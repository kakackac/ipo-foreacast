import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from data.collectors.krx_collector import KRXCollector
from data.pipelines.event_model_inputs import collect_days, read_day, new_frames, run
from data.pipelines.research_dataset import FEATURES
from features.source_contracts import DART_STRUCTURAL_AGGREGATE_STATUS


class EventModelInputTests(unittest.TestCase):
    def test_no_new_events_never_requests_market_data(self):
        krx=Mock()
        with tempfile.TemporaryDirectory() as root:
            _,days,report=collect_days(krx,root,self.seed(),pd.DataFrame())
        self.assertEqual(report['required_weekdays'],0)
        self.assertEqual(days,{})
        krx._get_daily_records.assert_not_called()

    def test_full_input_connection_with_simulated_market_and_separate_spac_policy(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);raw=root/'raw';processed=root/'processed';source=root/'events';dart=root/'dart';repair=root/'repair'
            for p in (raw,processed,source,dart,repair):p.mkdir();(p/'manifest.json').write_text('{}')
            dates=pd.bdate_range('2026-07-01','2026-09-09')
            pd.DataFrame({'date':dates,'index_code':'KOSPI','close':[100.+i for i in range(len(dates))]}).to_parquet(raw/'kospi_index.parquet')
            common={'listing_date':pd.Timestamp('2026-09-15'),'status':'official_identity_family_and_price_verified',
                    'corp_code':'01378005','rcept_no':'20260908000001','offering_price':100.}
            events=pd.DataFrame([dict(common,event_id='one',ticker='123450',event_class='general_ipo'),
                                 dict(common,event_id='spac',ticker='234560',event_class='spac_ipo')])
            events.to_parquet(dart/'events.parquet')
            prices=events[['event_id','ticker','listing_date']].assign(market='KOSDAQ',
                price_resolution_status='official_price_verified',open_price=150.,close_price=180.)
            prices.to_parquet(source/'events.parquet')
            observations=pd.DataFrame([{'event_id':e,'feature':f,'value':v,'receipt':'20260908000001',
                'is_missing':False,'human_review_required':False,'validation_status':'verified_official_source',
                'available_at':pd.Timestamp('2026-09-09',tz='Asia/Seoul')}
                for e in ['one','spac'] for f,v in [('institutional_demand_ratio',1000.),('lockup_commitment_ratio',.2)]])
            observations.to_parquet(dart/'observations.parquet')
            lineage=pd.DataFrame([{'event_id':e,'corp_code':'01378005','rcept_no':'20260908000001','rcept_dt':'2026-09-08'}
                                  for e in ['one','spac']]);lineage.to_parquet(dart/'lineage.parquet')
            old={f:float('nan') for f in FEATURES}
            old.update(event_id='old',corp_code='01378005',rcept_no='20260701000001',offering_price_rcept_no='20260701000001',
                listing_date=pd.Timestamp('2026-08-01'),market='KOSDAQ',event_class='general_ipo',offering_price=100.,
                offering_price_review_status='verified_currency_unit',price_target_validation_status='official_price_verified',
                institutional_demand_ratio=100.,open_return_pct=10.,close_return_pct=20.)
            pd.DataFrame([old]).to_parquet(repair/'features_all.parquet')
            pd.DataFrame([{'event_id':'old','feature_name':'institutional_demand_ratio','raw_value':100.,
                'available_at':pd.Timestamp('2026-07-01'),'source_reference':'20260701000001',
                'validation_status':DART_STRUCTURAL_AGGREGATE_STATUS,'is_missing':False,'human_review_required':False}]).to_parquet(repair/'feature_observations.parquet')
            pd.DataFrame([{'event_id':'old','corp_code':'01378005','rcept_no':'20260701000001','rcept_dt':'2026-07-01'}]).to_parquet(repair/'disclosure_lineage.parquet')
            krx=KRXCollector(api_key='test',request_delay=0)
            krx._get_daily_records=Mock(side_effect=lambda endpoint,day:[{'BAS_DD':day,'IDX_NM':'KOSPI','CLSPRC_IDX':'151'}])
            with patch('data.pipelines.event_model_inputs.current_event_delivery',return_value=(source,{'version':'source'})), \
                 patch('data.pipelines.event_model_inputs.current_disclosures',return_value=(dart,{'event_version':'source'})), \
                 patch('data.pipelines.event_model_inputs.current_financial_repair',return_value=repair):
                result=run(raw,processed,krx=krx)
                self.assertEqual(result['ready_general_ipo_rows'],1)
                self.assertEqual(result['pending_reasons'],{'separate_spac_policy_not_general_model':1})
                self.assertEqual(result['time_violations'],0)
                self.assertFalse(result['training_executed'])
                krx._get_daily_records=Mock(side_effect=AssertionError('must reuse completed days'))
                second=run(raw,processed,krx=krx)
                self.assertTrue(second['reused'])
                self.assertEqual(result['version'],second['version'])
                self.assertEqual(second['current_collection']['new_daily_requests'],0)

    def seed(self):
        return pd.DataFrame([{'date':pd.Timestamp('2026-09-09'),'index_code':'KOSPI','close':100.}])

    def test_missing_key_never_reuses_stale_seed_as_completed_collection(self):
        krx=KRXCollector(api_key='',request_delay=0)
        krx._get_daily_records=Mock()
        with tempfile.TemporaryDirectory() as root:
            index,days,status=collect_days(krx,root,self.seed(),pd.DataFrame([{'listing_date':'2026-09-15'}]))
        self.assertEqual(status['status'],'market_context_requires_krx_api')
        self.assertEqual(status['required_weekdays'],2)
        self.assertEqual(days,{})
        krx._get_daily_records.assert_not_called()

    def test_partial_failure_resumes_only_remaining_dates_and_reuses_empty_response(self):
        krx=KRXCollector(api_key='test',request_delay=0)
        krx._get_daily_records=Mock(side_effect=[[{'BAS_DD':'20260910','IDX_NM':'KOSPI','CLSPRC_IDX':'101'}],RuntimeError('private exception')])
        events=pd.DataFrame([{'listing_date':'2026-09-15'}])
        with tempfile.TemporaryDirectory() as root:
            _,days,report=collect_days(krx,root,self.seed(),events)
            self.assertEqual(set(days),{'20260910'})
            self.assertEqual(report['status'],'partial_failure_retryable')
            self.assertNotIn('private',json.dumps(report))
            krx._get_daily_records=Mock(return_value=[])
            index,days,report=collect_days(krx,root,self.seed(),events)
            krx._get_daily_records.assert_called_once_with('idx/kospi_dd_trd','20260911')
            self.assertEqual(report['status'],'complete')
            self.assertEqual(len(index),2)
            krx._get_daily_records=Mock(side_effect=AssertionError('unexpected network'))
            collect_days(krx,root,self.seed(),events)
            krx._get_daily_records.assert_not_called()
            path=Path(root)/'krx_index_days/KOSPI/20260910.json'
            payload=json.loads(path.read_text());payload['data']['selected_row']['CLSPRC_IDX']='999'
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):read_day(path,'20260910')

    def test_wrong_index_date_or_zero_close_never_becomes_checkpoint(self):
        events=pd.DataFrame([{'listing_date':'2026-09-14'}])
        for row in [{'BAS_DD':'20260909','IDX_NM':'KOSPI','CLSPRC_IDX':'100'},
                    {'BAS_DD':'20260910','IDX_NM':'KOSPI','CLSPRC_IDX':'0'},
                    {'BAS_DD':'20260910','IDX_NM':'KOSPI 200','CLSPRC_IDX':'100'}]:
            krx=KRXCollector(api_key='test',request_delay=0);krx._get_daily_records=Mock(return_value=[row])
            with tempfile.TemporaryDirectory() as root:
                _,_,report=collect_days(krx,root,self.seed(),events)
                self.assertEqual(report['status'],'partial_failure_retryable')
                self.assertEqual(list(Path(root).rglob('*.json')),[])

    def test_publication_uses_each_features_actual_receipt_not_parent_date(self):
        disclosures=pd.DataFrame([{'event_id':'one','ticker':'123450','listing_date':'2026-11-04',
            'status':'official_identity_family_and_price_verified','corp_code':'01378005',
            'rcept_no':'20261021000363','event_class':'general_ipo','offering_price':100.}])
        prices=pd.DataFrame([{'event_id':'one','ticker':'123450','listing_date':'2026-11-04','market':'KOSDAQ',
            'price_resolution_status':'official_price_verified','open_price':150.,'close_price':200.}])
        obs=pd.DataFrame([{'event_id':'one','feature':field,'is_missing':False,'human_review_required':False,
            'validation_status':'verified_official_source','receipt':'20261022000370',
            'available_at':pd.Timestamp('2026-10-23',tz='Asia/Seoul'),'value':value}
            for field,value in [('institutional_demand_ratio',1000.),('lockup_commitment_ratio',.2)]])
        lineage=pd.DataFrame([{'event_id':'one','rcept_no':'20261022000370','rcept_dt':'2026-10-22'}])
        features,ledger=new_frames(disclosures,obs,lineage,prices)
        self.assertEqual(features.open_return_pct.iloc[0],50.)
        self.assertEqual(features.close_return_pct.iloc[0],100.)
        self.assertEqual(ledger.available_at.iloc[0],pd.Timestamp('2026-10-22'))
        obs['available_at']=pd.Timestamp('2026-10-22',tz='Asia/Seoul')
        with self.assertRaisesRegex(ValueError,'publication_conflict'):new_frames(disclosures,obs,lineage,prices)


if __name__=='__main__':unittest.main()
