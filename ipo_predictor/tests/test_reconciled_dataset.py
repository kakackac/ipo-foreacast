import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from data.pipelines.data_delivery import digest
from data.pipelines.financial_history import atomic_json
from data.pipelines.reconciled_dataset import run,market_asof,normalise_observation_times,prior_targets_asof


class ReconciledDatasetTests(unittest.TestCase):
    def test_recent_targets_are_selected_at_cutoff_not_computed_then_discarded(self):
        frame=pd.DataFrame({'event_id':['a','b','c'],'listing_date':pd.date_range('2024-01-01',periods=3),
            'event_class':['general_ipo']*3,'offering_price_review_status':['verified_currency_unit']*3,
            'price_target_validation_status':['official_price_verified']*3,'open_return_pct':[10,999,100]})
        result=prior_targets_asof(frame)
        self.assertEqual(result.loc[2,'recent_ipo_avg_return_all'],10)
        self.assertEqual(json.loads(result.loc[2,'recent_ipo_avg_return_all_source_event_ids']),['a'])
        self.assertTrue(pd.isna(result.loc[0,'recent_ipo_avg_return_all']))
    def test_mixed_date_objects_strings_and_timezones_share_one_storage_dtype(self):
        frame=pd.DataFrame({'available_at':[pd.Timestamp('2024-01-01'),'2024-01-01T00:00:00+09:00',None]})
        result=normalise_observation_times(frame)
        self.assertEqual(str(result.available_at.dtype),'datetime64[ns, UTC]')
        self.assertEqual(result.available_at.iloc[0],result.available_at.iloc[1])
        with tempfile.TemporaryDirectory() as root:
            result.to_parquet(Path(root)/'observations.parquet',index=False)
            self.assertEqual(str(pd.read_parquet(Path(root)/'observations.parquet').available_at.dtype),'datetime64[ns, UTC]')
    def test_market_availability_is_recomputed_not_stale_or_double_shifted(self):
        dates=pd.bdate_range('2024-01-01','2024-03-25')
        index=pd.DataFrame({'date':dates,'close':range(100,100+len(dates))})
        features=pd.DataFrame({'listing_date':['2024-03-20','2024-04-20'],
                               'kospi_momentum_5d':[0.,0.],'kospi_momentum_20d':[0.,0.]})
        result=market_asof(features,index,'kospi')
        self.assertEqual(result.loc[0,'kospi_momentum_5d_available_at'],pd.Timestamp('2024-03-19'))
        self.assertTrue(pd.isna(result.loc[1,'kospi_momentum_5d']))
        self.assertEqual(result.loc[1,'kospi_momentum_5d_missing_reason'],'market_context_not_collected')

    def test_full_pinned_build_reuses_versions_and_keeps_raw_unapproved_values(self):
        with tempfile.TemporaryDirectory() as root:
            raw=Path(root)/'raw';raw.mkdir();processed=Path(root)/'processed'
            source=processed/'historical_reconciliations'/'source';source.mkdir(parents=True)
            financial=processed/'historical_financial_reconciliations'/'finance';financial.mkdir(parents=True)
            event=dict(event_id='e',corp_name='Issuer',ticker='000001',listing_date=pd.Timestamp('2024-03-01'),
                       event_class='general_ipo',market='KOSDAQ',offering_price=10000,open_price=12000,
                       close_price=13000,price_resolution_status='official_price_verified')
            record=dict(event_id='e',corp_name='Issuer',corp_code='01234567',rcept_no='20240201000001',
                        rcept_dt='2024-02-01',listing_date='2024-03-01',offering_price=10000,
                        offering_price_review_status='verified_currency_unit',institutional_demand_ratio=100,
                        institutional_available_at='2024-02-01',institutional_validation_status='verified_dart_structural_aggregate_v1')
            inputs={'events':pd.DataFrame([event]),'records':pd.DataFrame([record]),
                    'features':pd.DataFrame([{'event_id':'e'}]),'input_sha256':{'base':'pinned'}}
            pd.DataFrame([{'event_id':'e','status':'official_sources_connected','reason':None,
                           'record':json.dumps(record),'proof':'{}'}]).to_parquet(source/'connection_results.parquet')
            atomic_json(source/'manifest.json',{'inputs':{'inputs':inputs['input_sha256']},'output_sha256':{
                'connection_results.parquet':digest(source/'connection_results.parquet')}})
            pd.DataFrame([dict(event_id='e',operating_margin=.1,financial_feature_provenance=json.dumps({
                'operating_margin':{'validation_status':'verified_dart_annual_financial_asof_v1',
                                    'available_at':'2024-02-02T00:00:00+09:00','source_reference':'20240201000001'}}))]).to_parquet(financial/'summaries.parquet')
            atomic_json(financial/'manifest.json',{'inputs':inputs['input_sha256'],
                'connection_manifest':digest(source/'manifest.json'),'scope':'all_connected_general_ipo_financial_sections',
                'status_counts':{'verified_dart_annual_financial_asof_v1':1},'output_sha256':{
                    'summaries.parquet':digest(financial/'summaries.parquet')}})
            dates=pd.date_range('2024-01-01','2024-02-29',freq='B')
            for name in ['kospi','kosdaq']:
                pd.DataFrame({'date':dates,'close':range(100,100+len(dates))}).to_parquet(raw/(name+'_index.parquet'))
            report=json.loads((source/'manifest.json').read_text())
            with patch('data.pipelines.reconciled_dataset.load_inputs',return_value=inputs), \
                    patch('data.pipelines.reconciled_dataset.pin_connections',return_value=(source,report)):
                first=run(raw,processed,financial);second=run(raw,processed,financial)
            self.assertEqual(first['version'],second['version'])
            self.assertFalse(first['deployment_authorized']);self.assertFalse(first['training_executed'])
            self.assertEqual(first['validation']['time_violations'],0)
            features=pd.read_parquet(Path(first['path'])/'features_all.parquet')
            self.assertAlmostEqual(features.loc[0,'operating_margin'],.1)
            self.assertAlmostEqual(features.loc[0,'open_return_pct'],20)
            self.assertFalse((processed/'current_financial_repair.json').exists())
            pointer=json.loads((processed/'current_reconciled_data.json').read_text())
            self.assertEqual(pointer['sha256'],digest(Path(first['path'])/'manifest.json'))
            self.assertFalse(pointer['deployment_authorized'])
            self.assertTrue((Path(first['path'])/'kospi_index_snapshot.parquet').exists())
            self.assertTrue((Path(first['path'])/'event_master.parquet').exists())
