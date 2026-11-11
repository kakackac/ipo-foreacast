import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
import requests

from data.pipelines.disclosure_connections import company_identity, public_text
from data.pipelines.event_disclosures import accepted_values, discover_candidates, run, connect_event, cached_calendar_candidates
from data.collectors.dart_collector import DARTCollector


def company(corp='01378005', ticker='495910', market='코스닥시장'):
    return (f"<button onclick=\"searchPopReport('{corp}','name')\"></button><table>"
            f'<tr><th>종목코드</th><td>{ticker}</td></tr><tr><th>법인구분</th><td>{market}</td></tr></table>')


class EventDisclosureTests(unittest.TestCase):
    def test_receipt_specific_iso_time_and_timestamp_write_one_parquet_type(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);event=root/'source';event.mkdir()
            (event/'manifest.json').write_text('{}')
            pd.DataFrame([{'event_id':'one','ticker':'495910','corp_name':'EST','listing_date':pd.Timestamp('2026-11-04'),
                'event_class':'general_ipo','offering_price':12400}]).to_parquet(event/'added_events.parquet')
            pd.DataFrame([{'event_id':'one','candidate_corp_codes':'["01378005"]',
                'candidate_receipts':'["20261021000363"]'}]).to_parquet(event/'disclosure_links.parquet')
            from data.pipelines.event_disclosures import FIELDS
            values=dict.fromkeys(FIELDS);values.update(offering_price=12400,public_float_ratio_disclosed=.376)
            metadata={'rcept_no':'20261021000363'}
            proof={'corp_code':'01378005','family':[],'sources':[],
                'field_sources':{'public_float_ratio_disclosed':{'receipt':'20261021000370',
                    'available_at':'2026-10-22T00:00:00+09:00','source_url':'https://dart.fss.or.kr/dsaf001/main.do?rcpNo=20261021000370'}}}
            result=(values,metadata,proof,{},pd.Timestamp('2026-10-22',tz='Asia/Seoul'),pd.Timestamp('2026-11-03T21:00:00+09:00'))
            with patch('data.pipelines.event_disclosures.current_event_delivery',return_value=(event,{'version':'source'})), \
                 patch('data.pipelines.event_disclosures.connect_event',return_value=result):
                report=run(root/'raw',root/'processed',dart=Mock(),cached_only=True)
            ledger=pd.read_parquet(Path(report['path'])/'observations.parquet')
            self.assertEqual(str(ledger.available_at.dtype),'datetime64[ns, Asia/Seoul]')
            self.assertEqual(ledger[ledger.feature.eq('public_float_ratio_disclosed')].receipt.iloc[0],'20261021000370')

    def test_saved_calendar_requires_hash_period_and_both_subscription_dates(self):
        import hashlib
        import os
        html='''<select id="year"><option selected value="2026"></option></select>
        <select id="month"><option selected value="9"></option></select>
        <li class="day"><div class="date">10</div><a href="/a?rcpNo=20260909000307"><div class="gm 01876251">회사[시작]</div></a></li>
        <li class="day"><div class="date">11</div><a href="/a?rcpNo=20260909000307"><div class="gm 01876251">회사[종료]</div></a></li>'''
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'dart_calendar_audit'/'snapshot';root.mkdir(parents=True)
            path=root/'2026_09.html';path.write_text(html)
            report={'status':'complete','source':'https://dart.fss.or.kr/dsac008/main.do',
                'collected_at':'2026-09-29T12:00:00+09:00','requests':[{'year':2026,'month':9,
                'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}]}
            (root/'report.json').write_text(json.dumps(report))
            candidates=cached_calendar_candidates(folder)
            self.assertEqual(candidates,cached_calendar_candidates(os.path.relpath(folder)))
            self.assertEqual(candidates[0]['corp_code'],'01876251')
            self.assertEqual(candidates[0]['subscription_end'],'2026-09-11')
            path.write_text('tampered')
            with self.assertRaisesRegex(ValueError,'hash_mismatch'):cached_calendar_candidates(folder)
            report['status']='failed';(root/'report.json').write_text(json.dumps(report))
            self.assertEqual(cached_calendar_candidates(folder),[])

    def test_correction_index_refreshes_daily_not_on_every_resume(self):
        import hashlib
        receipt='20261021000363'
        url='https://dart.fss.or.kr/dsaf001/main.do?rcpNo='+receipt
        event=SimpleNamespace(ticker='495910',market='KOSDAQ',listing_date='2026-11-04')
        candidates=[{'corp_code':'01378005','rcept_no':receipt}]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'dart_connection_sources';path.mkdir()
            cache=path/(hashlib.sha256(url.encode()).hexdigest()+'.json')
            for days,cached_only,expected in [(0,False,False),(2,False,True),(2,True,False)]:
                cache.write_text(json.dumps({'collected_at':(pd.Timestamp.now(tz='Asia/Seoul')-pd.Timedelta(days=days)).isoformat()}))
                with patch('data.pipelines.event_disclosures.public_text',return_value=(company(),'a'*64)), \
                     patch('data.pipelines.event_disclosures.official_family_document',side_effect=ValueError('stop_after_refresh_decision')) as family:
                    with self.assertRaisesRegex(ValueError,'stop_after_refresh_decision'):
                        connect_event(Mock(),folder,event,candidates,cached_only)
                    self.assertEqual(family.call_args.kwargs['refresh_index'],expected)

    def test_institutional_scope_can_be_explicit_in_heading_and_category_schema(self):
        table=[['구분','운용사(집합) 공모','운용사(고유)','투자일임사','합계'],
               ['경쟁률','100','200','300','600']]
        extract=DARTCollector._extract_aggregate_demand_ratio
        self.assertIsNone(extract(table))
        self.assertIsNone(extract(table,'일반 청약 수요예측 참여 내역'))
        self.assertEqual(extract(table,'(가) 수요예측 참여 내역')['institutional_demand_ratio'],600)
        table[0][1]='개인 청약'
        self.assertIsNone(extract(table,'(가) 수요예측 참여 내역'))

    def test_allocation_regulation_footnotes_do_not_hide_actual_lockup_results(self):
        html = ('<table><tr><td>주1)</td><td>기관투자자는 15일 의무보유 확약 시 10% 이상 배정</td></tr>'
                '<tr><td>주2)</td><td>기관투자자는 15일 의무보유 확약 시 30% 이상 배정</td></tr></table>'
                '<p>기관투자자 수요예측 결과</p><table><tr><td>의무보유확약</td><td>기관 수요예측</td></tr>'
                '<tr><td>6개월 확약</td><td>1%</td></tr><tr><td>3개월 확약</td><td>2%</td></tr>'
                '<tr><td>1개월 확약</td><td>3%</td></tr><tr><td>15일 확약</td><td>4%</td></tr></table>')
        result=DARTCollector._extract_lockup_ratios_from_tables(html)
        self.assertAlmostEqual(result['lockup_commitment_ratio'],.1)
        self.assertEqual(result['parser_validation_status'],'structurally_verified')

    def test_dash_is_zero_only_when_full_aggregate_reconciles_independently(self):
        table = [['구분','외국 기관투자자','합계','합계'],['구분','수량','건수','수량'],
                 ['6개월 확약','0','1','100'],['1개월 확약','-','-','-'],
                 ['미확약','200','4','900'],['합계','200','5','1000']]
        extract = DARTCollector._extract_aggregate_lockup_shares
        self.assertAlmostEqual(extract(table,1000)['lockup_commitment_ratio'],.1)
        self.assertIsNone(extract(table))
        self.assertIsNone(extract(table,999))
        table[4][-1]='800'
        self.assertIsNone(extract(table,1000))

    def test_corporate_identity_requires_official_code_and_market_not_name(self):
        self.assertEqual(company_identity(company(),'01378005','495910','KOSDAQ')['ticker'],'495910')
        for html in [company(corp='12345678'),company(ticker='999999'),company(market='유가증권시장'),
                     company().replace('</table>','<tr><th>종목코드</th><td>999999</td></tr></table>')]:
            with self.assertRaises(ValueError): company_identity(html,'01378005','495910','KOSDAQ')

    def test_no_guessing_price_missing_values_or_period_lockups(self):
        document = {'offering_price':12400,'offering_price_review_status':'verified_currency_unit',
                    'offering_price_finality':'confirmed_price_language','institutional_demand_ratio':630.1,
                    'institutional_demand_parser_validation_status':'structurally_verified',
                    'institutional_demand_rule_id':'DART_DEMAND_DIRECT_LABEL_V1',
                    'lockup_commitment_ratio':0.,'lockup_parser_validation_status':'structurally_verified',
                    'lockup_rule_id':'explicit_aggregate','lockup_6m_ratio':None,'secondary_shares':0,
                    'secondary_shares_parser_validation_status':'structurally_verified','secondary_shares_parse_method':'explicit'}
        values = accepted_values(document,12400)
        self.assertEqual(values['institutional_demand_ratio'],630.1)
        self.assertEqual(values['lockup_commitment_ratio'],0.)
        self.assertEqual(values['secondary_shares'],0)
        self.assertIsNone(values['price_band_low'])
        document['institutional_demand_parser_validation_status']='candidate_rejected'
        self.assertIsNone(accepted_values(document,12400)['institutional_demand_ratio'])
        with self.assertRaises(ValueError):accepted_values(document,12300)
        document['offering_price_finality']='planned'
        with self.assertRaises(ValueError):accepted_values(document,12400)

    def test_cached_only_cannot_accidentally_use_api(self):
        dart=Mock();dart.is_configured=False
        events=pd.DataFrame([{'listing_date':'2026-11-04'}])
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError,'requires_dart_api'):discover_candidates(dart,root,events,True)
        dart.get_ipo_disclosure_list.assert_not_called()

    def test_discovery_window_reused_and_tamper_detected(self):
        dart=Mock();dart.is_configured=True
        dart.get_ipo_disclosure_list.return_value=pd.DataFrame([{'corp_code':'01378005','rcept_no':'20261021000363'}])
        with tempfile.TemporaryDirectory() as root:
            events=pd.DataFrame([{'listing_date':'2026-11-04'}])
            discover_candidates(dart,root,events,False)
            discover_candidates(dart,root,events,True)
            dart.get_ipo_disclosure_list.assert_called_once_with('20260806','20261103')
            path=next(Path(root).rglob('*.json'))
            payload=json.loads(path.read_text());payload['rows'][0]['corp_code']='99999999'
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError,'tampered'):discover_candidates(dart,root,events,True)

    def test_discovery_access_failure_is_not_cached_as_no_filings(self):
        dart=Mock();dart.is_configured=True
        dart.get_ipo_disclosure_list.side_effect=requests.Timeout('not a no-data result')
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(requests.Timeout):
                discover_candidates(dart,root,pd.DataFrame([{'listing_date':'2026-11-04'}]),False)
            self.assertEqual(list(Path(root).rglob('*.json')),[])

    def test_new_correction_index_can_refresh_but_cached_only_never_does(self):
        dart=Mock();dart.session.get.return_value=Mock(text='first')
        with tempfile.TemporaryDirectory() as root:
            public_text(dart,'https://dart.fss.or.kr/index',root)
            dart.session.get.return_value=Mock(text='corrected')
            self.assertEqual(public_text(dart,'https://dart.fss.or.kr/index',root,refresh=True)[0],'corrected')
            self.assertEqual(public_text(dart,'https://dart.fss.or.kr/index',root,cached_only=True,refresh=True)[0],'corrected')
            self.assertEqual(dart.session.get.call_count,2)

    def test_missing_candidates_persist_null_evidence_without_approval(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);event=root/'source';event.mkdir()
            (event/'manifest.json').write_text('{}')
            pd.DataFrame([{'event_id':'event','ticker':'495910','corp_name':'name','listing_date':pd.Timestamp('2026-11-04'),
                           'event_class':'general_ipo','offering_price':12400}]).to_parquet(event/'added_events.parquet')
            pd.DataFrame([{'event_id':'event','candidate_corp_codes':'[]','candidate_receipts':'[]'}]).to_parquet(event/'disclosure_links.parquet')
            with patch('data.pipelines.event_disclosures.current_event_delivery',return_value=(event,{'version':'source'})):
                report=run(root/'raw',root/'processed',dart=Mock(),cached_only=True)
                again=run(root/'raw',root/'processed',dart=Mock(),cached_only=True)
            self.assertEqual(report,again)
            self.assertFalse(report['training_executed'])
            self.assertEqual(report['status_counts'],{'official_candidate_not_linked':1})
            rows=pd.read_parquet(Path(report['path'])/'observations.parquet')
            self.assertTrue(rows.value.isna().all())
            self.assertTrue(rows.is_missing.all())
            self.assertTrue(rows.human_review_required.all())
            self.assertEqual(report['time_violations'],0)

    def test_empty_new_supported_population_finishes_without_network(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);event=root/'source';event.mkdir()
            (event/'manifest.json').write_text('{}')
            pd.DataFrame(columns=['event_id','ticker','listing_date','event_class']).to_parquet(event/'added_events.parquet')
            pd.DataFrame(columns=['event_id','candidate_corp_codes','candidate_receipts']).to_parquet(event/'disclosure_links.parquet')
            dart=Mock()
            with patch('data.pipelines.event_disclosures.current_event_delivery',return_value=(event,{'version':'source'})):
                result=run(root/'raw',root/'processed',dart=dart,discover=True)
            self.assertEqual(result['events'],0)
            self.assertEqual(result['time_violations'],0)
            dart.get_ipo_disclosure_list.assert_not_called()


if __name__=='__main__':unittest.main()
