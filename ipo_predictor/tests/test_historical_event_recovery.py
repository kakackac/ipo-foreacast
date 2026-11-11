import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from data.pipelines.historical_event_recovery import (parse_company_search, parse_filing_search, select_body_seed,
    capture_sources, source_record, run, verify_classification_table, resolve_registered_classification)
from data.collectors.dart_collector import DARTCollector
from data.pipelines.historical_ipo_pipeline import HistoricalIPOPipeline
from data.pipelines.disclosure_connections import company_identity
from data.processors.feature_engineer import FeatureEngineer
from tests.test_actual_data_pipeline import _FakeDART, _FakeKRX


class HistoricalEventRecoveryTests(unittest.TestCase):
    def test_non_public_classification_requires_one_complete_official_table(self):
        rule={'required_table_values':['설립방법','포괄적 주식이전','2019년 02월 13일']}
        table='<TABLE><TR><TD>설립방법</TD><TD>포괄적 주식이전</TD></TR><TR><TD>2019년 02월 13일</TD></TR></TABLE>'
        self.assertEqual(len(verify_classification_table(table,rule)),2)
        for body in [table+table,table.replace('2019년 02월 13일','2019년 02월 14일'),
                     '<table><tr><td>설립방법</td><td>포괄적 주식이전</td></tr></table><table><tr><td>2019년 02월 13일</td></tr></table>']:
            with self.assertRaises(ValueError):verify_classification_table(body,rule)

    def test_classification_registry_cannot_exclude_a_different_or_priced_ipo(self):
        from types import SimpleNamespace
        rule={'event_id':'one','ticker':'123456','corp_code':'12345678','listing_date':'2019-02-13'}
        for event in [SimpleNamespace(event_id='other',ticker='123456',listing_date='2019-02-13',offering_price=None),
                      SimpleNamespace(event_id='one',ticker='123456',listing_date='2019-02-13',offering_price=1000)]:
            dart=Mock()
            with self.assertRaisesRegex(ValueError,'identity_conflict'):
                resolve_registered_classification(dart,'unused',event,'12345678',rule,True)
            dart.session.get.assert_not_called()

    def test_recovered_record_uses_approved_bundles_not_raw_primary_values(self):
        from types import SimpleNamespace
        from data.pipelines.event_disclosures import FIELDS
        event=SimpleNamespace(event_id='one',ticker='123456',corp_name='Historical',listing_date='2015-03-01')
        values=dict.fromkeys(FIELDS);values.update(offering_price=1000,new_shares=100,secondary_shares=50)
        meta={'rcept_no':'20150201000001','rcept_dt':'2015-02-01','report_nm':'terms'}
        proof={'corp_code':'12345678','resolved_structure':{'new_shares':100,'secondary_shares':50,
            'offering_structure_rcept_no':'20150202000001','offering_structure_rcept_dt':'2015-02-02',
            'new_shares_parser_validation_status':'structurally_verified',
            'secondary_shares_parser_validation_status':'structurally_verified'},
            'resolved_float':{'public_float_ratio_disclosed':None,
                'public_float_parse_method':'conflicting_float_ratios_review_required'}}
        record=source_record(event,values,meta,proof,{'new_shares':999,'secondary_shares':None,
                                                    'public_float_ratio_disclosed':.2})
        self.assertEqual((record['new_shares'],record['secondary_shares']),(100,50))
        self.assertEqual(record['offering_structure_rcept_no'],'20150202000001')
        self.assertIsNone(record['public_float_ratio_disclosed'])

    def test_dated_body_correction_is_not_attachment_or_old_final_terms(self):
        rows = [{'rcept_no':'1','rcept_dt':'2015-04-01','report_nm':'[발행조건확정]증권신고서(지분증권)'},
                {'rcept_no':'2','rcept_dt':'2015-04-02','report_nm':'[기재정정]증권신고서(지분증권)'},
                {'rcept_no':'3','rcept_dt':'2015-04-03','report_nm':'[첨부정정]증권신고서(지분증권)'}]
        self.assertEqual(select_body_seed(rows)['rcept_no'],'2')

    def test_source_content_is_checked_not_only_stored_hash(self):
        import hashlib, json
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'dart_connection_sources';path.mkdir()
            url='https://dart.fss.or.kr/example';sha=hashlib.sha256(b'original').hexdigest()
            (path/(hashlib.sha256(url.encode()).hexdigest()+'.json')).write_text(json.dumps(
                {'url':url,'sha256':sha,'text':'tampered'}))
            with self.assertRaisesRegex(ValueError,'source_changed'):
                capture_sources(folder,[{'url':url,'sha256':sha}],{})

    def test_no_target_recovery_is_empty_success_without_network(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);raw=root/'raw';raw.mkdir();base=root/'base';base.mkdir()
            frame=pd.DataFrame(columns=['event_id','event_class','listing_date'])
            frame.to_parquet(base/'features_all.parquet')
            frame.to_parquet(base/'feature_observations.parquet')
            frame.to_parquet(raw/'krx_official_event_master.parquet')
            for name in ('kospi','kosdaq'):frame.to_parquet(raw/(name+'_index.parquet'))
            dart=Mock()
            with patch('data.pipelines.historical_event_recovery.current_financial_repair',return_value=base):
                report=run(raw,root/'processed',dart=dart,cached_only=True)
            self.assertEqual(report['status_counts'],{})
            self.assertEqual(report['processed_events'],0)
            dart.session.get.assert_not_called()

    def test_old_bookbuilding_summary_named_column_not_arbitrary_last_number(self):
        table=[['참여건수','신청수량','단순 경쟁률'],['726','814,926,000','643.70:1']]
        extract=DARTCollector._extract_aggregate_demand_ratio
        self.assertEqual(extract(table,'(12) 수요예측 결과① 수요예측 참여내역')['institutional_demand_ratio'],643.7)
        for title in ['일반청약 경쟁률','수요예측 실시 예정','수요예측 결과']:
            self.assertIsNone(extract(table,title))
        table[0][2]='개인 경쟁률'
        self.assertIsNone(extract(table,'수요예측 결과① 수요예측 참여내역'))

    def test_current_delisting_status_is_not_historical_market_identity(self):
        from tests.test_event_disclosures import company
        html = company(market='기타법인')
        with self.assertRaisesRegex(ValueError, 'market_mismatch'):
            company_identity(html,'01378005','495910','KOSDAQ')
        self.assertEqual(company_identity(html,'01378005','495910','KOSDAQ',
            verify_current_market=False)['current_corporation_type'], '기타법인')
        with self.assertRaisesRegex(ValueError, 'trading_code_mismatch'):
            company_identity(html,'01378005','999999','KOSDAQ',verify_current_market=False)

    def test_official_company_search_binds_code_not_name(self):
        html = "<table><tbody><tr><td><a href=\"javascript:select('00620558');\">New Name</a></td><td>195990</td></tr></tbody></table>"
        self.assertEqual(parse_company_search(html, '195990')['corp_code'], '00620558')
        for content in [html.replace('195990', '195991'), html + html, '<html>access denied</html>']:
            with self.assertRaises(ValueError): parse_company_search(content, '195990')

    def test_search_response_uses_displayed_date_not_receipt_prefix(self):
        html = """<table><tbody id="tbody"><tr><td><a onclick="openCorpInfoNew('00620558')">Name</a></td>
        <td><a id="r_20150331000001" href="/dsaf001/main.do?rcpNo=20150331000001">증권신고서(지분증권)</a></td>
        <td>2015.04.01</td></tr></tbody></table>[총 1건]"""
        rows, total = parse_filing_search(html, '00620558', '20150101', '20150412')
        self.assertEqual(total, 1)
        self.assertEqual(rows[0]['rcept_dt'], '2015-04-01')
        for content in [html.replace("openCorpInfoNew('00620558')", "openCorpInfoNew('99999999')"),
                        html.replace('2015.04.01', '2015.05.01'), html.replace('[총 1건]', '')]:
            with self.assertRaises(ValueError): parse_filing_search(content, '00620558', '20150101', '20150412')

    def test_empty_search_is_not_access_failure_or_proof_of_nonpublication(self):
        self.assertEqual(parse_filing_search('<tbody id="tbody">조회 결과가 없습니다.</tbody>',
            '00620558', '20150101', '20150412'), ([], 0))
        with self.assertRaises(ValueError):
            parse_filing_search('<html>maintenance</html>', '00620558', '20150101', '20150412')

    def test_stock_code_recovers_renamed_company_and_rejects_name_collision(self):
        for code, expected in [('123456', 1), ('999999', 0)]:
            dart = _FakeDART()
            listing = _FakeKRX().get_official_listing_events('20240101', '20241231')
            frame = _FakeDART().get_ipo_disclosure_list('20240101', '20241231').assign(stock_code=code)
            if code == '123456': frame['corp_name'] = 'Different Current Name'
            dart.get_ipo_disclosure_list = Mock(return_value=frame)
            with tempfile.TemporaryDirectory() as folder:
                pipeline = HistoricalIPOPipeline(dart_collector=dart, raw_dir=Path(folder)/'raw',
                                                 processed_dir=Path(folder)/'processed')
                rows, _ = pipeline._collect_dart_records(listing, 2024, 2024, include_dart_demand_audit=False)
                self.assertEqual(len(rows), expected)
                self.assertLess(dart.get_ipo_disclosure_list.call_args_list[0].args[0], '20240101')

    def test_event_id_join_survives_names_and_never_joins_other_event(self):
        dart = pd.DataFrame([{'event_id':'old', 'corp_name':'Current', 'listing_date':'2015-04-13'}])
        krx = pd.DataFrame([{'event_id':'old','corp_name':'Historical','listing_date':'2015-04-13'},
                            {'event_id':'new','corp_name':'Current','listing_date':'2026-01-01'}])
        merged = FeatureEngineer()._merge_base(dart, krx)
        self.assertEqual(merged.event_id.tolist(), ['old'])
        self.assertEqual(merged.corp_name.tolist(), ['Historical'])
        with self.assertRaises(ValueError):
            FeatureEngineer()._merge_base(dart, pd.concat([krx,krx.iloc[:1]]))


if __name__ == '__main__':
    unittest.main()
