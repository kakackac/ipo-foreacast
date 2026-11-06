import unittest
import tempfile
import json
import hashlib
from pathlib import Path
from unittest.mock import patch
import pandas as pd

from data.pipelines.stage_snapshots import demand_schedule, stage_documents, values_from_document, amendment_view, horizontal_after_cells, original_cache_hashes, reusable_checkpoint, document_analysis, section_text, StageResponseInvalid
from features.model_profiles import get_model_profile
from features.prediction_time_contract import validate_prediction_times


def schedule(start='2024년 02월 08일',end='02월 09일'):
    return f'<table><tr><td>수요예측 일시</td><td>{start} ~ {end}</td></tr></table>'


class StageSnapshotTests(unittest.TestCase):
    def test_http_200_error_html_is_not_saved_as_a_completed_original(self):
        class Response:
            text='<html>temporary service error</html>'
            def raise_for_status(self):pass
        class Session:
            def get(self,*args,**kwargs):return Response()
        class Transport:session=Session()
        with tempfile.TemporaryDirectory() as root, patch('data.pipelines.disclosure_connections.time.sleep'):
            with self.assertRaises(StageResponseInvalid):section_text(Transport(),'https://dart.fss.or.kr/report/viewer.do?rcpNo=20240101000001',Path(root),False)
            self.assertEqual(list(Path(root).glob('*.json')),[])
    def test_document_numbers_parse_once_and_a_bad_cache_is_not_trusted(self):
        class Parser:
            calls=0
            def _parse_offering_html(self,*args):self.calls+=1;return {}
            def _parse_demand_forecast_html(self,*args):self.calls+=1;return {}
        parser=Parser()
        with tempfile.TemporaryDirectory() as root:
            first=document_analysis(parser,'<p>original</p>','20240101000001','00000001',root)
            self.assertEqual(document_analysis(parser,'<p>original</p>','20240101000001','00000001',root),first)
            self.assertEqual(parser.calls,2)
            path=next((Path(root)/'stage_document_analyses').glob('*.json'));saved=json.loads(path.read_text());saved['sha256']='bad';path.write_text(json.dumps(saved))
            with self.assertRaisesRegex(RuntimeError,'analysis_integrity_failed'):
                document_analysis(parser,'<p>original</p>','20240101000001','00000001',root)

    def test_horizontal_after_column_is_only_a_consistency_view(self):
        body='<table><tr><td>항 목</td><td>정 정 전</td><td>정 정 후</td></tr><tr><td>공통 정정사항</td><td>희망가 5,900원</td><td>모집(매출)확정가액 : 4,500원</td></tr></table>'
        selected,proof=horizontal_after_cells(body)
        self.assertIn('4,500원',selected);self.assertNotIn('5,900원',selected)
        self.assertEqual(proof['rule'],'horizontal_after_cells_consistency_only_v1')
        with self.assertRaises(ValueError):horizontal_after_cells('<p>정정 전 5900원 정정 후 4500원</p>')
    def test_missing_original_checkpoint_is_not_a_permanent_negative_cache(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);url='https://dart.fss.or.kr/report/viewer.do?rcpNo=20240101000001'
            hashes=original_cache_hashes(root,{'url':url})
            saved={'source_cache_sha256':hashes,'attempts':[{'status':'official_source_not_cached'}]}
            self.assertTrue(reusable_checkpoint(root,saved,True))
            self.assertFalse(reusable_checkpoint(root,saved,False))
            directory=root/'dart_connection_sources';directory.mkdir()
            path=directory/(hashlib.sha256(url.encode()).hexdigest()+'.json');path.write_text('{}')
            self.assertFalse(reusable_checkpoint(root,saved,True))
            saved['source_cache_sha256']=original_cache_hashes(root,{'url':url})
            self.assertTrue(reusable_checkpoint(root,saved,True))
            path.write_text('{"changed":true}')
            with self.assertRaisesRegex(RuntimeError,'original_missing_or_changed'):reusable_checkpoint(root,saved,True)

    def test_angle_amendment_markers_are_explicit_but_horizontal_header_is_not(self):
        body='<p>(주1)&lt;정정 전&gt;</p><table><tr><td>old</td></tr></table><p>&lt;정정 후&gt;</p><table><tr><td>new</td></tr></table>'
        selected,proof=amendment_view(body)
        self.assertIn('new',selected);self.assertNotIn('old',selected)
        self.assertEqual(proof['pair_count'],1)
        self.assertIn('normalization',proof)
        bullet=body.replace('&lt;정정 전&gt;','■정정 전').replace('&lt;정정 후&gt;','■정정 후')
        self.assertIn('new',amendment_view(bullet)[0])
        with self.assertRaises(ValueError):amendment_view('<table><tr><td>정정 전</td><td>정정 후</td></tr></table>')

    def test_schedule_uses_labelled_full_range_not_incidental_dates(self):
        start,end,proof=demand_schedule(schedule())
        self.assertEqual(start,pd.Timestamp('2024-02-08'))
        self.assertEqual(end,pd.Timestamp('2024-02-09'))
        self.assertIn('cells',proof)
        with self.assertRaisesRegex(ValueError,'missing_or_conflicting'):
            demand_schedule('<p>2024.02.08~2024.02.09</p>')
        with self.assertRaises(ValueError):demand_schedule(schedule()+schedule('2024.02.10','2024.02.11'))

    def test_short_end_day_and_separate_domestic_foreign_windows(self):
        start,end,_=demand_schedule(schedule('2016년 12월 06일','07일'))
        self.assertEqual(end,pd.Timestamp('2016-12-07'))
        body='<table><tr><td>수요예측 일시</td><td>국내</td><td>2025년 01월 09일 ~ 2025년 01월 15일</td></tr><tr><td>수요예측 일시</td><td>해외</td><td>2025년 01월 06일 ~ 2025년 01월 15일</td></tr></table>'
        start,end,proof=demand_schedule(body)
        self.assertEqual(start,pd.Timestamp('2025-01-06'))
        self.assertIn('scoped_windows',proof)
        with self.assertRaisesRegex(ValueError,'invalid_or_reversed'):
            demand_schedule(schedule('2024년 1월 8일','2023년 1월 12일'))

    def test_explicit_scheduled_period_sentence_does_not_pick_subscription_dates(self):
        body='<p>당사의 수요예측 예정일은 2022년 09월 27일(화) ~ 28일(수)입니다. 청약 예정일은 2022년 10월 4일 ~ 5일입니다.</p>'
        start,end,proof=demand_schedule(body)
        self.assertEqual((start,end),(pd.Timestamp('2022-09-27'),pd.Timestamp('2022-09-28')))

    def test_actual_older_document_selected_for_pre_and_first_result_for_post(self):
        family=[{'rcept_no':'a','rcept_dt':'2024-01-01','report_nm':'증권신고서(지분증권)'},
                {'rcept_no':'b','rcept_dt':'2024-01-20','report_nm':'정정 증권신고서(지분증권)'},
                {'rcept_no':'c','rcept_dt':'2024-02-12','report_nm':'투자설명서'},
                {'rcept_no':'d','rcept_dt':'2024-02-15','report_nm':'정정 투자설명서'}]
        def load(m):
            if m['rcept_no']=='a':raise ValueError('unnecessary_old_original_requested')
            return schedule()+(' result' if m['rcept_no'] in ['c','d'] else ''),{'corp_code':'00000001','source_url':'official/'+m['rcept_no']}
        def demand(html,corp):return {'institutional_demand_parser_validation_status':'structurally_verified' if 'result' in html else 'value_not_found'}
        def price(html,receipt):return {'offering_price_review_status':'verified_currency_unit',
                                        'offering_price_finality':'confirmed_price_language'}
        with patch('data.pipelines.stage_snapshots.DARTCollector._parse_demand_forecast_html',side_effect=demand), \
                patch('data.pipelines.stage_snapshots.DARTCollector._parse_offering_html',side_effect=price):
            result=stage_documents(family,load,'2024-02-20')
        self.assertEqual(result['pre_demand']['metadata']['rcept_no'],'b')
        self.assertEqual(result['pre_demand']['prediction_at'],pd.Timestamp('2024-02-07 21:00',tz='Asia/Seoul'))
        self.assertEqual(result['post_demand']['metadata']['rcept_no'],'c')
        self.assertEqual(result['post_demand']['prediction_at'],pd.Timestamp('2024-02-13 00:01',tz='Asia/Seoul'))

    def test_missing_original_does_not_fall_back_to_final_document(self):
        family=[{'rcept_no':'a','rcept_dt':'2024-01-01','report_nm':'증권신고서(지분증권)'}]
        def load(m):raise ValueError('official_source_not_cached')
        result=stage_documents(family,load,'2024-02-20')
        self.assertTrue(all(x['status']=='official_source_not_cached' for x in result.values()))

    def test_later_event_date_anchor_needs_independent_pre_cutoff_schedule(self):
        family=[{'rcept_no':'old','rcept_dt':'2023-12-01','report_nm':'증권신고서(지분증권)'},
                {'rcept_no':'new','rcept_dt':'2024-02-12','report_nm':'투자설명서'}]
        def load(m):return (schedule('2024년 01월 08일','01월 09일') if m['rcept_no']=='old' else schedule()),{'corp_code':'00000001'}
        with patch('data.pipelines.stage_snapshots.DARTCollector._parse_demand_forecast_html',return_value={}), \
                patch('data.pipelines.stage_snapshots.DARTCollector._parse_offering_html',return_value={}):
            result=stage_documents(family,load,'2024-02-20')
        self.assertEqual(result['pre_demand']['status'],'canonical_period_not_announced_in_pre_cutoff_document')

    def test_pre_feature_values_never_receive_final_price_or_demand(self):
        class Parser:
            def _parse_offering_html(self,*args):return {'offering_price':20000,'offering_price_finality':'confirmed_price_language'}
            def _parse_demand_forecast_html(self,*args):return {}
        values,_=values_from_document(Parser(),{'html':'<p>old document</p>','metadata':{'rcept_no':'a'}},'00000001','pre_demand')
        self.assertNotIn('offering_price',values)
        self.assertNotIn('institutional_demand_ratio',values)

    def test_effective_midnight_is_not_delayed_twice_and_schedule_bounds_pre(self):
        frame=pd.DataFrame([{'event_id':'e','listing_date':'2024-02-20','prediction_at':'2024-02-13T00:01:00+09:00',
            'prediction_time_source':'first_verified_result','demand_result_available_at':'2024-02-13T00:00:00+09:00',
            'demand_result_time_source':'official','demand_result_time_is_effective':True,'institutional_demand_ratio':100}])
        audit=pd.DataFrame([{'event_id':'e','feature_name':'institutional_demand_ratio','available_at':'2024-02-13T00:00:00+09:00',
            'available_at_is_effective':True,'time_validation_status':'pre_listing_verified','is_missing':False}])
        self.assertTrue(validate_prediction_times(frame,get_model_profile('post_demand'),audit).iloc[0])
        frame=frame.drop(columns=['institutional_demand_ratio']);frame['underwriter_tier']=1
        frame['demand_start_at']='2024-02-14';frame['demand_start_time_source']='official_schedule'
        audit.feature_name='underwriter_tier'
        self.assertTrue(validate_prediction_times(frame,get_model_profile('pre_demand'),audit).iloc[0])
        frame['demand_start_at']='2024-02-12'
        self.assertFalse(validate_prediction_times(frame,get_model_profile('pre_demand'),audit).iloc[0])
