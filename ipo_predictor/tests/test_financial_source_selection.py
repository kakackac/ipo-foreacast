import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from data.pipelines.financial_source_selection import nonfinancial_terms_scope, resolve_financial_section


ROOT='20230316000002'
PROSPECTUS='20230315000001'
CORP='12345678'


def index(receipt,day,financial=False,reverse=True,newer=False):
    title='투자설명서' if financial else '증권신고서(지분증권)'
    text=f'<title>테스트/{title}/{day}</title>openCorpInfoNew("{CORP}");'
    text+=f'<select id="family"><option selected value="rcpNo={receipt}">{day} {title}</option>'
    if newer:text+='<option value="rcpNo=20230317000001">2023.03.17 [정정]투자설명서</option>'
    text+='</select><select id="ref">'
    if financial and reverse:text+=f'<option value="rcpNo={ROOT}">2023.03.16 증권신고서</option>'
    if not financial:text+=f'<option title="투자설명서" value="rcpNo={PROSPECTUS}">2023.03.15 투자설명서</option>'
    text+='</select>var node1 = {};'
    values=dict(text='III. 재무에 관한 사항' if financial else '증권발행조건확정',
                rcpNo=receipt,dcmNo='1',eleId='2',offset='3',length='4',dtd='dart3.xsd')
    return text+''.join(f'node1[\'{k}\'] = "{v}";' for k,v in values.items())


def notice():
    return ('<html><body><table><tr><td>항목</td><td>정정 전</td><td>정정 후</td></tr>'
            '<tr><td>I. 모집 또는 매출에 관한 일반사항</td>'
            '<td>I. 모집 또는 매출에 관한 일반사항</td><td>I. 모집 또는 매출에 관한 일반사항</td></tr>'
            '<tr><td>1. 공모개요</td><td>주1)</td><td>주1)</td></tr></table>'
            '<p>주1)</p><p>[정정 전]</p><p>공모가액 예정</p><p>[정정 후]</p><p>공모가액 확정</p></body></html>')


class FinancialSourceSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.raw=Path(self.temp.name);self.cache=self.raw/'dart_connection_sources'
        self.dart=Mock()
        self.put_index(ROOT,index(ROOT,'2023.03.16'))
        self.put_index(PROSPECTUS,index(PROSPECTUS,'2023.03.15',financial=True))
        self.put(self.viewer(ROOT),notice())
        self.put(self.viewer(PROSPECTUS),'<html><body>재무<table><tr><td>재무</td></tr></table></body></html>')

    def put(self,url,text):
        self.cache.mkdir(exist_ok=True)
        (self.cache/(hashlib.sha256(url.encode()).hexdigest()+'.json')).write_text(json.dumps(
            {'url':url,'text':text,'sha256':hashlib.sha256(text.encode()).hexdigest()},ensure_ascii=False))

    def put_index(self,receipt,text):
        self.put('https://dart.fss.or.kr/dsaf001/main.do?rcpNo='+receipt,text)

    def viewer(self,receipt):
        return 'https://dart.fss.or.kr/report/viewer.do?rcpNo='+receipt+'&dcmNo=1&eleId=2&offset=3&length=4&dtd=dart3.xsd'

    def resolve(self,**kwargs):
        result=resolve_financial_section(self.dart,self.raw,ROOT,'테스트',CORP,
            kwargs.get('cutoff','2023-03-20 21:00'),'2023-03-16',cached_only=True)
        self.dart.session.get.assert_not_called()
        return result

    def test_bidirectional_prospectus_requires_complete_unchanged_financial_scope(self):
        result=self.resolve()
        self.assertEqual(result['status'],'official_section_downloaded')
        self.assertEqual(result['receipt'],PROSPECTUS)
        proof=result['selection_evidence']
        self.assertEqual(proof['requested_receipt'],ROOT)
        self.assertEqual(len(proof['later_terms_corrections']),1)
        self.assertEqual(proof['later_terms_corrections'][0]['scope']['note_numbers'],[1])
        self.assertFalse(proof['absence_confirmed'])
        self.assertEqual(self.resolve(),result)

    def test_financial_change_in_overview_is_not_replaced_by_older_values(self):
        self.put(self.viewer(ROOT),notice().replace('1. 공모개요','1. 요약재무정보'))
        self.assertEqual(self.resolve()['status'],'financial_terms_correction_may_affect_annual_statements')

    def test_unrecognised_chapter_is_not_assumed_nonfinancial(self):
        self.put(self.viewer(ROOT),notice().replace('I. 모집 또는 매출에 관한 일반사항','II. 사업의 내용'))
        self.assertEqual(self.resolve()['status'],'financial_terms_amendment_scope_unverified')

    def test_incomplete_or_ambiguous_overview_is_not_accepted(self):
        for body in [notice().replace('<td>정정 후</td>','<td>기타</td>'),notice()+notice()]:
            with self.subTest(body=body[:30]):
                self.put(self.viewer(ROOT),body)
                self.assertEqual(self.resolve()['status'],'financial_terms_amendment_overview_not_unique')

    def test_note_pairs_and_actual_after_body_must_both_exist(self):
        self.put(self.viewer(ROOT),notice().replace('<p>[정정 후]</p>',''))
        self.assertNotEqual(self.resolve()['status'],'official_section_downloaded')
        with self.assertRaises(ValueError):
            nonfinancial_terms_scope(notice().replace('<td>주1)</td><td>주1)</td>','<td>주1)</td><td>주2)</td>'))

    def test_reverse_link_and_latest_prospectus_are_required(self):
        self.put_index(PROSPECTUS,index(PROSPECTUS,'2023.03.15',financial=True,reverse=False))
        self.assertEqual(self.resolve()['status'],'financial_related_prospectus_reverse_link_missing')
        self.put_index(PROSPECTUS,index(PROSPECTUS,'2023.03.15',financial=True,newer=True))
        self.assertEqual(self.resolve()['status'],'financial_related_prospectus_not_latest')

    def test_future_candidate_and_withdrawal_are_not_used(self):
        self.assertEqual(self.resolve(cutoff='2023-03-16 21:00')['status'],'financial_actual_publication_after_cutoff')
        self.put_index(ROOT,index(ROOT,'2023.03.16').replace('</select><select id="ref">',
            '<option value="rcpNo=20230317000003">2023.03.17 철회신고서</option></select><select id="ref">'))
        self.assertEqual(self.resolve()['status'],'financial_terms_family_withdrawn')

    def test_another_later_full_or_ambiguous_correction_blocks_fallback(self):
        self.put_index(ROOT,index(ROOT,'2023.03.16').replace('증권발행조건확정','정정신고서'))
        self.assertEqual(self.resolve()['status'],'financial_later_correction_scope_requires_review')

    def test_missing_cache_never_turns_into_a_live_request_or_nonpublication(self):
        (self.cache/(hashlib.sha256(self.viewer(ROOT).encode()).hexdigest()+'.json')).unlink()
        result=self.resolve()
        self.assertEqual(result['status'],'official_source_not_cached')
        self.assertFalse(result['absence_confirmed'])

    def test_stage_path_reuses_selection_but_not_numeric_approval_or_future_documents(self):
        from data.pipelines.stage_snapshots import financials_from_selected_document
        url='https://dart.fss.or.kr/dsaf001/main.do?rcpNo='+ROOT
        text=index(ROOT,'2023.03.16')
        selected={'proof':{'receipt':ROOT,'source_url':url,'published_at':'2023-03-16',
                           'index_sha256':hashlib.sha256(text.encode()).hexdigest()}}
        frame,evidence=financials_from_selected_document(self.dart,self.raw,selected,CORP,True,
                                                       cutoff='2023-03-20 21:00')
        self.assertTrue(frame.empty)
        self.assertEqual(evidence['source_selection']['selected_receipt'],PROSPECTUS)
        self.assertNotEqual(evidence['status'],'annual_accounts_extracted')
        frame,evidence=financials_from_selected_document(self.dart,self.raw,selected,CORP,True,
                                                       cutoff='2023-03-15 21:00')
        self.assertTrue(frame.empty)
        self.assertEqual(evidence['status'],'financial_actual_publication_after_cutoff')
        self.dart.session.get.assert_not_called()


if __name__=='__main__':
    unittest.main()
