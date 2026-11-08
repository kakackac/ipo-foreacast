import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from data.pipelines.disclosure_financials import (
    collect_disclosure_financials, parse_disclosure_financials, public_financial_section,
    select_financial_node, financial_records,
)
from data.pipelines.financial_history import summarize_asof


def table(rows):
    return '<table>' + ''.join('<tr>' + ''.join('<td>' + c + '</td>' for c in r) + '</tr>' for r in rows) + '</table>'


def summary(header='2022년', standard=True, period=None):
    rows = [['구분', header]]
    if standard:
        rows.append(['회계처리 기준', 'K-IFRS'])
    rows += [['자산총계', '300'], ['부채총계', '100'], ['자본총계', '200']]
    if period:
        rows.append(['구분', period])
    rows += [['매출액', '1000'], ['영업이익', '(50)'], ['당기순이익', '(60)']]
    return rows


def parse(html):
    return parse_disclosure_financials(html, '12345678', '20230315000001', '2023-03-15', 'https://dart.fss.or.kr/report/viewer.do')


def financial_index(receipt='20230315000001',date='2023.03.15',corp='12345678',name='테스트'):
    text=f"<title>{name}/투자설명서/{date}</title>openCorpInfoNew('{corp}');"
    text+=f'<select id="family"><option selected value="rcpNo={receipt}">{date} 투자설명서</option></select>'
    text+='var node1 = {};'
    for key,value in dict(text='III. 재무에 관한 사항',rcpNo=receipt,dcmNo='1',eleId='2',offset='3',length='4',dtd='dart3.xsd').items():
        text+=f"node1['{key}'] = \"{value}\";"
    return text


class DisclosureFinancialTests(unittest.TestCase):
    def test_document_geometry_reuse_does_not_reuse_financial_approval(self):
        from data.pipelines import disclosure_financials as module
        body='1. 요약별도재무정보 (단위: 원)'+table(summary())
        with tempfile.TemporaryDirectory() as root,patch.object(module,'BeautifulSoup',wraps=module.BeautifulSoup) as soup,patch.object(
                module,'primary_annual_balances',wraps=module.primary_annual_balances) as interpretation:
            first=module.parse_disclosure_financials(body,'12345678','20230315000001','2023-03-15','official',document_cache=root)
            second=module.parse_disclosure_financials(body,'12345678','20230315000001','2023-03-15','official',document_cache=root)
            pd.testing.assert_frame_equal(first,second)
            self.assertEqual(soup.call_count,1);self.assertEqual(interpretation.call_count,2)

    def test_document_geometry_invalidates_changed_body_or_contract_and_rejects_corruption(self):
        from data.pipelines import disclosure_financials as module
        with tempfile.TemporaryDirectory() as root:
            module.financial_document_structure('<table><tr><td>one</td></tr></table>',root)
            module.financial_document_structure('<table><tr><td>two</td></tr></table>',root)
            self.assertEqual(len(list(Path(root).glob('*.json'))),2)
            with patch.object(module,'financial_document_structure_contract',return_value='changed'):
                module.financial_document_structure('<table><tr><td>one</td></tr></table>',root)
            self.assertEqual(len(list(Path(root).glob('*.json'))),3)
            for path in Path(root).glob('*.json'):
                saved=json.loads(path.read_text());saved['result']['tables']=[];path.write_text(json.dumps(saved))
            with self.assertRaisesRegex(RuntimeError,'document_structure_integrity_failed'):
                module.financial_document_structure('<table><tr><td>one</td></tr></table>',root)

    def test_document_geometry_contract_excludes_approval_rules_but_tracks_normalizer_and_runtime(self):
        from data.pipelines import disclosure_financials as module
        before=module.financial_document_structure_contract()
        with patch.object(module,'ifrs_preparation_statements',return_value=[]):
            self.assertEqual(module.financial_document_structure_contract(),before)
        original=module.clean
        def changed(value):return original(value)
        with patch.object(module,'clean',changed):
            self.assertNotEqual(module.financial_document_structure_contract(),before)
        with patch.object(module.bs4,'__version__','changed'):
            self.assertNotEqual(module.financial_document_structure_contract(),before)
        with patch.dict(module.ACCOUNTS,{'새계정':'net_income'}):
            self.assertNotEqual(module.financial_document_structure_contract(),before)

    def interim_comparative(self):
        html='2. 연결재무제표 연결재무상태표 '
        html+='제14기 3분기 2023년 9월 30일 현재 제13기 2022년 12월 31일 현재 (단위: 원)'
        html+=table([['과목','제14기 3분기','제13기'],['자산총계','400','300'],
                     ['부채총계','110','100'],['자본총계','290','200']])
        html+='연결손익계산서 제14기 3분기 2023년 1월 1일부터 2023년 9월 30일까지 '
        html+='제13기 2022년 1월 1일부터 2022년 12월 31일까지 (단위: 원)'
        html+=table([['과목','제14기 3분기','제13기'],['매출액','900','1000'],
                     ['영업이익','45','50'],['당기순이익','36','40']])
        html+='3. 연결재무제표 주석 '
        html+='요약분기연결재무제표는 2023년 9월 30일 현재 한국채택국제회계기준에 따라 작성되었습니다. '
        html+='회계추정은 2022년 12월 31일로 종료하는 회계연도에 대한 연차연결재무제표와 동일합니다.'
        return html

    def test_same_statement_interim_framework_binds_only_identified_prior_annual_columns(self):
        html=self.interim_comparative()
        frame=parse_disclosure_financials(html,'12345678','20231101000001','2023-11-01','official')
        self.assertEqual(len(frame),6);self.assertEqual(set(frame.year),{2022})
        self.assertEqual(set(frame.fs_div),{'CFS'})
        self.assertEqual(frame.loc[frame.account_name_en.eq('revenue'),'amount'].iloc[0],1000)
        self.assertTrue(frame.accounting_standard_evidence.str.contains('same_scope_ifrs_interim_prior_annual_comparative_v1').all())

    def test_interim_basis_does_not_approve_unlinked_year_scope_or_planned_compliance(self):
        for original,replacement in [
                ('회계추정은 2022년 12월 31일로','회계추정은 2021년 12월 31일로'),
                ('연차연결재무제표','연차별도재무제표'),
                ('한국채택국제회계기준에 따라 작성되었습니다','한국채택국제회계기준에 따라 작성할 예정입니다'),
                ('한국채택국제회계기준에 따라 작성되었습니다','한국채택국제회계기준 개정사항을 검토 중입니다')]:
            with self.subTest(replacement=replacement):
                html=self.interim_comparative().replace(original,replacement)
                self.assertTrue(parse_disclosure_financials(html,'12345678','20231101000001','2023-11-01','official').empty)

    def test_statutory_issuer_preparation_keeps_scope_and_annual_columns(self):
        base='1. 요약연결재무정보 (단위: 원)'+table(summary(standard=False))
        note=('연결실체는 주식회사의외부감사에관한법률 제13조에서 규정하고 있는 '
              '국제회계기준위원회의 기준을 채택하여 정한 한국채택국제회계기준에 따라 연결재무제표를 작성하였습니다.')
        frame=parse(base+note)
        self.assertEqual(len(frame),6);self.assertEqual(set(frame.fs_div),{'CFS'})
        self.assertEqual(len(parse(base+'종속기업의 현황은 주석4에서 설명하고 있습니다. 2. 재무제표 작성기준 '+note)),6)
        for body in [base.replace('요약연결','요약별도')+note,
                     base+note.replace('작성하였습니다','작성할 예정입니다'),
                     base+note.replace('연결실체는','비교회사의 연결실체는'),
                     base+note.replace('연결실체는','종속회사인 회사는')]:
            with self.subTest(body=body[-120:]):self.assertTrue(parse(body).empty)
        ofs='1. 요약별도재무정보 (단위: 원)'+table(summary(standard=False))
        self.assertEqual(len(parse(ofs+note.replace('연결실체는','당사는').replace('연결재무제표','재무제표'))),6)
        self.assertTrue(parse(base.replace('2022년','2022년 반기')+note).empty)

    def test_consolidated_total_net_income_is_not_owner_specific_income(self):
        body='1. 요약연결재무정보 (단위: 원)'+table(summary())
        total=body.replace('당기순이익','연결총당기순이익')
        frame=parse(total)
        self.assertEqual(frame.loc[frame.account_name_en.eq('net_income'),'amount'].tolist(),[-60])
        for label in ['지배기업소유주에게귀속되는당기순이익','비지배지분당기순이익']:
            self.assertNotIn('net_income',parse(body.replace('당기순이익',label)).account_name_en.tolist())

    def test_interim_1034_parentheses_and_annual_period_wording(self):
        html=self.interim_comparative().replace(
            '한국채택국제회계기준','한국채택국제회계기준 기업회계기준서 제1034호(중간재무보고)')
        html=html.replace('종료하는 회계연도','종료되는 회계기간')
        frame=parse_disclosure_financials(html,'12345678','20231101000001','2023-11-01','official')
        self.assertEqual(len(frame),6);self.assertEqual(set(frame.year),{2022})
        for original,replacement in [('제1034호','제9999호'),('연차연결재무제표','연차별도재무제표'),
                ('2022년 12월 31일로','2021년 12월 31일로'),('연차연결재무제표','분기연결재무제표'),
                ('따라 작성되었습니다','따라 작성할 예정입니다')]:
            with self.subTest(replacement=replacement):
                self.assertTrue(parse_disclosure_financials(html.replace(original,replacement),
                    '12345678','20231101000001','2023-11-01','official').empty)
        no_label=html.replace('요약분기연결','중간요약연결').replace('(중간재무보고)','')
        self.assertEqual(len(parse_disclosure_financials(no_label,'12345678','20231101000001','2023-11-01','official')),6)

    def test_explicit_previous_annual_reference_without_end_suffix(self):
        html=self.interim_comparative().replace('2022년 12월 31일로 종료하는 회계연도에 대한 연차연결재무제표',
                                              '전기 연차연결재무제표')
        frame=parse_disclosure_financials(html,'12345678','20231101000001','2023-11-01','official')
        self.assertEqual(len(frame),6);self.assertEqual(set(frame.year),{2022})
        for original,replacement in [('전기 연차연결','전기 연결'),('전기 연차연결','전기 연차별도'),
                ('한국채택국제회계기준','일반기업회계기준')]:
            with self.subTest(replacement=replacement):
                self.assertTrue(parse_disclosure_financials(html.replace(original,replacement),
                    '12345678','20231101000001','2023-11-01','official').empty)

    def test_verified_primary_annual_framework_reaches_summary_when_primary_columns_are_unhandled(self):
        summary_html='1. 요약연결재무정보 (단위: 원)'+table(summary(standard=False))
        primary=self.interim_comparative().replace('<td>제13기</td>','<td>전기</td>')
        html=summary_html+primary
        frame=parse_disclosure_financials(html,'12345678','20231101000001','2023-11-01','official')
        self.assertEqual(len(frame),6);self.assertEqual(set(frame.year),{2022})
        self.assertTrue(frame.extraction_rule_id.eq('summary_issuer_statement_scope_v2').all())
        self.assertTrue(frame.accounting_standard_evidence.str.contains('same_scope_ifrs_interim_prior_annual_comparative_v1').all())
        for body in [html.replace('요약연결재무정보','요약별도재무정보'),
                     html.replace('2022년 12월 31일로 종료하는','2021년 12월 31일로 종료하는'),
                     html.replace('요약연결재무정보','요약연결재무정보 K-GAAP'),
                     html.replace('따라 작성되었습니다','따라 작성할 예정입니다')]:
            with self.subTest(body=body[-100:]):
                self.assertTrue(parse_disclosure_financials(body,'12345678','20231101000001','2023-11-01','official').empty)

    def test_interim_comparative_cannot_override_explicit_gaap_or_invent_full_year_income(self):
        html=self.interim_comparative().replace("<td>제13기</td>","<td>제13기 K-GAAP</td>")
        self.assertTrue(parse_disclosure_financials(html,'12345678','20231101000001','2023-11-01','official').empty)
        html=self.interim_comparative().replace(
            '제13기 2022년 1월 1일부터 2022년 12월 31일까지',
            '제13기 3분기 2022년 1월 1일부터 2022년 9월 30일까지')
        frame=parse_disclosure_financials(html,'12345678','20231101000001','2023-11-01','official')
        self.assertEqual(set(frame.account_name_en),{'total_assets','total_liabilities','equity'})

    def test_kifrs_1034_preparation_and_explicit_prior_annual_reference(self):
        html=self.interim_comparative().replace('한국채택국제회계기준',"기업회계기준서 제1034호 '중간재무보고'")
        html=html.replace('2022년 12월 31일로 종료하는 회계연도에 대한 연차연결재무제표','전기말 연차연결재무제표')
        frame=parse_disclosure_financials(html,'12345678','20231101000001','2023-11-01','official')
        self.assertEqual(len(frame),6);self.assertEqual(set(frame.year),{2022})
        self.assertTrue(frame.accounting_standard_evidence.str.contains('resolved_from_interim_year=2023').all())
        for original,replacement in [
                ('제1034호','제9999호'), ('전기말 연차연결재무제표','전기 연결재무제표'),
                ('전기말 연차연결재무제표','전기말 연차별도재무제표'),
                ('따라 작성되었습니다','따라 작성할 예정입니다')]:
            with self.subTest(replacement=replacement):
                bad=html.replace(original,replacement)
                self.assertTrue(parse_disclosure_financials(bad,'12345678','20231101000001','2023-11-01','official').empty)

    def test_remote_prior_annual_reference_cannot_bind_an_unrelated_interim_declaration(self):
        html=self.interim_comparative().replace('2022년 12월 31일로 종료하는 회계연도에 대한 연차연결재무제표',
                                               '다른 공시 설명입니다. '+'설명 '*300+'전기말 연차연결재무제표')
        self.assertTrue(parse_disclosure_financials(html,'12345678','20231101000001','2023-11-01','official').empty)

    def test_local_audit_adoption_since_year_never_approves_earlier_converted_year(self):
        rows=[['구분','2014년','2013년','2012년']]
        rows += [[r[0],r[1],r[1],r[1]] for r in summary(standard=False)[1:]]
        note=('주1) 2012년 사업연도까지 일반기업회계기준으로 수감하였으며, '
              '2013년 사업연도부터 한국채택국제회계기준(K-IFRS)를 적용한 감사보고서를 발행하였습니다. '
              '주2) 2012년 수치는 K-IFRS로 변환된 감사받지 않은 수치입니다.')
        html='1. 요약별도재무정보 (단위: 원)'+table(rows)+table([[note]])
        frame=parse(html)
        self.assertEqual(set(frame.year),{2013,2014})
        self.assertEqual(set(frame.fs_div),{'OFS'})
        self.assertEqual(len(frame),12)
        self.assertTrue(frame.accounting_standard_evidence.str.contains('2013년 사업연도부터',regex=False).all())
        self.assertTrue(parse(html.replace('발행하였습니다','발행할 예정입니다')).empty)
        self.assertTrue(parse(html.replace('적용한 감사보고서','적용하지 않은 감사보고서')).empty)

    def test_subsidiary_adoption_note_does_not_approve_issuer(self):
        note='주1) 종속회사는 2013년 사업연도부터 한국채택국제회계기준(K-IFRS)를 적용한 감사보고서를 발행하였습니다.'
        html='1. 요약별도재무정보 (단위: 원)'+table(summary('2014년',standard=False))+table([[note]])
        self.assertTrue(parse(html).empty)

    def test_local_summary_standard_can_bind_primary_without_guessing_missing_summary_unit(self):
        html='1. 요약별도재무정보 당사의 재무제표는 한국채택국제회계기준에 따라 작성되었습니다.'
        html+=table(summary(standard=False))
        html+='4. 재무제표 재무상태표 제13기 2022년 12월 31일 현재 (단위: 원)'
        html+=table([['과목','제13기'],['자산총계','300'],['부채총계','100'],['자본총계','200']])
        html+='손익계산서 제13기 2022년 1월 1일부터 2022년 12월 31일까지 (단위: 원)'
        html+=table([['과목','제13기'],['매출액','1000'],['영업이익','50'],['당기순이익','40']])
        frame=parse(html)
        self.assertEqual(set(frame.year),{2022});self.assertEqual(len(frame),6)
        self.assertEqual(set(frame.table_id),{1,2})
        self.assertTrue(frame.extraction_rule_id.eq('detailed_annual_period_and_basis_v1').all())
        self.assertTrue(frame.accounting_standard_evidence.str.contains('issuer_summary_year_standard_bound_to_primary_v1').all())
        self.assertTrue(parse(html.replace('4. 재무제표','2. 연결재무제표')).empty)
        self.assertTrue(parse(html.replace('한국채택국제회계기준에 따라 작성되었습니다','한국채택국제회계기준을 적용할 예정입니다')).empty)

    def test_unqualified_summary_basis_is_corroborated_by_primary_balances(self):
        html='1. 요약재무정보 (단위: 원)'+table(summary())
        html+='2. 연결재무제표 연결재무상태표 (단위: 원)'+table([
            ['과목','2022년 12월말'],['자산','300'],['부채','100'],['자본','200']])
        html+='4. 재무제표 재무상태표 (단위: 원)'+table([
            ['과목','2022년 12월말'],['자산','200'],['부채','50'],['자본','150']])
        frame=parse(html)
        self.assertEqual(set(frame.fs_div),{'CFS'})
        self.assertTrue(frame.primary_balance_scope_evidence.map(bool).all())
        self.assertTrue(frame.statement_basis_explicit.eq(False).all())
        # Explicit declarations are not overwritten by a numeric resemblance.
        self.assertEqual(set(parse(html.replace('1. 요약재무정보','1. 요약별도재무정보')).fs_div),{'OFS'})

    def test_ambiguous_or_unmatched_primary_fingerprints_do_not_approve_a_basis(self):
        body='1. 요약재무정보 (단위: 원)'+table(summary())
        connected='2. 연결재무제표 연결재무상태표 (단위: 원)'+table([
            ['과목','2022년 12월말'],['자산','300'],['부채','100'],['자본','200']])
        same='4. 재무제표 재무상태표 (단위: 원)'+table([
            ['과목','2022년 12월말'],['자산','300'],['부채','100'],['자본','200']])
        self.assertTrue(parse(body+connected+same).empty)
        self.assertTrue(parse(body+connected.replace('300','400')).empty)

    def test_summary_basis_aliases_reset_scope_instead_of_inheriting(self):
        for connected, separate in [
                ('연결 요약재무현황', '별도 요약재무현황'),
                ('요약 재무정보 - 연결', '요약 별도 재무정보 - 별도'),
                ('요약재무정보(연결)', '요약재무정보(별도)'),
                ('요약(연결) 재무정보', '요약(별도) 재무정보'),
                ('요약연결재무정보', '요약개별재무정보')]:
            with self.subTest(connected=connected, separate=separate):
                html=connected+' (단위: 원)'+table(summary())
                other=summary();other[-2]=['영업이익','70']
                html+=separate+' (단위: 원)'+table(other)
                frame=parse(html)
                self.assertEqual(set(frame.loc[frame.table_id.eq(0),'fs_div']),{'CFS'})
                self.assertEqual(set(frame.loc[frame.table_id.eq(1),'fs_div']),{'OFS'})
                self.assertTrue(frame.entity_scope.eq('issuer').all())
                self.assertEqual(summarize_asof(frame,'12345678','2023-04-01')['operating_income'],-50)

    def test_subsidiary_summary_owner_is_preserved_until_next_issuer_block(self):
        html='가. 요약연결재무정보 (단위: 원)'+table(summary())
        html+='다. 종속회사의 요약재무정보 본 요약재무정보는 당사의 종속회사인 자회사A의 별도재무제표입니다. (단위: 원)'
        html+=table(summary())+'(2) 자회사B (단위: 원)'+table(summary())
        html+='라. 요약개별재무정보 (단위: 원)'+table(summary())
        details={}
        frame=parse_disclosure_financials(html,'12345678','20230315000001','2023-03-15','https://dart.fss.or.kr/report/viewer.do',details)
        self.assertEqual(set(frame.table_id),{0,3})
        self.assertEqual(set(frame.loc[frame.table_id.eq(3),'fs_div']),{'OFS'})
        self.assertEqual(details['rejected_table_or_column_counts']['non_issuer_statement_scope'],2)

    def test_multicell_title_grid_preserves_explicit_scope_and_fiscal_mapping(self):
        title=table([['요약연결재무현황'],['제7기말 2022.12.31 현재'],['','','(단위: 원)']])
        html='1. 요약재무정보'+title+table(summary('제7기'))
        frame=parse(html)
        self.assertEqual(set(frame.fs_div),{'CFS'})
        self.assertEqual(set(frame.year),{2022})
        self.assertEqual(frame.table_id.unique().tolist(),[0])

    def test_unqualified_summary_explicit_separate_introduction_and_conflict(self):
        html='요약재무정보 본 요약재무정보는 별도재무제표입니다. (단위: 원)'+table(summary())
        self.assertEqual(set(parse(html).fs_div),{'OFS'})
        self.assertTrue(parse('요약재무정보 - 연결 본 요약재무정보는 별도재무제표입니다. (단위: 원)'+table(summary())).empty)

    def test_previous_ifrs_note_does_not_override_next_explicit_gaap_heading(self):
        for title in ['요약재무정보(K-GAAP 기준)','K-GAAP 요약재무제표']:
            with self.subTest(title=title):
                html='1. 요약재무정보 (단위: 원)'+table(summary())
                html+='주) 요약재무제표는 한국채택국제회계기준(K-IFRS)에 따라 작성되었습니다. 라. '+title+' (단위: 원)'
                html+=table(summary('2021년',standard=False))
                self.assertEqual(set(parse(html).year),{2022})

    def test_adjustment_amount_columns_are_not_annual_account_values(self):
        for label in ['수정금액','수정 금액','조정금액','차이금액']:
            with self.subTest(label=label):
                html='1. 요약재무정보 K-IFRS (단위: 원)'+table(summary('2022년 '+label,False))
                self.assertTrue(parse(html).empty)

    def test_summary_child_statement_headings_set_scope_without_repeating_summary(self):
        html='1. 요약재무정보 가. 연결재무제표 (단위: 원)'+table(summary())
        html+='나. 재무제표 (단위: 원)'+table(summary())
        frame=parse(html)
        self.assertEqual(set(frame.loc[frame.table_id.eq(0),'fs_div']),{'CFS'})
        self.assertEqual(set(frame.loc[frame.table_id.eq(1),'fs_div']),{'OFS'})
        self.assertTrue(parse('2. 연결재무제표 가. 연결재무제표 (단위: 원)'+table(summary())).empty)
        self.assertTrue(parse('1. 요약재무정보 해당사항 없습니다. 2. 연결재무제표 가. 연결재무제표 (단위: 원)'+table(summary())).empty)
        html='1. 요약재무정보 아래 재무제표는 한국채택국제회계기준에 따라 작성되었습니다. 가. 연결재무제표 (단위: 원)'+table(summary(standard=False))
        self.assertEqual(set(parse(html).fs_div),{'CFS'})

    def test_child_statement_heading_cannot_leave_a_subsidiary_block(self):
        html='종속회사의 요약재무정보 (단위: 원)'+table(summary())
        html+='가. 별도재무제표 (단위: 원)'+table(summary())
        self.assertTrue(parse(html).empty)

    def test_conflicting_summary_and_primary_balance_units_are_not_silently_corrected(self):
        html='1. 요약재무정보 (단위: 백만원)'+table(summary())
        html+='4. 재무제표 재무상태표 (단위: 원)'+table([
            ['구분','2022년 12월말'],['자산총계','300'],['부채총계','100'],['자본총계','200']])
        details={}
        frame=parse_disclosure_financials(html,'12345678','20230315000001','2023-03-15','https://dart.fss.or.kr/report/viewer.do',details)
        self.assertTrue(frame.empty)
        self.assertEqual(len(details['official_statement_unit_conflicts']),1)
        self.assertEqual(len(details['official_statement_unit_conflicts'][0]['evidence']),3)
        from data.pipelines.historical_financial_reconciliation import missing_category
        self.assertEqual(missing_category('annual_financial_table_format_unhandled',details),
                         'official_statement_unit_conflict_requires_review')
        alternate=html.replace('자산총계','자산').replace('부채총계','부채').replace('자본총계','자본')
        # Keep explicit summary totals; only the primary statement uses short labels.
        alternate=html[:html.index('4. 재무제표')]+alternate[alternate.index('4. 재무제표'):]
        self.assertTrue(parse(alternate).empty)

    def test_large_financial_amount_serialization_preserves_precision(self):
        amount=float('35579597308')*1_000_000
        self.assertEqual(financial_records(pd.DataFrame([{'amount':amount}]))[0]['amount'],amount)

    def test_public_empty_financial_chapter_is_retained_without_fabricated_values(self):
        body='<html><head><link href="/css/report_xml.css"></head><body><p class="section-1">Ⅲ. 재무에 관한 사항</p><p class="section-2">1. 요약재무정보</p></body></html>'
        dart=Mock();index=Mock();index.text=financial_index();response=Mock();response.text=body
        dart.session.get.side_effect=[index,response]
        with tempfile.TemporaryDirectory() as root:
            payload=public_financial_section(dart,'20230315000001','테스트',Path(root)/'sections',corp_code='12345678')
            self.assertEqual(payload['status'],'official_section_downloaded')
            self.assertEqual(payload['html'],body)
            self.assertTrue(parse(body).empty)
        from data.pipelines.historical_financial_reconciliation import missing_category
        self.assertEqual(missing_category('official_financial_section_without_tables',{}),
                         'official_section_without_value_tables_requires_lineage_review')

    def test_financial_chapter_not_its_other_financial_matters_subsection(self):
        index=financial_index()
        other=index[index.index('var node1'):].replace('node1','node2').replace('III. 재무에 관한 사항','8. 기타 재무에 관한 사항').replace('"3"','"300"')
        for title in ['III. 재무에 관한 사항','Ⅲ. 재무에 관한 사항','3. 재무에 관한 사항']:
            with self.subTest(title=title):
                selected=select_financial_node(index.replace('III. 재무에 관한 사항',title)+other,'20230315000001')
                self.assertEqual(selected['offset'],'3')
        self.assertIsNone(select_financial_node(index+other,'20230315000002'))
        self.assertIsNone(select_financial_node(other,'20230315000001'))

    def test_identical_viewer_nodes_deduplicate_but_different_financial_chapters_do_not(self):
        index=financial_index();node=index[index.index('var node1'):].replace('node1','node2')
        self.assertEqual(select_financial_node(index+node,'20230315000001')['offset'],'3')
        self.assertIsNone(select_financial_node(index+node.replace('"3"','"300"'),'20230315000001'))

    def test_financial_index_javascript_whitespace_does_not_change_selection(self):
        index=financial_index().replace('var node1 = {};','var  node1={ };')
        self.assertEqual(select_financial_node(index,'20230315000001')['eleId'],'2')

    def test_explicit_annual_consolidated_reference_approves_only_its_year_and_basis(self):
        body='1. 요약연결재무정보 (단위: 원)'+table(summary(standard=False))
        note='이 중간연결재무제표는 2023년 9월 30일 현재의 자료입니다. 동 연결재무제표에 대한 이해를 위해서는 한국채택국제회계기준에 따라 작성된 2022년 12월 31일자로 종료하는 회계연도에 대한 연차연결재무제표를 함께 이용해야 합니다.'
        self.assertEqual(len(parse(body+note)),6)
        self.assertTrue(parse(body.replace('요약연결재무정보','요약별도재무정보')+note).empty)
        self.assertTrue(parse(body+note.replace('2022년 12월 31일자로','2021년 12월 31일자로')).empty)

    def test_spaced_consolidated_preparation_cannot_approve_separate_values(self):
        note='연결 재무제표는 한국채택국제회계기준에 따라 작성되었습니다.'
        ofs='1. 요약별도재무정보 (단위: 원)'+table(summary(standard=False))
        self.assertTrue(parse(ofs+note).empty)
        cfs='1. 요약연결재무정보 (단위: 원)'+table(summary(standard=False))
        self.assertEqual(set(parse(cfs+note).fs_div),{'CFS'})
        for wording in ['분기 연결 재무제표는','반기 재무제표는','중간 재무제표 는']:
            self.assertTrue(parse(ofs+wording+' 한국채택국제회계기준에 따라 작성되었습니다.').empty)

    def test_interim_footnote_and_non_year_end_preparation_do_not_approve_annual_columns(self):
        body='1. 요약별도재무정보 (단위: 원)'+table(summary(standard=False))
        for note in ['주1) 제9기 반기 재무제표는 한국채택국제회계기준에 따라 작성되었습니다.',
                     '주1) 재무제표는 2022년 9월 30일 현재 유효한 한국채택국제회계기준에 따라 작성되었습니다.']:
            self.assertTrue(parse(body+table([[note]])).empty)

    def test_year_restricted_preparation_does_not_leak_to_another_year(self):
        body='1. 요약별도재무정보 (단위: 원)'+table(summary(standard=False))
        note='재무제표는 2021년 12월 31일 현재 한국채택국제회계기준에 따라 작성되었습니다.'
        self.assertTrue(parse(body+note).empty)

    def test_detailed_annual_statements_need_scope_period_and_year_bound_basis(self):
        balance=table([['연 결 재 무 상 태 표'],['제13기 2022년 12월 31일 현재'],['회사 (단위: 원)']])
        balance+=table([['과목','제13기말'],['자산총계','300'],['부채총계','100'],['자본총계','200']])
        income=table([['연 결 포 괄 손 익 계 산 서'],['제14기 반기 2023년 01월 01일부터 2023년 06월 30일까지'],
                      ['제13기 2022년 01월 01일부터 2022년 12월 31일까지'],['회사 (단위: 원)']])
        income+=table([['과목','제14기 반기','제13기'],['매출액','400','1000'],['영업손실','10','50'],['당기순손실','12','60']])
        notes='<p>3. 연결재무제표 주석</p><p>동 중간재무제표를 이해하려면 한국채택국제회계기준에 따라 작성된 2022년 12월 31일자로 종료하는 회계연도에 대한 연차재무제표를 함께 이용하여야 합니다.</p>'
        html='<p>1. 요약재무정보 해당사항 없습니다.</p><p>2. 연결재무제표</p>'+balance+income+notes
        frame=parse(html);result=summarize_asof(frame,'12345678','2023-04-01')
        self.assertEqual(result['revenue'],1000);self.assertEqual(result['operating_income'],-50)
        self.assertEqual(result['net_income'],-60);self.assertEqual(result['total_assets'],300)
        self.assertEqual(set(frame.year),{2022});self.assertEqual(set(frame.fs_div),{'CFS'})
        self.assertTrue(frame.extraction_rule_id.eq('detailed_annual_period_and_basis_v1').all())
        self.assertTrue(parse(html.replace('2022년 12월 31일자로 종료하는 회계연도','2021년 12월 31일자로 종료하는 회계연도')).empty)
        self.assertTrue(parse(html.replace('(단위: 원)','(단위: USD)')).empty)
        self.assertTrue(parse(html.replace(notes,'<p>3. 연결재무제표 주석</p>분기재무제표는 한국채택국제회계기준에 따라 작성되었습니다.')).empty)

    def test_detailed_ifrs_scope_cannot_leak_into_separate_statement(self):
        body='<p>4. 재무제표</p>'+table([['재무상태표'],['제13기 2022년 12월 31일 현재'],['회사 (단위: 원)']])
        body+=table([['과목','제13기말'],['자산총계','300'],['부채총계','100'],['자본총계','200']])
        html='<p>2. 연결재무제표</p><p>3. 연결재무제표 주석</p><p>연결재무제표는 한국채택국제회계기준에 따라 작성되었습니다.</p>'+body
        self.assertTrue(parse(html).empty)

    def test_intertable_scope_boundaries_do_not_reassign_previous_notes(self):
        balance=table([['재무상태표'],['제13기 2022년 12월 31일 현재'],['회사 (단위: 원)']])
        balance+=table([['과목','제13기말'],['자산총계','300'],['부채총계','100'],['자본총계','200']])
        note='재무제표는 한국채택국제회계기준에 따라 작성되었습니다.'
        crossing='2. 연결재무제표 3. 연결재무제표 주석 '+note+'4. 재무제표 '+balance
        self.assertTrue(parse(crossing).empty)
        cfs='2. 연결재무제표 '+balance+'3. 연결재무제표 주석 4. 재무제표 작성기준 '+note
        frame=parse(cfs)
        self.assertEqual(set(frame.fs_div),{'CFS'});self.assertEqual(len(frame),3)

    def test_partial_summary_does_not_prevent_a_coherent_primary_statement_group(self):
        balance=table([['재무상태표'],['제13기 2022년 12월 31일 현재'],['회사 (단위: 원)']])
        balance+=table([['과목','제13기말'],['자산총계','300000001'],['부채총계','100000001'],['자본총계','200000000']])
        income=table([['손익계산서'],['제13기 2022년 1월 1일부터 2022년 12월 31일까지'],['회사 (단위: 원)']])
        income+=table([['과목','제13기'],['매출액','1000000000'],['영업이익','50000000'],['당기순이익','40000000']])
        note='5. 재무제표 주석 한국채택국제회계기준에 따라 작성된 2022년 12월 31일자로 종료하는 회계연도에 대한 연차별도재무제표'
        html='1. 요약별도재무정보 (단위: 백만원)'+table(summary()[:5])+'4. 재무제표'+balance+income+note
        result=summarize_asof(parse(html),'12345678','2023-04-01')
        self.assertEqual(result['revenue'],1000000000)
        self.assertEqual(result['total_assets'],300000001)
        self.assertAlmostEqual(result['operating_margin'],.05)
        details={}
        conflict=html.replace("<td>300000001</td>","<td>400000001</td>")
        frame=parse_disclosure_financials(conflict,'12345678','20230315000001','2023-03-15','https://dart.fss.or.kr/report/viewer.do',details)
        self.assertTrue(frame.empty)
        self.assertEqual(len(details['official_statement_amount_conflicts']),1)
        predecessor=parse(html.replace('제13기 2022','제13기(분할전) 2022'))
        self.assertNotIn('revenue',predecessor.get('account_name_en',pd.Series(dtype=str)).tolist())

    def test_complete_precise_primary_group_is_not_replaced_by_rounded_summary(self):
        summary_rows=summary(standard=False);summary_rows[-2]=['영업이익','50'];summary_rows[-1]=['당기순이익','40']
        summary_html='1. 요약연결재무정보 (단위: 백만원)'+table(summary_rows)
        primary=self.interim_comparative()
        for old,new in [('300','300000001'),('100','100000001'),('200','200000000'),
                        ('1000','1000000000'),('50','50000000'),('40','40000000')]:
            primary=primary.replace('<td>'+old+'</td>','<td>'+new+'</td>')
        frame=parse(summary_html+primary)
        self.assertEqual(frame.loc[frame.account_name_en.eq('total_assets'),'amount'].tolist(),[300000001])
        self.assertTrue(frame.extraction_rule_id.eq('detailed_annual_period_and_basis_v1').all())
        incomplete=primary.replace('<tr><td>당기순이익</td><td>36</td><td>40000000</td></tr>','')
        frame=parse(summary_html+incomplete)
        self.assertEqual(frame.loc[frame.account_name_en.eq('total_assets'),'amount'].tolist(),[300000000])
        self.assertEqual(frame.loc[frame.account_name_en.eq('net_income'),'amount'].tolist(),[40000000])

    def test_partial_verified_summary_does_not_hide_complete_primary_income(self):
        summary_rows=summary(standard=False)
        summary_rows[-2]=['영업이익','50']
        summary_rows=summary_rows[:-1]
        head='1. 요약연결재무정보 (단위: 원)'+table(summary_rows)
        primary=self.interim_comparative()
        frame=parse(head+primary)
        income=frame[frame.account_name_en.isin(['revenue','operating_income','net_income'])]
        self.assertEqual(set(income.account_name_en),{'revenue','operating_income','net_income'})
        self.assertTrue(income.extraction_rule_id.eq('detailed_annual_period_and_basis_v1').all())
        self.assertEqual(income[income.account_name_en.eq('net_income')].amount.tolist(),[40])
        conflict=primary.replace('<td>50</td>','<td>500</td>')
        self.assertTrue(parse(head+conflict).empty)

    def test_shared_primary_framework_retains_precision_for_both_statement_scopes(self):
        rows=summary(standard=False);rows[-2]=['영업이익','50'];rows[-1]=['당기순이익','40']
        head='1. 요약연결재무정보 (단위: 백만원)'+table(rows)
        head+='나. 요약별도재무정보 (단위: 백만원)'+table(rows)
        cfs=self.interim_comparative()
        for old,new in [('300','300000001'),('100','100000001'),('200','200000000'),
                        ('1000','1000000000'),('50','50000000'),('40','40000000')]:
            cfs=cfs.replace('<td>'+old+'</td>','<td>'+new+'</td>')
        ofs=cfs.replace('2. 연결재무제표','4. 재무제표').replace('3. 연결재무제표 주석','5. 재무제표 주석').replace('연결','')
        frame=parse(head+cfs+ofs)
        for scope in ['CFS','OFS']:
            with self.subTest(scope=scope):
                assets=frame[frame.fs_div.eq(scope)&frame.account_name_en.eq('total_assets')]
                self.assertEqual(assets.amount.tolist(),[300000001])

    def test_adjacent_explicit_ifrs_preparation_basis_variant(self):
        html='1. 요약연결재무정보 (단위: 원)'+table(summary(standard=False))
        html+=table([['주1) 한국채택국제회계기준 작성기준에 따라 작성되었습니다.'],
                     ['주2) 전기 재무제표는 감사를 받았습니다.'],['주3) 비교표시는 감사를 받지 않았습니다.']])
        self.assertEqual(len(parse(html)),6)

    def test_korean_unit_spacing_does_not_change_currency_or_scale(self):
        for unit in ['백만원','백 만원','백 만 원']:
            result=summarize_asof(parse('1. 요약재무정보 (단위: '+unit+')'+table(summary())), '12345678','2023-04-01')
            self.assertEqual(result['revenue'],1_000_000_000)
        self.assertTrue(parse('1. 요약재무정보 (단위: 백만달러)'+table(summary())).empty)
    def test_annual_summary_without_date_range_and_negative_income(self):
        frame = parse('1. 요약재무정보 (단위: 백만원)' + table(summary()))
        result = summarize_asof(frame, '12345678', '2023-04-01')
        self.assertEqual(result['revenue'], 1_000_000_000)
        self.assertEqual(result['operating_margin'], -.05)
        self.assertEqual(result['debt_ratio'], .5)

    def test_unit_in_separate_one_cell_table_is_not_lost(self):
        html = '1. 요약재무정보' + table([['(단위: 백만원)']]) + table(summary())
        self.assertEqual(summarize_asof(parse(html), '12345678', '2023-04-01')['revenue'], 1_000_000_000)

    def test_heading_and_unit_in_separate_multicell_table_are_not_lost(self):
        html='1. 요약재무정보'+table([['요약재무제표(K-IFRS)'],['제7기 2022년 12월 31일 현재'],
                                           ['주식회사 테스트','','(단위 : 원)']])+table(summary(standard=False))
        self.assertEqual(summarize_asof(parse(html),'12345678','2023-04-01')['revenue'],1000)

    def test_numeric_non_total_accounts_are_not_part_of_year_header(self):
        rows = summary()
        rows.insert(2, ['유동자산', '2019'])
        self.assertEqual(set(parse('1. 요약재무정보 (단위: 원)' + table(rows)).year), {2022})

    def test_previous_cfs_footnote_cannot_change_next_ofs_scope(self):
        html = '1. 요약연결재무정보 (단위: 원)' + table(summary())
        html += '요약연결재무제표는 한국채택국제회계기준에 따라 작성되었습니다. 나. 요약별도재무정보 (단위: 원)'
        html += table(summary())
        frame = parse(html)
        self.assertEqual(set(frame.loc[frame.table_id.eq(0), 'fs_div']), {'CFS'})
        self.assertEqual(set(frame.loc[frame.table_id.eq(1), 'fs_div']), {'OFS'})

    def test_summary_statement_heading_is_not_dropped_when_trimming_scope(self):
        frame = parse('1. 요약재무정보 가. 요약재무제표(K-IFRS) (단위: 원)' + table(summary()))
        self.assertEqual(len(frame), 6)

    def test_income_period_overrides_annual_balance_header(self):
        frame = parse('1. 요약재무정보 (단위: 원)' + table(summary(period='2022.01.01~2022.09.30')))
        self.assertEqual(set(frame.account_name_en), {'total_assets', 'total_liabilities', 'equity'})

    def test_quarterly_column_is_not_annual(self):
        self.assertTrue(parse('1. 요약재무정보 (단위: 원)' + table(summary('2022년 3분기'))).empty)

    def test_month_range_is_not_misread_as_annual_income(self):
        frame = parse('1. 요약 연결 재무제표 한국채택국제회계기준 (단위: 원)'
                      + table(summary(period='2022년 1월~3월')))
        self.assertEqual(set(frame.account_name_en), {'total_assets', 'total_liabilities', 'equity'})

    def test_wrapped_annual_period_and_heading_accounting_standard(self):
        rows = summary(standard=False)
        rows[4:4] = [['기간', '(2022.01.01 ~'], ['', '2022.12.31)']]
        frame = parse('1. 요약재무정보 K-IFRS (단위: 천원)' + table(rows))
        self.assertEqual(frame.loc[frame.account_name_en.eq('revenue'), 'amount'].iloc[0], 1000000)

    def test_explicit_gaap_column_does_not_inherit_ifrs_from_balance(self):
        html = '1. 요약재무정보 (단위: 원)' + table(summary())
        html += table([['구분', '2022년'], ['회계기준', 'K-GAAP'],
                       ['매출액', '999'], ['영업이익', '99'], ['당기순이익', '88']])
        self.assertNotIn(999, parse(html).amount.tolist())

    def test_zero_padded_month_is_not_an_annual_balance(self):
        self.assertTrue(parse('1. 요약재무정보 (단위: 원)' + table(summary('2022년03월말'))).empty)

    def test_company_rename_requires_matching_official_corp_code(self):
        dart = Mock()
        index = financial_index(date='2023.03.16',name='변경회사')
        first, second = Mock(), Mock()
        first.text = index
        second.text = '재무' + table(summary())
        dart.session.get.side_effect = [first, second]
        with tempfile.TemporaryDirectory() as root:
            payload = public_financial_section(dart, '20230315000001', '이전회사', Path(root)/'sections',
                                               corp_code='12345678', published='2023-03-16')
        self.assertEqual(payload['status'], 'official_section_downloaded')
        self.assertEqual(payload['official_publication_date'], '2023-03-16')

    def test_matching_name_does_not_override_wrong_corp_code(self):
        dart = Mock()
        dart.session.get.return_value.text = "<title>테스트/투자설명서/2023.03.15</title>openCorpInfoNew('87654321', 'info');"
        with tempfile.TemporaryDirectory() as root:
            result = public_financial_section(dart, '20230315000001', '테스트', Path(root)/'sections', corp_code='12345678')
        self.assertEqual(result['status'], 'official_document_corp_code_mismatch')

    def test_adjacent_ifrs_footnote(self):
        frame = parse('1. 요약 연결재무정보 (단위: 원)' + table(summary(standard=False))
                      + table([['주1 한국채택국제회계기준(K-IFRS)에 따라 작성']]))
        self.assertEqual(set(frame.fs_div), {'CFS'})
        self.assertEqual(len(frame), 6)

    def test_adjacent_future_ifrs_plan_does_not_approve_current_accounts(self):
        html='1. 요약재무정보 (단위: 원)'+table(summary(standard=False))
        html+=table([['차기부터 한국채택국제회계기준에 따라 작성할 예정입니다.']])
        self.assertTrue(parse(html).empty)

    def test_fiscal_term_requires_explicit_calendar_mapping(self):
        html = '1. 요약재무정보 (단위: 백만원)' + table(summary('제10기', False))
        self.assertTrue(parse(html).empty)
        html += '*한국채택국제회계기준에 따라 작성 2. 연결재무제표 4. 재무제표 재무상태표 '
        html += '제 10(전)기 2017년 12월 31일 현재 (단위: 백만원)'
        html += table([['과목','제10기'],['자산총계','300'],['부채총계','100'],['자본총계','200']])
        self.assertEqual(set(parse(html).year), {2017})

    def test_two_year_income_typo_cannot_relabel_a_dated_annual_balance_term(self):
        html='1. 요약별도재무정보 (단위: 원)'+table(summary('제17기 기말'))
        html+='4. 재무제표 재무상태표 제18기 기말 2021년 12월 31일 현재 '
        html+='제17기 기말 2020년 12월 31일 현재 (단위: 원)'
        html+=table([['과목','제18기','제17기'],['자산총계','400','300'],
                     ['부채총계','110','100'],['자본총계','290','200']])
        html+='손익계산서 제17기 : 2020년 1월 1일부터 2021년 12월 31일까지 (단위: 원)'
        html+=table([['과목','제17기'],['매출액','9999'],['영업이익','999'],['당기순이익','888']])
        frame=parse(html)
        self.assertEqual(set(frame.year),{2020})
        self.assertEqual(frame.loc[frame.account_name_en.eq('revenue'),'amount'].iloc[0],1000)
        self.assertTrue(frame.fiscal_term_year_evidence.dropna().str.contains('issuer_primary_year_end_fiscal_term_v1').all())

    def test_note_year_reference_alone_cannot_establish_fiscal_term_identity(self):
        html='1. 요약별도재무정보 (단위: 원)'+table(summary('제17기 기말'))
        html+='주석 제17기 : 2020년 1월 1일부터 2021년 12월 31일까지'
        self.assertTrue(parse(html).empty)

    def test_separate_period_grid_and_attachment_note_keep_the_primary_statement_context(self):
        html='1. 요약별도재무정보 (단위: 원)'+table(summary('제13기'))
        html+='4. 재무제표 재무상태표'
        html+=table([['제14기 반기말 2023년 6월 30일 현재'],['제13기말 2022년 12월 31일 현재']])
        html+=table([['별첨 주석은 본 반기재무제표의 일부입니다.']])+'(단위: 원)'
        html+=table([['과목','제14기 반기','제13기'],['자산총계','400','300'],
                     ['부채총계','110','100'],['자본총계','290','200']])
        frame=parse(html)
        self.assertEqual(set(frame.year),{2022})
        self.assertEqual(frame.loc[frame.account_name_en.eq('revenue'),'amount'].iloc[0],1000)

    def test_parenthesized_equivalent_summary_scope_does_not_inherit_previous_consolidated_scope(self):
        html='가. 요약연결재무제표 (단위: 원)'+table(summary())
        html+='나. 요약별도(개별)재무제표 (단위: 원)'+table(summary())
        frame=parse(html)
        self.assertEqual(set(frame[frame.table_id.eq(0)].fs_div),{'CFS'})
        self.assertEqual(set(frame[frame.table_id.eq(1)].fs_div),{'OFS'})
        self.assertTrue(parse('요약연결(개별)재무정보 (단위: 원)'+table(summary())).empty)

    def test_fiscal_term_uses_own_column_full_year_range_not_another_columns_range(self):
        rows=summary('제13기')
        rows.insert(5,['손익현황','제13기 2022.01.01~2022.12.31'])
        html='1. 요약별도재무정보 (단위: 원)'+table(rows)
        self.assertEqual(set(parse(html).year),{2022})
        self.assertTrue(parse(html.replace('제13기 2022.01.01','제12기 2022.01.01')).empty)
        self.assertTrue(parse(html.replace('2022.01.01~2022.12.31','2021.01.01~2022.12.31')).empty)

    def test_interim_named_sibling_does_not_consume_adjacent_annual_term_dates(self):
        html='1. 요약별도재무정보 (단위: 원)'+table(summary('제13기'))
        html+='4. 재무제표 재무상태표 제13(당)기 2022년 12월 31일 현재 '
        html+='제13(당)분기말 2022년 9월 30일 현재 (단위: 원)'
        html+=table([['과목','제13(당)기','제13(당)분기'],['자산총계','300','250'],
                     ['부채총계','100','90'],['자본총계','200','160']])
        self.assertEqual(set(parse(html).year),{2022})

    def test_income_term_with_wrong_single_calendar_year_cannot_override_balance_year(self):
        html=self.interim_comparative().replace(
            '제13기 2022년 1월 1일부터 2022년 12월 31일까지',
            '제13기 2021년 1월 1일부터 2021년 12월 31일까지')
        frame=parse_disclosure_financials(html,'12345678','20231101000001','2023-11-01','official')
        self.assertEqual(set(frame.account_name_en),{'total_assets','total_liabilities','equity'})

    def test_non_summary_tables_do_not_inherit_summary_approval(self):
        html = '1. 요약재무정보 (단위: 원)' + table(summary())
        html += '2. 연결재무제표 4. 재무제표' + table(summary('2021년'))
        self.assertEqual(set(parse(html).year), {2022})

    def test_unknown_unit_and_accounting_basis_are_not_approved(self):
        self.assertTrue(parse('1. 요약재무정보' + table(summary())).empty)
        self.assertTrue(parse('1. 요약재무정보 (단위: 원)' + table(summary(standard=False))).empty)

    def test_preparation_statement_is_not_generic_future_ifrs_mention(self):
        html = '1. 요약재무정보 (단위: 원)' + table(summary(standard=False))
        self.assertTrue(parse(html + '한국채택국제회계기준을 적용할 예정입니다.').empty)
        frame = parse(html + '2. 중요한 회계정책 재무제표는 한국채택국제회계기준에 따라 작성되었습니다.')
        self.assertEqual(len(frame), 6)
        self.assertTrue(frame.accounting_standard_evidence.str.contains('따라 작성').all())

    def test_carve_out_history_is_not_issuer_annual_history(self):
        self.assertTrue(parse('1. 요약재무정보 (단위: 원)' + table(summary('설립 전 2022년'))).empty)

    def test_cfs_framework_does_not_approve_unlabelled_ofs_accounts(self):
        html='1. 요약별도재무정보 (단위: 원)'+table(summary(standard=False))
        self.assertTrue(parse(html+'연결재무제표는 한국채택국제회계기준에 따라 작성되었습니다.').empty)

    def test_affirmative_issuer_ifrs_application_not_future_plan(self):
        html='1. 요약재무정보 (단위: 원)'+table(summary(standard=False))
        self.assertEqual(len(parse(html+'당사는 한국채택국제회계기준을 적용하여 재무제표를 작성하고 있습니다.')),6)
        self.assertTrue(parse(html+'당사는 한국채택국제회계기준을 적용하여 재무제표를 작성할 예정입니다.').empty)

    def test_gaap_heading_cannot_use_ifrs_statement_elsewhere_in_report(self):
        html = '1. 요약재무정보(K-GAAP) (단위: 원)' + table(summary(standard=False))
        html += '다른 재무제표는 한국채택국제회계기준에 따라 작성되었습니다.'
        self.assertTrue(parse(html).empty)

    def test_absent_summary_does_not_enable_detailed_note_tables(self):
        html = '1. 요약재무정보 해당사항 없습니다. 2. 연결재무제표 4. 재무제표 (단위: 원)'
        self.assertTrue(parse(html + table(summary())).empty)

    def test_same_day_complete_prospectus_selected_not_terms_only(self):
        with tempfile.TemporaryDirectory() as root:
            receipt = '20230315000001'
            html = '재무 1. 요약재무정보 (단위: 원)' + table(summary())
            cache = Path(root) / 'dart_ipo_financial_sections'
            cache.mkdir()
            (cache / f'{receipt}.json').write_text(json.dumps({
                'receipt': receipt, 'issuer': '테스트', 'status': 'official_section_downloaded',
                'html': html, 'sha256': hashlib.sha256(html.encode()).hexdigest(),
                'source_url': 'https://dart.fss.or.kr/report/viewer.do?rcpNo='+receipt+'&dcmNo=1&eleId=2&offset=3&length=4&dtd=dart3.xsd',
            }))
            lineage = pd.DataFrame([
                dict(event_id='event', corp_code='12345678', rcept_no=receipt,
                     rcept_dt='2023-03-15', filing_report_nm='[기재정정]투자설명서'),
                dict(event_id='event', corp_code='12345678', rcept_no='20230315000002',
                     rcept_dt='2023-03-15', filing_report_nm='[발행조건확정]증권신고서(지분증권)'),
            ])
            dart=Mock();dart.session.get.return_value.text=financial_index()
            result, status = collect_disclosure_financials(dart, root, 'event', '12345678',
                                                           '테스트', '2023-03-17', lineage)
            self.assertEqual(status, 'official_annual_financial_tables_found')
            self.assertEqual(set(result.rcept_no), {receipt})
            self.assertEqual(dart.session.get.call_count,1)

    def test_legacy_receipt_prefix_date_is_replaced_by_verified_official_date(self):
        receipt='20230315000001'
        html='재무 1. 요약재무정보 (단위: 원)'+table(summary())
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'sections';path.mkdir()
            (path/(receipt+'.json')).write_text(json.dumps({'receipt':receipt,'issuer':'테스트',
                'status':'official_section_downloaded','html':html,'sha256':hashlib.sha256(html.encode()).hexdigest(),
                'source_url':'https://dart.fss.or.kr/report/viewer.do?rcpNo='+receipt+'&dcmNo=1&eleId=2&offset=3&length=4&dtd=dart3.xsd'}))
            dart=Mock();dart.session.get.return_value.text=financial_index(date='2023.03.16')
            payload=public_financial_section(dart,receipt,'테스트',path,corp_code='12345678',published='2023-03-15')
            self.assertEqual(payload['official_publication_date'],'2023-03-16')
            self.assertTrue(payload['publication_date_corrected'])
            self.assertEqual(payload['identity_schema_version'],2)
            public_financial_section(dart,receipt,'테스트',path,corp_code='12345678',published='2023-03-15')
            self.assertEqual(dart.session.get.call_count,1)

    def test_public_document_issuer_mismatch_is_rejected(self):
        dart = Mock()
        dart.session.get.return_value.text = '<title>다른회사/투자설명서</title>'
        with tempfile.TemporaryDirectory() as root:
            result = public_financial_section(dart, '20230315000001', '테스트', Path(root)/'sections')
        self.assertEqual(result['status'], 'official_document_issuer_mismatch')


if __name__ == '__main__':
    unittest.main()
