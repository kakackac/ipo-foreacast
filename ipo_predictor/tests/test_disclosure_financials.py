import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

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
        html += '*한국채택국제회계기준에 따라 작성 2. 연결재무제표 4. 재무제표'
        html += table([['제 10(전)기 2017년 12월 31일 현재']])
        self.assertEqual(set(parse(html).year), {2017})

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
