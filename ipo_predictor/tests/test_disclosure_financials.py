import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import pandas as pd

from data.pipelines.disclosure_financials import (
    collect_disclosure_financials, parse_disclosure_financials, public_financial_section,
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


class DisclosureFinancialTests(unittest.TestCase):
    def test_annual_summary_without_date_range_and_negative_income(self):
        frame = parse('1. 요약재무정보 (단위: 백만원)' + table(summary()))
        result = summarize_asof(frame, '12345678', '2023-04-01')
        self.assertEqual(result['revenue'], 1_000_000_000)
        self.assertEqual(result['operating_margin'], -.05)
        self.assertEqual(result['debt_ratio'], .5)

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
        index = "<title>변경회사/투자설명서/2023.03.16</title>"
        index += "openCorpInfoNew('12345678', 'info');"
        index += "var node1 = {};"
        for key, value in dict(text='III. 재무에 관한 사항', rcpNo='20230315000001',
                               dcmNo='1', eleId='2', offset='3', length='4', dtd='dart3.xsd').items():
            index += f"node1['{key}'] = \"{value}\";"
        first, second = Mock(), Mock()
        first.text = index
        second.text = '재무' + table(summary())
        dart.session.get.side_effect = [first, second]
        with tempfile.TemporaryDirectory() as root:
            payload = public_financial_section(dart, '20230315000001', '이전회사', root,
                                               corp_code='12345678', published='2023-03-16')
        self.assertEqual(payload['status'], 'official_section_downloaded')
        self.assertEqual(payload['official_publication_date'], '2023-03-16')

    def test_matching_name_does_not_override_wrong_corp_code(self):
        dart = Mock()
        dart.session.get.return_value.text = "<title>테스트/투자설명서/2023.03.15</title>openCorpInfoNew('87654321', 'info');"
        with tempfile.TemporaryDirectory() as root:
            result = public_financial_section(dart, '20230315000001', '테스트', root, corp_code='12345678')
        self.assertEqual(result['status'], 'official_document_corp_code_mismatch')

    def test_adjacent_ifrs_footnote(self):
        frame = parse('1. 요약 연결재무정보 (단위: 원)' + table(summary(standard=False))
                      + table([['주1 한국채택국제회계기준(K-IFRS)에 따라 작성']]))
        self.assertEqual(set(frame.fs_div), {'CFS'})
        self.assertEqual(len(frame), 6)

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
                'source_url': 'https://dart.fss.or.kr/report/viewer.do',
            }))
            lineage = pd.DataFrame([
                dict(event_id='event', corp_code='12345678', rcept_no=receipt,
                     rcept_dt='2023-03-15', filing_report_nm='[기재정정]투자설명서'),
                dict(event_id='event', corp_code='12345678', rcept_no='20230315000002',
                     rcept_dt='2023-03-15', filing_report_nm='[발행조건확정]증권신고서(지분증권)'),
            ])
            result, status = collect_disclosure_financials(Mock(), root, 'event', '12345678',
                                                           '테스트', '2023-03-17', lineage)
            self.assertEqual(status, 'official_annual_financial_tables_found')
            self.assertEqual(set(result.rcept_no), {receipt})

    def test_public_document_issuer_mismatch_is_rejected(self):
        dart = Mock()
        dart.session.get.return_value.text = '<title>다른회사/투자설명서</title>'
        with tempfile.TemporaryDirectory() as root:
            result = public_financial_section(dart, '20230315000001', '테스트', root)
        self.assertEqual(result['status'], 'official_document_issuer_mismatch')


if __name__ == '__main__':
    unittest.main()
