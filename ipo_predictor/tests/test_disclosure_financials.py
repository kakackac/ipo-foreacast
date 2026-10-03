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
