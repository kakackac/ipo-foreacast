"""Regression cases transcribed from official DART sample documents.

Full local originals and hashes are retained by scripts/audit_official_samples.py.
These focused excerpts exercise the observed failures, not production approval.
"""
import unittest

from data.collectors.dart_collector import DARTCollector
from scripts.audit_official_samples import verify_record


class OfficialDocumentSampleTests(unittest.TestCase):
    def setUp(self):
        self.collector = DARTCollector(api_key="test")

    def test_reference_audit_rejects_changed_source_and_missing_outputs(self):
        reference = {"receipt": "1", "source_url": "official", "sha256": "abc",
                     "expected": {"new_shares": 100, "secondary_shares": None}}
        record = {**reference, "offering": {"new_shares": 100, "secondary_shares": None}}
        self.assertEqual(verify_record(record, reference), [])
        record["sha256"] = "changed"
        self.assertIn("source_mismatch:sha256", verify_record(record, reference))
        record["offering"].pop("secondary_shares")
        self.assertIn("missing_output:secondary_shares", verify_record(record, reference))
        record["offering"]["new_shares"] = 101
        self.assertIn("value_mismatch:new_shares", verify_record(record, reference))

    def test_wiseitech_quantity_percentage_not_participant_percentage(self):
        # DART 20200128000055: the heading is outside the table.
        html = """<p>③ 의무보유 확약 기관수 및 신청수량</p><table>
        <tr><th>구분</th><th>참여건수(건)</th><th>참여수량(주)</th></tr>
        <tr><td>2개월 확약</td><td>33</td><td>22,440,000</td></tr>
        <tr><td>1개월 확약</td><td>27</td><td>17,338,000</td></tr>
        <tr><td>2주 확약</td><td>20</td><td>12,396,000</td></tr>
        <tr><td>합 계</td><td>80</td><td>52,174,000</td></tr>
        <tr><td>총 수량 대비 비율</td><td>6.85%</td><td>6.95%</td></tr>
        </table>"""
        parsed = self.collector._parse_demand_forecast_html(html, "위세아이텍")
        self.assertAlmostEqual(parsed["lockup_commitment_ratio"], 0.0695)
        without_scope = html.replace("③ 의무보유 확약 기관수 및 신청수량", "기존주주 보호예수")
        self.assertIsNone(self.collector._parse_demand_forecast_html(
            without_scope, "위세아이텍"
        )["lockup_commitment_ratio"])

    def test_sensorview_band_units_and_explicit_all_new_shares(self):
        # DART 20230706000211.
        html = """<table><tr><td>주당 희망공모가액</td>
        <td>2,900원 ~ 3,600원</td></tr></table>
        신주모집 3,900,000주(공모주식의 100.0%)"""
        parsed = self.collector._parse_offering_html(html, "20230706000211")
        self.assertEqual((parsed["price_band_low"], parsed["price_band_high"]), (2900, 3600))
        self.assertEqual(parsed["new_shares"], 3_900_000)
        self.assertEqual(parsed["secondary_shares"], 0)

    def test_sensorview_total_quantity_reconciles(self):
        # Total columns from DART 20230706000211, with count kept adjacent.
        rows = [
            ["기관투자자", "합 계", "합 계"], ["구분", "건수", "수량"],
            ["6개월 확약", "22", "64,350,000"],
            ["3개월 확약", "47", "137,475,000"],
            ["1개월 확약", "46", "129,946,000"],
            ["15일 확약", "11", "31,082,000"],
            ["미확약", "1,594", "4,531,623,157"],
            ["합 계", "1,720", "4,894,476,157"],
        ]
        parsed = self.collector._extract_aggregate_lockup_shares(rows)
        self.assertAlmostEqual(parsed["lockup_commitment_ratio"], 362_853_000 / 4_894_476_157)
        rows[-1][-1] = "4,894,476,158"
        self.assertIsNone(self.collector._extract_aggregate_lockup_shares(rows))

    def test_rznomics_correction_conflict_is_not_first_value(self):
        # DART 20251205000229 contains both before/after tables.
        def table(ratio):
            return f"""<table><tr><th>구분</th><th>유통가능 주식수 비율</th></tr>
            <tr><td>상장일 유통가능</td><td>{ratio}%</td></tr></table>"""
        parsed = self.collector._parse_offering_html(table(35.23) + table(35.27), "20251205000229")
        self.assertIsNone(parsed["public_float_ratio_disclosed"])
        self.assertEqual(parsed["public_float_parse_method"], "conflicting_float_ratios_review_required")
        final = self.collector._parse_offering_html(table(35.27), "final-body")
        self.assertAlmostEqual(final["public_float_ratio_disclosed"], 0.3527)

    def test_ns_shopping_post_offering_float_column_and_all_secondary(self):
        # DART 20150210000401: source table has pre/post offering columns.
        html = """구주매출 878,181 주. 공모는 100% 구주매출로 진행됩니다.
        <table><tr><th rowspan="2">구분</th><th colspan="2">공모전 기준</th>
        <th colspan="2">공모후 기준</th></tr>
        <tr><th>주식수</th><th>지분율 (%)</th><th>주식수</th><th>지분율 (%)</th></tr>
        <tr><td>유통가능물량 소계</td><td>1,415,380</td><td>42.00%</td>
        <td>1,552,330</td><td>46.07%</td></tr></table>"""
        parsed = self.collector._parse_offering_html(html, "20150210000401")
        self.assertEqual(parsed["new_shares"], 0)
        self.assertEqual(parsed["secondary_shares"], 878181)
        self.assertAlmostEqual(parsed["public_float_ratio_disclosed"], 0.4607)

    def test_wiseitech_float_header_selects_unrestricted_total(self):
        html = """<table><tr><th rowspan="2">구분</th><th colspan="2">유통가능물량</th>
        <th colspan="2">매도금지물량</th></tr><tr><th>주식수(주)</th><th>지분율</th>
        <th>주식수(주)</th><th>지분율</th></tr><tr><td>합계</td>
        <td>2,501,275</td><td>58.27%</td><td>1,791,225</td><td>41.73%</td></tr></table>"""
        parsed = self.collector._parse_offering_html(html, "20200128000055")
        self.assertAlmostEqual(parsed["public_float_ratio_disclosed"], 0.5827)

    def test_merged_total_units_require_two_matching_group_schemas(self):
        # DART 20251205000229 merged all total units into '합계'.
        table = [
            ["기관투자자", "건수", "수량", "신청가격", "건수", "수량", "신청가격", "합계", "합계", "합계"],
            ["6개월 확약", "1", "10", "20", "1", "10", "20", "2", "20", "20"],
            ["미확약", "1", "10", "20", "1", "10", "20", "2", "20", "20"],
            ["합계", "2", "20", "20", "2", "20", "20", "4", "40", "20"],
        ]
        parsed = self.collector._extract_aggregate_lockup_shares(table, application_total=40)
        self.assertEqual(parsed["lockup_commitment_ratio"], 0.5)
        self.assertIsNone(self.collector._extract_aggregate_lockup_shares(table))
        self.assertIsNone(self.collector._extract_aggregate_lockup_shares(table, application_total=41))
        table[0][2] = "배정수량"
        self.assertIsNone(self.collector._extract_aggregate_lockup_shares(table, application_total=40))

    def test_ns_shopping_summary_heading_and_quantity_unit(self):
        # DART 20150312000834, not a broker-local retail result.
        html = """<p>(12) 수요예측 결과① 기관투자자 수요예측 참여내역</p>
        <table><tr><th>참여건수(건)</th><th>신청수량(주)</th><th>단순 경쟁률</th></tr>
        <tr><td>644</td><td>212,580,000</td><td>302.59</td></tr></table>
        <p>③ 의무보유확약 신청내역</p><table>
        <tr><th>구분</th><th>신청수량(단위: 주)</th><th>비고</th></tr>
        <tr><td>합계</td><td>33,295,000</td><td>-</td></tr>
        <tr><td>총 수량 대비 비율(%)</td><td>15.66%</td><td>-</td></tr></table>"""
        parsed = self.collector._parse_demand_forecast_html(html, "엔에스쇼핑")
        self.assertEqual(parsed["institutional_demand_ratio"], 302.59)
        self.assertAlmostEqual(parsed["lockup_commitment_ratio"], .1566)
        retail = html.replace("수요예측 결과① 기관투자자 수요예측 참여내역", "일반청약 결과")
        self.assertIsNone(self.collector._parse_demand_forecast_html(retail, "test")["institutional_demand_ratio"])
        allocated = html.replace("신청수량(단위: 주)", "배정수량(단위: 주)")
        self.assertIsNone(self.collector._parse_demand_forecast_html(allocated, "test")["lockup_commitment_ratio"])

    def test_inferred_total_requires_independent_quantity_reference(self):
        table = [
            ["기관투자자", "건수", "수량", "신청가격", "건수", "수량", "신청가격", "합계", "합계", "합계"],
            ["6개월 확약", "1", "10", "20", "1", "11", "20", "2", "20", "20"],
            ["미확약", "1", "10", "20", "1", "10", "20", "2", "20", "20"],
            ["합계", "2", "20", "20", "2", "21", "20", "4", "40", "20"],
        ]
        self.assertIsNone(self.collector._extract_aggregate_lockup_shares(table))


if __name__ == "__main__":
    unittest.main()
