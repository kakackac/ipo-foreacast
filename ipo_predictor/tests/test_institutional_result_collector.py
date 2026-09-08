import unittest
from unittest.mock import Mock

import pandas as pd

from data.collectors.institutional_result_collector import (
    OfficialInstitutionalResultCollector,
    OfficialInstitutionalResultSource,
)
from data.pipelines.historical_ipo_pipeline import HistoricalIPOPipeline


class OfficialInstitutionalResultCollectorTests(unittest.TestCase):
    @staticmethod
    def _event_context():
        return {
            "corp_name": "테스트", "lead_underwriter": "KB증권",
            "listing_date": "2026-01-20", "offering_price": 10000,
        }

    @staticmethod
    def _source(**overrides):
        values = {
            "event_id": "event-1", "corp_name": "테스트", "lead_underwriter": "KB증권",
            "notice_url": "https://www.kbsec.com/notice/1", "published_at": "2026-01-10",
            "source_offering_price": "10000", "notice_underwriter": "KB증권",
            "aggregate_scope_verification": "manual_verified_aggregate_institutional",
        }
        values.update(overrides)
        return OfficialInstitutionalResultSource(**values)

    def _collector_with_html(self, html: str):
        session = Mock()
        response = Mock()
        response.content = html.encode()
        response.headers = {"Content-Type": "text/html; charset=utf-8"}
        response.raise_for_status.return_value = None
        session.get.return_value = response
        return OfficialInstitutionalResultCollector(session=session)

    def test_complete_same_notice_bundle_is_approved(self):
        collector = self._collector_with_html("""
            <html><body><p>기관투자자 수요예측 경쟁률 850.00 : 1</p>
            <p>의무보유확약 비율</p><p>6개월 10.0%</p><p>3개월 20.0%</p>
            <p>1개월 30.0%</p><p>15일 40.0%</p><p>확약 없음 0.0%</p></body></html>
        """)

        record = collector.collect_notice(self._source(), self._event_context())

        self.assertEqual(record["validation_status"], "verified_official_underwriter_aggregate_bundle")
        self.assertEqual(record["institutional_demand_ratio"], 850.0)
        self.assertEqual(record["lockup_commitment_ratio"], 1.0)
        self.assertEqual(record["lockup_6m_ratio"], 0.1)
        self.assertEqual(record["lockup_15d_ratio"], 0.4)
        self.assertFalse(record["human_review_required"])

    def test_incomplete_notice_never_approves_a_partial_bundle(self):
        collector = self._collector_with_html(
            "<html><body>기관투자자 수요예측 경쟁률 850.00 : 1 의무보유확약 비율 6개월 10.0%</body></html>"
        )

        record = collector.collect_notice(self._source(), self._event_context())

        self.assertEqual(record["validation_status"], "official_notice_incomplete_institutional_bundle")
        self.assertTrue(record["human_review_required"])

    def test_direct_total_lockup_ratio_is_approved_without_period_details(self):
        collector = self._collector_with_html(
            "<html><body>기관투자자 수요예측 경쟁률 63.41 : 1 의무보유확약 0.17%</body></html>"
        )

        record = collector.collect_notice(self._source(), self._event_context())

        self.assertEqual(record["validation_status"], "verified_official_underwriter_aggregate_bundle")
        self.assertAlmostEqual(record["lockup_commitment_ratio"], 0.0017)

    def test_pipeline_does_not_mix_dart_and_underwriter_partial_values(self):
        dart = pd.DataFrame([{
            "event_id": "event-1", "institutional_validation_status": "dart_final_terms_value_not_found",
            "lockup_validation_status": "dart_final_terms_value_not_found",
        }])
        underwriter = pd.DataFrame([{
            "event_id": "event-1", "validation_status": "verified_official_underwriter_aggregate_bundle",
            "event_context_validation_status": "verified_event_context", "institutional_demand_ratio": 850.0,
            "lockup_commitment_ratio": None,
            "lockup_6m_ratio": 0.1, "lockup_3m_ratio": None,
            "lockup_1m_ratio": 0.3, "lockup_15d_ratio": 0.4,
        }])

        merged = HistoricalIPOPipeline._merge_official_underwriter_institutional_results(dart, underwriter)

        self.assertNotIn("institutional_demand_ratio", merged.columns)

    def test_pipeline_uses_one_complete_underwriter_bundle_only_when_dart_missing(self):
        dart = pd.DataFrame([{
            "event_id": "event-1", "institutional_validation_status": "dart_final_terms_value_not_found",
            "lockup_validation_status": "dart_final_terms_value_not_found",
        }])
        underwriter = pd.DataFrame([{
            "event_id": "event-1", "validation_status": "verified_official_underwriter_aggregate_bundle",
            "event_context_validation_status": "verified_event_context", "institutional_demand_ratio": 850.0,
            "lockup_commitment_ratio": 1.0,
            "lockup_6m_ratio": 0.1, "lockup_3m_ratio": 0.2,
            "lockup_1m_ratio": 0.3, "lockup_15d_ratio": 0.4,
            "notice_url": "https://www.kbsec.com/notice/1", "available_at": "2026-01-10",
            "institutional_evidence": "기관투자자 수요예측 경쟁률 850.00 : 1",
            "lockup_evidence": "의무보유확약 비율 6개월 10.0%",
            "published_at": "2026-01-10", "collected_at": "2026-01-10", "is_correction": False,
            "revision_of_notice_id": None,
        }])

        merged = HistoricalIPOPipeline._merge_official_underwriter_institutional_results(dart, underwriter)

        self.assertEqual(merged.loc[0, "institutional_demand_ratio"], 850.0)
        self.assertEqual(merged.loc[0, "lockup_commitment_ratio"], 1.0)
        self.assertEqual(
            merged.loc[0, "institutional_validation_status"],
            "verified_official_underwriter_institutional",
        )

    def test_pipeline_uses_separately_verified_official_documents_for_each_field(self):
        dart = pd.DataFrame([{
            "event_id": "event-1", "institutional_demand_ratio": None,
            "lockup_commitment_ratio": None,
            "institutional_validation_status": "dart_final_terms_value_not_found",
            "lockup_validation_status": "dart_final_terms_value_not_found",
        }])
        common = {
            "event_id": "event-1", "event_context_validation_status": "verified_event_context",
            "aggregate_scope_verification": "manual_verified_aggregate_institutional",
            "validation_status": "official_notice_incomplete_institutional_bundle",
            "collected_at": "2026-01-10",
        }
        underwriter = pd.DataFrame([
            {**common, "institutional_demand_ratio": 850.0, "lockup_commitment_ratio": None,
             "notice_url": "https://www.kbsec.com/notice/demand", "available_at": "2026-01-10",
             "published_at": "2026-01-10", "institutional_evidence": "기관투자자 수요예측 경쟁률 850:1"},
            {**common, "institutional_demand_ratio": None, "lockup_commitment_ratio": 0.25,
             "notice_url": "https://www.kbsec.com/notice/lockup", "available_at": "2026-01-11",
             "published_at": "2026-01-11", "lockup_evidence": "기관투자자 의무보유확약 25%"},
        ])

        merged = HistoricalIPOPipeline._merge_official_underwriter_institutional_results(dart, underwriter)

        self.assertEqual(merged.loc[0, "institutional_demand_ratio"], 850.0)
        self.assertEqual(merged.loc[0, "lockup_commitment_ratio"], 0.25)
        self.assertEqual(merged.loc[0, "institutional_source_url"], "https://www.kbsec.com/notice/demand")
        self.assertEqual(merged.loc[0, "lockup_source_url"], "https://www.kbsec.com/notice/lockup")

    def test_dart_selector_never_combines_partial_values_from_two_receipts(self):
        first_final = pd.Series({"rcept_no": "20260101000001", "rcept_dt": "2026-01-01", "is_final_conditions": True})
        second_final = pd.Series({"rcept_no": "20260102000001", "rcept_dt": "2026-01-02", "is_final_conditions": True})
        first_document = {
            "dart_final_terms_demand_parser_version": 4,
            "institutional_demand_ratio": 850.0,
            "lockup_commitment_ratio": None,
            "lockup_6m_ratio": None, "lockup_3m_ratio": None,
            "lockup_1m_ratio": None, "lockup_15d_ratio": None,
        }
        second_document = {
            "dart_final_terms_demand_parser_version": 4,
            "institutional_demand_ratio": None,
            "lockup_commitment_ratio": 1.0,
            "lockup_6m_ratio": 0.1, "lockup_3m_ratio": 0.2,
            "lockup_1m_ratio": 0.3, "lockup_15d_ratio": 0.4,
        }

        values, metadata = HistoricalIPOPipeline._select_dart_final_terms_demand([
            (first_final, first_document),
            (second_final, second_document),
        ])

        self.assertEqual(values, {})
        self.assertIsNone(metadata["institutional_rcept_no"])
        self.assertIsNone(metadata["lockup_rcept_no"])


if __name__ == "__main__":
    unittest.main()
