import unittest
from unittest.mock import Mock, patch

import pandas as pd

from data.collectors.underwriter_collector import (
    OfficialNoticeSource,
    OfficialUnderwriterCollector,
)
from data.collectors.underwriter_registry import build_underwriter_priorities
from data.pipelines.historical_ipo_pipeline import HistoricalIPOPipeline


class OfficialUnderwriterCollectorTests(unittest.TestCase):
    def test_only_integrated_retail_ratio_is_auto_approved(self):
        session = Mock()
        response = Mock()
        response.content = (
            b"<html><body>\xed\x86\xb5\xed\x95\xa9\xea\xb2\xbd\xec\x9f\x81\xeb\xa5\xa0 587.00 : 1 "
            b"\xea\xb8\xb0\xea\xb4\x80\xed\x88\xac\xec\x9e\x90\xec\x9e\x90 \xea\xb2\xbd\xec\x9f\x81\xeb\xa5\xa0 1,200 : 1</body></html>"
        )
        response.headers = {"Content-Type": "text/html; charset=utf-8"}
        response.raise_for_status.return_value = None
        session.get.return_value = response
        collector = OfficialUnderwriterCollector(session=session)

        record = collector.collect_notice(OfficialNoticeSource(
            event_id="event-1", corp_name="테스트", lead_underwriter="KB증권",
            notice_url="https://www.kbsec.com/notice/1", published_at="2026-01-01",
        ))

        self.assertEqual(record["retail_subscription_ratio"], 587.0)
        self.assertEqual(record["retail_ratio_scope"], "integrated")
        self.assertEqual(record["institutional_demand_ratio"], 1200.0)
        self.assertEqual(record["validation_status"], "official_notice_integrated_retail_ratio")
        self.assertFalse(record["human_review_required"])

    def test_underwriter_only_ratio_stays_out_of_auto_approval(self):
        session = Mock()
        response = Mock()
        response.content = b"<html><body>\xec\x9d\xbc\xeb\xb0\x98\xec\xb2\xad\xec\x95\xbd \xea\xb2\xbd\xec\x9f\x81\xeb\xa5\xa0 100 : 1</body></html>"
        response.headers = {"Content-Type": "text/html"}
        response.raise_for_status.return_value = None
        session.get.return_value = response

        record = OfficialUnderwriterCollector(session=session).collect_notice(OfficialNoticeSource(
            event_id="event-1", corp_name="테스트", lead_underwriter="KB증권",
            notice_url="https://www.kbsec.com/notice/1",
        ))

        self.assertEqual(record["retail_ratio_scope"], "underwriter_only_or_unknown")
        self.assertEqual(record["validation_status"], "official_notice_value_requires_scope_review")

    def test_proportional_allocation_ratio_stays_out_of_auto_approval(self):
        session = Mock()
        response = Mock()
        response.content = b"<html><body>\xeb\xb9\x84\xeb\xa1\x80\xeb\xb0\xb0\xec\xa0\x95 \xea\xb2\xbd\xec\x9f\x81\xeb\xa5\xa0 1,364.23:1</body></html>"
        response.headers = {"Content-Type": "text/html; charset=utf-8"}
        response.raise_for_status.return_value = None
        session.get.return_value = response

        record = OfficialUnderwriterCollector(session=session).collect_notice(OfficialNoticeSource(
            event_id="event-1", corp_name="테스트", lead_underwriter="KB증권",
            notice_url="https://www.kbsec.com/notice/1",
        ))

        self.assertEqual(record["retail_subscription_ratio"], 1364.23)
        self.assertEqual(record["retail_ratio_scope"], "underwriter_only_or_unknown")
        self.assertEqual(record["validation_status"], "official_notice_value_requires_scope_review")

    def test_label_and_ratio_must_be_directly_connected(self):
        session = Mock()
        response = Mock()
        response.content = (
            b"<html><body><p>\xed\x86\xb5\xed\x95\xa9 \xea\xb2\xbd\xec\x9f\x81\xeb\xa5\xa0\xec\x9d\x80 \xea\xb3\xb5\xec\xa7\x80\xed\x95\xa9\xeb\x8b\x88\xeb\x8b\xa4.</p>"
            b"<p>587 : 1</p></body></html>"
        )
        response.headers = {"Content-Type": "text/html; charset=utf-8"}
        response.raise_for_status.return_value = None
        session.get.return_value = response

        record = OfficialUnderwriterCollector(session=session).collect_notice(OfficialNoticeSource(
            event_id="event-1", corp_name="테스트", lead_underwriter="KB증권",
            notice_url="https://www.kbsec.com/notice/1",
        ))

        self.assertIsNone(record["retail_subscription_ratio"])
        self.assertEqual(record["validation_status"], "official_notice_no_supported_value")

    def test_pdf_text_is_accepted_only_when_the_direct_label_survives_extraction(self):
        session = Mock()
        response = Mock()
        response.content = b"%PDF-test"
        response.headers = {"Content-Type": "application/pdf"}
        response.raise_for_status.return_value = None
        session.get.return_value = response
        page = Mock()
        page.extract_text.return_value = "통합경쟁률 587.00:1"
        reader = Mock()
        reader.pages = [page]

        with patch("pypdf.PdfReader", return_value=reader):
            record = OfficialUnderwriterCollector(session=session).collect_notice(OfficialNoticeSource(
                event_id="event-1", corp_name="테스트", lead_underwriter="KB증권",
                notice_url="https://www.kbsec.com/notice/1.pdf",
            ))

        self.assertEqual(record["retail_subscription_ratio"], 587.0)
        self.assertEqual(record["validation_status"], "official_notice_integrated_retail_ratio")

    def test_non_official_host_is_rejected_without_request(self):
        session = Mock()
        record = OfficialUnderwriterCollector(session=session).collect_notice(OfficialNoticeSource(
            event_id="event-1", corp_name="테스트", lead_underwriter="KB증권",
            notice_url="https://example.com/notice/1",
        ))

        self.assertEqual(record["validation_status"], "rejected_non_official_underwriter_host")
        session.get.assert_not_called()

    def test_event_context_mismatch_prevents_auto_approval(self):
        session = Mock()
        response = Mock()
        response.content = b"<html><body>\xed\x86\xb5\xed\x95\xa9\xea\xb2\xbd\xec\x9f\x81\xeb\xa5\xa0 587 : 1</body></html>"
        response.headers = {"Content-Type": "text/html; charset=utf-8"}
        response.raise_for_status.return_value = None
        session.get.return_value = response
        source = OfficialNoticeSource(
            event_id="event-1", corp_name="테스트", lead_underwriter="KB증권",
            notice_url="https://www.kbsec.com/notice/1", published_at="2026-01-10",
            source_offering_price="10000", subscription_end="2026-01-12",
        )
        context = {
            "corp_name": "다른회사", "lead_underwriter": "KB증권", "listing_date": "2026-01-20",
            "offering_price": 10000,
        }

        record = OfficialUnderwriterCollector(session=session).collect_notice(source, context)

        self.assertEqual(record["event_context_validation_status"], "needs_review_corp_name_mismatch")
        self.assertEqual(record["validation_status"], "official_notice_event_link_review_required")
        self.assertTrue(record["human_review_required"])

    def test_complete_event_context_allows_integrated_retail_approval(self):
        session = Mock()
        response = Mock()
        response.content = b"<html><body>\xed\x86\xb5\xed\x95\xa9\xea\xb2\xbd\xec\x9f\x81\xeb\xa5\xa0 587 : 1</body></html>"
        response.headers = {"Content-Type": "text/html; charset=utf-8"}
        response.raise_for_status.return_value = None
        session.get.return_value = response
        source = OfficialNoticeSource(
            event_id="event-1", corp_name="테스트", lead_underwriter="KB증권",
            notice_url="https://www.kbsec.com/notice/1", published_at="2026-01-10",
            source_offering_price="10000", subscription_start="2026-01-08", subscription_end="2026-01-12",
        )
        context = {
            "corp_name": "테스트", "lead_underwriter": "KB증권", "listing_date": "2026-01-20",
            "offering_price": 10000,
        }

        record = OfficialUnderwriterCollector(session=session).collect_notice(source, context)

        self.assertEqual(record["event_context_validation_status"], "verified_event_context")
        self.assertEqual(record["validation_status"], "official_notice_integrated_retail_ratio")
        self.assertEqual(record["retail_subscription_ratio"], 587.0)

    def test_supported_underwriter_priorities_use_normalized_krx_names(self):
        events = pd.DataFrame({
            "event_class": ["general_ipo"] * 5,
            "lead_underwriter": [
                "한국투자증권(주)", "미래에셋증권 주식회사", "엔에이치투자증권(주)",
                "KB증권(주)", "대신증권(주)",
            ],
        })

        priorities = build_underwriter_priorities(events, top_n=5)

        self.assertEqual(priorities["lead_underwriter"].tolist(), [
            "한국투자증권", "미래에셋증권", "NH투자증권", "KB증권",
        ])
        self.assertEqual(priorities["priority"].tolist(), [1, 2, 3, 4])

    def test_pipeline_merges_only_unambiguous_integrated_result(self):
        dart = pd.DataFrame({
            "event_id": ["a", "b", "c"],
            "corp_name": ["A", "B", "C"],
            "retail_subscription_ratio": [999.0, 888.0, 777.0],
        })
        notices = pd.DataFrame([
            {
                "event_id": "a", "retail_subscription_ratio": 500.0,
                "notice_url": "https://www.kbsec.com/a",
                "validation_status": "official_notice_integrated_retail_ratio",
                "event_context_validation_status": "verified_event_context",
                "collected_at": pd.Timestamp("2026-01-01", tz="Asia/Seoul"),
            },
            {
                "event_id": "b", "retail_subscription_ratio": 100.0,
                "notice_url": "https://www.kbsec.com/b1",
                "validation_status": "official_notice_integrated_retail_ratio",
                "event_context_validation_status": "verified_event_context",
                "collected_at": pd.Timestamp("2026-01-01", tz="Asia/Seoul"),
            },
            {
                "event_id": "b", "retail_subscription_ratio": 200.0,
                "notice_url": "https://www.kbsec.com/b2",
                "validation_status": "official_notice_integrated_retail_ratio",
                "event_context_validation_status": "verified_event_context",
                "collected_at": pd.Timestamp("2026-01-02", tz="Asia/Seoul"),
            },
        ])

        merged = HistoricalIPOPipeline._merge_official_underwriter_results(dart, notices)

        self.assertEqual(merged.loc[0, "retail_subscription_ratio"], 500.0)
        self.assertTrue(pd.isna(merged.loc[1, "retail_subscription_ratio"]))
        self.assertTrue(pd.isna(merged.loc[2, "retail_subscription_ratio"]))

    def test_pipeline_never_uses_an_unapproved_retail_value(self):
        dart = pd.DataFrame({
            "event_id": ["a"], "corp_name": ["A"], "retail_subscription_ratio": [999.0],
        })
        notices = pd.DataFrame([{
            "event_id": "a", "retail_subscription_ratio": 123.0,
            "validation_status": "official_notice_value_requires_scope_review",
            "event_context_validation_status": "verified_event_context",
        }])

        merged = HistoricalIPOPipeline._merge_official_underwriter_results(dart, notices)

        self.assertTrue(pd.isna(merged.loc[0, "retail_subscription_ratio"]))

    def test_pipeline_does_not_trust_legacy_notice_without_event_validation(self):
        dart = pd.DataFrame({"event_id": ["a"], "corp_name": ["A"]})
        legacy_notice = pd.DataFrame([{
            "event_id": "a", "retail_subscription_ratio": 123.0,
            "validation_status": "official_notice_integrated_retail_ratio",
        }])

        merged = HistoricalIPOPipeline._merge_official_underwriter_results(dart, legacy_notice)

        self.assertTrue(pd.isna(merged.loc[0, "retail_subscription_ratio"]))


if __name__ == "__main__":
    unittest.main()
