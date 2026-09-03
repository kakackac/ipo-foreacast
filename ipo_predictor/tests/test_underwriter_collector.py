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
            b"<html><body>\xec\xa0\x84\xec\xb2\xb4 \xec\xb0\xb8\xec\x97\xac \xec\xa6\x9d\xea\xb6\x8c\xec\x82\xac \xed\x86\xb5\xed\x95\xa9\xea\xb2\xbd\xec\x9f\x81\xeb\xa5\xa0 587.00 : 1 "
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
        self.assertEqual(record["retail_ratio_scope"], "integrated_all_participants")
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

        self.assertIsNone(record["retail_subscription_ratio"])
        self.assertEqual(record["proportional_allocation_ratio"], 1364.23)
        self.assertIsNone(record["retail_ratio_scope"])
        self.assertEqual(record["validation_status"], "official_notice_value_requires_scope_review")

    def test_kb_style_notice_keeps_raw_general_subscription_components_unapproved(self):
        """공식 KB 공지 형식은 전체 경쟁률 대신 비례 경쟁률과 원시 수치를 싣는다."""
        session = Mock()
        response = Mock()
        response.content = """
            <html><body>
              <h1>테스트 일반청약 배정주식 및 환불 안내</h1>
              <p>일반배정 주식수: 300,000주[균등배정 150,000주(50%), 비례배정 150,000주(50%)]</p>
              <p>청약주식수: 512,013,160주</p>
              <p>비례배정: 비례배정 경쟁률 3,412.42 : 1 기준으로 안분배정</p>
            </body></html>
        """.encode()
        response.headers = {"Content-Type": "text/html; charset=utf-8"}
        response.raise_for_status.return_value = None
        session.get.return_value = response

        record = OfficialUnderwriterCollector(session=session).collect_notice(OfficialNoticeSource(
            event_id="event-1", corp_name="테스트", lead_underwriter="KB증권",
            notice_url="https://www.kbsec.com/notice/1",
        ))

        self.assertIsNone(record["retail_subscription_ratio"])
        self.assertEqual(record["proportional_allocation_ratio"], 3412.42)
        self.assertEqual(record["retail_subscribed_shares"], 512_013_160)
        self.assertEqual(record["retail_allocation_shares"], 300_000)
        self.assertEqual(record["validation_status"], "official_notice_raw_retail_components_collected")

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
        page.extract_text.return_value = "전체 참여 증권사 통합경쟁률 587.00:1"
        reader = Mock()
        reader.pages = [page]

        with patch("pypdf.PdfReader", return_value=reader):
            record = OfficialUnderwriterCollector(session=session).collect_notice(OfficialNoticeSource(
                event_id="event-1", corp_name="테스트", lead_underwriter="KB증권",
                notice_url="https://www.kbsec.com/notice/1.pdf",
            ))

        self.assertEqual(record["retail_subscription_ratio"], 587.0)
        self.assertEqual(record["validation_status"], "official_notice_integrated_retail_ratio")

    def test_integrated_word_without_all_participant_scope_stays_out_of_auto_approval(self):
        session = Mock()
        response = Mock()
        response.content = b"<html><body>\xed\x86\xb5\xed\x95\xa9\xea\xb2\xbd\xec\x9f\x81\xeb\xa5\xa0 587.00 : 1</body></html>"
        response.headers = {"Content-Type": "text/html; charset=utf-8"}
        response.raise_for_status.return_value = None
        session.get.return_value = response

        record = OfficialUnderwriterCollector(session=session).collect_notice(OfficialNoticeSource(
            event_id="event-1", corp_name="테스트", lead_underwriter="KB증권",
            notice_url="https://www.kbsec.com/notice/1",
        ))

        self.assertEqual(record["retail_subscription_ratio"], 587.0)
        self.assertEqual(record["retail_ratio_scope"], "underwriter_only_or_unknown")
        self.assertEqual(record["validation_status"], "official_notice_value_requires_scope_review")

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
        response.content = b"<html><body>\xec\xa0\x84\xec\xb2\xb4 \xec\xa6\x9d\xea\xb6\x8c\xec\x82\xac \xed\x86\xb5\xed\x95\xa9\xea\xb2\xbd\xec\x9f\x81\xeb\xa5\xa0 587 : 1</body></html>"
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
        response.content = b"<html><body>\xec\xa0\x84\xec\xb2\xb4 \xec\xb0\xb8\xec\x97\xac \xec\xa6\x9d\xea\xb6\x8c\xec\x82\xac \xed\x86\xb5\xed\x95\xa9\xea\xb2\xbd\xec\x9f\x81\xeb\xa5\xa0 587 : 1</body></html>"
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

    def test_explicit_total_general_subscription_scope_is_approved_without_integrated_word(self):
        session = Mock()
        response = Mock()
        response.content = "<html><body>전체 일반청약 경쟁률 587 : 1</body></html>".encode()
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

        self.assertEqual(record["retail_ratio_scope"], "integrated_all_participants")
        self.assertEqual(record["validation_status"], "official_notice_integrated_retail_ratio")

    def test_sole_retail_intake_broker_is_approved_only_with_explicit_document_scope(self):
        session = Mock()
        response = Mock()
        response.content = "<html><body>단독 주관 일반청약 경쟁률 587 : 1</body></html>".encode()
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

        self.assertEqual(record["retail_ratio_scope"], "sole_retail_intake_broker")
        self.assertEqual(record["validation_status"], "official_notice_single_retail_intake_ratio")

    def test_all_participant_raw_shares_are_reconstructed_without_averaging_ratios(self):
        session = Mock()
        responses = []
        for body in (
            "일반청약 청약주식수 900,000주 일반청약 배정주식수 1,000주",
            "일반청약 청약주식수 1,100,000주 일반청약 배정주식수 1,000주",
        ):
            response = Mock()
            response.content = f"<html><body>{body}</body></html>".encode()
            response.headers = {"Content-Type": "text/html; charset=utf-8"}
            response.raise_for_status.return_value = None
            responses.append(response)
        session.get.side_effect = responses
        collector = OfficialUnderwriterCollector(session=session)
        sources = pd.DataFrame([
            {
                "event_id": "event-1", "corp_name": "테스트", "lead_underwriter": "한국투자증권",
                "notice_underwriter": "한국투자증권", "notice_url": "https://securities.koreainvestment.com/1",
                "published_at": "2026-01-10", "source_offering_price": "10000",
                "subscription_start": "2026-01-08", "subscription_end": "2026-01-12",
                "retail_participating_brokers": "한국투자증권,KB증권",
                "scope_verification_status": "manual_verified_official_source",
            },
            {
                "event_id": "event-1", "corp_name": "테스트", "lead_underwriter": "한국투자증권",
                "notice_underwriter": "KB증권", "notice_url": "https://www.kbsec.com/1",
                "published_at": "2026-01-10", "source_offering_price": "10000",
                "subscription_start": "2026-01-08", "subscription_end": "2026-01-12",
                "retail_participating_brokers": "한국투자증권,KB증권",
                "scope_verification_status": "manual_verified_official_source",
            },
        ])
        contexts = {"event-1": {
            "corp_name": "테스트", "lead_underwriter": "한국투자증권", "listing_date": "2026-01-20",
            "offering_price": 10000,
        }}

        records = collector.collect_sources(sources, event_contexts=contexts)
        resolved = collector.resolve_reconstructed_retail_ratios(records)
        reconstructed = resolved[
            resolved["validation_status"].eq("official_notice_reconstructed_retail_ratio")
        ].iloc[0]

        self.assertEqual(reconstructed["retail_subscription_ratio"], 1000.0)
        self.assertEqual(reconstructed["retail_subscribed_shares"], 2_000_000)
        self.assertEqual(reconstructed["retail_allocation_shares"], 2_000)

    def test_reconstruction_rejects_missing_participant_components(self):
        record = pd.DataFrame([{
            "event_id": "event-1", "notice_id": "one", "notice_underwriter": "한국투자증권",
            "retail_participating_brokers": "한국투자증권,KB증권",
            "scope_verification_status": "manual_verified_official_source",
            "retail_subscribed_shares": 900_000, "retail_allocation_shares": 1_000,
            "validation_status": "official_notice_raw_retail_components_collected",
            "event_context_validation_status": "verified_event_context",
        }])

        resolved = OfficialUnderwriterCollector(session=Mock()).resolve_reconstructed_retail_ratios(record)

        self.assertFalse(
            resolved["validation_status"].eq("official_notice_reconstructed_retail_ratio").any()
        )

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
        self.assertTrue(priorities["collection_policy"].str.startswith("manual_url_only").all())

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

    def test_pipeline_accepts_verified_reconstructed_all_participant_ratio(self):
        dart = pd.DataFrame({"event_id": ["a"], "corp_name": ["A"]})
        reconstructed = pd.DataFrame([{
            "event_id": "a", "retail_subscription_ratio": 1000.0,
            "notice_url": "https://official.example/a | https://official.example/b",
            "validation_status": "official_notice_reconstructed_retail_ratio",
            "event_context_validation_status": "verified_event_context",
            "available_at": "2026-01-15",
            "collected_at": pd.Timestamp("2026-01-16", tz="Asia/Seoul"),
        }])

        merged = HistoricalIPOPipeline._merge_official_underwriter_results(dart, reconstructed)

        self.assertEqual(merged.loc[0, "retail_subscription_ratio"], 1000.0)
        self.assertEqual(merged.loc[0, "retail_subscription_eligibility_status"], "verified_official_notice")

    def test_pipeline_uses_pre_listing_verified_dart_result_before_notice(self):
        dart = pd.DataFrame({
            "event_id": ["a"], "corp_name": ["A"], "listing_date": ["2026-01-20"],
            "retail_available_at": ["2026-01-15"],
            "retail_subscription_ratio": [123.0],
            "retail_validation_status": ["official_dart_issuer_total_retail_ratio"],
        })

        merged = HistoricalIPOPipeline._merge_official_underwriter_results(dart, pd.DataFrame())

        self.assertEqual(merged.loc[0, "retail_subscription_ratio"], 123.0)
        self.assertEqual(merged.loc[0, "retail_subscription_eligibility_status"], "verified_official_dart")
        self.assertTrue(merged.loc[0, "retail_subscription_eligible"])

    def test_pipeline_quarantines_conflicting_dart_and_notice_results(self):
        dart = pd.DataFrame({
            "event_id": ["a"], "corp_name": ["A"], "listing_date": ["2026-01-20"],
            "retail_available_at": ["2026-01-15"],
            "retail_subscription_ratio": [123.0],
            "retail_validation_status": ["official_dart_issuer_total_retail_ratio"],
        })
        notices = pd.DataFrame([{
            "event_id": "a", "retail_subscription_ratio": 456.0,
            "notice_url": "https://www.kbsec.com/a",
            "validation_status": "official_notice_integrated_retail_ratio",
            "event_context_validation_status": "verified_event_context",
            "available_at": "2026-01-15",
            "collected_at": pd.Timestamp("2026-01-16", tz="Asia/Seoul"),
        }])

        merged = HistoricalIPOPipeline._merge_official_underwriter_results(dart, notices)

        self.assertTrue(pd.isna(merged.loc[0, "retail_subscription_ratio"]))
        self.assertEqual(
            merged.loc[0, "retail_subscription_eligibility_status"],
            "official_source_conflict_review_required",
        )

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
