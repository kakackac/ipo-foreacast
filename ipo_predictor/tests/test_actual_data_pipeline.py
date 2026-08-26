import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock

import pandas as pd

from data.collectors.dart_collector import DARTCollector
from data.pipelines.historical_ipo_pipeline import HistoricalIPOPipeline


class _FakeDART:
    is_configured = True

    def get_ipo_disclosure_list(self, start_date, end_date):
        if not start_date.startswith("2024"):
            return pd.DataFrame()
        return pd.DataFrame([{
            "corp_code": "12345678", "corp_name": "테스트(주)",
            "rcept_no": "20240101000001", "rcept_dt": pd.Timestamp("2024-01-01"),
            "report_nm": "[발행조건확정]증권신고서(지분증권)",
        }])

    def get_offering_info(self, rcept_no):
        return {
            "rcept_no": rcept_no, "price_band_low": 9000, "price_band_high": 11000,
            "offering_price": 12000, "new_shares": 1000000, "secondary_shares": 200000,
            "total_post_listing_shares": 5000000, "lead_underwriter": "한국투자증권",
            "major_shareholder_lockup_months": 24, "risk_factor_count": 5,
            "offering_price_finality": "confirmed_price_language",
            "parse_success": True,
        }

    def find_demand_forecast_disclosure(self, corp_code, start_date, end_date):
        return "20240201000001"

    def get_demand_forecast(self, corp_code, rcept_no):
        return {
            "corp_code": corp_code, "institutional_demand_ratio": 850.0,
            "lockup_6m_ratio": 0.1, "lockup_3m_ratio": 0.2,
            "lockup_1m_ratio": 0.1, "lockup_15d_ratio": 0.1,
            "parse_success": True,
        }

    def get_equity_offering_prices(self, corp_code, start_date, end_date):
        return [{
            "rcept_no": "20240101000001",
            "offering_price": 12000,
            "security_type": "보통주",
        }]

    def get_financial_statements(self, corp_code, year):
        amounts = {
            "revenue": 100_000_000 + (year - 2021) * 10_000_000,
            "operating_income": 10_000_000,
            "net_income": 8_000_000,
            "total_assets": 200_000_000,
            "total_liabilities": 70_000_000,
            "equity": 130_000_000,
            "eps": 800.0,
        }
        return pd.DataFrame({
            "year": [year] * len(amounts),
            "account_name_en": list(amounts),
            "amount": list(amounts.values()),
        })


class _FakeKRX:
    official_listing_requests = []

    def get_official_listing_events(self, start_date, end_date):
        if not start_date.startswith("2024"):
            return pd.DataFrame()
        return pd.DataFrame([{
            "event_id": "krx_kind|123456|20240510|테스트",
            "ticker": "123456", "krx_standard_code": None, "corp_name": "테스트㈜",
            "listing_date": pd.Timestamp("2024-05-10"), "market": "KOSDAQ",
            "security_type": "주권", "stock_type": None, "listing_type": "신규상장",
            "offering_price": 12000, "offering_shares": 1_000_000,
            "lead_underwriter": "테스트증권", "industry_name": "소프트웨어",
            "industry_code": None, "country": "대한민국", "face_value": 500,
            "offering_amount": 12_000_000, "event_class": "general_ipo",
            "classification_reason": "test", "classification_confidence": "high",
            "classification_review_required": False, "source_name": "KRX_KIND_new_listing_company",
            "source_url": "https://kind.krx.co.kr", "source_request_id": "test",
            "collected_at": pd.Timestamp("2024-05-01", tz="Asia/Seoul"),
            "verification_status": "official_source", "listing_segment": None,
        }])

    def get_ipo_calendar(self, start_date, end_date):
        if not start_date.startswith("2024"):
            return pd.DataFrame()
        return pd.DataFrame([{
            "ticker": "123456", "isu_cd": "KR7123456000", "corp_name": "테스트㈜",
            "listing_date": pd.Timestamp("2024-05-10"), "market": "KOSDAQ",
            "sector": "소프트웨어", "same_day_ipo_count": 1,
        }])

    def get_listing_day_price(self, ticker, listing_date, isu_cd=None, market=None, corp_name=None):
        return {
            "ticker": ticker, "isu_cd": isu_cd, "listing_date": listing_date,
            "open_price": 18000, "close_price": 15000, "high_price": 19000,
            "low_price": 14000, "volume": 100000,
        }

    def get_index_ohlcv(self, index_code, start_date, end_date):
        dates = pd.bdate_range("2024-01-01", "2024-12-31")
        return pd.DataFrame({
            "date": dates, "index_code": "KOSPI" if index_code == "1" else "KOSDAQ",
            "close": range(2000, 2000 + len(dates)),
        })


class ActualDataPipelineTests(unittest.TestCase):
    def test_dart_no_data_status_is_normalised_to_an_empty_list(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"status": "013", "message": "조회된 데이타가 없습니다."}
        collector = DARTCollector(api_key="a" * 40)
        collector.session.get = Mock(return_value=response)

        result = collector._get("fnlttSinglAcntAll", {})

        self.assertEqual(result, {"status": "013", "list": []})

    def test_dart_ipo_list_uses_equity_offering_filter_in_three_month_chunks(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "status": "000",
            "total_page": 1,
            "list": [{
                "corp_code": "12345678", "corp_name": "테스트", "rcept_no": "20240101000001",
                "rcept_dt": "20240101", "report_nm": "증권신고서(지분증권)",
            }],
        }
        collector = DARTCollector(api_key="a" * 40)
        collector.session.get = Mock(return_value=response)

        disclosures = collector.get_ipo_disclosure_list("20240101", "20240430")

        self.assertEqual(len(disclosures), 1)
        self.assertEqual(collector.session.get.call_count, 2)
        first_params = collector.session.get.call_args_list[0].kwargs["params"]
        second_params = collector.session.get.call_args_list[1].kwargs["params"]
        self.assertEqual(first_params["pblntf_ty"], "C")
        self.assertEqual(first_params["pblntf_detail_ty"], "C001")
        self.assertEqual((first_params["bgn_de"], first_params["end_de"]), ("20240101", "20240330"))
        self.assertEqual((second_params["bgn_de"], second_params["end_de"]), ("20240331", "20240430"))

    def test_document_zip_is_converted_to_plain_text(self):
        content = io.BytesIO()
        with zipfile.ZipFile(content, "w") as archive:
            archive.writestr("document.xml", "<html><body>희망 공모가 10,000 ~ 12,000 원</body></html>")

        response = Mock()
        response.content = content.getvalue()
        response.raise_for_status.return_value = None
        collector = DARTCollector(api_key="a" * 40)
        collector.session.get = Mock(return_value=response)

        self.assertIn("희망 공모가 10,000 ~ 12,000 원", collector.get_document_text("20240101000001"))

    def test_pipeline_writes_raw_data_features_and_quality_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            summary = HistoricalIPOPipeline(
                dart_collector=_FakeDART(),
                krx_collector=_FakeKRX(),
                raw_dir=root / "raw",
                processed_dir=root / "processed",
            ).run(2024, 2024, feature_set="phase2")

            features = pd.read_parquet(root / "processed" / "features_all.parquet")
            self.assertEqual(summary["feature_rows"], 1)
            self.assertEqual(summary["open_target_rows"], 1)
            self.assertEqual(summary["close_target_rows"], 1)
            self.assertEqual(summary["offering_price_within_expected_range_rows"], 1)
            self.assertEqual(summary["offering_price_range_warning_rows"], 0)
            self.assertEqual(summary["offering_price_needs_review_rows"], 0)
            self.assertEqual(summary["listing_open_price_rows"], 1)
            self.assertEqual(summary["listing_close_price_rows"], 1)
            self.assertAlmostEqual(features.loc[0, "open_return_pct"], 50.0)
            self.assertAlmostEqual(features.loc[0, "close_return_pct"], 25.0)
            self.assertTrue((root / "raw" / "dart_ipo_raw.parquet").exists())
            self.assertTrue((root / "processed" / "feature_observations.parquet").exists())
            self.assertTrue((root / "processed" / "feature_time_validation.parquet").exists())
            audit = pd.read_parquet(root / "raw" / "dart_offering_price_audit.parquet")
            review_queue = pd.read_parquet(root / "raw" / "dart_offering_price_review_queue.parquet")
            self.assertIn("offering_price_review_status", audit.columns)
            self.assertIn("offering_price_audit_context", audit.columns)
            self.assertIn("filing_is_correction", audit.columns)
            self.assertEqual(len(review_queue), 0)
            self.assertTrue((root / "processed" / "data_collection_summary.json").exists())

            observations = pd.read_parquet(root / "processed" / "feature_observations.parquet")
            retail = observations[observations["feature_name"] == "retail_subscription_ratio"].iloc[0]
            self.assertTrue(retail["is_missing"])
            self.assertEqual(retail["missing_reason"], "official_underwriter_notice_not_collected")
            self.assertTrue(retail["human_review_required"])

    def test_document_014_tries_another_receipt_in_the_same_lineage(self):
        class FallbackDART(_FakeDART):
            def get_ipo_disclosure_list(self, start_date, end_date):
                return pd.DataFrame([
                    {
                        "corp_code": "12345678", "corp_name": "테스트(주)",
                        "rcept_no": "20240101000002", "rcept_dt": pd.Timestamp("2024-02-01"),
                        "report_nm": "[발행조건확정]증권신고서(지분증권)",
                    },
                    {
                        "corp_code": "12345678", "corp_name": "테스트(주)",
                        "rcept_no": "20240101000001", "rcept_dt": pd.Timestamp("2024-01-01"),
                        "report_nm": "증권신고서(지분증권)",
                    },
                ])

            def get_offering_info(self, rcept_no):
                if rcept_no == "20240101000002":
                    raise RuntimeError("DART 원문 ZIP 응답이 아닙니다: <status>014</status>")
                return super().get_offering_info(rcept_no)

            def get_equity_offering_prices(self, corp_code, start_date, end_date):
                return []

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            HistoricalIPOPipeline(
                dart_collector=FallbackDART(),
                krx_collector=_FakeKRX(),
                raw_dir=root / "raw",
                processed_dir=root / "processed",
            ).run(2024, 2024, feature_set="phase2")

            audit = pd.read_parquet(root / "raw" / "dart_document_failure_audit.parquet")
            raw = pd.read_parquet(root / "raw" / "dart_ipo_raw.parquet")
            self.assertIn("zip_file_missing_retry_required", set(audit["failure_classification"]))
            self.assertEqual(raw.loc[0, "rcept_no"], "20240101000001")

    def test_feature_time_audit_blocks_post_listing_feature(self):
        features = pd.DataFrame({
            "event_id": ["a", "b"],
            "corp_name": ["전", "후"],
            "listing_date": ["2024-01-10", "2024-01-10"],
            "feature_available_at": ["2024-01-09", "2024-01-11"],
        })
        audit = HistoricalIPOPipeline._build_feature_time_audit(features)

        self.assertEqual(audit["is_future_information"].tolist(), [False, True])
        self.assertEqual(audit.loc[1, "time_validation_status"], "future_information_blocked")

    def test_manual_price_override_promotes_audited_record_for_training(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manual_dir = root / "manual"
            manual_dir.mkdir()
            (manual_dir / "offering_price_overrides.csv").write_text(
                "rcept_no,offering_price,decision,note\n"
                "20240101000001,12000,verified,Confirmed against original\n",
                encoding="utf-8",
            )
            pipeline = HistoricalIPOPipeline(
                dart_collector=_FakeDART(),
                krx_collector=_FakeKRX(),
                raw_dir=root / "raw",
                processed_dir=root / "processed",
            )

            summary = pipeline.run(2024, 2024, feature_set="phase2")
            audit = pd.read_parquet(root / "raw" / "dart_offering_price_audit.parquet")

            self.assertEqual(summary["offering_price_manual_verified_rows"], 1)
            self.assertEqual(audit.loc[0, "offering_price_review_status"], "manual_verified")
            self.assertEqual(audit.loc[0, "offering_price"], 12000)

    def test_latest_correction_is_preferred_when_disclosure_dates_match(self):
        class CorrectionDART(_FakeDART):
            def get_ipo_disclosure_list(self, start_date, end_date):
                return pd.DataFrame([
                    {
                        "corp_code": "12345678", "corp_name": "테스트(주)",
                        "rcept_no": "20240101000001", "rcept_dt": pd.Timestamp("2024-01-01"),
                        "report_nm": "증권신고서(지분증권)",
                    },
                    {
                        "corp_code": "12345678", "corp_name": "테스트(주)",
                        "rcept_no": "20240101000002", "rcept_dt": pd.Timestamp("2024-01-01"),
                        "report_nm": "증권신고서(지분증권)(정정)",
                    },
                ])

            def get_offering_info(self, rcept_no):
                result = super().get_offering_info(rcept_no)
                result["rcept_no"] = rcept_no
                return result

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            HistoricalIPOPipeline(
                dart_collector=CorrectionDART(),
                krx_collector=_FakeKRX(),
                raw_dir=root / "raw",
                processed_dir=root / "processed",
            ).run(2024, 2024, feature_set="phase2")
            audit = pd.read_parquet(root / "raw" / "dart_offering_price_audit.parquet")

            self.assertEqual(audit.loc[0, "rcept_no"], "20240101000002")
            self.assertTrue(audit.loc[0, "filing_is_correction"])

    def test_structured_dart_price_is_used_when_text_price_needs_review(self):
        offering = {
            "offering_price": None,
            "offering_price_review_status": "needs_review_no_currency_unit",
            "offering_price_parse_method": "unverified_numeric_candidate",
        }

        reconciled = HistoricalIPOPipeline._reconcile_structured_offering_price(
            offering,
            {"rcept_no": "20240101000001", "offering_price": 12000, "security_type": "보통주"},
        )

        self.assertEqual(reconciled["offering_price"], 12000)
        self.assertEqual(reconciled["offering_price_review_status"], "verified_structured_api")
        self.assertEqual(reconciled["structured_price_check"], "structured_price_used")

    def test_structured_price_from_non_final_report_stays_in_audit(self):
        offering = {"offering_price": None, "offering_price_review_status": "missing"}

        reconciled = HistoricalIPOPipeline._reconcile_structured_offering_price(
            offering,
            {"rcept_no": "20240101000001", "offering_price": 9000, "security_type": "보통주"},
            is_final_price_disclosure=False,
        )

        self.assertIsNone(reconciled["offering_price"])
        self.assertEqual(reconciled["structured_price_check"], "structured_price_unverified_report_type")

    def test_second_run_reuses_listing_price_and_dart_document_cache(self):
        class CountingDART(_FakeDART):
            def __init__(self):
                self.offering_calls = 0

            def get_offering_info(self, rcept_no):
                self.offering_calls += 1
                return super().get_offering_info(rcept_no)

        class CountingKRX(_FakeKRX):
            def __init__(self):
                self.price_calls = 0

            def get_listing_day_price(self, *args, **kwargs):
                self.price_calls += 1
                return super().get_listing_day_price(*args, **kwargs)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dart = CountingDART()
            krx = CountingKRX()
            pipeline = HistoricalIPOPipeline(
                dart_collector=dart,
                krx_collector=krx,
                raw_dir=root / "raw",
                processed_dir=root / "processed",
            )
            pipeline.run(2024, 2024, feature_set="phase2")
            pipeline.run(2024, 2024, feature_set="phase2")

            self.assertEqual(dart.offering_calls, 1)
            self.assertEqual(krx.price_calls, 1)
            cached_prices = pd.read_parquet(root / "raw" / "ipo_listing_prices.parquet")
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(cached_prices["listing_date"]))


if __name__ == "__main__":
    unittest.main()
