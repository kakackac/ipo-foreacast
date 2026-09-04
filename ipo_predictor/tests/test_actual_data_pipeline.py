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
            "market": market,
            "open_price": 18000, "close_price": 15000, "high_price": 19000,
            "low_price": 14000, "volume": 100000,
            "price_match_status": "matched", "price_match_method": "ticker_or_short_issue_code",
            "price_failure_reason": None, "price_markets_queried": market,
            "price_api_rows_returned": 1, "price_raw_response_evidence": "{}",
            "price_matched_ticker": ticker, "price_matched_isu_cd": isu_cd,
            "price_matched_corp_name": corp_name,
        }

    def get_index_ohlcv(self, index_code, start_date, end_date):
        dates = pd.bdate_range("2024-01-01", "2024-12-31")
        return pd.DataFrame({
            "date": dates, "index_code": "KOSPI" if index_code == "1" else "KOSDAQ",
            "close": range(2000, 2000 + len(dates)),
        })


class ActualDataPipelineTests(unittest.TestCase):
    def test_underwriter_notice_review_queue_uses_event_master_without_web_discovery(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw"
            manual_dir = root / "manual"
            raw_dir.mkdir()
            manual_dir.mkdir()
            pd.DataFrame([
                {
                    "event_id": "event-1", "event_class": "general_ipo",
                    "offering_type": "common_stock_ipo", "ticker": "000001", "corp_name": "가나다",
                    "lead_underwriter": "한국투자증권(주)", "market": "KOSDAQ",
                    "listing_date": "2024-01-10", "offering_price": 10000,
                    "source_url": "https://kind.krx.co.kr/event-1",
                },
                {
                    "event_id": "event-2", "event_class": "spac_ipo",
                    "offering_type": "spac_ipo", "ticker": "000002", "corp_name": "테스트스팩",
                    "lead_underwriter": "KB증권(주)", "market": "KOSDAQ",
                    "listing_date": "2024-01-11", "offering_price": 2000,
                    "source_url": "https://kind.krx.co.kr/event-2",
                },
                {
                    "event_id": "event-3", "event_class": "general_ipo",
                    "offering_type": "common_stock_ipo", "ticker": "000003", "corp_name": "제외",
                    "lead_underwriter": "다른증권", "market": "KOSDAQ",
                    "listing_date": "2024-01-12", "offering_price": 10000,
                    "source_url": "https://kind.krx.co.kr/event-3",
                },
            ]).to_parquet(raw_dir / "krx_official_event_master.parquet", index=False)
            (manual_dir / "underwriter_notice_sources.csv").write_text(
                "event_id,corp_name,lead_underwriter,notice_url\n"
                "event-1,가나다,한국투자증권,https://securities.koreainvestment.com/notice/1\n",
                encoding="utf-8",
            )

            queue = HistoricalIPOPipeline(
                dart_collector=_FakeDART(), krx_collector=_FakeKRX(), raw_dir=raw_dir,
                processed_dir=root / "processed",
            ).prepare_underwriter_notice_review_queue()

            self.assertEqual(queue["event_id"].tolist(), ["event-1", "event-2"])
            self.assertEqual(queue.loc[0, "review_status"], "official_notice_url_already_linked")
            self.assertEqual(queue.loc[1, "review_status"], "official_notice_url_required")
            self.assertTrue(queue["collection_policy"].str.startswith("manual_url_only").all())
            self.assertTrue((raw_dir / "official_underwriter_notice_review_queue.parquet").exists())
            self.assertTrue((manual_dir / "underwriter_notice_review_queue.csv").exists())

    def test_dart_no_data_status_is_normalised_to_an_empty_list(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"status": "013", "message": "조회된 데이타가 없습니다."}
        collector = DARTCollector(api_key="a" * 40)
        collector.session.get = Mock(return_value=response)

        result = collector._get("fnlttSinglAcntAll", {})

        self.assertEqual(result, {"status": "013", "list": []})

    def test_demand_parser_does_not_treat_any_competition_ratio_as_institutional(self):
        collector = DARTCollector(api_key="a" * 40)
        result = collector._parse_demand_forecast_html(
            "비례배정 경쟁률 1,364.23 : 1", "12345678"
        )

        self.assertIsNone(result["institutional_demand_ratio"])

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

    def test_demand_forecast_candidate_keeps_receipt_date(self):
        collector = DARTCollector(api_key="a" * 40)
        collector.get_company_disclosure_list = Mock(return_value=pd.DataFrame([{
            "rcept_no": "20240201000001", "rcept_dt": pd.Timestamp("2024-02-01"),
            "report_nm": "수요예측결과", "corp_code": "12345678",
        }]))

        record = collector.find_demand_forecast_disclosure_record("12345678", "20240101", "20240501")

        self.assertEqual(record["rcept_no"], "20240201000001")
        self.assertEqual(record["rcept_dt"], pd.Timestamp("2024-02-01"))

    def test_offering_result_candidate_requires_statutory_report_title(self):
        collector = DARTCollector(api_key="a" * 40)
        collector.get_company_disclosure_list = Mock(return_value=pd.DataFrame([
            {
                "rcept_no": "20240201000001", "rcept_dt": pd.Timestamp("2024-02-01"),
                "report_nm": "유상증자결정", "corp_code": "12345678",
            },
            {
                "rcept_no": "20240202000001", "rcept_dt": pd.Timestamp("2024-02-02"),
                "report_nm": "증권발행실적보고서", "corp_code": "12345678",
            },
        ]))

        record = collector.find_offering_result_disclosure_record("12345678", "20240101", "20240501")

        self.assertEqual(record["rcept_no"], "20240202000001")
        self.assertEqual(record["report_nm"], "증권발행실적보고서")

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
            for stage_name in ("pre_demand", "post_demand"):
                stage_path = root / "processed" / "model_stage_datasets" / f"{stage_name}.parquet"
                self.assertTrue(stage_path.exists())
                stage = pd.read_parquet(stage_path)
                self.assertEqual(len(stage), 1)
                self.assertIn("stage_model_candidate", stage.columns)
            self.assertTrue((root / "processed" / "model_stage_readiness.json").exists())
            self.assertIn("model_stage_readiness", summary)
            audit = pd.read_parquet(root / "raw" / "dart_offering_price_audit.parquet")
            review_queue = pd.read_parquet(root / "raw" / "dart_offering_price_review_queue.parquet")
            self.assertIn("offering_price_review_status", audit.columns)
            self.assertIn("offering_price_audit_context", audit.columns)
            self.assertIn("filing_is_correction", audit.columns)
            self.assertEqual(len(review_queue), 0)
            self.assertTrue((root / "processed" / "data_collection_summary.json").exists())

            observations = pd.read_parquet(root / "processed" / "feature_observations.parquet")
            self.assertNotIn("retail_subscription_ratio", observations["feature_name"].tolist())

    def test_default_collection_skips_retail_audit_sources(self):
        class RetailAuditSpy(_FakeDART):
            def __init__(self):
                self.offering_result_lookup_calls = 0

            def find_offering_result_disclosure_record(self, corp_code, start_date, end_date):
                self.offering_result_lookup_calls += 1
                raise AssertionError("기본 수집은 개인청약 감사 원천을 조회하면 안 됩니다.")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dart = RetailAuditSpy()
            summary = HistoricalIPOPipeline(
                dart_collector=dart,
                krx_collector=_FakeKRX(),
                raw_dir=root / "raw",
                processed_dir=root / "processed",
            ).run(2024, 2024, feature_set="phase2")

            self.assertEqual(summary["retail_audit_mode"], "deferred")
            self.assertEqual(dart.offering_result_lookup_calls, 0)
            self.assertFalse((root / "raw" / "dart_offering_result_audit.parquet").exists())

    def test_listing_price_audit_resolves_only_verified_target_events(self):
        calendar = pd.DataFrame([
            {"event_id": "matched", "ticker": "000001", "corp_name": "가", "listing_date": "2024-01-02",
             "market": "KOSDAQ", "event_class": "general_ipo", "offering_type": "common_stock_ipo"},
            {"event_id": "empty", "ticker": "000002", "corp_name": "나", "listing_date": "2024-01-03",
             "market": "KOSDAQ", "event_class": "general_ipo", "offering_type": "common_stock_ipo"},
            {"event_id": "identifier", "ticker": "000003", "corp_name": "다", "listing_date": "2024-01-04",
             "market": "KOSDAQ", "event_class": "general_ipo", "offering_type": "common_stock_ipo"},
            {"event_id": "relisting", "ticker": "000004", "corp_name": "라", "listing_date": "2024-01-05",
             "market": "KOSDAQ", "event_class": "relisting", "offering_type": "relisting"},
        ])
        prices = pd.DataFrame([
            {"ticker": "000001", "listing_date": "20240102", "open_price": 10000, "close_price": 11000,
             "price_match_status": "matched", "price_match_method": "krx_standard_code",
             "price_raw_response_evidence": "{}"},
            {"ticker": "000002", "listing_date": "20240103", "price_match_status": "unmatched",
             "price_failure_reason": "daily_price_api_response_empty", "price_raw_response_evidence": "{}"},
            {"ticker": "000003", "listing_date": "20240104", "price_match_status": "unmatched",
             "price_failure_reason": "daily_rows_code_and_company_name_unmatched",
             "price_raw_response_evidence": "{}"},
            {"ticker": "000004", "listing_date": "20240105", "price_match_status": "unmatched",
             "price_failure_reason": "daily_rows_code_and_company_name_unmatched",
             "price_raw_response_evidence": "{}"},
        ])

        audit = HistoricalIPOPipeline._build_listing_price_audit(calendar, prices)

        status = audit.set_index("event_id")["price_resolution_status"].to_dict()
        self.assertEqual(status["matched"], "official_price_verified")
        self.assertEqual(status["empty"], "official_price_unconfirmed")
        self.assertEqual(status["identifier"], "historical_identifier_review_required")
        self.assertEqual(status["relisting"], "excluded_non_target_event")

        enriched = HistoricalIPOPipeline._attach_price_resolution(calendar, audit)
        resolution = enriched.set_index("event_id")["price_resolution_status"].to_dict()
        self.assertEqual(resolution["matched"], "official_price_verified")
        self.assertEqual(resolution["identifier"], "historical_identifier_review_required")

    def test_cached_price_without_raw_evidence_is_not_officially_verified(self):
        calendar = pd.DataFrame([{
            "event_id": "legacy", "ticker": "000001", "corp_name": "가",
            "listing_date": "2024-01-02", "market": "KOSDAQ",
            "event_class": "general_ipo", "offering_type": "common_stock_ipo",
        }])
        prices = pd.DataFrame([{
            "ticker": "000001", "listing_date": "20240102", "open_price": 10000,
            "close_price": 11000, "price_match_status": "matched",
            "price_match_method": "prior_cache",
        }])

        audit = HistoricalIPOPipeline._build_listing_price_audit(calendar, prices)

        self.assertEqual(
            audit.loc[0, "price_resolution_status"],
            "historical_price_cache_reaudit_required",
        )

    def test_pipeline_keeps_dart_offering_result_as_audit_candidate_only(self):
        class ResultDART(_FakeDART):
            def find_offering_result_disclosure_record(self, corp_code, start_date, end_date):
                return {
                    "rcept_no": "20240501000001", "rcept_dt": pd.Timestamp("2024-05-01"),
                    "report_nm": "증권발행실적보고서",
                }

            def get_offering_result(self, corp_code, rcept_no):
                return {
                    "corp_code": corp_code,
                    "retail_subscription_ratio": 1234.56,
                    "retail_subscription_ratio_candidate": 1234.56,
                    "retail_ratio_scope": "dart_issuer_total_general_subscription",
                    "retail_parse_evidence": "전체 일반청약 경쟁률 1,234.56 : 1",
                    "retail_parse_method": "dart_offering_result_table_row",
                    "retail_validation_status": "official_dart_issuer_total_retail_ratio",
                    "retail_human_review_required": False,
                    "retail_parse_success": True,
                }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            summary = HistoricalIPOPipeline(
                dart_collector=ResultDART(),
                krx_collector=_FakeKRX(),
                raw_dir=root / "raw",
                processed_dir=root / "processed",
            ).run(2024, 2024, feature_set="phase2", include_retail_audit=True)

            raw = pd.read_parquet(root / "raw" / "dart_ipo_raw.parquet")
            audit = pd.read_parquet(root / "raw" / "dart_offering_result_audit.parquet")
            observations = pd.read_parquet(root / "processed" / "feature_observations.parquet")

            self.assertEqual(summary["dart_offering_result_candidate_rows"], 1)
            self.assertEqual(summary["dart_offering_result_approved_retail_rows"], 1)
            self.assertTrue(pd.isna(raw.loc[0, "retail_subscription_ratio"]))
            self.assertEqual(raw.loc[0, "retail_subscription_ratio_candidate"], 1234.56)
            self.assertEqual(audit.loc[0, "retail_result_rcept_no"], "20240501000001")
            self.assertNotIn("retail_subscription_ratio", observations["feature_name"].tolist())

    def test_second_run_reuses_versioned_dart_offering_result_audit(self):
        class CachedResultDART(_FakeDART):
            def __init__(self):
                self.result_calls = 0

            def find_offering_result_disclosure_record(self, corp_code, start_date, end_date):
                return {
                    "rcept_no": "20240501000001", "rcept_dt": pd.Timestamp("2024-05-01"),
                    "report_nm": "증권발행실적보고서",
                }

            def get_offering_result(self, corp_code, rcept_no):
                self.result_calls += 1
                return {
                    "corp_code": corp_code,
                    "retail_parser_version": 2,
                    "retail_subscription_ratio": None,
                    "retail_subscription_ratio_candidate": None,
                    "retail_ratio_scope": None,
                    "retail_parse_evidence": "일반공모 / 청약현황",
                    "retail_parse_method": "dart_offering_result_general_offering_table",
                    "retail_validation_status": "dart_offering_result_general_offering_not_retail_scope",
                    "retail_human_review_required": True,
                    "retail_parse_success": True,
                }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dart = CachedResultDART()
            pipeline = HistoricalIPOPipeline(
                dart_collector=dart,
                krx_collector=_FakeKRX(),
                raw_dir=root / "raw",
                processed_dir=root / "processed",
            )
            pipeline.run(2024, 2024, feature_set="phase2", include_retail_audit=True)
            pipeline.run(2024, 2024, feature_set="phase2", include_retail_audit=True)

            self.assertEqual(dart.result_calls, 1)

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

    def test_demand_document_014_is_cached_for_the_retry_window(self):
        class DemandZipMissingDART(_FakeDART):
            demand_calls = 0

            def get_demand_forecast(self, corp_code, rcept_no):
                type(self).demand_calls += 1
                raise RuntimeError("DART 원문 ZIP 응답이 아닙니다: <status>014</status>")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = HistoricalIPOPipeline(
                dart_collector=DemandZipMissingDART(),
                krx_collector=_FakeKRX(),
                raw_dir=root / "raw",
                processed_dir=root / "processed",
            )
            first.run(2024, 2024, feature_set="phase2")
            second = HistoricalIPOPipeline(
                dart_collector=DemandZipMissingDART(),
                krx_collector=_FakeKRX(),
                raw_dir=root / "raw",
                processed_dir=root / "processed",
            )
            second.run(2024, 2024, feature_set="phase2")

            demand_failures = pd.read_parquet(root / "raw" / "dart_demand_document_failures.parquet")
            self.assertEqual(DemandZipMissingDART.demand_calls, 1)
            self.assertEqual(len(demand_failures), 1)
            self.assertEqual(demand_failures.loc[0, "reason"], "zip_file_missing_retry_required")

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

    def test_legacy_listing_price_cache_without_evidence_is_reaudited(self):
        class CountingKRX(_FakeKRX):
            def __init__(self):
                self.price_calls = 0

            def get_listing_day_price(self, *args, **kwargs):
                self.price_calls += 1
                return super().get_listing_day_price(*args, **kwargs)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            pd.DataFrame([{
                "ticker": "123456", "listing_date": "20240510", "open_price": 18000,
                "close_price": 15000, "price_match_status": "matched",
            }]).to_parquet(raw_dir / "ipo_listing_prices.parquet", index=False)
            krx = CountingKRX()
            pipeline = HistoricalIPOPipeline(
                dart_collector=_FakeDART(),
                krx_collector=krx,
                raw_dir=raw_dir,
                processed_dir=root / "processed",
            )

            prices = pipeline._collect_listing_prices(
                krx.get_official_listing_events("20240101", "20241231")
            )

            self.assertEqual(krx.price_calls, 1)
            self.assertEqual(prices.loc[0, "price_match_status"], "matched")
            self.assertTrue(pd.notna(prices.loc[0, "price_raw_response_evidence"]))

    def test_attach_prices_replaces_prior_enrichment_without_duplicate_market_columns(self):
        calendar = pd.DataFrame([{
            "ticker": "123456", "listing_date": "2024-01-10", "market": "KOSDAQ",
            "market_price": "KOSDAQ", "isu_cd": "OLD", "open_price": 10_000,
            "close_price": 11_000, "verification_status": "previous",
        }])
        prices = pd.DataFrame([{
            "ticker": "123456", "listing_date": "2024-01-10", "market": "KOSPI",
            "isu_cd": "NEW", "open_price": 12_000, "close_price": 13_000,
            "high_price": 14_000, "low_price": 9_000, "volume": 1_000,
        }])

        merged = HistoricalIPOPipeline._attach_prices(calendar, prices)

        self.assertFalse(merged.columns.duplicated().any())
        self.assertEqual(merged.loc[0, "market"], "KOSPI")
        self.assertEqual(merged.loc[0, "krx_standard_code"], "NEW")
        self.assertEqual(merged.loc[0, "open_price"], 12_000)
        self.assertNotIn("market_price", merged.columns)

    def test_official_source_resolution_preserves_null_and_records_reason(self):
        observations = pd.DataFrame([{
            "event_id": "event-1", "feature_name": "institutional_demand_ratio", "is_missing": True,
            "missing_reason": "official_source_field_unavailable_or_unverified", "source_reference": None,
            "collected_at": pd.NaT, "validation_status": "needs_review", "human_review_required": True,
        }])
        resolutions = pd.DataFrame([{
            "event_id": "event-1", "feature_name": "institutional_demand_ratio",
            "resolution_status": "official_source_not_published", "checked_at": "2026-01-10",
            "checked_sources": "DART;KRX;official_notice", "reviewed_by": "reviewer", "note": "checked",
        }])

        result = HistoricalIPOPipeline._apply_official_source_resolutions(observations, resolutions)

        self.assertTrue(result.loc[0, "is_missing"])
        self.assertEqual(result.loc[0, "missing_reason"], "official_source_not_published")
        self.assertEqual(result.loc[0, "source_reference"], "DART;KRX;official_notice")

    def test_feature_time_audit_uses_each_feature_available_at(self):
        features = pd.DataFrame({
            "event_id": ["event-1"], "corp_name": ["테스트"], "listing_date": ["2026-01-20"],
            "feature_available_at": ["2026-01-10"],
        })
        observations = pd.DataFrame({
            "event_id": ["event-1"], "corp_name": ["테스트"], "listing_date": ["2026-01-20"],
            "feature_name": ["institutional_demand_ratio"], "available_at": ["2026-01-21"],
        })

        audit = HistoricalIPOPipeline._build_feature_time_audit(features, observations)

        self.assertTrue(audit.loc[0, "is_future_information"])
        self.assertEqual(audit.loc[0, "time_validation_status"], "future_information_blocked")


if __name__ == "__main__":
    unittest.main()
