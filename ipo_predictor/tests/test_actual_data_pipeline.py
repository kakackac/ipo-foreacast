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
            "lockup_commitment_ratio": 0.4,
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
    def test_underwriter_institutional_review_queue_uses_event_master_without_web_discovery(self):
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
            (manual_dir / "underwriter_institutional_sources.csv").write_text(
                "event_id,corp_name,lead_underwriter,notice_url\n"
                "event-1,가나다,한국투자증권,https://securities.koreainvestment.com/notice/1\n",
                encoding="utf-8",
            )

            queue = HistoricalIPOPipeline(
                dart_collector=_FakeDART(), krx_collector=_FakeKRX(), raw_dir=raw_dir,
                processed_dir=root / "processed",
            ).prepare_underwriter_institutional_review_queue()

            self.assertEqual(queue["event_id"].tolist(), ["event-1", "event-2"])
            self.assertEqual(queue.loc[0, "review_status"], "official_institutional_url_already_linked")
            self.assertEqual(queue.loc[1, "review_status"], "official_institutional_url_required")
            self.assertTrue(queue["collection_policy"].str.startswith("manual_url_only").all())
            self.assertTrue((raw_dir / "official_underwriter_institutional_review_queue.parquet").exists())
            self.assertTrue((manual_dir / "underwriter_institutional_review_queue.csv").exists())

            readiness = HistoricalIPOPipeline(
                dart_collector=_FakeDART(), krx_collector=_FakeKRX(), raw_dir=raw_dir,
                processed_dir=root / "processed",
            ).audit_underwriter_source_readiness()
            self.assertEqual(set(readiness["lead_underwriter"]), {
                "한국투자증권", "미래에셋증권", "NH투자증권", "KB증권",
            })
            self.assertTrue((raw_dir / "official_underwriter_source_readiness.parquet").exists())
            self.assertTrue((manual_dir / "official_underwriter_source_readiness.csv").exists())

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

    def test_demand_forecast_candidates_keep_final_and_alternative_documents(self):
        collector = DARTCollector(api_key="a" * 40)
        collector.get_company_disclosure_list = Mock(return_value=pd.DataFrame([
            {
                "rcept_no": "20240201000001", "rcept_dt": pd.Timestamp("2024-02-01"),
                "report_nm": "증권신고서(지분증권)", "corp_code": "12345678",
            },
            {
                "rcept_no": "20240210000001", "rcept_dt": pd.Timestamp("2024-02-10"),
                "report_nm": "[발행조건확정]증권신고서(지분증권)", "corp_code": "12345678",
            },
            {
                "rcept_no": "20240212000001", "rcept_dt": pd.Timestamp("2024-02-12"),
                "report_nm": "수요예측결과", "corp_code": "12345678",
            },
        ]))

        records = collector.find_demand_forecast_disclosure_records("12345678", "20240101", "20240501")

        self.assertEqual([record["rcept_no"] for record in records], [
            "20240212000001", "20240210000001", "20240201000001",
        ])

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
            ).run(2024, 2024, feature_set="phase2", include_dart_demand_audit=True)

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
            self.assertEqual(features.loc[0, "underwriter_tier"], 1.0)
            self.assertTrue((root / "raw" / "dart_ipo_raw.parquet").exists())
            self.assertTrue((root / "raw" / "dart_institutional_extraction_audit.parquet").exists())
            self.assertTrue((root / "processed" / "feature_observations.parquet").exists())
            self.assertTrue((root / "processed" / "feature_coverage_audit.parquet").exists())
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
            coverage = pd.read_parquet(root / "processed" / "feature_coverage_audit.parquet")
            self.assertNotIn("retail_subscription_ratio", observations["feature_name"].tolist())
            self.assertIn("institutional_demand_ratio", summary["feature_coverage"])
            demand_coverage = coverage.set_index("feature_name").loc["institutional_demand_ratio"]
            self.assertEqual(demand_coverage["observed_rows"], 0)
            self.assertEqual(demand_coverage["missing_rows"], 1)
            raw = pd.read_parquet(root / "raw" / "dart_ipo_raw.parquet")
            self.assertEqual(
                raw.loc[0, "institutional_validation_status"],
                "dart_aggregate_value_not_verified",
            )

    def test_default_collection_uses_dart_final_terms_aggregate_demand(self):
        class FinalTermsDemandDART(_FakeDART):
            def get_offering_info(self, rcept_no):
                record = super().get_offering_info(rcept_no)
                record.update({
                    "institutional_demand_ratio": 850.0,
                    "lockup_commitment_ratio": 1.0,
                    "institutional_demand_parser_validation_status": "structurally_verified",
                    "lockup_parser_validation_status": "structurally_verified",
                    "institutional_demand_rule_id": "DART_DEMAND_DIRECT_LABEL_V1",
                    "lockup_rule_id": "DART_LOCKUP_DIRECT_AGGREGATE_V1",
                    "institutional_demand_parse_method": "demand_ratio_same_table_row_or_sentence",
                    "institutional_demand_evidence": "기관 수요예측 경쟁률 | 850.00 : 1",
                    "lockup_6m_ratio": 0.1,
                    "lockup_3m_ratio": 0.2,
                    "lockup_1m_ratio": 0.3,
                    "lockup_15d_ratio": 0.4,
                    "lockup_none_ratio": 0.0,
                    "lockup_parse_method": "lockup_same_table_row_percent",
                    "lockup_parse_evidence": "의무보유확약기간 | 6개월 | 10.0%",
                })
                return record

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            summary = HistoricalIPOPipeline(
                dart_collector=FinalTermsDemandDART(),
                krx_collector=_FakeKRX(),
                raw_dir=root / "raw",
                processed_dir=root / "processed",
            ).run(2024, 2024, feature_set="phase2")

            raw = pd.read_parquet(root / "raw" / "dart_ipo_raw.parquet")
            coverage = pd.read_parquet(root / "processed" / "feature_coverage_audit.parquet")
            self.assertEqual(raw.loc[0, "institutional_demand_ratio"], 850.0)
            self.assertEqual(raw.loc[0, "lockup_commitment_ratio"], 1.0)
            self.assertEqual(raw.loc[0, "institutional_rcept_no"], "20240101000001")
            self.assertEqual(raw.loc[0, "lockup_rcept_no"], "20240101000001")
            self.assertEqual(
                raw.loc[0, "institutional_source_url"],
                "https://dart.fss.or.kr/dsaf001/main.do?rcpNo=20240101000001",
            )
            self.assertEqual(
                raw.loc[0, "institutional_validation_status"],
                "verified_dart_structural_aggregate_v1",
            )
            self.assertEqual(
                coverage.set_index("feature_name").loc["institutional_demand_ratio", "observed_rows"],
                1,
            )
            self.assertEqual(summary["feature_coverage"]["institutional_demand_ratio"]["observed_rows"], 1)

            observations = pd.read_parquet(root / "processed" / "feature_observations.parquet")
            institutional_observation = observations.loc[
                observations["feature_name"] == "institutional_demand_ratio"
            ].iloc[0]
            self.assertEqual(
                institutional_observation["source_reference"],
                "https://dart.fss.or.kr/dsaf001/main.do?rcpNo=20240101000001",
            )

    def test_default_collection_searches_dart_demand_lineage(self):
        class DemandAuditSpy(_FakeDART):
            def __init__(self):
                self.demand_lookup_calls = 0
                self.demand_parse_calls = 0

            def find_demand_forecast_disclosure(self, corp_code, start_date, end_date):
                self.demand_lookup_calls += 1
                return "20240201000001"

            def get_demand_forecast(self, corp_code, rcept_no):
                self.demand_parse_calls += 1
                return {"corp_code": corp_code, "institutional_demand_ratio": None,
                        "lockup_commitment_ratio": None, "parse_success": False}

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dart = DemandAuditSpy()
            HistoricalIPOPipeline(
                dart_collector=dart,
                krx_collector=_FakeKRX(),
                raw_dir=root / "raw",
                processed_dir=root / "processed",
            ).run(2024, 2024, feature_set="phase2")

            raw = pd.read_parquet(root / "raw" / "dart_ipo_raw.parquet")
            self.assertEqual(dart.demand_lookup_calls, 1)
            self.assertEqual(dart.demand_parse_calls, 2)
            self.assertNotIn("institutional_demand_ratio", raw.columns)
            self.assertEqual(
                raw.loc[0, "institutional_validation_status"],
                "dart_aggregate_value_not_verified",
            )

    def test_offering_parser_v4_reparses_v3_cached_float_value(self):
        class VersionedOfferingDART(_FakeDART):
            def __init__(self):
                self.offering_calls = 0

            def get_offering_info(self, rcept_no):
                self.offering_calls += 1
                return super().get_offering_info(rcept_no)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            pd.DataFrame([{
                "rcept_no": "20240101000001",
                "offering_price": 12000,
                "public_float_shares": 999_999,
                "structured_price_check_version": 3,
                "offering_price_parser_version": 3,
            }]).to_parquet(raw_dir / "dart_offering_document_cache.parquet", index=False)

            dart = VersionedOfferingDART()
            HistoricalIPOPipeline(
                dart_collector=dart,
                krx_collector=_FakeKRX(),
                raw_dir=raw_dir,
                processed_dir=root / "processed",
            ).run(2024, 2024, feature_set="phase2")

            cache = pd.read_parquet(raw_dir / "dart_offering_document_cache.parquet")
            latest = cache.loc[cache["rcept_no"] == "20240101000001"].iloc[-1]
            self.assertEqual(dart.offering_calls, 1)
            self.assertEqual(latest["offering_price_parser_version"], 4)
            self.assertEqual(latest["dart_final_terms_demand_parser_version"], 5)
            self.assertNotEqual(latest["public_float_shares"], 999_999)

    def test_final_terms_document_parses_offering_and_demand_without_cross_overwriting_status(self):
        collector = DARTCollector(api_key="a" * 40)
        collector.get_document_text = Mock(return_value=(
            "희망 공모가 10,000원 ~ 12,000원. 기관투자자 수요예측 경쟁률 850.00 : 1."
        ))
        collector._parse_offering_html = Mock(return_value={
            "rcept_no": "20240101000001", "parse_success": True,
        })

        parsed = collector.get_offering_info("20240101000001")

        self.assertTrue(parsed["parse_success"])
        self.assertTrue(parsed["dart_final_terms_demand_parse_success"])
        self.assertEqual(parsed["institutional_demand_ratio"], 850.0)
        self.assertEqual(parsed["dart_final_terms_demand_parser_version"], 5)

    def test_collection_omits_removed_personal_subscription_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            HistoricalIPOPipeline(
                dart_collector=_FakeDART(),
                krx_collector=_FakeKRX(),
                raw_dir=root / "raw",
                processed_dir=root / "processed",
            ).run(2024, 2024, feature_set="phase2")

            raw = pd.read_parquet(root / "raw" / "dart_ipo_raw.parquet")
            self.assertNotIn("retail_subscription_ratio", raw.columns)
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

    def test_lineage_uses_prior_filing_for_price_band_without_replacing_final_price_source(self):
        class LineageFieldsDART(_FakeDART):
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
                base = super().get_offering_info(rcept_no)
                if rcept_no == "20240101000002":
                    base.update({"price_band_low": None, "price_band_high": None, "offering_price": 12000})
                else:
                    base.update({"price_band_low": 9000, "price_band_high": 11000, "offering_price": None})
                return base

            def get_equity_offering_prices(self, corp_code, start_date, end_date):
                return []

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            HistoricalIPOPipeline(
                dart_collector=LineageFieldsDART(),
                krx_collector=_FakeKRX(),
                raw_dir=root / "raw",
                processed_dir=root / "processed",
            ).run(2024, 2024, feature_set="phase2")

            raw = pd.read_parquet(root / "raw" / "dart_ipo_raw.parquet")
            self.assertEqual(raw.loc[0, "rcept_no"], "20240101000002")
            self.assertEqual(raw.loc[0, "offering_price"], 12000)
            self.assertEqual(raw.loc[0, "price_band_low"], 9000)
            self.assertEqual(raw.loc[0, "price_band_high"], 11000)
            self.assertEqual(raw.loc[0, "price_band_rcept_no"], "20240101000001")
            self.assertEqual(raw.loc[0, "price_band_rcept_dt"], pd.Timestamp("2024-01-01"))

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
            first.run(2024, 2024, feature_set="phase2", include_dart_demand_audit=True)
            second = HistoricalIPOPipeline(
                dart_collector=DemandZipMissingDART(),
                krx_collector=_FakeKRX(),
                raw_dir=root / "raw",
                processed_dir=root / "processed",
            )
            second.run(2024, 2024, feature_set="phase2", include_dart_demand_audit=True)

            demand_failures = pd.read_parquet(root / "raw" / "dart_demand_document_failures.parquet")
            self.assertEqual(DemandZipMissingDART.demand_calls, 2)
            self.assertEqual(len(demand_failures), 2)
            self.assertTrue((demand_failures["reason"] == "zip_file_missing_retry_required").all())

    def test_demand_014_does_not_block_another_candidate_in_the_same_lineage(self):
        class AlternateDemandDART(_FakeDART):
            def find_demand_forecast_disclosure_records(self, corp_code, start_date, end_date):
                return [
                    {
                        "rcept_no": "20240201000001", "rcept_dt": pd.Timestamp("2024-02-01"),
                        "report_nm": "수요예측결과", "candidate_score": 100,
                    },
                    {
                        "rcept_no": "20240101000001", "rcept_dt": pd.Timestamp("2024-01-01"),
                        "report_nm": "[발행조건확정]증권신고서(지분증권)", "candidate_score": 80,
                    },
                ]

            def get_demand_forecast(self, corp_code, rcept_no):
                if rcept_no == "20240201000001":
                    raise RuntimeError("DART 원문 ZIP 응답이 아닙니다: <status>014</status>")
                return {
                    "corp_code": corp_code, "institutional_demand_ratio": 850.0,
                    "demand_offering_price": 12000,
                    "lockup_commitment_ratio": 1.0,
                    "institutional_demand_parser_validation_status": "structurally_verified",
                    "lockup_parser_validation_status": "structurally_verified",
                    "institutional_demand_rule_id": "DART_DEMAND_DIRECT_LABEL_V1",
                    "lockup_rule_id": "DART_LOCKUP_DIRECT_AGGREGATE_V1",
                    "lockup_6m_ratio": 0.1, "lockup_3m_ratio": 0.2,
                    "lockup_1m_ratio": 0.3, "lockup_15d_ratio": 0.4,
                    "lockup_none_ratio": 0.0, "parse_success": True,
                }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            HistoricalIPOPipeline(
                dart_collector=AlternateDemandDART(),
                krx_collector=_FakeKRX(),
                raw_dir=root / "raw",
                processed_dir=root / "processed",
            ).run(2024, 2024, feature_set="phase2", include_dart_demand_audit=True)

            raw = pd.read_parquet(root / "raw" / "dart_ipo_raw.parquet")
            failures = pd.read_parquet(root / "raw" / "dart_demand_document_failures.parquet")
            cache = pd.read_parquet(root / "raw" / "dart_demand_document_cache.parquet")
            self.assertEqual(raw.loc[0, "institutional_demand_ratio"], 850.0)
            self.assertEqual(raw.loc[0, "institutional_validation_status"], "verified_dart_structural_aggregate_v1")
            self.assertEqual(cache.loc[cache["rcept_no"] == "20240101000001", "institutional_demand_ratio"].iloc[0], 850.0)
            self.assertEqual(raw.loc[0, "institutional_rcept_no"], "20240101000001")
            self.assertEqual(raw.loc[0, "lockup_rcept_no"], "20240101000001")
            self.assertIn("20240201000001", set(failures["rcept_no"].astype(str)))

    def test_dart_lineage_accepts_latest_valid_institutional_fields_from_separate_documents(self):
        class SplitInstitutionalDART(_FakeDART):
            def find_demand_forecast_disclosure_records(self, corp_code, start_date, end_date):
                return [
                    {"rcept_no": "20240215000001", "rcept_dt": pd.Timestamp("2024-02-15"),
                     "report_nm": "수요예측결과", "candidate_score": 100},
                    {"rcept_no": "20240214000001", "rcept_dt": pd.Timestamp("2024-02-14"),
                     "report_nm": "[기재정정]투자설명서", "candidate_score": 40},
                ]

            def get_demand_forecast(self, corp_code, rcept_no):
                base = {"corp_code": corp_code, "demand_offering_price": 12000,
                        "institutional_demand_ratio": None, "lockup_commitment_ratio": None,
                        "parse_success": True}
                if rcept_no == "20240215000001":
                    return {**base, "institutional_demand_ratio": 850.0,
                            "institutional_demand_parser_validation_status": "structurally_verified",
                            "institutional_demand_rule_id": "DART_DEMAND_DIRECT_LABEL_V1",
                            "institutional_demand_evidence": "기관투자자 수요예측 경쟁률 850 : 1"}
                if rcept_no == "20240214000001":
                    return {**base, "lockup_commitment_ratio": 0.25,
                            "lockup_parser_validation_status": "structurally_verified",
                            "lockup_rule_id": "DART_LOCKUP_DIRECT_AGGREGATE_V1",
                            "lockup_parse_evidence": "기관투자자 의무보유확약 25.0%"}
                return base

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            HistoricalIPOPipeline(
                dart_collector=SplitInstitutionalDART(), krx_collector=_FakeKRX(),
                raw_dir=root / "raw", processed_dir=root / "processed",
            ).run(2024, 2024, feature_set="phase2")

            raw = pd.read_parquet(root / "raw" / "dart_ipo_raw.parquet")
            self.assertEqual(raw.loc[0, "institutional_demand_ratio"], 850.0)
            self.assertEqual(raw.loc[0, "lockup_commitment_ratio"], 0.25)
            self.assertEqual(raw.loc[0, "institutional_rcept_no"], "20240215000001")
            self.assertEqual(raw.loc[0, "lockup_rcept_no"], "20240214000001")
            self.assertEqual(raw.loc[0, "institutional_validation_status"], "verified_dart_structural_aggregate_v1")
            self.assertEqual(raw.loc[0, "lockup_validation_status"], "verified_dart_structural_aggregate_v1")

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
