import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import requests

from data.collectors.dart_collector import DARTCollector
from data.pipelines.financial_history import FINANCIAL_STATUS, get_annual_history, summarize_asof, run_repair
from data.processors.feature_engineer import FeatureEngineer
from data.pipelines.research_dataset import run as build_research


def api_items(year=2022, receipt="20230315000001", currency="KRW"):
    values = {"ifrs-full_Revenue": "1,000", "dart_OperatingIncomeLoss": "-50",
              "ifrs-full_Liabilities": "100", "ifrs-full_Equity": "200",
              "ifrs-full_BasicEarningsLossPerShare": "10"}
    return [{"rcept_no": receipt, "corp_code": "12345678", "reprt_code": "11011",
             "bsns_year": str(year), "account_id": name, "thstrm_amount": amount,
             "sj_div": "BS" if name in {"ifrs-full_Liabilities", "ifrs-full_Equity"} else "IS",
             "currency": currency} for name, amount in values.items()]


class FakeFinancialDART:
    is_configured = True

    def __init__(self):
        self.calls = 0

    def get_financial_statements(self, corp_code, year):
        self.calls += 1
        receipt = f"{year+1}0315000001"
        return DARTCollector.normalize_financial_statements(api_items(year, receipt), corp_code, year, "11011", "CFS")

    def get_company_disclosure_list(self, corp_code, start_date, end_date):
        return pd.DataFrame([{"rcept_no": start_date + "000001", "corp_code": corp_code,
                              "rcept_dt": pd.Timestamp(start_date),
                              "report_nm": f"사업보고서 ({int(start_date[:4])-1}.12)"}])


class FinancialHistoryTests(unittest.TestCase):
    def history(self):
        frame = DARTCollector.normalize_financial_statements(api_items(), "12345678", 2022, "11011", "CFS")
        frame["publication_verified"] = True
        frame["published_at"] = "2023-03-15"
        return frame

    def test_negative_operating_income_and_receipt_metadata_are_preserved(self):
        frame = self.history()
        self.assertEqual(frame.loc[frame.account_name_en.eq("operating_income"), "amount"].iloc[0], -50)
        summary = summarize_asof(frame, "12345678", "2023-04-01 21:00")
        self.assertEqual(summary["operating_margin"], -0.05)
        self.assertEqual(summary["debt_ratio"], 0.5)
        self.assertEqual(summary["financial_time_validation_status"], FINANCIAL_STATUS)
        self.assertIsNone(summary["revenue_growth_3y"])

    def test_future_corrected_receipt_wrong_company_and_currency_are_excluded(self):
        for field, value in [("published_at", "2023-05-01"), ("corp_code", "87654321"),
                             ("currency", "USD"), ("publication_verified", False)]:
            frame = self.history()
            frame[field] = value
            self.assertNotIn("revenue", summarize_asof(frame, "12345678", "2023-04-01"))
        self.assertNotIn("revenue", summarize_asof(self.history(), "12345678", "2023-03-15 21:00"))

    def test_conflicting_duplicate_account_is_not_first_row_wins(self):
        frame = self.history()
        conflict = frame[frame.account_name_en.eq("revenue")].copy()
        conflict["amount"] = 2000
        summary = summarize_asof(pd.concat([frame, conflict]), "12345678", "2023-04-01")
        self.assertNotIn("operating_margin", summary)

    def test_three_year_growth_requires_exact_year_and_same_basis(self):
        frame = self.history()
        old = frame[frame.account_name_en.eq("revenue")].copy()
        old["year"], old["response_year"], old["rcept_no"] = 2019, "2019", "20200315000001"
        old["published_at"], old["amount"] = "2020-03-15", 125
        combined = pd.concat([frame, old])
        self.assertEqual(summarize_asof(combined, "12345678", "2023-04-01")["revenue_growth_3y"], 1)
        old["fs_div"] = "OFS"
        self.assertIsNone(summarize_asof(pd.concat([frame, old]), "12345678", "2023-04-01")["revenue_growth_3y"])

    def test_cache_reuses_successes_and_failure_is_not_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            dart = FakeFinancialDART()
            frame = get_annual_history(dart, "12345678", "2023-04-01", directory)
            self.assertEqual(dart.calls, 4)
            self.assertTrue(frame.publication_verified.all())
            get_annual_history(dart, "12345678", "2023-04-01", directory)
            self.assertEqual(dart.calls, 4)
            dart.get_financial_statements = Mock(side_effect=RuntimeError("status=020"))
            with self.assertRaises(RuntimeError):
                get_annual_history(dart, "12345678", "2024-04-01", directory)
            self.assertFalse((Path(directory) / "12345678_2023_11011_v1.json").exists())

    def test_ofs_fallback_only_after_no_cfs_data(self):
        collector = DARTCollector(api_key="test", document_cache_dir=None)
        collector._get = Mock(side_effect=[{"list": []}, {"list": api_items()}])
        frame = collector.get_financial_statements("12345678", 2022)
        self.assertTrue(frame.fs_div.eq("OFS").all())

    def test_rate_limit_is_failure_and_network_logs_do_not_expose_request_url(self):
        collector = DARTCollector(api_key="do-not-log-this", document_cache_dir=None)
        collector.session.get = Mock(return_value=Mock(json=Mock(return_value={"status": "020"})))
        with self.assertRaisesRegex(RuntimeError, "status=020"):
            collector._get("list", {})
        collector.session.get = Mock(side_effect=requests.ConnectionError("https://example/?crtfc_key=do-not-log-this"))
        with patch("data.collectors.dart_collector.time.sleep"), self.assertLogs("data.collectors.dart_collector", level="WARNING") as logs:
            with self.assertRaises(RuntimeError) as error:
                collector._get("list", {})
        self.assertNotIn("do-not-log-this", str(logs.output) + str(error.exception))

    def test_repair_publishes_features_and_evidence_without_overwriting_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            raw, processed = Path(directory)/"raw", Path(directory)/"processed"
            raw.mkdir(); processed.mkdir()
            features = pd.DataFrame([{"event_id": "event", "event_class": "general_ipo", "market": "KOSDAQ",
                "listing_date": pd.Timestamp("2023-04-01"), "corp_name": "test", "industry_name": "IT",
                "offering_price": 100, "revenue_growth_3y": None, "operating_margin": None, "debt_ratio": None}])
            features.to_parquet(processed/"features_all.parquet", index=False)
            pd.DataFrame([{"event_id": "event", "corp_code": "12345678"}]).to_parquet(raw/"dart_ipo_raw.parquet", index=False)
            pd.DataFrame().to_parquet(raw/"dart_financials.parquet")
            before = (processed/"features_all.parquet").read_bytes()
            report = run_repair(FakeFinancialDART(), raw, processed)
            root = Path(report["path"])
            result = pd.read_parquet(root/"features_all.parquet")
            observations = pd.read_parquet(root/"feature_observations.parquet")
            self.assertEqual(result.operating_margin.iloc[0], -0.05)
            used = observations[observations.feature_name.eq("operating_margin")].iloc[0]
            self.assertEqual(used.validation_status, FINANCIAL_STATUS)
            self.assertFalse(used.human_review_required)
            self.assertIn("20230315000001", used.source_reference)
            self.assertEqual(before, (processed/"features_all.parquet").read_bytes())

    def test_financial_repair_connects_to_research_dataset_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            raw, processed = Path(directory)/"raw", Path(directory)/"processed"
            raw.mkdir(); processed.mkdir()
            features = pd.DataFrame([{"event_id": "event", "event_class": "general_ipo", "market": "KOSDAQ",
                "listing_date": pd.Timestamp("2023-04-01"), "corp_name": "test", "industry_name": "IT",
                "offering_price": 100, "offering_price_review_status": "verified_currency_unit",
                "price_target_validation_status": "official_price_verified", "open_return_pct": 10,
                "close_return_pct": 5, "rcept_no": "20230315000002",
                "institutional_demand_ratio": 100, "institutional_rcept_no": "20230320000001",
                "institutional_available_at": pd.Timestamp("2023-03-20"),
                "institutional_validation_status": "verified_dart_structural_aggregate_v1",
                "revenue_growth_3y": None, "operating_margin": None, "debt_ratio": None}])
            features.to_parquet(processed/"features_all.parquet", index=False)
            FeatureEngineer(feature_set="phase2").build_feature_observations(features).to_parquet(processed/"feature_observations.parquet", index=False)
            pd.DataFrame([{"event_id": "event", "corp_code": "12345678"}]).to_parquet(raw/"dart_ipo_raw.parquet", index=False)
            pd.DataFrame().to_parquet(raw/"dart_financials.parquet")
            pd.DataFrame([{"event_id": "event", "rcept_no": "20230315000002", "rcept_dt": "2023-03-15"},
                          {"event_id": "event", "rcept_no": "20230320000001", "rcept_dt": "2023-03-20"}]).to_parquet(raw/"dart_disclosure_lineage.parquet", index=False)
            pd.DataFrame({"date": pd.date_range("2023-01-01", periods=95), "close": range(100, 195)}).to_parquet(raw/"kospi_index.parquet", index=False)
            repair = run_repair(FakeFinancialDART(), raw, processed)
            dataset = build_research(raw, processed, financial_repair=repair["path"])
            self.assertEqual(dataset["rows"], 1)
            frame = pd.read_parquet(Path(dataset["path"])/"dataset.parquet")
            self.assertEqual(frame.operating_margin.iloc[0], -0.05)
            self.assertEqual(frame.debt_ratio.iloc[0], 0.5)
            (Path(repair["path"])/"features_all.parquet").write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "hash_mismatch"):
                build_research(raw, processed, financial_repair=repair["path"])


if __name__ == "__main__":
    unittest.main()
