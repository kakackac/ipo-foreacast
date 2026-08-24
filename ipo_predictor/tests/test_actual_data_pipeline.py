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
        }])

    def get_offering_info(self, rcept_no):
        return {
            "rcept_no": rcept_no, "price_band_low": 9000, "price_band_high": 11000,
            "offering_price": 12000, "new_shares": 1000000, "secondary_shares": 200000,
            "total_post_listing_shares": 5000000, "lead_underwriter": "한국투자증권",
            "major_shareholder_lockup_months": 24, "risk_factor_count": 5,
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
    def get_ipo_calendar(self, start_date, end_date):
        if not start_date.startswith("2024"):
            return pd.DataFrame()
        return pd.DataFrame([{
            "ticker": "123456", "isu_cd": "KR7123456000", "corp_name": "테스트㈜",
            "listing_date": pd.Timestamp("2024-05-10"), "market": "KOSDAQ",
            "sector": "소프트웨어", "same_day_ipo_count": 1,
        }])

    def get_listing_day_price(self, ticker, listing_date, isu_cd=None):
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
            self.assertAlmostEqual(features.loc[0, "open_return_pct"], 50.0)
            self.assertAlmostEqual(features.loc[0, "close_return_pct"], 25.0)
            self.assertTrue((root / "raw" / "dart_ipo_raw.parquet").exists())
            self.assertTrue((root / "processed" / "data_collection_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
