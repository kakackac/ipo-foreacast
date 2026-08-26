import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data.collectors.dart_collector import DARTCollector
from data.processors.feature_engineer import FeatureEngineer, build_demo_dataset
from features.definitions import get_phase2_feature_names
import models.baseline.gradient_boost_model as gradient_boost_model
from models.baseline.gradient_boost_model import IPOPriceModel


class Phase2FeatureTests(unittest.TestCase):
    def test_phase2_demo_dataset_transforms_without_missing_values(self):
        df = build_demo_dataset(n=80, seed=7, phase="phase2")
        feature_names = get_phase2_feature_names()

        missing_columns = [col for col in feature_names if col not in df.columns]
        self.assertEqual(missing_columns, [])

        engineer = FeatureEngineer(feature_set="phase2")
        X = engineer.fit_transform(df)

        self.assertEqual(list(X.columns), feature_names)
        self.assertEqual(int(X.isna().sum().sum()), 0)
        self.assertEqual(int(df[["open_return_pct", "close_return_pct"]].isna().sum().sum()), 0)

    def test_listing_day_returns_use_offer_price_as_base(self):
        df = pd.DataFrame(
            {
                "offering_price": [10_000],
                "open_price": [12_000],
                "close_price": [11_000],
            }
        )

        result = FeatureEngineer()._calc_target(df)

        self.assertAlmostEqual(result.loc[0, "open_return_pct"], 20.0)
        self.assertAlmostEqual(result.loc[0, "close_return_pct"], 10.0)

    def test_merge_excludes_market_transfer_with_different_listing_date(self):
        dart = pd.DataFrame({
            "corp_name": ["테스트기업"],
            "listing_date": ["2020-01-10"],
            "offering_price": [10_000],
        })
        krx = pd.DataFrame({
            "corp_name": ["테스트기업", "테스트기업"],
            "listing_date": ["2020-01-10", "2022-03-15"],
            "open_price": [12_000, 15_000],
            "close_price": [11_000, 14_000],
        })

        merged = FeatureEngineer()._merge_base(dart, krx)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged.loc[0, "listing_date"], pd.Timestamp("2020-01-10"))

    def test_dart_offering_parser_extracts_phase2_fields(self):
        html = """
        <html><body>
        희망 공모가액 12,000 ~ 15,000 원
        확정 공모가 16,000 원
        신주모집 1,200,000 주
        구주매출 300,000 주
        상장 후 총 발행주식수 10,000,000 주
        대표주관회사 한국투자증권
        상장 예정일 2025년 03월 10일
        최대주주 의무보유기간 2년 6개월
        투자위험요소 가. 시장위험 나. 영업위험 다. 재무위험
        </body></html>
        """

        parsed = DARTCollector(api_key="test")._parse_offering_html(html, "20250101000001")

        self.assertTrue(parsed["parse_success"])
        self.assertEqual(parsed["price_band_low"], 12000)
        self.assertEqual(parsed["price_band_high"], 15000)
        self.assertEqual(parsed["offering_price"], 16000)
        self.assertEqual(parsed["new_shares"], 1200000)
        self.assertEqual(parsed["secondary_shares"], 300000)
        self.assertEqual(parsed["total_post_listing_shares"], 10000000)
        self.assertEqual(parsed["lead_underwriter"], "한국투자증권")
        self.assertEqual(parsed["listing_date"], "2025-03-10")
        self.assertEqual(parsed["major_shareholder_lockup_months"], 30)
        self.assertEqual(parsed["risk_factor_count"], 3)

    def test_dart_offering_parser_ignores_table_number_before_price(self):
        html = """
        확정 공모가 4 제 1 호 45,000 원
        희망 공모가 40,000 ~ 45,000 원
        """

        parsed = DARTCollector(api_key="test")._parse_offering_html(html, "20250101000002")

        self.assertEqual(parsed["offering_price"], 45_000)
        self.assertEqual(parsed["offering_price_review_status"], "verified_currency_unit")
        self.assertEqual(parsed["offering_price_extracted_amount"], 45_000)

    def test_dart_offering_parser_quarantines_number_without_currency_unit(self):
        parsed = DARTCollector(api_key="test")._parse_offering_html(
            "확정 공모가 4 제 1 호", "20250101000003"
        )

        self.assertIsNone(parsed["offering_price"])
        self.assertEqual(parsed["offering_price_extracted_amount"], 4)
        self.assertEqual(parsed["offering_price_review_status"], "needs_review_no_currency_unit")
        self.assertIn("확정 공모가 4", parsed["offering_price_audit_context"])

    def test_currency_unit_price_is_kept_even_when_outside_expected_range(self):
        parsed = DARTCollector(api_key="test")._parse_offering_html(
            "확정 공모가 50 원", "20250101000004"
        )

        self.assertEqual(parsed["offering_price"], 50)
        self.assertEqual(parsed["offering_price_review_status"], "verified_currency_unit")
        self.assertTrue(parsed["offering_price_range_warning"])

    def test_demand_forecast_extracts_price_for_source_comparison(self):
        parsed = DARTCollector(api_key="test")._parse_demand_forecast_html(
            "최종 공모가 45,000 원 기관 경쟁률 1,200 : 1", "12345678"
        )

        self.assertEqual(parsed["demand_offering_price"], 45_000)

    def test_equity_offering_price_flattens_dart_group_response(self):
        collector = DARTCollector(api_key="test")
        collector._get = lambda endpoint, params: {
            "group": [{"list": [{"rcept_no": "20250101000001", "slprc": "45,000", "stksen": "보통주"}]}]
        }

        prices = collector.get_equity_offering_prices("12345678", "20250101", "20250131")

        self.assertEqual(prices, [{"rcept_no": "20250101000001", "offering_price": 45_000, "security_type": "보통주"}])

    def test_preliminary_price_statement_does_not_treat_one_share_as_offer_price(self):
        parsed = DARTCollector._extract_offering_price_details(
            "모집가액의 확정은 수요예측 결과를 반영하여 1주당 확정공모가액을 최종 결정할 예정이며 "
            "모집가액 확정시 정정신고서를 제출할 예정입니다."
        )

        self.assertIsNone(parsed["offering_price"])
        self.assertIsNone(parsed["offering_price_extracted_amount"])
        self.assertEqual(parsed["offering_price_review_status"], "preliminary_price_language")
        self.assertEqual(parsed["offering_price_finality"], "preliminary_price_language")

    def test_unusual_offer_price_is_preserved_for_audit(self):
        df = pd.DataFrame({
            "offering_price": [4],
            "open_price": [40_000],
            "close_price": [38_000],
        })

        result = FeatureEngineer()._calc_target(df)

        self.assertEqual(result.loc[0, "offering_price"], 4)
        self.assertAlmostEqual(result.loc[0, "open_return_pct"], 999900.0)
        self.assertAlmostEqual(result.loc[0, "close_return_pct"], 949900.0)

    def test_classifier_fallback_is_preserved_after_load(self):
        df = build_demo_dataset(n=30, seed=13, phase="phase2")
        feature_names = get_phase2_feature_names()
        X = df[feature_names]
        y = pd.Series([5.0] * len(df))
        model = IPOPriceModel(n_estimators=5, max_depth=1)
        model.fit(X, y)

        original_model_dir = gradient_boost_model.MODEL_DIR
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                gradient_boost_model.MODEL_DIR = Path(temp_dir)
                model.save("fallback_persistence_test")
                loaded = IPOPriceModel.load("fallback_persistence_test")
                prediction = loaded.predict(X.head(1))
            finally:
                gradient_boost_model.MODEL_DIR = original_model_dir

        self.assertEqual(len(prediction), 1)
        self.assertEqual(float(prediction.iloc[0]["up_probability"]), 0.8)


if __name__ == "__main__":
    unittest.main()
