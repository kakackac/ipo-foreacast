import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data.collectors.dart_collector import DARTCollector
from data.processors.feature_engineer import FeatureEngineer, build_demo_dataset
from features.definitions import get_phase2_feature_names
from features.model_profiles import build_stage_dataset, get_model_profile
from pipeline import assess_training_readiness
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

    def test_unverified_krx_listing_price_is_not_used_as_target(self):
        df = pd.DataFrame(
            {
                "offering_price": [10_000],
                "open_price": [12_000],
                "close_price": [11_000],
                "price_resolution_status": ["historical_identifier_review_required"],
            }
        )

        result = FeatureEngineer()._calc_target(df)

        self.assertTrue(pd.isna(result.loc[0, "open_return_pct"]))
        self.assertTrue(pd.isna(result.loc[0, "close_return_pct"]))
        self.assertEqual(
            result.loc[0, "price_target_validation_status"],
            "blocked_nonverified_krx_listing_price",
        )

    def test_missing_band_and_supply_values_are_not_changed_to_zero(self):
        engineer = FeatureEngineer(feature_set="phase2")
        base = pd.DataFrame({
            "offering_price": [None],
            "price_band_low": [None],
            "price_band_high": [None],
            "new_shares": [None],
            "secondary_shares": [None],
            "total_post_listing_shares": [None],
        })
        result = engineer._calc_band_position(base.copy())
        result = engineer._calc_supply_structure_features(result)

        self.assertTrue(pd.isna(result.loc[0, "offering_price_band_position"]))
        self.assertTrue(pd.isna(result.loc[0, "band_exceeded"]))
        self.assertTrue(pd.isna(result.loc[0, "secondary_offering_ratio"]))
        self.assertTrue(pd.isna(result.loc[0, "float_share_ratio"]))

    def test_partial_lockup_periods_do_not_create_a_weighted_score(self):
        result = FeatureEngineer()._calc_lockup_features(pd.DataFrame({
            "lockup_6m_ratio": [0.2],
            "lockup_3m_ratio": [0.1],
            "lockup_1m_ratio": [0.1],
            "lockup_15d_ratio": [None],
        }))

        self.assertTrue(result.loc[0, "lockup_components_missing"])
        self.assertTrue(pd.isna(result.loc[0, "lockup_weighted_score"]))

    def test_training_readiness_rejects_small_general_ipo_population(self):
        df = pd.DataFrame({
            "event_class": ["general_ipo"] * 46,
            "offering_price_review_status": ["verified_currency_unit"] * 46,
            "listing_date": pd.date_range("2024-01-01", periods=46, freq="7D"),
            "feature_available_at": pd.date_range("2023-12-01", periods=46, freq="7D"),
            "open_return_pct": [1.0] * 46,
            "close_return_pct": [1.0] * 46,
        })
        readiness = assess_training_readiness(df, phase="phase2")

        self.assertFalse(readiness["eligible"])
        self.assertEqual(readiness["general_ipo_dual_target_rows"], 46)
        self.assertTrue(any("최소 100건" in reason for reason in readiness["reasons"]))
        self.assertTrue(any("KRX 상장일 가격 검증 상태" in reason for reason in readiness["reasons"]))
        self.assertEqual(readiness["source_time_validated_core_complete_rows"], 0)

    def test_prediction_profiles_exclude_retail_subscription_ratio(self):
        pre_demand = get_model_profile("pre_demand")
        post_demand = get_model_profile("post_demand")

        self.assertNotIn("retail_subscription_ratio", pre_demand.feature_names)
        self.assertNotIn("retail_subscription_ratio", post_demand.feature_names)
        self.assertIn("institutional_demand_ratio", post_demand.feature_names)
        with self.assertRaises(ValueError):
            get_model_profile("post_retail")

    def test_stage_datasets_reuse_one_ipo_population_with_different_feature_contracts(self):
        features = pd.DataFrame({
            "event_id": ["common", "spac"],
            "event_class": ["general_ipo", "spac_ipo"],
            "offering_type": ["common_stock_ipo", "spac_ipo"],
            "offering_price_review_status": ["verified_currency_unit"] * 2,
            "open_return_pct": [10.0, 5.0],
            "close_return_pct": [8.0, 4.0],
            "listing_date": ["2024-01-10", "2024-01-11"],
        })
        for profile_name in ("pre_demand", "post_demand"):
            profile = get_model_profile(profile_name)
            for feature_name in profile.feature_names:
                features[feature_name] = 1.0
        audit = pd.DataFrame([
            {
                "event_id": event_id, "feature_name": feature_name,
                "is_missing": False, "time_validation_status": "pre_listing_or_same_day",
            }
            for event_id in features["event_id"]
            for feature_name in set().union(*(p.feature_names for p in (
                get_model_profile("pre_demand"), get_model_profile("post_demand")
            )))
        ])

        pre = build_stage_dataset(features, "pre_demand", audit)
        post_demand = build_stage_dataset(features, "post_demand", audit)

        self.assertEqual(len(pre), 2)
        self.assertEqual(len(post_demand), 2)
        self.assertTrue(pre["stage_model_candidate"].all())
        self.assertTrue(post_demand["stage_model_candidate"].all())

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

    def test_dart_offering_parser_accepts_only_direct_same_sentence_price(self):
        parsed = DARTCollector(api_key="test")._parse_offering_html(
            "확정 공모가는 20,000원입니다.", "20250101000005"
        )

        self.assertEqual(parsed["offering_price"], 20_000)
        self.assertEqual(parsed["offering_price_parse_method"], "final_price_same_sentence")

    def test_dart_offering_parser_rejects_nearby_face_value_and_price_band(self):
        parsed = DARTCollector(api_key="test")._parse_offering_html(
            "확정 공모가 관련 표에는 액면가 500원, 희망 공모가 12,000원 ~ 15,000원 및 "
            "총공모금액 100억원이 있습니다.",
            "20250101000006",
        )

        self.assertIsNone(parsed["offering_price"])
        self.assertEqual(parsed["offering_price_review_status"], "missing")

    def test_dart_offering_parser_accepts_confirmed_price_in_same_table_row(self):
        parsed = DARTCollector(api_key="test")._parse_offering_html(
            "<table><tr><th>확정 공모가</th><td>20,000원</td></tr>"
            "<tr><th>액면가</th><td>500원</td></tr></table>",
            "20250101000007",
        )

        self.assertEqual(parsed["offering_price"], 20_000)
        self.assertEqual(parsed["offering_price_parse_method"], "final_price_table_row")

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

    def test_dart_offering_result_approves_only_explicit_total_general_investor_subscription_ratio(self):
        parsed = DARTCollector(api_key="test")._parse_offering_result_html(
            "<table><tr><th>전체 일반청약자 경쟁률</th><td>1,234.56 : 1</td></tr></table>",
            "12345678",
        )

        self.assertEqual(parsed["retail_subscription_ratio"], 1234.56)
        self.assertEqual(parsed["retail_ratio_scope"], "dart_issuer_total_general_subscription")
        self.assertEqual(parsed["retail_validation_status"], "official_dart_issuer_total_retail_ratio")
        self.assertFalse(parsed["retail_human_review_required"])

    def test_dart_offering_result_with_institutional_subscription_is_not_retail(self):
        parsed = DARTCollector(api_key="test")._parse_offering_result_html(
            "<table><tr><th>전체 일반청약자 경쟁률</th><td>1,234.56 : 1</td></tr></table>"
            "일반공모에는 기관투자자의 청약이 포함되어 있습니다.",
            "12345678",
        )

        self.assertIsNone(parsed["retail_subscription_ratio"])
        self.assertEqual(
            parsed["retail_validation_status"],
            "dart_offering_result_institutional_included_not_retail",
        )

    def test_dart_general_offering_table_is_not_assumed_to_be_retail(self):
        parsed = DARTCollector(api_key="test")._parse_offering_result_html(
            "청약 및 배정현황 일반공모 최초 배정내역 청약현황 배정현황",
            "12345678",
        )

        self.assertIsNone(parsed["retail_subscription_ratio"])
        self.assertEqual(
            parsed["retail_validation_status"],
            "dart_offering_result_general_offering_not_retail_scope",
        )

    def test_dart_offering_result_quarantines_general_ratio_without_total_scope(self):
        parsed = DARTCollector(api_key="test")._parse_offering_result_html(
            "일반청약 경쟁률 1,234.56 : 1", "12345678"
        )

        self.assertIsNone(parsed["retail_subscription_ratio"])
        self.assertEqual(parsed["retail_subscription_ratio_candidate"], 1234.56)
        self.assertEqual(parsed["retail_validation_status"], "dart_offering_result_scope_review_required")

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
