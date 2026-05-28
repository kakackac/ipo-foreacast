import unittest

from data.collectors.dart_collector import DARTCollector
from data.processors.feature_engineer import FeatureEngineer, build_demo_dataset
from features.definitions import get_phase2_feature_names


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


if __name__ == "__main__":
    unittest.main()
