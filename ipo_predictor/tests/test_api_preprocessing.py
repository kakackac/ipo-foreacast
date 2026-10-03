import unittest

import pandas as pd

from api.server import _features_to_df


class _Model:
    feature_names = ["float_share_ratio", "float_share_ratio__missing", "underwriter_tier"]
    imputation_values = {"float_share_ratio": 0.27, "underwriter_tier": 2.0}


class APIPreprocessingTests(unittest.TestCase):
    def test_serving_uses_training_imputation_and_missing_indicator(self):
        result = _features_to_df({"float_share_ratio": None, "underwriter_tier": None}, _Model())

        self.assertAlmostEqual(result.loc[0, "float_share_ratio"], 0.27)
        self.assertEqual(result.loc[0, "float_share_ratio__missing"], 1.0)
        self.assertEqual(result.loc[0, "underwriter_tier"], 2.0)

    def test_serving_rejects_old_model_without_required_imputation_metadata(self):
        class LegacyModel:
            feature_names = ["float_share_ratio"]
            imputation_values = {}

        with self.assertRaisesRegex(RuntimeError, "결측 보정 메타데이터"):
            _features_to_df({"float_share_ratio": pd.NA}, LegacyModel())


if __name__ == "__main__":
    unittest.main()
