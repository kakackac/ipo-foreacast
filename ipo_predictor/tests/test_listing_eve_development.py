import unittest
import pandas as pd
from scripts.run_listing_eve_development import prepare_inputs, validate_fold


class ListingEveDevelopmentTests(unittest.TestCase):
    def test_imputation_uses_training_only(self):
        train = pd.DataFrame({"a": [1., 3., None]})
        validation = pd.DataFrame({"a": [10000., None]})
        X, V = prepare_inputs(train, validation, ["a"])
        self.assertEqual(X.loc[2, "a"], 2.)
        self.assertEqual(V.loc[1, "a"], 2.)
        self.assertEqual(V.loc[1, "a__missing"], 1)

    def test_empty_training_feature_aborts(self):
        with self.assertRaises(ValueError):
            prepare_inputs(pd.DataFrame({"a": [None]}), pd.DataFrame({"a": [3.]}), ["a"])

    def test_holdout_or_overlap_aborts(self):
        frame = pd.DataFrame({"listing_date": ["2024-01-01"] * 100 + ["2026-01-01"] * 20})
        fold = {"train_event_ids": list(range(100)), "validation_event_ids": list(range(100, 120))}
        with self.assertRaises(ValueError):
            validate_fold(frame, fold)
        fold["validation_event_ids"] = [0] + list(range(101, 120))
        with self.assertRaises(ValueError):
            validate_fold(frame, fold)
