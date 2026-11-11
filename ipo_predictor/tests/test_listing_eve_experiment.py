import unittest
import pandas as pd
from scripts.prepare_listing_eve_experiment import make_splits, validate_evidence


class ListingEveExperimentTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame([{"event_id": f"{year}-{i}", "listing_date": f"{year}-06-01"}
            for year, count in ((2022, 100), (2023, 25), (2024, 25), (2026, 20)) for i in range(count)])

    def test_holdout_not_used_in_nonoverlapping_development_folds(self):
        dev, holdout, folds, _ = make_splits(self.frame())
        self.assertEqual((len(dev), len(holdout)), (150, 20))
        validation_ids = []
        for fold in folds:
            self.assertFalse(set(fold["train_event_ids"]) & set(fold["validation_event_ids"]))
            self.assertFalse(set(holdout.event_id) & set(fold["train_event_ids"] + fold["validation_event_ids"]))
            validation_ids.extend(fold["validation_event_ids"])
        self.assertEqual(len(validation_ids), len(set(validation_ids)))

    def test_duplicate_event_aborts(self):
        frame = self.frame()
        with self.assertRaises(ValueError):
            make_splits(pd.concat([frame, frame.iloc[:1]]))

    def test_evidence_must_match_value_and_precede_listing(self):
        frame = pd.DataFrame([{"event_id": "a", "listing_date": "2025-01-02", "feature": 3.0}])
        evidence = pd.DataFrame([{"event_id": "a", "feature_name": "feature", "raw_value": 3.0,
            "available_at": "2025-01-01", "source_reference": "receipt", "is_missing": False,
            "human_review_required": False}])
        validate_evidence(frame, ["feature"], evidence)
        for column, value in (("raw_value", 4.0), ("available_at", "2025-01-02"), ("human_review_required", True)):
            invalid = evidence.copy()
            invalid[column] = value
            with self.assertRaises(ValueError):
                validate_evidence(frame, ["feature"], invalid)
