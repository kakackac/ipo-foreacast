import unittest
import pandas as pd
from features.model_profiles import MODEL_PROFILES
from scripts.audit_training_cohorts import cohort_mask


class TrainingCohortAuditTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({
            "event_class": ["general_ipo"] * 3, "market": ["KOSDAQ"] * 3,
            "institutional_demand_ratio": [100, 200, 300],
            "offering_price_band_position": [1, 1, 1],
            "lockup_commitment_ratio": [None, None, None],
            "institutional_validation_status": ["verified_dart_structural_aggregate_v1", "needs_review", "verified_dart_structural_aggregate_v1"],
            "lockup_validation_status": ["missing"] * 3,
            "stage_offering_price_verified": [True] * 3,
            "stage_dual_target_ready": [True] * 3,
            "stage_time_valid": [True, True, False],
            "stage_model_candidate": [True] * 3,
            "price_target_validation_status": ["official_price_verified"] * 3,
            "open_return_pct": [1., 2., 3.], "close_return_pct": [1., 2., 3.],
        })

    def test_quarantined_rows_do_not_poison_valid_subset(self):
        self.assertEqual(cohort_mask(self.frame(), MODEL_PROFILES["post_demand"]).tolist(), [True, False, False])

    def test_spac_and_nonfinite_target_excluded(self):
        frame = self.frame()
        frame.loc[0, "event_class"] = "spac_ipo"
        self.assertFalse(cohort_mask(frame, MODEL_PROFILES["post_demand"]).any())
        frame = self.frame()
        frame.loc[0, "open_return_pct"] = float("inf")
        self.assertFalse(cohort_mask(frame, MODEL_PROFILES["post_demand"]).any())
