import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from scripts.finalize_listing_eve_experiment import paired_improvement, check_partitions, evaluate, digest, SPEC


class ListingEveFinalTests(unittest.TestCase):
    def test_used_holdout_is_rejected_before_data_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps(SPEC))
            (root / "development_results").mkdir()
            predictions = root / "development_results/predictions.parquet"
            predictions.write_bytes(b"test-only-placeholder")
            (root / "final_policy.json").write_text(json.dumps({
                "manifest_sha256": digest(root / "manifest.json"),
                "development_predictions_sha256": digest(predictions),
                "feature_set": "post_demand",
                "selected_estimators": {"open_return_pct": "gradient_boosting", "close_return_pct": "training_target_median"}}))
            (root / "final_evaluation_started.json").write_text("{}")
            with patch("scripts.finalize_listing_eve_experiment.pd.read_parquet") as read:
                with self.assertRaises(FileExistsError):
                    evaluate(root)
                read.assert_not_called()

    def test_paired_improvement_has_correct_sign_and_day_clusters(self):
        result = paired_improvement([1, 1, 1], [1, 1, 1], [3, 3, 3],
            ["2026-01-01", "2026-01-01", "2026-01-02"], repetitions=100)
        self.assertEqual(result["mae_improvement_pp"], 2)
        self.assertEqual(result["day_clusters"], 2)
        np.testing.assert_allclose(result["day_cluster_bootstrap_95_interval_pp"], [2, 2])

    def test_partition_overlap_and_time_leak_rejected(self):
        development = pd.DataFrame({"event_id": [f"d{i}" for i in range(100)], "listing_date": ["2025-01-01"] * 100})
        holdout = pd.DataFrame({"event_id": [f"h{i}" for i in range(20)], "listing_date": ["2026-01-01"] * 20})
        check_partitions(development, holdout)
        holdout.loc[0, "event_id"] = "d0"
        with self.assertRaises(ValueError):
            check_partitions(development, holdout)
        holdout.loc[0, "event_id"] = "h0"
        holdout.loc[0, "listing_date"] = "2025-12-31"
        with self.assertRaises(ValueError):
            check_partitions(development, holdout)
