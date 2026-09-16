import unittest
import pandas as pd
from scripts.apply_verified_sample_repairs import patch_event


class SampleRepairTests(unittest.TestCase):
    def test_patch_preserves_other_fields_and_events(self):
        original = pd.DataFrame({"event_id": ["a", "b"], "price": [1, 2], "source": ["old", "other"]})
        result = patch_event(original, "a", {"price": 3})
        self.assertEqual(result.price.tolist(), [3, 2])
        self.assertEqual(result.source.tolist(), ["old", "other"])
        self.assertEqual(original.price.tolist(), [1, 2])

    def test_ambiguous_or_missing_event_aborts(self):
        frame = pd.DataFrame({"event_id": ["a", "a"]})
        for event in ("a", "missing"):
            with self.assertRaises(RuntimeError):
                patch_event(frame, event, {"price": 3})
