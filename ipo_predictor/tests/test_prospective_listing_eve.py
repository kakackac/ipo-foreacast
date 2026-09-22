import copy
import json
import sqlite3
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
import pandas as pd

from scripts import prospective_listing_eve as p


NOW = datetime(2026, 9, 16, 22, tzinfo=p.KST)
FIELDS = ["institutional_demand_ratio", "lockup_commitment_ratio", "spac_flag"]


def proof(value=0):
    return {"source": "DART", "source_reference": "20260916000001", "verification_status": "verified",
        "human_review_required": False, "available_at": "2026-09-16T15:00:00+09:00",
        "collected_at": "2026-09-16T16:00:00+09:00", "raw_value": value}


def candidate():
    values = dict(zip(FIELDS, [100, .17, 0]))
    return {"event_id": "test_20260917", "ticker": "123450", "listing_date": "2026-09-17", "offering_type": "general_ipo",
        "market": "KOSDAQ", "features": values, "evidence": {k: proof(v) for k, v in values.items()},
        "offering_price": 10000, "offering_price_evidence": proof(10000)}


class ProspectiveTests(unittest.TestCase):
    def test_korean_kind_event_identity_accepted(self):
        item = candidate()
        item["event_id"] = "krx_kind|123450|20260917|테스트"
        p.validate_input(item, FIELDS, NOW)

    def test_prediction_end_to_end_without_reading_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "post_demand").mkdir()
            train = pd.DataFrame({"event_id": [f"old_{i}" for i in range(100)],
                "listing_date": ["2025-01-01"] * 100, "institutional_demand_ratio": range(100),
                "lockup_commitment_ratio": [.1] * 100, "spac_flag": [0] * 100,
                "open_return_pct": range(100), "close_return_pct": range(100)})
            path = root / "post_demand/development.parquet"
            train.to_parquet(path)
            manifest = dict(p.SPEC, run_id="test-only", cohorts={"post_demand": {
                "features": FIELDS, "development_sha256": p.digest(path)}})
            (root / "manifest.json").write_text(json.dumps(manifest))
            (root / "final_policy.json").write_text(json.dumps({"manifest_sha256": p.digest(root / "manifest.json"),
                "selected_estimators": {"open_return_pct": "gradient_boosting", "close_return_pct": "training_target_median"}}))
            with patch.object(p, "datetime", wraps=datetime) as clock:
                clock.now.return_value = NOW
                result = p.predict(root, candidate(), root / "ledger.sqlite")
                self.assertFalse(result["deployment_authorized"])
                with self.assertRaises(sqlite3.IntegrityError):
                    p.predict(root, candidate(), root / "ledger.sqlite")
            db = p.connect(root / "ledger.sqlite")
            record = p.read_prediction(db, candidate()["event_id"])
            self.assertIn("model", record["predictions"]["open_return_pct"])
            self.assertNotIn("model", record["predictions"]["close_return_pct"])
            db.close()

    def test_validated_input_and_unit_rejection(self):
        p.validate_input(candidate(), FIELDS, NOW)
        for field, value in (("lockup_commitment_ratio", 17), ("spac_flag", 1), ("institutional_demand_ratio", -1)):
            item = candidate()
            item["features"][field] = value
            item["evidence"][field]["raw_value"] = value
            with self.assertRaises(ValueError):
                p.validate_input(item, FIELDS, NOW)

    def test_no_backdated_or_early_predictions(self):
        for now in (NOW.replace(day=17), NOW.replace(hour=20), NOW.replace(day=15)):
            with self.assertRaises(ValueError):
                p.validate_input(candidate(), FIELDS, now)

    def test_unverified_late_or_mismatched_evidence(self):
        for key, value in (("verification_status", "needs_review"), ("available_at", "2026-09-17T00:00:00+09:00"),
                           ("source_reference", "url?api_key=secret"), ("raw_value", 999)):
            item = candidate()
            item["evidence"][FIELDS[0]][key] = value
            with self.assertRaises(ValueError):
                p.validate_input(item, FIELDS, NOW)

    def test_append_only_and_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            db = p.connect(Path(directory) / "ledger.sqlite")
            try:
                p.append(db, "predictions", "test", {"value": 1})
                self.assertEqual(p.read_prediction(db, "test"), {"value": 1})
                with self.assertRaises(sqlite3.IntegrityError):
                    p.append(db, "predictions", "test", {"value": 2})
                for command in ("UPDATE predictions SET payload='{}'", "DELETE FROM predictions"):
                    with self.assertRaises(sqlite3.IntegrityError):
                        db.execute(command)
                with self.assertRaises(ValueError):
                    p.read_prediction(db, "missing")
            finally:
                db.close()

    def test_outcome_matching_timing_and_repeat_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.sqlite"
            db = p.connect(path)
            prediction = {"listing_date": "2026-09-16", "offering_price": 10000,
                "predictions": {"open_return_pct": {"model": 15}, "close_return_pct": {"median": 5}}}
            p.append(db, "predictions", "test", prediction)
            db.close()
            source = dict(proof(), source="KRX", event_id="test", price_date="2026-09-16", open_price=12000,
                close_price=11000, available_at="2026-09-16T16:00:00+09:00", collected_at="2026-09-16T21:00:00+09:00")
            item = {"event_id": "test", "price_date": "2026-09-16", "session": "KRX_REGULAR",
                "open_price": 12000, "close_price": 11000, "evidence": source}
            with patch.object(p, "datetime", wraps=datetime) as clock:
                clock.now.return_value = NOW.replace(hour=20)
                with self.assertRaises(ValueError):
                    p.resolve(path, item)
                clock.now.return_value = NOW
                bad = copy.deepcopy(item)
                bad["evidence"]["event_id"] = "other"
                with self.assertRaises(ValueError):
                    p.resolve(path, bad)
                p.resolve(path, item)
                with self.assertRaises(sqlite3.IntegrityError):
                    p.resolve(path, item)
            db = p.connect(path)
            result = json.loads(db.execute("SELECT payload FROM outcomes").fetchone()[0])
            self.assertAlmostEqual(result["actual"]["open_return_pct"], 20)
            self.assertAlmostEqual(result["absolute_error_pp"]["open_return_pct"]["model"], 5)
            db.close()
