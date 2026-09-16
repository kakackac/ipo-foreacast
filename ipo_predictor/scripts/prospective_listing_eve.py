"""Research-only prospective prediction ledger. No service model promotion."""
import argparse
import hashlib
import json
import math
import re
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from scripts.prepare_listing_eve_experiment import SPEC
from scripts.finalize_listing_eve_experiment import digest
from scripts.run_listing_eve_development import prepare_inputs

KST = ZoneInfo("Asia/Seoul")


def timestamp(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("Timezone required")
    return result.astimezone(KST)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def connect(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("CREATE TABLE IF NOT EXISTS predictions (event_id TEXT PRIMARY KEY, payload TEXT NOT NULL, sha256 TEXT NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS outcomes (event_id TEXT PRIMARY KEY REFERENCES predictions(event_id), payload TEXT NOT NULL, sha256 TEXT NOT NULL)")
    for table in ("predictions", "outcomes"):
        for action in ("UPDATE", "DELETE"):
            db.execute(f"CREATE TRIGGER IF NOT EXISTS no_{action.lower()}_{table} BEFORE {action} ON {table} BEGIN SELECT RAISE(ABORT, 'Append-only research ledger'); END")
    db.commit()
    return db


def append(db, table, event_id, payload):
    serialized = canonical(payload)
    checksum = hashlib.sha256(serialized.encode()).hexdigest()
    with db:
        db.execute(f"INSERT INTO {table} VALUES (?, ?, ?)", (event_id, serialized, checksum))
    return checksum


def read_prediction(db, event_id):
    row = db.execute("SELECT payload, sha256 FROM predictions WHERE event_id=?", (event_id,)).fetchone()
    if row is None or hashlib.sha256(row[0].encode()).hexdigest() != row[1]:
        raise ValueError("Missing or altered prediction")
    return json.loads(row[0])


def evidence(item, now):
    # Store document identifiers, not arbitrary URLs or headers containing credentials.
    reference = item.get("source_reference", "")
    if not isinstance(reference, str) or not re.fullmatch(r"[A-Za-z0-9_./:-]{1,160}", reference):
        raise ValueError("Use a public document/request identifier without credentials")
    if item.get("source") not in ("DART", "KRX", "OFFICIAL_UNDERWRITER", "DERIVED_OFFICIAL"):
        raise ValueError("Unsupported source")
    if item.get("verification_status") != "verified" or item.get("human_review_required") is not False:
        raise ValueError("Unverified evidence")
    if not timestamp(item["available_at"]) <= timestamp(item["collected_at"]) <= now:
        raise ValueError("Evidence was unavailable at prediction time")
    return {key: item[key] for key in ("source", "source_reference", "available_at", "collected_at", "verification_status", "human_review_required")}


def validate_input(item, fields, now):
    listing = datetime.strptime(item["listing_date"], "%Y-%m-%d").replace(tzinfo=KST)
    if now.date() != (listing - timedelta(days=1)).date() or now.hour < 21:
        raise ValueError("Predictions require listing eve, 21:00-23:59 KST; no backdating")
    if item.get("offering_type") != "general_ipo" or item.get("market") not in ("KOSPI", "KOSDAQ"):
        raise ValueError("Research cohort only supports general KOSPI/KOSDAQ IPOs")
    if not isinstance(item.get("event_id"), str) or not re.fullmatch(r"[A-Za-z0-9_:-]{1,120}", item["event_id"]):
        raise ValueError("Event identifier required")
    if set(item["features"]) != set(fields):
        raise ValueError("Feature contract mismatch")
    approved = {}
    for field in fields:
        value = item["features"][field]
        proof = item["evidence"][field]
        if value is None:
            if proof.get("missing_reason") not in ("not_yet_published", "source_missing", "parser_failed", "source_access_failed", "needs_review"):
                raise ValueError("Missing reason required")
            approved[field] = {"missing_reason": proof["missing_reason"]}
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Invalid numeric feature")
            if proof.get("raw_value") != value:
                raise ValueError("Feature/evidence mismatch")
            approved[field] = evidence(proof, now)
    for key in ("institutional_demand_ratio", "lockup_commitment_ratio"):
        if item["features"].get(key) is None:
            raise ValueError("Required institutional feature missing")
    if item["features"]["institutional_demand_ratio"] < 0 or not 0 <= item["features"]["lockup_commitment_ratio"] <= 1:
        raise ValueError("Invalid institutional units; commitment must be a 0-1 fraction")
    for flag in ("spac_flag", "offering_type_spac_ipo"):
        if flag in fields and item["features"][flag] != 0:
            raise ValueError("SPAC feature conflicts with general IPO cohort")
    price = item["offering_price"]
    if isinstance(price, bool) or not isinstance(price, (int, float)) or not math.isfinite(price) or price <= 0:
        raise ValueError("Invalid offering price")
    if item["offering_price_evidence"].get("raw_value") != price:
        raise ValueError("Offering price/evidence mismatch")
    approved_price = evidence(item["offering_price_evidence"], now)
    return approved, approved_price


def predict(experiment, item, db_path):
    now = datetime.now(KST)
    root = Path(experiment)
    manifest = json.loads((root / "manifest.json").read_text())
    policy = json.loads((root / "final_policy.json").read_text())
    if digest(root / "manifest.json") != policy["manifest_sha256"] or any(manifest.get(k) != v for k, v in SPEC.items()):
        raise ValueError("Frozen experiment mismatch")
    if policy.get("selected_estimators") != {"open_return_pct": "gradient_boosting", "close_return_pct": "training_target_median"}:
        raise ValueError("Unsupported frozen selection")
    cohort = manifest["cohorts"]["post_demand"]
    fields = cohort["features"]
    approved, approved_price = validate_input(item, fields, now)
    development_path = root / "post_demand/development.parquet"
    if digest(development_path) != cohort["development_sha256"]:
        raise ValueError("Training snapshot changed")
    train = pd.read_parquet(development_path)
    if item["event_id"] in set(train.event_id) or (pd.to_datetime(train.listing_date) >= pd.Timestamp(SPEC["holdout_start"])).any():
        raise ValueError("Training overlap or unexpected training period")
    X, future = prepare_inputs(train, pd.DataFrame([item["features"]]), fields)
    outputs = {}
    for target in SPEC["targets"]:
        y = pd.to_numeric(train[target], errors="raise")
        if not y.map(math.isfinite).all():
            raise ValueError("Invalid training target")
        outputs[target] = {"mean": float(y.mean()), "median": float(y.median())}
        if target == "open_return_pct":
            model = GradientBoostingRegressor(**{k: v for k, v in SPEC["model"].items() if k != "class"})
            model.fit(X, y)
            outputs[target]["model"] = float(model.predict(future)[0])
    payload = {"version": 1, "recorded_at": datetime.now(KST).isoformat(), "input_cutoff": now.isoformat(),
        "event_id": item["event_id"], "listing_date": item["listing_date"], "offering_price": item["offering_price"],
        "features": item["features"], "evidence": approved, "offering_price_evidence": approved_price,
        "predictions": outputs, "experiment_id": manifest["run_id"], "training_sha256": cohort["development_sha256"],
        "model_spec": SPEC["model"], "deployment_authorized": False}
    if timestamp(payload["recorded_at"]).date() != now.date():
        raise ValueError("Prediction crossed listing-day boundary")
    db = connect(db_path)
    try:
        return {"event_id": item["event_id"], "sha256": append(db, "predictions", item["event_id"], payload), "deployment_authorized": False}
    finally:
        db.close()


def resolve(db_path, item):
    now = datetime.now(KST)
    db = connect(db_path)
    try:
        prediction = read_prediction(db, item["event_id"])
        close_time = datetime.strptime(prediction["listing_date"], "%Y-%m-%d").replace(hour=21, tzinfo=KST)
        if now < close_time or item["price_date"] != prediction["listing_date"]:
            raise ValueError("Wait until listing day 21:00 KST and use that day's prices")
        if item.get("session") != "KRX_REGULAR" or item["evidence"].get("source") != "KRX":
            raise ValueError("Official KRX regular-session prices required")
        if item["evidence"].get("event_id") != item["event_id"] or item["evidence"].get("price_date") != item["price_date"]:
            raise ValueError("Price evidence event/date mismatch")
        proof = evidence(item["evidence"], now)
        if timestamp(proof["available_at"]) < close_time.replace(hour=15, minute=30):
            raise ValueError("Final regular-session prices were not yet available")
        actual, errors = {}, {}
        for target, field in (("open_return_pct", "open_price"), ("close_return_pct", "close_price")):
            price = item[field]
            if isinstance(price, bool) or not isinstance(price, (int, float)) or not math.isfinite(price) or price <= 0:
                raise ValueError("Invalid official price")
            if item["evidence"].get(field) != price:
                raise ValueError("Price differs from verified observation")
            actual[target] = (price / prediction["offering_price"] - 1) * 100
            errors[target] = {name: abs(value - actual[target]) for name, value in prediction["predictions"][target].items()}
        payload = {"recorded_at": now.isoformat(), "actual": actual, "absolute_error_pp": errors,
            "price_date": item["price_date"], "open_price": item["open_price"], "close_price": item["close_price"],
            "evidence": proof, "deployment_authorized": False}
        return {"event_id": item["event_id"], "sha256": append(db, "outcomes", item["event_id"], payload)}
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["predict", "resolve", "status"])
    parser.add_argument("--database", required=True)
    parser.add_argument("--experiment")
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    if args.mode == "status":
        db = connect(args.database)
        try:
            print(canonical({table: db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in ("predictions", "outcomes")}))
        finally:
            db.close()
    else:
        if args.input is None or (args.mode == "predict" and not args.experiment):
            parser.error("--input required; predict also requires --experiment")
        item = json.loads(args.input.read_text())
        print(canonical(predict(args.experiment, item, args.database) if args.mode == "predict" else resolve(args.database, item)))
