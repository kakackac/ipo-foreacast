"""One retryable research outcome batch using official KRX regular-session data."""
import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.collectors.krx_collector import KRXCollector
from scripts.prospective_listing_eve import KST, connect, read_prediction, resolve


def build_outcome(prediction, price, collected_at):
    ticker = prediction.get("ticker", "")
    market = prediction.get("market")
    if not re.fullmatch(r"[0-9A-Z]{6}", ticker) or market not in ("KOSPI", "KOSDAQ"):
        raise ValueError("Legacy prediction lacks verified trading identity")
    if (price.get("price_match_status") != "matched" or price.get("market") != market
            or str(price.get("price_matched_ticker", "")).zfill(6) != ticker
            or not price.get("price_raw_response_evidence")
            or str(price.get("listing_date")) != prediction["listing_date"].replace("-", "")):
        raise ValueError("Exact official price identity not verified")
    proof = {"source": "KRX", "source_reference": f"daily:{market}:{ticker}:{price['listing_date']}",
        "verification_status": "verified", "human_review_required": False,
        "available_at": collected_at, "collected_at": collected_at,
        "event_id": prediction["event_id"], "price_date": prediction["listing_date"],
        "open_price": price.get("open_price"), "close_price": price.get("close_price")}
    return {"event_id": prediction["event_id"], "price_date": prediction["listing_date"],
        "session": "KRX_REGULAR", "open_price": price.get("open_price"),
        "close_price": price.get("close_price"), "evidence": proof}


def run(database, output, collector=None):
    now = datetime.now(KST)
    destination = Path(output) / now.strftime("%Y%m%dT%H%M%S%f")
    destination.mkdir(parents=True, exist_ok=False)
    collector = collector or KRXCollector()
    db = connect(database)
    report = {"started_at": now.isoformat(), "deployment_authorized": False, "events": []}
    try:
        pending = db.execute("SELECT event_id FROM predictions WHERE event_id NOT IN (SELECT event_id FROM outcomes) ORDER BY event_id").fetchall()
        for (event_id,) in pending:
            entry = {"event_id": event_id}
            report["events"].append(entry)
            try:
                prediction = read_prediction(db, event_id)
                due = datetime.strptime(prediction["listing_date"], "%Y-%m-%d").replace(hour=21, tzinfo=KST)
                if now < due:
                    entry["status"] = "not_due"
                    continue
                if not prediction.get("ticker") or not prediction.get("market"):
                    entry["status"] = "legacy_identity_review_required"
                    continue
                if not collector.is_configured:
                    entry["status"] = "credentials_missing"
                    continue
                price = collector.get_listing_day_price(prediction["ticker"], prediction["listing_date"], market=prediction["market"])
                item = build_outcome(prediction, price, datetime.now(KST).isoformat())
                # Preserve the matched raw-response evidence before accepting the outcome.
                evidence_path = destination / f"price_{len(report['events'])}.json"
                evidence_path.write_text(json.dumps(price, ensure_ascii=False, default=str), encoding="utf-8")
                resolve(database, item)
                entry.update(status="resolved", evidence_file=evidence_path.name)
            except Exception as exc:
                entry.update(status="retry_or_review_required", error_type=type(exc).__name__)
        report["status"] = "attention_required" if any(e["status"] not in ("resolved", "not_due") for e in report["events"]) else "complete"
    finally:
        db.close()
        report["finished_at"] = datetime.now(KST).isoformat()
        (destination / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    logging.getLogger("data.collectors.krx_collector").disabled = True
    report = run(args.database, args.output)
    print(json.dumps(report, ensure_ascii=False))
    sys.exit(0 if report["status"] == "complete" else 1)
