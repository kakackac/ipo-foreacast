import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from data.collectors.upcoming_ipo_collector import HEADERS, parse_candidates
from scripts.prospective_listing_eve import connect, append
from scripts.resolve_prospective_prices import build_outcome, run


def documents():
    cells = ["테스트", "2026-06-01", "2026-09-01 ~ 2026-09-02", "2026-09-03 ~ 2026-09-04",
             "2026-09-07", "10,000", "20,000", "2026-09-23", "공식증권"]
    export = "<table><tr>" + "".join(f"<th>{c}</th>" for c in HEADERS) + "</tr><tr>"
    export += "".join(f"<td>{c}</td>" for c in ["코스닥 상장추진", *cells]) + "</tr></table>"
    listing = "<table><tr onclick=\"fnDetailView('20260101000001')\">"
    listing += "".join(f"<td>{c}</td>" for c in cells) + "</tr></table>전체 1 건"
    return export, listing


class UpcomingTests(unittest.TestCase):
    def test_source_identity_and_pending_approval(self):
        rows = parse_candidates(*documents(), "KOSDAQ", "2026-09-22T10:00:00+09:00")
        self.assertEqual(rows[0]["kind_process_id"], "20260101000001")
        self.assertEqual(rows[0]["offering_price_candidate"], 10000)
        self.assertFalse(rows[0]["model_eligible"])

    def test_schema_pagination_and_values_fail_closed(self):
        export, listing = documents()
        for a, b in ((export.replace("확정공모가", "예정공모가"), listing),
                     (export, listing.replace("전체 1 건", "전체 101 건")),
                     (export.replace("10,000", "11,000"), listing)):
            with self.assertRaises(ValueError):
                parse_candidates(a, b, "KOSDAQ", "2026-09-22T10:00:00+09:00")

class OutcomeBatchTests(unittest.TestCase):
    def test_exact_identity_required(self):
        prediction = {"event_id": "test", "ticker": "123450", "market": "KOSDAQ", "listing_date": "2025-01-01"}
        price = {"price_match_status": "matched", "market": "KOSDAQ", "price_matched_ticker": "123450",
            "listing_date": "20250101", "price_raw_response_evidence": "{}", "open_price": 100, "close_price": 110}
        self.assertEqual(build_outcome(prediction, price, "2026-09-22T22:00:00+09:00")["open_price"], 100)
        for field, value in (("market", "KOSPI"), ("price_matched_ticker", "999999"), ("listing_date", "20250102")):
            with self.assertRaises(ValueError):
                build_outcome(prediction, dict(price, **{field: value}), "2026-09-22T22:00:00+09:00")

    def test_batch_resolves_once_and_skips_resolved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db_path = root / "ledger.sqlite"
            db = connect(db_path)
            append(db, "predictions", "test", {"event_id": "test", "ticker": "123450", "market": "KOSDAQ",
                "listing_date": "2025-01-01", "offering_price": 100,
                "predictions": {"open_return_pct": {"model": 5}, "close_return_pct": {"median": 5}}})
            db.close()
            collector = Mock(is_configured=True)
            collector.get_listing_day_price.return_value = {"price_match_status": "matched", "market": "KOSDAQ",
                "price_matched_ticker": "123450", "listing_date": "20250101", "price_raw_response_evidence": "{}",
                "open_price": 100, "close_price": 110}
            result = run(db_path, root / "runs", collector)
            self.assertEqual(result["events"][0]["status"], "resolved")
            self.assertEqual(run(db_path, root / "runs", collector)["events"], [])
            self.assertEqual(collector.get_listing_day_price.call_count, 1)

    def test_failed_event_does_not_block_other_events_or_leak_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = connect(root / "ledger.sqlite")
            for event in ("a", "b"):
                append(db, "predictions", event, {"event_id": event, "ticker": "123450", "market": "KOSDAQ", "listing_date": "2025-01-01"})
            db.close()
            collector = Mock(is_configured=True)
            collector.get_listing_day_price.side_effect = RuntimeError("sensitive-do-not-log")
            report = run(root / "ledger.sqlite", root / "runs", collector)
            self.assertEqual(collector.get_listing_day_price.call_count, 2)
            self.assertEqual(report["status"], "attention_required")
            self.assertNotIn("sensitive-do-not-log", json.dumps(report))
