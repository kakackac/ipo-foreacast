"""Replay saved official HTML and compare screenshot names without inventing dates."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.collectors.upcoming_ipo_collector import parse_candidates


def audit(snapshots, reference):
    events = {}
    windows = []
    for folder in snapshots:
        folder = Path(folder)
        manifest = json.loads((folder / "manifest.json").read_text())
        if manifest["status"] != "complete":
            raise ValueError("Failed snapshot cannot establish coverage")
        windows.append({"start": manifest["submission_start"], "end": manifest["submission_end"],
                        "collected_at": manifest["finished_at"]})
        for market in ("KOSPI", "KOSDAQ"):
            contents = {}
            for form in ("sub", "down"):
                contents[form] = (folder / f"{market}_{form}.html").read_bytes()
                request = next(r for r in manifest["requests"] if r["market"] == market and r["form"] == form)
                if hashlib.sha256(contents[form]).hexdigest() != request["sha256"]:
                    raise ValueError("Official snapshot hash mismatch")
            for row in parse_candidates(contents["down"], contents["sub"], market, manifest["finished_at"]):
                old = events.get(row["candidate_id"])
                if old is None or old["collected_at"] < row["collected_at"]:
                    events[row["candidate_id"]] = row
    names = {}
    for row in events.values():
        names.setdefault(row["corp_name"].replace(" ", ""), []).append(row)
    comparisons = []
    for item in reference["companies"]:
        matches = names.get(item["name"].replace(" ", ""), [])
        comparisons.append({"name": item["name"], "screenshot_d_day": item["d_day"],
            "status": "name_candidate_requires_identity_check" if matches else "not_found_in_queried_kind_snapshots",
            "candidate_ids": [r["candidate_id"] for r in matches], "cause": "unresolved_not_proof_of_source_absence" if not matches else None})
    return {"windows": windows, "official_candidate_count": len(events), "reference_count": len(comparisons),
        "name_candidate_count": sum(bool(r["candidate_ids"]) for r in comparisons), "comparisons": comparisons,
        "candidates": list(events.values()), "capture_date_verified": False,
        "model_eligible": False, "warning": "Name comparisons are not identity verification; no D-day dates inferred."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", nargs="+", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path(__file__).resolve().parents[1] / "data/manual/screenshot_schedule_reference.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.snapshots, json.loads(args.reference.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: report[key] for key in ("official_candidate_count", "reference_count", "name_candidate_count")}, ensure_ascii=False))
