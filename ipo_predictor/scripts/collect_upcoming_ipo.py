"""Collect official upcoming candidates without changing historical/model datasets."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RAW_DIR
from data.collectors.upcoming_ipo_collector import UpcomingIPOCollector


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-start", required=True)
    parser.add_argument("--submission-end", required=True)
    parser.add_argument("--output", type=Path, default=RAW_DIR / "upcoming_snapshots")
    args = parser.parse_args()
    try:
        result = UpcomingIPOCollector().collect(args.submission_start, args.submission_end, args.output)
        print(json.dumps(result, ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}), file=sys.stderr)
        sys.exit(1)
