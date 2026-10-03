"""Check two previously reviewed IPOs through the actual DART ZIP API; no training."""
import hashlib
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RAW_DIR
from data.collectors.dart_collector import DARTCollector, DEMAND_PARSER_VERSION

EXPECTED = {
    "20220808000054": {"demand_offering_price": 9000, "institutional_demand_ratio": 1934.89,
                       "lockup_commitment_ratio": .0457},
    "20210111000425": {"demand_offering_price": 19000, "institutional_demand_ratio": 1425.3,
                       "lockup_commitment_ratio": .107},
}


def compare_values(actual, expected):
    return [key for key, value in expected.items()
            if not isinstance(actual.get(key), (float, int))
            or not math.isclose(actual[key], value, rel_tol=1e-9, abs_tol=1e-9)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cached-only", action="store_true")
    args = parser.parse_args()
    collector = DARTCollector()
    if not args.cached_only and not collector.is_configured:
        print("DART_API_KEY is not configured. No requests sent.")
        return 2
    output = RAW_DIR / "official_zip_sample_audit"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for receipt, expected in EXPECTED.items():
        record = {"receipt": receipt, "parser_version": DEMAND_PARSER_VERSION,
                  "source_url": "https://opendart.fss.or.kr/api/document.xml",
                  "expected": expected}
        try:
            source_path = output / f"{receipt}.xml"
            if args.cached_only:
                document = source_path.read_text(encoding="utf-8")
            else:
                document = collector.get_document_text(receipt)
                source_path.write_text(document, encoding="utf-8")
            record["cache_used"] = args.cached_only
            actual = collector._parse_demand_forecast_html(document, "sample_audit")
            failed = compare_values(actual, expected)
            record.update(status="mismatch" if failed else "sample_passed",
                          failed_fields=failed, actual=actual,
                          decoded_document_sha256=hashlib.sha256(document.encode()).hexdigest())
        except Exception as exc:
            # Request exception strings may contain a URL with the API key.
            record.update(status="source_access_failed", error_type=type(exc).__name__)
        records.append(record)
        print(receipt, record["status"])
    report = {"samples": records, "model_training_authorized": False,
              "full_collection_authorized": False,
              "remaining_checks": "IPO linkage, publication cutoff and untested document formats"}
    (output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if all(r["status"] == "sample_passed" for r in records) else 1


if __name__ == "__main__":
    sys.exit(main())
