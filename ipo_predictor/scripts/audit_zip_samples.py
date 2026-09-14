"""Check two previously reviewed IPOs through the actual DART ZIP API; no training."""
import hashlib
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
    collector = DARTCollector()
    if not collector.is_configured:
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
            document = collector.get_document_text(receipt)
            (output / f"{receipt}.xml").write_text(document, encoding="utf-8")
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
