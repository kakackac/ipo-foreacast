"""Download bounded public DART samples and retain reproducible parser evidence.

This audit does not change production data or approve model training.
"""
import hashlib
import argparse
import math
import json
import re
import sys
import time
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.collectors.dart_collector import DARTCollector

SAMPLES = {
    "sensorview_2023": "20230706000211",
    "ns_shopping_2015": "20150210000401",
    "wiseitech_2020": "20200128000055",
    "rznomics_2025": "20251205000229",
    "sensorview_prospectus_2023": "20230706000212",
    "rznomics_prospectus_2025": "20251205000237",
    "rznomics_risks_2025": "20251205000237",
    "ns_shopping_final_2015": "20150312000834",
}


def select_document(index, receipt, section=None):
    nodes = []
    for block in re.split(r"var node\d+ = \{\};", index)[1:]:
        fields = dict(re.findall(r"node\d+\['(text|rcpNo|dcmNo|eleId|offset|length|dtd)'\]\s*=\s*\"([^\"]*)\"", block))
        if fields.get("rcpNo") == receipt and "dcmNo" in fields:
            nodes.append(fields)
    body = next((n for n in nodes if "본문" in re.sub(r"\s+", "", n.get("text", ""))), None)
    offering = next((n for n in nodes if "제1부" in n.get("text", "")), None)
    selected = (next((n for n in nodes if n.get("text") == section), None)
                if section else offering or body or (nodes[0] if len(nodes) == 1 else None))
    if selected is None:
        return None
    return {k: selected[k] for k in ("rcpNo", "dcmNo", "eleId", "offset", "length", "dtd")}


def download(url, path):
    temporary = path.with_suffix(".part")
    subprocess.run([
        "curl", "--compressed", "--max-time", "180", "--fail", "--silent", "--show-error",
        "--output", str(temporary), url,
    ], check=True, timeout=185)
    temporary.replace(path)
    time.sleep(0.5)


def verify_record(record, reference):
    """Compare to reviewed values; never regenerate expectations from parser output."""
    problems = []
    for key in ("receipt", "source_url", "sha256"):
        if record.get(key) != reference.get(key):
            problems.append(f"source_mismatch:{key}")
    values = {**record.get("offering", {}), **record.get("demand", {})}
    for key, expected in reference["expected"].items():
        if key not in values:
            problems.append(f"missing_output:{key}")
            continue
        actual = values[key]
        agrees = actual is None if expected is None else (
            isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)
        )
        if not agrees:
            problems.append(f"value_mismatch:{key}")
    return problems


def main():
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--cached-only", action="store_true")
    arguments.add_argument("--sample", choices=SAMPLES, action="append")
    args = arguments.parse_args()
    output = Path(__file__).resolve().parents[1] / "data/raw/official_sample_audit"
    output.mkdir(parents=True, exist_ok=True)
    collector = DARTCollector(api_key="unused_public_audit")
    reference_path = Path(__file__).with_name("official_sample_expectations.json")
    references = {r["sample"]: r for r in json.loads(reference_path.read_text())["samples"]}
    records = []
    for name, receipt in SAMPLES.items():
        if args.sample and name not in args.sample:
            continue
        print("auditing", name, flush=True)
        main_url = "https://dart.fss.or.kr/dsaf001/main.do?" + urlencode({"rcpNo": receipt})
        main_path = output / f"{name}_index.html"
        if not main_path.exists():
            if args.cached_only:
                records.append({"sample": name, "status": "cache_missing"})
                continue
            try:
                download(main_url, main_path)
            except (subprocess.SubprocessError, OSError) as exc:
                records.append({"sample": name, "receipt": receipt,
                                "status": "source_access_failed", "error_type": type(exc).__name__})
                continue
        index = main_path.read_text(encoding="utf-8")
        section = {"rznomics_prospectus_2025": "I. 모집 또는 매출에 관한 일반사항",
                   "rznomics_risks_2025": "III. 투자위험요소"}.get(name)
        parameters = select_document(index, receipt, section)
        if not parameters:
            records.append({"sample": name, "receipt": receipt, "status": "viewer_link_not_found"})
            continue
        url = "https://dart.fss.or.kr/report/viewer.do?" + urlencode(parameters)
        body_path = output / (f"{name}_section_{parameters['eleId']}.html" if section else f"{name}_body.html")
        if not body_path.exists():
            if args.cached_only:
                records.append({"sample": name, "status": "cache_missing"})
                continue
            try:
                download(url, body_path)
            except (subprocess.SubprocessError, OSError) as exc:
                records.append({"sample": name, "receipt": receipt,
                                "status": "source_access_failed", "error_type": type(exc).__name__})
                continue
        body = body_path.read_text(encoding="utf-8")
        record = {
            "sample": name, "receipt": receipt, "source_url": url,
            "document_path": str(body_path), "selection": parameters,
            "sha256": hashlib.sha256(body_path.read_bytes()).hexdigest(),
            "audited_at": datetime.now(timezone.utc).isoformat(),
            "parser_sha256": hashlib.sha256(Path(sys.modules[DARTCollector.__module__].__file__).read_bytes()).hexdigest(),
            "status": "parsed_not_independently_approved",
            "offering": collector._parse_offering_html(body, receipt),
            "demand": collector._parse_demand_forecast_html(body, name),
        }
        reference = references.get(name)
        record["comparison_failures"] = verify_record(record, reference) if reference else ["reference_missing"]
        record["status"] = "sample_passed" if not record["comparison_failures"] else "sample_failed"
        records.append(record)
    suffix = "" if not args.sample else "_" + hashlib.sha256("|".join(sorted(args.sample)).encode()).hexdigest()[:12]
    (output / f"results{suffix}.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "samples_requested": len(records),
        "samples_passed": sum(r["status"] == "sample_passed" for r in records),
        "all_reference_samples_checked": {r["sample"] for r in records} == set(references),
        "checked_fields": sum(len(references[r["sample"]]["expected"]) for r in records if r["status"] == "sample_passed"),
        "full_collection_authorized": False,
        "remaining_gate": "Authenticated ZIP transport, event linkage and time/source contracts not validated by this public HTML audit.",
    }
    (output / f"summary{suffix}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for record in records:
        print(record["sample"], {k: record.get("demand", {}).get(k) for k in ["institutional_demand_ratio", "lockup_commitment_ratio"]})
    print(json.dumps(summary, ensure_ascii=False))
    return 1 if any(r["status"] != "sample_passed" for r in records) else 0


if __name__ == "__main__":
    sys.exit(main())
