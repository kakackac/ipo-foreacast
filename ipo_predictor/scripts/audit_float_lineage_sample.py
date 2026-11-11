"""Replay the Rznomics official-source correction through selection and feature building."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from config import RAW_DIR, PROC_DIR
from data.collectors.dart_collector import DARTCollector
from data.pipelines.float_source_resolver import resolve_public_float
from data.processors.feature_engineer import FeatureEngineer


def main():
    references = json.loads(Path(__file__).with_name("official_sample_expectations.json").read_text())
    references = {row["sample"]: row for row in references["samples"]}
    directory = RAW_DIR / "official_sample_audit"
    documents = {}
    for name, filename in (
        ("rznomics_2025", "rznomics_2025_body.html"),
        ("rznomics_prospectus_2025", "rznomics_prospectus_2025_section_8.html"),
        ("rznomics_risks_2025", "rznomics_risks_2025_section_15.html"),
    ):
        content = (directory / filename).read_bytes()
        if hashlib.sha256(content).hexdigest() != references[name]["sha256"]:
            raise RuntimeError(f"Official reference hash mismatch: {name}")
        documents[name] = content.decode("utf-8")
    collector = DARTCollector(api_key="", document_cache_dir=None)
    final = collector._parse_offering_html(documents["rznomics_2025"], "20251205000229")
    prospectus = collector._parse_offering_html(
        documents["rznomics_prospectus_2025"] + documents["rznomics_risks_2025"], "20251205000237")
    # Both downloaded sections belong to the same receipt, never separate filings.
    metadata = {"rcept_no": "20251205000229", "rcept_dt": "2025-12-05"}
    calendar = pd.read_parquet(RAW_DIR / "krx_official_event_master.parquet")
    event = calendar[calendar["corp_name"].eq("알지노믹스")]
    if len(event) != 1:
        raise RuntimeError("Official event must be uniquely identified")
    listing = event.iloc[0]["listing_date"]
    conflict = resolve_public_float([(metadata, final)], 22500, listing)
    selected = resolve_public_float([(metadata, final), (
        dict(metadata, rcept_no="20251205000237", supplementary=True), prospectus)], 22500, listing)
    if conflict["public_float_ratio_disclosed"] is not None:
        raise RuntimeError("Conflicting filing was not blocked")
    feature = FeatureEngineer("phase2")._calc_supply_structure_features(pd.DataFrame([selected]))
    if abs(float(feature.iloc[0]["float_share_ratio"]) - .3527) > 1e-9:
        raise RuntimeError("Selected official value did not reach feature output")
    report = {"event_id": str(event.iloc[0]["event_id"]),
              "conflicting_filing_status": conflict["public_float_resolution_status"],
              "selected": selected, "float_share_ratio": float(feature.iloc[0]["float_share_ratio"]),
              "production_dataset_updated": False, "model_training_authorized": False}
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    (PROC_DIR / "float_lineage_sample_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(json.dumps({key: value for key, value in report.items() if key != "selected"}, ensure_ascii=False))
    print("source_receipt", selected["public_float_rcept_no"])


if __name__ == "__main__":
    main()
