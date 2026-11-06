"""Build an offline evidence worklist; never infer issuer-wide absence or approve values."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from config import RAW_DIR, PROC_DIR
from data.processors.feature_engineer import FeatureEngineer
from data.pipelines.data_delivery import current_financial_repair, checked_artifact


def main():
    repair = current_financial_repair(PROC_DIR)
    manifest = checked_artifact(repair)
    features = pd.read_parquet(repair / "features_all.parquet")
    observations = pd.read_parquet(repair / "feature_observations.parquet")
    missing = observations.loc[
        observations.feature_name.isin(["institutional_demand_ratio", "lockup_commitment_ratio", "float_share_ratio"])
        & observations.is_missing
    ].copy()
    lineage = pd.read_parquet(repair / 'disclosure_lineage.parquet' if 'disclosure_lineage.parquet' in manifest['output_sha256']
                              else RAW_DIR / "dart_disclosure_lineage.parquet")
    rows = []
    for event_id, group in lineage.groupby("event_id"):
        status = group.get('attempt_status', pd.Series('', index=group.index)).fillna("")
        rows.append({
            "event_id": event_id,
            "candidate_receipts": group.rcept_no.nunique(),
            "parsed_or_cached_receipts": group.loc[status.eq("document_parsed") | status.str.startswith("cached"), "rcept_no"].nunique(),
            "deferred_receipts": group.loc[status.eq("retry_deferred"), "rcept_no"].nunique(),
        })
    counts = pd.DataFrame(rows, columns=["event_id", "candidate_receipts", "parsed_or_cached_receipts", "deferred_receipts"])
    missing = missing.merge(counts, on="event_id", how="left", validate="many_to_one")
    missing["diagnosis_scope"] = "stored_parser_and_lineage_evidence_only_not_official_nonpublication"
    missing['data_version'] = repair.name
    missing.to_parquet(PROC_DIR / "missing_feature_worklist.parquet", index=False)
    report = missing.groupby(["feature_name", "missing_reason"], dropna=False).size().rename("rows").reset_index()
    (PROC_DIR / "missing_feature_worklist_summary.json").write_text(report.to_json(orient="records", force_ascii=False, indent=2))
    checked_artifact(repair)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
