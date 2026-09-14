"""Apply two reviewed field groups and rebuild dependent audits, with rollback copies."""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from config import RAW_DIR, PROC_DIR
from data.pipelines.historical_ipo_pipeline import HistoricalIPOPipeline
from data.processors.feature_engineer import FeatureEngineer


def patch_event(frame, event_id, changes):
    indices = frame.index[frame["event_id"].eq(event_id)]
    if len(indices) != 1:
        raise RuntimeError(f"Event is not unique: {event_id}")
    result = frame.copy()
    for field, value in changes.items():
        if field not in result:
            result[field] = None
        result.at[indices[0], field] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    # Regenerate evidence before reading repair values; neither audit uses the network.
    from audit_ns_price_lineage import main as audit_ns
    from audit_float_lineage_sample import main as audit_float
    audit_ns()
    audit_float()
    source = RAW_DIR / "dart_ipo_raw.parquet"
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    original = pd.read_parquet(source)
    ns = pd.read_parquet(PROC_DIR / "ns_price_lineage_sample.parquet").iloc[0]
    ns_fields = {key: value for key, value in ns.items() if key.startswith("offering_price")}
    ns_fields.update(offering_price_rcept_no=ns["rcept_no"], offering_price_rcept_dt=ns["rcept_dt"])
    if original.loc[original.event_id.eq(ns["event_id"]), "offering_price_review_status"].eq("manual_verified").any():
        raise RuntimeError("Manual price approval must not be overwritten")
    changed = patch_event(original, ns["event_id"], ns_fields)
    report = json.loads((PROC_DIR / "float_lineage_sample_audit.json").read_text())
    float_changes = report["selected"]
    float_changes["public_float_rcept_dt"] = pd.Timestamp(float_changes["public_float_rcept_dt"])
    changed = patch_event(changed, report["event_id"], float_changes)
    events = pd.read_parquet(RAW_DIR / "krx_official_event_master.parquet")
    kospi = pd.read_parquet(RAW_DIR / "kospi_index.parquet")
    kosdaq = pd.read_parquet(RAW_DIR / "kosdaq_index.parquet")
    engineer = FeatureEngineer("phase2")
    features = engineer.build_features(changed, events, kospi, kosdaq)
    before_features = pd.read_parquet(PROC_DIR / "features_all.parquet")
    if set(features.event_id) != set(before_features.event_id) or len(features) != len(before_features):
        raise RuntimeError("Rebuild changed the event population")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    with tempfile.TemporaryDirectory(dir=PROC_DIR) as directory:
        root = Path(directory)
        pipeline = HistoricalIPOPipeline(raw_dir=root / "raw", processed_dir=root / "processed")
        live = HistoricalIPOPipeline()
        changed.to_parquet(root / "raw/dart_ipo_raw.parquet", index=False)
        features.to_parquet(root / "processed/features_all.parquet", index=False)
        observations = engineer.build_feature_observations(features)
        observations = live._apply_official_source_resolutions(observations, live._load_official_source_resolutions())
        observations.to_parquet(root / "processed/feature_observations.parquet", index=False)
        coverage = pipeline._build_feature_coverage_audit(observations)
        coverage.to_parquet(root / "processed/feature_coverage_audit.parquet", index=False)
        time_audit = pipeline._build_feature_time_audit(features, observations)
        if time_audit.is_future_information.any():
            raise RuntimeError("Time audit found future information; apply blocked")
        time_audit.to_parquet(root / "processed/feature_time_validation.parquet", index=False)
        stages = pipeline._write_stage_datasets(features, time_audit)
        prices = pipeline._build_offering_price_audit(changed)
        prices.to_parquet(root / "raw/dart_offering_price_audit.parquet", index=False)
        approved = {"verified_currency_unit", "verified_text_and_structured", "verified_structured_api", "manual_verified"}
        prices[~prices.offering_price_review_status.isin(approved)].to_parquet(
            root / "raw/dart_offering_price_review_queue.parquet", index=False)
        summary = json.loads((PROC_DIR / "data_collection_summary.json").read_text())
        summary.update(pipeline._build_summary(events, changed,
            pd.read_parquet(RAW_DIR / "ipo_listing_prices.parquet"), features))
        summary.update(model_stage_readiness=stages,
            offline_sample_repair_id=stamp, full_collection_rerun=False)
        summary["feature_coverage"] = {str(row.feature_name): {
            "observed_rows": int(row.observed_rows), "coverage_rate": float(row.coverage_rate),
            "human_review_required_rows": int(row.human_review_required_rows)}
            for row in coverage.itertuples(index=False)}
        summary.update(future_information_violations=int(time_audit.is_future_information.sum()),
                       feature_time_validation_rows=len(time_audit))
        (root / "processed/data_collection_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        manifest = {"repair_id": stamp, "events": [ns["event_id"], report["event_id"]],
            "source_before_sha256": original_hash, "feature_rows": len(features),
            "open_targets_before": int(before_features.open_return_pct.notna().sum()),
            "open_targets_after": int(features.open_return_pct.notna().sum()),
            "verified_offer_rows_before": int(original.offering_price_review_status.isin(approved).sum()),
            "verified_offer_rows_after": int(changed.offering_price_review_status.isin(approved).sum()),
            "future_information_rows": int(time_audit.is_future_information.sum()),
            "training_authorized": False, "applied": False}
        files = [(path, (RAW_DIR if path.relative_to(root).parts[0] == "raw" else PROC_DIR)
                  / Path(*path.relative_to(root).parts[1:]))
                 for path in root.rglob("*") if path.is_file()]
        if args.apply:
            if hashlib.sha256(source.read_bytes()).hexdigest() != original_hash:
                raise RuntimeError("Source changed during audit; no changes applied")
            backup = PROC_DIR / "repair_backups" / stamp
            backup.mkdir(parents=True)
            replaced = []
            try:
                for new, destination in files:
                    relative = new.relative_to(root)
                    saved = backup / relative
                    saved.parent.mkdir(parents=True, exist_ok=True)
                    existed = destination.exists()
                    if existed:
                        shutil.copy2(destination, saved)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    replaced.append((destination, saved, existed))
                    os.replace(new, destination)
            except Exception:
                for destination, saved, existed in reversed(replaced):
                    if existed:
                        shutil.copy2(saved, destination)
                    else:
                        destination.unlink(missing_ok=True)
                raise
            manifest.update(applied=True, backup=str(backup))
        (PROC_DIR / "verified_sample_repair_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
