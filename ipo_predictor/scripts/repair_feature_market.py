"""Restore missing market metadata by exact official event ID, without API or training."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from config import RAW_DIR, PROC_DIR
from data.pipelines.historical_ipo_pipeline import HistoricalIPOPipeline
from pipeline import assess_training_readiness


def main():
    path = PROC_DIR / "features_all.parquet"
    features = pd.read_parquet(path)
    master = pd.read_parquet(RAW_DIR / "krx_official_event_master.parquet")
    reference = master[["event_id", "market"]].rename(columns={"market": "official_market"})
    repaired = features.merge(reference, on="event_id", how="left", validate="many_to_one", sort=False)
    if repaired["official_market"].isna().any():
        raise RuntimeError("Some event IDs lack an official market; repair aborted.")
    if "market" in features and (features["market"].notna() & features["market"].ne(repaired["official_market"])).any():
        raise RuntimeError("Existing market conflicts with official master; repair aborted.")
    backup = path.with_name("features_before_market_repair.parquet")
    if not backup.exists():
        features.to_parquet(backup, index=False)
    repaired["market"] = repaired.pop("official_market")
    repaired.to_parquet(path, index=False)
    audit = pd.read_parquet(PROC_DIR / "feature_time_validation.parquet")
    stages = HistoricalIPOPipeline()._write_stage_datasets(repaired, audit)
    summary_path = PROC_DIR / "data_collection_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        summary["model_stage_readiness"] = stages
        summary["offline_market_metadata_repair"] = True
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    report = {}
    for stage in stages:
        frame = pd.read_parquet(PROC_DIR / "model_stage_datasets" / f"{stage}.parquet")
        report[stage] = assess_training_readiness(frame, prediction_stage=stage)
    (PROC_DIR / "market_repair_readiness.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
