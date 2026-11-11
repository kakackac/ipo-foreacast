"""Freeze a source-checked listing-eve experiment without fitting or scoring models."""
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from config import PROC_DIR
from features.model_profiles import MODEL_PROFILES, build_stage_dataset
from scripts.audit_training_cohorts import EXPERIMENT_EXCLUSIONS, cohort_mask

SPEC = {
    "version": 1, "prediction_time": "listing_eve_end_of_day_Asia_Seoul",
    "service_stage_claim": False, "holdout_start": "2026-01-01",
    "minimum_train_rows": 100, "minimum_validation_rows": 20,
    "validation_windows": "non_overlapping_calendar_years_before_2026",
    "targets": ["open_return_pct", "close_return_pct"],
    "metrics": ["mae_percentage_points", "median_absolute_error_percentage_points", "rmse_percentage_points"],
    "baselines": ["training_target_mean", "training_target_median", "zero_return"],
    "model": {"class": "sklearn.ensemble.GradientBoostingRegressor",
              "n_estimators": 100, "max_depth": 2, "learning_rate": 0.05,
              "min_samples_leaf": 10, "random_state": 42},
    "tuning": "none_for_initial_baseline", "confidence_intervals": "not_estimated",
    "preprocessing": "fit_median_and_missing_indicators_on_each_training_fold_only; reject_all_missing_columns",
    "comparison": "each_feature_set_vs_its_own_baselines; different_cohorts_not_head_to_head",
    "final_holdout_evaluation": "separate_explicit_step_after_development_review",
}


def make_splits(frame):
    if frame.event_id.isna().any() or frame.event_id.duplicated().any():
        raise ValueError("Missing or duplicate event identifiers")
    dates = pd.to_datetime(frame.listing_date, errors="raise")
    if dates.isna().any():
        raise ValueError("Missing listing date")
    development = frame.loc[dates < pd.Timestamp(SPEC["holdout_start"])].sort_values(["listing_date", "event_id"])
    holdout = frame.loc[dates >= pd.Timestamp(SPEC["holdout_start"])].sort_values(["listing_date", "event_id"])
    dev_dates = pd.to_datetime(development.listing_date)
    folds, skipped = [], []
    for year in sorted(dev_dates.dt.year.unique()):
        train = development[dev_dates < pd.Timestamp(int(year), 1, 1)]
        validation = development[dev_dates.dt.year == year]
        counts = {"year": int(year), "train_rows": len(train), "validation_rows": len(validation)}
        if len(train) < SPEC["minimum_train_rows"] or len(validation) < SPEC["minimum_validation_rows"]:
            skipped.append(counts)
            continue
        folds.append(dict(counts, train_event_ids=train.event_id.tolist(),
                          validation_event_ids=validation.event_id.tolist()))
    if not folds or len(holdout) < SPEC["minimum_validation_rows"]:
        raise ValueError("Insufficient development folds or final holdout")
    return development, holdout, folds, skipped


def validate_evidence(frame, fields, observations):
    if observations.duplicated(["event_id", "feature_name"]).any():
        raise ValueError("Duplicate observation evidence")
    evidence = observations.set_index(["event_id", "feature_name"])
    for _, row in frame.iterrows():
        cutoff = pd.Timestamp(row.listing_date).normalize()
        for field in fields:
            value = row[field]
            if pd.isna(value):
                continue
            if (row.event_id, field) not in evidence.index:
                raise ValueError("Observed value has no source evidence")
            item = evidence.loc[(row.event_id, field)]
            available = pd.to_datetime(item.available_at, errors="coerce")
            if (item.is_missing or item.human_review_required or pd.isna(available)
                    or available >= cutoff or pd.isna(item.source_reference)):
                raise ValueError(f"Unapproved or late evidence: {row.event_id} {field}")
            if float(item.raw_value) != float(value):
                raise ValueError("Feature differs from audited raw value")


def main():
    paths = [PROC_DIR / name for name in ("features_all.parquet", "feature_observations.parquet", "feature_time_validation.parquet")]
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    features, observations, time_audit = [pd.read_parquet(path) for path in paths]
    config = dict(SPEC, input_sha256=hashes,
                  excluded_features={name: sorted(values) for name, values in EXPERIMENT_EXCLUSIONS.items()})
    run_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    parent = PROC_DIR / "experiments/listing_eve_v1"
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / run_id
    if destination.exists():
        raise RuntimeError(f"Frozen experiment already exists: {destination}; it will not be overwritten")
    with tempfile.TemporaryDirectory(dir=parent) as temp:
        staging = Path(temp) / "snapshot"
        staging.mkdir()
        manifests = {}
        for name, profile in MODEL_PROFILES.items():
            stage = build_stage_dataset(features, name, time_audit)
            selected = stage[cohort_mask(stage, profile)].copy()
            fields = [field for field in profile.feature_names if field not in EXPERIMENT_EXCLUSIONS[name]]
            validate_evidence(selected, fields, observations)
            columns = ["event_id", "corp_name", "listing_date", *fields, *SPEC["targets"]]
            selected = selected[columns]
            development, holdout, folds, skipped = make_splits(selected)
            for fold in folds:
                train = development[development.event_id.isin(fold["train_event_ids"])]
                if train[fields].isna().all().any():
                    raise ValueError("A training fold has an entirely missing feature")
            group = staging / name
            group.mkdir()
            development.to_parquet(group / "development.parquet", index=False)
            holdout.to_parquet(group / "locked_holdout.parquet", index=False)
            manifests[name] = {"feature_set_origin": name, "prediction_time": SPEC["prediction_time"],
                "features": fields, "development_rows": len(development), "holdout_rows": len(holdout),
                "folds": folds, "skipped_windows": skipped,
                "development_sha256": hashlib.sha256((group / "development.parquet").read_bytes()).hexdigest(),
                "holdout_sha256": hashlib.sha256((group / "locked_holdout.parquet").read_bytes()).hexdigest()}
        if any(hashlib.sha256(path.read_bytes()).hexdigest() != hashes[path.name] for path in paths):
            raise RuntimeError("Inputs changed during experiment preparation")
        manifest = dict(config, run_id=run_id, cohorts=manifests, training_executed=False, holdout_scored=False)
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        os.rename(staging, destination)
    print(json.dumps({"run_id": run_id, "path": str(destination), "cohorts": {
        name: {"development": value["development_rows"], "holdout": value["holdout_rows"],
               "folds": [{k: fold[k] for k in ("year", "train_rows", "validation_rows")} for fold in value["folds"]]}
        for name, value in manifests.items()}, "training_executed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
