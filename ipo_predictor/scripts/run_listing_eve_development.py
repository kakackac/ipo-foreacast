"""Run frozen development folds only; never open the final holdout file."""
import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from scripts.prepare_listing_eve_experiment import SPEC


def prepare_inputs(train, validation, fields):
    train = train[fields].apply(pd.to_numeric, errors="raise")
    validation = validation[fields].apply(pd.to_numeric, errors="raise")
    if np.isinf(train.to_numpy(dtype=float)).any() or np.isinf(validation.to_numpy(dtype=float)).any():
        raise ValueError("Nonfinite feature")
    medians = train.median()
    if medians.isna().any():
        raise ValueError("Entire training feature is missing")
    def transform(frame):
        return pd.concat([frame.fillna(medians), frame.isna().astype(int).add_suffix("__missing")], axis=1)
    return transform(train), transform(validation)


def validate_fold(frame, fold):
    train_ids, validation_ids = fold["train_event_ids"], fold["validation_event_ids"]
    if len(set(train_ids)) != len(train_ids) or len(set(validation_ids)) != len(validation_ids):
        raise ValueError("Duplicate fold members")
    if set(train_ids) & set(validation_ids):
        raise ValueError("Train/validation overlap")
    train = frame.loc[train_ids]
    validation = frame.loc[validation_ids]
    if len(train) < SPEC["minimum_train_rows"] or len(validation) < SPEC["minimum_validation_rows"]:
        raise ValueError("Undersized fold")
    if pd.to_datetime(train.listing_date).max() >= pd.to_datetime(validation.listing_date).min():
        raise ValueError("Nonchronological fold")
    if (pd.to_datetime(validation.listing_date) >= pd.Timestamp(SPEC["holdout_start"])).any():
        raise ValueError("Holdout leaked into development")
    return train, validation


def summarize(predictions):
    rows = []
    for keys, group in predictions.groupby(["feature_set", "target", "estimator"], sort=True):
        error = group.prediction - group.actual
        rows.append(dict(zip(("feature_set", "target", "estimator"), keys),
            n=len(group), mae_pp=float(error.abs().mean()), median_ae_pp=float(error.abs().median()),
            rmse_pp=float(np.sqrt((error ** 2).mean()))))
    return rows


def run(experiment):
    experiment = Path(experiment)
    manifest = json.loads((experiment / "manifest.json").read_text())
    if any(manifest.get(key) != value for key, value in SPEC.items()):
        raise ValueError("Frozen specification differs from runner specification")
    output = experiment / "development_results"
    if output.exists():
        raise RuntimeError("Development results already exist; no silent rerun or overwrite")
    records = []
    for name, cohort in manifest["cohorts"].items():
        path = experiment / name / "development.parquet"
        if hashlib.sha256(path.read_bytes()).hexdigest() != cohort["development_sha256"]:
            raise ValueError("Development data hash mismatch")
        frame = pd.read_parquet(path)
        if frame.event_id.duplicated().any() or frame.event_id.isna().any():
            raise ValueError("Invalid event identity")
        if (pd.to_datetime(frame.listing_date) >= pd.Timestamp(SPEC["holdout_start"])).any():
            raise ValueError("Final holdout data in development file")
        frame = frame.set_index("event_id", drop=False)
        seen_validation = set()
        for fold in cohort["folds"]:
            if seen_validation & set(fold["validation_event_ids"]):
                raise ValueError("Repeated validation predictions")
            seen_validation.update(fold["validation_event_ids"])
            train, validation = validate_fold(frame, fold)
            X_train, X_validation = prepare_inputs(train, validation, cohort["features"])
            for target in SPEC["targets"]:
                y_train = pd.to_numeric(train[target], errors="raise")
                actual = pd.to_numeric(validation[target], errors="raise")
                if not np.isfinite(y_train).all() or not np.isfinite(actual).all():
                    raise ValueError("Invalid target")
                parameters = {key: value for key, value in SPEC["model"].items() if key != "class"}
                model = GradientBoostingRegressor(**parameters)
                model.fit(X_train, y_train)
                predictions = {"gradient_boosting": model.predict(X_validation),
                    "training_target_mean": np.full(len(actual), y_train.mean()),
                    "training_target_median": np.full(len(actual), y_train.median()),
                    "zero_return": np.zeros(len(actual))}
                for estimator, prediction in predictions.items():
                    records.extend({"feature_set": name, "target": target, "year": fold["year"],
                        "event_id": event_id, "estimator": estimator, "actual": float(y), "prediction": float(p)}
                        for event_id, y, p in zip(validation.event_id, actual, prediction))
    predictions = pd.DataFrame(records)
    report = {"experiment_id": manifest["run_id"], "scope": "listing_eve_development_only",
        "holdout_opened": False, "holdout_scored": False, "deployment_authorized": False,
        "metrics": summarize(predictions),
        "manifest_sha256": hashlib.sha256((experiment / "manifest.json").read_bytes()).hexdigest()}
    with tempfile.TemporaryDirectory(dir=experiment) as directory:
        staging = Path(directory) / "results"
        staging.mkdir()
        predictions.to_parquet(staging / "predictions.parquet", index=False)
        (staging / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        os.rename(staging, output)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.experiment), ensure_ascii=False))
