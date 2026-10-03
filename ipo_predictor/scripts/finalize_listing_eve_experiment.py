"""Freeze the decision before a single held-out evaluation; no deployment actions."""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from scripts.prepare_listing_eve_experiment import SPEC
from scripts.run_listing_eve_development import prepare_inputs, summarize


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired_improvement(actual, model, baseline, dates, repetitions=2000):
    differences = np.abs(np.asarray(actual) - np.asarray(baseline)) - np.abs(np.asarray(actual) - np.asarray(model))
    groups = pd.DataFrame({"day": pd.to_datetime(dates).to_numpy(), "difference": differences}).groupby("day").difference.agg(["sum", "count"])
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(groups), size=(repetitions, len(groups)))
    bootstrap = groups["sum"].to_numpy()[draws].sum(axis=1) / groups["count"].to_numpy()[draws].sum(axis=1)
    return {"mae_improvement_pp": float(differences.mean()),
            "day_cluster_bootstrap_95_interval_pp": np.quantile(bootstrap, [.025, .975]).tolist(),
            "day_clusters": len(groups), "fraction_lower_absolute_error": float((differences > 0).mean())}


def check_partitions(development, holdout):
    for frame in (development, holdout):
        if frame.event_id.isna().any() or frame.event_id.duplicated().any():
            raise ValueError("Invalid event identity")
    if set(development.event_id) & set(holdout.event_id):
        raise ValueError("Development/holdout overlap")
    boundary = pd.Timestamp(SPEC["holdout_start"])
    before = pd.to_datetime(development.listing_date, errors="coerce")
    after = pd.to_datetime(holdout.listing_date, errors="coerce")
    if before.isna().any() or after.isna().any() or not (before < boundary).all() or not (after >= boundary).all():
        raise ValueError("Invalid temporal partition")
    if len(development) < 100 or len(holdout) < 20:
        raise ValueError("Insufficient final evaluation samples")


def freeze(experiment):
    experiment = Path(experiment)
    manifest = json.loads((experiment / "manifest.json").read_text())
    if any(manifest.get(key) != value for key, value in SPEC.items()):
        raise ValueError("Experiment specification mismatch")
    predictions = pd.read_parquet(experiment / "development_results/predictions.parquet")
    development = pd.read_parquet(experiment / "post_demand/development.parquet")
    if digest(experiment / "post_demand/development.parquet") != manifest["cohorts"]["post_demand"]["development_sha256"]:
        raise ValueError("Development file changed")
    diagnostics = {}
    for target in SPEC["targets"]:
        target_rows = predictions[predictions.feature_set.eq("post_demand") & predictions.target.eq(target)]
        if target_rows.duplicated(["event_id", "estimator"]).any():
            raise ValueError("Duplicate validation predictions")
        wide = target_rows.pivot(index="event_id", columns="estimator", values="prediction")
        actual = target_rows.drop_duplicates("event_id").set_index("event_id").actual.reindex(wide.index)
        dates = development.set_index("event_id").listing_date.reindex(wide.index)
        diagnostics[target] = paired_improvement(actual, wide.gradient_boosting, wide.training_target_mean, dates)
    policy = {"version": 1, "experiment_id": manifest["run_id"],
        "frozen_at": datetime.now(timezone.utc).isoformat(), "feature_set": "post_demand",
        "scope": "listing_eve_selected_general_IPOs_only",
        "selected_estimators": {"open_return_pct": "gradient_boosting", "close_return_pct": "training_target_median"},
        "selection_reason": "Open model beat both simple baselines in every development year; close model did not beat the median overall.",
        "manifest_sha256": digest(experiment / "manifest.json"),
        "development_predictions_sha256": digest(experiment / "development_results/predictions.parquet"),
        "development_diagnostics": diagnostics,
        "research_signal_rule": "Open MAE lower than mean and median baselines AND bootstrap improvement vs mean lower bound > 0.",
        "deployment_authorized": False, "holdout_accessed": False,
        "post_evaluation_policy": "Never retune on this holdout or claim it remains unseen; no automatic deployment."}
    with (experiment / "final_policy.json").open("x", encoding="utf-8") as handle:
        json.dump(policy, handle, ensure_ascii=False, indent=2)
    return policy


def evaluate(experiment):
    experiment = Path(experiment)
    policy = json.loads((experiment / "final_policy.json").read_text())
    manifest = json.loads((experiment / "manifest.json").read_text())
    if digest(experiment / "manifest.json") != policy["manifest_sha256"] or any(manifest.get(key) != value for key, value in SPEC.items()):
        raise ValueError("Frozen manifest changed")
    if digest(experiment / "development_results/predictions.parquet") != policy["development_predictions_sha256"]:
        raise ValueError("Development predictions changed")
    if policy["feature_set"] != "post_demand" or policy["selected_estimators"] != {
        "open_return_pct": "gradient_boosting", "close_return_pct": "training_target_median"}:
        raise ValueError("Unsupported evaluation policy")
    marker = experiment / "final_evaluation_started.json"
    with marker.open("x", encoding="utf-8") as handle:
        json.dump({"started_at": datetime.now(timezone.utc).isoformat(),
                   "policy_sha256": digest(experiment / "final_policy.json")}, handle)
    cohort = manifest["cohorts"][policy["feature_set"]]
    folder = experiment / policy["feature_set"]
    for filename, expected in (("development.parquet", cohort["development_sha256"]),
                               ("locked_holdout.parquet", cohort["holdout_sha256"])):
        if digest(folder / filename) != expected:
            raise ValueError("Snapshot hash mismatch; evaluation attempt recorded, not silently retried")
    development = pd.read_parquet(folder / "development.parquet")
    holdout = pd.read_parquet(folder / "locked_holdout.parquet")
    check_partitions(development, holdout)
    X, H = prepare_inputs(development, holdout, cohort["features"])
    records = []
    comparisons = {}
    for target, selected in policy["selected_estimators"].items():
        y = pd.to_numeric(development[target], errors="raise")
        actual = pd.to_numeric(holdout[target], errors="raise")
        if not np.isfinite(y).all() or not np.isfinite(actual).all():
            raise ValueError("Nonfinite target")
        outputs = {"training_target_mean": np.full(len(holdout), y.mean()),
                   "training_target_median": np.full(len(holdout), y.median()), "zero_return": np.zeros(len(holdout))}
        if selected == "gradient_boosting":
            model = GradientBoostingRegressor(**{k: v for k, v in SPEC["model"].items() if k != "class"})
            model.fit(X, y)
            outputs[selected] = model.predict(H)
        for estimator, prediction in outputs.items():
            records.extend({"feature_set": policy["feature_set"], "event_id": event, "target": target,
                "estimator": estimator, "actual": float(a), "prediction": float(p)}
                for event, a, p in zip(holdout.event_id, actual, prediction))
        comparisons[target] = paired_improvement(actual, outputs[selected], outputs["training_target_mean"], holdout.listing_date)
    frame = pd.DataFrame(records)
    metrics = summarize(frame)
    opening = {r["estimator"]: r["mae_pp"] for r in metrics if r["target"] == "open_return_pct"}
    signal = (opening["gradient_boosting"] < min(opening["training_target_mean"], opening["training_target_median"])
              and comparisons["open_return_pct"]["day_cluster_bootstrap_95_interval_pp"][0] > 0)
    report = {"experiment_id": manifest["run_id"], "scope": policy["scope"], "n": len(holdout),
        "metrics": metrics, "paired_comparisons_vs_mean": comparisons,
        "predeclared_research_signal_passed": bool(signal), "deployment_authorized": False,
        "holdout_now_consumed": True, "policy_sha256": digest(experiment / "final_policy.json"),
        "warning": "Small selected sample; bootstrap interval is not a guarantee. No pre/post-demand stage or investment-profit claim."}
    output = experiment / "final_results"
    output.mkdir(exist_ok=False)
    frame.to_parquet(output / "predictions.parquet", index=False)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["freeze", "evaluate"])
    parser.add_argument("--experiment", required=True)
    arguments = parser.parse_args()
    print(json.dumps((freeze if arguments.mode == "freeze" else evaluate)(arguments.experiment), ensure_ascii=False))
