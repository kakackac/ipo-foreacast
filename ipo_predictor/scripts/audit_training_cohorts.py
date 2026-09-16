"""Audit all missing cells and bounded verified cohorts; never fit a model."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from config import PROC_DIR
from features.model_profiles import MODEL_PROFILES, build_stage_dataset, stage_model_candidate_mask
from pipeline import assess_training_readiness
from models.evaluation.backtester import WalkForwardBacktester

EXPERIMENT_EXCLUSIONS = {
    "pre_demand": {"major_shareholder_lockup_months", "risk_factor_count"},
    "post_demand": {"offering_price_band_position", "band_exceeded"},
}

def cohort_mask(frame, profile):
    """A quarantine outside the cohort must not poison the eligible cohort."""
    numeric = frame.reindex(columns=["open_return_pct", "close_return_pct"]).apply(pd.to_numeric, errors="coerce")
    return (frame.event_class.eq("general_ipo") & stage_model_candidate_mask(frame, profile)
            & frame.price_target_validation_status.eq("official_price_verified")
            & numeric.notna().all(axis=1) & ~numeric.isin([float("inf"), float("-inf")]).any(axis=1))


def audit(features, observations, time_audit):
    if observations.duplicated(["event_id", "feature_name"]).any():
        raise RuntimeError("Duplicate observation keys; audit aborted")
    missing = observations[observations.is_missing]
    reasons = missing.groupby("missing_reason", dropna=False).size()
    per_feature = []
    for name, group in observations.groupby("feature_name"):
        absent = group[group.is_missing]
        present = group[~group.is_missing]
        per_feature.append({"feature": name, "total": len(group), "missing": len(absent),
            "observed_review_required": int(present.human_review_required.sum()),
            "observed_approved": int((~present.human_review_required).sum()),
            "missing_reasons": {str(k): int(v) for k, v in absent.groupby("missing_reason", dropna=False).size().items()}})
    profiles = {}
    cohorts = {}
    for name, profile in MODEL_PROFILES.items():
        stage = build_stage_dataset(features, name, time_audit)
        selected = stage.loc[cohort_mask(stage, profile)].copy()
        relevant = observations[observations.feature_name.isin(profile.feature_names)]
        unapproved_ids = set(relevant.loc[(~relevant.is_missing) & relevant.human_review_required, "event_id"])
        strict = selected[~selected.event_id.isin(unapproved_ids)].copy()
        experimental_features = [field for field in profile.feature_names
                                 if field not in EXPERIMENT_EXCLUSIONS[name]]
        experiment_evidence = relevant[relevant.feature_name.isin(experimental_features)
                                       & relevant.event_id.isin(selected.event_id)]
        experiment_invalid = experiment_evidence[
            ~experiment_evidence.is_missing & experiment_evidence.human_review_required]
        windows = WalkForwardBacktester()._build_windows(selected, "listing_date") if len(selected) else []
        all_general = stage[stage.event_class.eq("general_ipo")]
        profiles[name] = {
            "whole_dataset_gate": assess_training_readiness(stage, prediction_stage=name),
            "verified_contract_subset_gate": assess_training_readiness(selected, prediction_stage=name),
            "strict_observation_subset_gate": assess_training_readiness(strict, prediction_stage=name),
            "contract_subset_rows": len(selected), "all_observed_fields_approved_rows": len(strict),
            "selected_years": selected.groupby(pd.to_datetime(selected.listing_date).dt.year).size().to_dict(),
            "population_years": all_general.groupby(pd.to_datetime(all_general.listing_date).dt.year).size().to_dict(),
            "unapproved_observed_fields_in_subset": relevant[
                relevant.event_id.isin(selected.event_id) & ~relevant.is_missing & relevant.human_review_required
            ].groupby("feature_name").size().to_dict(),
            "planned_window_sizes": [{"train": len(train), "validation": len(val)} for train, val in windows],
            "phase_specific_prediction_cutoff_verified": False,
            "phase_cutoff_note": "Current time contract checks before listing, not actual pre/post-demand prediction timestamps.",
            "bounded_experiment": {
                "prediction_cutoff": "listing_eve_only_not_pre_demand_or_post_demand_service",
                "features": experimental_features,
                "excluded_unapproved_features": sorted(EXPERIMENT_EXCLUSIONS[name]),
                "unapproved_observed_cells": len(experiment_invalid),
                "rows": len(selected),
                "proposed_holdout_start": "2026-01-01",
                "development_rows": int((pd.to_datetime(selected.listing_date) < pd.Timestamp("2026-01-01")).sum()),
                "holdout_rows": int((pd.to_datetime(selected.listing_date) >= pd.Timestamp("2026-01-01")).sum()),
                "source_contract_passed": len(experiment_invalid) == 0,
                "training_executed": False,
                "release_authorized": False,
            },
        }
        cohorts[name] = selected
    return {"rows": len(features), "feature_cells": len(observations), "missing_cells": len(missing),
        "missing_reasons": {str(k): int(v) for k, v in reasons.items()}, "features": per_feature,
        "profiles": profiles, "training_executed": False, "deployment_authorized": False}, cohorts


def main():
    features = pd.read_parquet(PROC_DIR / "features_all.parquet")
    observations = pd.read_parquet(PROC_DIR / "feature_observations.parquet")
    times = pd.read_parquet(PROC_DIR / "feature_time_validation.parquet")
    report, cohorts = audit(features, observations, times)
    output = PROC_DIR / "training_cohort_audit"
    output.mkdir(exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    for name, frame in cohorts.items():
        # Candidate audit artifacts are deliberately separate from training entrypoint files.
        frame.to_parquet(output / f"{name}_candidate_review.parquet", index=False)
    lines = ["# 결측 원인과 검증 표본 학습 조건 재평가", "", 
        "2026-09-15. 모델 학습과 성능평가는 실행하지 않았다. 전체 데이터의 완벽한 수집을 학습 선행조건으로 삼지 않는다.", "",
        f"전체 {report['rows']}행, {report['feature_cells']}개 피처 칸 중 결측 {report['missing_cells']}칸이다. 아래 사유는 저장된 증거 수준이며 공식 미공개 확정이 아니다.", "",
        "| 피처 | 결측 | 값 있음·검토 필요 | 값 있음·승인 |", "|---|---:|---:|---:|"]
    for item in report["features"]:
        lines.append(f"| {item['feature']} | {item['missing']} | {item['observed_review_required']} | {item['observed_approved']} |")
    lines.extend(["", "## 결측 원인", ""])
    lines.extend(f"- {reason}: {count}칸" for reason, count in report["missing_reasons"].items())
    for name, profile in report["profiles"].items():
        lines.extend(["", f"## {name}", "",
            f"기존 계약 후보 {profile['contract_subset_rows']}건. 이 표본만 평가하면 기존 최소 기준 통과={profile['verified_contract_subset_gate']['eligible']}.",
            f"모든 비결측 피처에 관측 원장 승인을 요구하면 {profile['all_observed_fields_approved_rows']}건이다. 기존 계약 통과를 모든 피처 검증 완료로 해석하면 안 된다.",
            f"승인되지 않은 관측 필드: {profile['unapproved_observed_fields_in_subset']}.",
            f"선택 표본 연도별 분포: {profile['selected_years']}.",
            f"축소 실험 피처: {', '.join(profile['bounded_experiment']['features'])}.",
            f"축소 실험 내 미승인 비결측 칸: {profile['bounded_experiment']['unapproved_observed_cells']}."])
        lines.append(f"제안 분할: 2026년 이전 개발 {profile['bounded_experiment']['development_rows']}건, 2026년 잠금 평가 {profile['bounded_experiment']['holdout_rows']}건. 아직 모델을 학습하거나 이 평가 구간의 성능을 계산하지 않았다.")
    lines.extend(["", "## 결정 및 중단 조건", "",
        "1. 전체 결측을 채우기 위한 파서 확장과 전체 재수집 반복을 중단한다. 일반 IPO의 검증된 표본만 별도 실험 대상으로 고정한다.",
        "2. 기존 100건·3개 연도·연도당 5건·평균 70% 기준은 프로젝트 내부 최소값이지 성능 보장 기준이 아니다. 표본 필터링 후 충족률은 전체 모집단 커버리지와 함께 보고한다.",
        "3. 미승인 피처를 제외한 축소안은 상장 전날 기준 내부 기준선 실험 후보로만 사용한다. pre/post-demand 실제 시점 서비스의 승인이 아니다. 제외 피처는 값 또는 검증 계약을 확보한 후 별도 비교 실험으로만 추가한다.",
        "4. 2026년을 최종 평가로 잠그고 이전 자료 안에서만 튜닝한다. 기존 윈도우는 초기 학습이 20~30건까지 작아질 수 있으므로 실험에서는 학습 100건·검증 20건 이상인 구간만 쓰고 중첩 평가의 중복 예측을 합산하지 않는다. 보정/인코딩은 각 학습 구간에서만 맞춘다. 단순 기준선과 비교하고 모집단 대비 연도·주관사·규모 편향도 보고해야 한다.",
        "5. 미래시점/원천 혼합/중복 이벤트처럼 기존 검증을 무효화하는 오류만 즉시 수집·파서 수정 대상으로 삼는다. 단순 결측률은 재수집 지시 사유가 아니다.",
        "6. 저장된 단계 시점 검사는 현재 상장일 이전 여부만 확인한다. 실제 수요예측 전/후 시점의 데이터 스냅샷 검증 없이 해당 단계 성능이나 출시를 주장하지 않는다.",
        "7. 이번 산출물은 별도 검토 폴더에 저장했고 기본 학습 진입점과 안전장치를 자동 해제하지 않았다. 성능 수치 없음, 배포 승인 없음.", ""])
    document = Path(__file__).resolve().parents[2] / "docs/결측_및_학습표본_재평가.md"
    document.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"rows": report["rows"], "missing_cells": report["missing_cells"],
        "missing_reasons": report["missing_reasons"], "profiles": {name: {
            key: profile[key] for key in ["contract_subset_rows", "all_observed_fields_approved_rows",
                "unapproved_observed_fields_in_subset", "selected_years"]}
            | {"subset_gate": profile["verified_contract_subset_gate"]["eligible"],
               "strict_gate": profile["strict_observation_subset_gate"]["eligible"]}
        for name, profile in report["profiles"].items()}}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
