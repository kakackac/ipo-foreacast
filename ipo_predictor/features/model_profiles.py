"""IPO 정보 공개 단계별 모델 입력 계약이다.

각 모델은 해당 단계보다 늦게 공개된 피처를 요구하지 않는다. 실제 학습은
각 행의 ``available_at`` 검증과 데이터 품질 기준을 모두 통과한 뒤에만 허용한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from features.source_contracts import (
    APPROVED_INSTITUTIONAL_DEMAND_STATUSES,
    APPROVED_LOCKUP_STATUSES,
    APPROVED_OFFERING_STRUCTURE_STATUSES,
    APPROVED_PUBLIC_FLOAT_STATUSES,
    APPROVED_UNDERWRITER_TIER_STATUSES,
)


@dataclass(frozen=True)
class PredictionProfile:
    name: str
    description: str
    feature_names: tuple[str, ...]
    critical_features: tuple[str, ...]


MODEL_PROFILES = {
    "pre_demand": PredictionProfile(
        name="pre_demand",
        description="수요예측 결과 공시 전: 공모 구조·주관사·시장 상황 중심",
        feature_names=(
            "kospi_momentum_5d", "kospi_momentum_20d", "recent_ipo_avg_return_sector",
            "recent_ipo_avg_return_all", "float_share_ratio", "secondary_offering_ratio",
            "major_shareholder_lockup_months", "risk_factor_count",
            "underwriter_tier", "offering_type_spac_ipo",
        ),
        critical_features=("float_share_ratio", "secondary_offering_ratio", "underwriter_tier"),
    ),
    "post_demand": PredictionProfile(
        name="post_demand",
        description="수요예측 결과 공개 후: 확정 공모가·기관 수요·확약을 추가",
        feature_names=(
            "institutional_demand_ratio", "lockup_commitment_ratio",
            "offering_price_band_position", "band_exceeded", "kospi_momentum_5d",
            "kospi_momentum_20d", "recent_ipo_avg_return_sector", "recent_ipo_avg_return_all",
            "offering_type_spac_ipo",
        ),
        critical_features=("institutional_demand_ratio", "offering_price_band_position"),
    ),
}


def get_model_profile(name: str) -> PredictionProfile:
    try:
        return MODEL_PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"지원하지 않는 예측 시점 모델입니다: {name}") from exc


VERIFIED_OFFERING_PRICE_STATUSES = frozenset({
    "verified_currency_unit",
    "verified_text_and_structured",
    "verified_structured_api",
    "manual_verified",
})

FEATURE_SOURCE_APPROVAL = {
    "float_share_ratio": (
        "float_share_validation_status",
        APPROVED_PUBLIC_FLOAT_STATUSES,
    ),
    "secondary_offering_ratio": (
        "offering_structure_validation_status",
        APPROVED_OFFERING_STRUCTURE_STATUSES,
    ),
    "underwriter_tier": (
        "underwriter_validation_status",
        APPROVED_UNDERWRITER_TIER_STATUSES,
    ),
    "institutional_demand_ratio": (
        "institutional_validation_status",
        APPROVED_INSTITUTIONAL_DEMAND_STATUSES,
    ),
    "lockup_commitment_ratio": (
        "lockup_validation_status",
        APPROVED_LOCKUP_STATUSES,
    ),
}


def build_stage_dataset(
    features: pd.DataFrame,
    prediction_stage: str,
    feature_time_audit: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """하나의 원천 피처 표에서 공개 단계별 학습 후보 표를 만든다.

    공개 단계는 서로 다른 표본을 나눠 갖지 않는다. 동일 IPO 행을 보존하면서
    해당 단계의 피처 계약, 공모가 승인, 타깃, 공개시각 검사를 각각 표시한다.
    결측 보정이나 행 삭제는 여기서 수행하지 않는다.
    """
    profile = get_model_profile(prediction_stage)
    frame = features.copy()
    for feature in profile.feature_names:
        if feature not in frame.columns:
            frame[feature] = pd.NA

    frame["prediction_stage"] = profile.name
    frame["stage_features_complete"] = frame[list(profile.feature_names)].notna().all(axis=1)
    frame["stage_critical_features_complete"] = frame[
        list(profile.critical_features)
    ].notna().all(axis=1)
    price_status = frame.get(
        "offering_price_review_status", pd.Series("missing", index=frame.index)
    ).fillna("missing").astype(str)
    frame["stage_offering_price_verified"] = price_status.isin(VERIFIED_OFFERING_PRICE_STATUSES)
    open_ready = pd.to_numeric(
        frame.get("open_return_pct", pd.Series(index=frame.index, dtype=float)), errors="coerce"
    ).notna()
    close_ready = pd.to_numeric(
        frame.get("close_return_pct", pd.Series(index=frame.index, dtype=float)), errors="coerce"
    ).notna()
    frame["stage_dual_target_ready"] = open_ready & close_ready

    frame["stage_time_valid"] = _stage_time_valid(frame, profile, feature_time_audit)
    frame["stage_source_valid"] = stage_source_valid(frame, profile)
    frame["stage_model_candidate"] = stage_model_candidate_mask(frame, profile)
    return frame


def stage_source_valid(features: pd.DataFrame, profile: PredictionProfile) -> pd.Series:
    """의미 검증 계약이 있는 피처는 승인 상태까지 통과해야 한다."""
    valid = pd.Series(True, index=features.index, dtype=bool)
    for feature_name in profile.feature_names:
        contract = FEATURE_SOURCE_APPROVAL.get(feature_name)
        if contract is None:
            continue
        status_column, approved_statuses = contract
        if status_column not in features.columns:
            return pd.Series(False, index=features.index, dtype=bool)
        observed = pd.to_numeric(features[feature_name], errors="coerce").notna()
        approved = features[status_column].fillna("").astype(str).isin(approved_statuses)
        valid &= ~observed | approved
    return valid


def stage_model_candidate_mask(features: pd.DataFrame, profile: PredictionProfile) -> pd.Series:
    """현재 계약으로 학습 후보를 다시 계산한다.

    선택 피처 결측은 훈련 구간에서만 보정하므로 모든 피처의 완비를 요구하지
    않는다. 대신 단계별 핵심 피처, 공모가, 두 타깃, 공개시각, 원천 승인을
    모두 요구한다. 저장된 과거 ``stage_model_candidate`` 값은 사용하지 않는다.
    """
    required_columns = {
        "stage_offering_price_verified", "stage_dual_target_ready", "stage_time_valid",
    }
    if not required_columns.issubset(features.columns):
        return pd.Series(False, index=features.index, dtype=bool)
    critical_complete = features.reindex(columns=profile.critical_features).notna().all(axis=1)
    return (
        critical_complete
        & features["stage_offering_price_verified"].fillna(False).astype(bool)
        & features["stage_dual_target_ready"].fillna(False).astype(bool)
        & features["stage_time_valid"].fillna(False).astype(bool)
        & stage_source_valid(features, profile)
    ).astype(bool)


def _stage_time_valid(
    features: pd.DataFrame, profile: PredictionProfile, feature_time_audit: pd.DataFrame | None
) -> pd.Series:
    """값이 있는 단계 피처는 모두 상장일 이전 공개시각을 가져야 한다."""
    valid = pd.Series(True, index=features.index, dtype=bool)
    if feature_time_audit is None or feature_time_audit.empty or "event_id" not in features:
        return pd.Series(False, index=features.index, dtype=bool)
    required = feature_time_audit[
        feature_time_audit.get("feature_name", pd.Series(dtype=str)).isin(profile.feature_names)
    ].copy()
    if required.empty:
        return pd.Series(False, index=features.index, dtype=bool)
    observed = required[~required.get("is_missing", pd.Series(True, index=required.index)).astype(bool)].copy()
    if observed.empty:
        return valid
    status = observed.get("time_validation_status", pd.Series("", index=observed.index)).astype(str)
    invalid_event_ids = set(
        observed.loc[status != "pre_listing_verified", "event_id"].dropna().astype(str)
    )
    # 피처가 실제로 채워졌는데 그 피처의 관측 원장이 없으면 검증할 수 없으므로 차단한다.
    observed_pairs = set(zip(observed["event_id"].astype(str), observed["feature_name"].astype(str)))
    missing_pairs: set[tuple[str, str]] = set()
    for row in features.itertuples(index=False):
        event_id = str(getattr(row, "event_id", ""))
        for feature in profile.feature_names:
            if pd.notna(getattr(row, feature, pd.NA)) and (event_id, feature) not in observed_pairs:
                missing_pairs.add((event_id, feature))
    invalid_event_ids.update(event_id for event_id, _ in missing_pairs)
    if invalid_event_ids:
        valid = ~features["event_id"].astype(str).isin(invalid_event_ids)
    return valid.astype(bool)


def stage_readiness_by_offering_type(dataset: pd.DataFrame) -> list[dict[str, object]]:
    """유형별로 원천 행부터 실제 모델 후보까지의 감소 사유를 집계한다."""
    if dataset.empty:
        return []
    type_column = "offering_type" if "offering_type" in dataset.columns else "event_class"
    grouping = dataset.assign(**{type_column: dataset[type_column].fillna("review_required")}).groupby(
        type_column, dropna=False
    )
    rows: list[dict[str, object]] = []
    for offering_type, group in grouping:
        rows.append({
            "offering_type": str(offering_type),
            "event_rows": int(len(group)),
            "verified_offering_price_rows": int(group["stage_offering_price_verified"].sum()),
            "dual_target_rows": int(group["stage_dual_target_ready"].sum()),
            "feature_complete_rows": int(group["stage_features_complete"].sum()),
            "critical_feature_complete_rows": int(group["stage_critical_features_complete"].sum()),
            "time_valid_rows": int(group["stage_time_valid"].sum()),
            "source_valid_rows": int(group["stage_source_valid"].sum()),
            "model_candidate_rows": int(group["stage_model_candidate"].sum()),
        })
    return rows
