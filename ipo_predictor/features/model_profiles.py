"""IPO 정보 공개 단계별 모델 입력 계약이다.

각 모델은 해당 단계보다 늦게 공개된 피처를 요구하지 않는다. 실제 학습은
각 행의 ``available_at`` 검증과 데이터 품질 기준을 모두 통과한 뒤에만 허용한다.
"""

from __future__ import annotations

from dataclasses import dataclass


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
            "major_shareholder_lockup_months", "same_day_ipo_count", "risk_factor_count",
            "underwriter_tier",
        ),
        critical_features=("float_share_ratio", "secondary_offering_ratio", "underwriter_tier"),
    ),
    "post_demand": PredictionProfile(
        name="post_demand",
        description="수요예측 결과 공개 후: 확정 공모가·기관 수요·확약을 추가",
        feature_names=(
            "institutional_demand_ratio", "lockup_6m_ratio", "lockup_3m_ratio",
            "lockup_1m_ratio", "lockup_15d_ratio", "lockup_weighted_score",
            "offering_price_band_position", "band_exceeded", "kospi_momentum_5d",
            "kospi_momentum_20d", "recent_ipo_avg_return_sector", "recent_ipo_avg_return_all",
        ),
        critical_features=("institutional_demand_ratio", "offering_price_band_position"),
    ),
    "post_retail": PredictionProfile(
        name="post_retail",
        description="일반청약 마감 후·상장 전: 검증된 통합 개인 청약 경쟁률을 추가",
        feature_names=(
            "institutional_demand_ratio", "retail_subscription_ratio", "lockup_6m_ratio",
            "lockup_3m_ratio", "lockup_1m_ratio", "lockup_15d_ratio", "lockup_weighted_score",
            "offering_price_band_position", "band_exceeded", "kospi_momentum_5d",
            "kospi_momentum_20d", "recent_ipo_avg_return_sector", "recent_ipo_avg_return_all",
        ),
        critical_features=(
            "institutional_demand_ratio", "retail_subscription_ratio", "offering_price_band_position",
        ),
    ),
}


def get_model_profile(name: str) -> PredictionProfile:
    try:
        return MODEL_PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"지원하지 않는 예측 시점 모델입니다: {name}") from exc
