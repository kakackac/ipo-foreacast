"""
features/definitions.py
───────────────────────
모든 피처의 정의, 출처, 계산 공식, 결측 처리 방법을 한 곳에서 관리한다.

피처를 추가하거나 제거할 때 이 파일만 수정하면 된다.
모델 학습 코드와 API 서빙 코드가 모두 이 정의를 참조한다.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class FeatureGroup(str, Enum):
    SUBSCRIPTION  = "subscription"   # 청약/수요예측 관련
    VALUATION     = "valuation"      # 공모가 매력도/밸류에이션
    FINANCIAL     = "financial"      # 기업 재무제표
    MARKET        = "market"         # 시장 컨텍스트
    SUPPLY        = "supply"         # 수급 (유통물량, 확약)
    IPO_STRUCTURE = "ipo_structure"  # 공모 구조


class Importance(str, Enum):
    CORE      = "core"       # Phase 1 — 반드시 있어야 함
    SECONDARY = "secondary"  # Phase 2 — 성능 개선에 기여
    OPTIONAL  = "optional"   # Phase 3 — 실험적 추가


@dataclass
class FeatureDef:
    name:        str
    group:       FeatureGroup
    importance:  Importance
    dtype:       str           # "float", "int", "bool"
    description: str
    source:      str           # 데이터 출처
    formula:     str           # 계산 공식 or 추출 방법
    fill_na:     str           # "mean", "median", "zero", "flag" (학습 분할 뒤 결측 처리)
    clip:        Optional[tuple] = None   # (min, max) 클리핑 범위

    def __repr__(self):
        return f"[{self.importance.value.upper()}] {self.name} ({self.group.value})"


# ═══════════════════════════════════════════════════════════════
# PHASE 1 — CORE FEATURES
# 예측력이 높고 DART에서 명확하게 추출 가능한 정량 피처
# ═══════════════════════════════════════════════════════════════

CORE_FEATURES: list[FeatureDef] = [

    # ── 청약/수요예측 ──────────────────────────────────────────

    FeatureDef(
        name        = "institutional_demand_ratio",
        group       = FeatureGroup.SUBSCRIPTION,
        importance  = Importance.CORE,
        dtype       = "float",
        description = "기관 수요예측 경쟁률",
        source      = "DART 수요예측 결과 공시 또는 주관사 공식 수요예측 결과 공지",
        formula     = "기관 신청 물량 합계 / 배정 가능 물량",
        fill_na     = "median",
        clip        = (0, 3000),
    ),

    FeatureDef(
        name        = "retail_subscription_ratio",
        group       = FeatureGroup.SUBSCRIPTION,
        importance  = Importance.CORE,
        dtype       = "float",
        description = "개인 청약 경쟁률",
        source      = "주관사 공식 청약 결과 공지(통합 경쟁률이 명시된 경우만 자동 승인)",
        formula     = "전체 참여 증권사를 포함한다고 명시된 개인 청약 경쟁률",
        fill_na     = "median",
        clip        = (0, 5000),
    ),

    FeatureDef(
        name        = "lockup_6m_ratio",
        group       = FeatureGroup.SUPPLY,
        importance  = Importance.CORE,
        dtype       = "float",
        description = "기관 6개월 의무보유확약 비율",
        source      = "DART 수요예측 결과 공시",
        formula     = "6개월 확약 신청 물량 / 전체 기관 배정 물량",
        fill_na     = "median",
        clip        = (0, 1),
    ),

    FeatureDef(
        name        = "lockup_3m_ratio",
        group       = FeatureGroup.SUPPLY,
        importance  = Importance.CORE,
        dtype       = "float",
        description = "기관 3개월 의무보유확약 비율",
        source      = "DART 수요예측 결과 공시",
        formula     = "3개월 확약 신청 물량 / 전체 기관 배정 물량",
        fill_na     = "median",
        clip        = (0, 1),
    ),

    FeatureDef(
        name        = "lockup_1m_ratio",
        group       = FeatureGroup.SUPPLY,
        importance  = Importance.CORE,
        dtype       = "float",
        description = "기관 1개월 의무보유확약 비율",
        source      = "DART 수요예측 결과 공시",
        formula     = "1개월 확약 신청 물량 / 전체 기관 배정 물량",
        fill_na     = "median",
        clip        = (0, 1),
    ),

    FeatureDef(
        name        = "lockup_15d_ratio",
        group       = FeatureGroup.SUPPLY,
        importance  = Importance.CORE,
        dtype       = "float",
        description = "기관 15일 의무보유확약 비율",
        source      = "DART 수요예측 결과 공시",
        formula     = "15일 확약 신청 물량 / 전체 기관 배정 물량",
        fill_na     = "median",
        clip        = (0, 1),
    ),

    FeatureDef(
        name        = "lockup_weighted_score",
        group       = FeatureGroup.SUPPLY,
        importance  = Importance.CORE,
        dtype       = "float",
        description = "확약기간 가중 점수 (파생 피처)",
        source      = "lockup_6m/3m/1m/15d_ratio 파생",
        formula     = "6m×1.0 + 3m×0.75 + 1m×0.5 + 15d×0.25",
        fill_na     = "median",
        clip        = (0, 1),
    ),

    FeatureDef(
        name        = "offering_price_band_position",
        group       = FeatureGroup.VALUATION,
        importance  = Importance.CORE,
        dtype       = "float",
        description = "공모가 희망밴드 내 위치 (0=하단, 1=상단, >1=초과)",
        source      = "DART 증권신고서",
        formula     = "(확정공모가 - 밴드하단) / (밴드상단 - 밴드하단)",
        fill_na     = "median",
        clip        = (-0.5, 2.0),
    ),

    FeatureDef(
        name        = "band_exceeded",
        group       = FeatureGroup.VALUATION,
        importance  = Importance.CORE,
        dtype       = "bool",
        description = "공모가가 희망밴드 상단을 초과했는지 여부",
        source      = "offering_price_band_position 파생",
        formula     = "offering_price_band_position > 1.0",
        fill_na     = "median",
    ),

    FeatureDef(
        name        = "kospi_momentum_5d",
        group       = FeatureGroup.MARKET,
        importance  = Importance.CORE,
        dtype       = "float",
        description = "상장일 기준 KOSPI 5일 수익률",
        source      = "KRX 시장 데이터",
        formula     = "KOSPI[상장일-1] / KOSPI[상장일-6] - 1",
        fill_na     = "median",
        clip        = (-0.2, 0.2),
    ),

    FeatureDef(
        name        = "kospi_momentum_20d",
        group       = FeatureGroup.MARKET,
        importance  = Importance.CORE,
        dtype       = "float",
        description = "상장일 기준 KOSPI 20일 수익률",
        source      = "KRX 시장 데이터",
        formula     = "KOSPI[상장일-1] / KOSPI[상장일-21] - 1",
        fill_na     = "median",
        clip        = (-0.3, 0.3),
    ),

    FeatureDef(
        name        = "recent_ipo_avg_return_sector",
        group       = FeatureGroup.MARKET,
        importance  = Importance.CORE,
        dtype       = "float",
        description = "동일 섹터 최근 5개 IPO 상장일 평균 수익률",
        source      = "내부 히스토리 DB",
        formula     = "해당 섹터 직전 5개 상장 종목의 시초가 수익률 평균",
        fill_na     = "mean",
        clip        = (-0.5, 2.0),
    ),

    FeatureDef(
        name        = "recent_ipo_avg_return_all",
        group       = FeatureGroup.MARKET,
        importance  = Importance.CORE,
        dtype       = "float",
        description = "전체 최근 10개 IPO 상장일 평균 수익률",
        source      = "내부 히스토리 DB",
        formula     = "직전 10개 상장 종목의 시초가 수익률 평균",
        fill_na     = "mean",
        clip        = (-0.5, 2.0),
    ),
]


# ═══════════════════════════════════════════════════════════════
# PHASE 2 — SECONDARY FEATURES
# Rule-based 추출 가능, 수집 난이도 중간
# ═══════════════════════════════════════════════════════════════

SECONDARY_FEATURES: list[FeatureDef] = [

    FeatureDef(
        name        = "offering_type_spac_ipo",
        group       = FeatureGroup.IPO_STRUCTURE,
        importance  = Importance.SECONDARY,
        dtype       = "bool",
        description = "공모 유형이 스팩 IPO인지 여부",
        source      = "KRX KIND 신규상장종목 현황 분류",
        formula     = "offering_type == 'spac_ipo'",
        fill_na     = "flag",
    ),

    FeatureDef(
        name        = "offering_type_foreign_common_stock",
        group       = FeatureGroup.IPO_STRUCTURE,
        importance  = Importance.SECONDARY,
        dtype       = "bool",
        description = "공모 유형이 외국기업 보통주 상장인지 여부",
        source      = "KRX KIND 신규상장종목 현황 분류",
        formula     = "offering_type == 'foreign_common_stock_listing'",
        fill_na     = "flag",
    ),

    FeatureDef(
        name        = "float_share_ratio",
        group       = FeatureGroup.SUPPLY,
        importance  = Importance.SECONDARY,
        dtype       = "float",
        description = "상장 직후 유통 가능 물량 비율",
        source      = "DART 증권신고서 주식분포표",
        formula     = "(공모 신주 + 구주매출) / 상장 후 총 발행주식수",
        fill_na     = "median",
        clip        = (0, 1),
    ),

    FeatureDef(
        name        = "secondary_offering_ratio",
        group       = FeatureGroup.SUPPLY,
        importance  = Importance.SECONDARY,
        dtype       = "float",
        description = "구주매출 비율 (기존 주주 엑싯 비중)",
        source      = "DART 증권신고서",
        formula     = "구주매출 물량 / (신주발행 + 구주매출) 합계",
        fill_na     = "median",
        clip        = (0, 1),
    ),

    FeatureDef(
        name        = "major_shareholder_lockup_months",
        group       = FeatureGroup.SUPPLY,
        importance  = Importance.SECONDARY,
        dtype       = "int",
        description = "최대주주 의무보유기간 (월)",
        source      = "DART 증권신고서 주요주주 현황",
        formula     = "보호예수기간 텍스트 → 월 단위 정수 변환 (6개월=6, 1년=12)",
        fill_na     = "median",
        clip        = (0, 36),
    ),

    FeatureDef(
        name        = "same_day_ipo_count",
        group       = FeatureGroup.MARKET,
        importance  = Importance.SECONDARY,
        dtype       = "int",
        description = "동일 상장일 IPO 종목 수 (수급 분산 효과)",
        source      = "KRX IPO 일정",
        formula     = "같은 날짜 상장 종목 수 카운트",
        fill_na     = "zero",
        clip        = (0, 20),
    ),

    FeatureDef(
        name        = "risk_factor_count",
        group       = FeatureGroup.IPO_STRUCTURE,
        importance  = Importance.SECONDARY,
        dtype       = "int",
        description = "투자설명서 위험요소 항목 수",
        source      = "DART 투자설명서 — 위험요소 섹션 Rule-based 파싱",
        formula     = "'위험요소' 섹션의 소항목(가., 나., ...) 개수 카운트",
        fill_na     = "median",
        clip        = (0, 60),
    ),

    FeatureDef(
        name        = "underwriter_tier",
        group       = FeatureGroup.IPO_STRUCTURE,
        importance  = Importance.SECONDARY,
        dtype       = "int",
        description = "주관사 등급 (1=대형IB, 2=중형, 3=소형)",
        source      = "주관사명 → 내부 등급 매핑 테이블",
        formula     = "주관사명 정규화 후 등급 코드 매핑",
        fill_na     = "median",
        clip        = (1, 3),
    ),

    # ── 밸류에이션 / 재무 ──────────────────────────────────────

    FeatureDef(
        name        = "offering_per",
        group       = FeatureGroup.VALUATION,
        importance  = Importance.SECONDARY,
        dtype       = "float",
        description = "공모 기준 PER (주가수익비율)",
        source      = "공모가 / 주당순이익(EPS, DART 재무제표)",
        formula     = "공모가 / EPS (최근 12개월 또는 추정)",
        fill_na     = "median",
        clip        = (0, 500),
    ),

    FeatureDef(
        name        = "per_vs_sector_median",
        group       = FeatureGroup.VALUATION,
        importance  = Importance.SECONDARY,
        dtype       = "float",
        description = "공모 PER / 동일 섹터 상장사 PER 중앙값",
        source      = "offering_per / 섹터 PER 중앙값",
        formula     = "offering_per / sector_median_per (>1이면 섹터 대비 비싸다)",
        fill_na     = "median",
        clip        = (0, 10),
    ),

    FeatureDef(
        name        = "revenue_growth_3y",
        group       = FeatureGroup.FINANCIAL,
        importance  = Importance.SECONDARY,
        dtype       = "float",
        description = "매출액 3년 연평균 성장률 (CAGR)",
        source      = "DART 감사보고서 재무제표",
        formula     = "(매출[T] / 매출[T-3])^(1/3) - 1",
        fill_na     = "median",
        clip        = (-0.5, 5.0),
    ),

    FeatureDef(
        name        = "operating_margin",
        group       = FeatureGroup.FINANCIAL,
        importance  = Importance.SECONDARY,
        dtype       = "float",
        description = "영업이익률 (최근 연도)",
        source      = "DART 재무제표",
        formula     = "영업이익 / 매출액",
        fill_na     = "median",
        clip        = (-1.0, 1.0),
    ),

    FeatureDef(
        name        = "debt_ratio",
        group       = FeatureGroup.FINANCIAL,
        importance  = Importance.SECONDARY,
        dtype       = "float",
        description = "부채비율",
        source      = "DART 재무제표",
        formula     = "총부채 / 자기자본",
        fill_na     = "median",
        clip        = (0, 10),
    ),
]


# ═══════════════════════════════════════════════════════════════
# PHASE 3 — OPTIONAL FEATURES (에러 분석 후 필요시 추가)
# ═══════════════════════════════════════════════════════════════

OPTIONAL_FEATURES: list[FeatureDef] = [

    FeatureDef(
        name        = "kospi_momentum_60d",
        group       = FeatureGroup.MARKET,
        importance  = Importance.OPTIONAL,
        dtype       = "float",
        description = "상장일 기준 KOSPI 60일 수익률 (약세장 감지)",
        source      = "KRX 시장 데이터",
        formula     = "KOSPI[상장일-1] / KOSPI[상장일-61] - 1",
        fill_na     = "zero",
        clip        = (-0.5, 0.5),
    ),

    FeatureDef(
        name        = "kosdaq_momentum_20d",
        group       = FeatureGroup.MARKET,
        importance  = Importance.OPTIONAL,
        dtype       = "float",
        description = "상장일 기준 KOSDAQ 20일 수익률",
        source      = "KRX 시장 데이터",
        formula     = "KOSDAQ[상장일-1] / KOSDAQ[상장일-21] - 1",
        fill_na     = "zero",
        clip        = (-0.3, 0.3),
    ),

    FeatureDef(
        name        = "news_mention_count_7d",
        group       = FeatureGroup.MARKET,
        importance  = Importance.OPTIONAL,
        dtype       = "int",
        description = "상장 전 7일 뉴스 언급 건수 (감성 아님, 횟수만)",
        source      = "금융 뉴스 크롤러",
        formula     = "종목명 포함 뉴스 기사 수 (7일)",
        fill_na     = "zero",
        clip        = (0, 500),
    ),
]


# ═══════════════════════════════════════════════════════════════
# 유틸 함수
# ═══════════════════════════════════════════════════════════════

ALL_FEATURES = CORE_FEATURES + SECONDARY_FEATURES + OPTIONAL_FEATURES
FEATURE_MAP: dict[str, FeatureDef] = {f.name: f for f in ALL_FEATURES}


def get_feature_names(importance: Importance | None = None) -> list[str]:
    """특정 중요도 이상의 피처 이름 목록 반환"""
    order = [Importance.CORE, Importance.SECONDARY, Importance.OPTIONAL]
    if importance is None:
        return [f.name for f in ALL_FEATURES]
    idx = order.index(importance)
    allowed = set(order[:idx+1])
    return [f.name for f in ALL_FEATURES if f.importance in allowed]


def get_core_feature_names() -> list[str]:
    return get_feature_names(Importance.CORE)


def get_phase2_feature_names() -> list[str]:
    return get_feature_names(Importance.SECONDARY)


def fill_na_strategy(feature_name: str) -> str:
    """피처별 결측값 처리 전략 반환"""
    return FEATURE_MAP[feature_name].fill_na


def print_feature_summary():
    from collections import Counter
    cnt = Counter(f.importance.value for f in ALL_FEATURES)
    print(f"\n피처 요약: 전체 {len(ALL_FEATURES)}개")
    print(f"  CORE:      {cnt['core']}개 (Phase 1)")
    print(f"  SECONDARY: {cnt['secondary']}개 (Phase 2)")
    print(f"  OPTIONAL:  {cnt['optional']}개 (Phase 3, 에러 분석 후 결정)")
    print()
    for grp in FeatureGroup:
        items = [f for f in ALL_FEATURES if f.group == grp]
        if items:
            print(f"  [{grp.value}]")
            for f in items:
                tag = "✓" if f.importance == Importance.CORE else "○"
                print(f"    {tag} {f.name}")


if __name__ == "__main__":
    print_feature_summary()
