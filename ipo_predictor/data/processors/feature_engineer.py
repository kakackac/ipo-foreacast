"""
data/processors/feature_engineer.py
─────────────────────────────────────
수집된 원본 데이터를 모델 입력 피처로 변환한다.

입력: DART + KRX 수집 데이터 (raw parquet)
출력: 피처 행렬 DataFrame (features/definitions.py 정의 기반)

핵심 변환:
  1. 의무확약 가중 점수 계산
  2. 공모가 밴드 위치 정규화
  3. 시장 모멘텀 계산 (KOSPI rolling return)
  4. 섹터 IPO 온도 계산 (최근 N개 평균 수익률)
  5. 밸류에이션 (PER, PER vs 섹터 비교)
  6. 결측값 처리 (피처별 전략 적용)
"""

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd

from features.definitions import (
    FEATURE_MAP, get_core_feature_names, get_phase2_feature_names,
    fill_na_strategy, FeatureGroup
)
from config import FEATURE_CFG, PROC_DIR, RAW_DIR

logger = logging.getLogger(__name__)


UNDERWRITER_TIER_1 = (
    "NH투자증권",
    "한국투자증권",
    "미래에셋증권",
    "KB증권",
    "삼성증권",
    "신한투자증권",
    "하나증권",
)

UNDERWRITER_TIER_2 = (
    "대신증권",
    "키움증권",
    "신영증권",
    "유안타증권",
    "교보증권",
    "하이투자증권",
    "IBK투자증권",
    "DB금융투자",
    "메리츠증권",
    "현대차증권",
    "유진투자증권",
    "한화투자증권",
    "SK증권",
)


class FeatureEngineer:
    """
    원본 수집 데이터 → 모델 피처 행렬 변환기.

    사용법:
        fe = FeatureEngineer()
        fe.fit(train_df)          # 결측값 통계 학습
        X_train = fe.transform(train_df)
        X_test  = fe.transform(test_df)
    """

    def __init__(self, feature_set: str = "core"):
        """
        feature_set: "core" | "phase2" | "all"
        """
        if feature_set == "core":
            self.feature_names = get_core_feature_names()
        elif feature_set == "phase2":
            self.feature_names = get_phase2_feature_names()
        else:
            from features.definitions import get_feature_names
            self.feature_names = get_feature_names()

        self.fill_values: dict[str, float] = {}
        self._fitted = False
        logger.info("FeatureEngineer 초기화: %d개 피처 (%s)", len(self.feature_names), feature_set)

    # ── 메인 파이프라인 ────────────────────────────────────────

    def build_features(
        self,
        dart_df:   pd.DataFrame,   # DART 수집 데이터
        krx_df:    pd.DataFrame,   # KRX IPO 캘린더 + 상장일 가격
        kospi_df:  pd.DataFrame,   # KOSPI 지수 히스토리
        kosdaq_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        전체 피처 빌드 파이프라인.
        반환: 피처 행렬 + target (open_return_pct)
        """
        # 1. 기본 병합
        df = self._merge_base(dart_df, krx_df)
        if "offering_price_review_status" not in df.columns:
            df["offering_price_review_status"] = "needs_review_missing_audit"

        # 2. 핵심 파생 피처 계산
        df = self._calc_lockup_features(df)
        df = self._calc_band_position(df)
        df = self._calc_supply_structure_features(df)
        df = self._calc_offering_type_features(df)
        df = self._calc_same_day_ipo_count(df)
        df = self._calc_underwriter_tier(df)
        df = self._calc_risk_factor_count(df)
        df = self._calc_market_momentum(df, kospi_df, "kospi")
        if kosdaq_df is not None:
            df = self._calc_market_momentum(df, kosdaq_df, "kosdaq")
        # 3. 타깃 변수 계산 (시초가 / 상장일 종가 수익률)
        df = self._calc_target(df)
        df = self._calc_sector_ipo_temperature(df)
        df = self._calc_valuation_features(df)
        df = self._calc_financial_features(df)

        # 4. 피처만 선택
        available = [f for f in self.feature_names if f in df.columns]
        missing   = [f for f in self.feature_names if f not in df.columns]
        if missing:
            logger.warning("누락된 피처 %d개: %s", len(missing), missing[:5])
            for feature in missing:
                df[feature] = np.nan

        identity_columns = [
            "event_id", "corp_name", "listing_date", "event_class", "offering_type",
            "retail_subscription_eligibility_status", "retail_subscription_eligible",
            "industry_name", "listing_segment",
            "offering_price", "offering_price_review_status", "open_return_pct", "close_return_pct",
            "rcept_no", "corp_code", "feature_available_at", "event_source_url",
            "verification_status", "lineage_validation_status", "retail_source_url",
            "retail_validation_status", "retail_available_at", "retail_collected_at",
            "demand_rcept_no", "institutional_available_at", "lockup_available_at",
            "institutional_validation_status", "lockup_validation_status",
        ]
        for column in identity_columns:
            if column not in df.columns:
                df[column] = np.nan
        result = df[self.feature_names + identity_columns].copy()
        for feature in self.feature_names:
            result[f"{feature}__missing"] = result[feature].isna()
        result = result.sort_values("listing_date").reset_index(drop=True)
        logger.info("피처 빌드 완료: %d행 × %d피처", len(result), len(self.feature_names))
        return result

    # ── 병합 ──────────────────────────────────────────────────

    def _merge_base(self, dart_df: pd.DataFrame, krx_df: pd.DataFrame) -> pd.DataFrame:
        """DART 공시와 KRX 상장 실적을 원본을 바꾸지 않고 정합한다."""
        dart = dart_df.copy()
        krx = krx_df.copy()
        if "corp_name" not in dart.columns or "corp_name" not in krx.columns:
            raise ValueError("DART와 KRX 데이터에는 corp_name 컬럼이 필요합니다.")

        # 법인 표기, 공백, 문장부호 차이 때문에 이름이 달라도 같은 회사인
        # 사례를 줄인다. DART 쪽에는 아직 ticker가 없으므로 이름이 기본 키다.
        for frame in (dart, krx):
            frame["corp_name_clean"] = (
                frame["corp_name"].astype(str)
                .str.replace(r"(주식회사|㈜|\(주\)|\(株\))", "", regex=True)
                .str.replace(r"[\s·.()\-]", "", regex=True)
                .str.upper()
                .str.strip()
            )

        merged = pd.merge(
            dart, krx,
            on="corp_name_clean",
            how="inner",
            suffixes=("_dart", "_krx"),
        )

        # 동일 회사가 KOSDAQ 최초 상장 후 KOSPI로 이전 상장한 경우처럼,
        # 이름만 같은 다른 상장 이벤트가 붙는 일을 막는다. 수집 파이프라인이
        # 기록한 DART 연결 상장일이 있을 때만 엄격하게 비교해 기존 입력 호환성은
        # 유지한다.
        if {"listing_date_dart", "listing_date_krx"}.issubset(merged.columns):
            dart_listing = pd.to_datetime(merged["listing_date_dart"], errors="coerce")
            krx_listing = pd.to_datetime(merged["listing_date_krx"], errors="coerce")
            mismatched_listing = dart_listing.notna() & krx_listing.notna() & (dart_listing != krx_listing)
            if mismatched_listing.any():
                logger.info("DART 연결 상장일과 다른 KRX 이벤트 %d건을 제외했습니다.", int(mismatched_listing.sum()))
                merged = merged.loc[~mismatched_listing].copy()

        # 병합 뒤 suffix가 생긴 공통 컬럼을 후속 계산의 표준 이름으로
        # 되돌린다. KRX 상장일/가격을 우선하고, DART 공시 값은 보완용이다.
        canonical_sources = {
            "corp_name": ["corp_name_krx", "corp_name_dart"],
            "listing_date": ["listing_date_krx", "listing_date", "listing_date_dart"],
            # KRX 공식 공모가는 DART 원문 승인값을 교차검증하는 보조 원천이다.
            # 타깃 계산에는 감사 상태가 함께 있는 DART 값을 우선 사용한다.
            "offering_price": ["offering_price_dart", "offering_price", "offering_price_krx"],
            "ticker": ["ticker", "ticker_krx", "ticker_dart"],
            "event_id": ["event_id_krx", "event_id_dart", "event_id"],
            "event_class": ["event_class_krx", "event_class_dart", "event_class"],
            "offering_type": ["offering_type_krx", "offering_type_dart", "offering_type"],
            "retail_subscription_eligibility_status": [
                "retail_subscription_eligibility_status_krx",
                "retail_subscription_eligibility_status_dart",
                "retail_subscription_eligibility_status",
            ],
            "industry_name": ["industry_name_krx", "industry_name_dart", "industry_name"],
            "listing_segment": ["listing_segment_krx", "listing_segment_dart", "sector_krx", "sector"],
            "market": ["market", "market_krx", "market_dart"],
            "event_source_url": ["source_url", "source_url_krx", "event_source_url"],
            "verification_status": ["verification_status", "verification_status_krx"],
            "lineage_validation_status": ["lineage_validation_status", "lineage_validation_status_dart"],
            "retail_source_url": ["retail_source_url", "retail_source_url_dart"],
            "retail_validation_status": ["retail_validation_status", "retail_validation_status_dart"],
            "retail_available_at": ["retail_available_at", "retail_available_at_dart"],
            "retail_collected_at": ["retail_collected_at", "retail_collected_at_dart"],
            "demand_rcept_no": ["demand_rcept_no", "demand_rcept_no_dart"],
            "institutional_available_at": ["institutional_available_at", "institutional_available_at_dart"],
            "lockup_available_at": ["lockup_available_at", "lockup_available_at_dart"],
            "institutional_validation_status": [
                "institutional_validation_status", "institutional_validation_status_dart",
            ],
            "lockup_validation_status": ["lockup_validation_status", "lockup_validation_status_dart"],
            "same_day_ipo_count": ["same_day_ipo_count", "same_day_ipo_count_krx"],
            "open_price": ["open_price", "open_price_krx"],
            "close_price": ["close_price", "close_price_krx"],
        }
        for target, candidates in canonical_sources.items():
            existing = [col for col in candidates if col in merged.columns]
            if not existing:
                continue
            series = merged[existing[0]]
            for col in existing[1:]:
                series = series.combine_first(merged[col])
            merged[target] = series

        if "listing_date" in merged.columns:
            merged["listing_date"] = pd.to_datetime(merged["listing_date"], errors="coerce")
        logger.info("병합 결과: %d건 (DART %d × KRX %d)", len(merged), len(dart_df), len(krx_df))
        return merged

    def _calc_offering_type_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """공모 유형은 범주형 원문을 보존하고, 모델에는 독립 이진 피처로 넣는다.

        유형에 임의의 순서를 부여하는 숫자 코드 대신 이진 피처를 쓴다. 스팩
        여부는 DART 신고서 회사명으로 상장 전에 알 수 있어 모델에 쓸 수 있다.
        외국기업 분류는 KIND 사후 이벤트 마스터 기반이므로 감사/평가용으로만
        보존하고, 모델 프로필에는 넣지 않는다.
        """
        if "offering_type" not in df.columns:
            event_class = df.get("event_class", pd.Series(index=df.index, dtype=object))
            df["offering_type"] = event_class.map({
                "general_ipo": "common_stock_ipo",
                "spac_ipo": "spac_ipo",
                "foreign_listing": "foreign_common_stock_listing",
            }).fillna("review_required")
        offering_type = df["offering_type"].fillna("review_required").astype(str)
        pre_listing_name = df.get("corp_name", pd.Series("", index=df.index)).fillna("").astype(str)
        df["offering_type_spac_ipo"] = pre_listing_name.str.contains("스팩|SPAC", case=False, regex=True)
        df["offering_type_foreign_common_stock"] = offering_type.eq("foreign_common_stock_listing")
        return df

    # ── 확약 피처 ─────────────────────────────────────────────

    def _calc_lockup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """의무보유확약 가중 점수 계산"""
        for col in ["lockup_6m_ratio", "lockup_3m_ratio", "lockup_1m_ratio", "lockup_15d_ratio"]:
            if col not in df.columns:
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(0, 1)

        weighted = pd.concat([
            df["lockup_6m_ratio"] * 1.00,
            df["lockup_3m_ratio"] * 0.75,
            df["lockup_1m_ratio"] * 0.50,
            df["lockup_15d_ratio"] * 0.25,
        ], axis=1)
        df["lockup_components_missing"] = df[
            ["lockup_6m_ratio", "lockup_3m_ratio", "lockup_1m_ratio", "lockup_15d_ratio"]
        ].isna().any(axis=1)
        # 부분 기간만 찾은 경우를 완전한 확약 점수처럼 쓰지 않는다.
        df["lockup_weighted_score"] = weighted.sum(axis=1, min_count=4)
        return df

    # ── 공모가 밴드 위치 ───────────────────────────────────────

    def _calc_band_position(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        공모가 밴드 내 위치 계산.
        - 0.0  = 하단 확정
        - 0.5  = 중간
        - 1.0  = 상단 확정
        - >1.0 = 상단 초과 확정 (강한 수요 신호)
        """
        required = ["offering_price", "price_band_low", "price_band_high"]
        for col in required:
            if col not in df.columns:
                df["offering_price_band_position"] = np.nan
                df["band_exceeded"] = np.nan
                return df

        band_range = df["price_band_high"] - df["price_band_low"]

        valid_band = (
            (band_range > 0)
            & df["offering_price"].notna()
            & df["price_band_low"].notna()
            & df["price_band_high"].notna()
        )
        df["offering_price_band_position"] = np.where(
            valid_band,
            (df["offering_price"] - df["price_band_low"]) / band_range,
            np.nan,
        )
        df["offering_price_band_position"] = df["offering_price_band_position"].clip(-0.5, 2.0)
        df["band_exceeded"] = np.where(
            df["offering_price_band_position"].notna(),
            (df["offering_price_band_position"] > 1.0).astype(float),
            np.nan,
        )
        return df

    # ── 공모 구조 / 수급 피처 ─────────────────────────────────

    def _calc_supply_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """신주·구주·상장 후 주식수 기반 Phase 2 수급 피처 계산"""
        numeric_cols = [
            "new_shares",
            "secondary_shares",
            "public_float_shares",
            "total_post_listing_shares",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "secondary_offering_ratio" not in df.columns:
            new_shares = df.get("new_shares", pd.Series(np.nan, index=df.index))
            secondary_shares = df.get("secondary_shares", pd.Series(np.nan, index=df.index))
            offered_shares = new_shares + secondary_shares
            df["secondary_offering_ratio"] = np.where(
                new_shares.notna() & secondary_shares.notna() & (offered_shares > 0),
                secondary_shares / offered_shares,
                np.nan,
            )
        df["secondary_offering_ratio"] = pd.to_numeric(
            df["secondary_offering_ratio"], errors="coerce"
        ).clip(0, 1)

        if "float_share_ratio" not in df.columns:
            total_shares = df.get("total_post_listing_shares", pd.Series(np.nan, index=df.index))
            if "public_float_shares" in df.columns:
                float_shares = df["public_float_shares"]
            else:
                new_shares = df.get("new_shares", pd.Series(np.nan, index=df.index))
                secondary_shares = df.get("secondary_shares", pd.Series(np.nan, index=df.index))
                float_shares = new_shares + secondary_shares

            df["float_share_ratio"] = np.where(
                total_shares.notna() & float_shares.notna() & (total_shares > 0),
                float_shares / total_shares,
                np.nan,
            )
        df["float_share_ratio"] = pd.to_numeric(
            df["float_share_ratio"], errors="coerce"
        ).clip(0, 1)
        return df

    def _calc_same_day_ipo_count(self, df: pd.DataFrame) -> pd.DataFrame:
        """같은 상장일에 몰린 IPO 수를 계산해 수급 분산을 반영"""
        if "same_day_ipo_count" in df.columns:
            df["same_day_ipo_count"] = pd.to_numeric(
                df["same_day_ipo_count"], errors="coerce"
            ).clip(0, 20)
            return df

        if "listing_date" not in df.columns:
            df["same_day_ipo_count"] = np.nan
            return df

        listing_dates = pd.to_datetime(df["listing_date"], errors="coerce")
        df["same_day_ipo_count"] = listing_dates.map(listing_dates.value_counts()).astype("Float64")
        return df

    def _calc_underwriter_tier(self, df: pd.DataFrame) -> pd.DataFrame:
        """주관사명을 대형/중형/소형 등급으로 단순 매핑"""
        if "underwriter_tier" in df.columns:
            df["underwriter_tier"] = pd.to_numeric(
                df["underwriter_tier"], errors="coerce"
            ).clip(1, 3)
            return df

        name_col = next(
            (col for col in ["lead_underwriter", "underwriter", "underwriters", "manager"] if col in df.columns),
            None,
        )
        if name_col is None:
            df["underwriter_tier"] = np.nan
            return df

        df["underwriter_tier"] = df[name_col].map(self._map_underwriter_tier)
        return df

    @staticmethod
    def _map_underwriter_tier(name: object) -> float:
        if pd.isna(name):
            return np.nan
        normalized = re.sub(r"\s+", "", str(name))
        if any(tier_name.replace(" ", "") in normalized for tier_name in UNDERWRITER_TIER_1):
            return 1
        if any(tier_name.replace(" ", "") in normalized for tier_name in UNDERWRITER_TIER_2):
            return 2
        return 3

    def _calc_risk_factor_count(self, df: pd.DataFrame) -> pd.DataFrame:
        """위험요소 텍스트가 있으면 항목 마커 수를 세고, 이미 있으면 숫자로 정리"""
        if "risk_factor_count" in df.columns:
            df["risk_factor_count"] = pd.to_numeric(
                df["risk_factor_count"], errors="coerce"
            ).clip(0, 60)
            return df

        text_col = next(
            (col for col in ["risk_factor_text", "risk_factors", "prospectus_risk_text"] if col in df.columns),
            None,
        )
        if text_col is None:
            df["risk_factor_count"] = np.nan
            return df

        df["risk_factor_count"] = df[text_col].map(self._count_risk_items).clip(0, 60)
        return df

    @staticmethod
    def _count_risk_items(text: object) -> float:
        if pd.isna(text):
            return np.nan
        normalized = re.sub(r"\s+", " ", str(text))
        markers = re.findall(r"(?:^|\s)(?:[가-하]\.|[0-9]{1,2}\.|[①-⑳])", normalized)
        return float(len(markers)) if markers else np.nan

    # ── 시장 모멘텀 ───────────────────────────────────────────

    def _calc_market_momentum(
        self,
        df:       pd.DataFrame,
        index_df: pd.DataFrame,
        index_name: str = "kospi",
    ) -> pd.DataFrame:
        """
        상장일 기준 지수 N일 수익률 계산.
        index_df: date(datetime), close(float)
        """
        index_df = index_df.sort_values("date").reset_index(drop=True)
        closes = index_df.set_index("date")["close"]

        windows = FEATURE_CFG.momentum_windows  # [5, 20, 60]

        for window in windows:
            col_name = f"{index_name}_momentum_{window}d"
            returns = []

            for _, row in df.iterrows():
                listing_dt = pd.to_datetime(row.get("listing_date"))
                # 상장일 전날까지의 데이터에서 수익률 계산
                past = closes[closes.index < listing_dt]
                if len(past) > window:
                    ret = past.iloc[-1] / past.iloc[-1 - window] - 1
                else:
                    ret = np.nan
                returns.append(round(ret, 6))

            df[col_name] = returns
            logger.debug("%s 모멘텀 %dd 계산 완료", index_name.upper(), window)

        return df

    # ── 섹터 IPO 온도 ─────────────────────────────────────────

    def _calc_sector_ipo_temperature(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        각 종목 상장 시점 기준으로:
          - 동일 섹터 직전 N개 IPO 평균 수익률
          - 전체 직전 M개 IPO 평균 수익률
        시간 순서로 정렬 후 expanding window 적용 (미래 누수 방지).
        """
        if "open_return_pct" not in df.columns:
            # 타깃이 없으면 임시 NaN으로 채움 (추론 시)
            df["open_return_pct"] = np.nan

        df = df.sort_values("listing_date").reset_index(drop=True)

        n_sector = FEATURE_CFG.sector_lookback_n   # 5
        n_all    = 10

        sector_temps = []
        all_temps    = []

        for _, row in df.iterrows():
            # 같은 상장일의 다른 종목 수익률도 아직 장 마감 전에는 알 수 없으므로 제외한다.
            past = df.loc[df["listing_date"] < row["listing_date"]]

            # 전체 최근 N개
            past_valid = past["open_return_pct"].dropna()
            all_temp = past_valid.tail(n_all).mean() if len(past_valid) > 0 else np.nan
            all_temps.append(all_temp)

            # 섹터별
            if "industry_name" in df.columns:
                sect = row.get("industry_name")
                past_sect = past[past["industry_name"] == sect]["open_return_pct"].dropna()
                sect_temp = past_sect.tail(n_sector).mean() if len(past_sect) > 0 else all_temp
            else:
                sect_temp = all_temp
            sector_temps.append(sect_temp)

        df["recent_ipo_avg_return_sector"] = sector_temps
        df["recent_ipo_avg_return_all"]    = all_temps
        return df

    # ── 밸류에이션 피처 ───────────────────────────────────────

    def _calc_valuation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        PER 및 섹터 대비 PER 계산.
        EPS가 없는 종목(적자 등)은 결측으로 남긴다. 이 값의 보정은 학습
        분할 이후 훈련 데이터 통계로만 수행한다.
        """
        if "eps" not in df.columns or "offering_price" not in df.columns:
            df["offering_per"] = np.nan
            df["per_vs_sector_median"] = np.nan
            return df

        # 기본 PER
        df["eps"] = pd.to_numeric(df.get("eps"), errors="coerce")
        df["offering_per"] = np.where(
            (df["eps"].notna()) & (df["eps"] > 0),
            df["offering_price"] / df["eps"],
            np.nan,
        )
        df["offering_per"] = df["offering_per"].clip(0, 500)

        # 전체 기간 중앙값은 미래 IPO의 값을 과거 행에 섞는다. 동일 상장일도
        # 제외하고, 실제 산업 업종의 엄격히 이전 행만으로 중앙값을 만든다.
        df["per_vs_sector_median"] = np.nan
        if "industry_name" in df.columns:
            for index, row in df.iterrows():
                prior = df[
                    (df["listing_date"] < row["listing_date"])
                    & (df["industry_name"] == row["industry_name"])
                ]
                median = prior["offering_per"].dropna().median()
                if pd.notna(median) and median > 0 and pd.notna(row["offering_per"]):
                    df.at[index, "per_vs_sector_median"] = row["offering_per"] / median

        return df

    # ── 재무 피처 ─────────────────────────────────────────────

    def _calc_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """DART 재무제표 기반 성장성·수익성·안정성 피처 계산"""
        if "revenue_growth_3y" not in df.columns:
            if "revenue" in df.columns and "revenue_3y_ago" in df.columns:
                revenue = pd.to_numeric(df["revenue"], errors="coerce")
                revenue_3y_ago = pd.to_numeric(df["revenue_3y_ago"], errors="coerce")
                df["revenue_growth_3y"] = np.where(
                    (revenue > 0) & (revenue_3y_ago > 0),
                    (revenue / revenue_3y_ago) ** (1 / 3) - 1,
                    np.nan,
                )
            else:
                df["revenue_growth_3y"] = np.nan
        df["revenue_growth_3y"] = pd.to_numeric(
            df["revenue_growth_3y"], errors="coerce"
        ).clip(-0.5, 5.0)

        if "operating_margin" not in df.columns:
            if "operating_income" in df.columns and "revenue" in df.columns:
                operating_income = pd.to_numeric(df["operating_income"], errors="coerce")
                revenue = pd.to_numeric(df["revenue"], errors="coerce")
                df["operating_margin"] = np.where(revenue > 0, operating_income / revenue, np.nan)
            else:
                df["operating_margin"] = np.nan
        df["operating_margin"] = pd.to_numeric(
            df["operating_margin"], errors="coerce"
        ).clip(-1.0, 1.0)

        if "debt_ratio" not in df.columns:
            if "total_liabilities" in df.columns and "equity" in df.columns:
                liabilities = pd.to_numeric(df["total_liabilities"], errors="coerce")
                equity = pd.to_numeric(df["equity"], errors="coerce")
                df["debt_ratio"] = np.where(equity > 0, liabilities / equity, np.nan)
            else:
                df["debt_ratio"] = np.nan
        df["debt_ratio"] = pd.to_numeric(df["debt_ratio"], errors="coerce").clip(0, 10)
        return df

    # ── 타깃 계산 ─────────────────────────────────────────────

    def _calc_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        두 가지 상장일 타깃을 계산한다.
          - open_return_pct: (상장일 시초가 / 공모가 - 1) × 100
          - close_return_pct: (상장일 종가 / 공모가 - 1) × 100

        시초가는 09:00 개장 직후 형성된 가격이며, 종가는 상장 첫날 전체
        매매를 반영한다. 상장 전에는 두 값을 각각 예측하고, 장 마감 후
        실제값을 확정해 모델 성능을 기록한다.
        """
        if "offering_price" not in df.columns:
            df["open_return_pct"] = np.nan
            df["close_return_pct"] = np.nan
            return df

        df["offering_price"] = pd.to_numeric(df["offering_price"], errors="coerce")
        out_of_expected_range = (df["offering_price"] < 100) | (df["offering_price"] > 10_000_000)
        if out_of_expected_range.any():
            logger.warning(
                "예상 공모가 범위를 벗어난 값 %d건을 감사 로그에서 확인하세요. 값은 삭제하지 않습니다.",
                int(out_of_expected_range.sum()),
            )
        for price_col, target_col in [
            ("open_price", "open_return_pct"),
            ("close_price", "close_return_pct"),
        ]:
            if price_col not in df.columns:
                df[target_col] = np.nan
                continue
            df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
            df[target_col] = np.where(
                (df["offering_price"] > 0) & df[price_col].notna(),
                (df[price_col] / df["offering_price"] - 1) * 100,
                np.nan,
            )

        # 극단값 리포트 (제거하지 않고 로그만)
        extreme = df[df["open_return_pct"].abs() > 200]
        if len(extreme):
            names = extreme["corp_name"].tolist()[:5] if "corp_name" in extreme.columns else []
            logger.info("극단 수익률 종목 %d건 (±200%%+): %s", len(extreme),
                        names)

        return df

    # ── 결측값 처리 (fit / transform) ─────────────────────────

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        """학습 데이터에서 결측값 대체 통계 계산"""
        for feat_name in self.feature_names:
            if feat_name not in df.columns:
                self.fill_values[feat_name] = np.nan
                continue
            strategy = fill_na_strategy(feat_name)
            col = pd.to_numeric(df[feat_name], errors="coerce")
            if strategy == "mean":
                fill_value = col.mean()
            elif strategy == "median":
                fill_value = col.median()
            elif strategy in ("zero", "flag"):
                fill_value = 0.0
            else:
                fill_value = 0.0
            # 원시 피처가 전부 비어 있으면 임의의 0 또는 구간 중간값을 만들지
            # 않는다. 학습 적격성 검사에서 해당 데이터셋을 차단한다.
            self.fill_values[feat_name] = float(fill_value)
        self._fitted = True
        logger.info("FeatureEngineer fit 완료: %d개 피처 통계 계산", len(self.fill_values))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """피처 행렬 반환 (결측값 처리 포함)"""
        if not self._fitted:
            raise RuntimeError("fit()을 먼저 호출하세요.")

        result = pd.DataFrame(index=df.index)
        for feat_name in self.feature_names:
            if feat_name in df.columns:
                col = df[feat_name].copy()
            else:
                col = pd.Series([np.nan] * len(df), index=df.index)
            col = pd.to_numeric(col, errors="coerce")

            # 클리핑
            feat_def = FEATURE_MAP.get(feat_name)
            if feat_def and feat_def.clip:
                lo, hi = feat_def.clip
                col = col.clip(lo, hi)

            # 결측값 대체
            col = col.fillna(self.fill_values.get(feat_name, np.nan))
            if feat_def and feat_def.clip:
                lo, hi = feat_def.clip
                col = col.clip(lo, hi)
            result[feat_name] = col

        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def build_feature_observations(self, features: pd.DataFrame) -> pd.DataFrame:
        """원시 피처별 값·결측·출처·검증 정보를 long 형식으로 보존한다.

        ``features_all``은 모델 입력 표이고, 이 표는 감사와 사람 검토를 위한
        관측 원장이다. 결측은 수치 0으로 바꾸지 않고 원인과 원천을 함께 남긴다.
        """
        source_by_group = {
            FeatureGroup.MARKET: "KRX_KIND_or_KRX_OpenAPI",
            FeatureGroup.SUBSCRIPTION: "DART_or_official_underwriter_notice",
            FeatureGroup.SUPPLY: "DART_disclosure",
            FeatureGroup.IPO_STRUCTURE: "DART_disclosure",
            FeatureGroup.VALUATION: "DART_disclosure",
            FeatureGroup.FINANCIAL: "OpenDART_financial_statement",
        }
        records: list[dict[str, object]] = []
        for row in features.itertuples(index=False):
            values = row._asdict()
            for feature_name in self.feature_names:
                if feature_name not in features.columns:
                    continue
                value = values.get(feature_name)
                missing = pd.isna(value)
                feature = FEATURE_MAP[feature_name]
                source = source_by_group[feature.group]
                if feature_name == "retail_subscription_ratio":
                    retail_status = values.get("retail_validation_status")
                    missing_reason = (
                        str(retail_status)
                        if missing and pd.notna(retail_status) and str(retail_status).strip()
                        else "official_underwriter_notice_not_collected" if missing else None
                    )
                elif feature.group == FeatureGroup.FINANCIAL and missing:
                    missing_reason = "financial_publication_time_unverified"
                elif missing:
                    missing_reason = "official_source_field_unavailable_or_unverified"
                else:
                    missing_reason = None
                source_ref = values.get("rcept_no")
                available_at = values.get("feature_available_at")
                if feature_name == "offering_type_spac_ipo":
                    source = "DART_disclosure"
                elif feature_name == "offering_type_foreign_common_stock":
                    # KIND의 최종 상장 분류는 과거 유형별 평가에는 쓰되, 상장 전
                    # 입력으로 쓸 수 없으므로 공개시각을 꾸며내지 않는다.
                    source = "KRX_KIND_post_listing_event_classification"
                    source_ref = values.get("event_source_url")
                    available_at = pd.NA
                if feature_name == "institutional_demand_ratio":
                    source_ref = values.get("demand_rcept_no")
                    available_at = values.get("institutional_available_at")
                if feature_name.startswith("lockup_"):
                    source_ref = values.get("demand_rcept_no")
                    available_at = values.get("lockup_available_at")
                if feature_name == "retail_subscription_ratio" and pd.notna(values.get("retail_source_url")):
                    source_ref = values.get("retail_source_url")
                    available_at = values.get("retail_available_at")
                if source.startswith("KRX"):
                    event_source_url = values.get("event_source_url")
                    source_ref = (
                        event_source_url if pd.notna(event_source_url)
                        else "KRX_KIND_new_listing_company"
                    )
                validation = values.get("offering_price_review_status")
                if feature_name == "retail_subscription_ratio":
                    validation = values.get("retail_validation_status")
                elif feature_name == "institutional_demand_ratio":
                    validation = values.get("institutional_validation_status")
                elif feature_name.startswith("lockup_"):
                    validation = values.get("lockup_validation_status")
                if pd.isna(validation) or str(validation).strip() == "":
                    validation = values.get("verification_status")
                if pd.isna(validation) or str(validation).strip() == "":
                    validation = values.get("lineage_validation_status")
                if pd.isna(validation) or str(validation).strip() == "":
                    validation = "needs_review"
                validation = str(validation)
                deferred_retail_feature = (
                    feature_name == "retail_subscription_ratio"
                    and validation == "retail_feature_deferred_not_collected"
                )
                records.append({
                    "event_id": values.get("event_id"),
                    "corp_name": values.get("corp_name"),
                    "listing_date": values.get("listing_date"),
                    "feature_name": feature_name,
                    "raw_value": value,
                    "is_missing": bool(missing),
                    "missing_reason": missing_reason,
                    "source": source,
                    "source_reference": source_ref,
                    "available_at": available_at,
                    "collected_at": pd.Timestamp.now(tz="Asia/Seoul"),
                    "validation_status": validation,
                    "human_review_required": bool(not deferred_retail_feature and (missing or validation not in {
                        "verified_currency_unit", "verified_text_and_structured",
                        "verified_structured_api", "manual_verified",
                        "official_source_krx_code_enriched", "official_source_collected",
                        "official_dart_issuer_total_retail_ratio", "official_dart_and_notice_match",
                    })),
                })
        return pd.DataFrame(records)

    # ── 피처 요약 ─────────────────────────────────────────────

    def feature_stats(self, X: pd.DataFrame) -> pd.DataFrame:
        """피처별 기술 통계 (데이터 품질 체크용)"""
        stats = X.describe().T
        stats["missing_pct"] = (X.isna().sum() / len(X) * 100).round(2)
        stats["zero_pct"]    = ((X == 0).sum() / len(X) * 100).round(2)
        return stats


def build_demo_dataset(n: int = 200, seed: int = 42, phase: str = "core") -> pd.DataFrame:
    """
    실제 데이터 없이 모델 개발/테스트용 가상 데이터셋 생성.
    실제 한국 IPO 통계에서 추정한 분포 파라미터를 사용한다.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=n, freq="7D")

    # 핵심 피처 시뮬레이션 (실제 분포 근사)
    lockup_6m  = rng.beta(2, 5, n).clip(0, 1)   # 우편향, 대부분 낮음
    lockup_3m  = rng.beta(2, 4, n).clip(0, 1)
    lockup_1m  = rng.beta(3, 4, n).clip(0, 1)
    lockup_15d = rng.beta(2, 3, n).clip(0, 1)

    institutional_demand = rng.lognormal(5.5, 1.2, n).clip(1, 3000)
    retail_demand        = rng.lognormal(6.0, 1.5, n).clip(1, 5000)
    band_position        = rng.uniform(-0.1, 1.3, n)

    kospi_5d  = rng.normal(0.005, 0.025, n)
    kospi_20d = rng.normal(0.010, 0.050, n)
    same_day_ipo_count = rng.choice([1, 1, 1, 2, 2, 3], size=n, p=[0.45, 0.25, 0.15, 0.08, 0.05, 0.02])
    float_share_ratio = rng.beta(2.5, 6.0, n).clip(0.03, 0.85)
    secondary_offering_ratio = rng.beta(1.2, 5.5, n).clip(0, 0.9)
    major_lockup_months = rng.choice([6, 12, 18, 24, 30, 36], size=n, p=[0.12, 0.28, 0.14, 0.28, 0.08, 0.10])
    risk_factor_count = rng.poisson(14, n).clip(2, 45)
    underwriter_tier = rng.choice([1, 2, 3], size=n, p=[0.55, 0.32, 0.13])
    offering_per = rng.lognormal(3.0, 0.55, n).clip(3, 220)
    per_vs_sector = rng.lognormal(0.0, 0.35, n).clip(0.2, 4.0)
    revenue_growth = rng.normal(0.25, 0.45, n).clip(-0.5, 3.0)
    operating_margin = rng.normal(0.08, 0.18, n).clip(-0.6, 0.8)
    debt_ratio = rng.lognormal(0.0, 0.55, n).clip(0, 6)

    lockup_score = lockup_6m * 1.0 + lockup_3m * 0.75 + lockup_1m * 0.5 + lockup_15d * 0.25

    # 타깃 생성: 실제 관계를 반영한 수익률 시뮬레이션
    # 한국 IPO 실제 통계: 약 65% 양수, 35% 음수 or 0
    signal = (
        lockup_score          * 30  +
        np.log1p(institutional_demand) * 3  +
        band_position         * 15  +
        kospi_20d             * 60  +
        (1 - float_share_ratio) * 12 +
        (1 - secondary_offering_ratio) * 8 +
        (major_lockup_months / 36) * 7 +
        (underwriter_tier == 1) * 5 -
        risk_factor_count * 0.25 -
        np.maximum(per_vs_sector - 1, 0) * 4 +
        np.maximum(revenue_growth, 0) * 4 +
        operating_margin * 10 -
        debt_ratio * 1.2 +
        rng.normal(0, 25, n)         # 충분한 노이즈로 음수 포함
    )
    # 중앙값을 0 근처로 이동시켜 양수/음수 혼재 보장
    signal = signal - signal.mean() + 12.0   # 평균 +12% (실제 통계 근사)
    open_return_pct = signal.clip(-50, 300)
    intraday_move = (
        lockup_score * 4 +
        kospi_5d * 30 -
        secondary_offering_ratio * 3 +
        rng.normal(0, 10, n)
    )
    close_return_pct = (open_return_pct + intraday_move).clip(-60, 300)

    df = pd.DataFrame({
        "corp_name":                    [f"종목_{i:04d}" for i in range(n)],
        "listing_date":                 dates,
        "offering_price":               rng.integers(5000, 100000, n),
        "lockup_6m_ratio":              lockup_6m,
        "lockup_3m_ratio":              lockup_3m,
        "lockup_1m_ratio":              lockup_1m,
        "lockup_15d_ratio":             lockup_15d,
        "lockup_weighted_score":        lockup_score,
        "institutional_demand_ratio":   institutional_demand,
        "retail_subscription_ratio":    retail_demand,
        "offering_price_band_position": band_position,
        "band_exceeded":                (band_position > 1.0).astype(int),
        "kospi_momentum_5d":            kospi_5d,
        "kospi_momentum_20d":           kospi_20d,
        "recent_ipo_avg_return_sector": rng.normal(15, 20, n),
        "recent_ipo_avg_return_all":    rng.normal(12, 18, n),
        "open_return_pct":              open_return_pct,
        "close_return_pct":             close_return_pct,
    })
    if phase in ("phase2", "all"):
        df = df.assign(
            float_share_ratio=float_share_ratio,
            secondary_offering_ratio=secondary_offering_ratio,
            major_shareholder_lockup_months=major_lockup_months,
            same_day_ipo_count=same_day_ipo_count,
            risk_factor_count=risk_factor_count,
            underwriter_tier=underwriter_tier,
            offering_per=offering_per,
            per_vs_sector_median=per_vs_sector,
            revenue_growth_3y=revenue_growth,
            operating_margin=operating_margin,
            debt_ratio=debt_ratio,
        )
        # 데모 데이터도 실제 파이프라인과 같은 범주형 입력 계약을 따른다.
        # 기준 범주인 일반 보통주 IPO 외에 일부 스팩·외국기업 플래그를 섞는다.
        offering_type = rng.choice(
            ["common_stock_ipo", "spac_ipo", "foreign_common_stock_listing"],
            size=n,
            p=[0.80, 0.15, 0.05],
        )
        df["offering_type"] = offering_type
        df["offering_type_spac_ipo"] = (offering_type == "spac_ipo").astype(int)
        df["offering_type_foreign_common_stock"] = (
            offering_type == "foreign_common_stock_listing"
        ).astype(int)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = build_demo_dataset(300)
    print(df.describe())
    print(f"\n타깃 분포:")
    print(f"  평균:   {df['open_return_pct'].mean():.1f}%")
    print(f"  중앙값: {df['open_return_pct'].median():.1f}%")
    print(f"  std:    {df['open_return_pct'].std():.1f}%")
    print(f"  양수 비율: {(df['open_return_pct'] > 0).mean() * 100:.1f}%")
