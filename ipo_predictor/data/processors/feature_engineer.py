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

        result = df[available + [
            "corp_name", "listing_date", "offering_price",
            "offering_price_review_status",
            "open_return_pct", "close_return_pct",
        ]].copy()
        result = result.sort_values("listing_date").reset_index(drop=True)
        logger.info("피처 빌드 완료: %d행 × %d피처", len(result), len(available))
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

        # 병합 뒤 suffix가 생긴 공통 컬럼을 후속 계산의 표준 이름으로
        # 되돌린다. KRX 상장일/가격을 우선하고, DART 공시 값은 보완용이다.
        canonical_sources = {
            "corp_name": ["corp_name_krx", "corp_name_dart"],
            "listing_date": ["listing_date_krx", "listing_date", "listing_date_dart"],
            "ticker": ["ticker", "ticker_krx", "ticker_dart"],
            "sector_name": ["sector_name", "sector_krx", "sector", "sector_dart"],
            "market": ["market", "market_krx", "market_dart"],
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

    # ── 확약 피처 ─────────────────────────────────────────────

    def _calc_lockup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """의무보유확약 가중 점수 계산"""
        for col in ["lockup_6m_ratio", "lockup_3m_ratio", "lockup_1m_ratio", "lockup_15d_ratio"]:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).clip(0, 1)

        df["lockup_weighted_score"] = (
            df["lockup_6m_ratio"]  * 1.00 +
            df["lockup_3m_ratio"]  * 0.75 +
            df["lockup_1m_ratio"]  * 0.50 +
            df["lockup_15d_ratio"] * 0.25
        )
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
                df["band_exceeded"] = False
                return df

        band_range = df["price_band_high"] - df["price_band_low"]

        df["offering_price_band_position"] = np.where(
            band_range > 0,
            (df["offering_price"] - df["price_band_low"]) / band_range,
            0.5  # 밴드 정보 없으면 중간값
        )
        df["offering_price_band_position"] = df["offering_price_band_position"].clip(-0.5, 2.0)
        df["band_exceeded"] = (df["offering_price_band_position"] > 1.0).astype(int)
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
            offered_shares = new_shares.fillna(0) + secondary_shares.fillna(0)
            df["secondary_offering_ratio"] = np.where(
                offered_shares > 0,
                secondary_shares.fillna(0) / offered_shares,
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
                float_shares = new_shares.fillna(0) + secondary_shares.fillna(0)

            df["float_share_ratio"] = np.where(
                total_shares > 0,
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
        df["same_day_ipo_count"] = listing_dates.map(listing_dates.value_counts()).fillna(0).astype(int)
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
                    ret = 0.0
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

        for i, row in df.iterrows():
            past = df.loc[:i-1]  # 현재 행 이전 데이터만 사용

            # 전체 최근 N개
            past_valid = past["open_return_pct"].dropna()
            all_temp = past_valid.tail(n_all).mean() if len(past_valid) > 0 else np.nan
            all_temps.append(all_temp)

            # 섹터별
            if "sector_name" in df.columns:
                sect = row.get("sector_name")
                past_sect = past[past["sector_name"] == sect]["open_return_pct"].dropna()
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
        EPS가 없는 종목(적자 등)은 섹터 중앙값으로 대체.
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

        # 섹터별 PER 중앙값 대비 (적자 기업 패널티 반영)
        if "sector_name" in df.columns:
            sector_median_per = df.groupby("sector_name")["offering_per"].transform("median")
            df["per_vs_sector_median"] = np.where(
                sector_median_per > 0,
                df["offering_per"] / sector_median_per,
                np.nan,
            )
        else:
            overall_median = df["offering_per"].median()
            df["per_vs_sector_median"] = df["offering_per"] / overall_median if overall_median else np.nan

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
                self.fill_values[feat_name] = 0.0
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
            if pd.isna(fill_value):
                feat_def = FEATURE_MAP.get(feat_name)
                if strategy == "median" and feat_def and feat_def.clip and feat_def.clip[0] > 0:
                    lo, hi = feat_def.clip
                    fill_value = (lo + hi) / 2
                else:
                    fill_value = 0.0
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
            col = col.fillna(self.fill_values.get(feat_name, 0.0))
            if feat_def and feat_def.clip:
                lo, hi = feat_def.clip
                col = col.clip(lo, hi)
            result[feat_name] = col

        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

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
