"""
models/evaluation/outlier_analyzer.py
───────────────────────────────────────
에러 주도 피처 엔지니어링의 핵심 모듈.

백테스트 예측 오차가 ±2σ를 초과하는 아웃라이어 종목을 수집하고,
그 원인을 자동으로 분류한 뒤 다음 피처 추가 방향을 제안한다.

원인 유형:
  A. 시장 국면 오류  — 오차 발생 시점이 KOSPI 급락 구간과 겹침
  B. 수급 이벤트    — 동시 상장 급증, 유통 물량 급변
  C. 섹터 쏠림      — 특정 섹터에 오차가 집중
  D. 구조적 이슈    — 구주매출 비율, 확약 파싱 오류 등
  E. 설명 불가      — 위 어디에도 해당 없음 (모델 한계)

분석 결과는 reports/ 폴더에 저장되고,
추가할 피처 후보를 features/definitions.py의 OPTIONAL 목록과 대조한다.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import BACKTEST_CFG, REPORT_DIR

logger = logging.getLogger(__name__)


# ── 원인 유형 정의 ────────────────────────────────────────────

CAUSE_TYPES = {
    "A_market":   "시장 국면 오류 (KOSPI 급락 구간)",
    "B_supply":   "수급 이벤트 (동시 상장 / 유통물량)",
    "C_sector":   "섹터 쏠림",
    "D_structure":"구조적 이슈 (구주매출 / 데이터 품질)",
    "E_unknown":  "설명 불가 (모델 한계)",
}

# KOSPI 급락 판단 기준
BEAR_KOSPI_20D_THRESHOLD = -0.08   # 20일 수익률 -8% 이하


@dataclass
class OutlierRecord:
    """단일 아웃라이어 종목 레코드"""
    corp_name:    str
    listing_date: str
    actual:       float
    pred:         float
    error:        float        # actual - pred
    abs_error:    float
    sigma:        float        # 오차가 몇 σ인지
    cause:        str          # CAUSE_TYPES 키
    cause_detail: str          # 구체적 근거
    feature_hint: list[str]    # 추가 검토 피처 후보


@dataclass
class OutlierReport:
    """전체 아웃라이어 분석 결과"""
    n_total:      int
    n_outliers:   int
    outlier_rate: float
    sigma_threshold: float
    records:      list[OutlierRecord] = field(default_factory=list)
    cause_summary: dict = field(default_factory=dict)
    feature_recommendations: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "═══ 아웃라이어 분석 보고서 ═══",
            f"  전체 예측:    {self.n_total}건",
            f"  아웃라이어:   {self.n_outliers}건 ({self.outlier_rate*100:.1f}%)",
            f"  기준 (σ):     ±{self.sigma_threshold}",
            "",
            "  원인 분포:",
        ]
        for cause, cnt in sorted(self.cause_summary.items(), key=lambda x: -x[1]):
            label = CAUSE_TYPES.get(cause, cause)
            lines.append(f"    {label}: {cnt}건")
        if self.feature_recommendations:
            lines += ["", "  추가 피처 후보:"]
            for f in self.feature_recommendations:
                lines.append(f"    → {f}")
        return "\n".join(lines)


class OutlierAnalyzer:
    """
    백테스트 결과에서 아웃라이어를 추출하고 원인을 분류한다.

    사용법:
        analyzer = OutlierAnalyzer()
        report = analyzer.analyze(backtest_result.predictions, kospi_df)
        print(report.summary())
    """

    def __init__(
        self,
        sigma_threshold: float = BACKTEST_CFG.outlier_sigma,
        min_abs_error:   float = BACKTEST_CFG.outlier_min_error_pct,
    ):
        self.sigma_threshold = sigma_threshold
        self.min_abs_error   = min_abs_error

    # ── 메인 분석 ─────────────────────────────────────────────

    def analyze(
        self,
        predictions: pd.DataFrame,
        kospi_df:    Optional[pd.DataFrame] = None,
    ) -> OutlierReport:
        """
        백테스트 예측 DataFrame을 받아 아웃라이어를 분석한다.

        predictions 필수 컬럼:
          corp_name, listing_date, actual, pred, error, abs_error
        kospi_df (선택):
          date, close — 시장 국면 감지용
        """
        df = predictions.copy()
        df["listing_date"] = pd.to_datetime(df["listing_date"])

        # 오차 통계
        error_std  = df["abs_error"].std()
        error_mean = df["abs_error"].mean()

        # 아웃라이어 추출: ±2σ AND 최소 절대 오차 이상
        df["error_sigma"] = df["abs_error"] / error_std if error_std > 0 else 0
        outlier_mask = (
            (df["error_sigma"] >= self.sigma_threshold) &
            (df["abs_error"]   >= self.min_abs_error)
        )
        outliers = df[outlier_mask].copy()

        logger.info(
            "아웃라이어 추출: %d / %d건 (σ≥%.1f, 절대오차≥%.0f%%)",
            len(outliers), len(df), self.sigma_threshold, self.min_abs_error
        )

        # 시장 모멘텀 붙이기 (KOSPI 데이터 있을 때)
        if kospi_df is not None:
            outliers = self._attach_market_context(outliers, kospi_df)

        # 원인 분류
        records = []
        for _, row in outliers.iterrows():
            record = self._classify_cause(row)
            records.append(record)

        # 원인 집계
        from collections import Counter
        cause_cnt = Counter(r.cause for r in records)

        # 피처 추천
        recommendations = self._recommend_features(cause_cnt, records)

        report = OutlierReport(
            n_total          = len(df),
            n_outliers       = len(outliers),
            outlier_rate     = len(outliers) / len(df) if len(df) else 0,
            sigma_threshold  = self.sigma_threshold,
            records          = records,
            cause_summary    = dict(cause_cnt),
            feature_recommendations = recommendations,
        )

        self._save_report(report, outliers)
        return report

    # ── 시장 컨텍스트 부착 ────────────────────────────────────

    def _attach_market_context(
        self,
        outliers: pd.DataFrame,
        kospi_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """각 아웃라이어에 상장일 기준 KOSPI 모멘텀을 추가"""
        kospi = kospi_df.copy()
        kospi["date"] = pd.to_datetime(kospi["date"])
        kospi = kospi.sort_values("date").set_index("date")["close"]

        mom_5d  = []
        mom_20d = []
        mom_60d = []

        for _, row in outliers.iterrows():
            dt   = row["listing_date"]
            past = kospi[kospi.index < dt]

            def _ret(window):
                return (past.iloc[-1] / past.iloc[-1-window] - 1) if len(past) > window else 0.0

            mom_5d.append(round(_ret(5),  6))
            mom_20d.append(round(_ret(20), 6))
            mom_60d.append(round(_ret(60), 6))

        outliers = outliers.copy()
        outliers["kospi_mom_5d"]  = mom_5d
        outliers["kospi_mom_20d"] = mom_20d
        outliers["kospi_mom_60d"] = mom_60d
        return outliers

    # ── 원인 분류 ─────────────────────────────────────────────

    def _classify_cause(self, row: pd.Series) -> OutlierRecord:
        """
        단일 아웃라이어 종목의 원인 분류.
        우선순위: A(시장) → B(수급) → C(섹터) → D(구조) → E(불명)
        """
        cause        = "E_unknown"
        cause_detail = "명확한 패턴 없음"
        hints        = []

        # ── A: 시장 국면 ────────────────────────────────────
        kospi_20d = row.get("kospi_mom_20d", 0)
        kospi_5d  = row.get("kospi_mom_5d",  0)

        if kospi_20d < BEAR_KOSPI_20D_THRESHOLD:
            cause        = "A_market"
            cause_detail = f"KOSPI 20일 수익률 {kospi_20d*100:.1f}% (약세장 구간)"
            hints        = ["kospi_momentum_60d", "bear_market_flag", "volatility_index"]

        elif kospi_5d < -0.04:
            cause        = "A_market"
            cause_detail = f"상장 직전 KOSPI 급락 {kospi_5d*100:.1f}%"
            hints        = ["kospi_momentum_5d_extreme_flag", "market_stress_flag"]

        # ── B: 수급 이벤트 ───────────────────────────────────
        elif row.get("same_day_ipo_count", 0) >= 4:
            cause        = "B_supply"
            cause_detail = f"동일일 상장 {int(row['same_day_ipo_count'])}개 (수급 분산)"
            hints        = ["same_day_ipo_count", "same_day_total_offer_size"]

        elif row.get("float_share_ratio", 0) > 0.6:
            cause        = "B_supply"
            cause_detail = f"유통 물량 과다 ({row['float_share_ratio']*100:.0f}%)"
            hints        = ["float_share_ratio", "secondary_offering_ratio"]

        # ── C: 섹터 쏠림 ────────────────────────────────────
        elif abs(row.get("recent_ipo_avg_return_sector", 0)) > 50:
            cause        = "C_sector"
            cause_detail = f"섹터 평균 수익률 이상 ({row.get('recent_ipo_avg_return_sector', 0):.1f}%)"
            hints        = ["sector_heat_index", "recent_ipo_avg_return_sector"]

        # ── D: 구조적 이슈 ──────────────────────────────────
        elif row.get("secondary_offering_ratio", 0) > 0.5:
            cause        = "D_structure"
            cause_detail = f"구주매출 과다 ({row.get('secondary_offering_ratio', 0)*100:.0f}%)"
            hints        = ["secondary_offering_ratio", "vc_exit_flag"]

        # 예측 방향이 완전히 반대인 경우 추가 플래그
        if row["error"] * row["actual"] < 0:
            cause_detail += " | 방향 예측 오류"

        return OutlierRecord(
            corp_name    = str(row.get("corp_name", "unknown")),
            listing_date = str(row.get("listing_date", ""))[:10],
            actual       = round(float(row["actual"]), 2),
            pred         = round(float(row["pred"]), 2),
            error        = round(float(row["error"]), 2),
            abs_error    = round(float(row["abs_error"]), 2),
            sigma        = round(float(row["error_sigma"]), 2),
            cause        = cause,
            cause_detail = cause_detail,
            feature_hint = hints,
        )

    # ── 피처 추천 ─────────────────────────────────────────────

    def _recommend_features(
        self,
        cause_cnt: dict,
        records:   list[OutlierRecord],
    ) -> list[str]:
        """
        원인 분포를 보고 다음 단계에서 추가할 피처를 추천한다.
        전체 아웃라이어 중 30% 이상을 차지하는 원인에 대해서만 추천.
        """
        total = sum(cause_cnt.values())
        recommendations = []

        for cause, cnt in cause_cnt.items():
            ratio = cnt / total if total else 0
            if ratio < 0.30:
                continue

            if cause == "A_market":
                recommendations += [
                    "kospi_momentum_60d (60일 장기 모멘텀 — 약세장 국면 감지)",
                    "bear_market_flag (KOSPI 20일 수익률 < -8% 이진 플래그)",
                    "market_volatility_20d (20일 변동성 — 불확실성 환경 반영)",
                ]
            elif cause == "B_supply":
                recommendations += [
                    "same_day_ipo_count (동일 상장일 종목 수)",
                    "same_day_total_market_cap (동일일 공모 총 시총 — 수급 희소성)",
                    "float_share_ratio (유통 물량 비율 — Phase 2 우선 추가)",
                ]
            elif cause == "C_sector":
                recommendations += [
                    "sector_heat_index (섹터 최근 30일 IPO 평균 수익률)",
                    "sector_ipo_frequency (섹터 IPO 빈도 — 과열 감지)",
                ]
            elif cause == "D_structure":
                recommendations += [
                    "secondary_offering_ratio (구주매출 비율 — Phase 2 우선 추가)",
                    "vc_backed_flag (VC 투자 기업 여부)",
                    "audit_firm_tier (회계법인 등급)",
                ]

        # 중복 제거 및 정렬
        seen = set()
        result = []
        for r in recommendations:
            key = r.split(" ")[0]
            if key not in seen:
                seen.add(key)
                result.append(r)

        return result

    # ── 보고서 저장 ───────────────────────────────────────────

    def _save_report(self, report: OutlierReport, outliers: pd.DataFrame):
        """보고서를 CSV + 텍스트로 저장"""
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

        # 아웃라이어 CSV
        csv_path = REPORT_DIR / f"outliers_{ts}.csv"
        rows = [{
            "corp_name":    r.corp_name,
            "listing_date": r.listing_date,
            "actual":       r.actual,
            "pred":         r.pred,
            "error":        r.error,
            "sigma":        r.sigma,
            "cause":        r.cause,
            "cause_detail": r.cause_detail,
            "feature_hint": "|".join(r.feature_hint),
        } for r in report.records]
        pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

        # 텍스트 요약
        txt_path = REPORT_DIR / f"outlier_report_{ts}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(report.summary())
            f.write("\n\n── 아웃라이어 상세 ─────────────────────\n")
            for r in report.records:
                f.write(
                    f"\n[{r.listing_date}] {r.corp_name}\n"
                    f"  실제: {r.actual:+.1f}%  예측: {r.pred:+.1f}%  오차: {r.error:+.1f}% ({r.sigma:.1f}σ)\n"
                    f"  원인: {CAUSE_TYPES.get(r.cause, r.cause)}\n"
                    f"  근거: {r.cause_detail}\n"
                    f"  피처 힌트: {', '.join(r.feature_hint) or '없음'}\n"
                )

        logger.info("보고서 저장: %s", txt_path)

    # ── 시각화 데이터 ─────────────────────────────────────────

    def error_distribution(self, predictions: pd.DataFrame) -> dict:
        """
        오차 분포 통계 (앱/대시보드용 시각화 데이터).
        """
        errors = predictions["abs_error"].dropna()
        sigma  = errors.std()

        return {
            "mean":          round(errors.mean(), 2),
            "median":        round(errors.median(), 2),
            "std":           round(sigma, 2),
            "p90":           round(errors.quantile(0.90), 2),
            "within_1sigma": round((errors < sigma).mean() * 100, 1),
            "within_2sigma": round((errors < 2 * sigma).mean() * 100, 1),
            "n_outliers":    int((errors >= 2 * sigma).sum()),
            "buckets": {
                "0-5%":    int((errors < 5).sum()),
                "5-10%":   int(((errors >= 5)  & (errors < 10)).sum()),
                "10-20%":  int(((errors >= 10) & (errors < 20)).sum()),
                "20-30%":  int(((errors >= 20) & (errors < 30)).sum()),
                "30%+":    int((errors >= 30).sum()),
            },
        }

    def temporal_error_pattern(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """
        시간대별 오차 집중 여부 확인.
        특정 연도/분기에 오차가 몰려 있으면 시장 국면 원인 가능성 높음.
        """
        df = predictions.copy()
        df["listing_date"] = pd.to_datetime(df["listing_date"])
        df["year"]    = df["listing_date"].dt.year
        df["quarter"] = df["listing_date"].dt.quarter

        pattern = (
            df.groupby(["year", "quarter"])
            .agg(
                n=("abs_error", "count"),
                mae=("abs_error", "mean"),
                direction_acc=(
                    "error",
                    lambda e: ((e + df.loc[e.index, "actual"]) * df.loc[e.index, "actual"] > 0).mean()
                    if len(e) else np.nan
                ),
            )
            .reset_index()
        )
        return pattern


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parents[2]))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from models.evaluation.backtester import run_quick_backtest

    result = run_quick_backtest(n_samples=600)
    analyzer = OutlierAnalyzer()
    report = analyzer.analyze(result.predictions)

    print(report.summary())
    print(f"\n오차 분포:")
    import json
    print(json.dumps(analyzer.error_distribution(result.predictions), ensure_ascii=False, indent=2))
