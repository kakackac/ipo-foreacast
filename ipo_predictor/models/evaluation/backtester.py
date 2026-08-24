"""
models/evaluation/backtester.py
─────────────────────────────────
Walk-forward 백테스트 엔진.

일반 k-fold 대신 시간 기반 슬라이딩 윈도우를 사용해
미래 데이터 누수를 완전히 차단한다.

핵심 원칙:
  - 항상 과거 데이터로만 학습 → 미래 데이터로 예측
  - 각 윈도우의 예측 결과를 누적해 전체 성능 산출
  - 결과 DataFrame에 모든 예측 + 실제 + 날짜를 보존
    → 아웃라이어 분석기(outlier_analyzer.py)로 전달
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from config import BACKTEST_CFG, BacktestConfig
from models.baseline.gradient_boost_model import IPOPriceModel

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """백테스트 전체 결과 컨테이너"""

    # 예측 vs 실제 전체 레코드
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)

    # 윈도우별 메트릭
    window_metrics: list[dict] = field(default_factory=list)

    # 집계 메트릭
    overall_mae:           float = 0.0
    overall_direction_acc: float = 0.0
    coverage_90:           float = 0.0
    n_windows:             int   = 0

    def summary(self) -> str:
        lines = [
            "═══ Walk-Forward 백테스트 결과 ═══",
            f"  윈도우 수:       {self.n_windows}",
            f"  총 예측 건수:    {len(self.predictions)}",
            f"  전체 MAE:        {self.overall_mae:.2f}%",
            f"  방향 정확도:     {self.overall_direction_acc*100:.1f}%",
            f"  90% CI 커버리지: {self.coverage_90*100:.1f}%",
        ]
        return "\n".join(lines)

    def by_year(self) -> pd.DataFrame:
        """연도별 성능 분해"""
        df = self.predictions.copy()
        df["year"] = pd.to_datetime(df["listing_date"]).dt.year
        grouped = df.groupby("year").apply(
            lambda g: pd.Series({
                "n":            len(g),
                "mae":          (g["actual"] - g["pred"]).abs().mean(),
                "direction_acc": ((g["actual"] > 0) == (g["pred"] > 0)).mean(),
                "actual_mean":  g["actual"].mean(),
                "pred_mean":    g["pred"].mean(),
            })
        ).reset_index()
        return grouped


class WalkForwardBacktester:
    """
    Walk-Forward 백테스트 실행기.

    사용법:
        bt = WalkForwardBacktester(cfg=BACKTEST_CFG)
        result = bt.run(df, feature_cols)
        print(result.summary())
        result.predictions → 아웃라이어 분석기에 전달
    """

    def __init__(self, cfg: BacktestConfig = BACKTEST_CFG):
        self.cfg = cfg

    def run(
        self,
        df:           pd.DataFrame,   # 전체 피처 + 타깃 포함
        feature_cols: list[str],
    target_col:   str = "open_return_pct",  # open_return_pct | close_return_pct
        date_col:     str = "listing_date",
        model_kwargs: Optional[dict] = None,
    ) -> BacktestResult:
        """
        전체 Walk-Forward 백테스트 실행.

        Args:
            df:           listing_date 컬럼 포함, 시간순 정렬된 DataFrame
            feature_cols: 모델 입력 피처 컬럼 목록
            target_col:   타깃 컬럼명
            date_col:     날짜 컬럼명
        """
        df = df.sort_values(date_col).reset_index(drop=True)
        df[date_col] = pd.to_datetime(df[date_col])

        # 유효한 타깃이 있는 행만 사용
        valid_mask = df[target_col].notna()
        df = df[valid_mask].reset_index(drop=True)
        logger.info("백테스트 시작: 전체 %d건", len(df))

        windows = self._build_windows(df, date_col)
        logger.info("생성된 윈도우 수: %d", len(windows))

        all_preds    = []
        window_mets  = []

        for i, (train_idx, val_idx) in enumerate(windows):
            wm = self._run_window(
                df, train_idx, val_idx,
                feature_cols, target_col, date_col,
                window_num=i+1,
                model_kwargs=model_kwargs or {},
            )
            if wm is None:
                continue
            preds_df, metrics = wm
            all_preds.append(preds_df)
            window_mets.append(metrics)
            logger.info(
                "윈도우 %02d | 학습 %d건 → 검증 %d건 | MAE=%.1f%% Dir=%.1f%%",
                i+1, len(train_idx), len(val_idx),
                metrics["mae"], metrics["direction_acc"] * 100,
            )

        if not all_preds:
            logger.error("유효한 백테스트 윈도우가 없습니다.")
            return BacktestResult()

        pred_df = pd.concat(all_preds, ignore_index=True)

        # 전체 메트릭 집계
        overall_mae  = (pred_df["actual"] - pred_df["pred"]).abs().mean()
        dir_acc      = ((pred_df["actual"] > 0) == (pred_df["pred"] > 0)).mean()
        cov_90       = (
            (pred_df["actual"] >= pred_df["ci_90_low"]) &
            (pred_df["actual"] <= pred_df["ci_90_high"])
        ).mean()

        result = BacktestResult(
            predictions           = pred_df,
            window_metrics        = window_mets,
            overall_mae           = round(overall_mae, 3),
            overall_direction_acc = round(dir_acc, 4),
            coverage_90           = round(cov_90, 4),
            n_windows             = len(window_mets),
        )
        logger.info(result.summary())
        return result

    # ── 윈도우 생성 ───────────────────────────────────────────

    def _build_windows(
        self,
        df:       pd.DataFrame,
        date_col: str,
    ) -> list[tuple[list[int], list[int]]]:
        """
        Expanding Window 생성.
        각 윈도우: train = 처음~분할점, val = 분할점~다음 분할점
        """
        dates   = pd.to_datetime(df[date_col])
        min_dt  = dates.min()
        max_dt  = dates.max()

        min_train_days = self.cfg.min_train_years * 365
        val_days       = self.cfg.val_window_months * 30
        step_days      = self.cfg.step_months * 30

        windows = []
        split_dt = min_dt + pd.Timedelta(days=min_train_days)

        while split_dt + pd.Timedelta(days=val_days) <= max_dt:
            val_end = split_dt + pd.Timedelta(days=val_days)

            train_mask = dates < split_dt
            val_mask   = (dates >= split_dt) & (dates < val_end)

            train_idx = df.index[train_mask].tolist()
            val_idx   = df.index[val_mask].tolist()

            if len(train_idx) >= 20 and len(val_idx) >= 5:
                windows.append((train_idx, val_idx))

            split_dt += pd.Timedelta(days=step_days)

        return windows

    # ── 단일 윈도우 실행 ──────────────────────────────────────

    def _run_window(
        self,
        df:           pd.DataFrame,
        train_idx:    list[int],
        val_idx:      list[int],
        feature_cols: list[str],
        target_col:   str,
        date_col:     str,
        window_num:   int,
        model_kwargs: dict,
    ) -> Optional[tuple[pd.DataFrame, dict]]:
        """단일 윈도우 학습 + 예측"""
        try:
            train_df = df.loc[train_idx]
            val_df   = df.loc[val_idx]

            available_feats = [c for c in feature_cols if c in train_df.columns]

            # 간단히 결측값만 처리 (이미 빌드된 피처 가정)
            X_train = train_df[available_feats].copy()
            X_val   = val_df[available_feats].copy()
            y_train = train_df[target_col]
            y_val   = val_df[target_col]

            # 결측값 처리 (학습셋 중앙값으로)
            fill_vals = X_train.median()
            X_train = X_train.fillna(fill_vals)
            X_val   = X_val.fillna(fill_vals)

            # 학습셋의 20%를 Conformal 보정셋으로 분리
            cal_size = max(5, int(len(X_train) * 0.15))
            X_cal   = X_train.iloc[-cal_size:]
            y_cal   = y_train.iloc[-cal_size:]
            X_tr    = X_train.iloc[:-cal_size]
            y_tr    = y_train.iloc[:-cal_size]

            model = IPOPriceModel(**model_kwargs)
            model.fit(X_tr, y_tr, X_cal, y_cal)

            # 예측
            pred_df = model.predict(X_val)
            metrics = model.evaluate(X_val, y_val)

            # 결과 레코드 조합
            result = val_df[[date_col, "corp_name", target_col]].copy()
            if "offering_price" in val_df.columns:
                result["offering_price"] = val_df["offering_price"].values
            result = result.rename(columns={target_col: "actual"})
            result["pred"]      = pred_df["pred_return_pct"].values
            result["up_prob"]   = pred_df["up_probability"].values
            result["ci_90_low"]  = pred_df["ci_90_low"].values
            result["ci_90_high"] = pred_df["ci_90_high"].values
            result["risk_grade"] = pred_df["risk_grade"].values
            result["error"]      = result["actual"] - result["pred"]
            result["abs_error"]  = result["error"].abs()
            result["window"]     = window_num
            result["prediction_target"] = target_col

            return result, metrics

        except Exception as e:
            logger.warning("윈도우 %d 실패: %s", window_num, e)
            return None


def run_quick_backtest(n_samples: int = 400) -> BacktestResult:
    """
    데모 데이터로 백테스트를 빠르게 실행한다.
    실제 데이터 없이 파이프라인 전체를 검증하는 용도.
    """
    from data.processors.feature_engineer import build_demo_dataset
    from features.definitions import get_core_feature_names

    df = build_demo_dataset(n_samples)
    feat_cols = [c for c in get_core_feature_names() if c in df.columns]

    bt = WalkForwardBacktester()
    result = bt.run(df, feat_cols, model_kwargs={"n_estimators": 100})
    return result


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    result = run_quick_backtest(n_samples=500)
    print("\n" + result.summary())

    print("\n── 연도별 성능 ─────────────────────────")
    print(result.by_year().to_string(index=False))

    print("\n── 예측 샘플 (처음 5건) ─────────────────")
    cols = ["listing_date", "corp_name", "actual", "pred", "abs_error", "risk_grade"]
    print(result.predictions[cols].head().to_string(index=False))
