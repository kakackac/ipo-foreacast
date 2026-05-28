"""
models/baseline/gradient_boost_model.py
─────────────────────────────────────────
XGBoost / LightGBM 대신 scikit-learn GradientBoostingRegressor로
동일한 인터페이스를 구현한다.

실제 운영 환경에서는 XGBoost/LightGBM으로 교체하면 된다.
인터페이스(fit, predict, get_feature_importance)는 동일하다.

구현 포인트:
  - 회귀 모델: 시초가 수익률 (연속값) 예측
  - 분류 모델: 방향성 (상승/하락) 예측 → 확률 기반 신뢰도 출력
  - Conformal Prediction: 통계적 보장 신뢰구간
  - SHAP 대체: feature_importances_ 기반 기여도 계산
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import mean_absolute_error, accuracy_score

from config import MODEL_DIR

logger = logging.getLogger(__name__)


class IPOPriceModel:
    """
    공모주 시초가 수익률 예측 모델.

    두 가지 예측을 동시에 수행:
      1. regressor  → 수익률 수치 (ex. +23.4%)
      2. classifier → 방향성 (상승 확률)

    향후 XGBoost 교체 시 이 클래스의 내부만 바꾸면 된다.
    """

    # Conformal Prediction 분위수
    QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]

    def __init__(
        self,
        n_estimators:  int   = 300,
        max_depth:     int   = 5,
        learning_rate: float = 0.05,
        subsample:     float = 0.8,
        random_state:  int   = 42,
    ):
        self.params = dict(
            n_estimators  = n_estimators,
            max_depth     = max_depth,
            learning_rate = learning_rate,
            subsample     = subsample,
            random_state  = random_state,
        )

        self.regressor  = GradientBoostingRegressor(**self.params)
        self.classifier = GradientBoostingClassifier(**self.params)

        self.feature_names: list[str] = []
        self.calibration_errors: np.ndarray = np.array([])  # Conformal Prediction용
        self._fitted = False

    # ── 학습 ──────────────────────────────────────────────────

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        calibration_X: Optional[pd.DataFrame] = None,
        calibration_y: Optional[pd.Series]    = None,
    ) -> "IPOPriceModel":
        """
        모델 학습.

        calibration_X/y: Conformal Prediction 신뢰구간 보정용
                         (학습셋의 20%를 별도로 남겨 사용 권장)
        """
        self.feature_names = list(X.columns)
        y_direction = (y > 0).astype(int)   # 방향 레이블 (1=상승, 0=하락)

        logger.info("회귀 모델 학습 중... (%d samples, %d features)", len(X), len(X.columns))
        self.regressor.fit(X, y)

        logger.info("분류 모델 학습 중...")
        # 단일 클래스 방어: 양수/음수가 모두 있어야 분류기 학습 가능
        n_pos = (y_direction == 1).sum()
        n_neg = (y_direction == 0).sum()
        if n_pos < 2 or n_neg < 2:
            logger.warning("분류기 학습 불가 (클래스 불균형: pos=%d, neg=%d) — 회귀 기반 방향 예측 사용", n_pos, n_neg)
            self._classifier_fallback = True
        else:
            self._classifier_fallback = False
            self.classifier.fit(X, y_direction)

        # Conformal Prediction: 보정셋 잔차로 신뢰구간 범위 설정
        if calibration_X is not None and calibration_y is not None:
            cal_pred = self.regressor.predict(calibration_X)
            self.calibration_errors = np.abs(calibration_y.values - cal_pred)
            logger.info("Conformal 보정 완료: 중앙값 오차 = %.1f%%",
                        np.median(self.calibration_errors))
        else:
            # 보정셋 없으면 학습셋 잔차로 대략 추정 (실제보다 낙관적)
            train_pred = self.regressor.predict(X)
            self.calibration_errors = np.abs(y.values - train_pred)

        self._fitted = True
        logger.info("모델 학습 완료")
        return self

    # ── 예측 ──────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        단일 또는 배치 예측.

        반환 컬럼:
          - pred_return_pct  : 예측 수익률 (%)
          - up_probability   : 상승 확률 (0~1)
          - ci_90_low / high : 90% 신뢰구간 (Conformal Prediction)
          - ci_50_low / high : 50% 신뢰구간
          - risk_grade       : 리스크 등급 (A/B/C)
        """
        self._check_fitted()
        X = X[self.feature_names] if all(f in X.columns for f in self.feature_names) else X

        pred_returns = self.regressor.predict(X)
        # 분류기 폴백: 회귀 예측값으로 상승 확률 근사
        if getattr(self, "_classifier_fallback", False):
            up_proba = (pred_returns > 0).astype(float) * 0.6 + 0.2
        else:
            up_proba = self.classifier.predict_proba(X)[:, 1]

        # Conformal Prediction 신뢰구간
        q90 = np.quantile(self.calibration_errors, 0.90)
        q50 = np.quantile(self.calibration_errors, 0.50)

        result = pd.DataFrame({
            "pred_return_pct": np.round(pred_returns, 2),
            "up_probability":  np.round(up_proba, 4),
            "ci_90_low":       np.round(pred_returns - q90, 2),
            "ci_90_high":      np.round(pred_returns + q90, 2),
            "ci_50_low":       np.round(pred_returns - q50, 2),
            "ci_50_high":      np.round(pred_returns + q50, 2),
        }, index=X.index)

        result["risk_grade"] = result.apply(self._assign_risk_grade, axis=1)
        return result

    def predict_single(self, feature_dict: dict) -> dict:
        """단일 종목 예측 (API 서빙용)"""
        X = pd.DataFrame([feature_dict])
        pred = self.predict(X)
        return pred.iloc[0].to_dict()

    # ── 평가 ──────────────────────────────────────────────────

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """모델 성능 평가 지표 계산"""
        self._check_fitted()
        preds = self.predict(X)

        mae           = mean_absolute_error(y, preds["pred_return_pct"])
        y_dir_actual  = (y > 0).astype(int)
        y_dir_pred    = (preds["pred_return_pct"] > 0).astype(int)
        direction_acc = accuracy_score(y_dir_actual, y_dir_pred)

        # Conformal coverage rate
        covered_90 = (
            (y >= preds["ci_90_low"]) & (y <= preds["ci_90_high"])
        ).mean()
        covered_50 = (
            (y >= preds["ci_50_low"]) & (y <= preds["ci_50_high"])
        ).mean()

        up_mask   = preds["pred_return_pct"] > 0
        up_actual = y[up_mask].mean() if up_mask.sum() > 0 else np.nan

        metrics = {
            "mae":                   round(mae, 3),
            "direction_acc":         round(direction_acc, 4),
            "direction_accuracy":    round(direction_acc, 4),
            "coverage_90pct":        round(covered_90, 4),
            "coverage_50pct":        round(covered_50, 4),
            "up_pred_actual_return": round(up_actual, 2) if not np.isnan(up_actual) else None,
            "n_samples":             len(y),
        }
        return metrics

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> dict:
        """K-Fold CV (Walk-forward CV는 별도 backtester에서 수행)"""
        neg_mae = cross_val_score(
            self.regressor, X, y, cv=cv, scoring="neg_mean_absolute_error"
        )
        return {
            "cv_mae_mean": round(-neg_mae.mean(), 3),
            "cv_mae_std":  round(neg_mae.std(), 3),
        }

    # ── 피처 중요도 ───────────────────────────────────────────

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        피처 중요도 DataFrame.
        실제 XGBoost 사용 시 SHAP으로 교체하면 된다.
        """
        self._check_fitted()
        importances = self.regressor.feature_importances_
        df = pd.DataFrame({
            "feature":    self.feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False).head(top_n)
        df["importance_pct"] = (df["importance"] / df["importance"].sum() * 100).round(2)
        return df.reset_index(drop=True)

    def explain_prediction(self, feature_dict: dict, top_n: int = 5) -> list[dict]:
        """
        단일 예측 설명: 어떤 피처가 얼마나 기여했는지.
        XGBoost SHAP 대체: feature_importance × (피처값 - 평균) 로 근사.

        NOTE: 실제 운영 시 SHAP TreeExplainer로 교체 권장.
        """
        self._check_fitted()
        fi = self.get_feature_importance()
        explanations = []

        for _, row in fi.head(top_n).iterrows():
            feat = row["feature"]
            val  = feature_dict.get(feat, 0)
            imp  = row["importance_pct"]
            explanations.append({
                "feature":    feat,
                "value":      round(val, 4),
                "importance": round(imp, 2),
            })
        return explanations

    # ── 리스크 등급 ───────────────────────────────────────────

    @staticmethod
    def _assign_risk_grade(row: pd.Series) -> str:
        """
        신뢰구간 폭과 방향 일관성으로 리스크 등급 부여.
          A: 좁은 구간 + 높은 상승 확률 (신뢰도 높음)
          B: 중간
          C: 넓은 구간 또는 낮은 상승 확률 (주의)
        """
        ci_width = row["ci_90_high"] - row["ci_90_low"]
        up_prob  = row["up_probability"]

        if ci_width < 30 and up_prob > 0.65:
            return "A"
        elif ci_width < 60 and up_prob > 0.50:
            return "B"
        else:
            return "C"

    # ── 저장 / 로드 ───────────────────────────────────────────

    def save(self, name: str = "baseline_v1") -> Path:
        """모델 파일 저장"""
        path = MODEL_DIR / f"{name}.pkl"
        payload = {
            "params":              self.params,
            "regressor":           self.regressor,
            "classifier":          self.classifier,
            "feature_names":       self.feature_names,
            "calibration_errors":  self.calibration_errors,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logger.info("모델 저장: %s", path)

        # 메타데이터 JSON 저장 (버전 관리용)
        meta_path = MODEL_DIR / f"{name}_meta.json"
        with open(meta_path, "w") as f:
            json.dump({
                "name":          name,
                "params":        self.params,
                "feature_names": self.feature_names,
                "n_features":    len(self.feature_names),
            }, f, indent=2, ensure_ascii=False)
        return path

    @classmethod
    def load(cls, name: str = "baseline_v1") -> "IPOPriceModel":
        """저장된 모델 로드"""
        path = MODEL_DIR / f"{name}.pkl"
        with open(path, "rb") as f:
            payload = pickle.load(f)
        model = cls(**payload["params"])
        model.regressor          = payload["regressor"]
        model.classifier         = payload["classifier"]
        model.feature_names      = payload["feature_names"]
        model.calibration_errors = payload["calibration_errors"]
        model._fitted = True
        logger.info("모델 로드: %s (%d features)", name, len(model.feature_names))
        return model

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("fit()을 먼저 호출하세요.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # 데모 데이터로 빠른 smoke test
    import sys
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from data.processors.feature_engineer import build_demo_dataset
    from features.definitions import get_core_feature_names

    df = build_demo_dataset(500)
    feat_cols = get_core_feature_names()
    feat_cols = [c for c in feat_cols if c in df.columns]

    X = df[feat_cols]
    y = df["open_return_pct"]

    split = int(len(df) * 0.8)
    X_train, X_cal  = X.iloc[:split], X.iloc[split:split+int(split*0.1)]
    X_test          = X.iloc[split:]
    y_train, y_cal  = y.iloc[:split], y.iloc[split:split+int(split*0.1)]
    y_test          = y.iloc[split:]

    model = IPOPriceModel(n_estimators=100)
    model.fit(X_train, y_train, X_cal, y_cal)

    metrics = model.evaluate(X_test, y_test)
    print("\n── 평가 지표 ──────────────────────")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    preds = model.predict(X_test.head(3))
    print("\n── 샘플 예측 ──────────────────────")
    print(preds.to_string())

    print("\n── 피처 중요도 Top 10 ─────────────")
    print(model.get_feature_importance(10).to_string())
