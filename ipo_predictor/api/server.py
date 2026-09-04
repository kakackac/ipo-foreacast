"""
api/server.py
─────────────
공모주 가격 예측 FastAPI 서버.

엔드포인트:
  POST /predict          단일 종목 예측
  POST /predict/batch    배치 예측
  GET  /health           헬스 체크
  GET  /model/info       현재 모델 메타 정보
  GET  /features         피처 정의 목록

실제 배포 시 uvicorn으로 실행:
  uvicorn api.server:app --host 0.0.0.0 --port 8000 --workers 4
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)

app = FastAPI(
    title="IPO 상장일 수익률 예측 API",
    description="공모주 상장일 시초가 및 종가 수익률 예측 서비스",
    version="2.0.0",
)


# ── 요청/응답 스키마 ──────────────────────────────────────────

class IPOFeatures(BaseModel):
    """단일 종목 예측 요청 피처"""

    # 필수 — CORE 피처
    institutional_demand_ratio:    float = Field(...,  ge=0, le=3000,  description="기관 수요예측 경쟁률")
    lockup_6m_ratio:               float = Field(...,  ge=0, le=1,     description="6개월 확약 비율")
    lockup_3m_ratio:               float = Field(0.0,  ge=0, le=1)
    lockup_1m_ratio:               float = Field(0.0,  ge=0, le=1)
    lockup_15d_ratio:              float = Field(0.0,  ge=0, le=1)
    offering_price_band_position:  float = Field(...,  description="밴드 위치 (0=하단, 1=상단, >1=초과)")
    kospi_momentum_5d:             float = Field(0.0,  description="KOSPI 5일 수익률")
    kospi_momentum_20d:            float = Field(0.0,  description="KOSPI 20일 수익률")
    recent_ipo_avg_return_sector:  float = Field(10.0, description="섹터 최근 IPO 평균 수익률(%)")
    recent_ipo_avg_return_all:     float = Field(10.0, description="전체 최근 IPO 평균 수익률(%)")

    # 선택 — SECONDARY 피처 (없으면 모델 내부에서 중앙값 대체)
    float_share_ratio:             Optional[float] = Field(None, ge=0, le=1)
    secondary_offering_ratio:      Optional[float] = Field(None, ge=0, le=1)
    major_shareholder_lockup_months: Optional[int] = Field(None, ge=0)
    same_day_ipo_count:            Optional[int]   = Field(None, ge=0)
    risk_factor_count:             Optional[int]   = Field(None, ge=0)
    underwriter_tier:              Optional[int]   = Field(None, ge=1, le=3)
    offering_per:                  Optional[float] = Field(None, ge=0)
    per_vs_sector_median:          Optional[float] = Field(None, ge=0)
    revenue_growth_3y:             Optional[float] = Field(None)
    operating_margin:              Optional[float] = Field(None)
    debt_ratio:                    Optional[float] = Field(None, ge=0)

    # 메타 정보 (예측에 사용되지 않음, 로깅용)
    corp_name:      Optional[str] = None
    listing_date:   Optional[str] = None

    @validator("lockup_6m_ratio", "lockup_3m_ratio", "lockup_1m_ratio", "lockup_15d_ratio")
    def lockup_sum_max_one(cls, v, values):
        # 개별 비율은 0~1 사이 (합이 1 초과해도 허용 — 실제 DART 데이터에서 발생)
        return min(v, 1.0)

    class Config:
        json_schema_extra = {
            "example": {
                "corp_name":                     "테스트AI",
                "listing_date":                  "2025-03-10",
                "institutional_demand_ratio":    1250.0,
                "lockup_6m_ratio":               0.42,
                "lockup_3m_ratio":               0.18,
                "lockup_1m_ratio":               0.10,
                "lockup_15d_ratio":              0.05,
                "offering_price_band_position":  1.1,
                "kospi_momentum_5d":             0.012,
                "kospi_momentum_20d":            0.035,
                "recent_ipo_avg_return_sector":  18.5,
                "recent_ipo_avg_return_all":     14.2,
                "offering_per":                  22.0,
                "per_vs_sector_median":          0.78,
                "revenue_growth_3y":             0.35,
                "operating_margin":              0.12,
            }
        }


class ReturnPrediction(BaseModel):
    """하나의 수익률 타깃에 대한 예측 결과"""
    pred_return_pct:      float       = Field(..., description="예측 수익률 (%)")
    up_probability:       float       = Field(..., description="상승 확률 (0~1)")
    ci_90_low:            float       = Field(..., description="90% 신뢰구간 하단")
    ci_90_high:           float       = Field(..., description="90% 신뢰구간 상단")
    ci_50_low:            float       = Field(..., description="50% 신뢰구간 하단")
    ci_50_high:           float       = Field(..., description="50% 신뢰구간 상단")
    risk_grade:           str         = Field(..., description="리스크 등급 (A/B/C)")
    top_features:         list[dict]  = Field(default_factory=list)


class PredictionResponse(BaseModel):
    """상장 전 시초가·종가 동시 예측 응답"""
    corp_name:            Optional[str]
    listing_date:         Optional[str]
    opening:              ReturnPrediction
    closing:              ReturnPrediction
    lockup_weighted_score: float      = Field(..., description="확약 가중 점수")
    disclaimer:           str         = "이 예측은 투자 조언이 아닙니다. 과거 성과가 미래를 보장하지 않습니다."
    predicted_at:         str         = Field(default_factory=lambda: datetime.now().isoformat())


class BatchRequest(BaseModel):
    items: list[IPOFeatures]


class BatchResponse(BaseModel):
    count:       int
    predictions: list[PredictionResponse]


class ModelInfo(BaseModel):
    opening_model: str
    closing_model: str
    n_features:   dict[str, int]
    feature_set:  str
    trained_at:   Optional[str]
    overall_mae:  Optional[float]
    direction_acc: Optional[float]


# ── 모델 로더 (싱글톤) ────────────────────────────────────────

_models: dict = {}
_model_meta: dict = {}

def get_models() -> dict:
    """시초가·종가 모델 싱글톤 반환."""
    global _models, _model_meta
    if not _models:
        try:
            import sys, os
            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            from models.baseline.gradient_boost_model import IPOPriceModel
            _models = {
                "opening": IPOPriceModel.load("baseline_open_v1"),
                "closing": IPOPriceModel.load("baseline_close_v1"),
            }
            logger.info("시초가·종가 모델 로드 완료")
        except Exception as e:
            logger.warning("저장된 상장일 모델 없음, 데모 모델 사용: %s", e)
            _models = _create_demo_models()
    return _models

def _create_demo_models() -> dict:
    """저장된 모델이 없을 때 사용할 시초가·종가 데모 모델"""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from models.baseline.gradient_boost_model import IPOPriceModel
    from data.processors.feature_engineer import build_demo_dataset
    from features.definitions import get_phase2_feature_names

    df = build_demo_dataset(400, phase="phase2")
    feat_cols = [c for c in get_phase2_feature_names() if c in df.columns]
    X = df[feat_cols]
    split = int(len(df) * 0.8)
    models = {}
    for target_name, target_col in [("opening", "open_return_pct"), ("closing", "close_return_pct")]:
        model = IPOPriceModel(n_estimators=100)
        model.fit(
            X.iloc[:split], df[target_col].iloc[:split],
            X.iloc[split:], df[target_col].iloc[split:],
        )
        model.feature_names = feat_cols
        models[target_name] = model
    logger.info("상장일 데모 모델 학습 완료 (%d 피처)", len(feat_cols))
    return models


# ── 피처 변환 유틸 ────────────────────────────────────────────

def _feature_dict(feat: IPOFeatures) -> dict:
    """요청 피처를 모델 입력용 숫자 사전으로 정리한다."""
    raw = feat.dict(exclude={"corp_name", "listing_date"})
    d = {key: (0.0 if value is None else value) for key, value in raw.items()}
    d["lockup_weighted_score"] = (
        d["lockup_6m_ratio"] * 1.00 +
        d["lockup_3m_ratio"] * 0.75 +
        d["lockup_1m_ratio"] * 0.50 +
        d["lockup_15d_ratio"] * 0.25
    )
    d["band_exceeded"] = int(d["offering_price_band_position"] > 1.0)
    return d


def _features_to_df(feature_dict: dict, model):
    """IPOFeatures → 모델 입력 DataFrame 변환"""
    import pandas as pd
    df = pd.DataFrame([feature_dict])

    # 모델이 요구하는 피처만 선택, 없는 피처는 0으로
    for col in model.feature_names:
        if col not in df.columns:
            df[col] = 0.0
    return df[model.feature_names].astype(float).fillna(0.0)


def _return_prediction(model, feature_dict: dict) -> ReturnPrediction:
    X = _features_to_df(feature_dict, model)
    pred = model.predict(X).iloc[0]
    return ReturnPrediction(
        pred_return_pct=float(pred["pred_return_pct"]),
        up_probability=float(pred["up_probability"]),
        ci_90_low=float(pred["ci_90_low"]),
        ci_90_high=float(pred["ci_90_high"]),
        ci_50_low=float(pred["ci_50_low"]),
        ci_50_high=float(pred["ci_50_high"]),
        risk_grade=pred["risk_grade"],
        top_features=model.explain_prediction(feature_dict, top_n=5),
    )


# ── 엔드포인트 ────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/model/info", response_model=ModelInfo)
async def model_info():
    models = get_models()
    return ModelInfo(
        opening_model = "baseline_open_v1",
        closing_model = "baseline_close_v1",
        n_features   = {name: len(model.feature_names) for name, model in models.items()},
        feature_set  = "core+secondary",
        trained_at   = _model_meta.get("trained_at"),
        overall_mae  = _model_meta.get("overall_mae"),
        direction_acc = _model_meta.get("direction_acc"),
    )


@app.get("/features")
async def list_features():
    """피처 정의 목록 (앱 개발팀 참조용)"""
    from features.definitions import ALL_FEATURES
    return [
        {
            "name":        f.name,
            "group":       f.group.value,
            "importance":  f.importance.value,
            "description": f.description,
            "fill_na":     f.fill_na,
        }
        for f in ALL_FEATURES
    ]


@app.post("/predict", response_model=PredictionResponse)
async def predict(feat: IPOFeatures, background_tasks: BackgroundTasks):
    """단일 종목 예측"""
    try:
        models = get_models()
        feature_dict = _feature_dict(feat)

        response = PredictionResponse(
            corp_name             = feat.corp_name,
            listing_date          = feat.listing_date,
            opening               = _return_prediction(models["opening"], feature_dict),
            closing               = _return_prediction(models["closing"], feature_dict),
            lockup_weighted_score = round(
                feat.lockup_6m_ratio * 1.0 + feat.lockup_3m_ratio * 0.75 +
                feat.lockup_1m_ratio * 0.5 + feat.lockup_15d_ratio * 0.25, 3
            ),
        )

        # 로깅은 백그라운드 태스크로 (응답 속도 영향 없음)
        background_tasks.add_task(
            _log_prediction, feat.corp_name, feat.listing_date,
            response.opening.pred_return_pct, response.opening.risk_grade,
        )
        return response

    except Exception as e:
        logger.error("예측 실패: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"예측 중 오류: {str(e)}")


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(batch: BatchRequest):
    """배치 예측 (최대 50건)"""
    if len(batch.items) > 50:
        raise HTTPException(status_code=400, detail="배치 최대 50건")

    try:
        if not batch.items:
            return BatchResponse(count=0, predictions=[])
        models = get_models()
        responses = []
        for feat in batch.items:
            feature_dict = _feature_dict(feat)
            responses.append(PredictionResponse(
                corp_name=feat.corp_name,
                listing_date=feat.listing_date,
                opening=_return_prediction(models["opening"], feature_dict),
                closing=_return_prediction(models["closing"], feature_dict),
                lockup_weighted_score=round(feature_dict["lockup_weighted_score"], 3),
            ))
        return BatchResponse(count=len(responses), predictions=responses)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _log_prediction(corp_name, listing_date, pred_return, risk_grade):
    """예측 결과 비동기 로깅 (DB 저장 / 모니터링 훅)"""
    logger.info(
        "예측 기록: %s (%s) → %.1f%% [%s]",
        corp_name, listing_date, pred_return, risk_grade
    )


# ── 서버 실행 진입점 ──────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)
