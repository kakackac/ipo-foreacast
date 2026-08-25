"""
config.py
─────────
전체 프로젝트 설정. 환경변수(.env)에서 민감 정보를 읽고,
나머지 상수는 여기서 중앙 관리한다.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

# ── 프로젝트 루트 ──────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR  = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
MANUAL_DATA_DIR = DATA_DIR / "manual"
MODEL_DIR = ROOT_DIR / "models" / "saved"
REPORT_DIR = ROOT_DIR / "reports"

for d in [RAW_DIR, PROC_DIR, MANUAL_DATA_DIR, MODEL_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── DART API ───────────────────────────────────────────────────
DART_API_KEY = os.getenv("DART_API_KEY", "YOUR_DART_API_KEY")
DART_BASE_URL = "https://opendart.fss.or.kr/api"

# ── KRX OpenAPI (한국거래소) ───────────────────────────────────
# 개인 웹 로그인 계정은 사용하지 않는다. API 키는 서버 환경변수에만 둔다.
KRX_API_KEY = os.getenv("KRX_API_KEY", "")
KRX_OPENAPI_BASE_URL = os.getenv("KRX_OPENAPI_BASE_URL", "https://data-dbg.krx.co.kr/svc/apis")

# ── 상장일 실적 확정 ────────────────────────────────
# KRX 종가를 타곟으로 사용하되, NXT 애프터마켓 종료 후에 실측치를 확정한다.
POST_LISTING_RECONCILIATION_TIME = os.getenv("POST_LISTING_RECONCILIATION_TIME", "21:00")
POST_LISTING_RECONCILIATION_TIMEZONE = "Asia/Seoul"

# ── DB (PostgreSQL + TimescaleDB) ─────────────────────────────
DB_HOST     = os.getenv("DB_HOST", "localhost")
DB_PORT     = int(os.getenv("DB_PORT", "5432"))
DB_NAME     = os.getenv("DB_NAME", "ipo_db")
DB_USER     = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_URL      = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


# ── 모델 하이퍼파라미터 기본값 ───────────────────────────────
@dataclass
class XGBConfig:
    n_estimators:     int   = 500
    max_depth:        int   = 6
    learning_rate:    float = 0.05
    subsample:        float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int   = 5
    reg_alpha:        float = 0.1   # L1
    reg_lambda:       float = 1.0   # L2
    random_state:     int   = 42
    n_jobs:           int   = -1


@dataclass
class LGBMConfig:
    n_estimators:    int   = 500
    max_depth:       int   = 6
    learning_rate:   float = 0.05
    num_leaves:      int   = 63
    subsample:       float = 0.8
    colsample_bytree:float = 0.8
    min_child_samples:int  = 10
    reg_alpha:       float = 0.1
    reg_lambda:      float = 1.0
    random_state:    int   = 42
    n_jobs:          int   = -1
    verbose:         int   = -1


# ── Walk-forward 백테스트 설정 ─────────────────────────────────
@dataclass
class BacktestConfig:
    # 최초 학습에 사용할 최소 연수
    min_train_years: int = 3
    # 검증 윈도우 크기 (개월)
    val_window_months: int = 6
    # 슬라이딩 스텝 (개월)
    step_months: int = 3
    # 아웃라이어 판단 기준 (σ 배수)
    outlier_sigma: float = 2.0
    # 아웃라이어 최소 오차 (% — 너무 작은 오차는 제외)
    outlier_min_error_pct: float = 15.0


# ── 피처 엔지니어링 설정 ──────────────────────────────────────
@dataclass
class FeatureConfig:
    # 시장 모멘텀 계산 기간 (거래일)
    momentum_windows: list = field(default_factory=lambda: [5, 20, 60])
    # 동일 섹터 직전 IPO 참조 건수
    sector_lookback_n: int = 5
    # 재무 데이터 최소 보유 연도 수 (없으면 결측 처리)
    min_financial_years: int = 1


XGB_CFG      = XGBConfig()
LGBM_CFG     = LGBMConfig()
BACKTEST_CFG = BacktestConfig()
FEATURE_CFG  = FeatureConfig()
