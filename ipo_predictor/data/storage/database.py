"""
data/storage/database.py
─────────────────────────
DB 스키마 정의 (SQLAlchemy ORM).

테이블 구조:
  ipo_master        공모주 기본 정보 (1종목 1행)
  ipo_features      모델 입력 피처 (학습·추론 공유)
  ipo_predictions   예측 결과 이력 (모델 버전별)
  model_registry    학습된 모델 메타 정보
  backtest_results  백테스트 윈도우별 결과
  outlier_log       아웃라이어 분석 기록

실제 운영 시 PostgreSQL + TimescaleDB를 사용한다.
개발·테스트 시에는 SQLite로 대체 가능하다.

사용법:
    from data.storage.database import get_engine, get_session, init_db
    engine = get_engine()
    init_db(engine)          # 최초 1회 테이블 생성
    with get_session() as s:
        s.add(...)
        s.commit()
"""

import os
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    create_engine, Column, String, Float, Integer,
    Boolean, DateTime, Text, JSON, ForeignKey, Index,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship
from sqlalchemy.pool import StaticPool

from config import DB_URL, ROOT_DIR


# ── 엔진 팩토리 ───────────────────────────────────────────────

def get_engine(url: Optional[str] = None, echo: bool = False):
    """
    SQLAlchemy 엔진 반환.
    url 미지정 시 config.DB_URL 사용.
    SQLite fallback: DB_URL이 비어 있거나 개발 환경일 때.
    """
    target = url or DB_URL
    if not target or target.startswith("postgresql://postgres:@"):
        # 개발 환경 — SQLite 사용
        sqlite_path = ROOT_DIR / "data" / "storage" / "ipo_dev.db"
        target = f"sqlite:///{sqlite_path}"
        return create_engine(
            target,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=echo,
        )
    return create_engine(target, echo=echo, pool_pre_ping=True)


@contextmanager
def get_session(engine=None):
    """세션 컨텍스트 매니저"""
    if engine is None:
        engine = get_engine()
    with Session(engine) as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise


# ── Base ──────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── 테이블 정의 ───────────────────────────────────────────────

class IPOMaster(Base):
    """
    공모주 기본 마스터 테이블.
    수집 → 가공 전 단계의 원본 정보 보존.
    """
    __tablename__ = "ipo_master"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    corp_code     = Column(String(8),  nullable=False, index=True)   # DART 기업 코드
    corp_name     = Column(String(100), nullable=False)
    ticker        = Column(String(10),  nullable=True, index=True)   # 종목 코드 (KRX)
    market        = Column(String(10),  nullable=True)               # KOSPI / KOSDAQ
    sector_code   = Column(String(20),  nullable=True)
    sector_name   = Column(String(50),  nullable=True)

    # 공모 일정
    rcept_no      = Column(String(14),  nullable=True, unique=True)  # DART 접수번호
    demand_start  = Column(DateTime,    nullable=True)               # 수요예측 시작일
    demand_end    = Column(DateTime,    nullable=True)               # 수요예측 종료일
    sub_start     = Column(DateTime,    nullable=True)               # 청약 시작일
    sub_end       = Column(DateTime,    nullable=True)               # 청약 종료일
    listing_date  = Column(DateTime,    nullable=True, index=True)   # 상장일

    # 공모 구조
    price_band_low  = Column(Integer,  nullable=True)
    price_band_high = Column(Integer,  nullable=True)
    offering_price  = Column(Integer,  nullable=True)
    new_shares      = Column(Integer,  nullable=True)
    secondary_shares = Column(Integer, nullable=True)
    total_shares_post = Column(Integer, nullable=True)
    lead_underwriter  = Column(String(50), nullable=True)

    # 상장 결과 (상장 후 채움)
    open_price    = Column(Float,  nullable=True)
    close_price   = Column(Float,  nullable=True)
    open_return   = Column(Float,  nullable=True)   # (시초가/공모가 - 1) × 100
    close_return  = Column(Float,  nullable=True)

    created_at    = Column(DateTime, default=datetime.utcnow)
    updated_at    = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    features      = relationship("IPOFeatures", back_populates="ipo", uselist=False)
    predictions   = relationship("IPOPrediction", back_populates="ipo")

    __table_args__ = (
        Index("ix_ipo_listing_sector", "listing_date", "sector_code"),
    )


class IPOFeatures(Base):
    """
    모델 입력 피처 테이블.
    IPOMaster와 1:1 관계. 피처 추가 시 이 테이블에 컬럼 추가.
    """
    __tablename__ = "ipo_features"

    id     = Column(Integer, primary_key=True, autoincrement=True)
    ipo_id = Column(Integer, ForeignKey("ipo_master.id"), unique=True, nullable=False)

    # ── CORE 피처 ─────────────────────────────────────────────
    institutional_demand_ratio    = Column(Float, nullable=True)
    retail_subscription_ratio     = Column(Float, nullable=True)
    lockup_6m_ratio               = Column(Float, nullable=True, default=0.0)
    lockup_3m_ratio               = Column(Float, nullable=True, default=0.0)
    lockup_1m_ratio               = Column(Float, nullable=True, default=0.0)
    lockup_15d_ratio              = Column(Float, nullable=True, default=0.0)
    lockup_weighted_score         = Column(Float, nullable=True)
    offering_price_band_position  = Column(Float, nullable=True)
    band_exceeded                 = Column(Boolean, nullable=True)
    kospi_momentum_5d             = Column(Float, nullable=True)
    kospi_momentum_20d            = Column(Float, nullable=True)
    recent_ipo_avg_return_sector  = Column(Float, nullable=True)
    recent_ipo_avg_return_all     = Column(Float, nullable=True)

    # ── SECONDARY 피처 ────────────────────────────────────────
    float_share_ratio             = Column(Float, nullable=True)
    secondary_offering_ratio      = Column(Float, nullable=True)
    major_shareholder_lockup_months = Column(Integer, nullable=True)
    same_day_ipo_count            = Column(Integer, nullable=True)
    risk_factor_count             = Column(Integer, nullable=True)
    underwriter_tier              = Column(Integer, nullable=True)
    offering_per                  = Column(Float, nullable=True)
    per_vs_sector_median          = Column(Float, nullable=True)
    revenue_growth_3y             = Column(Float, nullable=True)
    operating_margin              = Column(Float, nullable=True)
    debt_ratio                    = Column(Float, nullable=True)

    # ── OPTIONAL 피처 (에러 분석 후 추가) ─────────────────────
    kospi_momentum_60d            = Column(Float, nullable=True)
    kosdaq_momentum_20d           = Column(Float, nullable=True)
    bear_market_flag              = Column(Boolean, nullable=True)
    news_mention_count_7d         = Column(Integer, nullable=True)

    # 재무 매력도 점수 (financial_analyzer 출력)
    financial_attractiveness      = Column(Float, nullable=True)
    attractiveness_grade          = Column(String(1), nullable=True)

    feature_version = Column(String(20), default="v1")
    computed_at     = Column(DateTime, default=datetime.utcnow)

    ipo = relationship("IPOMaster", back_populates="features")


class IPOPrediction(Base):
    """
    예측 결과 이력 테이블.
    모델 버전별로 예측 결과를 보존 → 챔피언-챌린저 비교 가능.
    """
    __tablename__ = "ipo_predictions"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    ipo_id        = Column(Integer, ForeignKey("ipo_master.id"), nullable=False, index=True)
    model_name    = Column(String(50),  nullable=False)    # "baseline_v1", "lgbm_v2"
    model_version = Column(String(20),  nullable=False)

    pred_return_pct  = Column(Float, nullable=False)
    up_probability   = Column(Float, nullable=True)
    ci_90_low        = Column(Float, nullable=True)
    ci_90_high       = Column(Float, nullable=True)
    ci_50_low        = Column(Float, nullable=True)
    ci_50_high       = Column(Float, nullable=True)
    risk_grade       = Column(String(1), nullable=True)
    lockup_score     = Column(Float, nullable=True)
    top_features     = Column(JSON, nullable=True)        # SHAP 기여도 top5

    # 실제 결과 (상장 후 채움)
    actual_return    = Column(Float, nullable=True)
    prediction_error = Column(Float, nullable=True)       # actual - pred
    direction_correct = Column(Boolean, nullable=True)    # 방향 맞혔는지

    predicted_at  = Column(DateTime, default=datetime.utcnow)
    resolved_at   = Column(DateTime, nullable=True)       # 상장 후 실제값 채운 시점

    ipo = relationship("IPOMaster", back_populates="predictions")

    __table_args__ = (
        Index("ix_pred_model_date", "model_name", "predicted_at"),
        UniqueConstraint("ipo_id", "model_name", "model_version", name="uq_pred_ipo_model"),
    )


class ModelRegistry(Base):
    """
    학습된 모델 메타 정보 레지스트리.
    챔피언 모델 지정 및 버전 이력 관리.
    """
    __tablename__ = "model_registry"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    name          = Column(String(50),  nullable=False)
    version       = Column(String(20),  nullable=False)
    algorithm     = Column(String(30),  nullable=True)    # "GBM", "LGBM", "XGB"
    feature_set   = Column(String(20),  nullable=True)    # "core", "phase2"
    n_features    = Column(Integer,     nullable=True)
    feature_names = Column(JSON,        nullable=True)

    # 백테스트 성능
    overall_mae       = Column(Float, nullable=True)
    direction_acc     = Column(Float, nullable=True)
    coverage_90       = Column(Float, nullable=True)
    n_backtest_samples = Column(Integer, nullable=True)

    # 학습 조건
    train_start   = Column(DateTime, nullable=True)
    train_end     = Column(DateTime, nullable=True)
    n_train_samples = Column(Integer, nullable=True)
    hyperparams   = Column(JSON, nullable=True)

    # 운영 상태
    status        = Column(String(20), default="staging")  # staging / champion / retired
    is_champion   = Column(Boolean, default=False)
    file_path     = Column(String(200), nullable=True)

    trained_at    = Column(DateTime, default=datetime.utcnow)
    promoted_at   = Column(DateTime, nullable=True)       # champion 승격 시점
    retired_at    = Column(DateTime, nullable=True)

    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_model_name_version"),
        Index("ix_model_champion", "is_champion"),
    )


class BacktestResult(Base):
    """백테스트 윈도우별 결과 저장"""
    __tablename__ = "backtest_results"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    model_name    = Column(String(50), nullable=False)
    model_version = Column(String(20), nullable=False)
    window_num    = Column(Integer,    nullable=False)

    train_start   = Column(DateTime, nullable=True)
    train_end     = Column(DateTime, nullable=True)
    val_start     = Column(DateTime, nullable=True)
    val_end       = Column(DateTime, nullable=True)

    n_train       = Column(Integer, nullable=True)
    n_val         = Column(Integer, nullable=True)
    mae           = Column(Float,   nullable=True)
    direction_acc = Column(Float,   nullable=True)
    coverage_90   = Column(Float,   nullable=True)

    run_at        = Column(DateTime, default=datetime.utcnow)


class OutlierLog(Base):
    """아웃라이어 분석 기록 (에러 주도 개발 이력)"""
    __tablename__ = "outlier_log"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    ipo_id        = Column(Integer, ForeignKey("ipo_master.id"), nullable=True)
    corp_name     = Column(String(100), nullable=True)
    listing_date  = Column(DateTime,    nullable=True)

    actual_return = Column(Float,   nullable=True)
    pred_return   = Column(Float,   nullable=True)
    abs_error     = Column(Float,   nullable=True)
    sigma         = Column(Float,   nullable=True)

    cause         = Column(String(20),  nullable=True)    # A_market / B_supply / ...
    cause_detail  = Column(Text,        nullable=True)
    feature_hints = Column(JSON,        nullable=True)    # 추천 피처 목록

    analyzed_at   = Column(DateTime, default=datetime.utcnow)
    resolved      = Column(Boolean, default=False)        # 해당 피처 추가로 해결됐는지
    resolved_note = Column(Text, nullable=True)


class DataQualityLog(Base):
    """데이터 품질 체크 결과 로그"""
    __tablename__ = "data_quality_log"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    source      = Column(String(30), nullable=False)   # "DART", "KRX"
    check_name  = Column(String(50), nullable=False)
    status      = Column(String(10), nullable=False)   # "pass" / "warn" / "fail"
    detail      = Column(Text, nullable=True)
    n_records   = Column(Integer, nullable=True)
    checked_at  = Column(DateTime, default=datetime.utcnow)


# ── 초기화 ────────────────────────────────────────────────────

def init_db(engine=None) -> None:
    """전체 테이블 생성 (없는 테이블만 생성, 기존 데이터 보존)"""
    if engine is None:
        engine = get_engine()
    Base.metadata.create_all(engine)
    print(f"DB 초기화 완료: {engine.url}")


# ── Repository 패턴 ───────────────────────────────────────────

class IPORepository:
    """IPO 데이터 CRUD 헬퍼"""

    def __init__(self, session: Session):
        self.session = session

    def upsert_master(self, data: dict) -> IPOMaster:
        """corp_code + listing_date 기준 upsert"""
        existing = (
            self.session.query(IPOMaster)
            .filter_by(corp_code=data["corp_code"])
            .first()
        )
        if existing:
            for k, v in data.items():
                if hasattr(existing, k) and v is not None:
                    setattr(existing, k, v)
            existing.updated_at = datetime.utcnow()
            return existing
        obj = IPOMaster(**{k: v for k, v in data.items() if hasattr(IPOMaster, k)})
        self.session.add(obj)
        self.session.flush()
        return obj

    def upsert_features(self, ipo_id: int, features: dict) -> IPOFeatures:
        existing = self.session.query(IPOFeatures).filter_by(ipo_id=ipo_id).first()
        if existing:
            for k, v in features.items():
                if hasattr(existing, k):
                    setattr(existing, k, v)
            existing.computed_at = datetime.utcnow()
            return existing
        obj = IPOFeatures(ipo_id=ipo_id, **{k: v for k, v in features.items() if hasattr(IPOFeatures, k)})
        self.session.add(obj)
        self.session.flush()
        return obj

    def save_prediction(self, ipo_id: int, model_name: str, version: str, pred: dict) -> IPOPrediction:
        # 동일 ipo_id + 모델 조합이면 업데이트
        existing = (
            self.session.query(IPOPrediction)
            .filter_by(ipo_id=ipo_id, model_name=model_name, model_version=version)
            .first()
        )
        if existing:
            for k, v in pred.items():
                if hasattr(existing, k):
                    setattr(existing, k, v)
            return existing
        obj = IPOPrediction(
            ipo_id=ipo_id, model_name=model_name, model_version=version,
            **{k: v for k, v in pred.items() if hasattr(IPOPrediction, k)}
        )
        self.session.add(obj)
        self.session.flush()
        return obj

    def resolve_prediction(self, ipo_id: int, actual_return: float) -> None:
        """상장 후 실제 수익률 채우기"""
        preds = self.session.query(IPOPrediction).filter_by(ipo_id=ipo_id).all()
        for p in preds:
            p.actual_return    = actual_return
            p.prediction_error = actual_return - p.pred_return_pct
            p.direction_correct = (actual_return > 0) == (p.pred_return_pct > 0)
            p.resolved_at      = datetime.utcnow()

        master = self.session.query(IPOMaster).filter_by(id=ipo_id).first()
        if master:
            master.open_return = actual_return

    def get_champion_model(self) -> Optional[ModelRegistry]:
        return (
            self.session.query(ModelRegistry)
            .filter_by(is_champion=True)
            .order_by(ModelRegistry.promoted_at.desc())
            .first()
        )

    def promote_champion(self, model_name: str, version: str) -> None:
        """새 챔피언 모델 승격 (기존 챔피언 retire)"""
        old = self.session.query(ModelRegistry).filter_by(is_champion=True).all()
        for m in old:
            m.is_champion = False
            m.status      = "retired"
            m.retired_at  = datetime.utcnow()

        new = (
            self.session.query(ModelRegistry)
            .filter_by(name=model_name, version=version)
            .first()
        )
        if new:
            new.is_champion  = True
            new.status       = "champion"
            new.promoted_at  = datetime.utcnow()


if __name__ == "__main__":
    engine = get_engine()
    init_db(engine)

    # smoke test
    with get_session(engine) as s:
        repo = IPORepository(s)
        master = repo.upsert_master({
            "corp_code":    "00000001",
            "corp_name":    "테스트기업",
            "market":       "KOSDAQ",
            "offering_price": 15000,
            "listing_date": datetime(2024, 3, 15),
        })
        print(f"IPOMaster 생성: id={master.id}, name={master.corp_name}")

        feats = repo.upsert_features(master.id, {
            "institutional_demand_ratio": 850.0,
            "lockup_6m_ratio": 0.35,
            "lockup_weighted_score": 0.48,
            "offering_price_band_position": 1.05,
            "kospi_momentum_20d": 0.023,
        })
        print(f"IPOFeatures 생성: id={feats.id}")

    print("DB smoke test 통과 ✅")
