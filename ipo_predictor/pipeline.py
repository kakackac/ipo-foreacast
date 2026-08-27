"""
pipeline.py
────────────
전체 학습 파이프라인 실행 진입점.

실행 모드:
  python pipeline.py --mode train      학습 + 백테스트 + 아웃라이어 분석
  python pipeline.py --mode backtest   백테스트만 재실행
  python pipeline.py --mode analyze    저장된 백테스트 결과로 아웃라이어 재분석
  python pipeline.py --mode demo       실제 데이터 없이 전 과정 시뮬레이션
  python pipeline.py --mode collect    OpenDART + KRX 실제 이력 수집 및 피처 생성

에러 주도 개발 루프:
  1. train 모드로 초기 모델 학습 + 백테스트
  2. 아웃라이어 리포트 확인
  3. 원인이 A(시장 국면)면 → config에서 OPTIONAL 피처 활성화 후 재실행
  4. 성능 개선 확인 → 반복
"""

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

MIN_GENERAL_IPO_TARGET_ROWS = 100
MIN_GENERAL_IPO_YEARS = 3
MIN_GENERAL_IPOS_PER_YEAR = 5
MIN_CORE_FEATURE_COMPLETENESS = 0.70
MIN_CRITICAL_FEATURE_COMPLETENESS = 0.60


class TrainingReadinessError(RuntimeError):
    """공식 일반 IPO 학습의 최소 데이터 품질 기준이 충족되지 않았을 때 발생한다."""


# ── 파이프라인 스텝 ───────────────────────────────────────────

def step_load_or_build_data(demo: bool = False, phase: str = "core"):
    """데이터 로드 또는 데모 데이터 생성"""
    from data.processors.feature_engineer import build_demo_dataset, FeatureEngineer
    from features.definitions import get_core_feature_names, get_phase2_feature_names

    if demo:
        logger.info("── 데모 데이터 생성 (실제 API 없이 전 과정 테스트) ──")
        df = build_demo_dataset(n=600, seed=42, phase=phase)
        logger.info("데모 데이터: %d행 × %d열", len(df), len(df.columns))
    else:
        from config import PROC_DIR
        proc_path = PROC_DIR / "features_all.parquet"
        if not proc_path.exists():
            logger.error("피처 파일 없음: %s", proc_path)
            logger.error("먼저 데이터 수집을 실행하세요 (dart_collector, krx_collector)")
            sys.exit(1)
        df = pd.read_parquet(proc_path)
        logger.info("피처 로드: %d행", len(df))

    return df


def step_feature_selection(df, phase: str = "core", prediction_stage: str = "post_retail"):
    """사용할 피처 컬럼 선택"""
    from features.model_profiles import get_model_profile

    profile = get_model_profile(prediction_stage)
    feat_cols = list(profile.feature_names)
    available = [c for c in feat_cols if c in df.columns]
    missing_indicators = [f"{column}__missing" for column in available if f"{column}__missing" in df.columns]
    available = available + missing_indicators
    missing   = [c for c in feat_cols if c not in df.columns]

    if missing:
        logger.warning("누락 피처 %d개: %s", len(missing), missing[:5])
    logger.info("사용 피처: %d개 (요청 %d개 중, %s)", len(available), len(feat_cols), prediction_stage)
    return available


def assess_training_readiness(
    df: pd.DataFrame, phase: str = "core", prediction_stage: str = "post_retail"
) -> dict:
    """일반 IPO만 대상으로 학습 가능 여부와 미달 사유를 계산한다."""
    from features.model_profiles import get_model_profile

    profile = get_model_profile(prediction_stage)

    report: dict[str, object] = {
        "eligible": False,
        "reasons": [],
        "minimum_general_ipo_target_rows": MIN_GENERAL_IPO_TARGET_ROWS,
        "minimum_years": MIN_GENERAL_IPO_YEARS,
        "minimum_per_year": MIN_GENERAL_IPOS_PER_YEAR,
        "minimum_core_feature_completeness": MIN_CORE_FEATURE_COMPLETENESS,
        "minimum_critical_feature_completeness": MIN_CRITICAL_FEATURE_COMPLETENESS,
        "prediction_stage": profile.name,
        "prediction_stage_description": profile.description,
    }
    if "event_class" not in df.columns:
        report["reasons"].append("공식 이벤트 분류(event_class)가 없는 이전 피처 파일입니다.")
        return report
    general = df[df["event_class"].eq("general_ipo")].copy()
    report["general_ipo_rows"] = int(len(general))
    verified = step_filter_unreviewed_offering_prices(general)
    report["verified_general_ipo_rows"] = int(len(verified))
    target_mask = verified.get("open_return_pct", pd.Series(dtype=float)).notna() & verified.get(
        "close_return_pct", pd.Series(dtype=float)
    ).notna()
    candidates = verified.loc[target_mask].copy()
    report["general_ipo_dual_target_rows"] = int(len(candidates))
    if len(candidates) < MIN_GENERAL_IPO_TARGET_ROWS:
        report["reasons"].append(
            f"일반 IPO의 검증된 시초가·종가 타깃이 {len(candidates)}건으로 최소 {MIN_GENERAL_IPO_TARGET_ROWS}건에 미달합니다."
        )
    if "listing_date" not in candidates.columns:
        report["reasons"].append("상장일이 없어 시간 분할을 검증할 수 없습니다.")
        return report
    candidates["listing_date"] = pd.to_datetime(candidates["listing_date"], errors="coerce")
    yearly = candidates.dropna(subset=["listing_date"]).groupby(candidates["listing_date"].dt.year).size()
    report["yearly_general_ipo_targets"] = {str(year): int(count) for year, count in yearly.items()}
    eligible_years = int((yearly >= MIN_GENERAL_IPOS_PER_YEAR).sum())
    report["years_meeting_minimum"] = eligible_years
    if eligible_years < MIN_GENERAL_IPO_YEARS:
        report["reasons"].append(
            f"연도별 {MIN_GENERAL_IPOS_PER_YEAR}건 이상인 일반 IPO 타깃 연도가 {eligible_years}개뿐입니다."
        )
    core_features = [feature for feature in profile.feature_names if feature in candidates.columns]
    if len(core_features) != len(profile.feature_names):
        missing = sorted(set(profile.feature_names) - set(core_features))
        report["reasons"].append(f"{profile.name} 모델의 피처 열이 완전하지 않습니다: {', '.join(missing)}")
        report["core_feature_completeness"] = 0.0
        report["source_time_validated_core_complete_rows"] = 0
        report["source_time_validated_core_complete_rate"] = 0.0
    else:
        completeness = candidates[core_features].notna().mean()
        strict_complete = candidates[core_features].notna().all(axis=1)
        report["core_feature_completeness_by_feature"] = {
            feature: round(float(rate), 4) for feature, rate in completeness.items()
        }
        report["core_feature_completeness"] = round(float(completeness.mean()), 4)
        report["source_time_validated_core_complete_rows"] = int(strict_complete.sum())
        report["source_time_validated_core_complete_rate"] = round(float(strict_complete.mean()), 4)
        if float(completeness.mean()) < MIN_CORE_FEATURE_COMPLETENESS:
            report["reasons"].append(
                f"핵심 피처 평균 충족률이 {float(completeness.mean()):.1%}로 최소 {MIN_CORE_FEATURE_COMPLETENESS:.0%}에 미달합니다."
            )
        for feature in profile.critical_features:
            if feature not in completeness:
                continue
            rate = float(completeness[feature])
            if rate < MIN_CRITICAL_FEATURE_COMPLETENESS:
                report["reasons"].append(
                    f"핵심 원천 피처 {feature} 충족률이 {rate:.1%}로 최소 "
                    f"{MIN_CRITICAL_FEATURE_COMPLETENESS:.0%}에 미달합니다."
                )
    if "feature_available_at" not in candidates.columns:
        report["reasons"].append("피처 공개시각이 없어 미래 정보 누출을 검증할 수 없습니다.")
    else:
        available_at = pd.to_datetime(candidates["feature_available_at"], errors="coerce")
        violations = available_at.notna() & candidates["listing_date"].notna() & (available_at > candidates["listing_date"])
        report["future_information_violations"] = int(violations.sum())
        if violations.any():
            report["reasons"].append(f"상장일 이후에 공개된 피처 행이 {int(violations.sum())}건 있습니다.")
    report["eligible"] = not report["reasons"]
    return report


def require_training_ready(
    df: pd.DataFrame, phase: str = "core", prediction_stage: str = "post_retail"
) -> pd.DataFrame:
    """학습 안전장치를 적용하고, 통과한 일반 IPO 행만 반환한다."""
    report = assess_training_readiness(df, phase=phase, prediction_stage=prediction_stage)
    if not report["eligible"]:
        details = " | ".join(report["reasons"])
        raise TrainingReadinessError(f"학습·성능평가 차단: {details}")
    return step_filter_unreviewed_offering_prices(
        df[df["event_class"].eq("general_ipo")].copy()
    ).reset_index(drop=True)


def step_filter_unreviewed_offering_prices(df):
    """원문 근거가 부족한 공모가는 보존하되 모델 학습·검증에서는 격리한다."""
    if "offering_price_review_status" not in df.columns:
        raise RuntimeError(
            "공모가 감사 상태가 없는 이전 데이터입니다. collect 모드로 데이터를 다시 수집하세요."
        )
    status = df["offering_price_review_status"].fillna("missing").astype(str)
    usable = status.isin({
        "verified_currency_unit",
        "verified_text_and_structured",
        "verified_structured_api",
        "manual_verified",
    })
    quarantined = int((~usable).sum())
    if quarantined:
        logger.warning("원문 검증 필요 공모가 %d건을 학습·백테스트에서 격리합니다.", quarantined)
    return df[usable].copy().reset_index(drop=True)


def step_backtest(df, feature_cols, target_col: str):
    """Walk-forward 백테스트 실행"""
    from models.evaluation.backtester import WalkForwardBacktester
    from config import BACKTEST_CFG

    logger.info("── %s Walk-Forward 백테스트 시작 ──", target_col)
    bt = WalkForwardBacktester(cfg=BACKTEST_CFG)
    result = bt.run(
        df           = df,
        feature_cols = feature_cols,
        target_col   = target_col,
        model_kwargs = {"n_estimators": 200, "max_depth": 5},
    )
    logger.info(result.summary())
    return result


def step_train_final_model(df, feature_cols, target_col: str, model_name: str):
    """전체 데이터로 최종 모델 학습 후 저장"""
    import pandas as pd
    from models.baseline.gradient_boost_model import IPOPriceModel

    logger.info("── 최종 모델 학습 (전체 데이터) ──")
    valid = df[target_col].notna()
    df_clean = df[valid].reset_index(drop=True)

    available = [c for c in feature_cols if c in df_clean.columns]
    # 최근 15%를 Conformal 보정셋으로 분리
    cal_size = max(10, int(len(df_clean) * 0.15))
    X = df_clean[available]
    y = df_clean[target_col]
    X_tr, X_cal = X.iloc[:-cal_size].copy(), X.iloc[-cal_size:].copy()
    y_tr, y_cal = y.iloc[:-cal_size], y.iloc[-cal_size:]
    fill_values = X_tr.median(numeric_only=True)
    if fill_values.isna().any():
        missing = fill_values[fill_values.isna()].index.tolist()
        raise TrainingReadinessError(f"훈련 구간에서 전부 결측인 피처가 있습니다: {missing}")
    X_tr = X_tr.fillna(fill_values)
    X_cal = X_cal.fillna(fill_values)

    model = IPOPriceModel(n_estimators=300, max_depth=5)
    model.fit(X_tr, y_tr, X_cal, y_cal)
    model.feature_names = available

    path = model.save(model_name)
    logger.info("모델 저장: %s", path)

    # 피처 중요도 출력
    fi = model.get_feature_importance(top_n=15)
    logger.info("\n── 피처 중요도 Top 15 ──────────────")
    for _, row in fi.iterrows():
        bar = "█" * int(row["importance_pct"] / 2)
        logger.info("  %-40s %5.1f%% %s", row["feature"], row["importance_pct"], bar)

    return model


def step_outlier_analysis(backtest_result):
    """아웃라이어 분석 + 피처 추천"""
    from models.evaluation.outlier_analyzer import OutlierAnalyzer

    logger.info("── 아웃라이어 분석 ──")
    analyzer = OutlierAnalyzer()
    report = analyzer.analyze(backtest_result.predictions)

    logger.info(report.summary())

    # 오차 분포
    dist = analyzer.error_distribution(backtest_result.predictions)
    logger.info("\n── 오차 분포 ──")
    logger.info("  평균 절대 오차: %.1f%%", dist["mean"])
    logger.info("  중앙값 오차:    %.1f%%", dist["median"])
    logger.info("  90분위 오차:    %.1f%%", dist["p90"])
    logger.info("  1σ 이내:        %.1f%%", dist["within_1sigma"])
    logger.info("  아웃라이어 수:  %d건", dist["n_outliers"])

    return report


def step_year_analysis(backtest_result):
    """연도별 성능 분석"""
    logger.info("\n── 연도별 성능 ──")
    by_year = backtest_result.by_year()
    for _, row in by_year.iterrows():
        logger.info(
            "  %d년 | n=%3d | MAE=%.1f%% | 방향=%.0f%% | 실제평균=%.1f%% | 예측평균=%.1f%%",
            int(row["year"]), int(row["n"]),
            row["mae"], row["direction_acc"] * 100,
            row["actual_mean"], row["pred_mean"],
        )
    return by_year


def step_save_results(results: dict[str, tuple]):
    """시초가·종가 타깃별 백테스트 결과를 JSON으로 저장"""
    from config import REPORT_DIR
    import json

    summary = {"targets": {}}
    for target_name, (backtest_result, outlier_report) in results.items():
        summary["targets"][target_name] = {
            "backtest": {
                "n_windows":    backtest_result.n_windows,
                "n_predictions": len(backtest_result.predictions),
                "overall_mae":  backtest_result.overall_mae,
                "direction_acc": backtest_result.overall_direction_acc,
                "coverage_90":  backtest_result.coverage_90,
            },
            "outliers": {
                "n_total":    outlier_report.n_total,
                "n_outliers": outlier_report.n_outliers,
                "rate":       outlier_report.outlier_rate,
                "cause_summary": outlier_report.cause_summary,
                "feature_recommendations": outlier_report.feature_recommendations,
            },
        }

    out = REPORT_DIR / "pipeline_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    predictions = pd.concat(
        [backtest_result.predictions for backtest_result, _ in results.values()],
        ignore_index=True,
    )
    predictions.to_parquet(REPORT_DIR / "predictions_latest.parquet", index=False)
    logger.info("결과 저장: %s", out)
    return summary


# ── 메인 실행 ─────────────────────────────────────────────────

def run_demo(phase: str = "core"):
    """데모 모드: 전 과정을 실제 데이터 없이 시뮬레이션"""
    import pandas as pd

    logger.info("════ IPO 예측 파이프라인 — 데모 모드 (%s) ════", phase)

    df           = step_load_or_build_data(demo=True, phase=phase)
    feature_cols = step_feature_selection(df, phase=phase)
    results = {}
    model_names = {
        "open_return_pct": "baseline_open_v1",
        "close_return_pct": "baseline_close_v1",
    }
    for target_col, model_name in model_names.items():
        bt_result = step_backtest(df, feature_cols, target_col)
        _ = step_train_final_model(df, feature_cols, target_col, model_name)
        outlier_rpt = step_outlier_analysis(bt_result)
        step_year_analysis(bt_result)
        results[target_col] = (bt_result, outlier_rpt)
    summary = step_save_results(results)

    logger.info("\n════ 파이프라인 완료 ════")
    for target_col, result in summary["targets"].items():
        metrics = result["backtest"]
        logger.info(
            "  %s | MAE %.2f%% | 방향정확도 %.1f%% | 90%% CI %.1f%%",
            target_col, metrics["overall_mae"], metrics["direction_acc"] * 100,
            metrics["coverage_90"] * 100,
        )

    return summary


def run_train(phase: str = "core", prediction_stage: str = "post_retail"):
    """실제 데이터 학습 모드"""
    import pandas as pd

    logger.info("════ IPO 예측 파이프라인 — 학습 모드 (%s, %s) ════", phase, prediction_stage)
    df           = step_load_or_build_data(demo=False, phase=phase)
    df           = require_training_ready(df, phase=phase, prediction_stage=prediction_stage)
    feature_cols = step_feature_selection(df, phase=phase, prediction_stage=prediction_stage)
    results = {}
    model_names = {
        "open_return_pct": "baseline_open_v1",
        "close_return_pct": "baseline_close_v1",
    }
    for target_col, model_name in model_names.items():
        bt_result = step_backtest(df, feature_cols, target_col)
        _ = step_train_final_model(df, feature_cols, target_col, model_name)
        outlier_rpt = step_outlier_analysis(bt_result)
        step_year_analysis(bt_result)
        results[target_col] = (bt_result, outlier_rpt)
    step_save_results(results)


def run_backtest(phase: str = "core", prediction_stage: str = "post_retail"):
    """실제 데이터로 백테스트만 실행"""
    logger.info("════ IPO 예측 파이프라인 — 백테스트 모드 (%s, %s) ════", phase, prediction_stage)
    df           = step_load_or_build_data(demo=False, phase=phase)
    df           = require_training_ready(df, phase=phase, prediction_stage=prediction_stage)
    feature_cols = step_feature_selection(df, phase=phase, prediction_stage=prediction_stage)
    results = {}
    for target_col in ["open_return_pct", "close_return_pct"]:
        bt_result = step_backtest(df, feature_cols, target_col)
        outlier_rpt = step_outlier_analysis(bt_result)
        step_year_analysis(bt_result)
        results[target_col] = (bt_result, outlier_rpt)
    step_save_results(results)


def run_analyze():
    """기존 백테스트 결과 파일로 아웃라이어만 재분석"""
    from config import REPORT_DIR
    import pandas as pd

    pred_path = REPORT_DIR / "predictions_latest.parquet"
    if not pred_path.exists():
        logger.error("예측 결과 파일 없음: %s", pred_path)
        sys.exit(1)

    preds = pd.read_parquet(pred_path)
    from models.evaluation.outlier_analyzer import OutlierAnalyzer
    report = OutlierAnalyzer().analyze(preds)
    logger.info(report.summary())


def run_collect(start_year: int, end_year: int, phase: str = "phase2"):
    """공식 원천 데이터를 수집해 학습용 피처 파일을 생성한다."""
    from data.pipelines.historical_ipo_pipeline import HistoricalIPOPipeline

    logger.info("════ 실제 IPO 데이터 수집 (%d~%d) ════", start_year, end_year)
    summary = HistoricalIPOPipeline().run(start_year, end_year, feature_set=phase)
    logger.info(
        "수집 완료 | KRX 일정 %d | DART 정합 %d | 학습 행 %d | 시초가 타깃 %d | 종가 타깃 %d",
        summary["calendar_rows"], summary["dart_matched_rows"], summary["feature_rows"],
        summary["open_target_rows"], summary["close_target_rows"],
    )
    return summary


def run_collect_events(start_year: int, end_year: int, force_refresh: bool = False):
    """DART·가격 API 없이 KRX 공식 신규상장 이벤트 마스터만 갱신한다."""
    from data.pipelines.historical_ipo_pipeline import HistoricalIPOPipeline

    logger.info("════ KRX 공식 신규상장 이벤트 수집 (%d~%d) ════", start_year, end_year)
    calendar, manifest = HistoricalIPOPipeline().collect_official_event_master(
        start_year, end_year, force_refresh=force_refresh
    )
    logger.info("공식 이벤트 저장 완료 | %d건 | 실행 매니페스트 %s", len(calendar), manifest["path"])
    return calendar, manifest


def run_audit_dart_failures():
    """저장된 DART 원문 실패를 접수번호별로 다시 감사한다."""
    from data.pipelines.historical_ipo_pipeline import HistoricalIPOPipeline

    audit = HistoricalIPOPipeline().audit_document_failures()
    counts = audit["failure_classification"].value_counts(dropna=False).to_dict() if not audit.empty else {}
    logger.info("DART 원문 실패 재감사 완료 | %d행 | %s", len(audit), counts)
    return audit


if __name__ == "__main__":
    import pandas as pd

    parser = argparse.ArgumentParser(description="IPO 예측 파이프라인")
    parser.add_argument(
        "--mode",
        choices=["train", "backtest", "analyze", "demo", "collect", "collect-events", "audit-dart-failures"],
        default="demo",
        help="실행 모드",
    )
    parser.add_argument(
        "--refresh-events",
        action="store_true",
        help="KRX 공식 이벤트 캐시를 무시하고 요청 기간을 다시 수집",
    )
    parser.add_argument(
        "--phase",
        choices=["core", "phase2"],
        default="core",
        help="피처 세트",
    )
    parser.add_argument(
        "--prediction-stage",
        choices=["pre_demand", "post_demand", "post_retail"],
        default="post_retail",
        help="예측 기준 공개 단계. 실제 학습은 해당 단계의 품질 기준을 통과한 경우에만 허용",
    )
    parser.add_argument("--start-year", type=int, default=2015, help="실제 수집 시작 연도")
    parser.add_argument(
        "--end-year",
        type=int,
        default=date.today().year,
        help="실제 수집 종료 연도 (기본값: 실행일의 연도, 미래 날짜는 제외)",
    )
    args = parser.parse_args()

    if args.mode == "demo":
        run_demo(phase=args.phase)
    elif args.mode == "train":
        run_train(phase=args.phase, prediction_stage=args.prediction_stage)
    elif args.mode == "analyze":
        run_analyze()
    elif args.mode == "backtest":
        run_backtest(phase=args.phase, prediction_stage=args.prediction_stage)
    elif args.mode == "collect":
        try:
            run_collect(args.start_year, args.end_year, phase=args.phase)
        except RuntimeError as exc:
            logger.error("실제 데이터 수집 중단: %s", exc)
            sys.exit(2)
    elif args.mode == "collect-events":
        try:
            run_collect_events(args.start_year, args.end_year, force_refresh=args.refresh_events)
        except RuntimeError as exc:
            logger.error("KRX 공식 이벤트 수집 중단: %s", exc)
            sys.exit(2)
    elif args.mode == "audit-dart-failures":
        try:
            run_audit_dart_failures()
        except RuntimeError as exc:
            logger.error("DART 원문 실패 재감사 중단: %s", exc)
            sys.exit(2)
    else:
        logger.error("지원하지 않는 모드: %s", args.mode)
