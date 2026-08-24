"""
pipeline.py
────────────
전체 학습 파이프라인 실행 진입점.

실행 모드:
  python pipeline.py --mode train      학습 + 백테스트 + 아웃라이어 분석
  python pipeline.py --mode backtest   백테스트만 재실행
  python pipeline.py --mode analyze    저장된 백테스트 결과로 아웃라이어 재분석
  python pipeline.py --mode demo       실제 데이터 없이 전 과정 시뮬레이션

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


def step_feature_selection(df, phase: str = "core"):
    """사용할 피처 컬럼 선택"""
    from features.definitions import get_core_feature_names, get_phase2_feature_names

    if phase == "core":
        feat_cols = get_core_feature_names()
    else:
        feat_cols = get_phase2_feature_names()

    available = [c for c in feat_cols if c in df.columns]
    missing   = [c for c in feat_cols if c not in df.columns]

    if missing:
        logger.warning("누락 피처 %d개: %s", len(missing), missing[:5])
    logger.info("사용 피처: %d개 (요청 %d개 중)", len(available), len(feat_cols))
    return available


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
    X = df_clean[available].fillna(0)
    y = df_clean[target_col]

    # 최근 15%를 Conformal 보정셋으로 분리
    cal_size = max(10, int(len(X) * 0.15))
    X_tr, X_cal = X.iloc[:-cal_size], X.iloc[-cal_size:]
    y_tr, y_cal = y.iloc[:-cal_size], y.iloc[-cal_size:]

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


def run_train(phase: str = "core"):
    """실제 데이터 학습 모드"""
    import pandas as pd

    logger.info("════ IPO 예측 파이프라인 — 학습 모드 (%s) ════", phase)
    df           = step_load_or_build_data(demo=False, phase=phase)
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
    step_save_results(results)


def run_backtest(phase: str = "core"):
    """실제 데이터로 백테스트만 실행"""
    logger.info("════ IPO 예측 파이프라인 — 백테스트 모드 (%s) ════", phase)
    df           = step_load_or_build_data(demo=False, phase=phase)
    feature_cols = step_feature_selection(df, phase=phase)
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


if __name__ == "__main__":
    import pandas as pd

    parser = argparse.ArgumentParser(description="IPO 예측 파이프라인")
    parser.add_argument(
        "--mode",
        choices=["train", "backtest", "analyze", "demo"],
        default="demo",
        help="실행 모드",
    )
    parser.add_argument(
        "--phase",
        choices=["core", "phase2"],
        default="core",
        help="피처 세트",
    )
    args = parser.parse_args()

    if args.mode == "demo":
        run_demo(phase=args.phase)
    elif args.mode == "train":
        run_train(phase=args.phase)
    elif args.mode == "analyze":
        run_analyze()
    elif args.mode == "backtest":
        run_backtest(phase=args.phase)
    else:
        logger.error("지원하지 않는 모드: %s", args.mode)
