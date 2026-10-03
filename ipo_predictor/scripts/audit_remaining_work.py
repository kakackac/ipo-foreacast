"""Snapshot historical data quality and release blockers without training or mutation."""
import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from config import PROC_DIR
from data.processors.feature_engineer import FeatureEngineer
from features.model_profiles import MODEL_PROFILES, build_stage_dataset
from pipeline import assess_training_readiness


def audit():
    paths = {name: PROC_DIR / name for name in ('features_all.parquet', 'feature_time_validation.parquet')}
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    frame = pd.read_parquet(paths['features_all.parquet'])
    times = pd.read_parquet(paths['feature_time_validation.parquet'])
    observations = FeatureEngineer(feature_set='phase2').build_feature_observations(frame)
    counts = []
    for name, group in observations.groupby('feature_name'):
        counts.append({'feature': name, 'rows': len(group), 'missing': int(group.is_missing.sum()),
                       'observed_review_required': int((~group.is_missing & group.human_review_required).sum()),
                       'missing_reasons': group.loc[group.is_missing].missing_reason.value_counts().to_dict()})
    stage_reports = {}
    for name in MODEL_PROFILES:
        stage = build_stage_dataset(frame, name, times)
        stage_reports[name] = assess_training_readiness(stage, prediction_stage=name, feature_time_audit=times)
    consumed = sorted(str(p.relative_to(PROC_DIR)) for p in (PROC_DIR / 'experiments').rglob('final_evaluation_started.json'))
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != hashes[name] for name, path in paths.items()):
        raise RuntimeError('Input changed during audit')
    return {'generated_at': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
            'input_sha256': hashes, 'historical_rows': len(frame),
            'year_counts': pd.to_datetime(frame.listing_date).dt.year.value_counts().sort_index().to_dict(),
            'event_classes': frame.event_class.value_counts(dropna=False).to_dict(),
            'duplicate_event_rows': int(frame.event_id.duplicated().sum()),
            'features': counts, 'stages': stage_reports, 'consumed_evaluation_markers': consumed,
            'training_executed': False, 'historical_data_changed': False, 'deployment_authorized': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, default=str)
    print(json.dumps({'rows': result['historical_rows'], 'duplicates': result['duplicate_event_rows'],
                      'stage_cutoff_verified': {k: v.get('actual_stage_cutoff_verified_rows', 0) for k, v in result['stages'].items()},
                      'training_eligible': {k: v['eligible'] for k, v in result['stages'].items()}}, ensure_ascii=False))
