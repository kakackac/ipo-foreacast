"""Pin hash-checked research inputs without changing production data or model approvals."""
import hashlib
import json
from pathlib import Path

import pandas as pd

from data.pipelines.financial_history import atomic_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_artifact(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, expected in manifest['output_sha256'].items():
        path = (root / name).resolve()
        if path.parent != root or digest(path) != expected:
            raise RuntimeError('data_delivery_artifact_integrity_failed')
    return manifest


def publish(processed_dir, financial_repair, research_dataset):
    processed = Path(processed_dir).resolve()
    repair, research = Path(financial_repair).resolve(), Path(research_dataset).resolve()
    if not repair.is_relative_to(processed / 'financial_repairs') or not research.is_relative_to(processed / 'research_datasets'):
        raise RuntimeError('data_delivery_path_outside_workspace')
    financial, study = checked_artifact(repair), checked_artifact(research)
    if (financial['input_sha256'] != digest(processed / 'features_all.parquet')
            or study['source_sha256']['features'] != financial['output_sha256']['features_all.parquet']
            or study['source_sha256']['observations'] != financial['output_sha256']['feature_observations.parquet']):
        raise RuntimeError('data_delivery_sources_not_connected')
    data = pd.read_parquet(research / 'dataset.parquet')
    evidence = pd.read_parquet(research / 'evidence.parquet')
    if data.event_id.duplicated().any():
        raise RuntimeError('data_delivery_duplicate_event')
    used = evidence[evidence.used.eq(True) & evidence.event_id.isin(data.event_id)].merge(data[['event_id', 'prediction_at']], on='event_id',
                                                how='left', validate='many_to_one')
    available = pd.to_datetime(used.available_at, utc=True, errors='coerce')
    cutoffs = pd.to_datetime(used.prediction_at, utc=True, errors='coerce')
    violations = available.isna() | cutoffs.isna() | (available >= cutoffs)
    if violations.any():
        raise RuntimeError('data_delivery_future_or_missing_publication_time')
    if checked_artifact(repair) != financial or checked_artifact(research) != study:
        raise RuntimeError('data_delivery_inputs_changed_during_validation')
    identity = {
        'financial_repair': str(repair.relative_to(processed)),
        'financial_manifest_sha256': digest(repair / 'manifest.json'),
        'research_dataset': str(research.relative_to(processed)),
        'research_manifest_sha256': digest(research / 'manifest.json'),
        'source_sha256': study['source_sha256'],
    }
    version = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    report = {**identity, 'version': version, 'scope': 'listing_eve_research_only',
              'rows': len(data), 'year_counts': pd.to_datetime(data.listing_date).dt.year.value_counts().sort_index().to_dict(),
              'financial_status_counts': financial['status_counts'],
              'feature_observed_rows': {f: int(data[f].notna().sum()) for f in study['features']},
              'used_evidence_rows': len(used), 'time_violations': int(violations.sum()),
              'training_executed': False, 'deployment_authorized': False,
              'production_files_replaced': False}
    destination = processed / 'data_deliveries' / version / 'manifest.json'
    if destination.exists():
        if json.loads(destination.read_text()) != json.loads(json.dumps(report)):
            raise RuntimeError('data_delivery_existing_version_changed')
    else:
        atomic_json(destination, report)
    atomic_json(processed / 'current_research_data.json', {
        'manifest': str(destination.relative_to(processed)), 'sha256': digest(destination),
        'training_authorized': False, 'deployment_authorized': False,
    })
    return {**report, 'path': str(destination)}


def current_financial_repair(processed_dir):
    processed = Path(processed_dir).resolve()
    pointer = json.loads((processed / 'current_research_data.json').read_text())
    path = (processed / pointer['manifest']).resolve()
    if not path.is_relative_to(processed / 'data_deliveries') or digest(path) != pointer['sha256']:
        raise RuntimeError('current_research_pointer_integrity_failed')
    entry = json.loads(path.read_text())
    repair = (processed / entry['financial_repair']).resolve()
    research = (processed / entry['research_dataset']).resolve()
    for root, field, folder in [(repair, 'financial_manifest_sha256', 'financial_repairs'),
                                (research, 'research_manifest_sha256', 'research_datasets')]:
        if not root.is_relative_to(processed / folder) or digest(root / 'manifest.json') != entry[field]:
            raise RuntimeError('current_research_artifact_manifest_changed')
        checked_artifact(root)
    if digest(processed / 'features_all.parquet') != json.loads((repair / 'manifest.json').read_text())['input_sha256']:
        raise RuntimeError('current_research_base_input_changed')
    return repair
