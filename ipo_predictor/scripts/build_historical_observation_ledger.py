"""Recover reviewed historical observations without replacing frozen model data."""
import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from config import RAW_DIR
from data.collectors import dart_collector
from scripts.audit_official_samples import verify_record


def bind_event(record, lineage, index_content):
    receipt = record['receipt']
    if not re.fullmatch(r'\d{14}', receipt):
        raise ValueError('invalid_receipt')
    matched = lineage[lineage.rcept_no.eq(receipt)].copy()
    identities = set(re.findall(r"openCorpInfoNew\('([0-9]{8})'", index_content))
    if matched.empty or matched.event_id.nunique() != 1 or matched.corp_code.isna().any():
        raise ValueError('event_identity_ambiguous')
    if identities != set(matched.corp_code.astype(str)) or len(identities) != 1:
        raise ValueError('corporate_identity_mismatch')
    published = pd.to_datetime(matched.rcept_dt, errors='coerce').dt.normalize()
    listing = pd.to_datetime(matched.listing_date, errors='coerce').dt.normalize()
    receipt_date = pd.Timestamp(datetime.strptime(receipt[:8], '%Y%m%d'))
    if published.isna().any() or listing.isna().any() or not published.eq(receipt_date).all() or listing.nunique() != 1:
        raise ValueError('publication_or_listing_date_conflict')
    available = (receipt_date + pd.Timedelta(days=1)).tz_localize('Asia/Seoul')
    if not available < listing.iloc[0].tz_localize('Asia/Seoul'):
        raise ValueError('not_available_before_listing')
    return {'event_id': str(matched.event_id.iloc[0]), 'corp_code': next(iter(identities)),
            'listing_date': listing.iloc[0].date().isoformat(),
            'published_date': receipt_date.date().isoformat(), 'available_at': available.isoformat()}


def select_asof(observations, event_id, feature, cutoff):
    """Research lookup only; missing/latest conflicts never backfill older values."""
    moment = pd.Timestamp(cutoff)
    if moment.tzinfo is None:
        raise ValueError('timezone_required')
    rows = [row for row in observations if row['event_id'] == event_id and row['feature_name'] == feature
            and pd.Timestamp(row['available_at']) < moment]
    if not rows:
        return {'status': 'no_reviewed_observation_before_cutoff', 'value': None, 'model_eligible': False}
    latest = max(pd.Timestamp(row['available_at']) for row in rows)
    rows = [row for row in rows if pd.Timestamp(row['available_at']) == latest]
    values = {json.dumps(row['value'], sort_keys=True) for row in rows}
    if len(values) != 1 or any(row['value'] is None for row in rows):
        return {'status': 'latest_observation_missing_or_conflicting', 'value': None, 'model_eligible': False}
    return {'status': 'reviewed_sample_value_not_complete_lineage_snapshot', 'value': rows[0]['value'],
            'evidence_ids': [row['observation_id'] for row in rows], 'model_eligible': False}


def build(raw_dir=RAW_DIR):
    root = Path(raw_dir) / 'official_sample_audit'
    references = {row['sample']: row for row in json.loads(Path(__file__).with_name('official_sample_expectations.json').read_text())['samples']}
    results = json.loads((root / 'results.json').read_text())
    lineage_path = Path(raw_dir) / 'dart_disclosure_lineage.parquet'
    lineage_sha = hashlib.sha256(lineage_path.read_bytes()).hexdigest()
    lineage = pd.read_parquet(lineage_path)
    parser_sha = hashlib.sha256(Path(dart_collector.__file__).read_bytes()).hexdigest()
    observations, rejected = [], []
    for record in results:
        try:
            reference = references[record['sample']]
            if record['status'] != 'sample_passed' or verify_record(record, reference):
                raise ValueError('reference_audit_failed')
            document = Path(record['document_path']).resolve()
            if document.parent != root.resolve() or hashlib.sha256(document.read_bytes()).hexdigest() != record['sha256']:
                raise ValueError('document_hash_mismatch')
            if record['parser_sha256'] != parser_sha:
                raise ValueError('parser_changed_reaudit_required')
            index = (root / (record['sample'] + '_index.html')).read_bytes()
            identity = bind_event(record, lineage, index.decode('utf-8'))
            actual = {**record.get('offering', {}), **record.get('demand', {})}
            for feature in reference['expected']:
                observations.append({**identity, 'observation_id': f"{record['sample']}:{feature}:{record['sha256']}",
                    'feature_name': feature, 'value': actual[feature], 'is_missing': actual[feature] is None,
                    'rcept_no': record['receipt'], 'sample': record['sample'], 'source_url': record['source_url'],
                    'document_sha256': record['sha256'], 'index_sha256': hashlib.sha256(index).hexdigest(),
                    'parser_sha256': parser_sha, 'demand_parser_version': dart_collector.DEMAND_PARSER_VERSION,
                    'validation_status': 'reviewed_sample_value_and_event_identity',
                    'availability_policy': 'publication_date_next_day_KST_conservative',
                    'model_eligible': False})
        except (ValueError, KeyError, OSError) as exc:
            rejected.append({'sample': record.get('sample'), 'status': 'review_required', 'error_type': type(exc).__name__})
    if hashlib.sha256(lineage_path.read_bytes()).hexdigest() != lineage_sha:
        raise RuntimeError('lineage_changed_during_audit')
    return {'created_at': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
            'lineage_sha256': lineage_sha,
            'observations': observations, 'rejected_samples': rejected, 'model_eligible': False,
            'scope': 'reviewed_reference_fields_only_not_complete_stage_snapshots',
            'historical_training_data_modified': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({'observations': len(result['observations']),
                      'events': len({row['event_id'] for row in result['observations']}),
                      'rejected_samples': len(result['rejected_samples']), 'model_approved': 0}))
    sys.exit(1 if result['rejected_samples'] else 0)
