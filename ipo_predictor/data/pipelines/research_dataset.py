"""Complete offline dataset build from accepted source observations; no model fitting."""
import hashlib
import json
import math
import re
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from urllib.parse import urlsplit, parse_qs

import pandas as pd

from features.model_profiles import VERIFIED_OFFERING_PRICE_STATUSES
from features.source_contracts import (APPROVED_INSTITUTIONAL_DEMAND_STATUSES, APPROVED_LOCKUP_STATUSES,
                                      DART_ANNUAL_FINANCIAL_STATUS, DART_STRUCTURAL_AGGREGATE_STATUS)

POLICY = 'listing_eve_2100_core_v2'
FEATURES = ['institutional_demand_ratio', 'lockup_commitment_ratio',
            'kospi_momentum_5d', 'kospi_momentum_20d', 'recent_ipo_avg_return_all',
            'revenue_growth_3y', 'operating_margin', 'debt_ratio']


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def local_time(value):
    try:
        result = pd.Timestamp(value)
        if pd.isna(result):
            return pd.NaT
        return result.tz_localize('Asia/Seoul') if result.tzinfo is None else result.tz_convert('Asia/Seoul')
    except (ValueError, TypeError):
        return pd.NaT


def available_day(value):
    value = local_time(value)
    return value.normalize() + pd.Timedelta(days=1) if pd.notna(value) else pd.NaT


def build_frames(features, observations, index, lineage):
    if features.event_id.duplicated().any() or observations.duplicated(['event_id', 'feature_name']).any():
        raise ValueError('duplicate_event_or_observation')
    index = index.copy()
    index['available_at'] = index.date.map(available_day)
    if index.available_at.isna().any() or index.available_at.duplicated().any():
        raise ValueError('invalid_or_duplicate_market_dates')
    index['close'] = pd.to_numeric(index.close, errors='coerce')
    if not index.close.map(lambda x: number(x) is not None and x > 0).all():
        raise ValueError('invalid_market_prices')
    index = index.sort_values('available_at')
    observed = {(str(r.event_id), r.feature_name): r._asdict() for r in observations.itertuples(index=False)}
    filing_dates = {}
    for key, group in lineage.dropna(subset=['event_id', 'rcept_no', 'rcept_dt']).groupby(['event_id', 'rcept_no']):
        dates = {local_time(value).normalize() for value in group.rcept_dt if pd.notna(local_time(value))}
        receipt = str(key[1])
        if len(dates) == 1 and re.fullmatch(r'\d{14}', receipt):
            published = next(iter(dates))
            if published.strftime('%Y%m%d') == receipt[:8]:
                filing_dates[(str(key[0]), receipt)] = published
    rows, excluded, evidence = [], [], []
    eligible_targets = features[
        features.event_class.eq('general_ipo') & features.market.isin(['KOSPI', 'KOSDAQ'])
        & features.offering_price_review_status.isin(VERIFIED_OFFERING_PRICE_STATUSES)
        & features.price_target_validation_status.eq('official_price_verified')].copy()
    eligible_targets['target_available_at'] = eligible_targets.listing_date.map(available_day)
    eligible_targets = eligible_targets.sort_values(['target_available_at', 'event_id'])
    for _, row in features.iterrows():
        event = str(row.event_id)
        listing = local_time(row.listing_date)
        cutoff = listing.normalize() - pd.Timedelta(hours=3) if pd.notna(listing) else pd.NaT
        reason = None
        if row.event_class != 'general_ipo' or row.market not in ('KOSPI', 'KOSDAQ'):
            reason = 'outside_general_ipo_research_scope'
        elif pd.isna(cutoff):
            reason = 'listing_date_missing'
        elif row.offering_price_review_status not in VERIFIED_OFFERING_PRICE_STATUSES or number(row.offering_price) is None or row.offering_price <= 0:
            reason = 'offering_price_unverified'
        elif row.price_target_validation_status != 'official_price_verified' or any(number(row[x]) is None for x in ('open_return_pct', 'close_return_pct')):
            reason = 'price_targets_unverified'
        price_receipt = row.get('offering_price_rcept_no')
        if pd.isna(price_receipt):
            price_receipt = row.get('rcept_no')
        price_at = available_day(filing_dates.get((event, str(price_receipt))))
        if reason is None and (pd.isna(price_at) or not price_at < cutoff):
            reason = 'offering_price_not_available_at_cutoff'
        values = {}
        row_evidence = []
        contracts = [('institutional_demand_ratio', APPROVED_INSTITUTIONAL_DEMAND_STATUSES),
                     ('lockup_commitment_ratio', APPROVED_LOCKUP_STATUSES)]
        contracts.extend((name, {DART_ANNUAL_FINANCIAL_STATUS}) for name in
                         ('revenue_growth_3y', 'operating_margin', 'debt_ratio'))
        for feature, approved in contracts:
            obs = observed.get((event, feature), {})
            value = number(obs.get('raw_value'))
            financial = feature in ('revenue_growth_3y', 'operating_margin', 'debt_ratio')
            published = local_time(obs.get('available_at')) if financial else available_day(obs.get('available_at'))
            receipt_bound = True
            if obs.get('validation_status') == DART_STRUCTURAL_AGGREGATE_STATUS:
                ref = str(obs.get('source_reference', ''))
                parsed = urlsplit(ref)
                receipt = ref if re.fullmatch(r'\d{14}', ref) else (
                    parse_qs(parsed.query).get('rcpNo', [''])[0]
                    if parsed.hostname == 'dart.fss.or.kr' and parsed.scheme == 'https' else '')
                actual_date = filing_dates.get((event, receipt))
                receipt_bound = (actual_date is not None
                                 and available_day(actual_date) == published)
            range_valid = value is not None and (financial or value >= 0)
            if feature == 'debt_ratio':
                range_valid = value is not None and value >= 0
            accepted = (range_valid and (feature != 'lockup_commitment_ratio' or value <= 1)
                        and obs.get('human_review_required') == False and obs.get('is_missing') == False
                        and obs.get('validation_status') in approved
                        and isinstance(obs.get('source_reference'), str) and bool(obs['source_reference'].strip())
                        and receipt_bound
                        and pd.notna(published) and pd.notna(cutoff) and published < cutoff
                        and number(row.get(feature)) is not None
                        and math.isclose(value, number(row.get(feature)), rel_tol=1e-9, abs_tol=1e-9))
            values[feature] = value if accepted else None
            row_evidence.append({'event_id': event, 'feature': feature, 'used': bool(accepted),
                                 'source_reference': obs.get('source_reference'),
                                 'available_at': published.isoformat() if pd.notna(published) else None,
                                 'reason': 'approved_asof_observation' if accepted else 'missing_unverified_or_after_cutoff'})
        if reason is None and values['institutional_demand_ratio'] is None:
            reason = 'required_institutional_observation_unverified_at_cutoff'
        if reason:
            excluded.append({'event_id': event, 'reason': reason})
            continue
        row_evidence.append({'event_id': event, 'feature': 'offering_price', 'used': True,
                             'source_reference': str(price_receipt), 'available_at': price_at.isoformat(),
                             'reason': 'approved_price_receipt_bound_to_event_lineage'})
        past = index[index.available_at < cutoff]
        for window in (5, 20):
            field = f'kospi_momentum_{window}d'
            values[field] = float(past.close.iloc[-1] / past.close.iloc[-1-window] - 1) if len(past) > window else None
            row_evidence.append({'event_id': event, 'feature': field, 'used': values[field] is not None,
                                 'available_at': past.available_at.iloc[-1].isoformat() if len(past) > window else None,
                                 'source_reference': 'KRX_kospi_daily_close', 'reason': 'recomputed_at_cutoff'})
        history = eligible_targets[(eligible_targets.target_available_at < cutoff)
                                   & eligible_targets.open_return_pct.map(lambda x: number(x) is not None)].tail(10)
        values['recent_ipo_avg_return_all'] = float(history.open_return_pct.mean()) if len(history) else None
        row_evidence.append({'event_id': event, 'feature': 'recent_ipo_avg_return_all', 'used': bool(len(history)),
                             'available_at': history.target_available_at.max().isoformat() if len(history) else None,
                             'source_reference': '|'.join(history.event_id.astype(str)), 'reason': 'recomputed_from_prior_verified_targets'})
        rows.append({'event_id': event, 'listing_date': listing.isoformat(), 'prediction_at': cutoff.isoformat(),
                     'prediction_time_source': POLICY, **values,
                     'open_return_pct': float(row.open_return_pct), 'close_return_pct': float(row.close_return_pct),
                     'split': 'development' if listing.year < 2026 else 'previously_exposed_2026_not_fresh_test'})
        evidence.extend(row_evidence)
    dataset = pd.DataFrame(rows, columns=['event_id', 'listing_date', 'prediction_at', 'prediction_time_source',
                                        *FEATURES, 'open_return_pct', 'close_return_pct', 'split'])
    for feature in FEATURES:
        dataset[feature] = pd.to_numeric(dataset[feature], errors='coerce')
        dataset[feature + '__missing'] = dataset[feature].isna()
    return dataset, pd.DataFrame(excluded, columns=['event_id', 'reason']), pd.DataFrame(evidence)


def run(raw_dir, processed_dir, financial_repair=None):
    raw, processed = Path(raw_dir), Path(processed_dir)
    sources = {'features': processed / 'features_all.parquet', 'observations': processed / 'feature_observations.parquet',
               'index': raw / 'kospi_index.parquet', 'lineage': raw / 'dart_disclosure_lineage.parquet'}
    if financial_repair is not None:
        repair = Path(financial_repair)
        manifest = json.loads((repair / 'manifest.json').read_text())
        if manifest['input_sha256'] != hashlib.sha256(sources['features'].read_bytes()).hexdigest():
            raise RuntimeError('financial_repair_base_input_mismatch')
        for name, digest in manifest['output_sha256'].items():
            if hashlib.sha256((repair / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError('financial_repair_output_hash_mismatch')
        sources['features'] = repair / 'features_all.parquet'
        sources['observations'] = repair / 'feature_observations.parquet'
    hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in sources.items()}
    frames = {key: pd.read_parquet(path) for key, path in sources.items()}
    dataset, excluded, evidence = build_frames(**frames)
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != hashes[key] for key, path in sources.items()):
        raise RuntimeError('inputs_changed_during_build')
    fingerprint = hashlib.sha256(json.dumps({'sources': hashes, 'policy': POLICY,
        'accepted_statuses': sorted(VERIFIED_OFFERING_PRICE_STATUSES | APPROVED_INSTITUTIONAL_DEMAND_STATUSES
                                   | APPROVED_LOCKUP_STATUSES | {DART_ANNUAL_FINANCIAL_STATUS}),
        'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, sort_keys=True).encode()).hexdigest()[:20]
    root = processed / 'research_datasets' / fingerprint
    if root.exists():
        manifest = json.loads((root / 'manifest.json').read_text())
        for name, digest in manifest['output_sha256'].items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError('existing_output_hash_mismatch')
        return {**manifest, 'path': str(root), 'reused': True}
    destination = root
    root.parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=f'.{fingerprint}.', dir=root.parent))
    outputs = {'dataset.parquet': dataset, 'development.parquet': dataset[dataset.split.eq('development')],
               'excluded.parquet': excluded, 'evidence.parquet': evidence}
    for name, frame in outputs.items():
        frame.to_parquet(root / name, index=False)
    manifest = {'status': 'complete', 'policy': POLICY, 'created_at': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
                'source_sha256': hashes, 'rows': len(dataset), 'development_rows': len(outputs['development.parquet']),
                'excluded_reasons': excluded.reason.value_counts().to_dict(), 'features': FEATURES,
                'optional_missing_counts': dataset[FEATURES].isna().sum().to_dict(),
                'output_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in outputs},
                'training_executed': False, 'deployment_authorized': False,
                'scope': 'accepted_stored_observations_listing_eve_only_not_pre_demand_or_new_source_audit'}
    (root / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    root.rename(destination)
    return {**manifest, 'path': str(destination), 'reused': False}
