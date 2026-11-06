"""Version official event/price deltas without overwriting historical/model inputs."""
import hashlib
import json
import re
import tempfile
from pathlib import Path

import pandas as pd

from data.collectors.krx_collector import KRXCollector
from data.pipelines.data_delivery import checked_artifact, digest
from data.pipelines.financial_history import atomic_json


def identity(frame):
    ticker = frame.ticker.astype(str).str.strip().str.upper().str.zfill(6)
    dates = pd.to_datetime(frame.listing_date, errors='coerce', utc=True).dt.tz_convert('Asia/Seoul').dt.strftime('%Y-%m-%d')
    if not ticker.str.fullmatch('[0-9A-Z]{6}').all() or dates.isna().any():
        raise ValueError('event_trading_identity_invalid')
    return ticker + '|' + dates


def disclosure_candidate(row, registry):
    candidates = [r for r in (registry or {}).get('rows', [])
                  if KRXCollector._normalise_company_name(r['corp_name']) == KRXCollector._normalise_company_name(row['corp_name'])
                  and pd.Timestamp(r['subscription_end']) < pd.Timestamp(row['listing_date'])]
    return {'event_id':row['event_id'], 'candidate_receipts':json.dumps(sorted({r['rcept_no'] for r in candidates})),
            'candidate_corp_codes':json.dumps(sorted({r['corp_code'] for r in candidates})),
            'link_status':'candidate_only_requires_corp_code_and_terms_verification' if candidates else 'official_disclosure_not_linked',
            'model_eligible':False}


def merge_frames(base, snapshot, audit, registry=None):
    for frame in (base, snapshot, audit):
        if frame.event_id.duplicated().any() or identity(frame).duplicated().any():
            raise ValueError('duplicate_event_or_trading_identity')
    if set(snapshot.event_id) != set(audit.event_id):
        raise ValueError('snapshot_audit_population_mismatch')
    source = snapshot.copy()
    proof = audit.set_index('event_id')
    for row in source.itertuples(index=False):
        evidence = proof.loc[row.event_id]
        if identity(pd.DataFrame([row._asdict()])).iloc[0] != identity(pd.DataFrame([evidence.to_dict()])).iloc[0]:
            raise ValueError('snapshot_audit_trading_identity_mismatch')
        if row.market != evidence.price_market or row.market not in ('KOSPI', 'KOSDAQ'):
            raise ValueError('snapshot_price_market_mismatch')
        if evidence.price_match_status != 'matched':
            raise ValueError('snapshot_price_not_matched')
        actual = str(evidence.price_matched_ticker or '')
        if actual in ('None', 'nan', ''):
            actual = str(evidence.price_matched_isu_cd or '')
        if len(actual) != 6 or actual.upper().zfill(6) != str(row.ticker).upper().zfill(6):
            raise ValueError('snapshot_price_trading_code_unverified')
        payload = json.loads(evidence.price_raw_response_evidence)
        if (row.market not in payload or not payload[row.market].get('row_count', 0)
                or not re.fullmatch('[a-f0-9]{64}', str(payload[row.market].get('sha256','')))):
            raise ValueError('snapshot_price_raw_evidence_missing')
        for field, audit_field in [('open_price','listing_open_price'),('close_price','listing_close_price')]:
            value, expected = float(getattr(row, field)), float(evidence[audit_field])
            if not pd.notna(value) or not 0 < value < float('inf') or value != expected:
                raise ValueError('snapshot_price_value_mismatch')
    source = KRXCollector.reclassify_official_listing_events(source)
    base = KRXCollector.reclassify_official_listing_events(base)
    # ISU_CD in daily prices may be a trading code, not an ISIN.
    for frame in (base, source):
        if 'krx_standard_code' not in frame: frame['krx_standard_code'] = None
        valid = frame.krx_standard_code.astype(str).str.fullmatch(r'KR[A-Z0-9]{9}[0-9]', na=False)
        frame['legacy_standard_code_value'] = frame.krx_standard_code.where(~valid)
        frame['krx_standard_code'] = frame.krx_standard_code.where(valid)
        frame['ticker'] = frame.ticker.astype(str).str.upper().str.zfill(6)
        frame['listing_date'] = pd.to_datetime(frame.listing_date).dt.normalize()
    base['_identity'], source['_identity'] = identity(base), identity(source)
    base = base.set_index('_identity', drop=False)
    links, delta = [], []
    for _, row in source.iterrows():
        key = row['_identity']
        if key in base.index:
            old = base.loc[key]
            if old.market != row.market: raise ValueError('event_market_changed_requires_review')
            row['source_event_id'] = row.event_id
            row['event_id'] = old.event_id
            if pd.isna(row.krx_standard_code) and pd.notna(old.krx_standard_code):
                row['krx_standard_code'] = old.krx_standard_code
            combined = old.to_dict()
            combined.update(row.to_dict())
            for column, value in combined.items():
                if column not in base: base[column] = None
                base.at[key,column] = value
        else:
            row['source_event_id'] = row.event_id
            base = pd.concat([base, pd.DataFrame([row], index=[key])])
            delta.append(row.to_dict())
        # Names discover candidates only; they never prove corporate identity.
        links.append(disclosure_candidate(row,registry))
    merged = base.reset_index(drop=True).drop(columns='_identity')
    added = pd.DataFrame(delta, columns=list(source.columns)+['source_event_id']).drop(columns='_identity')
    return merged, added, pd.DataFrame(links)


def run(raw_dir, processed_dir, snapshot_path, price_dir, registry_path=None):
    raw, processed, price_dir = Path(raw_dir), Path(processed_dir), Path(price_dir)
    sources = {'base':raw/'krx_official_event_master.parquet','snapshot':Path(snapshot_path),
               'prices':price_dir/'events_with_prices.parquet','audit':price_dir/'krx_listing_price_audit.parquet',
               'price_manifest':price_dir/'snapshot_prices_manifest.json'}
    if registry_path: sources['registry'] = Path(registry_path)
    hashes = {k:digest(v) for k,v in sources.items()}
    receipt = json.loads(sources['price_manifest'].read_text())
    if receipt['status'] != 'complete' or receipt['snapshot_sha256'] != hashes['snapshot']:
        raise ValueError('price_collection_incomplete_or_snapshot_mismatch')
    all_events = pd.read_parquet(sources['snapshot'])
    prices, audit = pd.read_parquet(sources['prices']), pd.read_parquet(sources['audit'])
    if len(prices) != receipt['result_rows'] or len(prices) != receipt['due_events']:
        raise ValueError('price_collection_row_count_mismatch')
    finished = pd.Timestamp(receipt['finished_at'])
    if finished.tzinfo is None: raise ValueError('price_run_timezone_missing')
    available = pd.to_datetime(prices.listing_date).dt.tz_localize('Asia/Seoul') + pd.Timedelta(hours=21)
    if available.gt(finished).any(): raise ValueError('price_collected_before_listing_day_2100')
    expected = all_events[all_events.event_id.isin(prices.event_id)]
    if set(expected.event_id) != set(prices.event_id): raise ValueError('price_events_outside_snapshot')
    paired = expected[['event_id','ticker','listing_date','market']].merge(
        prices[['event_id','ticker','listing_date','market']],on='event_id',suffixes=('_expected','_actual'),validate='one_to_one')
    if not all(paired[f'{k}_expected'].astype(str).eq(paired[f'{k}_actual'].astype(str)).all() for k in ('ticker','listing_date','market')):
        raise ValueError('price_event_identity_not_bound_to_snapshot')
    registry = json.loads(sources['registry'].read_text()) if registry_path else None
    base = pd.read_parquet(sources['base'])
    merged, added, links = merge_frames(base, prices, audit, registry)
    pending = all_events[~all_events.event_id.isin(prices.event_id)].copy()
    pending['collection_status'] = 'not_collected_in_source_run_recheck_due_time'
    # The official population must not shrink merely because a price is pending.
    pending_events = KRXCollector.reclassify_official_listing_events(pending)
    pending_events = pending_events.loc[~identity(pending_events).isin(set(identity(merged)))].copy()
    pending_events['ticker'] = pending_events.ticker.astype(str).str.upper().str.zfill(6)
    pending_events['listing_date'] = pd.to_datetime(pending_events.listing_date).dt.normalize()
    standard = pending_events.get('krx_standard_code',pd.Series(index=pending_events.index,dtype=object))
    valid = standard.astype(str).str.fullmatch(r'KR[A-Z0-9]{9}[0-9]',na=False)
    pending_events['legacy_standard_code_value'] = standard.where(~valid)
    pending_events['krx_standard_code'] = standard.where(valid)
    pending_events['price_resolution_status'] = 'pending_collection'
    pending_events['open_price'], pending_events['close_price'] = float('nan'),float('nan')
    if len(pending_events):
        merged = pd.concat([merged,pending_events],ignore_index=True)
        added = pd.concat([added,pending_events],ignore_index=True)
        links = pd.concat([links,pd.DataFrame([disclosure_candidate(r,registry) for _,r in pending_events.iterrows()])],ignore_index=True)
    for frame in (merged, added): frame['model_eligible'] = False
    fingerprint = hashlib.sha256(json.dumps({'sources':hashes,'builder':digest(__file__)},sort_keys=True).encode()).hexdigest()[:20]
    root = processed/'event_deliveries'/fingerprint
    if root.exists():
        result = checked_artifact(root)
    else:
        root.parent.mkdir(parents=True,exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix='.events.',dir=root.parent))
        outputs = {'events.parquet':merged,'added_events.parquet':added,'disclosure_links.parquet':links,
                   'pending_prices.parquet':pending,'source_price_events.parquet':prices,
                   'source_price_audit.parquet':audit,'source_event_snapshot.parquet':all_events}
        for name,frame in outputs.items(): frame.to_parquet(staging/name,index=False)
        if any(digest(path)!=hashes[key] for key,path in sources.items()): raise RuntimeError('event_delivery_sources_changed')
        result = {'version':fingerprint,'source_sha256':hashes,'source_price_run_started_at':receipt['started_at'],
                  'source_price_run_finished_at':receipt['finished_at'],'base_rows':len(base),'rows':len(merged),'added_rows':len(added),
                  'processed_price_rows':len(prices),'pending_price_rows':len(pending),
                  'added_class_counts':added.event_class.value_counts().to_dict(),
                  'disclosure_link_counts':links.link_status.value_counts().to_dict(),
                  'output_sha256':{n:digest(staging/n) for n in outputs},
                  'scope':'official_event_price_integration_not_model_dataset',
                  'research_data_replaced':False,'training_executed':False,'deployment_authorized':False}
        atomic_json(staging/'source_price_manifest.json',receipt)
        result['output_sha256']['source_price_manifest.json'] = digest(staging/'source_price_manifest.json')
        atomic_json(staging/'manifest.json',result)
        staging.rename(root)
    atomic_json(processed/'current_event_data.json',{'manifest':str((root/'manifest.json').relative_to(processed)),
                                                  'sha256':digest(root/'manifest.json')})
    return {**result,'path':str(root)}


def current_event_delivery(processed_dir):
    processed = Path(processed_dir).resolve()
    pointer = json.loads((processed/'current_event_data.json').read_text())
    manifest = (processed/pointer['manifest']).resolve()
    if not manifest.is_relative_to(processed/'event_deliveries') or digest(manifest)!=pointer['sha256']:
        raise RuntimeError('event_delivery_pointer_invalid')
    return manifest.parent, checked_artifact(manifest.parent)
