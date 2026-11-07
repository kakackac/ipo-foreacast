"""Reconcile stored historical events through one official source contract.

Successful event checkpoints are reusable only with identical identities, parser
code and original-source hashes. Collection never trains or authorizes a model.
"""
import hashlib
import json
import tempfile
from pathlib import Path

import pandas as pd
import requests

from data.collectors.dart_collector import DARTCollector
from data.collectors.krx_collector import KRXCollector
from data.pipelines.data_delivery import checked_artifact, current_financial_repair, digest
from data.pipelines.disclosure_connections import official_family_document
from data.pipelines.event_disclosures import connect_event
from data.pipelines.financial_history import atomic_json
from data.pipelines.historical_event_recovery import capture_sources, discover, select_body_seed, source_record
from features.model_profiles import VERIFIED_OFFERING_PRICE_STATUSES


def json_records(frame):
    return json.loads(frame.to_json(orient='records',date_format='iso'))


def load_inputs(raw_dir, processed_dir):
    raw, processed = Path(raw_dir).resolve(), Path(processed_dir).resolve()
    base = current_financial_repair(processed)
    base_manifest = checked_artifact(base)
    pointer = json.loads((processed/'current_historical_event_recovery.json').read_text())
    path = (processed/pointer['manifest']).resolve()
    if (not path.is_relative_to(processed/'historical_event_recoveries')
            or digest(path)!=pointer['sha256']):
        raise RuntimeError('reconciliation_recovery_pointer_invalid')
    recovery = checked_artifact(path.parent)
    if recovery['input']['base'] != digest(base/'features_all.parquet'):
        raise RuntimeError('reconciliation_financial_and_recovery_bases_differ')
    events = KRXCollector.reclassify_official_listing_events(pd.read_parquet(raw/'krx_official_event_master.parquet'))
    additional_inputs = {}
    if (processed/'current_event_data.json').exists():
        from data.pipelines.event_delivery import current_event_delivery
        delivery, delivery_manifest = current_event_delivery(processed)
        delivered_events = pd.read_parquet(delivery/'events.parquet')
        if not set(events.event_id).issubset(delivered_events.event_id) or delivered_events.event_id.duplicated().any():
            raise RuntimeError('reconciliation_event_delivery_population_conflict')
        events = delivered_events
        additional_inputs['event_delivery_manifest'] = digest(delivery/'manifest.json')
    overrides = pd.read_parquet(path.parent/'event_classification_overrides.parquet')
    for row in overrides.itertuples(index=False):
        selected = events.event_id.eq(row.event_id)
        if selected.sum()!=1:
            raise RuntimeError('reconciliation_classification_event_not_unique')
        events.loc[selected,'event_class'] = row.after
        events.loc[selected,'offering_type'] = ('spac_ipo' if row.after=='spac_ipo' else row.offering_type)
    records = pd.concat([pd.read_parquet(raw/'dart_ipo_raw.parquet'),
                         pd.read_parquet(path.parent/'dart_records.parquet')],ignore_index=True)
    features = pd.read_parquet(path.parent/'features_all.parquet')
    if (processed/'current_event_disclosures.json').exists():
        from data.pipelines.event_disclosures import current_disclosures
        disclosure, disclosure_manifest = current_disclosures(processed)
        if 'delivery_manifest' in locals() and disclosure_manifest['event_version']!=delivery_manifest['version']:
            raise RuntimeError('reconciliation_new_disclosure_event_version_conflict')
        existing = pd.read_parquet(disclosure/'events.parquet')
        missing = existing[~existing.event_id.isin(records.event_id)]
        if not missing.empty:
            extra = missing[['event_id','ticker','listing_date','corp_code','rcept_no']].merge(
                events[['event_id','corp_name','event_class','market']],on='event_id',validate='one_to_one')
            if extra[['corp_code','rcept_no']].isna().any().any():
                raise RuntimeError('reconciliation_existing_disclosure_seed_incomplete')
            records = pd.concat([records,extra],ignore_index=True)
            placeholders = extra.copy()
            placeholders['offering_price_review_status'] = 'verified_currency_unit'
            features = pd.concat([features,placeholders],ignore_index=True)
        additional_inputs['existing_disclosure_manifest'] = digest(disclosure/'manifest.json')
    if records.event_id.duplicated().any() or set(records.event_id)!=set(features.event_id):
        raise RuntimeError('reconciliation_raw_feature_population_mismatch')
    return {'events':events,'records':records,'features':features,'base':base,'recovery':path.parent,
            'input_sha256':{'financial_manifest':digest(base/'manifest.json'),
                'recovery_manifest':digest(path),'events':digest(raw/'krx_official_event_master.parquet'),
                'records':digest(raw/'dart_ipo_raw.parquet'),**additional_inputs},'financial_manifest':base_manifest}


def connection_contract():
    here = Path(__file__)
    paths = {'parser':here.parents[1]/'collectors/dart_collector.py',
             'connection':here.with_name('disclosure_connections.py'),
             'resolver':here.with_name('event_disclosures.py'),
             'record':here.with_name('historical_event_recovery.py'),
             'float':here.with_name('float_source_resolver.py'),
             'float_tables':here.parents[1]/'collectors/float_tables.py',
             'band':here.with_name('band_source_resolver.py'),
             'corrections':here.with_name('correction_views.py'), 'adapter':here}
    result = {name:digest(path) for name,path in paths.items()}
    import ast
    selection = {}
    for filename,names in [('lockup_reconciliation.py',{'documents','original_body'}),
                           ('stage_snapshots.py',{'amendment_view'})]:
        tree = ast.parse(here.with_name(filename).read_text())
        selected = {node.name:ast.dump(node,include_attributes=False) for node in tree.body
                    if isinstance(node,ast.FunctionDef) and node.name in names}
        if set(selected) != names: raise RuntimeError('band_selection_dependency_missing')
        selection.update(selected)
    result['band'] = hashlib.sha256(json.dumps({'resolver':result['band'],
        'selection':selection},sort_keys=True).encode()).hexdigest()
    return result


def checkpoint_identity(event, record, contract):
    return {'event_id':event.event_id,'ticker':str(event.ticker).zfill(6),
            'listing_date':pd.Timestamp(event.listing_date).isoformat(),
            'price':None if pd.isna(event.offering_price) else float(event.offering_price),
            'corp_code':str(record.corp_code),'seed_receipt':str(record.rcept_no), 'contract':contract}


def connect_one(dart, raw, event, record, cached_only):
    candidate = {'corp_code':str(record.corp_code),'rcept_no':str(record.rcept_no)}
    try:
        result = connect_event(dart,raw,event,[candidate],cached_only,verify_current_market=False)
    except ValueError as exc:
        if str(exc)=='price_source_predates_latest_partial_correction':
            from data.pipelines.correction_views import connect_corrected_notice
            return connect_corrected_notice(dart,raw,event,record)[0]
        # Discover a new seed only for linkage failures. Select the latest body
        # from a complete official date-bounded list, never search for a value fit.
        if str(exc) not in {'official_selected_receipt_conflict','official_family_reverse_link_missing',
                           'multiple_offering_families_require_review','latest_official_price_krx_mismatch'}:
            raise
        candidates, identity = discover(dart,raw,event,cached_only)
        if identity['corp_code']!=str(record.corp_code):
            raise ValueError('historical_corp_code_conflict_requires_review')
        seed = select_body_seed(candidates)
        result = connect_event(dart,raw,event,[seed],cached_only,verify_current_market=False)
        result[2]['candidate_discovery'] = identity
        result[2]['sources'].extend(identity['sources'])
    values, metadata, proof, document, _, _ = result
    originals = {}
    capture_sources(raw,proof['sources'],originals)
    parsed = source_record(event,values,metadata,proof,document)
    return {'status':'official_sources_connected','record':json_records(pd.DataFrame([parsed]))[0],
            'proof':proof,'sources':proof['sources'],
            'candidate_values':{key:document.get(key) for key in values},
            'field_missing_reasons':{field:('selected_complete_source_field_unverified' if value is None else None)
                                     for field,value in values.items()}}


def reusable_connections(raw, processed, inputs, contract):
    """Reuse only a complete, source-checked version of the same execution contract."""
    if not (processed/'current_historical_source_connections.json').exists():
        return None
    from data.pipelines.reconciled_dataset import pin_connections
    path, report = pin_connections(processed)
    stored_contract = report.get('replay_contract', report['inputs']['contract'])
    if report['inputs']['inputs'] != inputs['input_sha256'] or stored_contract != contract:
        return None
    rows = pd.read_parquet(path/'connection_results.parquet')
    target = inputs['events'].loc[inputs['events'].event_class.eq('general_ipo'), 'event_id']
    if (rows.event_id.duplicated().any() or set(rows.event_id) != set(target)
            or not rows.status.eq('official_sources_connected').all()):
        return None
    originals = {}
    for row in rows.itertuples(index=False):
        capture_sources(raw, json.loads(row.proof).get('sources', []), originals)
    return {**report, 'path':str(path), 'reused':True}


def feature_tasks(results):
    """Account for every event/field, including optional fields missed by old gates."""
    fields = {
        'offering_price': ('offering_price', 'offering_price_review_status'),
        'institutional_demand_ratio': ('institutional_demand_ratio', 'institutional_demand_parser_validation_status'),
        'lockup_commitment_ratio': ('lockup_commitment_ratio', 'lockup_parser_validation_status'),
        'price_band': ('price_band_low', 'price_band_validation_status'),
        'public_float': ('public_float_ratio_disclosed', 'public_float_resolution_status'),
        'new_shares': ('new_shares', 'new_shares_parser_validation_status'),
        'secondary_shares': ('secondary_shares', 'secondary_shares_parser_validation_status'),
    }
    tasks=[]
    for row in results:
        record=json.loads(row['record']);proof=json.loads(row['proof'])
        for field,(value_key,status_key) in fields.items():
            status=record.get(status_key)
            value=record.get(value_key)
            if field=='public_float' and value is None and record.get('public_float_shares') is not None:
                value=record['public_float_shares']
            reason=(row['reason'] if row['status']!='official_sources_connected' else
                    None if value is not None else status or 'selected_document_no_extraction_record')
            tasks.append({'event_id':row['event_id'],'corp_name':row['corp_name'],'feature':field,
                'processed':row['status']=='official_sources_connected','value_present':value is not None,
                'validation_status':status,'missing_reason':reason,'absence_confirmed':False,
                'attempt':'cached_official_lineage_and_latest_related_prospectus',
                'source_urls':json.dumps([s['url'] for s in proof.get('sources',[])]),
                'source_hashes':json.dumps([s['sha256'] for s in proof.get('sources',[])]),
                'remaining_action':None if value is not None else 'inspect_recorded_source_and_rejection_evidence_not_nonpublication',
                'record_evidence':json.dumps({k:v for k,v in record.items() if k.startswith(
                    ('public_float','total_post_listing'))},ensure_ascii=False) if field=='public_float' else
                    json.dumps({k:v for k,v in record.items() if k.startswith(
                        'institutional_demand_' if field=='institutional_demand_ratio' else
                        'lockup_' if field=='lockup_commitment_ratio' else value_key)},ensure_ascii=False)})
    return pd.DataFrame(tasks)


def collect_connections(raw_dir, processed_dir, dart=None, cached_only=False, event_limit=None):
    raw, processed = Path(raw_dir).resolve(), Path(processed_dir).resolve()
    inputs = load_inputs(raw,processed)
    contract = connection_contract()
    if event_limit is None:
        reused = reusable_connections(raw, processed, inputs, contract)
        if reused is not None:
            return reused
    dart = dart or DARTCollector()
    records = inputs['records'].set_index('event_id')
    features = inputs['features'].set_index('event_id')
    events = inputs['events']
    targets = events[events.event_class.eq('general_ipo') & events.event_id.isin(records.index)].copy()
    targets['price_held'] = ~features.loc[targets.event_id,'offering_price_review_status'].isin(VERIFIED_OFFERING_PRICE_STATUSES).to_numpy()
    targets = targets.sort_values(['price_held','listing_date','event_id'],ascending=[False,True,True])
    population = len(targets)
    if event_limit is not None:
        if event_limit<1: raise ValueError('reconciliation_limit_must_be_positive')
        held = targets[targets.price_held]
        sample = held if len(held) else targets
        positions=sorted({round(i*(len(sample)-1)/max(1,event_limit-1)) for i in range(min(len(sample),event_limit))})
        targets=sample.iloc[positions]
    results, checkpoints, originals, consecutive_errors = [], [], {}, 0
    checkpoint_dir = raw/'historical_reconciliation_connections'
    started = pd.Timestamp.now(tz='Asia/Seoul').isoformat()
    for event in targets.itertuples(index=False):
        record = records.loc[event.event_id]
        identity = checkpoint_identity(event,record,contract)
        key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        path=checkpoint_dir/(key+'.json')
        payload=None
        if path.exists():
            saved=json.loads(path.read_text())
            body_hash=hashlib.sha256(json.dumps(saved['result'],sort_keys=True).encode()).hexdigest()
            if saved['identity']!=identity or body_hash!=saved['result_sha256']:
                raise RuntimeError('reconciliation_checkpoint_integrity_failed')
            if saved['result']['status']=='official_sources_connected':
                capture_sources(raw,saved['result']['sources'],originals)
                payload=saved['result']
        if payload is None:
            try:
                payload=connect_one(dart,raw,event,record,cached_only or consecutive_errors>=3)
                capture_sources(raw,payload['sources'],originals)
                consecutive_errors=0
            except requests.RequestException as exc:
                consecutive_errors+=1
                payload={'status':'source_access_failed','reason':type(exc).__name__,'absence_confirmed':False}
            except ValueError as exc:
                payload={'status':'pending_source_download' if str(exc)=='official_source_not_cached' else 'review_required',
                         'reason':str(exc),'absence_confirmed':False}
            atomic_json(path,{'identity':identity,'result':payload,
                'result_sha256':hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),
                'collected_at':pd.Timestamp.now(tz='Asia/Seoul').isoformat()})
        checkpoints.append({'event_id':event.event_id,'path':str(path),'sha256':digest(path)})
        results.append({'event_id':event.event_id,'corp_name':event.corp_name,
                        'status':payload['status'],'reason':payload.get('reason'),
                        'record':json.dumps(payload.get('record',{}),ensure_ascii=False),
                        'proof':json.dumps(payload.get('proof',{}),ensure_ascii=False),
                        'checkpoint':str(path),'checkpoint_sha256':digest(path)})
        progress={'started_at':started,'population':population,'processed':len(results),
                  'status_counts':pd.Series([row['status'] for row in results]).value_counts().to_dict(),
                  'last_event':event.event_id,'training_executed':False,'input_sha256':inputs['input_sha256']}
        atomic_json(raw/'historical_reconciliation_progress.json',progress)
        if len(results)%10==0 or len(results)==len(targets): print(json.dumps(progress,ensure_ascii=False),flush=True)
    if connection_contract()!=contract:
        raise RuntimeError('reconciliation_connection_code_changed')
    pinned_inputs=load_inputs(raw,processed)['input_sha256']
    if pinned_inputs!=inputs['input_sha256']:
        raise RuntimeError('reconciliation_inputs_changed')
    identity={'inputs':pinned_inputs,'contract':contract,'checkpoints':checkpoints}
    version=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
    destination=processed/'historical_reconciliations'/version
    if not destination.exists():
        destination.parent.mkdir(parents=True,exist_ok=True)
        staging=Path(tempfile.mkdtemp(prefix='.reconcile.',dir=destination.parent))
        pd.DataFrame(results).to_parquet(staging/'connection_results.parquet',index=False)
        feature_tasks(results).to_parquet(staging/'feature_tasks.parquet',index=False)
        inputs['events'].to_parquet(staging/'events.parquet',index=False)
        atomic_json(staging/'source_documents.json',originals)
        report={'version':version,'phase':'source_connections','population':population,'processed':len(results),
                'status_counts':pd.Series([row['status'] for row in results],dtype=object).value_counts().to_dict(),
                'inputs':identity,'started_at':started,'finished_at':pd.Timestamp.now(tz='Asia/Seoul').isoformat(),
                'training_executed':False,'deployment_authorized':False,'production_files_replaced':False,
                'remaining_phases':['financial_reconciliation','feature_contracts','integrated_time_and_unit_validation'],
                'output_sha256':{name:digest(staging/name) for name in ['connection_results.parquet','feature_tasks.parquet','events.parquet','source_documents.json']}}
        atomic_json(staging/'manifest.json',report)
        staging.rename(destination)
    else: report=checked_artifact(destination)
    # A failed batch remains inspectable/resumable but cannot replace a complete
    # source version consumed by the financial and dataset pipelines.
    if event_limit is None and all(row['status']=='official_sources_connected' for row in results):
        atomic_json(processed/'current_historical_source_connections.json',
                    {'manifest':str((destination/'manifest.json').relative_to(processed)),'sha256':digest(destination/'manifest.json')})
    return {**report,'path':str(destination)}
