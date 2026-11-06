"""Reprocess one changed field from pinned originals, not whole IPO collections."""
import ast
import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from data.collectors.dart_collector import DARTCollector, DEMAND_PARSER_VERSION
from data.pipelines.data_delivery import checked_artifact, digest
from data.pipelines.financial_history import atomic_json
from data.pipelines.historical_reconciliation import connection_contract, feature_tasks, load_inputs
from data.pipelines.historical_event_recovery import capture_sources
from data.pipelines.reconciled_dataset import pin_connections
from features.source_contracts import DART_STRUCTURAL_AGGREGATE_STATUS

BASELINE = '9b4237a'
METHODS = {'_extract_lockup_ratios_from_tables', '_extract_disclosed_aggregate_lockup_ratio',
           '_extract_aggregate_lockup_shares', '_extract_aggregate_demand_ratio'}


def unchanged_other_parser_logic(before, after):
    def remainder(source):
        tree = ast.parse(source)
        tree.body = [n for n in tree.body if not (isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == 'DEMAND_PARSER_VERSION' for t in n.targets))]
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == 'DARTCollector':
                node.body = [n for n in node.body if not (isinstance(n, ast.FunctionDef) and n.name in METHODS)]
        return ast.dump(tree, include_attributes=False)
    return remainder(before) == remainder(after)


def original_body(raw, bundle):
    pieces = []
    for section in bundle.get('sections', []):
        path = raw/'dart_connection_sources'/(hashlib.sha256(section['url'].encode()).hexdigest()+'.json')
        payload = json.loads(path.read_text())
        if (payload['url'] != section['url'] or payload['sha256'] != section['sha256']
                or hashlib.sha256(payload['text'].encode()).hexdigest() != section['sha256']):
            raise RuntimeError('lockup_reprocessing_original_integrity_failed')
        pieces.append(payload['text'])
    if not pieces:
        raise RuntimeError('lockup_reprocessing_section_missing')
    return '\n'.join(pieces)


def documents(raw, record, proof):
    receipt = record['rcept_no']
    primary = next(m for m in proof['family'] if m['rcept_no'] == receipt)
    selected = []
    if proof.get('correction_selection'):
        from data.pipelines.correction_views import after_correction_view
        body, evidence = after_correction_view(original_body(raw, proof))
        if evidence != proof['correction_selection']:
            raise RuntimeError('lockup_reprocessing_correction_view_changed')
        selected.append((primary, proof, body))
    elif (proof.get('content_scope') != 'partial_or_terms_notice'
          and proof.get('price_resolution_status') != 'latest_related_prospectus_price_verified'):
        selected.append((primary, proof, original_body(raw, proof)))
    extra = proof.get('supplementary', {})
    if proof.get('supplementary_status') == 'official_related_prospectus_verified':
        extra_receipt = parse_qs(urlparse(extra['source_url']).query)['rcpNo'][0]
        metadata = next(m for m in extra['family'] if m['rcept_no'] == extra_receipt)
        if pd.Timestamp(metadata['rcept_dt']) >= pd.Timestamp(primary['rcept_dt']):
            selected.append((metadata, extra, original_body(raw, extra)))
    if not selected:
        raise RuntimeError('lockup_reprocessing_no_parent_approved_document')
    return sorted(selected, key=lambda x:(x[0]['rcept_dt'], x[0]['rcept_no']), reverse=True)


def reprocess_record(raw, record, proof, dart, *, allow_reuse=False):
    cutoff = pd.Timestamp(record['listing_date']).normalize()-pd.Timedelta(hours=3)
    selected_documents = documents(raw, record, proof)
    for metadata, _, _ in selected_documents:
        if pd.Timestamp(metadata['rcept_dt'])+pd.Timedelta(days=1) >= cutoff:
            raise RuntimeError('lockup_reprocessing_publication_after_cutoff')
    previous = proof.get('lockup_reprocessing', {})
    attempts_before = previous.get('attempts', [])
    # The parent manifest pins this exact parser. Only a selection change needs
    # extraction again; original_body has already checked every source hash.
    if (allow_reuse and record.get('lockup_parser_version') == DEMAND_PARSER_VERSION
            and previous.get('parser_version') == DEMAND_PARSER_VERSION and attempts_before
            and len(attempts_before) <= len(selected_documents)
            and all(a['receipt']==m['rcept_no'] and a['published']==m['rcept_dt'] and a['sections']==b['sections']
                    for a,(m,b,_) in zip(attempts_before,selected_documents))
            and (record.get('lockup_commitment_ratio') is not None
                 or attempts_before[-1]['parser_status']=='candidate_rejected'
                 or len(attempts_before)==len(selected_documents))):
        reused = json.loads(json.dumps(proof))
        reused['lockup_reprocessing']['extraction_reused'] = True
        return record.copy(), reused
    attempts = []
    chosen = None
    first = None
    for metadata, bundle, body in selected_documents:
        published = pd.Timestamp(metadata['rcept_dt'])
        if published+pd.Timedelta(days=1) >= cutoff:
            raise RuntimeError('lockup_reprocessing_publication_after_cutoff')
        parsed = dart._extract_lockup_ratios_from_tables(body)
        item = (metadata, bundle, parsed)
        first = first or item
        attempts.append({'receipt':metadata['rcept_no'], 'published':metadata['rcept_dt'],
            'source_url':bundle['source_url'], 'sections':bundle['sections'],
            'parser_status':parsed['parser_validation_status'], 'rule':parsed.get('rule_id'),
            'reason':parsed.get('rejection_reason'), 'evidence':parsed.get('structured_evidence')})
        if parsed['parser_validation_status'] == 'candidate_rejected':
            break
        value = parsed['lockup_commitment_ratio']
        if value is not None:
            if (parsed['parser_validation_status'] != 'structurally_verified' or not parsed.get('rule_id')
                    or not math.isfinite(value) or not 0 <= value <= 1):
                raise RuntimeError('lockup_reprocessing_invalid_candidate')
            chosen = item
            break
    metadata, bundle, parsed = chosen or first
    updated = {k:v for k,v in record.items() if not k.startswith('lockup_')}
    updated.update({k:v for k,v in parsed.items() if k.startswith('lockup_')})
    for target, source in [('parse_method','parse_method'), ('parse_evidence','evidence'),
                           ('rule_id','rule_id'), ('parser_validation_status','parser_validation_status'),
                           ('structured_evidence','structured_evidence'), ('rejection_reason','rejection_reason')]:
        updated['lockup_'+target] = parsed.get(source)
    updated.update(lockup_commitment_ratio=parsed['lockup_commitment_ratio'] if chosen else None,
        lockup_validation_status=DART_STRUCTURAL_AGGREGATE_STATUS if chosen else 'dart_aggregate_value_not_verified',
        lockup_rcept_no=metadata['rcept_no'], lockup_rcept_dt=metadata['rcept_dt'],
        lockup_available_at=metadata['rcept_dt'], lockup_source_url=bundle['source_url'],
        lockup_source_sha256='|'.join(sorted(s['sha256'] for s in bundle['sections'])),
        lockup_source_scope='dart_lineage_aggregate_institutional', lockup_parser_version=DEMAND_PARSER_VERSION)
    # All other values and their provenance must be byte-equivalent JSON values.
    if {k:v for k,v in updated.items() if not k.startswith('lockup_')} != {
            k:v for k,v in record.items() if not k.startswith('lockup_')}:
        raise RuntimeError('lockup_reprocessing_outside_field_changed')
    proof = json.loads(json.dumps(proof))
    proof.setdefault('field_sources', {}).pop('lockup_commitment_ratio', None)
    if chosen:
        proof['field_sources']['lockup_commitment_ratio'] = {
            'receipt':metadata['rcept_no'], 'source_url':bundle['source_url'],
            'available_at':(pd.Timestamp(metadata['rcept_dt']).tz_localize('Asia/Seoul')+pd.Timedelta(days=1)).isoformat()}
    proof['lockup_reprocessing'] = {'parser_version':DEMAND_PARSER_VERSION, 'attempts':attempts,
                                   'other_fields_unchanged':True}
    return updated, proof


def reprocess_institutional_record(raw, record, proof, dart):
    cutoff=pd.Timestamp(record['listing_date']).normalize()-pd.Timedelta(hours=3)
    attempts=[];chosen=None;first=None
    for metadata,bundle,body in documents(raw,record,proof):
        if pd.Timestamp(metadata['rcept_dt'])+pd.Timedelta(days=1)>=cutoff:
            raise RuntimeError('institutional_reprocessing_publication_after_cutoff')
        parsed=dart._parse_demand_forecast_html(body,str(record['corp_code']))
        first=first or (metadata,bundle,parsed)
        attempts.append({'receipt':metadata['rcept_no'],'published':metadata['rcept_dt'],
            'sections':bundle['sections'],'parser_status':parsed['institutional_demand_parser_validation_status'],
            'rule':parsed.get('institutional_demand_rule_id'),'reason':parsed.get('institutional_demand_rejection_reason')})
        if parsed['institutional_demand_parser_validation_status']=='candidate_rejected':break
        value=parsed.get('institutional_demand_ratio')
        if value is not None:
            if (parsed['institutional_demand_parser_validation_status']!='structurally_verified'
                    or not parsed.get('institutional_demand_rule_id') or not math.isfinite(value) or value<0):
                raise RuntimeError('institutional_reprocessing_invalid_candidate')
            chosen=(metadata,bundle,parsed);break
    metadata,bundle,parsed=chosen or first
    updated={k:v for k,v in record.items() if not k.startswith('institutional_') or k=='institutional_data_contract_version'}
    updated.update({k:v for k,v in parsed.items() if k.startswith('institutional_demand_')})
    updated.update(institutional_demand_ratio=parsed['institutional_demand_ratio'] if chosen else None,
        institutional_validation_status=DART_STRUCTURAL_AGGREGATE_STATUS if chosen else 'dart_aggregate_value_not_verified',
        institutional_rcept_no=metadata['rcept_no'],institutional_rcept_dt=metadata['rcept_dt'],
        institutional_available_at=metadata['rcept_dt'],institutional_source_url=bundle['source_url'],
        institutional_source_sha256='|'.join(sorted(s['sha256'] for s in bundle['sections'])),
        institutional_source_scope='dart_lineage_aggregate_institutional',institutional_parser_version=DEMAND_PARSER_VERSION)
    if {k:v for k,v in updated.items() if not k.startswith('institutional_')}!={
            k:v for k,v in record.items() if not k.startswith('institutional_')}:
        raise RuntimeError('institutional_reprocessing_outside_field_changed')
    proof=json.loads(json.dumps(proof));proof.setdefault('field_sources',{}).pop('institutional_demand_ratio',None)
    if chosen:
        proof['field_sources']['institutional_demand_ratio']={'receipt':metadata['rcept_no'],
            'source_url':bundle['source_url'],'available_at':(pd.Timestamp(metadata['rcept_dt']).tz_localize('Asia/Seoul')+pd.Timedelta(days=1)).isoformat()}
    proof['institutional_reprocessing']={'parser_version':DEMAND_PARSER_VERSION,'attempts':attempts}
    return updated,proof


def run(raw_dir, processed_dir, dart=None):
    raw, processed = Path(raw_dir).resolve(), Path(processed_dir).resolve()
    if not (processed/'current_historical_source_connections.json').exists():
        return None
    parent, report = pin_connections(processed)
    parser = Path(__file__).parents[1]/'collectors/dart_collector.py'
    code = {'parser':digest(parser), 'repair':digest(__file__)}
    if report.get('lockup_field_reprocessing', {}).get('code') == code:
        return {**report, 'path':str(parent), 'reused':True}
    if report['inputs']['contract']['parser'] == code['parser']:
        return None
    contract = connection_contract()
    previous_contract = report.get('replay_contract', report['inputs']['contract'])
    if {k:v for k,v in contract.items() if k!='parser'} != {
            k:v for k,v in previous_contract.items() if k!='parser'}:
        return None
    baseline = subprocess.check_output(['git','show',BASELINE+':ipo_predictor/data/collectors/dart_collector.py'],
                                       cwd=Path(__file__).parents[3])
    if (hashlib.sha256(baseline).hexdigest() != report['inputs']['contract']['parser']
            or not unchanged_other_parser_logic(baseline.decode(), parser.read_text())):
        return None
    inputs = load_inputs(raw, processed)
    if report['inputs']['inputs'] != inputs['input_sha256']:
        return None
    connections = pd.read_parquet(parent/'connection_results.parquet')
    general = inputs['events'].loc[inputs['events'].event_class.eq('general_ipo'), 'event_id']
    if (not connections.status.eq('official_sources_connected').all() or connections.event_id.duplicated().any()
            or set(connections.event_id)!=set(general)):
        return None
    dart = dart or DARTCollector()
    batch = hashlib.sha256(json.dumps({'parent':digest(parent/'manifest.json'), 'code':code,
                                     'contract':contract},sort_keys=True).encode()).hexdigest()[:20]
    checkpoint_root = raw/'lockup_field_repairs'/batch
    started = pd.Timestamp.now(tz='Asia/Seoul').isoformat()
    audit = []
    for index, row in connections.iterrows():
        old = json.loads(row.record)
        checkpoint = checkpoint_root/(hashlib.sha256(row.event_id.encode()).hexdigest()+'.json')
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            payload = saved['payload']
            if (saved['event_id']!=row.event_id or saved['sha256']!=hashlib.sha256(
                    json.dumps(payload,sort_keys=True).encode()).hexdigest()):
                raise RuntimeError('lockup_field_checkpoint_integrity_failed')
            capture_sources(raw, json.loads(row.proof).get('sources', []), {})
            record, proof = payload['record'], payload['proof']
        else:
            record, proof = reprocess_record(raw, old, json.loads(row.proof), dart,
                allow_reuse=report.get('lockup_field_reprocessing', {}).get('code', {}).get('parser')==code['parser'])
            record, proof = reprocess_institutional_record(raw, record, proof, dart)
            payload = {'record':record, 'proof':proof}
            atomic_json(checkpoint, {'event_id':row.event_id, 'payload':payload,
                'sha256':hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()})
        connections.at[index, 'record'] = json.dumps(record, ensure_ascii=False)
        connections.at[index, 'proof'] = json.dumps(proof, ensure_ascii=False)
        before, after = old.get('lockup_commitment_ratio'), record['lockup_commitment_ratio']
        if {k:v for k,v in old.items() if not k.startswith(('lockup_','institutional_'))}!={
                k:v for k,v in record.items() if not k.startswith(('lockup_','institutional_'))}:
            raise RuntimeError('demand_reprocessing_unrelated_field_changed')
        audit.append({'event_id':row.event_id, 'corp_name':row.corp_name, 'before':before, 'after':after,
            'disposition':'recovered' if before is None and after is not None else
                          'held' if after is None else 'retained_or_revalidated',
            'institutional_before':old.get('institutional_demand_ratio'),
            'institutional_after':record.get('institutional_demand_ratio'),
            'unrelated_fields_unchanged':True, 'receipt':record['lockup_rcept_no'],
            'rule':record['lockup_rule_id'], 'extraction_reused':proof['lockup_reprocessing'].get('extraction_reused',False),
            'attempts':json.dumps(proof['lockup_reprocessing']['attempts'],ensure_ascii=False)})
        if len(audit)%100 == 0:
            print(json.dumps({'lockup_reprocessed':len(audit),'population':len(connections)}), flush=True)
    if (code != {'parser':digest(parser), 'repair':digest(__file__)} or contract!=connection_contract()
            or inputs['input_sha256'] != load_inputs(raw,processed)['input_sha256']):
        raise RuntimeError('lockup_reprocessing_inputs_changed')
    identity = {'parent_manifest':str((parent/'manifest.json').relative_to(processed)),
        'parent_sha256':digest(parent/'manifest.json'), 'code':code,
        'scope':'institutional_and_lockup_fields_other_parser_AST_identical', 'baseline_commit':BASELINE}
    version = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    destination = processed/'historical_reconciliations'/version
    if not destination.exists():
        staging = Path(tempfile.mkdtemp(prefix='.lockup.', dir=destination.parent))
        connections.to_parquet(staging/'connection_results.parquet', index=False)
        pd.read_parquet(parent/'events.parquet').to_parquet(staging/'events.parquet', index=False)
        feature_tasks(connections.to_dict('records')).to_parquet(staging/'feature_tasks.parquet', index=False)
        pd.DataFrame(audit).to_parquet(staging/'lockup_reprocessing.parquet', index=False)
        atomic_json(staging/'source_documents.json', {})
        result = {**report, 'version':version, 'amendment':identity,
            'source_archive_policy':'parent_manifest_archive_plus_amendment_originals',
            'replay_contract':connection_contract(), 'lockup_field_reprocessing':{
                'code':code, 'processed':len(audit), 'dispositions':pd.DataFrame(audit).disposition.value_counts().to_dict(),
                'reprocessed_fields':['institutional_demand_ratio','lockup_commitment_ratio'],
                'unrelated_fields_unchanged':True, 'checkpoint_batch':batch,
                'unchanged_extractions_reused':sum(a['extraction_reused'] for a in audit),
                'started_at':started, 'finished_at':pd.Timestamp.now(tz='Asia/Seoul').isoformat()},
            'output_sha256':{f.name:digest(f) for f in staging.iterdir()}}
        atomic_json(staging/'manifest.json', result)
        staging.rename(destination)
    atomic_json(processed/'current_historical_source_connections.json', {
        'manifest':str((destination/'manifest.json').relative_to(processed)), 'sha256':digest(destination/'manifest.json')})
    return {**checked_artifact(destination), 'path':str(destination)}
