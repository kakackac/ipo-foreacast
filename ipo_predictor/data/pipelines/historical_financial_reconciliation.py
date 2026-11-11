"""Reprocess official annual sections without replacing the production version."""
import hashlib
import json
import errno
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from data.collectors.dart_collector import DARTCollector
from data.pipelines.data_delivery import checked_artifact, digest
from data.pipelines.disclosure_financials import collect_disclosure_financials, financial_document_structure_contract
from data.pipelines.financial_history import FINANCIAL_STATUS, atomic_json, get_annual_history, summarize_asof
from data.pipelines.historical_reconciliation import load_inputs
from data.pipelines.reconciliation_publication import reference_path


def missing_category(status, details):
    if status == FINANCIAL_STATUS:
        return 'not_missing'
    if 'access_failed' in status or 'not_cached' in status:
        return 'source_access_failed'
    if status == 'official_financial_section_without_tables':
        return 'official_section_without_value_tables_requires_lineage_review'
    if status in {'official_statement_unit_conflict_requires_review','official_statement_amount_conflict_requires_review'}:
        return status
    if status=='summary_statement_basis_unverified':
        return 'statement_basis_unverified'
    if 'format_unhandled' in status:
        if details.get('official_statement_unit_conflicts'):
            return 'official_statement_unit_conflict_requires_review'
        rejected=details.get('rejected_table_or_column_counts',{})
        detailed=details.get('detailed_statement_rejection_counts',{})
        if rejected.get('accounting_basis_unverified') or detailed.get('accounting_basis_unverified'):
            return 'accounting_basis_unverified'
        if details.get('selected_document_explicit_summary_absence'):
            return 'summary_absent_detailed_schema_unhandled'
        return 'format_unsupported'
    return 'annual_basis_identity_or_integrity_unverified'


def unresolved_event_ids(processed, fallback):
    """Use the latest verified cohort, not the original 632-row repair's hold list."""
    pointer=reference_path(Path(processed)/'current_reconciliation_run.json')
    if not pointer.exists():return set(fallback.loc[fallback.missing_category.ne('not_missing'),'event_id']),set(fallback.event_id),fallback
    reference=json.loads(pointer.read_text());manifest=(Path(processed)/reference['manifest']).resolve()
    if not manifest.is_relative_to(Path(processed).resolve()/'reconciliation_runs') or digest(manifest)!=reference['sha256']:
        raise RuntimeError('financial_previous_run_pointer_invalid')
    run=json.loads(manifest.read_text());financial=Path(run['financial']).resolve()
    if (not financial.is_relative_to(Path(processed).resolve()/'historical_financial_reconciliations')
            or digest(financial/'manifest.json')!=run['input_sha256']['financial']):
        raise RuntimeError('financial_previous_resolution_pointer_invalid')
    checked_artifact(financial)
    resolutions=pd.read_parquet(financial/'resolutions.parquet')
    if resolutions.event_id.duplicated().any():raise RuntimeError('financial_previous_resolution_duplicate_event')
    return set(resolutions.loc[resolutions.missing_category.ne('not_missing'),'event_id']),set(resolutions.event_id),resolutions


def connected_lineage(connections):
    rows=[]
    for row in connections.itertuples(index=False):
        if row.status!='official_sources_connected':raise RuntimeError('financial_connection_unverified')
        proof=json.loads(row.proof);record=json.loads(row.record)
        if str(proof['corp_code'])!=str(record['corp_code']):raise RuntimeError('financial_connection_corp_code_conflict')
        for bundle in [proof,proof.get('supplementary',{})]:
            rows.extend({'event_id':row.event_id,'corp_code':proof['corp_code'],**entry,
                'filing_report_nm':entry['report_nm']} for entry in bundle.get('family',[]))
    return pd.DataFrame(rows).drop_duplicates(['event_id','corp_code','rcept_no'])


def reuse_inputs(raw, targets, records, lineage, inputs, code):
    events=[{'event_id':r.event_id,'corp_code':str(records.loc[r.event_id].corp_code),
             'corp_name':r.corp_name,'listing_date':pd.Timestamp(r.listing_date).isoformat()}
            for r in targets.sort_values('event_id').itertuples(index=False)]
    files={};future_cache=False
    for folder in ('dart_connection_sources','dart_ipo_financial_sections','dart_financial_sources',
                   'dart_financial_parsed','financial_document_geometry'):
        for path in sorted((raw/folder).rglob('*.json')):
            files[str(path.relative_to(raw))]=digest(path)
            if folder=='dart_financial_sources':
                try:
                    stamp=pd.to_datetime(json.loads(path.read_text()).get('collected_at'),utc=True)
                    future_cache=future_cache or bool(pd.notna(stamp) and stamp>pd.Timestamp.now(tz='UTC'))
                except (ValueError,TypeError,KeyError):
                    pass
    lineage_json=lineage.sort_values(['event_id','corp_code','rcept_no']).to_json(orient='records',date_format='iso')
    return {'schema':1,'inputs':inputs,'code':code,
            'events_sha256':hashlib.sha256(json.dumps(events,sort_keys=True).encode()).hexdigest(),
            'lineage_sha256':hashlib.sha256(lineage_json.encode()).hexdigest(),
            'raw_sources_sha256':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),
            'raw_files':len(files),'future_annual_cache_present':future_cache}


def reuse_financials(processed, signature, connection_hash):
    pointer=reference_path(processed/'current_reconciliation_run.json')
    if not pointer.exists():return None
    from data.pipelines.reconciliation_publication import committed_reconciliation
    previous=committed_reconciliation(processed,reference=json.loads(pointer.read_text()))
    parent=previous['paths']['financial'];report=checked_artifact(parent)
    if (not report.get('reuse_eligible') or report.get('reuse_inputs')!=signature
            or signature['future_annual_cache_present']):return None
    if report['connection_manifest']==connection_hash:
        return {**report,'path':str(parent),'reused':True,'reused_events':report['population']}
    identity={'parent_manifest':digest(parent/'manifest.json'),'connection_manifest':connection_hash,
              'reuse_inputs':signature,'policy':'unchanged_financial_dependencies_rebound_v1'}
    version=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
    destination=processed/'historical_financial_reconciliations'/version
    if not destination.exists():
        staging=Path(tempfile.mkdtemp(prefix='.financial-reuse.',dir=destination.parent))
        for name in report['output_sha256']:
            try:os.link(parent/name,staging/name)
            except OSError as exc:
                if exc.errno!=errno.EXDEV:raise
                shutil.copyfile(parent/name,staging/name)
        atomic_json(staging/'manifest.json',{**report,'version':version,'connection_manifest':connection_hash,
            'rebound_from':str(parent.relative_to(processed)),'rebound_parent_sha256':identity['parent_manifest'],
            'reuse_reason':identity['policy']})
        staging.rename(destination)
    return {**checked_artifact(destination),'path':str(destination),'reused':True,'reused_events':report['population']}


def run(raw_dir, processed_dir, *, unresolved_only=False, dart=None):
    raw,processed=Path(raw_dir).resolve(),Path(processed_dir).resolve()
    inputs=load_inputs(raw,processed)
    resolutions=pd.read_parquet(inputs['base']/'source_resolutions.parquet')
    records=inputs['records'].set_index('event_id')
    events=inputs['events']
    targets=events[events.event_class.eq('general_ipo') & events.event_id.isin(records.index)].copy()
    pointer=json.loads(reference_path(processed/'current_historical_source_connections.json').read_text())
    manifest=(processed/pointer['manifest']).resolve()
    if not manifest.is_relative_to(processed/'historical_reconciliations') or digest(manifest)!=pointer['sha256']:
        raise RuntimeError('financial_connection_pointer_invalid')
    report=checked_artifact(manifest.parent)
    if report['inputs']['inputs']!=inputs['input_sha256']:
        raise RuntimeError('financial_connection_inputs_changed')
    connections=pd.read_parquet(manifest.parent/'connection_results.parquet')
    if set(connections.event_id)!=set(targets.event_id) or connections.event_id.duplicated().any():
        raise RuntimeError('financial_connection_population_incomplete')
    lineage=connected_lineage(connections)
    connection_hash=digest(manifest)
    scope='all_connected_general_ipo_financial_sections'
    if unresolved_only:
        held,previous,resolutions=unresolved_event_ids(processed,resolutions)
        targets=targets[targets.event_id.isin(held) | ~targets.event_id.isin(previous)]
        scope='latest_financial_unresolved_only_not_full_version'
    dart=dart or DARTCollector()
    parser=Path(__file__).with_name('disclosure_financials.py')
    code_paths={'parser':parser,'summary':Path(__file__).with_name('financial_history.py'),'runner':Path(__file__),
                'source_selection':Path(__file__).with_name('financial_source_selection.py'),
                'source_identity':Path(__file__).with_name('disclosure_connections.py'),
                'correction_views':Path(__file__).with_name('correction_views.py'),
                'source_contracts':Path(__file__).parents[2]/'features/source_contracts.py',
                'publication':Path(__file__).with_name('reconciliation_publication.py')}
    code={name:digest(path) for name,path in code_paths.items()}
    code['document_structure']=financial_document_structure_contract()
    code['pandas_version']=pd.__version__
    signature=reuse_inputs(raw,targets,records,lineage,inputs['input_sha256'],code)
    if not unresolved_only:
        reused=reuse_financials(processed,signature,connection_hash)
        if (reused is not None and signature==reuse_inputs(raw,targets,records,lineage,inputs['input_sha256'],code)
                and all(code[name]==digest(path) for name,path in code_paths.items())
                and code['document_structure']==financial_document_structure_contract()
                and code['pandas_version']==pd.__version__
                and inputs['input_sha256']==load_inputs(raw,processed)['input_sha256']):
            print(json.dumps({'financial_reused':reused['reused_events'],'financial_version':reused['version'],
                              'calculation_repeated':False}),flush=True)
            return reused
    rows,accounts,summaries,originals=[],[],[],{}
    started=pd.Timestamp.now(tz='Asia/Seoul').isoformat()
    for event in targets.sort_values(['listing_date','event_id']).itertuples(index=False):
        identity=records.loc[event.event_id]
        cutoff=pd.Timestamp(event.listing_date).normalize()-pd.Timedelta(hours=3)
        details={}
        extra,status=collect_disclosure_financials(dart,raw,event.event_id,str(identity.corp_code),
                                                   event.corp_name,cutoff,lineage,details)
        receipt=details.get('receipt')
        for proof in details.get('financial_source_selection',{}).get('sources',[]):
            path=raw/'dart_connection_sources'/(hashlib.sha256(proof['url'].encode()).hexdigest()+'.json')
            saved=json.loads(path.read_text())
            if (saved['url']!=proof['url'] or saved['sha256']!=proof['sha256']
                    or hashlib.sha256(saved['text'].encode()).hexdigest()!=proof['sha256']):
                raise RuntimeError('financial_reconciliation_selection_proof_integrity_failed')
            originals[proof['url']]=saved
        section=raw/'dart_ipo_financial_sections'/f'{receipt}.json'
        if details.get('document_sha256') and section.exists():
            payload=json.loads(section.read_text())
            if (hashlib.sha256(payload['html'].encode()).hexdigest()!=details['document_sha256']
                    or payload.get('official_corp_code')!=str(identity.corp_code)):
                raise RuntimeError('financial_reconciliation_original_integrity_failed')
            originals[payload['source_url']]=payload
            index=raw/'dart_connection_sources'/(hashlib.sha256(payload['index_url'].encode()).hexdigest()+'.json')
            source=json.loads(index.read_text())
            if hashlib.sha256(source['text'].encode()).hexdigest()!=payload['index_sha256']:
                raise RuntimeError('financial_reconciliation_index_integrity_failed')
            originals[payload['index_url']]=source
        history=get_annual_history(dart,str(identity.corp_code),event.listing_date,raw/'dart_financial_sources',cached_only=True)
        if not extra.empty:
            history=pd.concat([history,extra],ignore_index=True) if not history.empty else extra
        summary=summarize_asof(history,str(identity.corp_code),cutoff)
        if summary['financial_time_validation_status']!=FINANCIAL_STATUS:
            summary['financial_time_validation_status']=status if extra.empty else 'annual_basis_or_integrity_unverified'
        summaries.append({'event_id':event.event_id,**summary})
        if not history.empty:
            accounts.append(history.assign(event_id=event.event_id))
        rows.append({'event_id':event.event_id,'previous_category':(
                resolutions.set_index('event_id').missing_category.get(event.event_id,'not_previously_processed')),
            'result_status':summary['financial_time_validation_status'],
            'missing_category':missing_category(summary['financial_time_validation_status'],details),
            'absence_confirmed':False,'source_evidence':json.dumps(details,ensure_ascii=False)})
        atomic_json(raw/'historical_financial_reconciliation_progress.json',{
            'scope':scope,'population':len(targets),'processed':len(rows),'started_at':started,
            'status_counts':pd.Series([r['result_status'] for r in rows]).value_counts().to_dict()})
        if len(rows)%10==0 or len(rows)==len(targets):
            print(json.dumps({'financial_processed':len(rows),'population':len(targets),
                'status_counts':pd.Series([r['result_status'] for r in rows]).value_counts().to_dict()},ensure_ascii=False),flush=True)
    if (any(code[name]!=digest(path) for name,path in code_paths.items())
            or code['document_structure']!=financial_document_structure_contract()
            or code['pandas_version']!=pd.__version__
            or inputs['input_sha256']!=load_inputs(raw,processed)['input_sha256']):
        raise RuntimeError('financial_reconciliation_code_changed')
    after_signature=reuse_inputs(raw,targets,records,lineage,inputs['input_sha256'],code)
    eligible=(not unresolved_only and signature==after_signature and not signature['future_annual_cache_present']
              and all(r['missing_category'] in {'not_missing','official_statement_amount_conflict_requires_review',
                                                'official_statement_unit_conflict_requires_review'} for r in rows))
    identity={'inputs':inputs['input_sha256'],'connection_manifest':connection_hash,'code':code,'scope':scope,
              'resolutions':rows,'summaries':summaries,'reuse_inputs':after_signature,'reuse_eligible':eligible}
    version=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
    destination=processed/'historical_financial_reconciliations'/version
    if not destination.exists():
        destination.parent.mkdir(parents=True,exist_ok=True)
        staging=Path(tempfile.mkdtemp(prefix='.financial.',dir=destination.parent))
        pd.DataFrame(rows).to_parquet(staging/'resolutions.parquet',index=False)
        pd.DataFrame(summaries).to_parquet(staging/'summaries.parquet',index=False)
        (pd.concat(accounts,ignore_index=True) if accounts else pd.DataFrame()).to_parquet(staging/'accounts.parquet',index=False)
        atomic_json(staging/'source_documents.json',originals)
        manifest={'version':version,'scope':scope,'population':len(targets),'processed':len(rows),
            'inputs':inputs['input_sha256'],'connection_manifest':connection_hash,'code':code,
            'reuse_inputs':after_signature,'reuse_eligible':eligible,
            'status_counts':pd.Series([r['result_status'] for r in rows]).value_counts().to_dict(),
            'missing_categories':pd.Series([r['missing_category'] for r in rows]).value_counts().to_dict(),
            'started_at':started,'finished_at':pd.Timestamp.now(tz='Asia/Seoul').isoformat(),
            'training_executed':False,'production_files_replaced':False,
            'output_sha256':{name:digest(staging/name) for name in ['resolutions.parquet','summaries.parquet','accounts.parquet','source_documents.json']}}
        atomic_json(staging/'manifest.json',manifest)
        staging.rename(destination)
    return {**checked_artifact(destination),'path':str(destination)}
