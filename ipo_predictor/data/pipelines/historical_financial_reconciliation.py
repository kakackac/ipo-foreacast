"""Reprocess official annual sections without replacing the production version."""
import hashlib
import json
import tempfile
from pathlib import Path

import pandas as pd

from data.collectors.dart_collector import DARTCollector
from data.pipelines.data_delivery import checked_artifact, digest
from data.pipelines.disclosure_financials import collect_disclosure_financials
from data.pipelines.financial_history import FINANCIAL_STATUS, atomic_json, get_annual_history, summarize_asof
from data.pipelines.historical_reconciliation import load_inputs


def missing_category(status, details):
    if status == FINANCIAL_STATUS:
        return 'not_missing'
    if 'access_failed' in status or 'not_cached' in status:
        return 'source_access_failed'
    if details.get('selected_document_explicit_summary_absence'):
        return 'summary_explicitly_absent_detailed_statements_not_supported'
    if 'format_unhandled' in status:
        rejected=details.get('rejected_table_or_column_counts',{})
        return 'accounting_basis_unverified' if rejected.get('accounting_basis_unverified') else 'format_unsupported'
    return 'annual_basis_identity_or_integrity_unverified'


def run(raw_dir, processed_dir, *, unresolved_only=False, dart=None):
    raw,processed=Path(raw_dir).resolve(),Path(processed_dir).resolve()
    inputs=load_inputs(raw,processed)
    resolutions=pd.read_parquet(inputs['base']/'source_resolutions.parquet')
    records=inputs['records'].set_index('event_id')
    events=inputs['events']
    targets=events[events.event_class.eq('general_ipo') & events.event_id.isin(records.index)].copy()
    lineage=pd.read_parquet(raw/'dart_disclosure_lineage.parquet')
    connection_hash=None
    if unresolved_only:
        targets=targets[targets.event_id.isin(resolutions.loc[resolutions.missing_category.ne('not_missing'),'event_id'])]
        scope='previous_financial_unresolved_only_not_full_version'
    else:
        pointer=json.loads((processed/'current_historical_source_connections.json').read_text())
        manifest=(processed/pointer['manifest']).resolve()
        if not manifest.is_relative_to(processed/'historical_reconciliations') or digest(manifest)!=pointer['sha256']:
            raise RuntimeError('financial_connection_pointer_invalid')
        report=checked_artifact(manifest.parent)
        if report['inputs']['inputs']!=inputs['input_sha256']:
            raise RuntimeError('financial_connection_inputs_changed')
        connections=pd.read_parquet(manifest.parent/'connection_results.parquet')
        if set(connections.event_id)!=set(targets.event_id):
            raise RuntimeError('financial_connection_population_incomplete')
        verified_lineage=[]
        for row in connections.itertuples(index=False):
            if row.status!='official_sources_connected':
                continue
            proof=json.loads(row.proof)
            for bundle in [proof,proof.get('supplementary',{})]:
                verified_lineage.extend({'event_id':row.event_id,'corp_code':proof['corp_code'],
                    **entry,'filing_report_nm':entry['report_nm']} for entry in bundle.get('family',[]))
        lineage=pd.DataFrame(verified_lineage).drop_duplicates(['event_id','corp_code','rcept_no'])
        connection_hash=digest(manifest)
        scope='all_connected_general_ipo_financial_sections'
    dart=dart or DARTCollector()
    parser=Path(__file__).with_name('disclosure_financials.py')
    code={'parser':digest(parser),'summary':digest(Path(__file__).with_name('financial_history.py')),'runner':digest(Path(__file__))}
    rows,accounts,summaries,originals=[],[],[],{}
    started=pd.Timestamp.now(tz='Asia/Seoul').isoformat()
    for event in targets.sort_values(['listing_date','event_id']).itertuples(index=False):
        identity=records.loc[event.event_id]
        cutoff=pd.Timestamp(event.listing_date).normalize()-pd.Timedelta(hours=3)
        details={}
        extra,status=collect_disclosure_financials(dart,raw,event.event_id,str(identity.corp_code),
                                                   event.corp_name,cutoff,lineage,details)
        receipt=details.get('receipt')
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
    if (code['parser']!=digest(parser) or code['summary']!=digest(Path(__file__).with_name('financial_history.py'))
            or code['runner']!=digest(Path(__file__)) or inputs['input_sha256']!=load_inputs(raw,processed)['input_sha256']):
        raise RuntimeError('financial_reconciliation_code_changed')
    identity={'inputs':inputs['input_sha256'],'connection_manifest':connection_hash,'code':code,'scope':scope,
              'resolutions':rows,'summaries':summaries}
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
            'status_counts':pd.Series([r['result_status'] for r in rows]).value_counts().to_dict(),
            'missing_categories':pd.Series([r['missing_category'] for r in rows]).value_counts().to_dict(),
            'started_at':started,'finished_at':pd.Timestamp.now(tz='Asia/Seoul').isoformat(),
            'training_executed':False,'production_files_replaced':False,
            'output_sha256':{name:digest(staging/name) for name in ['resolutions.parquet','summaries.parquet','accounts.parquet','source_documents.json']}}
        atomic_json(staging/'manifest.json',manifest)
        staging.rename(destination)
    return {**checked_artifact(destination),'path':str(destination)}
