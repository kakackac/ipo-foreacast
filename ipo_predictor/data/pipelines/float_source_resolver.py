"""Select a dated, price-linked float disclosure without reviving superseded values."""
import math
import pandas as pd

FLOAT_FIELDS = (
    "public_float_shares", "public_float_ratio_disclosed", "public_float_parse_method",
    "public_float_parse_evidence", "total_post_listing_shares",
    "total_post_listing_shares_parse_method", "total_post_listing_shares_parse_evidence",
    "total_post_listing_shares_parser_validation_status",
    "public_float_basis_status",
)


def resolve_public_float(documents, offering_price, listing_date):
    result = {field: None for field in FLOAT_FIELDS if field.startswith("public_float_")}
    result.update(public_float_rcept_no=None, public_float_rcept_dt=None,
                  public_float_resolution_status="no_verified_source")
    cutoff = pd.Timestamp(listing_date)
    ordered = sorted(documents, key=lambda pair: (
        str(pair[0].get("rcept_dt", "")), str(pair[0].get("rcept_no", ""))), reverse=True)
    for metadata, document in ordered:
        published = pd.to_datetime(metadata.get("rcept_dt"), errors="coerce")
        if pd.isna(published) or published >= cutoff:
            continue
        method = document.get("public_float_parse_method")
        if method == "conflicting_float_ratios_review_required":
            result.update(public_float_parse_method=method,
                          public_float_parse_evidence=document.get("public_float_parse_evidence"),
                          public_float_resolution_status="latest_conflict_blocks_older_fallback")
            return result
        if method not in ("disclosed_public_float_ratio_direct_context",
                          "disclosed_public_float_shares_direct_context"):
            continue
        # Prospectus candidates are discovered by corp_code; price linkage is also required.
        if metadata.get("supplementary"):
            try:
                matches = math.isfinite(float(offering_price)) and float(offering_price) > 0 and (
                    float(document.get("offering_price")) == float(offering_price))
            except (TypeError, ValueError):
                matches = False
            if not matches:
                result["public_float_resolution_status"] = "latest_source_price_unverified"
                return result
        if method == "disclosed_public_float_shares_direct_context" and document.get(
            "total_post_listing_shares_parser_validation_status"
        ) != "structurally_verified":
            result["public_float_resolution_status"] = "latest_source_denominator_unverified"
            return result
        if (method == "disclosed_public_float_shares_direct_context" and document.get(
                "total_post_listing_shares_parse_method") != "same_table_post_offering_scope_v1"):
            result["public_float_resolution_status"] = "latest_source_denominator_scope_unverified"
            return result
        result.update({field: document.get(field) for field in FLOAT_FIELDS})
        result.update(public_float_rcept_no=str(metadata["rcept_no"]),
                      public_float_rcept_dt=published,
                      public_float_resolution_status="latest_disclosed_source_selected")
        return result
    return result


def refresh_cached_float(raw_dir, processed_dir, dart=None):
    """Apply only a changed float parser to already pinned official IPO originals."""
    import ast
    import hashlib
    import json
    import subprocess
    import tempfile
    from pathlib import Path
    from data.collectors.dart_collector import DARTCollector
    from data.pipelines.data_delivery import checked_artifact, digest
    from data.pipelines.financial_history import atomic_json
    from data.pipelines.historical_event_recovery import capture_sources
    from data.pipelines.historical_reconciliation import connection_contract, feature_tasks, load_inputs
    from data.pipelines.lockup_reconciliation import documents as admitted_documents, parser_snapshot
    from data.pipelines.reconciled_dataset import pin_connections

    raw,processed=Path(raw_dir).resolve(),Path(processed_dir).resolve()
    if not (processed/'current_historical_source_connections.json').exists():return None
    parent,report=pin_connections(processed)
    contract=connection_contract();previous=report.get('replay_contract',report['inputs']['contract'])
    tables_path=Path(__file__).parents[1]/'collectors/float_tables.py'
    parser_snapshot(raw,contract['float'],Path(__file__).read_text())
    parser_snapshot(raw,contract['float_tables'],tables_path.read_text())
    if previous==contract:return None
    owned={'float','float_tables'}
    if {k:v for k,v in previous.items() if k not in owned}!={k:v for k,v in contract.items() if k not in owned}:return None
    baseline='82839b0';repo=Path(__file__).parents[3]
    original_resolver=subprocess.check_output(['git','show',baseline+':ipo_predictor/data/pipelines/float_source_resolver.py'],cwd=repo)
    original_tables=subprocess.check_output(['git','show',baseline+':ipo_predictor/data/collectors/float_tables.py'],cwd=repo)
    parent_resolver=parser_snapshot(raw,previous['float'])
    parent_tables=parser_snapshot(raw,previous['float_tables'])
    if parent_resolver is None and previous['float']==hashlib.sha256(original_resolver).hexdigest():
        parent_resolver=original_resolver.decode()
    if parent_tables is None and previous['float_tables']==hashlib.sha256(original_tables).hexdigest():
        parent_tables=original_tables.decode()
    if (parent_resolver is None or parent_tables is None or
            Path(__file__).read_text().split('\ndef refresh_cached_float(')[0].rstrip()!=
            parent_resolver.split('\ndef refresh_cached_float(')[0].rstrip()):
        return None
    def outside_float_function(text):
        tree=ast.parse(text);tree.body=[node for node in tree.body if not (isinstance(node,ast.FunctionDef) and node.name=='ownership_float')]
        return ast.dump(tree,include_attributes=False)
    if outside_float_function(parent_tables)!=outside_float_function(tables_path.read_text()):return None
    inputs=load_inputs(raw,processed)
    if report['inputs']['inputs']!=inputs['input_sha256']:return None
    rows=pd.read_parquet(parent/'connection_results.parquet')
    targets=set(inputs['events'].loc[inputs['events'].event_class.eq('general_ipo'),'event_id'])
    if set(rows.event_id)!=targets or rows.event_id.duplicated().any() or not rows.status.eq('official_sources_connected').all():return None
    identity={'parent_manifest':str((parent/'manifest.json').relative_to(processed)),'parent_sha256':digest(parent/'manifest.json'),
              'contract':contract,'scope':'float_fields_only_existing_identity_price_and_other_features_unchanged',
              'fallback_baseline_commit':baseline,'parent_parser_source_hashes':{
                  field:previous[field] for field in owned},'repair':digest(__file__)}
    version=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
    checkpoints=raw/'float_field_repairs'/version;dart=dart or DARTCollector();audit=[]
    field_names=set(FLOAT_FIELDS)|{'public_float_rcept_no','public_float_rcept_dt','public_float_resolution_status'}
    def effective_ratio(record):
        if record.get('public_float_resolution_status')!='latest_disclosed_source_selected':return None
        value=record.get('public_float_ratio_disclosed')
        if value is not None:return float(value)
        n,d=record.get('public_float_shares'),record.get('total_post_listing_shares')
        if n is not None and d and record.get('total_post_listing_shares_parser_validation_status')=='structurally_verified':
            return n/d
        return None
    for index,row in rows.iterrows():
        record=json.loads(row.record);proof=json.loads(row.proof)
        if str(record['corp_code'])!=str(proof['corp_code']):raise RuntimeError('float_field_corp_code_conflict')
        capture_sources(raw,proof.get('sources',[]),{})
        checkpoint=checkpoints/(hashlib.sha256(row.event_id.encode()).hexdigest()+'.json')
        if checkpoint.exists():
            saved=json.loads(checkpoint.read_text());payload=saved['payload']
            if saved['event_id']!=row.event_id or saved['sha256']!=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest():
                raise RuntimeError('float_field_checkpoint_integrity_failed')
        else:
            documents=[];origins={}
            for metadata,bundle,body in admitted_documents(raw,record,proof):
                if pd.Timestamp(metadata['rcept_dt'])+pd.Timedelta(days=1)>=pd.Timestamp(record['listing_date']).normalize()-pd.Timedelta(hours=3):
                    raise RuntimeError('float_field_source_after_cutoff')
                parsed=dart._parse_offering_html(body,metadata['rcept_no'])
                meta={**metadata,'supplementary':True}
                documents.append((meta,parsed));origins[metadata['rcept_no']]=(metadata,bundle)
            resolved=dict.fromkeys(FLOAT_FIELDS);resolved.update(resolve_public_float(documents,record['offering_price'],record['listing_date']))
            receipt=resolved['public_float_rcept_no'];resolved.update(source_url=None,available_at=None)
            if receipt is not None:
                metadata,bundle=origins[receipt]
                resolved.update(public_float_rcept_dt=pd.Timestamp(metadata['rcept_dt']).isoformat(),source_url=bundle['source_url'],
                                available_at=(pd.Timestamp(metadata['rcept_dt']).tz_localize('Asia/Seoul')+pd.Timedelta(days=1)).isoformat())
            updated={**record,**{field:resolved.get(field) for field in field_names}}
            if {k:v for k,v in updated.items() if k not in field_names}!={k:v for k,v in record.items() if k not in field_names}:
                raise RuntimeError('float_field_unrelated_value_changed')
            proof['resolved_float']=resolved;proof.setdefault('field_sources',{}).pop('public_float_ratio_disclosed',None)
            if receipt is not None:proof['field_sources']['public_float_ratio_disclosed']={
                'receipt':receipt,'source_url':resolved['source_url'],'available_at':resolved['available_at']}
            payload={'record':updated,'proof':proof}
            atomic_json(checkpoint,{'event_id':row.event_id,'payload':payload,
                'sha256':hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()})
        updated=payload['record']
        if {k:v for k,v in updated.items() if k not in field_names}!={k:v for k,v in record.items() if k not in field_names}:
            raise RuntimeError('float_field_unrelated_value_changed')
        rows.at[index,'record']=json.dumps(updated,ensure_ascii=False);rows.at[index,'proof']=json.dumps(payload['proof'],ensure_ascii=False)
        audit.append({'event_id':row.event_id,'corp_name':row.corp_name,'before':effective_ratio(record),
            'after':effective_ratio(updated),'before_reported_ratio':record.get('public_float_ratio_disclosed'),
            'after_reported_ratio':updated.get('public_float_ratio_disclosed'),'numerator':updated.get('public_float_shares'),
            'denominator':updated.get('total_post_listing_shares'),'resolution_status':updated.get('public_float_resolution_status'),
            'unrelated_fields_unchanged':True,'evidence':updated.get('public_float_parse_evidence')})
        if len(audit)%100==0:print(json.dumps({'float_reprocessed':len(audit),'population':len(rows)}),flush=True)
    if contract!=connection_contract() or identity['repair']!=digest(__file__) or inputs['input_sha256']!=load_inputs(raw,processed)['input_sha256']:
        raise RuntimeError('float_field_inputs_changed')
    destination=processed/'historical_reconciliations'/version
    if not destination.exists():
        staging=Path(tempfile.mkdtemp(prefix='.float.',dir=destination.parent))
        rows.to_parquet(staging/'connection_results.parquet',index=False)
        pd.read_parquet(parent/'events.parquet').to_parquet(staging/'events.parquet',index=False)
        feature_tasks(rows.to_dict('records')).to_parquet(staging/'feature_tasks.parquet',index=False)
        pd.DataFrame(audit).to_parquet(staging/'float_reprocessing.parquet',index=False);atomic_json(staging/'source_documents.json',{})
        amended={**report,'version':version,'amendment':identity,'replay_contract':contract,
            'source_archive_policy':'parent_manifest_archive_plus_amendment_originals','float_field_reprocessing':{
                'processed':len(rows),'unrelated_fields_unchanged':True,'status_counts':pd.DataFrame(audit).resolution_status.value_counts().to_dict()},
            'output_sha256':{file.name:digest(file) for file in staging.iterdir()}}
        atomic_json(staging/'manifest.json',amended);staging.rename(destination)
    atomic_json(processed/'current_historical_source_connections.json',{
        'manifest':str((destination/'manifest.json').relative_to(processed)),'sha256':digest(destination/'manifest.json')})
    return {**checked_artifact(destination),'path':str(destination)}
