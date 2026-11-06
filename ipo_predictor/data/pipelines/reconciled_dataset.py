"""Publish a pinned data version, never an implicit model/deployment approval."""
import hashlib
import json
import tempfile
import re
import math
from pathlib import Path

import pandas as pd

from data.pipelines.data_delivery import checked_artifact, digest
from data.pipelines.financial_history import atomic_json
from data.pipelines.historical_reconciliation import load_inputs
from data.pipelines.research_dataset import local_time
from data.processors.feature_engineer import FeatureEngineer
from features.model_profiles import FEATURE_SOURCE_APPROVAL
from features.model_profiles import VERIFIED_OFFERING_PRICE_STATUSES
from features.source_contracts import DART_UNDERWRITER_TIER_STATUS

FINANCIAL_VALUES=('revenue','operating_income','net_income','total_assets','total_liabilities','equity',
                  'eps','revenue_growth_3y','operating_margin','debt_ratio','offering_per','per_vs_sector_median')


def market_asof(features,index,name,covered_through=None):
    index=index.copy().sort_values('date')
    index['available_at']=index.date.map(lambda value:local_time(value).normalize()+pd.Timedelta(days=1))
    last=local_time(covered_through if covered_through is not None else index.date.max()).normalize()
    for row in features.itertuples():
        cutoff=local_time(row.listing_date).normalize()-pd.Timedelta(hours=3)
        required_end=cutoff.normalize().tz_localize(None)-pd.Timedelta(days=1)
        pending=len(pd.bdate_range(last.tz_localize(None)+pd.Timedelta(days=1),required_end))>0
        past=index[index.available_at<cutoff]
        for window in (5,20,60):
            field=f'{name}_momentum_{window}d'
            if field not in features:continue
            enough=len(past)>window
            features.at[row.Index,field]=(float(past.close.iloc[-1]/past.close.iloc[-1-window]-1)
                                         if not pending and enough else float('nan'))
            features.at[row.Index,field+'_available_at']=(past.available_at.iloc[-1].tz_localize(None)
                                                         if not pending and enough else pd.NaT)
            features.at[row.Index,field+'_availability_is_effective']=True
            features.at[row.Index,field+'_missing_reason']=('market_context_not_collected' if pending else
                                                           'insufficient_verified_index_history' if not enough else None)
    return features


def normalise_observation_times(frame):
    frame=frame.copy()
    for column in ('listing_date','available_at','collected_at'):
        if column in frame:
            frame[column]=pd.to_datetime(frame[column].map(local_time),utc=True)
    return frame


def prior_targets_asof(features):
    eligible=features[features.event_class.eq('general_ipo')
        & features.offering_price_review_status.isin(VERIFIED_OFFERING_PRICE_STATUSES)
        & features.price_target_validation_status.eq('official_price_verified')
        & features.open_return_pct.notna()].copy()
    eligible['target_available_at']=eligible.listing_date.map(lambda value:local_time(value).normalize()+pd.Timedelta(hours=21))
    eligible=eligible.sort_values(['target_available_at','event_id'])
    for row in features.itertuples():
        cutoff=local_time(row.listing_date).normalize()-pd.Timedelta(hours=3)
        history=eligible[eligible.target_available_at<cutoff].tail(10)
        features.at[row.Index,'recent_ipo_avg_return_all']=float(history.open_return_pct.mean()) if len(history) else float('nan')
        features.at[row.Index,'recent_ipo_avg_return_all_available_at']=(history.target_available_at.max().tz_localize(None)
                                                                 if len(history) else pd.NaT)
        features.at[row.Index,'recent_ipo_avg_return_all_source_event_ids']=json.dumps(history.event_id.tolist())
    return features


def pin_connections(processed):
    pointer=json.loads((processed/'current_historical_source_connections.json').read_text())
    path=(processed/pointer['manifest']).resolve()
    if not path.is_relative_to(processed/'historical_reconciliations') or digest(path)!=pointer['sha256']:
        raise RuntimeError('reconciled_connection_pointer_invalid')
    report=checked_artifact(path.parent)
    seen={path}
    previous=report
    while 'amendment' in previous:
        amendment=previous['amendment'];parent=(processed/amendment['parent_manifest']).resolve()
        if (not parent.is_relative_to(processed/'historical_reconciliations') or parent in seen
                or digest(parent)!=amendment['parent_sha256']):
            raise RuntimeError('reconciled_connection_parent_invalid')
        seen.add(parent);previous=checked_artifact(parent.parent)
    return path.parent,report


def mask_unapproved(features, observations):
    """Retain raw candidates, but only pass separately approved values to models."""
    features=features.copy()
    observations=observations.copy()
    observations['model_eligible']=False
    observations['model_exclusion_reason']=None
    features=features.set_index('event_id',drop=False)
    for feature in observations.feature_name.unique():
        if pd.api.types.is_bool_dtype(features[feature]):features[feature]=features[feature].astype('Float64')
    for index,row in observations.iterrows():
        feature,event=row.feature_name,row.event_id
        value=features.at[event,feature]
        cutoff=local_time(features.at[event,'listing_date'])-pd.Timedelta(hours=3)
        available=local_time(row.available_at)
        if (pd.notna(available) and available==available.normalize()
                and row.get('available_at_is_effective',False)!=True):
            available+=pd.Timedelta(days=1)
        contract=FEATURE_SOURCE_APPROVAL.get(feature)
        approved=(not row.human_review_required and pd.notna(available) and available<cutoff)
        reason=('value_missing' if pd.isna(value) else 'source_contract_unapproved' if row.human_review_required else
                'availability_unverified_or_after_cutoff' if not approved else None)
        if contract:
            approved=approved and contract[0] in features and features.at[event,contract[0]] in contract[1]
            if not approved and reason is None:reason='source_contract_unapproved'
        if feature in {'major_shareholder_lockup_months','risk_factor_count','recent_ipo_avg_return_sector',
                       'offering_per','per_vs_sector_median','same_day_ipo_count'}:
            approved=False
            reason='pre_listing_derivation_or_share_basis_not_verified'
        if feature=='underwriter_tier' and features.loc[event].get('underwriter_validation_status')!=DART_UNDERWRITER_TIER_STATUS:
            approved=False;reason='underwriter_pre_listing_document_not_verified'
        if 'event_class' in features and features.at[event,'event_class']!='general_ipo':
            approved=False;reason='outside_general_ipo_reconciliation_scope'
        observations.at[index,'model_eligible']=bool(approved and pd.notna(value))
        observations.at[index,'model_exclusion_reason']=reason
        if pd.notna(value) and not approved:
            features.at[event,feature]=float('nan')
        features.at[event,feature+'__missing']=pd.isna(features.at[event,feature])
    observations['model_value']=[features.at[r.event_id,r.feature_name] for r in observations.itertuples(index=False)]
    return features.reset_index(drop=True),observations


def validate_snapshot(features,observations):
    if (features.event_id.duplicated().any() or observations.duplicated(['event_id','feature_name']).any()
            or not set(observations.event_id).issubset(features.event_id)):
        raise RuntimeError('reconciled_snapshot_identity_violation')
    names=set(FeatureEngineer('phase2').feature_names)
    if (set(observations.feature_name)!=names or set(observations.event_id)!=set(features.event_id)
            or len(observations)!=len(features)*len(names)):
        raise RuntimeError('reconciled_snapshot_observation_lattice_incomplete')
    lookup=features.set_index('event_id')
    for row in observations.itertuples(index=False):
        event=lookup.loc[row.event_id]
        value=event[row.feature_name]
        missing=pd.isna(value)
        if (bool(event[row.feature_name+'__missing'])!=missing
                or pd.isna(row.model_value)!=missing
                or bool(row.model_eligible)==missing
                or (not missing and not math.isclose(float(value),float(row.model_value),rel_tol=1e-10,abs_tol=1e-10))):
            raise RuntimeError('reconciled_snapshot_ledger_value_or_missing_flag_mismatch')
    time_violations=value_violations=scope_violations=0
    used=observations[observations.model_eligible]
    for row in used.itertuples(index=False):
        event=lookup.loc[row.event_id]
        available=local_time(row.available_at)
        if (pd.notna(available) and available==available.normalize()
                and row.available_at_is_effective!=True):available+=pd.Timedelta(days=1)
        cutoff=local_time(event.listing_date)-pd.Timedelta(hours=3)
        time_violations+=int(pd.isna(available) or not available<cutoff)
        value=event[row.feature_name]
        value_violations+=int(pd.isna(value) or not math.isfinite(float(value))
                             or not math.isclose(float(value),float(row.model_value),rel_tol=1e-10,abs_tol=1e-10))
        scope_violations+=int(event.event_class!='general_ipo')
    if time_violations or value_violations or scope_violations:
        raise RuntimeError('reconciled_snapshot_time_value_or_scope_violation')
    return {'status':'snapshot_valid_with_recorded_holds','rows':len(features),'observations':len(observations),
            'approved_observations':len(used),'time_violations':time_violations,'value_violations':value_violations,
            'non_general_model_observations':scope_violations,'storage_timezone':'UTC',
            'product_stage_or_training_authorization':False}


def run(raw_dir,processed_dir,financial_version):
    raw,processed=Path(raw_dir).resolve(),Path(processed_dir).resolve()
    inputs=load_inputs(raw,processed)
    paths={'builder':Path(__file__),'engineer':Path(__file__).parents[1]/'processors/feature_engineer.py',
           'source_contracts':Path(__file__).parents[2]/'features/source_contracts.py',
           'model_profiles':Path(__file__).parents[2]/'features/model_profiles.py',
           'feature_definitions':Path(__file__).parents[2]/'features/definitions.py',
           'underwriter_source':Path(__file__).with_name('underwriter_source.py'),
           'kospi':raw/'kospi_index.parquet','kosdaq':raw/'kosdaq_index.parquet'}
    pinned={k:digest(p) for k,p in paths.items()}
    source_path,source_report=pin_connections(processed)
    financial_path=Path(financial_version).resolve()
    if not financial_path.is_relative_to(processed/'historical_financial_reconciliations'):
        raise RuntimeError('reconciled_financial_version_outside_store')
    financial=checked_artifact(financial_path)
    if (source_report['inputs']['inputs']!=inputs['input_sha256']
            or financial['inputs']!=inputs['input_sha256']
            or financial['connection_manifest']!=digest(source_path/'manifest.json')
            or financial['scope']!='all_connected_general_ipo_financial_sections'):
        raise RuntimeError('reconciled_version_inputs_or_scope_mismatch')
    connections=pd.read_parquet(source_path/'connection_results.parquet')
    general=inputs['events'].loc[inputs['events'].event_class.eq('general_ipo'),'event_id']
    summaries=pd.read_parquet(financial_path/'summaries.parquet')
    if (set(connections.event_id)!=set(general) or set(summaries.event_id)!=set(general)
            or connections.event_id.duplicated().any() or summaries.event_id.duplicated().any()):
        raise RuntimeError('reconciled_general_population_incomplete')
    originals=inputs['records'].set_index('event_id',drop=False)
    records=[]
    for row in connections.itertuples(index=False):
        if row.status=='official_sources_connected':
            records.append(json.loads(row.record))
        else:
            identity=originals.loc[row.event_id]
            records.append({key:identity[key] for key in ('event_id','corp_code','corp_name','listing_date','rcept_no')})
    untouched=inputs['records'][~inputs['records'].event_id.isin(general)]
    candidates=pd.concat([untouched,pd.DataFrame(records)],ignore_index=True).copy()
    for field in FINANCIAL_VALUES:
        candidates[field]=float('nan')
    for field in ['financial_feature_provenance','financial_time_validation_status','financial_available_at']:
        candidates[field]=None
    candidates=candidates.set_index('event_id',drop=False)
    for row in summaries.to_dict('records'):
        for key,value in row.items():
            if key!='event_id':
                if key not in candidates:candidates[key]=None
                candidates.at[row['event_id'],key]=value
    candidates=candidates.reset_index(drop=True)
    for column in candidates:
        if column=='listing_date' or column.endswith(('_rcept_dt','_available_at')) or column=='rcept_dt':
            candidates[column]=candidates[column].map(lambda value:local_time(value).tz_localize(None)
                if pd.notna(local_time(value)) else pd.NaT)
    engineer=FeatureEngineer('phase2')
    features=engineer.build_features(candidates,inputs['events'],pd.read_parquet(raw/'kospi_index.parquet'),
                                    pd.read_parquet(raw/'kosdaq_index.parquet'))
    financial_columns=[c for c in summaries if c!='event_id']
    features=features.drop(columns=[c for c in financial_columns if c in features],errors='ignore').merge(
        summaries,on='event_id',how='left',validate='one_to_one')
    for field in engineer.feature_names:
        if field not in features:features[field]=float('nan')
    from data.pipelines.underwriter_source import from_connection
    features=features.set_index('event_id',drop=False)
    for row in connections.itertuples(index=False):
        if row.status!='official_sources_connected':continue
        underwriter=from_connection(raw,json.loads(row.record),json.loads(row.proof))
        for key,value in underwriter.items():
            if key not in features:features[key]=None
            features.at[row.event_id,key]=value
    features=features.reset_index(drop=True)
    from data.pipelines.event_model_inputs import collect_days
    from data.collectors.krx_collector import KRXCollector
    seed=pd.read_parquet(raw/'kospi_index.parquet')
    market_index,days,context=collect_days(KRXCollector(),raw,seed,inputs['events'][inputs['events'].event_id.isin(general)],cached_only=True)
    covered=max([pd.Timestamp(seed.date.max()),*[pd.Timestamp(day) for day in days]])
    if context['status']=='market_index_all_empty_requires_review':covered=pd.Timestamp(seed.date.max())
    features=market_asof(features,market_index,'kospi',covered)
    features=market_asof(features,pd.read_parquet(raw/'kosdaq_index.parquet'),'kosdaq')
    features=prior_targets_asof(features)
    observations=engineer.build_feature_observations(features)
    proofs={row.event_id:json.loads(row.proof) for row in connections.itertuples(index=False)
            if row.status=='official_sources_connected'}
    for row in observations.itertuples():
        if row.event_id not in proofs or row.feature_name in FINANCIAL_VALUES or not str(row.source).startswith('DART'):continue
        proof=proofs[row.event_id]
        receipts=set(re.findall(r'\d{14}',str(row.source_reference)))
        sources=[source for source in proof.get('sources',[]) if any(receipt in source['url'] for receipt in receipts)]
        publications={entry['rcept_no']:entry['rcept_dt'] for bundle in [proof,proof.get('supplementary',{})]
                      for entry in bundle.get('family',[]) if entry['rcept_no'] in receipts}
        observations.at[row.Index,'source_collected_at']=json.dumps({s['url']:s['collected_at'] for s in sources})
        observations.at[row.Index,'source_publication_dates']=json.dumps(publications)
        observations.at[row.Index,'source_urls']=json.dumps([s['url'] for s in sources])
        if not isinstance(row.source_sha256,str) or not row.source_sha256:
            observations.at[row.Index,'source_sha256']='|'.join(sorted(s['sha256'] for s in sources))
    approved,observations=mask_unapproved(features,observations)
    observations=normalise_observation_times(observations)
    validation=validate_snapshot(approved,observations)
    if set(approved.event_id)!=set(inputs['features'].event_id):
        raise RuntimeError('reconciled_feature_population_changed')
    event_dispositions=inputs['events'][['event_id','event_class','market','listing_date']].merge(
        connections[['event_id','status','reason']],on='event_id',how='left',validate='one_to_one')
    event_dispositions['status']=event_dispositions.status.fillna('outside_general_source_reconciliation_scope')
    if (pinned!={k:digest(p) for k,p in paths.items()}
            or inputs['input_sha256']!=load_inputs(raw,processed)['input_sha256']
            or any(digest(raw/'krx_index_days'/'KOSPI'/(day+'.json'))!=sha for day,sha in days.items())):
        raise RuntimeError('reconciled_inputs_changed_during_build')
    identity={'inputs':inputs['input_sha256'],'sources':digest(source_path/'manifest.json'),
              'financial':digest(financial_path/'manifest.json'),'code_and_market':pinned,'market_days':days}
    version=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
    destination=processed/'reconciled_datasets'/version
    if not destination.exists():
        destination.parent.mkdir(parents=True,exist_ok=True)
        staging=Path(tempfile.mkdtemp(prefix='.dataset.',dir=destination.parent))
        for name,frame in [('features_all',approved),('raw_candidates',features),('feature_observations',observations),
                           ('event_dispositions',event_dispositions),('kospi_index_snapshot',market_index),
                           ('kosdaq_index_snapshot',pd.read_parquet(raw/'kosdaq_index.parquet')),
                           ('event_master',inputs['events'])]:
            frame.to_parquet(staging/(name+'.parquet'),index=False)
        atomic_json(staging/'validation.json',validation)
        report={'version':version,'scope':'stored_general_ipo_listing_eve_reconciliation_not_product_stage_approval',
            'inputs':identity,'rows':len(approved),'general_ipo_rows':len(general),
            'source_status_counts':connections.status.value_counts().to_dict(),
            'financial_status_counts':financial['status_counts'],
            'market_context':context,
            'validation':validation,
            'feature_approved_counts':observations.groupby('feature_name').model_eligible.sum().astype(int).to_dict(),
            'training_executed':False,'deployment_authorized':False,'production_files_replaced':False,
            'remaining_gate':'Actual pre_demand/post_demand source snapshots and milestone cutoffs must be verified independently.',
            'output_sha256':{p.name:digest(p) for p in staging.iterdir()}}
        atomic_json(staging/'manifest.json',report);staging.rename(destination)
    atomic_json(processed/'current_reconciled_data.json',
                {'manifest':str((destination/'manifest.json').relative_to(processed)),
                 'sha256':digest(destination/'manifest.json'),'training_authorized':False,'deployment_authorized':False})
    return {**checked_artifact(destination),'path':str(destination)}
