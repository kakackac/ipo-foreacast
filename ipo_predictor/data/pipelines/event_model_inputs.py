"""Resume only missing KOSPI context and connect verified new research inputs."""
import hashlib
import json
import math
import tempfile
from pathlib import Path

import pandas as pd

from data.collectors.krx_collector import KRXCollector
from data.pipelines.data_delivery import checked_artifact, current_financial_repair, digest
from data.pipelines.event_delivery import current_event_delivery
from data.pipelines.event_disclosures import current_disclosures
from data.pipelines.financial_history import atomic_json
from data.pipelines.research_dataset import build_frames, FEATURES
from features.source_contracts import DART_STRUCTURAL_AGGREGATE_STATUS


def checksum(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def read_day(path, day):
    record=json.loads(Path(path).read_text())
    if record['sha256']!=checksum(record['data']) or record['data']['date']!=day:
        raise ValueError('market_day_cache_integrity_failed')
    data=record['data']
    if data['endpoint']!='idx/kospi_dd_trd': raise ValueError('market_day_endpoint_conflict')
    row=data['selected_row']
    if row is not None:
        if row['BAS_DD']!=day or KRXCollector._normalise_index_name(row['IDX_NM']) not in {'KOSPI','코스피'}:
            raise ValueError('market_day_response_identity_conflict')
        close=KRXCollector._to_number(row['CLSPRC_IDX'])
        if close is None or not math.isfinite(close) or close<=0: raise ValueError('market_day_close_invalid')
    elif data['response_rows']!=0: raise ValueError('market_day_missing_index_not_empty_response')
    return data


def collect_days(krx, raw_dir, seed, events, cached_only=False, *, start_date=None, end_date=None):
    if events.empty:
        return seed.copy(),{},{'status':'complete','error_type':None,'new_daily_requests':0,
            'checked_days':0,'required_weekdays':0,'start':None,'end':None}
    if seed.empty:raise ValueError('verified_historical_index_seed_required')
    start=pd.to_datetime(seed.date).max().normalize()+pd.Timedelta(days=1)
    end=pd.to_datetime(events.listing_date).max().normalize()-pd.Timedelta(days=2)
    if start_date is not None:start=pd.Timestamp(start_date).normalize()
    if end_date is not None:end=min(end,pd.Timestamp(end_date).normalize())
    if (start_date is not None or end_date is not None) and start>end:
        raise ValueError('market_context_invalid_requested_range')
    root=Path(raw_dir)/'krx_index_days'/'KOSPI';root.mkdir(parents=True,exist_ok=True)
    frames, checked, requested = [],{},0
    status='complete';error_type=None
    for timestamp in pd.bdate_range(start,end):
        day=timestamp.strftime('%Y%m%d');path=root/(day+'.json')
        if not path.exists():
            if cached_only or not krx.is_configured:
                status='market_context_requires_krx_api';break
            try:
                records=krx._get_daily_records('idx/kospi_dd_trd',day)
                requested+=1
                matches=[r for r in records if krx._normalise_index_name(r.get('IDX_NM')) in {'KOSPI','코스피'}]
                if records and len(matches)!=1: raise ValueError('market_index_not_unique_in_response')
                selected={k:matches[0].get(k) for k in ('BAS_DD','IDX_NM','CLSPRC_IDX')} if matches else None
                data={'endpoint':'idx/kospi_dd_trd','date':day,'selected_row':selected,'response_rows':len(records),
                      'collected_at':pd.Timestamp.now(tz='Asia/Seoul').isoformat()}
                payload={'data':data,'sha256':checksum(data)}
                # Validate before publishing a reusable checkpoint.
                if selected is not None:
                    close=krx._to_number(selected['CLSPRC_IDX'])
                    if (selected['BAS_DD']!=day or close is None or not math.isfinite(close) or close<=0):
                        raise ValueError('market_index_date_or_close_invalid')
                atomic_json(path,payload)
            except (RuntimeError,ValueError) as exc:
                status='partial_failure_retryable';error_type=type(exc).__name__;break
        record=read_day(path,day);checked[day]=digest(path)
        if record['selected_row']:
            frames.append({'date':timestamp,'index_code':'KOSPI','close':krx._to_number(record['selected_row']['CLSPRC_IDX'])})
    fresh=pd.DataFrame(frames,columns=['date','index_code','close'])
    requested_rows_present=bool(frames)
    prefix_paths={path.stem:path for path in root.glob('????????.json')
                  if path.stem.isdigit() and path.stem<pd.Timestamp(seed.date.min()).strftime('%Y%m%d')}
    prefix_missing=[];prefix=[]
    if prefix_paths:
        required_prefix=[d.strftime('%Y%m%d') for d in pd.bdate_range(
            min(prefix_paths),pd.Timestamp(seed.date.min()).normalize()-pd.Timedelta(days=1))]
        prefix_missing=[day for day in required_prefix if day not in prefix_paths]
        if not prefix_missing:
            for day in required_prefix:
                record=read_day(prefix_paths[day],day);checked[day]=digest(prefix_paths[day])
                if record['selected_row']:
                    prefix.append({'date':pd.Timestamp(day),'index_code':'KOSPI',
                                   'close':krx._to_number(record['selected_row']['CLSPRC_IDX'])})
    if prefix:fresh=pd.concat([fresh,pd.DataFrame(prefix)],ignore_index=True)
    if not fresh.empty:
        overlap=seed[['date','close']].merge(fresh[['date','close']],on='date',suffixes=('_seed','_fresh'))
        if any(not math.isclose(float(r.close_seed),float(r.close_fresh),rel_tol=1e-12,abs_tol=1e-12)
               for r in overlap.itertuples(index=False)):
            raise ValueError('market_index_seed_conflict')
    combined=(pd.concat([seed,fresh],ignore_index=True) if len(fresh) else seed.copy()).drop_duplicates('date').sort_values('date')
    if status=='complete' and len(pd.bdate_range(start,end)) and not requested_rows_present:
        status='market_index_all_empty_requires_review'
    return combined,checked,{'status':status,'error_type':error_type,'new_daily_requests':requested,
        'checked_days':len(checked),'required_weekdays':len(pd.bdate_range(start,end)),
        'prefix_status':('incomplete_checkpoints_not_merged' if prefix_missing else
                         'verified_cached_prefix_merged' if prefix_paths else 'no_cached_prefix'),
        'prefix_missing_days':prefix_missing,
        'start':start.strftime('%Y-%m-%d'),'end':end.strftime('%Y-%m-%d')}


def new_frames(disclosures, observations, lineage, price_events):
    prices=price_events.set_index('event_id')
    rows=[];ledger=[]
    for event in disclosures.itertuples(index=False):
        if event.status!='official_identity_family_and_price_verified': continue
        price=prices.loc[event.event_id]
        if price.price_resolution_status!='official_price_verified': continue
        if price.ticker!=event.ticker or pd.Timestamp(price.listing_date)!=pd.Timestamp(event.listing_date):
            raise ValueError('new_input_price_identity_conflict')
        row={field:float('nan') for field in FEATURES}
        row.update(event_id=event.event_id,corp_code=event.corp_code,rcept_no=event.rcept_no,
            offering_price_rcept_no=event.rcept_no,market=price.market,listing_date=event.listing_date,
            event_class=event.event_class,offering_price=event.offering_price,
            offering_price_review_status='verified_currency_unit',price_target_validation_status='official_price_verified',
            open_return_pct=(price.open_price/event.offering_price-1)*100,
            close_return_pct=(price.close_price/event.offering_price-1)*100)
        for field in ('institutional_demand_ratio','lockup_commitment_ratio'):
            matches=observations[observations.event_id.eq(event.event_id)&observations.feature.eq(field)]
            if len(matches)!=1: raise ValueError('new_input_observation_not_unique')
            obs=matches.iloc[0]
            if obs.is_missing or obs.human_review_required or obs.validation_status!='verified_official_source': continue
            published=lineage[lineage.event_id.eq(event.event_id)&lineage.rcept_no.eq(obs.receipt)].rcept_dt.dropna().unique()
            if len(published)!=1: raise ValueError('new_input_publication_not_unique')
            expected=pd.Timestamp(published[0]).tz_localize('Asia/Seoul')+pd.Timedelta(days=1)
            if expected!=pd.Timestamp(obs.available_at): raise ValueError('new_input_observation_publication_conflict')
            row[field]=float(obs.value)
            ledger.append({'event_id':event.event_id,'feature_name':field,'raw_value':float(obs.value),
                'available_at':pd.Timestamp(published[0]),'source_reference':obs.receipt,
                'validation_status':DART_STRUCTURAL_AGGREGATE_STATUS,'is_missing':False,'human_review_required':False})
        rows.append(row)
    columns=[*FEATURES,'event_id','corp_code','rcept_no','offering_price_rcept_no','market','listing_date',
        'event_class','offering_price','offering_price_review_status','price_target_validation_status','open_return_pct','close_return_pct']
    return pd.DataFrame(rows,columns=columns),pd.DataFrame(ledger,columns=['event_id','feature_name','raw_value',
        'available_at','source_reference','validation_status','is_missing','human_review_required'])


def run(raw_dir, processed_dir, krx=None, cached_only=False):
    raw,processed=Path(raw_dir).resolve(),Path(processed_dir).resolve()
    event_root,event_manifest=current_event_delivery(processed)
    disclosure_root,disclosure_manifest=current_disclosures(processed)
    if disclosure_manifest['event_version']!=event_manifest['version']: raise ValueError('new_input_event_version_conflict')
    disclosures=pd.read_parquet(disclosure_root/'events.parquet')
    observations=pd.read_parquet(disclosure_root/'observations.parquet')
    lineage=pd.read_parquet(disclosure_root/'lineage.parquet')
    seed=pd.read_parquet(raw/'kospi_index.parquet')
    seed['date']=pd.to_datetime(seed.date)
    index,days,collection=collect_days(krx or KRXCollector(),raw,seed,disclosures,cached_only)
    features,obs=new_frames(disclosures,observations,lineage,pd.read_parquet(event_root/'events.parquet'))
    repair=current_financial_repair(processed)
    historical=pd.read_parquet(repair/'features_all.parquet')
    if set(features.event_id)&set(historical.event_id): raise ValueError('new_input_overlaps_historical_features')
    old_observations=pd.read_parquet(repair/'feature_observations.parquet')
    old_lineage=pd.read_parquet(repair/'disclosure_lineage.parquet')
    dataset,excluded,evidence=build_frames(pd.concat([historical,features],ignore_index=True),
        pd.concat([old_observations,obs],ignore_index=True),index,pd.concat([old_lineage,lineage],ignore_index=True))
    dataset=dataset[dataset.event_id.isin(features.event_id)].copy()
    pending=[];ready=[]
    for event in features.itertuples(index=False):
        if event.event_class!='general_ipo':
            pending.append({'event_id':event.event_id,'reason':'separate_spac_policy_not_general_model'});continue
        end=pd.Timestamp(event.listing_date).normalize()-pd.Timedelta(days=2)
        missing=[day.strftime('%Y%m%d') for day in pd.bdate_range(pd.Timestamp(collection['start']),end)
                 if day.strftime('%Y%m%d') not in days]
        if missing:
            pending.append({'event_id':event.event_id,'reason':'market_context_not_collected','missing_days':len(missing)});continue
        if collection['status']=='market_index_all_empty_requires_review':
            pending.append({'event_id':event.event_id,'reason':'market_index_all_empty_requires_review'});continue
        selected=dataset[dataset.event_id.eq(event.event_id)]
        if selected.empty or selected[['kospi_momentum_5d','kospi_momentum_20d']].isna().any().any():
            pending.append({'event_id':event.event_id,'reason':'source_or_market_input_contract_failed'});continue
        ready.append(event.event_id)
    dataset=dataset[dataset.event_id.isin(ready)]
    evidence=evidence[evidence.event_id.isin(features.event_id)]
    evidence.loc[~evidence.event_id.isin(ready),'used']=False
    used=evidence[evidence.used].merge(dataset[['event_id','prediction_at']],on='event_id',validate='many_to_one')
    available,cutoffs=pd.to_datetime(used.available_at,utc=True),pd.to_datetime(used.prediction_at,utc=True)
    violations=int((available.isna()|cutoffs.isna()|(available>=cutoffs)).sum())
    if violations:raise ValueError('new_model_input_time_violation')
    source={'event':digest(event_root/'manifest.json'),'disclosure':digest(disclosure_root/'manifest.json'),
        'historical':digest(repair/'manifest.json'),'index_seed':digest(raw/'kospi_index.parquet'),'days':days,
        'builder':digest(__file__),'research_builder':digest(Path(__file__).with_name('research_dataset.py'))}
    version=checksum(source)[:20];destination=processed/'event_model_inputs'/version
    reused=destination.exists()
    if destination.exists(): report=checked_artifact(destination)
    else:
        destination.parent.mkdir(parents=True,exist_ok=True)
        staging=Path(tempfile.mkdtemp(prefix='.event-inputs.',dir=destination.parent))
        outputs={'dataset.parquet':dataset,'pending.parquet':pd.DataFrame(pending),'evidence.parquet':evidence,'kospi_index.parquet':index}
        for name,data in outputs.items():data.to_parquet(staging/name,index=False)
        report={'version':version,'source':source,'collection':collection,'new_events':len(features),
            'ready_general_ipo_rows':len(dataset),'pending_reasons':pd.DataFrame(pending).get('reason',pd.Series(dtype=str)).value_counts().to_dict(),
            'scope':'listing_eve_reconstruction_2026_not_fresh_test_or_product_stage',
            'time_violations':violations,
            'training_executed':False,'deployment_authorized':False,'production_files_replaced':False,
            'output_sha256':{name:digest(staging/name) for name in outputs}}
        atomic_json(staging/'manifest.json',report);staging.rename(destination)
    atomic_json(processed/'current_event_model_inputs.json',{'manifest':str((destination/'manifest.json').relative_to(processed)),
        'sha256':digest(destination/'manifest.json')})
    return {**report,'current_collection':collection,'reused':reused,'path':str(destination)}
