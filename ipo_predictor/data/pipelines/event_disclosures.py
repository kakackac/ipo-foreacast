"""Connect new official events to dated DART originals; never release a model."""
import hashlib
import json
import math
import tempfile
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests

from data.collectors.dart_collector import DARTCollector
from data.collectors.krx_collector import KRXCollector
from data.pipelines.data_delivery import checked_artifact, digest
from data.pipelines.disclosure_connections import company_identity, official_family_document, public_text
from data.pipelines.event_delivery import current_event_delivery
from data.pipelines.financial_history import atomic_json


FIELDS = ('offering_price', 'institutional_demand_ratio', 'lockup_commitment_ratio',
          'price_band_low', 'price_band_high', 'public_float_ratio_disclosed', 'new_shares', 'secondary_shares')


def cached_calendar_candidates(raw_dir):
    """Reuse official month originals that a narrower schedule registry omitted."""
    from scripts.audit_dart_subscription_calendar import parse_calendar
    raw_dir = Path(raw_dir).resolve()
    months = {}
    for report_path in sorted((Path(raw_dir)/'dart_calendar_audit').glob('*/report.json')):
        report = json.loads(report_path.read_text())
        if report.get('status') != 'complete' or report.get('source') != 'https://dart.fss.or.kr/dsac008/main.do':
            continue
        for request in report['requests']:
            period = (request['year'],request['month'])
            if period in months and months[period][0] >= report['collected_at']: continue
            source = report_path.parent/f'{period[0]}_{period[1]:02d}.html'
            proof = {'path':str(source),'sha256':request['sha256'],'collected_at':report['collected_at'],
                     'url':report['source']+'?'+urlencode({'year':period[0],'month':period[1]})}
            months[period] = (report['collected_at'],proof)
    groups = {}
    for period,(_,proof) in months.items():
        path = Path(proof['path'])
        if digest(path) != proof['sha256']: raise ValueError('event_calendar_source_hash_mismatch')
        events = parse_calendar(path.read_bytes(),*period)
        for row in events:
            row['calendar_source'] = proof
            groups.setdefault((row['corp_code'],row['rcept_no']),[]).append(row)
    result = []
    for rows in groups.values():
        starts={r['subscription_date'] for r in rows if r['event_type']=='start'}
        ends={r['subscription_date'] for r in rows if r['event_type']=='end'}
        if len(starts)!=1 or len(ends)!=1 or next(iter(starts)) > next(iter(ends)): continue
        result.append({**rows[0],'subscription_end':next(iter(ends)),
                       'calendar_sources':list({r['calendar_source']['path']:r['calendar_source'] for r in rows}.values())})
    return result


def accepted_values(document, expected_price):
    if (document.get('offering_price_review_status') != 'verified_currency_unit'
            or document.get('offering_price_finality') != 'confirmed_price_language'):
        raise ValueError('latest_official_price_not_explicitly_confirmed')
    if not math.isfinite(float(expected_price)) or float(expected_price) <= 0 or document['offering_price'] != float(expected_price):
        raise ValueError('latest_official_price_krx_mismatch')
    values = dict.fromkeys(FIELDS)
    values['offering_price'] = document['offering_price']
    for field, prefix, rule in [('institutional_demand_ratio','institutional_demand','institutional_demand_rule_id'),
                                ('lockup_commitment_ratio','lockup','lockup_rule_id'),
                                ('new_shares','new_shares','new_shares_parse_method'),
                                ('secondary_shares','secondary_shares','secondary_shares_parse_method')]:
        if document.get(prefix+'_parser_validation_status') == 'structurally_verified' and document.get(rule):
            values[field] = document.get(field)
    if document.get('price_band_validation_status') == 'direct_currency_band':
        values.update({f:document.get(f) for f in ('price_band_low','price_band_high')})
    if document.get('public_float_parse_method') == 'disclosed_public_float_ratio_direct_context':
        values['public_float_ratio_disclosed'] = document.get('public_float_ratio_disclosed')
    for field,value in values.items():
        if value is None: continue
        if not math.isfinite(float(value)) or value < 0:
            raise ValueError('parsed_official_value_invalid')
        if field in {'lockup_commitment_ratio','public_float_ratio_disclosed'} and value > 1:
            raise ValueError('parsed_official_fraction_outside_unit_range')
    return values


def discover_candidates(dart, raw_dir, events, cached_only):
    """One bounded list window, reused across companies; no repeated whole history."""
    start = (pd.to_datetime(events.listing_date).min()-pd.Timedelta(days=90)).strftime('%Y%m%d')
    end = (pd.to_datetime(events.listing_date).max()-pd.Timedelta(days=1)).strftime('%Y%m%d')
    path = Path(raw_dir)/'event_disclosure_lists'/f'{start}_{end}.json'
    if path.exists():
        payload = json.loads(path.read_text())
        if hashlib.sha256(json.dumps(payload['rows'],sort_keys=True).encode()).hexdigest() != payload['rows_sha256']:
            raise ValueError('event_disclosure_list_cache_tampered')
        return pd.DataFrame(payload['rows'])
    if cached_only or not dart.is_configured:
        raise ValueError('candidate_discovery_requires_dart_api')
    records = dart.get_ipo_disclosure_list(start,end)
    rows = json.loads(records.to_json(orient='records',date_format='iso'))
    atomic_json(path, {'start':start,'end':end,'source':'OpenDART/list.json', 'rows':rows,
                      'collected_at':pd.Timestamp.now(tz='Asia/Seoul').isoformat(),
                      'rows_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()})
    return records


def connect_event(dart, raw_dir, event, candidates, cached_only=False):
    corps = {str(c['corp_code']) for c in candidates}
    if len(corps) != 1: raise ValueError('candidate_corp_code_not_unique')
    corp = corps.pop()
    url = 'https://dart.fss.or.kr/dsae001/selectPopup.ax?' + urlencode({'selectKey':corp})
    content, company_hash = public_text(dart,url,Path(raw_dir)/'dart_connection_sources',
        lambda text:company_identity(text,corp,event.ticker,event.market),cached_only)
    company_identity(content,corp,event.ticker,event.market)
    seed = max(candidates,key=lambda c:(str(c.get('rcept_dt','')),c['rcept_no']))
    initial_url = 'https://dart.fss.or.kr/dsaf001/main.do?' + urlencode({'rcpNo':seed['rcept_no']})
    index_cache = Path(raw_dir)/'dart_connection_sources'/(hashlib.sha256(initial_url.encode()).hexdigest()+'.json')
    refresh_index = not cached_only
    if index_cache.exists():
        collected = pd.Timestamp(json.loads(index_cache.read_text())['collected_at'])
        if collected.tzinfo is None: raise ValueError('official_source_collection_timezone_missing')
        refresh_index = refresh_index and collected.tz_convert('Asia/Seoul').date()!=pd.Timestamp.now(tz='Asia/Seoul').date()
    metadata, body, proof = official_family_document(dart,raw_dir,seed['rcept_no'],corp,event.listing_date,
                                                   cached_only,refresh_index=refresh_index)
    if not {c['rcept_no'] for c in candidates}.issubset({r['rcept_no'] for r in proof['family']}):
        raise ValueError('multiple_offering_families_require_review')
    document = dart._parse_offering_html(body,metadata['rcept_no'])
    demand = dart._parse_demand_forecast_html(body,corp)
    document.update({k:v for k,v in demand.items() if k not in {'corp_code','parse_success'}})
    values = accepted_values(document,event.offering_price)
    cutoff = pd.Timestamp(event.listing_date).tz_localize('Asia/Seoul')-pd.Timedelta(hours=3)
    available = pd.Timestamp(metadata['rcept_dt']).tz_localize('Asia/Seoul')+pd.Timedelta(days=1)
    if available >= cutoff: raise ValueError('publication_not_before_listing_eve_cutoff')
    proof.update(company_url=url,company_sha256=company_hash)
    proof['candidate_calendars'] = sorted({s['path']:s for c in candidates for s in c.get('calendar_sources',[])}.values(),
                                           key=lambda s:s['path'])
    sources = []
    for source_url in {url,initial_url,proof['source_url'],*[r['url'] for r in proof['sections']]}:
        cache = Path(raw_dir)/'dart_connection_sources'/(hashlib.sha256(source_url.encode()).hexdigest()+'.json')
        record = json.loads(cache.read_text())
        if record['url'] != source_url or hashlib.sha256(record['text'].encode()).hexdigest() != record['sha256']:
            raise ValueError('official_connection_cache_integrity_failed')
        sources.append({'url':source_url,'sha256':record['sha256'],'collected_at':record['collected_at']})
    proof['sources'] = sorted(sources,key=lambda r:r['url'])
    return values, metadata, proof, document, available, cutoff


def run(raw_dir, processed_dir, dart=None, cached_only=False, discover=False):
    dart = dart or DARTCollector()
    root, manifest = current_event_delivery(processed_dir)
    events = pd.read_parquet(root/'added_events.parquet')
    events = events[events.event_class.isin(['general_ipo','spac_ipo'])].copy()
    links = pd.read_parquet(root/'disclosure_links.parquet').set_index('event_id')
    source_manifest = digest(root/'manifest.json')
    calendar = cached_calendar_candidates(raw_dir)
    calendar_events = {event.event_id:[c for c in calendar
        if KRXCollector._normalise_company_name(c['corp_name']) == KRXCollector._normalise_company_name(event.corp_name)
        and pd.Timestamp(event.listing_date)-pd.Timedelta(days=90) <= pd.Timestamp(c['subscription_end']) < pd.Timestamp(event.listing_date)]
        for event in events.itertuples(index=False)}
    listing = pd.DataFrame()
    discovery_reason = 'official_candidate_not_linked'
    needs_discovery = any(not json.loads(links.loc[e.event_id].candidate_receipts) and not calendar_events[e.event_id]
                          for e in events.itertuples(index=False))
    if discover and needs_discovery:
        try: listing = discover_candidates(dart,raw_dir,events,cached_only)
        except (ValueError,RuntimeError,requests.RequestException) as exc:
            discovery_reason = str(exc) if isinstance(exc,ValueError) else 'source_access_failed'
    results, observations, lineage = [], [], []
    consecutive_failures = 0
    for event in events.itertuples(index=False):
        link = links.loc[event.event_id]
        corps, receipts = json.loads(link.candidate_corp_codes),json.loads(link.candidate_receipts)
        candidates = [{'corp_code':c,'rcept_no':r} for c in corps for r in receipts]
        if not candidates:
            candidates = calendar_events[event.event_id]
        if not candidates and len(listing):
            names = listing.corp_name.map(KRXCollector._normalise_company_name)
            matches = listing[names.eq(KRXCollector._normalise_company_name(event.corp_name))].copy()
            dates = pd.to_datetime(matches.rcept_dt,errors='coerce')
            matches = matches[dates.lt(pd.Timestamp(event.listing_date)) & dates.ge(pd.Timestamp(event.listing_date)-pd.Timedelta(days=90))]
            candidates = matches.to_dict('records')
        row = {'event_id':event.event_id,'ticker':event.ticker,'listing_date':event.listing_date,
               'event_class':event.event_class,'status':discovery_reason,'model_eligible':False,
               **dict.fromkeys(FIELDS)}
        results.append(row)
        document, proof, metadata, available, cutoff = {}, {}, {}, None, None
        try:
            if not candidates: raise ValueError(discovery_reason)
            if consecutive_failures >= 3 and not cached_only:
                raise ValueError('source_cooldown_requires_retry')
            values, metadata, proof, document, available, cutoff = connect_event(dart,raw_dir,event,candidates,cached_only)
            consecutive_failures = 0
            row.update(values,corp_code=proof['corp_code'],rcept_no=metadata['rcept_no'],
                       status='official_identity_family_and_price_verified',available_at=available,prediction_at=cutoff,
                       evidence=json.dumps(proof,ensure_ascii=False))
            lineage.extend({'event_id':event.event_id,'corp_code':proof['corp_code'],**r} for r in proof['family'])
        except requests.RequestException:
            consecutive_failures += 1
            row['status'] = 'source_access_failed'
        except ValueError as exc:
            row['status'] = str(exc)
        for field in FIELDS:
            value = row[field]
            reason = None if value is not None else (row['status'] if not proof else
                document.get({'institutional_demand_ratio':'institutional_demand_rejection_reason',
                              'lockup_commitment_ratio':'lockup_rejection_reason'}.get(field,''))
                or 'not_verified_in_selected_official_sections')
            observations.append({'event_id':event.event_id,'feature':field,'value':value,'is_missing':value is None,
                'raw_value':document.get(field),'unit':('KRW' if field in {'offering_price','price_band_low','price_band_high'}
                    else 'shares' if field in {'new_shares','secondary_shares'} else 'ratio_to_one'
                    if field=='institutional_demand_ratio' else 'fraction'),
                'missing_reason':reason,'validation_status':'verified_official_source' if value is not None else 'needs_review',
                'source_url':proof.get('source_url'),'receipt':metadata.get('rcept_no'),'available_at':available,
                'prediction_at':cutoff,'collected_at':max((s['collected_at'] for s in proof.get('sources',[])),default=None),
                'human_review_required':value is None,'evidence':json.dumps(proof,ensure_ascii=False),
                'parser_evidence':json.dumps(document,ensure_ascii=False,default=str)})
    frame = pd.DataFrame(results,columns=['event_id','ticker','listing_date','event_class','status','model_eligible',
                                        *FIELDS,'corp_code','rcept_no','available_at','prediction_at','evidence'])
    ledger = pd.DataFrame(observations,columns=['event_id','feature','value','raw_value','unit','is_missing',
        'missing_reason','validation_status','source_url','receipt','available_at','prediction_at','collected_at',
        'human_review_required','evidence','parser_evidence'])
    used = ledger.loc[~ledger.is_missing.astype(bool)]
    available, cutoffs = pd.to_datetime(used.available_at,utc=True), pd.to_datetime(used.prediction_at,utc=True)
    violations = (available.isna() | cutoffs.isna() | (available >= cutoffs)).sum()
    if violations or current_event_delivery(processed_dir)[0] != root or digest(root/'manifest.json') != source_manifest:
        raise RuntimeError('event_disclosure_input_changed_or_time_violation')
    identity = {'event_manifest':source_manifest,'builder':digest(__file__),
                'parser':digest(Path(__file__).parents[1]/'collectors/dart_collector.py'),
                'connections':digest(Path(__file__).with_name('disclosure_connections.py')),
                'records':hashlib.sha256(frame.to_json(orient='records',date_format='iso').encode()).hexdigest()}
    version = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
    destination = Path(processed_dir)/'event_disclosures'/version
    if destination.exists(): report = checked_artifact(destination)
    else:
        destination.parent.mkdir(parents=True,exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix='.disclosures.',dir=destination.parent))
        outputs = {'events.parquet':frame,'observations.parquet':ledger,'lineage.parquet':pd.DataFrame(lineage)}
        for name,data in outputs.items(): data.to_parquet(staging/name,index=False)
        originals, calendars = {}, {}
        for row in results:
            for source in json.loads(row.get('evidence','{}')).get('sources',[]):
                path = Path(raw_dir)/'dart_connection_sources'/(hashlib.sha256(source['url'].encode()).hexdigest()+'.json')
                record = json.loads(path.read_text())
                if hashlib.sha256(record['text'].encode()).hexdigest() != source['sha256']:
                    raise RuntimeError('event_disclosure_original_changed')
                originals[source['url']] = record
            for source in json.loads(row.get('evidence','{}')).get('candidate_calendars',[]):
                path = Path(source['path'])
                if digest(path) != source['sha256']: raise RuntimeError('event_calendar_source_changed')
                calendars[source['url']] = {**source,'text':path.read_text()}
        atomic_json(staging/'source_documents.json', originals)
        atomic_json(staging/'calendar_sources.json', calendars)
        report = {'version':version,'event_version':manifest['version'],'source':identity,'events':len(frame),
                  'status_counts':frame.status.value_counts().to_dict(),'verified_values':{f:int(frame[f].notna().sum()) for f in FIELDS},
                  'time_violations':int(violations),'training_executed':False,'deployment_authorized':False,
                  'scope':'new_event_listing_eve_disclosures_not_pre_demand_or_post_demand_release',
                  'output_sha256':{name:digest(staging/name) for name in outputs}}
        report['output_sha256']['source_documents.json'] = digest(staging/'source_documents.json')
        report['output_sha256']['calendar_sources.json'] = digest(staging/'calendar_sources.json')
        atomic_json(staging/'manifest.json',report)
        staging.rename(destination)
    atomic_json(Path(processed_dir)/'current_event_disclosures.json',
                {'manifest':str((destination/'manifest.json').relative_to(processed_dir)),
                 'sha256':digest(destination/'manifest.json')})
    return {**report,'path':str(destination)}


def current_disclosures(processed_dir):
    processed = Path(processed_dir).resolve()
    pointer = json.loads((processed/'current_event_disclosures.json').read_text())
    manifest = (processed/pointer['manifest']).resolve()
    if not manifest.is_relative_to(processed/'event_disclosures') or digest(manifest)!=pointer['sha256']:
        raise RuntimeError('event_disclosure_pointer_invalid')
    return manifest.parent, checked_artifact(manifest.parent)
