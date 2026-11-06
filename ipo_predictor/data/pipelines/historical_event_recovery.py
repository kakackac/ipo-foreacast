"""Recover omitted historical events through official trading-code identities."""
import hashlib
import json
import re
import tempfile
from pathlib import Path
from urllib.parse import urlencode, urlparse, parse_qs

import pandas as pd
import requests
from bs4 import BeautifulSoup

from data.collectors.dart_collector import DARTCollector
from data.collectors.krx_collector import KRXCollector
from data.pipelines.data_delivery import current_financial_repair, digest
from data.pipelines.disclosure_connections import public_text, official_family_document
from data.pipelines.event_disclosures import connect_event
from data.pipelines.financial_history import atomic_json
from data.pipelines.historical_ipo_pipeline import HistoricalIPOPipeline, MAX_FILING_TO_LISTING_DAYS
from data.processors.feature_engineer import FeatureEngineer
from features.source_contracts import DART_STRUCTURAL_AGGREGATE_STATUS


def parse_company_search(text, ticker):
    soup = BeautifulSoup(text, 'html.parser')
    matches = []
    for row in soup.select('tbody tr'):
        cells = row.find_all('td', recursive=False)
        if len(cells) != 2 or cells[1].get_text(strip=True).upper() != ticker:
            continue
        links = cells[0].select('a[href]')
        ids = {m.group(1) for a in links
               if (m := re.fullmatch(r"javascript:select\('([0-9]{8})'\);?", a['href']))}
        if len(ids) != 1:
            raise ValueError('official_company_search_identity_ambiguous')
        matches.append({'corp_code': ids.pop(), 'corp_name': links[0].get_text(' ', strip=True)})
    if len(matches) != 1:
        raise ValueError('official_trading_code_not_unique_or_not_found')
    return matches[0]


def parse_filing_search(text, corp, start, end):
    soup = BeautifulSoup(text, 'html.parser')
    total = re.search(r'총\s*([\d,]+)\s*건', soup.get_text(' ', strip=True))
    rows = []
    table = soup.select_one('tbody#tbody')
    if table is None:
        raise ValueError('official_filing_search_response_invalid')
    for row in table.select('tr'):
        links = row.select('a[id^="r_"]')
        if not links:
            continue
        if len(links) != 1:
            raise ValueError('official_filing_search_receipt_ambiguous')
        receipt = links[0].get('id', '')[2:]
        ids = set(re.findall(r"openCorpInfoNew\('([0-9]{8})'", str(row)))
        href = parse_qs(urlparse(links[0].get('href', '')).query).get('rcpNo')
        dates = re.findall(r'\b([0-9]{4})\.([0-9]{2})\.([0-9]{2})\b', row.get_text(' ', strip=True))
        if ids != {corp} or not re.fullmatch(r'\d{14}', receipt) or href != [receipt] or len(dates) != 1:
            raise ValueError('official_filing_search_identity_conflict')
        day = '-'.join(dates[0])
        if not pd.Timestamp(start) <= pd.Timestamp(day) <= pd.Timestamp(end):
            raise ValueError('official_filing_search_date_outside_request')
        rows.append({'corp_code': corp, 'rcept_no': receipt, 'rcept_dt': day,
                     'report_nm': links[0].get_text(' ', strip=True)})
    if total is None:
        if rows or '조회 결과가 없습니다' not in soup.get_text(' ', strip=True):
            raise ValueError('official_filing_search_count_missing')
        count = 0
    else:
        count = int(total.group(1).replace(',', ''))
    return rows, count


def discover(dart, raw, event, cached_only=False):
    cache = Path(raw)/'dart_connection_sources'
    ticker = str(event.ticker).upper().zfill(6)
    url = 'https://dart.fss.or.kr/dsae001/search.ax?' + urlencode(
        {'textCrpNm': ticker, 'searchType': '1', 'currentPage': '1'})
    text, sha = public_text(dart, url, cache, lambda s: parse_company_search(s, ticker), cached_only)
    company = parse_company_search(text, ticker)
    start = (pd.Timestamp(event.listing_date)-pd.Timedelta(days=MAX_FILING_TO_LISTING_DAYS)).strftime('%Y%m%d')
    end = (pd.Timestamp(event.listing_date)-pd.Timedelta(days=1)).strftime('%Y%m%d')
    sources = [{'url': url, 'sha256': sha}]
    filings = []
    page, count = 1, None
    while count is None or len(filings) < count:
        url = 'https://dart.fss.or.kr/dsab001/search.ax?' + urlencode({
            'textCrpCik': company['corp_code'], 'startDate': start, 'endDate': end,
            'publicType': 'C001', 'maxResults': '100', 'currentPage': str(page),
            'sort': 'date', 'series': 'desc', 'pageGubun': 'corp'})
        text, sha = public_text(dart, url, cache,
            lambda s: parse_filing_search(s, company['corp_code'], start, end), cached_only)
        rows, response_count = parse_filing_search(text, company['corp_code'], start, end)
        if count is not None and count != response_count:
            raise ValueError('official_filing_search_population_changed')
        count = response_count
        if count and not rows:
            raise ValueError('official_filing_search_page_missing')
        filings.extend(rows)
        sources.append({'url': url, 'sha256': sha})
        page += 1
        if page > 20:
            raise ValueError('official_filing_search_page_limit')
    if len(filings) != count or len({r['rcept_no'] for r in filings}) != count:
        raise ValueError('official_filing_search_incomplete')
    candidates = [r for r in filings if re.search(r'증권신고서.*지분증권', r['report_nm'])]
    return candidates, {**company, 'sources': sources, 'filings': filings,
                        'search_start': start, 'search_end': end}


def source_record(event, values, meta, proof, document):
    record = {k: v for k, v in document.items() if pd.api.types.is_scalar(v)}
    record.update(values, event_id=event.event_id, ticker=str(event.ticker).zfill(6),
        corp_name=event.corp_name, corp_code=proof['corp_code'], listing_date=event.listing_date,
        rcept_no=meta['rcept_no'], rcept_dt=pd.Timestamp(meta['rcept_dt']),
        filing_report_nm=meta['report_nm'], feature_available_at=pd.Timestamp(meta['rcept_dt']),
        offering_price_rcept_no=meta['rcept_no'], offering_price_rcept_dt=pd.Timestamp(meta['rcept_dt']),
        lineage_validation_status='official_viewer_corp_and_bidirectional_family_verified')
    documents = [(meta, document)]
    if 'supplementary' in proof:
        supplement = proof['supplementary']
        receipt = parse_qs(urlparse(supplement['source_url']).query)['rcpNo'][0]
        supplemental_meta = next(r for r in supplement['family'] if r['rcept_no'] == receipt)
        documents.append((supplemental_meta, document['supplementary_document']))
    demand, demand_meta = HistoricalIPOPipeline._select_dart_lineage_demand(documents, values['offering_price'], event.listing_date)
    record.update(demand, **demand_meta)
    for prefix, field in [('institutional', 'institutional_demand_ratio'), ('lockup', 'lockup_commitment_ratio')]:
        if values[field] is not None:
            record[prefix+'_validation_status'] = DART_STRUCTURAL_AGGREGATE_STATUS
            record[prefix+'_available_at'] = demand_meta[prefix+'_rcept_dt']
    for group, field in [('price_band', 'price_band_low')]:
        if values[field] is None:
            continue
        origin = proof.get('field_sources', {}).get(field, {})
        receipt = origin.get('receipt', meta['rcept_no'])
        selected_meta, selected_doc = next((m,d) for m,d in documents if m['rcept_no'] == receipt)
        record[group+'_rcept_no'] = receipt
        record[group+'_rcept_dt'] = pd.Timestamp(selected_meta['rcept_dt'])
        prefixes = ('price_band_',)
        record.update({k:v for k,v in selected_doc.items()
                       if k.startswith(prefixes) and pd.api.types.is_scalar(v)})
    # These are resolved bundles, not independently filled raw scalar fields.
    for key in ('resolved_float', 'resolved_structure'):
        bundle = proof.get(key)
        if bundle is not None:
            record.update({field:value for field,value in bundle.items() if field not in {'source_url','available_at'}})
    for field in ('new_shares', 'secondary_shares'):
        if values[field] is None:
            record[field] = None
    for field in ('public_float_rcept_dt','offering_structure_rcept_dt'):
        if record.get(field) is not None:
            record[field] = pd.Timestamp(record[field])
    return record


def select_body_seed(candidates):
    body = [r for r in candidates if '첨부정정' not in r['report_nm']]
    if not body:
        raise ValueError('body_registration_candidate_not_found')
    return max(body, key=lambda r:(r['rcept_dt'],r['rcept_no']))


def capture_sources(raw, references, destination):
    for source in references:
        cache = Path(raw)/'dart_connection_sources'/(hashlib.sha256(source['url'].encode()).hexdigest()+'.json')
        payload = json.loads(cache.read_text())
        if (payload['url'] != source['url'] or payload['sha256'] != source['sha256']
                or hashlib.sha256(payload['text'].encode()).hexdigest() != source['sha256']):
            raise ValueError('recovery_source_changed')
        destination[source['url']] = payload


def run(raw_dir, processed_dir, dart=None, cached_only=False, event_limit=None):
    dart = dart or DARTCollector()
    raw, processed = Path(raw_dir), Path(processed_dir)
    base = current_financial_repair(processed)
    if event_limit is not None and event_limit < 1:
        raise ValueError('recovery_limit_must_be_positive')
    paths = {'base':base/'features_all.parquet', 'observations':base/'feature_observations.parquet',
        'events':raw/'krx_official_event_master.parquet', 'kospi':raw/'kospi_index.parquet',
        'kosdaq':raw/'kosdaq_index.parquet', 'builder':Path(__file__),
        'parser':Path(__file__).parents[1]/'collectors/dart_collector.py',
        'connections':Path(__file__).with_name('disclosure_connections.py'),
        'connector':Path(__file__).with_name('event_disclosures.py'),
        'selector':Path(__file__).with_name('historical_ipo_pipeline.py'),
        'float_resolver':Path(__file__).with_name('float_source_resolver.py'),
        'features':Path(__file__).parents[1]/'processors/feature_engineer.py',
        'definitions':Path(__file__).parents[2]/'features/definitions.py',
        'config':Path(__file__).parents[2]/'config.py'}
    pinned = {key:digest(path) for key,path in paths.items()}
    source_hash = digest(base/'features_all.parquet')
    original = pd.read_parquet(base/'features_all.parquet')
    events = KRXCollector.reclassify_official_listing_events(pd.read_parquet(raw/'krx_official_event_master.parquet'))
    event_hash = digest(raw/'krx_official_event_master.parquet')
    targets = events[events.event_class.eq('general_ipo') & ~events.event_id.isin(original.event_id)].sort_values('listing_date')
    total = len(targets)
    if event_limit is not None:
        targets = targets.head(event_limit)
    results, records, lineage, sources, classifications = [], [], [], {}, []
    for event in targets.itertuples(index=False):
        row = {'event_id': event.event_id, 'corp_name': event.corp_name,
               'listing_date': event.listing_date, 'status': 'pending', 'absence_confirmed': False}
        results.append(row)
        try:
            candidates, identity = discover(dart, raw, event, cached_only)
            row.update(corp_code=identity['corp_code'], current_corp_name=identity['corp_name'],
                       company_name_changed=HistoricalIPOPipeline._clean_name(identity['corp_name']) != HistoricalIPOPipeline._clean_name(event.corp_name),
                       discovery_evidence=json.dumps(identity, ensure_ascii=False))
            capture_sources(raw,identity['sources'],sources)
            if not candidates:
                raise ValueError('no_registration_in_bounded_official_search_requires_event_review')
            # Earlier withdrawn IPO attempts and duplicate same-day registrations
            # are not necessarily members of the final IPO's correction family.
            # Select the latest dated registration, never fall back after failure;
            # the selected family must still verify the KRX final price and cutoff.
            seed = select_body_seed(candidates)
            metadata, body, family = official_family_document(dart,raw,seed['rcept_no'],identity['corp_code'],event.listing_date,cached_only)
            row['family_evidence'] = json.dumps(family,ensure_ascii=False)
            references = [*family['sections'],
                {'url':'https://dart.fss.or.kr/dsaf001/main.do?'+urlencode({'rcpNo':seed['rcept_no']}), 'sha256':family['initial_index_sha256']},
                {'url':family['source_url'],'sha256':family['final_index_sha256']}]
            capture_sources(raw,references,sources)
            compact = re.sub(r'\s+', '',BeautifulSoup(body,'html.parser').get_text())
            issuer = re.search(r'회사명[:：](.{1,80}?)대표이사[:：]',compact)
            if issuer and '기업인수목적' in issuer.group(1):
                classifications.append({'event_id':event.event_id,'before':'general_ipo','after':'spac_ipo',
                    'issuer_name':issuer.group(1),'receipt':metadata['rcept_no'],'source_url':family['source_url']})
                row.update(status='classified_spac_excluded_from_general_recovery',receipt=metadata['rcept_no'])
                continue
            # Current listing/delisting status is not the market at the historical
            # IPO date. The official KRX event owns that date/market; DART still
            # must prove the exact current trading code and corporation identity.
            values, meta, proof, document, available, cutoff = connect_event(
                dart, raw, event, [seed], cached_only, verify_current_market=False)
            proof['discovery_candidates'] = candidates
            all_sources = identity['sources'] + proof['sources']
            capture_sources(raw,all_sources,sources)
            record = source_record(event,values,meta,proof,document)
            records.append(record)
            lineage.extend({'event_id':event.event_id, 'corp_code':proof['corp_code'], **r} for r in proof['family'])
            row.update(status='recovered_official_event', receipt=meta['rcept_no'],
                       evidence=json.dumps(proof, ensure_ascii=False,default=str))
        except requests.RequestException as exc:
            row.update(status='source_access_failed', reason=type(exc).__name__)
        except ValueError as exc:
            row.update(status='review_required', reason=str(exc))
        print(event.event_id, row['status'], row.get('reason',''), flush=True)
        atomic_json(raw/'historical_event_recovery_progress.json', {'input':pinned,
            'target_events':total,'processed_events':len(results), 'recovered_events':len(records),
            'dispositions':json.loads(pd.DataFrame(results).to_json(orient='records',date_format='iso'))})
    recovered = pd.DataFrame(records)
    engineer = FeatureEngineer('phase2')
    extra = engineer.build_features(recovered, events, pd.read_parquet(raw/'kospi_index.parquet'),
        pd.read_parquet(raw/'kosdaq_index.parquet')) if records else original.iloc[:0].copy()
    combined = pd.concat([original, extra], ignore_index=True)
    if combined.event_id.duplicated().any() or set(extra.event_id)-set(targets.event_id):
        raise RuntimeError('historical_recovery_population_conflict')
    observations = pd.concat([pd.read_parquet(base/'feature_observations.parquet'),
                              engineer.build_feature_observations(extra) if len(extra) else pd.DataFrame()], ignore_index=True)
    if {key:digest(path) for key,path in paths.items()} != pinned:
        raise RuntimeError('historical_recovery_inputs_changed')
    identity = {**pinned,
        'results':hashlib.sha256(pd.DataFrame(results).to_json(date_format='iso').encode()).hexdigest()}
    version = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
    destination = processed/'historical_event_recoveries'/version
    if not destination.exists():
        destination.parent.mkdir(parents=True,exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix='.recovery.', dir=destination.parent))
        outputs = {'features_all.parquet':combined, 'recovered_features.parquet':extra,
            'feature_observations.parquet':observations, 'dart_records.parquet':recovered,
            'dispositions.parquet':pd.DataFrame(results), 'disclosure_lineage.parquet':pd.DataFrame(lineage),
            'event_classification_overrides.parquet':pd.DataFrame(classifications)}
        for name, frame in outputs.items():
            frame.to_parquet(staging/name,index=False)
        atomic_json(staging/'source_documents.json', sources)
        report = {'version':version, 'input':identity, 'base_path':str(base), 'target_events':total,
            'processed_events':len(targets), 'recovered_events':len(extra), 'features_before':len(original),
            'features_after':len(combined), 'status_counts':pd.Series([r['status'] for r in results],dtype=object).value_counts().to_dict(),
            'classified_spac_events':len(classifications),
            'remaining_without_feature_row':total-len(extra), 'training_executed':False,
            'production_files_replaced':False, 'deployment_authorized':False,
            'scope':'historical_missing_event_recovery_not_full_cohort_recalculation_or_stage_release',
            'integration_remaining':['recalculate_population_dependent_features', 'financial_collection_for_recovered_events',
                                     'true_stage_cutoff_validation'],
            'output_sha256':{name:digest(staging/name) for name in [*outputs, 'source_documents.json']}}
        atomic_json(staging/'manifest.json', report)
        staging.rename(destination)
    else:
        from data.pipelines.data_delivery import checked_artifact
        report = checked_artifact(destination)
    pointer = processed/'current_historical_event_recovery.json'
    promote = event_limit is None
    if pointer.exists():
        from data.pipelines.data_delivery import checked_artifact
        prior = json.loads(pointer.read_text())
        prior_manifest = (processed/prior['manifest']).resolve()
        if not prior_manifest.is_relative_to((processed/'historical_event_recoveries').resolve()) or digest(prior_manifest) != prior['sha256']:
            raise ValueError('historical_recovery_pointer_invalid')
        previous = checked_artifact(prior_manifest.parent)
        if previous['input']['base'] == pinned['base']:
            old_ids = set(pd.read_parquet(prior_manifest.parent/'recovered_features.parquet').event_id)
            explained_exclusions = {row['event_id'] for row in classifications}
            promote = promote and old_ids.issubset(set(extra.event_id) | explained_exclusions)
    if promote:
        atomic_json(pointer, {'manifest':str((destination/'manifest.json').relative_to(processed)),
                              'sha256':digest(destination/'manifest.json')})
    report['current_pointer_advanced'] = promote
    return {**report, 'path':str(destination)}
