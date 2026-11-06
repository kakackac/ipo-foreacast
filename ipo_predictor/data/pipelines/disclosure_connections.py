"""Repair missing event fields from the same official DART correction family."""
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, parse_qs

import pandas as pd
import requests
from bs4 import BeautifulSoup

from data.pipelines.financial_history import atomic_json
from features.model_profiles import VERIFIED_OFFERING_PRICE_STATUSES
from features.source_contracts import DART_STRUCTURAL_AGGREGATE_STATUS, APPROVED_INSTITUTIONAL_DEMAND_STATUSES


def seed_reviewed_sources(raw_dir, records):
    """Reuse already reviewed official originals, not stale parsed numeric caches."""
    reference = Path(__file__).resolve().parents[2] / 'scripts/official_sample_expectations.json'
    if not reference.exists(): return
    raw = Path(raw_dir)
    for sample in json.loads(reference.read_text())['samples']:
        receipt = sample['receipt']
        corp = records.loc[records.rcept_no.astype(str).eq(receipt), 'corp_code'].dropna().astype(str).unique()
        if len(corp) != 1: continue
        index = raw / 'official_sample_audit' / (sample['sample'] + '_index.html')
        sources = list((raw / 'official_sample_audit').glob(sample['sample'] + '*.html'))
        bodies = [p for p in sources if hashlib.sha256(p.read_bytes()).hexdigest() == sample['sha256']]
        if not index.exists() or len(bodies) != 1: continue
        content = index.read_text()
        try:
            viewer_identity(content, receipt, corp[0])
        except ValueError:
            continue
        for url, text in [('https://dart.fss.or.kr/dsaf001/main.do?' + urlencode({'rcpNo': receipt}), content),
                          (sample['source_url'], bodies[0].read_text())]:
            path = raw / 'dart_connection_sources' / (hashlib.sha256(url.encode()).hexdigest() + '.json')
            if path.exists(): continue
            atomic_json(path, {'url': url, 'text': text, 'sha256': hashlib.sha256(text.encode()).hexdigest(),
                               'collected_at': datetime.now(timezone.utc).isoformat(),
                               'imported_from_reviewed_official_sample': sample['sample']})


def viewer_identity(content, receipt, corp_code, published=None):
    soup = BeautifulSoup(content, 'html.parser')
    ids = set(re.findall(r"openCorpInfoNew\(['\"](\d{8})['\"]", content))
    if ids != {str(corp_code)}:
        raise ValueError('official_document_corp_code_mismatch')
    title = soup.title.get_text(' ', strip=True) if soup.title else ''
    match = re.search(r'/(\d{4})[.](\d{2})[.](\d{2})\s*$', title)
    day = '-'.join(match.groups()) if match else None
    if day is None or (published is not None and day != str(published)[:10]):
        raise ValueError('official_publication_date_conflict')
    family = []
    selected = []
    for option in soup.select('select#family option'):
        number = parse_qs(option.get('value', '')).get('rcpNo', [''])[0]
        if not re.fullmatch(r'\d{14}', number):
            continue
        label = option.get_text(' ', strip=True)
        date_match = re.match(r'(\d{4})[.](\d{2})[.](\d{2})', label)
        if not date_match:
            raise ValueError('official_family_date_missing')
        family.append({'rcept_no': number, 'rcept_dt': '-'.join(date_match.groups()),
                       'report_nm': re.sub(r'\s+', ' ', label)})
        if option.has_attr('selected'):
            selected.append(number)
    if selected != [receipt] or not any(r['rcept_no'] == receipt and r['rcept_dt'] == day for r in family):
        raise ValueError('official_selected_receipt_conflict')
    return {'corp_code': str(corp_code), 'publication_date': day, 'title': title, 'family': family}


def company_identity(content, corp_code, ticker, market):
    """A name discovers a candidate; the official company popup binds its stock code."""
    soup = BeautifulSoup(content, 'html.parser')
    ids = set(re.findall(r"searchPopReport\(['\"](\d{8})['\"]", content))
    if ids != {str(corp_code)}:
        raise ValueError('official_company_corp_code_mismatch')
    fields = {}
    for row in soup.select('table tr'):
        label, value = row.find('th'), row.find('td')
        if label is None or value is None: continue
        key = re.sub(r'\s+', '', label.get_text())
        if key in fields: raise ValueError('official_company_field_ambiguous')
        fields[key] = value.get_text(' ', strip=True)
    code = fields.get('종목코드', '').strip().upper()
    if not re.fullmatch(r'[A-Z0-9]{6}', code) or code != str(ticker).upper().zfill(6):
        raise ValueError('official_company_trading_code_mismatch')
    if fields.get('법인구분') != {'KOSPI':'유가증권시장', 'KOSDAQ':'코스닥시장'}.get(market):
        raise ValueError('official_company_market_mismatch')
    return {'corp_code':str(corp_code), 'ticker':code, 'market':market,
            'company_name':fields.get('공시회사명')}


def public_text(dart, url, cache_dir, validator=None, cached_only=False, refresh=False):
    """Completed downloads survive parser changes; failures are never negative-cached."""
    path = Path(cache_dir) / (hashlib.sha256(url.encode()).hexdigest() + '.json')
    if path.exists() and not (refresh and not cached_only):
        payload = json.loads(path.read_text())
        if payload['url'] != url or hashlib.sha256(payload['text'].encode()).hexdigest() != payload['sha256']:
            raise ValueError('official_connection_cache_integrity_failed')
        if validator is not None:
            validator(payload['text'])
        return payload['text'], payload['sha256']
    if cached_only:
        raise ValueError('official_source_not_cached')
    for attempt in range(3):
        time.sleep(.5 if attempt == 0 else 2 ** attempt)
        try:
            response = dart.session.get(url, timeout=30)
            response.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 2: raise
    response.encoding = 'utf-8'
    content = response.text
    if validator is not None:
        validator(content)
    digest = hashlib.sha256(content.encode()).hexdigest()
    atomic_json(path, {'url': url, 'text': content, 'sha256': digest,
                       'collected_at': datetime.now(timezone.utc).isoformat()})
    return content, digest


def official_family_document(dart, raw_dir, receipt, corp_code, listing_date, cached_only=False, refresh_index=False):
    cache = Path(raw_dir) / 'dart_connection_sources'
    index_url = 'https://dart.fss.or.kr/dsaf001/main.do?' + urlencode({'rcpNo': receipt})
    content, initial_hash = public_text(dart, index_url, cache,
                                       lambda text: viewer_identity(text, receipt, corp_code), cached_only, refresh_index)
    initial = viewer_identity(content, receipt, corp_code)
    cutoff = pd.Timestamp(listing_date).normalize() - pd.Timedelta(hours=3)
    eligible = [r for r in initial['family']
                if pd.Timestamp(r['rcept_dt']) + pd.Timedelta(days=1) < cutoff]
    if not eligible:
        raise ValueError('official_pre_cutoff_family_missing')
    terms = [r for r in eligible if re.search(r'증권신고서.*지분증권|투자설명서', r['report_nm'])]
    if not terms:
        raise ValueError('official_latest_family_type_unsupported')
    latest = max(terms, key=lambda r: (r['rcept_dt'], r['rcept_no']))
    if any(re.search(r'철회|취소', r['report_nm']) and r['rcept_dt'] >= latest['rcept_dt'] for r in eligible):
        raise ValueError('official_family_withdrawn_or_cancelled')
    final_url = 'https://dart.fss.or.kr/dsaf001/main.do?' + urlencode({'rcpNo': latest['rcept_no']})
    if refresh_index and final_url == index_url:
        final_hash = initial_hash
    else:
        content, final_hash = public_text(dart, final_url, cache,
                                         lambda text: viewer_identity(text, latest['rcept_no'], corp_code, latest['rcept_dt']),
                                         cached_only, refresh_index)
    final = viewer_identity(content, latest['rcept_no'], corp_code, latest['rcept_dt'])
    if receipt not in {r['rcept_no'] for r in final['family']}:
        raise ValueError('official_family_reverse_link_missing')
    nodes = []
    for block in re.split(r'var\s+node\d+\s*=\s*\{\s*\}\s*;', content)[1:]:
        fields = dict(re.findall(r"node\d+\['(text|rcpNo|dcmNo|eleId|offset|length|dtd)'\]\s*=\s*\"([^\"]*)\"", block))
        if not re.search(r'증권발행조건확정|요약정보|모집.*매출.*일반사항|투자위험요소', fields.get('text', '')):
            continue
        if (fields.get('rcpNo') != latest['rcept_no'] or fields.get('dtd') not in ('dart3.xsd', 'dart4.xsd')
                or any(not re.fullmatch(r'\d+', fields.get(k, '')) for k in ('dcmNo', 'eleId', 'offset', 'length'))):
            raise ValueError('official_section_identity_conflict')
        nodes.append({k: v for k, v in fields.items() if k != 'text'})
    if not nodes or len(nodes) > 6:
        raise ValueError('official_terms_section_not_identified')
    parts, proofs = [], []
    for node in nodes:
        url = 'https://dart.fss.or.kr/report/viewer.do?' + urlencode(node)
        def validate_body(text):
            if '<table' not in text.lower():
                raise ValueError('official_terms_response_invalid')
        body, digest = public_text(dart, url, cache, validate_body, cached_only)
        if '<table' not in body.lower():
            raise ValueError('official_terms_response_invalid')
        parts.append(body)
        proofs.append({'url': url, 'sha256': digest})
    return latest, '\n'.join(parts), {'initial_receipt': receipt, 'initial_index_sha256': initial_hash,
             'final_index_sha256': final_hash, 'source_url': final_url,
             'corp_code': str(corp_code), 'family': final['family'], 'sections': proofs}


def official_related_prospectus(dart, raw_dir, terms_proof, listing_date, cached_only=False):
    """Follow a bidirectional official related-document link, not a name search."""
    cache = Path(raw_dir)/'dart_connection_sources'
    corp = terms_proof['corp_code']
    terms_receipt = parse_qs(terms_proof['source_url'].split('?',1)[1])['rcpNo'][0]
    terms, terms_hash = public_text(dart,terms_proof['source_url'],cache,
        lambda text:viewer_identity(text,terms_receipt,corp),cached_only)
    if terms_hash != terms_proof['final_index_sha256']:
        raise ValueError('terms_index_changed_during_supplement_selection')
    cutoff = pd.Timestamp(listing_date).normalize()-pd.Timedelta(hours=3)

    def related(content):
        rows = []
        for option in BeautifulSoup(content,'html.parser').select('select#ref option'):
            receipt = parse_qs(option.get('value','')).get('rcpNo',[''])[0]
            label = option.get_text(' ',strip=True)
            day = re.match(r'(\d{4})[.](\d{2})[.](\d{2})',label)
            if re.fullmatch(r'\d{14}',receipt) and day:
                rows.append({'rcept_no':receipt,'rcept_dt':'-'.join(day.groups()),
                             'report_nm':label,'title':option.get('title','')})
        return rows

    candidates = [r for r in related(terms) if r['title']=='투자설명서'
                  and pd.Timestamp(r['rcept_dt'])+pd.Timedelta(days=1)<cutoff]
    if not candidates: raise ValueError('no_pre_cutoff_related_prospectus')
    candidate = max(candidates,key=lambda r:(r['rcept_dt'],r['rcept_no']))
    url = 'https://dart.fss.or.kr/dsaf001/main.do?'+urlencode({'rcpNo':candidate['rcept_no']})
    index, index_hash = public_text(dart,url,cache,
        lambda text:viewer_identity(text,candidate['rcept_no'],corp,candidate['rcept_dt']),cached_only)
    identity = viewer_identity(index,candidate['rcept_no'],corp,candidate['rcept_dt'])
    eligible = [r for r in identity['family'] if '투자설명서' in r['report_nm']
                and pd.Timestamp(r['rcept_dt'])+pd.Timedelta(days=1)<cutoff]
    latest = max(eligible,key=lambda r:(r['rcept_dt'],r['rcept_no'])) if eligible else None
    if latest is None or latest['rcept_no'] != candidate['rcept_no']:
        raise ValueError('related_prospectus_not_latest_in_family')
    if any(re.search(r'철회|취소',r['report_nm']) and r['rcept_dt']>=latest['rcept_dt']
           and pd.Timestamp(r['rcept_dt'])+pd.Timedelta(days=1)<cutoff for r in identity['family']):
        raise ValueError('related_prospectus_withdrawn')
    if terms_receipt not in {r['rcept_no'] for r in related(index)}:
        raise ValueError('related_prospectus_reverse_link_missing')
    nodes = []
    for block in re.split(r'var\s+node\d+\s*=\s*\{\s*\}\s*;',index)[1:]:
        fields = dict(re.findall(r"node\d+\['(text|rcpNo|dcmNo|eleId|offset|length|dtd)'\]\s*=\s*\"([^\"]*)\"",block))
        if not re.search(r'모집.*매출.*일반사항|투자위험요소',fields.get('text','')): continue
        if (fields.get('rcpNo') != latest['rcept_no'] or fields.get('dtd') not in ('dart3.xsd','dart4.xsd')
                or any(not re.fullmatch(r'\d+',fields.get(k,'')) for k in ('dcmNo','eleId','offset','length'))):
            raise ValueError('related_prospectus_section_identity_conflict')
        nodes.append({k:v for k,v in fields.items() if k!='text'})
    if len(nodes)!=2: raise ValueError('related_prospectus_sections_not_unique')
    bodies, sections = [], []
    for node in nodes:
        source_url = 'https://dart.fss.or.kr/report/viewer.do?'+urlencode(node)
        def validate_body(text):
            if '<table' not in text.lower(): raise ValueError('related_prospectus_body_invalid')
        body, sha = public_text(dart,source_url,cache,validate_body,cached_only)
        bodies.append(body);sections.append({'url':source_url,'sha256':sha})
    return latest,'\n'.join(bodies),{'source_url':url,'final_index_sha256':index_hash,'corp_code':corp,
        'family':identity['family'],'sections':sections,'related_from_receipt':terms_receipt}


def repair_features(dart, raw_dir, original, event_limit=None, cached_only=False):
    """Only price/institutional-excluded general IPOs; never retail or fitting."""
    from data.pipelines.historical_ipo_pipeline import HistoricalIPOPipeline
    from data.processors.feature_engineer import FeatureEngineer
    raw = Path(raw_dir)
    records = pd.read_parquet(raw / 'dart_ipo_raw.parquet')
    seed_reviewed_sources(raw, records)
    events = pd.read_parquet(raw / 'krx_official_event_master.parquet')
    lineage = pd.read_parquet(raw / 'dart_disclosure_lineage.parquet')
    targets = original[original.event_class.eq('general_ipo') & original.market.isin(['KOSPI', 'KOSDAQ'])
                       & (~original.offering_price_review_status.isin(VERIFIED_OFFERING_PRICE_STATUSES)
                          | ~original.institutional_validation_status.isin(APPROVED_INSTITUTIONAL_DEMAND_STATUSES))]
    if event_limit:
        targets = targets.sort_values('listing_date').iloc[
            sorted({round(i*(len(targets)-1)/max(1,event_limit-1)) for i in range(min(event_limit,len(targets)))})]
    resolutions, added_lineage = [], []
    consecutive_failures = 0
    for target in targets.itertuples(index=False):
        source = records[records.event_id.eq(target.event_id)]
        event = events[events.event_id.eq(target.event_id)]
        entry = {'event_id': target.event_id, 'before_status': target.offering_price_review_status,
                 'status': 'review_required', 'absence_confirmed': False}
        resolutions.append(entry)
        initial_receipt = records.loc[records.event_id.eq(target.event_id), 'rcept_no']
        url = 'https://dart.fss.or.kr/dsaf001/main.do?' + urlencode({'rcpNo': str(initial_receipt.iloc[0])}) if len(initial_receipt)==1 else ''
        initial_cache = raw / 'dart_connection_sources' / (hashlib.sha256(url.encode()).hexdigest() + '.json')
        if consecutive_failures >= 3 and not initial_cache.exists():
            entry.update(status='pending_source_cooldown', reason='three_consecutive_transport_failures')
            continue
        try:
            if len(source) != 1 or len(event) != 1:
                raise ValueError('event_identity_not_unique')
            row = source.iloc[0]
            members = lineage[lineage.event_id.eq(target.event_id)].corp_code.dropna().astype(str).unique()
            if len(members) != 1 or members[0] != str(row.corp_code):
                raise ValueError('event_corp_identity_conflict')
            metadata, body, proof = official_family_document(dart, raw, str(row.rcept_no), str(row.corp_code), target.listing_date, cached_only)
            consecutive_failures = 0
            document = dart._parse_offering_html(body, metadata['rcept_no'])
            demand_document = dart._parse_demand_forecast_html(body, str(row.corp_code))
            document.update({k:v for k,v in demand_document.items() if k not in {'corp_code', 'parse_success'}})
            price = document.get('offering_price')
            if (document.get('offering_price_review_status') != 'verified_currency_unit'
                    or document.get('offering_price_finality') != 'confirmed_price_language'):
                raise ValueError('latest_official_price_not_explicitly_confirmed')
            if not pd.notna(event.iloc[0].offering_price) or float(price) != float(event.iloc[0].offering_price):
                raise ValueError('latest_official_price_krx_mismatch')
            changes = {k:v for k,v in document.items() if k.startswith('offering_price')}
            changes.update(offering_price_rcept_no=metadata['rcept_no'], offering_price_rcept_dt=pd.Timestamp(metadata['rcept_dt']))
            demand, sources = HistoricalIPOPipeline._select_dart_lineage_demand([(metadata, document)], price, target.listing_date)
            changes.update(demand, **sources)
            if (document.get('new_shares_parser_validation_status') == 'structurally_verified'
                    and document.get('secondary_shares_parser_validation_status') == 'structurally_verified'):
                changes.update({k:v for k,v in document.items() if k.startswith(('new_shares', 'secondary_shares'))})
                changes.update(offering_structure_rcept_no=metadata['rcept_no'],
                               offering_structure_rcept_dt=pd.Timestamp(metadata['rcept_dt']))
            from data.pipelines.float_source_resolver import resolve_public_float
            resolved_float = resolve_public_float([(metadata, document)], price, target.listing_date)
            if resolved_float.get('public_float_resolution_status') == 'latest_disclosed_source_selected':
                changes.update(resolved_float)
            for prefix, field in [('institutional', 'institutional_demand_ratio'), ('lockup', 'lockup_commitment_ratio')]:
                if demand.get(field) is not None:
                    changes[prefix + '_validation_status'] = DART_STRUCTURAL_AGGREGATE_STATUS
                    changes[prefix + '_available_at'] = pd.Timestamp(metadata['rcept_dt'])
            # A missing value in this bounded terms section does not prove nonpublication.
            # Preserve prior verified observations unless this source verifies a replacement.
            changes = {k:v for k,v in changes.items() if pd.api.types.is_scalar(v) and v is not None}
            for key, value in changes.items():
                if key not in records: records[key] = None
                records.at[source.index[0], key] = value
            entry.update(status='recovered_latest_official_family', receipt=metadata['rcept_no'],
                         price=float(price), evidence=json.dumps(proof, ensure_ascii=False))
            added_lineage.append({'event_id': target.event_id, 'corp_code': str(row.corp_code),
                'rcept_no': metadata['rcept_no'], 'rcept_dt': pd.Timestamp(metadata['rcept_dt']),
                'filing_report_nm': metadata['report_nm'], 'source_url': proof['source_url'],
                'lineage_validation_status': 'official_viewer_corp_and_bidirectional_family_verified'})
        except requests.RequestException as exc:
            consecutive_failures += 1
            entry.update(status='source_access_failed', reason=type(exc).__name__)
        except ValueError as exc:
            entry.update(status='pending_source_download' if str(exc)=='official_source_not_cached' else 'review_required', reason=str(exc))
    engineer = FeatureEngineer('phase2')
    rebuilt = engineer.build_features(records, events, pd.read_parquet(raw/'kospi_index.parquet'),
                                      pd.read_parquet(raw/'kosdaq_index.parquet'))
    if set(rebuilt.event_id) != set(original.event_id) or len(rebuilt) != len(original):
        raise RuntimeError('connection_repair_event_population_changed')
    return rebuilt, pd.DataFrame(resolutions), pd.concat([lineage, pd.DataFrame(added_lineage)], ignore_index=True)
