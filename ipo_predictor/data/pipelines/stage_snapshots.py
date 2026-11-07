"""Point-in-time IPO inputs from actual historical documents, not final-row masks."""
import hashlib
import fcntl
import json
import math
import re
import tempfile
import unicodedata
from copy import deepcopy
from decimal import Decimal, ROUND_HALF_UP
from html import escape
from pathlib import Path
from urllib.parse import urlencode, parse_qs

import pandas as pd
import requests
from bs4 import BeautifulSoup

from data.collectors.dart_collector import DARTCollector, _TableRowParser
from data.pipelines.data_delivery import checked_artifact, digest
from data.pipelines.disclosure_connections import public_text, viewer_identity
from data.pipelines.financial_history import atomic_json, summarize_asof, FINANCIAL_STATUS
from data.pipelines.disclosure_financials import select_financial_node, parse_disclosure_financials, validate_financial_section_response, financial_records
from data.pipelines.reconciled_dataset import pin_connections
from data.pipelines.underwriter_source import parse_underwriters
from features.model_profiles import get_model_profile
from features.prediction_time_contract import validate_prediction_times

POLICY = 'historical_stage_documents_v1'
EXTRA_FEATURES = ('price_band_low', 'price_band_high', 'revenue_growth_3y', 'operating_margin', 'debt_ratio')


class StageSourceMissing(ValueError):
    def __init__(self, url):
        super().__init__('official_source_not_cached')
        self.url = url


class StageResponseInvalid(ValueError):
    def __init__(self,url):
        super().__init__('stage_section_response_invalid');self.url=url


def section_text(dart,url,cache,cached_only,*,financial=False):
    def validate(text):
        if financial:
            try:validate_financial_section_response(text)
            except ValueError as exc:raise StageResponseInvalid(url) from exc
            return
        if '<table' in text.lower():return
        raise StageResponseInvalid(url)
    try:
        return public_text(dart,url,cache,validate,cached_only)
    except StageResponseInvalid:
        if cached_only:raise
        return public_text(dart,url,cache,validate,False,refresh=True)
    except ValueError as exc:
        if str(exc)=='official_source_not_cached':raise StageSourceMissing(url) from exc
        raise


def local(value):
    value = pd.Timestamp(value)
    return value.tz_localize('Asia/Seoul') if value.tzinfo is None else value.tz_convert('Asia/Seoul')


def available(metadata):
    return local(metadata['rcept_dt']).normalize() + pd.Timedelta(days=1)


def demand_schedule(html):
    """Read an explicitly labelled range; never infer dates from receipt numbers."""
    parser = _TableRowParser(); parser.feed(html); parser.close()
    candidates = {}
    def period(text):
        matches = re.findall(r'(?:(20\d{2})\s*(?:년|[./-])\s*)?(\d{1,2})\s*(?:월|[./-])\s*(\d{1,2})\s*일?', text)
        if not matches or not matches[0][0]:return None
        if len(matches)==1:
            short=re.search(r'[~～]\s*(\d{1,2})\s*(?:일|(?=\([월화수목금토일]\)))',text)
            if short:matches.append((matches[0][0],matches[0][1],short[1]))
        if len(matches)!=2:return None
        try:dates=[pd.Timestamp(int(y or matches[0][0]),int(m),int(d)) for y,m,d in matches]
        except ValueError:raise ValueError('demand_schedule_invalid_or_reversed_dates')
        if not pd.Timedelta(0)<=dates[1]-dates[0]<=pd.Timedelta(days=30):
            raise ValueError('demand_schedule_invalid_or_reversed_dates')
        return tuple(dates)
    for table_id, table in enumerate(parser.tables):
        for row_id, row in enumerate(table):
            if not row or not re.fullmatch(r'수요예측(?:접수|예정)?(?:기간|일시|일자|일정|일)', re.sub(r'\s+', '', row[0])):
                continue
            dates=period(' '.join(row[1:]))
            if dates is not None:
                scope=re.sub(r'\s+','',row[1]) if len(row)>2 and re.sub(r'\s+','',row[1]) in {'국내','해외'} else 'all'
                candidates[dates]={'table':table_id,'row':row_id,'cells':row,'scope':scope}
    if not candidates:
        text=DARTCollector._normalize_text(html)
        for match in re.finditer(r'수요\s*예측\s*예정일은\s*([^。]{8,140}?)입니다',text):
            dates=period(match[1])
            if dates is not None:candidates[dates]={'kind':'explicit_scheduled_period_sentence','text':match[0],'scope':'all'}
    if len(candidates)==2 and {item['scope'] for item in candidates.values()}=={'국내','해외'}:
        return min(pair[0] for pair in candidates),max(pair[1] for pair in candidates),{
            'rule':'before_any_domestic_or_foreign_bookbuilding_v1','scoped_windows':list(candidates.values())}
    if len(candidates) != 1:
        raise ValueError('demand_schedule_missing_or_conflicting')
    (start, end), evidence = next(iter(candidates.items()))
    return start, end, evidence


def read_document(dart, raw, metadata, corp, cached_only):
    """Read this exact receipt; a family dropdown must not switch to the latest filing."""
    receipt = metadata['rcept_no']; cache = Path(raw)/'dart_connection_sources'
    url = 'https://dart.fss.or.kr/dsaf001/main.do?'+urlencode({'rcpNo':receipt})
    def obtain(source,validator=None):
        try:return public_text(dart,source,cache,validator,cached_only)
        except ValueError as exc:
            if str(exc)=='official_source_not_cached':raise StageSourceMissing(source) from exc
            raise
    index, index_hash = obtain(url,lambda text:viewer_identity(text, receipt, corp, metadata['rcept_dt']))
    related={parse_qs(option.get('value','')).get('rcpNo',[''])[0]
             for option in BeautifulSoup(index,'html.parser').select('select#ref option')}
    nodes = []
    for block in re.split(r'var\s+node\d+\s*=\s*\{\s*\}\s*;', index)[1:]:
        node = dict(re.findall(r'node\d+\[\'(text|rcpNo|dcmNo|eleId|offset|length|dtd)\'\]\s*=\s*"([^\"]*)"', block))
        title = unicodedata.normalize('NFKC', node.get('text', '')).strip()
        if not re.search(r'모집.*매출.*일반사항|투자위험요소|증권발행조건확정', title):
            continue
        if (node.get('rcpNo') != receipt or node.get('dtd') not in ('dart3.xsd', 'dart4.xsd')
                or any(not re.fullmatch(r'\d+', node.get(k, '')) for k in ('dcmNo','eleId','offset','length'))):
            raise ValueError('stage_section_identity_conflict')
        nodes.append(node)
    full = [n for n in nodes if re.match(r'^[IVX]+[.]\s*(?:모집.*매출.*일반사항|투자위험요소)',
                                       unicodedata.normalize('NFKC', n['text']).strip())]
    full_scope=any('일반사항' in n['text'] for n in full)
    selected=full if full_scope else [n for n in nodes if '증권발행조건확정' in n['text']]
    if not selected:raise ValueError('stage_document_section_not_identified')
    pieces, sections = [], []
    for node in selected:
        source = 'https://dart.fss.or.kr/report/viewer.do?'+urlencode({k:v for k,v in node.items() if k!='text'})
        body, sha = section_text(dart,source,cache,cached_only)
        stored=json.loads((cache/(hashlib.sha256(source.encode()).hexdigest()+'.json')).read_text())
        pieces.append(body); sections.append({'url':source, 'sha256':sha,'collected_at':stored.get('collected_at')})
    return '\n'.join(pieces), {'receipt':receipt, 'published_at':metadata['rcept_dt'],
        'available_at':available(metadata).isoformat(), 'source_url':url, 'index_sha256':index_hash,
        'sections':sections, 'corp_code':corp,'content_scope':'full' if full_scope else 'partial_terms_notice',
        'related_receipts':sorted(number for number in related if re.fullmatch(r'\d{14}',number))}


def document_analysis(dart,body,receipt,corp,raw=None):
    identity={'schema':1,'receipt':receipt,'corp_code':corp,'body_sha256':hashlib.sha256(body.encode()).hexdigest(),
        'parser':digest(Path(__file__).parents[1]/'collectors/dart_collector.py'),
        'float_parser':digest(Path(__file__).parents[1]/'collectors/float_tables.py'),
        'underwriters':digest(Path(__file__).with_name('underwriter_source.py')),
        'tier_mapping':digest(Path(__file__).parents[1]/'processors/feature_engineer.py')}
    path=None if raw is None else Path(raw)/'stage_document_analyses'/(hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()+'.json')
    if path is not None and path.exists():
        saved=json.loads(path.read_text())
        if saved['identity']!=identity or saved['sha256']!=hashlib.sha256(json.dumps(saved['result'],sort_keys=True).encode()).hexdigest():
            raise RuntimeError('stage_document_analysis_integrity_failed')
        return saved['result']
    result={'offering':dart._parse_offering_html(body,receipt),'demand':dart._parse_demand_forecast_html(body,corp),
            'underwriter':parse_underwriters(body)}
    if path is not None:atomic_json(path,{'identity':identity,'result':result,
        'sha256':hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()})
    return result


def same_verified_lockup_table(verified, other):
    """An identical outcome table can lack its separate denominator in a full copy."""
    if (verified.get('lockup_parser_validation_status')!='structurally_verified'
            or other.get('lockup_parser_validation_status')!='candidate_rejected'
            or other.get('lockup_rejection_reason')!='lockup_outcome_quantity_schema_or_reconciliation_unverified'):
        return False
    try:
        left=verified.get('lockup_structured_evidence')
        right=other.get('lockup_structured_evidence')
        left=json.loads(left) if isinstance(left,str) else left
        right=json.loads(right) if isinstance(right,str) else right
        table=left.get('table') if isinstance(left,dict) else None
        candidates=right.get('candidate_tables') if isinstance(right,dict) else None
        return bool(table and candidates and all(c.get('table')==table for c in candidates))
    except (TypeError,ValueError):
        return False


def result_day_document(metadata, terms, load, analyse=None):
    """A same-day complete prospectus must agree with explicit terms amendments."""
    day=[(m,*load(m)) for m in terms if m['rcept_dt']==metadata['rcept_dt']]
    complete=[item for item in day if item[2].get('content_scope','full')=='full']
    if not complete:raise ValueError('stage_partial_document_requires_composition')
    prospectuses=[item for item in complete if '투자설명서' in item[0].get('report_nm','')]
    chosen=max(complete,key=lambda item:item[0]['rcept_no'])
    dart=DARTCollector(api_key='unused')
    analyse=analyse or (lambda m,b,c:document_analysis(dart,b,m['rcept_no'],c))
    role_selection=None
    if prospectuses and '증권신고서' in chosen[0].get('report_nm',''):
        prospectus=max(prospectuses,key=lambda item:item[0]['rcept_no'])
        registered=analyse(chosen[0],chosen[1],chosen[2]['corp_code'])
        final=analyse(prospectus[0],prospectus[1],prospectus[2]['corp_code'])
        if (registered['offering'].get('offering_price_finality')=='preliminary_price_language'
                and registered['demand'].get('institutional_demand_parser_validation_status')=='value_not_found'
                and final['offering'].get('offering_price_finality')=='confirmed_price_language'
                and final['offering'].get('offering_price_review_status')=='verified_currency_unit'
                and final['demand'].get('institutional_demand_parser_validation_status')=='structurally_verified'
                and chosen[2]['corp_code']==prospectus[2]['corp_code']
                and chosen[0]['rcept_no'] in prospectus[2].get('related_receipts',[])
                and prospectus[0]['rcept_no'] in chosen[2].get('related_receipts',[])):
            role_selection={'rule':'linked_final_prospectus_over_planned_unpopulated_registration_v1',
                'registration_receipt':chosen[0]['rcept_no'],'prospectus_receipt':prospectus[0]['rcept_no']}
            chosen=prospectus
    full_analysis=deepcopy(analyse(chosen[0],chosen[1],chosen[2]['corp_code']))
    full_price=full_analysis['offering'];full_demand=full_analysis['demand']
    notices=[];field_sources={};held_fields={};band_notices=[];full_checks=[]
    for other in sorted(complete,key=lambda item:item[0]['rcept_no']):
        m,body,origin=other
        if m['rcept_no']<=chosen[0]['rcept_no']:continue
        if (origin['corp_code']!=chosen[2]['corp_code']
                or m['rcept_no'] not in chosen[2].get('related_receipts',[])
                or chosen[0]['rcept_no'] not in origin.get('related_receipts',[])):
            raise ValueError('stage_later_full_document_not_linked')
        other_analysis=analyse(m,body,origin['corp_code']);other_price=other_analysis['offering']
        if (other_price.get('offering_price_finality')=='confirmed_price_language'
                and other_price.get('offering_price_review_status')=='verified_currency_unit'
                and other_price.get('offering_price')!=full_price.get('offering_price')):
            raise ValueError('stage_later_full_document_price_conflict')
        for field in ('new_shares','secondary_shares','price_band_low','price_band_high',
                      'public_float_shares','public_float_ratio_disclosed','total_post_listing_shares'):
            if (other_price.get(field) is not None and full_price.get(field) is not None
                    and other_price[field]!=full_price[field]):
                raise ValueError('stage_later_full_document_supply_or_band_conflict')
        other_demand=other_analysis['demand']
        for field,prefix in [('institutional_demand_ratio','institutional_demand'),('lockup_commitment_ratio','lockup')]:
            if (other_demand.get(prefix+'_parser_validation_status')=='structurally_verified'
                    and full_demand.get(prefix+'_parser_validation_status')=='structurally_verified'
                    and other_demand.get(field)!=full_demand.get(field)
                    and not (field=='institutional_demand_ratio' and same_demand_printed_rounding(other_demand,full_demand))):
                raise ValueError('stage_later_full_document_result_conflict')
        full_checks.append({'receipt':m['rcept_no'],'origin':origin,
            'comparison':'linked_same_day_registration_not_silently_skipped',
            'price_finality':other_price.get('offering_price_finality'),
            'institutional_status':other_demand.get('institutional_demand_parser_validation_status')})
    from data.pipelines.band_source_resolver import superseded_terms_evidence
    prior_terms=superseded_terms_evidence(day,chosen)
    superseded=[]
    for m,body,proof in sorted(day,key=lambda item:item[0]['rcept_no']):
        if proof.get('content_scope','full')=='full':continue
        if m['rcept_no'] in prior_terms:
            superseded.append(prior_terms[m['rcept_no']])
            continue
        linked=(m['rcept_no'] in chosen[2].get('related_receipts',[])
                and chosen[0]['rcept_no'] in proof.get('related_receipts',[])
                and chosen[2]['corp_code']==proof['corp_code']
                and '투자설명서' in chosen[0].get('report_nm',''))
        if re.search(r'정정\s*전',body) and re.search(r'정정\s*후',body):
            try:body,view=amendment_view(body)
            except ValueError as exc:
                view=None
                if (str(exc)=='correction_after_without_matching_before' and linked
                        and full_price.get('offering_price_finality')=='confirmed_price_language'
                        and full_price.get('offering_price_review_status')=='verified_currency_unit'):
                    try:body,view=added_note_after_view(body)
                    except ValueError:pass
                    if view is not None:view['strict_section_rejection']=str(exc)
                # A later, bidirectionally linked full prospectus is the source,
                # not a guessed composition of an incomplete correction notice.
                if view is None and (not linked or chosen[0]['rcept_no']<=m['rcept_no']
                        or full_price.get('offering_price_finality')!='confirmed_price_language'
                        or full_price.get('offering_price_review_status')!='verified_currency_unit'):
                    raise
                if view is None:
                    body,view=horizontal_after_cells(body)
                    view.update(strict_section_rejection=str(exc),
                        replacement_rule='later_bidirectionally_linked_complete_prospectus_v1',
                        replacement_receipt=chosen[0]['rcept_no'])
        else:view=None
        partial_analysis=analyse(m,body,proof['corp_code'])
        price=partial_analysis['offering'];demand=partial_analysis['demand']
        if view and view.get('rule')=='explicit_overview_empty_before_additions_v1':
            if (price.get('offering_price_review_status')!='verified_currency_unit'
                    or price.get('offering_price_finality')!='confirmed_price_language'
                    or price.get('offering_price')!=full_price.get('offering_price')):
                raise ValueError('stage_added_notice_price_not_reconciled')
            if chosen[0]['rcept_no']<m['rcept_no']:
                for field in ('new_shares','secondary_shares','price_band_low','price_band_high',
                              'public_float_shares','public_float_ratio_disclosed','total_post_listing_shares'):
                    if (price.get(field) is not None and full_price.get(field) is not None
                            and price[field]!=full_price[field]):
                        raise ValueError('stage_added_notice_supply_or_band_conflict')
        band_notices.append({'receipt':m['rcept_no'],'document':price,'origin':proof,
                             'metadata':m,'linked':linked,'view':view})
        if view and view.get('rule')=='horizontal_after_cells_consistency_only_v1':
            explicit_prices={int(x.replace(',','')) for x in re.findall(
                r'(?:모집\s*\(\s*매출\s*\)|모집\s*및\s*매출|모집|매출)\s*'
                r'(?:확정)?가액\s*(?:\(\s*(?:예정|확정)\s*\))?\s*[:：]\s*([0-9][0-9,]*)\s*원(?!\s*[~～])',
                DARTCollector._normalize_text(body))}
            if not explicit_prices or explicit_prices!={full_price.get('offering_price')}:
                raise ValueError('stage_horizontal_amendment_price_not_reconciled')
        if (price.get('offering_price_finality')=='confirmed_price_language'
                and price.get('offering_price_review_status')=='verified_currency_unit'
                and price.get('offering_price')!=full_price.get('offering_price')):
            raise ValueError('stage_same_day_amendment_price_conflict')
        checks={}
        for field,prefix in [('institutional_demand_ratio','institutional_demand'),('lockup_commitment_ratio','lockup')]:
            if demand.get(prefix+'_parser_validation_status')!='structurally_verified':continue
            if demand.get(field)==full_demand.get(field):continue
            full_status=full_demand.get(prefix+'_parser_validation_status')
            full_unpopulated=(full_status=='value_not_found' or full_status=='candidate_rejected'
                and full_demand.get(prefix+'_rejection_reason') in {
                    'risk_or_planned_threshold_context','selected_document_lockup_rules_without_identified_outcome'})
            same_table=field=='lockup_commitment_ratio' and same_verified_lockup_table(demand,full_demand)
            if (linked and (full_unpopulated or same_table)
                    and price.get('offering_price_finality')=='confirmed_price_language'
                    and price.get('offering_price_review_status')=='verified_currency_unit'
                    and price.get('offering_price')==full_price.get('offering_price')):
                for key,value in demand.items():
                    if key.startswith(prefix+'_') or key==field:full_demand[key]=value
                field_sources[field]={**proof,'source_selection_rule':'same_day_linked_confirmed_terms_field_v1'}
                checks[field]=('verified_terms_identical_full_outcome_table' if same_table else
                               'verified_terms_value_full_document_unpopulated')
                continue
            if field=='institutional_demand_ratio' and same_demand_printed_rounding(demand,full_demand):
                checks[field]='same_application_quantity_and_count_printed_rounding';continue
            if field=='lockup_commitment_ratio':
                held_fields[field]={'partial_receipt':m['rcept_no'],'partial_value':demand.get(field),
                    'complete_receipt':chosen[0]['rcept_no'],'complete_value':full_demand.get(field),
                    'complete_parser_rejection':full_demand.get(prefix+'_rejection_reason')}
                full_demand[field]=None;full_demand[prefix+'_parser_validation_status']='candidate_rejected'
                full_demand[prefix+'_rejection_reason']='same_day_official_lockup_conflict'
                field_sources.pop(field,None);checks[field]='held_optional_field_not_entire_stage'
                continue
            raise ValueError('stage_same_day_amendment_result_conflict')
        notices.append({'origin':proof,'after_correction_view':view,'field_checks':checks})
    from data.pipelines.band_source_resolver import select_notice_band
    band,band_donor,band_checks=select_notice_band(full_price,band_notices,full_price.get('offering_price'),
                                                complete_receipt=chosen[0]['rcept_no'])
    full_analysis['offering']=band
    if band_donor:
        def component(origin):
            return {key:origin[key] for key in ('receipt','corp_code','published_at','available_at',
                                              'source_url','index_sha256','sections')}
        band_origin={**component(band_donor['origin']),'band_after_view':band_donor['view'],
            'source_selection_rule':'same_day_linked_confirmed_after_notice_band_v1'}
        for field in ('price_band_low','price_band_high'):field_sources[field]=band_origin
        price_origin=component(chosen[2])
        sections={s['url']:s for source in (price_origin,band_origin) for s in source['sections']}
        for field in ('offering_price_band_position','band_exceeded'):
            field_sources[field]={**price_origin,'sections':list(sections.values()),
                'available_at':max(local(price_origin['available_at']),local(band_origin['available_at'])).isoformat(),
                'component_sources':{'band':band_origin,'price':price_origin},
                'source_selection_rule':'band_and_confirmed_price_independent_origins_v1'}
    elif band.get('price_band_validation_status')=='conflicting_bands_review_required':
        held_fields['price_band']={'reason':'same_day_official_band_conflict','checks':band_checks}
    return chosen[0],chosen[1],{**chosen[2],'same_day_terms_checks':notices,
        'result_field_sources':field_sources,'held_result_fields':held_fields,'band_source_checks':band_checks,
        'superseded_same_day_terms':superseded,'same_day_later_full_checks':full_checks,
        'final_prospectus_role_selection':role_selection},full_analysis


def added_note_after_view(html):
    """An explicit empty-before overview can declare a new, after-only note."""
    parser=_TableRowParser();parser.feed(html);parser.close();additions={}
    for table_index,table in enumerate(parser.tables):
        if not table or [re.sub(r'\s+','',c) for c in table[0]]!=['항목','정정전','정정후']:continue
        for row_index,row in enumerate(table[1:],1):
            if len(row)!=3 or re.sub(r'\s+','',row[1])!='-':continue
            note=re.fullmatch(r'주([1-9][0-9]*)\)',re.sub(r'\s+','',row[2]))
            if note is None:continue
            if note[1] in additions:raise ValueError('added_note_overview_reference_not_unique')
            additions[note[1]]={'table':table_index,'row':row_index,'cells':row}
    if not additions:raise ValueError('added_note_empty_before_overview_missing')
    from bs4 import Comment
    soup=BeautifulSoup(html,'html.parser');matched={}
    markers={'■정정후','[정정후]','(정정후)','정정후'}
    for node in list(soup.find_all(string=True)):
        if isinstance(node,Comment) or node.find_parent(['table','script','style']) is not None:continue
        note=re.fullmatch(r'주([1-9][0-9]*)\)',re.sub(r'\s+','',str(node)))
        if note is None or note[1] not in additions:continue
        text='';first=None;boundary=False
        for following in node.next_elements:
            if getattr(following,'name',None)=='table':break
            if not isinstance(following,str) or isinstance(following,Comment) or not following.strip():continue
            if following.find_parent(['script','style']) is not None:break
            first=first if first is not None else following
            text+=re.sub(r'\s+','',str(following))
            if text in markers:boundary=True;break
            if not any(marker.startswith(text) for marker in markers):break
        if not boundary:continue
        if note[1] in matched:raise ValueError('added_note_after_boundary_not_unique')
        before=soup.new_tag('span');before.string='[정정 전]'
        first.insert_before(soup.new_tag('br'));first.insert_before(before);first.insert_before(soup.new_tag('br'))
        matched[note[1]]=additions[note[1]]
    if set(matched)!=set(additions):raise ValueError('added_note_after_boundaries_unverified')
    expanded=str(soup);selected,evidence=amendment_view(expanded)
    return selected,{**evidence,'rule':'explicit_overview_empty_before_additions_v1',
        'original_sha256':hashlib.sha256(html.encode()).hexdigest(),
        'empty_before_expansion_sha256':hashlib.sha256(expanded.encode()).hexdigest(),
        'empty_before_notes':matched,'original_paired_sections':evidence['pair_count']-len(matched)}


def same_demand_printed_rounding(left,right):
    """A formatting difference requires the same independently printed raw demand."""
    def facts(result):
        evidence=result.get('institutional_demand_structured_evidence')
        evidence=json.loads(evidence) if isinstance(evidence,str) else evidence
        if not evidence or evidence.get('application_quantity') is None or evidence.get('application_count') is None:return None
        cell=evidence['table'][evidence['selected_row']][evidence['selected_column']]
        match=re.fullmatch(r'\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(?::\s*1)?\s*',cell)
        if not match:return None
        number=Decimal(match[1].replace(',',''))
        return evidence['application_quantity'],evidence['application_count'],number
    a,b=facts(left),facts(right)
    if a is None or b is None or a[:2]!=b[:2]:return False
    quantum=Decimal(1).scaleb(max(a[2].as_tuple().exponent,b[2].as_tuple().exponent))
    return a[2].quantize(quantum,rounding=ROUND_HALF_UP)==b[2].quantize(quantum,rounding=ROUND_HALF_UP)


def horizontal_after_cells(html):
    parser=_TableRowParser();parser.feed(html);parser.close();cells=[];positions=[]
    for index,table in enumerate(parser.tables):
        if not table or [re.sub(r'\s+','',c) for c in table[0]]!=['항목','정정전','정정후']:continue
        for row_index,row in enumerate(table[1:],1):
            if len(row)!=3:raise ValueError('horizontal_correction_column_alignment_unverified')
            cells.append(row[2]);positions.append([index,row_index,2])
    if not cells:raise ValueError('horizontal_correction_header_not_identified')
    # References such as "주3)" are not expanded or used as feature values.
    return ''.join('<p>'+escape(c)+'</p>' for c in cells),{
        'rule':'horizontal_after_cells_consistency_only_v1','selected_cells':positions,
        'original_sha256':hashlib.sha256(html.encode()).hexdigest()}


def amendment_view(html):
    """Normalize explicit text delimiters, never a bare before/after table header."""
    from data.pipelines.correction_views import after_correction_view
    soup=BeautifulSoup(html,'html.parser');normalized=False
    # Inline styling may split a visible marker ("■" + "정정 후").
    # Preserve explicit line breaks and leave all table cells untouched.
    for paragraph in soup.find_all('p'):
        if paragraph.find_parent('table') is not None or paragraph.find(['table','p']) is not None:continue
        for br in paragraph.find_all('br'):br.replace_with('\n')
        text=paragraph.get_text()
        if re.search(r'정정\s*(?:전|후)',text):
            paragraph.clear();paragraph.append(soup.new_string(text));normalized=True
    note=r'(?:\(\s*주(?:석)?\s*\d+\s*\)|주(?:석)?\s*\d+\s*\)|<\s*주(?:석)?\s*\d+\s*>)'
    marker_line=re.compile(r'(?m)^\s*(?:(?P<before>'+note+r')\s*(?:-\s*)?)?'
        r'(?:정정\s*(?P<plain>전|후)\)?|\[\s*정정\s*(?P<bracket>전|후)\s*\]'
        r'|\(\s*정정\s*(?P<parenthesis>전|후)(?:\s*주(?:석)?\s*\d+)?\s*\))'
        r'(?:\s*(?P<after>'+note+r'))?\s*$')
    for node in list(soup.find_all(string=True)):
        if node.find_parent('table') is not None:continue
        text=str(node);matches=list(marker_line.finditer(text))
        if matches:
            end=0
            for match in matches:
                node.insert_before(soup.new_string(text[end:match.start()]))
                node.insert_before(soup.new_string((match['before'] or match['after'] or '')+'\n'))
                marker=soup.new_tag('span')
                marker.string='[정정 '+(match['plain'] or match['bracket'] or match['parenthesis'])+']'
                node.insert_before(marker);end=match.end()
            node.insert_before(soup.new_string(text[end:]));node.extract();normalized=True
    pattern=r'(<\s*정정\s*(?:전|후)\s*>|■\s*정정\s*(?:전|후))'
    for node in list(soup.find_all(string=re.compile(pattern))):
        for piece in re.split(pattern,str(node)):
            match=re.fullmatch(r'(?:<\s*정정\s*(전|후)\s*>|■\s*정정\s*(전|후))',piece)
            text='[정정 '+(match[1] or match[2])+']' if match else piece
            if match:
                marker=soup.new_tag('span');marker.string=text;node.insert_before(marker)
            else:node.insert_before(soup.new_string(text))
            normalized=normalized or bool(match)
        node.extract()
    selected,evidence=after_correction_view(str(soup) if normalized else html)
    if normalized:
        evidence.update(normalization='explicit_line_note_parenthesis_angle_or_bullet_markers_to_brackets_v3',
            normalized_sha256=evidence['original_sha256'],original_sha256=hashlib.sha256(html.encode()).hexdigest())
    return selected,evidence


def stage_documents(family, load, listing, anchor=None, analyse=None):
    """Each stage is selected independently; missing earlier evidence never becomes a final value."""
    ordered = sorted([m for m in family if available(m)<local(listing)],key=lambda m:(m['rcept_dt'],m['rcept_no']))
    terms = [m for m in ordered if re.search(r'증권신고서.*지분증권|투자설명서', m['report_nm'])]
    if not terms:
        return {name:{'status':'offering_lineage_missing'} for name in ('pre_demand','post_demand')}
    withdrawals=[m['rcept_dt'] for m in ordered if re.search(r'철회|취소',m['report_nm'])]
    if withdrawals:
        boundary=max(withdrawals)
        terms=[m for m in terms if m['rcept_dt']>boundary]
        if not terms:return {name:{'status':'offering_withdrawn_without_later_active_document'} for name in ('pre_demand','post_demand')}
    output = {};canonical_period=None;post_not_before=None
    try:
        seed,body,proof=anchor if anchor is not None else (terms[-1],*load(terms[-1]))
        start, end, schedule = demand_schedule(body)
        if local(seed['rcept_dt']).normalize()<local(end):
            raise ValueError('canonical_demand_period_not_confirmed_by_later_document')
        canonical_period=(start,end)
        post_not_before=local(end).normalize()+pd.Timedelta(days=2 if 'scoped_windows' in schedule else 1)
        # This retrospective event-date anchor is not an input feature. The
        # actual input document must independently announce that same period.
        anchor_proof={'receipt':seed['rcept_no'],'published_at':seed['rcept_dt'],
                      'schedule':schedule,'source_url':proof.get('source_url'),
                      'sections':proof.get('sections',[]),'index_sha256':proof.get('index_sha256',proof.get('final_index_sha256')),
                      'use':'event_date_label_only_never_feature_values'}
        # Separate overseas windows may use local dates without an intraday
        # timezone. The prior KST day start precedes even the earliest window.
        cutoff = local(start).normalize()-pd.Timedelta(hours=24 if 'scoped_windows' in schedule else 3)
        if not cutoff < local(listing):raise ValueError('demand_period_not_before_listing')
        eligible = [m for m in terms if available(m)<cutoff]
        if not eligible:raise ValueError('pre_demand_schedule_not_known_before_cutoff')
        latest=eligible[-1]
        if any(re.search(r'철회|취소',m['report_nm']) and available(m)<cutoff
               and m['rcept_dt']>=latest['rcept_dt'] for m in ordered):
            raise ValueError('stage_family_withdrawn_before_cutoff')
        body, proof = load(latest)
        if proof.get('content_scope','full')!='full':
            raise ValueError('pre_demand_partial_document_requires_composition')
        announced_start,announced_end,schedule=demand_schedule(body)
        if (announced_start,announced_end)!=(start,end):
            raise ValueError('canonical_period_not_announced_in_pre_cutoff_document')
        output['pre_demand'] = {'status':'historical_document_selected','metadata':latest,
            'html':body,'proof':proof,'prediction_at':cutoff,
            'demand_start_at':local(start),'demand_end_at':local(end),
            'milestone_evidence':{'announced_before_cutoff':schedule,'event_date_anchor':anchor_proof}}
    except (ValueError, requests.RequestException) as exc:
        output['pre_demand'] = {'status':str(exc) if isinstance(exc,ValueError) else 'source_access_failed'}
    try:
        days=set()
        partial_days=[]
        final_terms=[m['rcept_dt'] for m in terms if '발행조건확정' in m['report_nm']]
        first_result_day=str(canonical_period[1].date()) if canonical_period else min(final_terms) if final_terms else None
        for metadata in terms:
            if first_result_day is not None and metadata['rcept_dt']<first_result_day:continue
            if metadata['rcept_dt'] in days:continue
            days.add(metadata['rcept_dt'])
            try:metadata,body,proof,analysis=result_day_document(metadata,terms,load,analyse)
            except ValueError as exc:
                if str(exc)=='stage_partial_document_requires_composition':
                    partial_days.append(metadata['rcept_dt']);continue
                raise
            demand=analysis['demand'];price=analysis['offering']
            if (demand.get('institutional_demand_parser_validation_status') != 'structurally_verified'
                    or price.get('offering_price_review_status') != 'verified_currency_unit'
                    or price.get('offering_price_finality') != 'confirmed_price_language'):
                continue
            result_available=available(metadata)
            result_origin=proof;first_result_receipt=metadata['rcept_no']
            cutoff = max(result_available,post_not_before) + pd.Timedelta(minutes=1) if post_not_before is not None else result_available+pd.Timedelta(minutes=1)
            if not cutoff < local(listing):
                raise ValueError('post_demand_result_not_available_before_listing')
            if any(re.search(r'철회|취소',m['report_nm']) and available(m)<cutoff
                   and m['rcept_dt']>=metadata['rcept_dt'] for m in ordered):
                raise ValueError('stage_family_withdrawn_before_cutoff')
            newer=[m for m in terms if available(m)<cutoff and m['rcept_dt']>metadata['rcept_dt']]
            if newer:
                latest=max(newer,key=lambda m:(m['rcept_dt'],m['rcept_no']))
                metadata,body,proof,analysis=result_day_document(latest,terms,load,analyse)
                if (analysis['demand'].get('institutional_demand_parser_validation_status')!='structurally_verified'
                        or analysis['offering'].get('offering_price_finality')!='confirmed_price_language'):
                    raise ValueError('later_pre_prediction_amendment_not_verified')
            from data.pipelines.band_source_resolver import historical_indicative_band, currency_band, BAND_FIELDS
            offering=analysis['offering']
            if (currency_band(offering) is None
                    and offering.get('price_band_validation_status')!='conflicting_bands_review_required'
                    and all(offering.get(f) is None for f in BAND_FIELDS[:2])):
                analyser=analyse or (lambda m,b,c:document_analysis(DARTCollector(api_key='unused'),b,m['rcept_no'],c))
                donor,history=historical_indicative_band((metadata,body,proof),ordered,load,analyser,cutoff)
                proof['historical_band_selection']=history
                if donor:
                    offering.update({f:donor['document'].get(f) for f in BAND_FIELDS})
                    def component(origin):
                        return {k:origin[k] for k in ('receipt','corp_code','published_at','available_at',
                                                     'source_url','index_sha256','sections')}
                    band_origin={**component(donor['origin']),'band_after_view':None,
                        'source_selection_rule':history['rule'],'band_history':history}
                    sources=proof.setdefault('result_field_sources',{})
                    for f in ('price_band_low','price_band_high'):sources[f]=band_origin
                    price_origin=component(proof)
                    sections={s['url']:s for source in (price_origin,band_origin) for s in source['sections']}
                    for f in ('offering_price_band_position','band_exceeded'):
                        sources[f]={**price_origin,'sections':list(sections.values()),
                            'available_at':max(local(price_origin['available_at']),local(band_origin['available_at'])).isoformat(),
                            'component_sources':{'band':band_origin,'price':price_origin},
                            'source_selection_rule':'band_and_confirmed_price_independent_origins_v1'}
            output['post_demand'] = {'status':'historical_document_selected','metadata':metadata,
                'html':body,'proof':proof,'analysis':analysis,'prediction_at':cutoff,
                'demand_result_available_at':result_available,'post_stage_not_before':post_not_before,
                'demand_result_time_source':result_origin['source_url'],
                'milestone_evidence':{'first_verified_complete_result_receipt':first_result_receipt,
                                      'result_origin':result_origin,
                                      'earlier_terms_only_days_not_used_for_features':partial_days}}
            break
        else:
            output['post_demand'] = {'status':'no_verified_result_in_checked_lineage'}
    except (ValueError, requests.RequestException) as exc:
        output['post_demand'] = {'status':str(exc) if isinstance(exc,ValueError) else 'source_access_failed'}
    return output


def values_from_document(dart, selected, corp, stage):
    body = selected['html']; receipt = selected['metadata']['rcept_no']
    analysis=selected.get('analysis') or document_analysis(dart,body,receipt,corp)
    document=analysis['offering'];demand=analysis['demand']
    values, reasons = {}, {}
    for field in ('price_band_low','price_band_high','offering_price_band_position','band_exceeded'):
        reasons[field]=document.get('price_band_validation_status') or 'band_not_identified_in_selected_section'
    for field in ('price_band_low','price_band_high'):
        if document.get('price_band_validation_status') == 'direct_currency_band':
            values[field] = document.get(field)
    structure = all(document.get(f+'_parser_validation_status')=='structurally_verified'
                    and document.get(f) is not None for f in ('new_shares','secondary_shares'))
    if structure:
        total = document['new_shares']+document['secondary_shares']
        if total>0:values['secondary_offering_ratio']=document['secondary_shares']/total
    reasons['secondary_offering_ratio']='offering_structure_pair_incomplete_or_unverified'
    reasons['float_share_ratio']=document.get('public_float_parse_method') or 'float_not_identified_in_selected_sections'
    if document.get('public_float_parse_method')=='disclosed_public_float_ratio_direct_context':
        values['float_share_ratio']=document.get('public_float_ratio_disclosed')
    elif (document.get('public_float_parse_method')=='disclosed_public_float_shares_direct_context'
          and document.get('total_post_listing_shares_parse_method')=='same_table_post_offering_scope_v1'
          and document.get('total_post_listing_shares_parser_validation_status')=='structurally_verified'):
        total=document.get('total_post_listing_shares');shares=document.get('public_float_shares')
        if total and shares is not None and 0<=shares<=total:values['float_share_ratio']=shares/total
    underwriter = analysis['underwriter']
    if underwriter.get('underwriter_tier') is not None:values['underwriter_tier']=underwriter['underwriter_tier']
    reasons['underwriter_tier']=underwriter['underwriter_validation_status']
    if stage=='post_demand':
        for field, prefix in [('institutional_demand_ratio','institutional_demand'),('lockup_commitment_ratio','lockup')]:
            if demand.get(prefix+'_parser_validation_status')=='structurally_verified':values[field]=demand.get(field)
            reasons[field]=demand.get(prefix+'_rejection_reason') or 'value_not_found_in_selected_document'
        if (document.get('offering_price_review_status')=='verified_currency_unit'
                and document.get('offering_price_finality')=='confirmed_price_language'):
            values['offering_price']=document['offering_price']
            low, high = values.get('price_band_low'), values.get('price_band_high')
            if low is not None and high is not None and high>low:
                values['offering_price_band_position']=(document['offering_price']-low)/(high-low)
                values['band_exceeded']=float(document['offering_price']>high)
    elif demand.get('institutional_demand_ratio') is not None:
        raise ValueError('pre_demand_document_already_contains_demand_result')
    selected['field_evidence']={
        'price_band_low':document.get('price_band_parse_evidence'),
        'price_band_high':document.get('price_band_parse_evidence'),
        'float_share_ratio':document.get('public_float_parse_evidence'),
        'secondary_offering_ratio':{f:document.get(f+'_parse_evidence') for f in ('new_shares','secondary_shares')},
        'underwriter_tier':underwriter.get('underwriter_source_evidence'),
        'institutional_demand_ratio':demand.get('institutional_demand_structured_evidence'),
        'lockup_commitment_ratio':demand.get('lockup_structured_evidence'),
        'offering_price_band_position':document.get('price_band_parse_evidence'),
        'band_exceeded':document.get('price_band_parse_evidence')}
    return values, reasons


def set_stage_offering_price_metadata(row, selected):
    """Keep the contemporaneous price separate from the eventual target price."""
    if row['prediction_stage']!='post_demand' or row['status']!='historical_document_selected':
        return
    price=selected['analysis']['offering'];origin=selected['proof']
    if (price.get('offering_price_review_status')!='verified_currency_unit'
            or price.get('offering_price_finality')!='confirmed_price_language'):
        return
    value=price.get('offering_price')
    if value is None or not math.isfinite(float(value)) or value<=0:
        raise ValueError('post_demand_input_price_invalid')
    if not local(origin['available_at'])<local(row['prediction_at']):
        raise ValueError('post_demand_input_price_not_available_at_cutoff')
    if not origin.get('sections'):
        raise ValueError('post_demand_input_price_original_sections_missing')
    source={key:origin[key] for key in ('receipt','corp_code','source_url','index_sha256','sections','available_at')}
    source.update(unit='KRW_per_share',validation_status='verified_stage_asof_offering_price_v1',
        parse_method=price.get('offering_price_parse_method'),
        raw_context=price.get('offering_price_audit_context'),role='stage_input_not_realized_target')
    row.update(offering_price=float(value),offering_price_available_at=origin['available_at'],
               offering_price_source=json.dumps(source,ensure_ascii=False,sort_keys=True))


def stage_collection_fields():
    """Retain collected observations independently of a model's feature subset."""
    return tuple(dict.fromkeys((*get_model_profile('pre_demand').feature_names,
                                *get_model_profile('post_demand').feature_names,*EXTRA_FEATURES)))


def stage_feature_missing_reason(stage, field, reasons, status):
    if stage=='pre_demand' and field in {'institutional_demand_ratio','lockup_commitment_ratio',
                                        'offering_price_band_position','band_exceeded'}:
        return 'not_available_before_demand_result'
    return reasons.get(field,status if status!='historical_document_selected' else 'selected_asof_source_not_verified')


def stage_feature_unit(field):
    if field in ('price_band_low','price_band_high'):return 'KRW_per_share'
    return {'institutional_demand_ratio':'times','underwriter_tier':'ordinal_tier',
            'offering_price_band_position':'dimensionless','major_shareholder_lockup_months':'months',
            'risk_factor_count':'count','band_exceeded':'flag','offering_type_spac_ipo':'flag',
            'recent_ipo_avg_return_all':'percent','recent_ipo_avg_return_sector':'percent'}.get(field,'fraction')


def stage_observation(row, field, value, origin, reasons):
    return {'event_id':row['event_id'],'stage':row['prediction_stage'],'feature_name':field,'value':value,
        'is_missing':value is None,'missing_reason':None if value is not None else
            stage_feature_missing_reason(row['prediction_stage'],field,reasons,row['status']),
        'unit':stage_feature_unit(field),
        'validation_status':origin.get('validation_status','verified_stage_asof_input_v1') if value is not None else 'unverified_or_missing',
        'available_at':origin.get('available_at'),'prediction_at':row['prediction_at'],
        'available_at_is_effective':True,
        'time_validation_status':'pre_listing_verified' if value is not None else 'missing',
        'source':json.dumps(origin,ensure_ascii=False),'absence_confirmed':False}


def financials_from_selected_document(dart,raw,selected,corp,cached_only):
    """An early stage needs its own annual section, not the eventual final prospectus."""
    origin=selected['proof'];receipt=origin['receipt'];cache=Path(raw)/'dart_connection_sources'
    index_path=cache/(hashlib.sha256(origin['source_url'].encode()).hexdigest()+'.json')
    payload=json.loads(index_path.read_text())
    if (payload['sha256']!=origin['index_sha256']
            or hashlib.sha256(payload['text'].encode()).hexdigest()!=origin['index_sha256']):
        raise RuntimeError('stage_financial_index_integrity_failed')
    node=select_financial_node(payload['text'],receipt)
    if node is None:return pd.DataFrame(),{'status':'selected_document_financial_section_not_identified','absence_confirmed':False}
    source='https://dart.fss.or.kr/report/viewer.do?'+urlencode(node)
    if (node['dtd'] not in ('dart3.xsd','dart4.xsd')
            or any(not re.fullmatch(r'\d+',node[key]) for key in ('dcmNo','eleId','offset','length'))):
        raise ValueError('stage_financial_section_identity_invalid')
    html,sha=section_text(dart,source,cache,cached_only,financial=True)
    identity={'body':sha,'receipt':receipt,'corp_code':corp,'published_at':origin['published_at'],
              'parser':digest(Path(__file__).with_name('disclosure_financials.py'))}
    parsed_cache=Path(raw)/'stage_financial_analyses'/(hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()+'.json')
    if parsed_cache.exists():
        saved=json.loads(parsed_cache.read_text());result=saved['result']
        if saved['identity']!=identity or saved['sha256']!=hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest():
            raise RuntimeError('stage_financial_analysis_integrity_failed')
        return pd.DataFrame(result['records']),result['evidence']
    diagnostics={}
    frame=parse_disclosure_financials(html,corp,receipt,origin['published_at'],source,diagnostics)
    if not frame.empty:
        stored=json.loads((cache/(hashlib.sha256(source.encode()).hexdigest()+'.json')).read_text())
        frame['source_collected_at']=stored.get('collected_at')
        frame['source_hash_scope']='original_official_section_html_utf8'
    evidence={'status':'annual_accounts_extracted' if len(frame) else
        'official_statement_unit_conflict_requires_review' if diagnostics.get('official_statement_unit_conflicts') else
        'official_statement_amount_conflict_requires_review' if diagnostics.get('official_statement_amount_conflicts') else
        'summary_statement_basis_unverified' if diagnostics.get('summary_primary_basis_unresolved_columns') else
        'official_financial_section_without_tables' if '<table' not in html.lower() else 'annual_format_or_basis_unverified',
              'url':source,'sha256':sha,'details':diagnostics,'absence_confirmed':False}
    result={'records':financial_records(frame),'evidence':evidence}
    atomic_json(parsed_cache,{'identity':identity,'result':result,
        'sha256':hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()})
    return frame,evidence


def original_cache_hashes(raw,objects):
    urls=set()
    def visit(value):
        if isinstance(value,dict):
            for child in value.values():visit(child)
        elif isinstance(value,list):
            for child in value:visit(child)
        elif isinstance(value,str) and value.startswith('https://dart.fss.or.kr/'):
            urls.add(value)
    visit(objects)
    result={}
    for url in urls:
        name=hashlib.sha256(url.encode()).hexdigest()+'.json';path=Path(raw)/'dart_connection_sources'/name
        result[name]=digest(path) if path.exists() else None
    return result


def reusable_checkpoint(raw,saved,cached_only):
    for name,sha in saved['source_cache_sha256'].items():
        path=Path(raw)/'dart_connection_sources'/name
        if sha is not None:
            if not path.exists() or digest(path)!=sha:raise RuntimeError('stage_checkpoint_original_missing_or_changed')
        elif path.exists() or not cached_only:return False
    return cached_only or not any(a['status'] in {'source_access_failed','stage_section_response_invalid'} for a in saved['attempts'])


def stage_checkpoint_identity(event, identity, index_hash, annual_history, target, reusable_annual_parser):
    return {'record':hashlib.sha256(event.record.encode()).hexdigest(),
        'proof':hashlib.sha256(event.proof.encode()).hexdigest(),
        'code':identity['builder'],'dependencies':identity['dependency_sha256'],'parser':identity['parser'],
        'index':index_hash,
        'annual':hashlib.sha256(annual_history.to_json(orient='records',date_format='iso').encode()).hexdigest(),
        'annual_parser_reusable':bool(reusable_annual_parser),
        'target':target[['offering_price','open_return_pct','close_return_pct','listing_date']].to_json(date_format='iso')}


def stage_financial_missing_reason(field, annual, collection_status):
    if annual.get(field) is not None:return None
    if annual.get('financial_time_validation_status') != FINANCIAL_STATUS:
        return (annual.get('financial_time_validation_status',collection_status)
                if collection_status in {'annual_accounts_extracted','verified_stored_annual_history'} else collection_status)
    if field=='revenue_growth_3y':return 'positive_same_basis_revenue_endpoints_three_years_apart_unverified'
    if field=='operating_margin':
        if annual.get('revenue') is not None and annual['revenue']<=0:return 'nonpositive_revenue_margin_undefined'
        return 'verified_same_period_operating_income_and_revenue_pair_unavailable'
    if field=='debt_ratio':
        if annual.get('equity') is not None and annual['equity']<=0:return 'nonpositive_equity_ratio_undefined'
        return 'verified_same_period_liabilities_and_equity_pair_unavailable'
    return collection_status


def current_annual_history(accounts, reusable_parser):
    if reusable_parser or accounts.empty:
        return accounts
    # Structured API rows are raw records; old disclosure extraction is not.
    if 'report_code' not in accounts:
        return accounts.iloc[0:0].copy()
    return accounts[accounts.report_code.eq('11011')].copy()


def run(raw_dir, processed_dir, *, cached_only=True, event_limit=None, event_ids=None, dart=None):
    processed=Path(processed_dir);processed.mkdir(parents=True,exist_ok=True)
    with (processed/'stage_snapshots.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return {'status':'already_running','training_executed':False}
        return _run(raw_dir,processed_dir,cached_only=cached_only,event_limit=event_limit,event_ids=event_ids,dart=dart)


def _run(raw_dir, processed_dir, *, cached_only, event_limit, event_ids, dart):
    raw, processed = Path(raw_dir).resolve(), Path(processed_dir).resolve()
    source, source_report = pin_connections(processed)
    pointer_names=('current_historical_source_connections.json','current_reconciled_data.json','current_reconciliation_run.json')
    pointer_hashes={name:digest(processed/name) for name in pointer_names}
    ptr=json.loads((processed/'current_reconciled_data.json').read_text());dataset=processed/ptr['manifest']
    if digest(dataset)!=ptr['sha256']:raise RuntimeError('stage_dataset_pointer_invalid')
    report=checked_artifact(dataset.parent)
    if report['inputs']['sources']!=digest(source/'manifest.json'):raise RuntimeError('stage_source_dataset_version_mismatch')
    records=pd.read_parquet(source/'connection_results.parquet').sort_values('event_id')
    if event_ids is not None:
        if event_limit is not None or not event_ids or not set(event_ids).issubset(set(records.event_id)):
            raise ValueError('stage_event_selection_invalid')
        records=records[records.event_id.isin(event_ids)]
    if event_limit is not None:
        if event_limit<=0:raise ValueError('stage_limit_must_be_positive')
        records=records.iloc[:event_limit]
    targets=pd.read_parquet(dataset.parent/'features_all.parquet').set_index('event_id',drop=False)
    index=pd.read_parquet(dataset.parent/'kospi_index_snapshot.parquet').sort_values('date')
    index['available_at']=index.date.map(lambda x:local(x).normalize()+pd.Timedelta(days=1))
    run_pointer=json.loads((processed/'current_reconciliation_run.json').read_text())
    financial=processed/run_pointer['manifest']
    if digest(financial)!=run_pointer['sha256']:raise RuntimeError('stage_run_pointer_invalid')
    run_report=json.loads(financial.read_text());accounts_path=Path(run_report['financial'])/'accounts.parquet'
    finance_report=checked_artifact(accounts_path.parent)
    if (run_report['input_sha256']['dataset']!=digest(dataset)
            or run_report['input_sha256']['financial']!=digest(accounts_path.parent/'manifest.json')):
        raise RuntimeError('stage_financial_dataset_version_mismatch')
    accounts=pd.read_parquet(accounts_path)
    dart=dart or DARTCollector();rows=[];ledger=[];attempts=[];milestones=[]
    identity={'policy':POLICY,'sources':digest(source/'manifest.json'),'dataset':digest(dataset),
              'accounts':digest(accounts_path),'builder':digest(__file__), 'parser':digest(Path(__file__).parents[1]/'collectors/dart_collector.py')}
    dependencies={'time_contract':Path(__file__).parents[2]/'features/prediction_time_contract.py',
        'profiles':Path(__file__).parents[2]/'features/model_profiles.py',
        'underwriters':Path(__file__).with_name('underwriter_source.py'),
        'tier_mapping':Path(__file__).parents[1]/'processors/feature_engineer.py',
        'financial_summary':Path(__file__).with_name('financial_history.py'),
        'correction_views':Path(__file__).with_name('correction_views.py'),
        'source_document_selection':Path(__file__).with_name('lockup_reconciliation.py'),
        'band_sources':Path(__file__).with_name('band_source_resolver.py'),
        'financial_parser':Path(__file__).with_name('disclosure_financials.py'),
        'float_tables':Path(__file__).parents[1]/'collectors/float_tables.py'}
    identity['dependency_sha256']={name:digest(path) for name,path in dependencies.items()}
    reusable_annual_parser=(finance_report['code']['parser']==identity['dependency_sha256']['financial_parser']
        and finance_report['code']['summary']==identity['dependency_sha256']['financial_summary'])
    accounts=current_annual_history(accounts,reusable_annual_parser)
    index_hash=digest(dataset.parent/'kospi_index_snapshot.parquet')
    reused_events=0
    for event in records.itertuples(index=False):
        record=json.loads(event.record);proof=json.loads(event.proof);memo={}
        event_identity=stage_checkpoint_identity(event,identity,index_hash,
            accounts[accounts.event_id.eq(event.event_id)],targets.loc[event.event_id],reusable_annual_parser)
        key=hashlib.sha256(json.dumps(event_identity,sort_keys=True).encode()).hexdigest()
        checkpoint=raw/'stage_event_checkpoints'/(key+'.json')
        if checkpoint.exists():
            saved=json.loads(checkpoint.read_text())
            payload=saved['payload']
            if saved['identity']!=event_identity or saved['sha256']!=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest():
                raise RuntimeError('stage_event_checkpoint_integrity_failed')
            if reusable_checkpoint(raw,payload,cached_only):
                rows.extend(payload['rows']);ledger.extend(payload['ledger']);attempts.extend(payload['attempts']);milestones.extend(payload['milestones'])
                reused_events+=1
                continue
        starts=[len(x) for x in (rows,ledger,attempts,milestones)]
        family={m['rcept_no']:m for b in [proof,proof.get('supplementary',{})] for m in b.get('family',[])}
        def load(metadata):
            receipt=metadata['rcept_no']
            if receipt not in memo:
                try:
                    memo[receipt]=read_document(dart,raw,metadata,str(record['corp_code']),cached_only)
                    attempts.append({'event_id':event.event_id,'receipt':receipt,'status':'original_verified',
                        'evidence':json.dumps(memo[receipt][1],ensure_ascii=False)})
                except (ValueError,requests.RequestException) as exc:
                    memo[receipt]=exc
                    attempts.append({'event_id':event.event_id,'receipt':receipt,'status':str(exc) if isinstance(exc,ValueError) else 'source_access_failed',
                        'evidence':json.dumps({'url':getattr(exc,'url','https://dart.fss.or.kr/dsaf001/main.do?rcpNo='+receipt),'absence_confirmed':False})})
                    raise
            if isinstance(memo[receipt],Exception):raise memo[receipt]
            return memo[receipt]
        from data.pipelines.lockup_reconciliation import documents as approved_documents
        try:
            anchor_metadata,anchor_bundle,anchor_body=approved_documents(raw,record,proof)[0]
            anchor=(anchor_metadata,anchor_body,anchor_bundle)
        except ValueError:anchor=None
        analyse=lambda m,b,c:document_analysis(dart,b,m['rcept_no'],c,raw)
        stages=stage_documents(list(family.values()),load,record['listing_date'],anchor,analyse)
        for stage,selected in stages.items():
            fields=stage_collection_fields()
            row={'event_id':event.event_id,'corp_code':record['corp_code'],'corp_name':event.corp_name,
                'listing_date':record['listing_date'],'prediction_stage':stage,'status':selected['status'],
                'prediction_at':None,'selected_receipt':None,'offering_price':None,
                'prediction_time_source':POLICY,'demand_start_at':None,'demand_start_time_source':None,
                'demand_result_available_at':None,'demand_result_time_source':None,
                'post_stage_not_before':None,
                'demand_result_time_is_effective':True,**dict.fromkeys(fields)}
            target=targets.loc[event.event_id]
            row.update(target_offering_price=float(target.offering_price),open_return_pct=float(target.open_return_pct),
                       close_return_pct=float(target.close_return_pct))
            values={};reasons={};origins={}
            if selected['status']=='historical_document_selected':
                cutoff=selected['prediction_at'];origin=selected['proof'];row.update(prediction_at=cutoff.isoformat(),selected_receipt=origin['receipt'])
                if stage=='pre_demand':
                    row.update(demand_start_at=selected['demand_start_at'].isoformat(),demand_start_time_source=origin['source_url'])
                else:
                    row.update(demand_result_available_at=selected['demand_result_available_at'].isoformat(),
                               demand_result_time_source=selected['demand_result_time_source'],
                               post_stage_not_before=selected['post_stage_not_before'].isoformat() if selected.get('post_stage_not_before') is not None else None)
                try:
                    if 'analysis' not in selected:
                        selected['analysis']=analyse(selected['metadata'],selected['html'],str(record['corp_code']))
                    values,reasons=values_from_document(dart,selected,str(record['corp_code']),stage)
                    set_stage_offering_price_metadata(row,selected)
                except ValueError as exc:row['status']=str(exc);values={}
                origins={f:{**origin,'extraction_evidence':selected.get('field_evidence',{}).get(f)}
                         for f in set(values)|set(reasons)}
                for field,source in origin.get('result_field_sources',{}).items():
                    if field in values:
                        origins[field]={**source,'extraction_evidence':selected.get('field_evidence',{}).get(field),
                            'linked_complete_receipt':origin['receipt']}
                past=index[index.available_at<cutoff]
                for window in [5,20]:
                    field=f'kospi_momentum_{window}d'
                    if len(past)>window:
                        values[field]=float(past.close.iloc[-1]/past.close.iloc[-window-1]-1)
                        origins[field]={'source_url':'pinned_kospi_index_snapshot','sha256':digest(dataset.parent/'kospi_index_snapshot.parquet'),
                                        'available_at':past.available_at.iloc[-1].isoformat()}
                history=targets[targets.event_class.eq('general_ipo')].reset_index(drop=True).copy()
                history['available_at']=history.listing_date.map(lambda x:local(x).normalize()+pd.Timedelta(hours=21))
                history=history[history.available_at<cutoff].sort_values(['available_at','event_id']).tail(10)
                if len(history):
                    values['recent_ipo_avg_return_all']=float(history.open_return_pct.mean())
                    origins['recent_ipo_avg_return_all']={'source_url':'pinned_verified_prior_ipo_targets',
                        'event_ids':history.event_id.tolist(),'available_at':history.available_at.max().isoformat(),'sha256':digest(dataset.parent/'features_all.parquet')}
                annual_history=accounts[accounts.event_id.eq(event.event_id)]
                annual=summarize_asof(annual_history,str(record['corp_code']),cutoff)
                financial_status='verified_stored_annual_history'
                selected_annual_already_parsed=(reusable_annual_parser and not annual_history.empty
                    and annual_history.rcept_no.astype(str).eq(origin['receipt']).any())
                if not selected_annual_already_parsed and any(annual.get(field) is None for field in EXTRA_FEATURES[2:]):
                    try:
                        extra,financial_evidence=financials_from_selected_document(dart,raw,selected,str(record['corp_code']),cached_only)
                        financial_status=financial_evidence['status']
                        attempts.append({'event_id':event.event_id,'receipt':origin['receipt'],
                            'status':financial_status,'evidence':json.dumps(financial_evidence,ensure_ascii=False)})
                        if not extra.empty:
                            annual_history=pd.concat([annual_history,extra],ignore_index=True) if not annual_history.empty else extra
                            annual=summarize_asof(annual_history,str(record['corp_code']),cutoff)
                    except (ValueError,requests.RequestException) as exc:
                        financial_status=str(exc) if isinstance(exc,ValueError) else 'source_access_failed'
                        attempts.append({'event_id':event.event_id,'receipt':origin['receipt'],'status':financial_status,
                            'evidence':json.dumps({'url':getattr(exc,'url',origin['source_url']),'absence_confirmed':False})})
                financial_proof=json.loads(annual.get('financial_feature_provenance','{}'))
                for field in EXTRA_FEATURES[2:]:
                    reasons[field]=stage_financial_missing_reason(field,annual,financial_status)
                    origins[field]={'receipt':origin['receipt'],'source_url':origin['source_url'],
                                    'financial_collection_status':financial_status}
                    if field in financial_proof and annual.get(field) is not None:
                        values[field]=annual[field];origins[field]=financial_proof[field]
                values['offering_type_spac_ipo']=0.0
                origins['offering_type_spac_ipo']={**origin,'classification':'general_ipo_document_identity'}
                milestones.append({'event_id':event.event_id,'stage':stage,'prediction_at':cutoff.isoformat(),
                    'demand_start_at':str(selected.get('demand_start_at','')),
                    'demand_result_available_at':str(selected.get('demand_result_available_at','')),
                    'evidence':json.dumps({'origin':origin,'milestone':selected['milestone_evidence']},ensure_ascii=False)})
            for field in fields:
                value=values.get(field);origin=origins.get(field,{})
                if value is not None and (not math.isfinite(float(value)) or not local(origin['available_at'])<local(row['prediction_at'])):
                    raise RuntimeError('stage_feature_time_or_value_violation')
                row[field]=value
                ledger.append(stage_observation(row,field,value,origin,reasons))
            critical=get_model_profile(stage).critical_features
            time_audit=pd.DataFrame([x for x in ledger[-len(fields):]])
            row['stage_time_valid']=bool(validate_prediction_times(pd.DataFrame([row]),get_model_profile(stage),time_audit).iloc[0])
            row['stage_data_ready']=row['status']=='historical_document_selected' and row['stage_time_valid'] and all(row.get(f) is not None for f in critical)
            rows.append(row)
        event_attempts=attempts[starts[2]:]
        evidence=[proof,*[json.loads(a['evidence']) for a in event_attempts]]
        payload={'rows':rows[starts[0]:],'ledger':ledger[starts[1]:],'attempts':event_attempts,
                 'milestones':milestones[starts[3]:],'source_cache_sha256':original_cache_hashes(raw,evidence)}
        atomic_json(checkpoint,{'identity':event_identity,'payload':payload,
            'sha256':hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()})
        if len(rows)%100==0:print(json.dumps({'stage_events_processed':len(rows)//2,'population':len(records)}),flush=True)
    if (pointer_hashes!={name:digest(processed/name) for name in pointer_names}
            or identity['builder']!=digest(__file__)
            or identity['parser']!=digest(Path(__file__).parents[1]/'collectors/dart_collector.py')
            or identity['dependency_sha256']!={name:digest(path) for name,path in dependencies.items()}):
        raise RuntimeError('stage_inputs_or_code_changed')
    outputs={'stages.parquet':pd.DataFrame(rows),'observations.parquet':pd.DataFrame(ledger),
             'milestones.parquet':pd.DataFrame(milestones),'source_attempts.parquet':pd.DataFrame(attempts)}
    identity['attempts']=hashlib.sha256(json.dumps(attempts,sort_keys=True).encode()).hexdigest()
    version=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
    dest=processed/'stage_snapshots'/version;dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():
        staging=Path(tempfile.mkdtemp(prefix='.stages.',dir=dest.parent))
        for name,frame in outputs.items():frame.to_parquet(staging/name,index=False)
        summary={'version':version,'policy':POLICY,'inputs':identity,'events':len(records),
            'scope':'all_connected_general_ipo' if event_limit is None and event_ids is None else 'limited_sample',
            'stages':{s:{'rows':int(len(g)),'ready_rows':int(g.stage_data_ready.sum()),'status_counts':g.status.value_counts().to_dict()}
                      for s,g in outputs['stages.parquet'].groupby('prediction_stage')},
            'training_executed':False,'deployment_authorized':False,
            'output_sha256':{name:digest(staging/name) for name in outputs}}
        atomic_json(staging/'manifest.json',summary);staging.rename(dest)
    if event_limit is None and event_ids is None:
        atomic_json(processed/'current_stage_snapshots.json',{'manifest':str((dest/'manifest.json').relative_to(processed)),'sha256':digest(dest/'manifest.json')})
    return {**checked_artifact(dest),'path':str(dest),'reused_events':reused_events}
