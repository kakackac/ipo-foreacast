"""Explicit before/after sections, never positional guesses in correction tables."""
import hashlib
import re
from html import escape
import json
import tempfile
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd

from bs4 import BeautifulSoup


def after_correction_view(html):
    soup=BeautifulSoup(html,'html.parser')
    root=soup.body or soup
    state=None
    blocks=[]
    pairs=0
    tokens=[]
    def visit(element):
        if getattr(element,'name',None)=='table':
            if re.search(r'\[\s*정정\s*(?:전|후)\s*\]',element.get_text(' ',strip=True)):
                raise ValueError('correction_boundary_inside_table_requires_review')
            tokens.append(('table',str(element)))
        elif getattr(element,'name',None) in {'script','style'}:
            return
        elif hasattr(element,'children'):
            for child in element.children:visit(child)
        else:
            tokens.append(('text',str(element)))
    visit(root)
    for position,(kind,text) in enumerate(tokens):
        marker=re.fullmatch(r'\s*\[\s*정정\s*(전|후)\s*\]\s*',text)
        if marker:
            if marker[1]=='전':
                if state=='before':
                    raise ValueError('correction_before_after_pair_incomplete')
                state='before'
            else:
                if state!='before':
                    raise ValueError('correction_after_without_matching_before')
                state='after';pairs+=1
            continue
        if re.search(r'\[\s*정정\s*(?:전|후)\s*\]',text):
            raise ValueError('nested_or_inline_correction_boundary_requires_review')
        if state=='after':
            blocks.append({'position':position,'html':text if kind=='table' else '<p>'+escape(text)+'</p>'})
    if state!='after' or not pairs or not blocks:
        raise ValueError('explicit_correction_pairs_missing')
    result=''.join(block['html'] for block in blocks)
    return result,{'rule':'explicit_marker_before_after_sections_v1','pair_count':pairs,
                   'selected_positions':[block['position'] for block in blocks],
                   'original_sha256':hashlib.sha256(html.encode()).hexdigest(),
                   'selected_sha256':hashlib.sha256(result.encode()).hexdigest()}


def connect_corrected_notice(dart,raw,event,record):
    from data.pipelines.disclosure_connections import official_family_document,company_identity,public_text
    from data.pipelines.event_disclosures import accepted_values,_resolve_supply_bundles
    from data.pipelines.historical_event_recovery import source_record,capture_sources
    from data.pipelines.historical_reconciliation import json_records
    corp=str(record.corp_code)
    metadata,body,proof=official_family_document(dart,raw,str(record.rcept_no),corp,event.listing_date,True)
    if proof.get('content_scope')!='partial_or_terms_notice':
        raise ValueError('explicit_correction_amendment_not_applicable')
    selected,evidence=after_correction_view(body)
    document=dart._parse_offering_html(selected,metadata['rcept_no'])
    document.update({k:v for k,v in dart._parse_demand_forecast_html(selected,corp).items()
                     if k not in {'corp_code','parse_success'}})
    values=accepted_values(document,event.offering_price)
    company_url='https://dart.fss.or.kr/dsae001/selectPopup.ax?'+urlencode({'selectKey':corp})
    company,sha=public_text(dart,company_url,Path(raw)/'dart_connection_sources',
        lambda text:company_identity(text,corp,event.ticker,event.market,verify_current_market=False),True)
    proof.update(correction_selection=evidence,corp_code=corp,field_sources={},company_url=company_url,
        company_identity=company_identity(company,corp,event.ticker,event.market,verify_current_market=False),
        source_selection_policy='latest_explicit_after_correction_values_only_v1')
    _resolve_supply_bundles(values,proof,[({**metadata,'source_url':proof['source_url']},document,values.copy())],
                            event.offering_price,event.listing_date)
    urls={company_url,proof['source_url'],
          'https://dart.fss.or.kr/dsaf001/main.do?'+urlencode({'rcpNo':str(record.rcept_no)}),
          *[s['url'] for s in proof['sections']]}
    expected={company_url:sha,proof['source_url']:proof['final_index_sha256'],
              'https://dart.fss.or.kr/dsaf001/main.do?'+urlencode({'rcpNo':str(record.rcept_no)}):proof['initial_index_sha256'],
              **{section['url']:section['sha256'] for section in proof['sections']}}
    sources=[]
    for url in urls:
        payload=json.loads((Path(raw)/'dart_connection_sources'/(hashlib.sha256(url.encode()).hexdigest()+'.json')).read_text())
        if payload['sha256']!=expected[url]:raise ValueError('correction_original_changed_during_selection')
        sources.append({'url':url,'sha256':payload['sha256'],'collected_at':payload['collected_at']})
    proof['sources']=sources
    originals={};capture_sources(raw,sources,originals)
    parsed=source_record(event,values,metadata,proof,document)
    return {'status':'official_sources_connected','record':json_records(pd.DataFrame([parsed]))[0],
            'proof':proof,'sources':sources},originals


def amend_connections(raw_dir,processed_dir,dart=None):
    from data.collectors.dart_collector import DARTCollector
    from data.pipelines.data_delivery import checked_artifact,digest
    from data.pipelines.financial_history import atomic_json
    from data.pipelines.historical_reconciliation import load_inputs
    from data.pipelines.reconciled_dataset import pin_connections
    raw,processed=Path(raw_dir).resolve(),Path(processed_dir).resolve()
    parent,report=pin_connections(processed)
    connections=pd.read_parquet(parent/'connection_results.parquet')
    selected=connections.reason.eq('price_source_predates_latest_partial_correction')
    if not selected.any():return {**report,'path':str(parent)}
    inputs=load_inputs(raw,processed);events=inputs['events'].set_index('event_id',drop=False)
    records=inputs['records'].set_index('event_id');originals={};patches=[]
    dart=dart or DARTCollector()
    for index,row in connections[selected].iterrows():
        event=next(events.loc[[row.event_id]].itertuples(index=False))
        try:
            result,sources=connect_corrected_notice(dart,raw,event,records.loc[row.event_id])
        except ValueError as exc:
            patches.append({'event_id':row.event_id,'status':'review_required','reason':str(exc)})
            continue
        originals.update(sources)
        connections.at[index,'record']=json.dumps(result['record'],ensure_ascii=False)
        connections.at[index,'proof']=json.dumps(result['proof'],ensure_ascii=False)
        connections.at[index,'status']=result['status'];connections.at[index,'reason']=None
        patches.append({'event_id':row.event_id,'status':result['status'],'rule':result['proof']['correction_selection']['rule']})
    identity={'parent_manifest':str((parent/'manifest.json').relative_to(processed)),
              'parent_sha256':digest(parent/'manifest.json'),'resolver':digest(Path(__file__)),'patches':patches}
    version=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
    destination=processed/'historical_reconciliations'/version
    if not destination.exists():
        staging=Path(tempfile.mkdtemp(prefix='.correction.',dir=destination.parent))
        connections.to_parquet(staging/'connection_results.parquet',index=False)
        pd.read_parquet(parent/'events.parquet').to_parquet(staging/'events.parquet',index=False)
        atomic_json(staging/'source_documents.json',originals)
        amended={**report,'version':version,'amendment':identity,
            'source_archive_policy':'parent_manifest_archive_plus_amendment_originals',
            'status_counts':connections.status.value_counts().to_dict(),
            'output_sha256':{name:digest(staging/name) for name in ['connection_results.parquet','events.parquet','source_documents.json']}}
        atomic_json(staging/'manifest.json',amended);staging.rename(destination)
    atomic_json(processed/'current_historical_source_connections.json',
        {'manifest':str((destination/'manifest.json').relative_to(processed)),'sha256':digest(destination/'manifest.json')})
    return {**checked_artifact(destination),'path':str(destination)}
