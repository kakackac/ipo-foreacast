"""Bind the underwriter tier to an explicit pre-listing DART underwriting table."""
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import parse_qs,urlparse

import pandas as pd

from data.collectors.dart_collector import DARTCollector
from data.pipelines.correction_views import after_correction_view
from data.processors.feature_engineer import FeatureEngineer
from features.source_contracts import DART_UNDERWRITER_TIER_STATUS


def parse_underwriters(html):
    candidates=[]
    for index,table in enumerate(DARTCollector._extract_table_cells(html)):
        if not table:continue
        header=[re.sub(r'\s+','',cell) for cell in table[0]]
        if not {'인수인','인수수량','인수방법'}.issubset(header):continue
        columns=[i for i,cell in enumerate(header) if cell=='인수인']
        if len(columns)!=2 or columns[1]!=columns[0]+1:continue
        leads=[];evidence=[]
        for row_index,row in enumerate(table[1:],1):
            if len(row)!=len(header):continue
            role=re.sub(r'\s+','',row[columns[0]])
            if not re.fullmatch(r'(?:공동)?대표(?:주관(?:회사|사)?)?',role):continue
            broker=row[columns[1]].strip()
            quantity=row[header.index('인수수량')].strip()
            if not re.fullmatch(r'[1-9][\d,]*',quantity):continue
            tier=FeatureEngineer._map_underwriter_tier(broker)
            if pd.isna(tier):
                return {'underwriter_validation_status':'official_underwriter_registry_mapping_unavailable',
                        'underwriter_tier':None,'underwriter_source_evidence':json.dumps({'table':index,'broker':broker},ensure_ascii=False)}
            leads.append((broker,int(tier)))
            evidence.append({'row':row_index,'role':role,'broker':broker,'underwritten_shares':int(quantity.replace(',',''))})
        if leads:candidates.append({'leads':list(dict.fromkeys(leads)),'table':index,'rows':evidence})
    if not candidates:return {'underwriter_validation_status':'official_underwriter_table_not_verified','underwriter_tier':None}
    groups={tuple(x['leads']) for x in candidates}
    if len(groups)!=1:return {'underwriter_validation_status':'conflicting_official_underwriter_tables','underwriter_tier':None}
    selected=candidates[0]
    return {'underwriter_tier':selected['leads'][0][1],
            'underwriter_validation_status':DART_UNDERWRITER_TIER_STATUS,
            'underwriter_source_evidence':json.dumps({'rule':'explicit_underwriter_role_and_quantity_v1',
                'matching_tables':candidates,'tier_policy':'first_explicit_lead_in_source_order'},ensure_ascii=False)}


def from_connection(raw,record,proof):
    receipt=record.get('offering_price_rcept_no')
    bundles=[p for p in [proof,proof.get('supplementary',{})]
             if parse_qs(urlparse(p.get('source_url','')).query).get('rcpNo')==[receipt]]
    if len(bundles)!=1:return {'underwriter_tier':None,'underwriter_validation_status':'underwriter_price_source_not_unique'}
    bundle=bundles[0];sections=[]
    for source in bundle['sections']:
        payload=json.loads((Path(raw)/'dart_connection_sources'/(hashlib.sha256(source['url'].encode()).hexdigest()+'.json')).read_text())
        if (payload['url']!=source['url'] or payload['sha256']!=source['sha256']
                or hashlib.sha256(payload['text'].encode()).hexdigest()!=source['sha256']):
            raise RuntimeError('underwriter_source_integrity_failed')
        sections.append(payload['text'])
    html='\n'.join(sections)
    if 'correction_selection' in proof:
        html,evidence=after_correction_view(html)
        if evidence!=proof['correction_selection']:raise RuntimeError('underwriter_correction_selection_changed')
    return {**parse_underwriters(html),'underwriter_rcept_no':receipt,
            'underwriter_available_at':record.get('offering_price_rcept_dt'),
            'underwriter_source_url':bundle['source_url'],
            'underwriter_source_sha256':'|'.join(sorted(source['sha256'] for source in bundle['sections']))}
