"""Resolve a complete annual section behind an official terms-only correction."""
import re
from pathlib import Path
from urllib.parse import parse_qs, urlencode

import pandas as pd
import requests
from bs4 import BeautifulSoup

from data.collectors.dart_collector import _TableRowParser
from data.pipelines.correction_views import after_correction_view
from data.pipelines.disclosure_connections import public_text, viewer_identity


def compact(text):
    return re.sub(r"\s+", "", str(text))


def related_prospectuses(index):
    rows=[]
    for option in BeautifulSoup(index,'html.parser').select('select#ref option'):
        receipt=parse_qs(option.get('value','')).get('rcpNo',[''])[0]
        label=option.get_text(' ',strip=True)
        day=re.match(r'(\d{4})[.](\d{2})[.](\d{2})',label)
        if (re.fullmatch(r'\d{14}',receipt) and day
                and option.get('title')=='투자설명서' and not re.search('철회|취소',label)):
            rows.append({'rcept_no':receipt,'rcept_dt':'-'.join(day.groups())})
    return rows


def nonfinancial_terms_scope(html):
    """Only a complete explicit amendment overview can prove the unchanged scope."""
    parser=_TableRowParser();parser.feed(html);parser.close()
    candidates=[(i,grid) for i,grid in enumerate(parser.tables)
                if grid and [compact(c) for c in grid[0]]==['항목','정정전','정정후']]
    if len(candidates)!=1:
        raise ValueError('financial_terms_amendment_overview_not_unique')
    table_id,grid=candidates[0]
    allowed={'I.모집또는매출에관한일반사항','Ⅰ.모집또는매출에관한일반사항',
             'III.투자위험요소','Ⅲ.투자위험요소',
             'IV.인수인의의견(분석기관의평가의견)','Ⅳ.인수인의의견(분석기관의평가의견)',
             'V.자금의사용목적','Ⅴ.자금의사용목적'}
    active=None;notes=set()
    for cells in grid[1:]:
        if len(cells)!=3 or any(not compact(c) for c in cells):
            raise ValueError('financial_terms_amendment_overview_incomplete')
        label=compact(cells[0]);joined=compact(' '.join(cells))
        if re.search('재무|회계|감사|제2부|발행인에관한',joined):
            raise ValueError('financial_terms_correction_may_affect_annual_statements')
        if label in allowed:
            if any(compact(c)!=label for c in cells):
                raise ValueError('financial_terms_amendment_heading_conflict')
            active=label
        elif label=='공통정정사항':
            if not all(re.match(r'(?:모집|매출)',compact(c)) for c in cells[1:]):
                raise ValueError('financial_terms_common_change_scope_unverified')
        elif label=='요약정보':
            if not all('제1부모집또는매출에관한사항' in compact(c) and '동일' in compact(c)
                       for c in cells[1:]):
                raise ValueError('financial_terms_summary_change_scope_unverified')
        elif active is not None and re.match(r'\d+[.)]',label):
            before,after=(re.fullmatch(r'주\s*(\d+)\s*\)',c.strip()) for c in cells[1:])
            if before is None or after is None or before[1]!=after[1]:
                raise ValueError('financial_terms_amendment_note_pair_unverified')
            notes.add(int(before[1]))
        else:
            raise ValueError('financial_terms_amendment_scope_unverified')
    _,pairs=after_correction_view(html)
    if not notes or notes!=set(range(1,max(notes)+1)) or pairs['pair_count']!=len(notes):
        raise ValueError('financial_terms_amendment_body_incomplete')
    return {'rule':'complete_terms_overview_excludes_annual_financial_chapter_v1',
            'table_id':table_id,'table':grid,'note_numbers':sorted(notes),'correction_view':pairs}


def resolve_financial_section(dart, raw_dir, receipt, corp_name, corp_code, cutoff, published, *, cached_only=False):
    from data.pipelines.disclosure_financials import public_financial_section
    raw=Path(raw_dir);cache=raw/'dart_connection_sources';cut=pd.Timestamp(cutoff).tz_localize(None)
    def before_cutoff(day):
        return pd.Timestamp(day)+pd.Timedelta(days=1)<cut
    def index(number,day=None):
        url='https://dart.fss.or.kr/dsaf001/main.do?'+urlencode({'rcpNo':number})
        text,sha=public_text(dart,url,cache,lambda body:viewer_identity(body,number,corp_code,day),cached_only)
        return text,{'url':url,'sha256':sha},viewer_identity(text,number,corp_code,day)
    original=public_financial_section(dart,receipt,corp_name,raw/'dart_ipo_financial_sections',
                                      corp_code=corp_code,published=published,cached_only=cached_only)
    if original.get('status')!='financial_section_not_found':return original
    try:
        root,root_source,identity=index(receipt)
        if not before_cutoff(identity['publication_date']):
            raise ValueError('financial_actual_publication_after_cutoff')
        candidates=[r for r in related_prospectuses(root) if before_cutoff(r['rcept_dt'])]
        if not candidates:raise ValueError('financial_related_prospectus_not_identified')
        chosen=max(candidates,key=lambda r:(r['rcept_dt'],r['rcept_no']))
        body,prospectus_source,prospectus=index(chosen['rcept_no'],chosen['rcept_dt'])
        eligible=[r for r in prospectus['family'] if before_cutoff(r['rcept_dt'])]
        if any(re.search('철회|취소',r['report_nm']) for r in eligible):
            raise ValueError('financial_related_prospectus_withdrawn')
        filings=[r for r in eligible if '투자설명서' in r['report_nm']]
        if not filings or max(filings,key=lambda r:(r['rcept_dt'],r['rcept_no']))['rcept_no']!=chosen['rcept_no']:
            raise ValueError('financial_related_prospectus_not_latest')
        reverse={parse_qs(o.get('value','')).get('rcpNo',[''])[0]
                 for o in BeautifulSoup(body,'html.parser').select('select#ref option')}
        if receipt not in reverse:raise ValueError('financial_related_prospectus_reverse_link_missing')
        family=[r for r in identity['family'] if before_cutoff(r['rcept_dt'])]
        if any(re.search('철회|취소',r['report_nm']) for r in family):
            raise ValueError('financial_terms_family_withdrawn')
        later=[r for r in family if r['rcept_dt']>=chosen['rcept_dt']]
        if receipt not in {r['rcept_no'] for r in later}:
            raise ValueError('financial_related_prospectus_chronology_conflict')
        sources=[root_source,prospectus_source];corrections=[]
        for entry in sorted(later,key=lambda r:(r['rcept_dt'],r['rcept_no'])):
            number=entry['rcept_no'];text,origin,official=index(number,entry['rcept_dt'])
            nodes=[]
            for block in re.split(r'var\s+node\d+\s*=\s*\{\s*\}\s*;',text)[1:]:
                fields=dict(re.findall(r"node\d+\['(text|rcpNo|dcmNo|eleId|offset|length|dtd)'\]\s*=\s*\"([^\"]*)\"",block))
                if fields:nodes.append(fields)
            if (len(nodes)!=1 or compact(nodes[0].get('text'))!='증권발행조건확정'
                    or nodes[0].get('rcpNo')!=number or nodes[0].get('dtd') not in ('dart3.xsd','dart4.xsd')
                    or any(not re.fullmatch(r'\d+',nodes[0].get(k,'')) for k in ('dcmNo','eleId','offset','length'))):
                raise ValueError('financial_later_correction_scope_requires_review')
            url='https://dart.fss.or.kr/report/viewer.do?'+urlencode({k:v for k,v in nodes[0].items() if k!='text'})
            terms,sha=public_text(dart,url,cache,nonfinancial_terms_scope,cached_only)
            corrections.append({'receipt':number,'published_at':official['publication_date'],
                                'scope':nonfinancial_terms_scope(terms),'url':url,'sha256':sha})
            sources.extend([origin,{'url':url,'sha256':sha}])
        result=public_financial_section(dart,chosen['rcept_no'],corp_name,raw/'dart_ipo_financial_sections',
                                        corp_code=corp_code,published=chosen['rcept_dt'],cached_only=cached_only)
        if result.get('status')!='official_section_downloaded':return result
        result=dict(result)
        result['selection_evidence']={'rule':'bidirectional_related_prospectus_unchanged_financial_scope_v1',
            'requested_receipt':receipt,'selected_receipt':chosen['rcept_no'],'cutoff':str(cut),
            'sources':list({s['url']:s for s in sources}.values()),'later_terms_corrections':corrections,
            'absence_confirmed':False}
        return result
    except ValueError as exc:
        return {'status':str(exc),'receipt':receipt,'absence_confirmed':False}
    except requests.RequestException as exc:
        return {'status':'financial_public_source_access_failed','receipt':receipt,
                'error_type':type(exc).__name__,'absence_confirmed':False}
