"""Issuer annual financial tables from the official pre-listing DART section."""
import hashlib
import ast
import inspect
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from types import SimpleNamespace
from textwrap import dedent
from urllib.parse import urlencode, urlparse, parse_qs

import pandas as pd
import requests
import bs4
from bs4 import BeautifulSoup

from data.collectors.dart_collector import _TableRowParser
from data.pipelines.financial_history import atomic_json


def clean(value):
    return re.sub(r"[^가-힣A-Za-z0-9]", "", str(value))


def issuer_name(value):
    for prefix in ("주식회사", "(주)", "㈜"):
        value = str(value).replace(prefix, "")
    return clean(value)


def select_financial_node(index, receipt):
    matches = []
    for block in re.split(r"var\s+node\d+\s*=\s*\{\s*\}\s*;", index)[1:]:
        fields = dict(re.findall(r"node\d+\['(text|rcpNo|dcmNo|eleId|offset|length|dtd)'\]\s*=\s*\"([^\"]*)\"", block))
        # The chapter and its "기타 재무에 관한 사항" subsection are distinct.
        title = clean(fields.get("text", "")).upper()
        if fields.get("rcpNo") == receipt and re.fullmatch(r"(?:[IVXLC]+|\d+)?재무에관한사항", title):
            if all(key in fields for key in ("rcpNo", "dcmNo", "eleId", "offset", "length", "dtd")):
                matches.append({key: fields[key] for key in ("rcpNo", "dcmNo", "eleId", "offset", "length", "dtd")})
    unique = {tuple(node.items()): node for node in matches}
    return next(iter(unique.values())) if len(unique) == 1 else None


def validate_financial_section_response(html):
    if '<table' in html.lower() and '재무' in html:
        return
    soup=BeautifulSoup(html,'html.parser')
    heading=soup.select_one('p.section-1')
    if (soup.body is not None and soup.select_one('link[href="/css/report_xml.css"]') is not None
            and heading is not None and re.fullmatch(r'(?:[IVXLC]+|\d+)?재무에관한사항',clean(heading.get_text()).upper())
            and soup.find('script') is None):
        return
    raise ValueError('financial_section_response_invalid')


def public_financial_section(dart, receipt, corp_name, cache_dir, corp_code=None, published=None, *, cached_only=False):
    """Read a bounded public section, never the first arbitrary viewer link."""
    path = Path(cache_dir) / f"{receipt}.json"
    cached = None
    if path.exists():
        cached = json.loads(path.read_text())
        if (corp_code is not None and cached.get('official_corp_code') is not None
                and str(corp_code) != cached['official_corp_code']):
            raise RuntimeError('financial_disclosure_cache_corp_code_mismatch')
        if (cached.get('receipt') != receipt or
                hashlib.sha256(cached['html'].encode()).hexdigest()!=cached.get('sha256')):
            raise RuntimeError('financial_disclosure_cache_integrity_failed')
    if not hasattr(dart, "session"):
        return {"status": "financial_public_transport_unavailable"}
    url = "https://dart.fss.or.kr/dsaf001/main.do?" + urlencode({"rcpNo": receipt})
    try:
        from data.pipelines.disclosure_connections import public_text, viewer_identity
        def validate_index(text):
            if corp_code is not None:
                viewer_identity(text,receipt,str(corp_code))
        index,index_hash=public_text(dart,url,Path(cache_dir).parent/'dart_connection_sources',validate_index,cached_only)
        title = re.search(r"<title>([^<]+)</title>", index, re.I)
        issuer = unescape(title.group(1)).split("/")[0].strip() if title else ""
        ids = set(re.findall(r"openCorpInfoNew\(['\"](\d{8})['\"]", index))
        official_corp = next(iter(ids)) if len(ids) == 1 else None
        date_match = re.search(r"/(\d{4})[.](\d{2})[.](\d{2})\s*$", title.group(1)) if title else None
        official_date = "-".join(date_match.groups()) if date_match else None
        if corp_code is not None and official_corp is not None and official_corp != str(corp_code):
            return {"status": "official_document_corp_code_mismatch"}
        if issuer_name(issuer) != issuer_name(corp_name) and not (corp_code is not None and official_corp == str(corp_code)):
            return {"status": "official_document_issuer_mismatch"}
        if official_date is None or (corp_code is not None and official_corp!=str(corp_code)):
            return {"status":"financial_official_identity_or_date_unverified"}
        node = select_financial_node(index, receipt)
        if node is None:
            return {"status": "financial_section_not_found"}
        if (node['dtd'] not in ('dart3.xsd','dart4.xsd')
                or any(not re.fullmatch(r'\d+',node[key]) for key in ('dcmNo','eleId','offset','length'))):
            return {"status":"financial_section_identity_unverified"}
        section_url = "https://dart.fss.or.kr/report/viewer.do?" + urlencode(node)
        cached_url=urlparse(cached.get('source_url','')) if cached is not None else None
        official_url=urlparse(section_url)
        if (cached_url is not None and (cached_url.scheme,cached_url.netloc,cached_url.path)==(
                official_url.scheme,official_url.netloc,official_url.path)
                and parse_qs(cached_url.query)==parse_qs(official_url.query)):
            html=cached['html']
        else:
            html,_=public_text(dart,section_url,Path(cache_dir).parent/'dart_connection_sources',
                validate_financial_section_response,cached_only)
    except requests.RequestException as exc:
        return {"status": "financial_public_source_access_failed", "error_type": type(exc).__name__}
    except ValueError as exc:
        return {'status':str(exc)}
    try:validate_financial_section_response(html)
    except ValueError:return {"status": "financial_section_response_invalid"}
    payload = {"receipt": receipt, "issuer": issuer, "status": "official_section_downloaded",
               "identity_schema_version":2,"index_url":url,"index_sha256":index_hash,
               "official_corp_code": official_corp, "official_publication_date": official_date,
               "requested_publication_date":str(published)[:10] if published is not None else None,
               "publication_date_corrected":published is not None and official_date!=str(published)[:10],
               "html": html, "sha256": hashlib.sha256(html.encode()).hexdigest(),
               "source_url": section_url, "collected_at": (
                   cached.get('collected_at') if cached is not None and cached.get('sha256')==hashlib.sha256(html.encode()).hexdigest()
                   and cached.get('collected_at') else datetime.now(timezone.utc).isoformat())}
    if cached is not None and cached!=payload:
        atomic_json(Path(cache_dir)/'history'/f'{receipt}_{hashlib.sha256(json.dumps(cached,sort_keys=True).encode()).hexdigest()}.json',cached)
    if cached!=payload: atomic_json(path, payload)
    return payload


ACCOUNTS = {
    "매출액": "revenue", "매출": "revenue", "영업수익": "revenue", "수익매출액": "revenue",
    "영업이익": "operating_income", "영업이익손실": "operating_income", "영업손익": "operating_income",
    "당기순이익": "net_income", "당기순이익손실": "net_income", "당기순손익": "net_income",
    "연결총당기순이익": "net_income",
    "자산총계": "total_assets", "부채총계": "total_liabilities", "자본총계": "equity",
    "자산합계": "total_assets", "부채합계": "total_liabilities", "자본합계": "equity",
    "주당순이익원": "eps", "기본주당순이익원": "eps",
    "당분기순이익": "net_income", "당기순손익손실": "net_income",
    "영업이익영업손실": "operating_income", "당기순이익당기순손실": "net_income",
}


def account_name(value):
    return ACCOUNTS.get(clean(re.sub(r"^\s*(?:[IVX]+|\d+)[.)]\s*", "", value)))


def standalone_non_ifrs_caption(text):
    return bool(re.search(r'\[\s*(?:K\s*-?\s*GAAP|K\s*-?\s*GAPP|일반기업회계기준)\s*\]\s*'
                         r'(?:\(\s*단위\s*[:：][^)]*\))?\s*$',text,re.I))


def financial_table_document(html, soup=None):
    """Keep statement title/period grids in the following value table's context."""
    soup = soup if soup is not None else BeautifulSoup(html, "html.parser")
    for table in list(soup.find_all("table")):
        text=table.get_text(" ",strip=True)
        cells=table.find_all(["td","th"])
        period_only=bool(cells) and all(
            re.match(r'제\s*\d+\s*(?:\([^)]*\)\s*)?기',cell.get_text(' ',strip=True))
            and re.search(r'(?:19|20)\d{2}\s*(?:년|[./-])\s*\d{1,2}\s*(?:월|[./-])\s*\d{1,2}',cell.get_text(' ',strip=True))
            and re.search(r'(?:현재|까지)\s*$',cell.get_text(' ',strip=True)) for cell in cells)
        attachment_note=(len(cells)==1 and re.fullmatch(
            r'["“”]?(?:별첨|첨부된)주석은본(?:연결|별도|개별|반기|분기|요약)*재무제표의일부입니다[.。"“”]*',
            re.sub(r'\s+','',text)))
        heading_only=((re.match(r"(?:요약(?:연결|별도|개별){0,2}재무|(?:분기|반기)?(?:연결|별도|개별)?(?:재무제표|재무상태표|(?:포괄)?손익계산서))",clean(text))
                       or re.search(r"단위\s*[:：]",text))
                      and not any(account_name(cell.get_text(" ",strip=True)) for cell in cells)
                      and not any(re.fullmatch(r"\(?-?\d[\d,]*(?:\.\d+)?\)?",cell.get_text(strip=True)) for cell in cells))
        if table.find("table") is not None:
            # Layout wrappers are not financial grids. Preserve their inner tables.
            table.unwrap()
        elif (heading_only or period_only or attachment_note or len(cells) == 1
              and re.search(r"단위\s*[:：]|요약\s*(?:연결|별도)?\s*재무", text)):
            paragraph = soup.new_tag("p")
            paragraph.string = table.get_text(" ", strip=True)
            table.replace_with(paragraph)
    for tag in list(soup.find_all(['br','p','div','h1','h2','h3','li'])):
        if tag.find_parent('table') is not None:continue
        if tag.name=='br':tag.replace_with(soup.new_string(' '))
        else:tag.insert_after(soup.new_string(' '))
    return str(soup)


def financial_document_structure_contract():
    sources={function.__name__:ast.dump(ast.parse(dedent(inspect.getsource(function))),include_attributes=False)
             for function in (financial_table_document,clean,account_name,_TableRowParser)}
    return hashlib.sha256(json.dumps({'schema':2,'sources':sources,'accounts':ACCOUNTS,'beautifulsoup':bs4.__version__,
        'python':list(sys.version_info[:3])},sort_keys=True).encode()).hexdigest()


def financial_document_structure(html, cache_dir=None):
    """Cache raw document geometry only; interpretation and approval always run."""
    identity={'html_sha256':hashlib.sha256(html.encode()).hexdigest(),
              'structure_contract':financial_document_structure_contract()}
    path=None if cache_dir is None else Path(cache_dir)/(hashlib.sha256(
        json.dumps(identity,sort_keys=True).encode()).hexdigest()+'.json')
    if path is not None and path.exists():
        saved=json.loads(path.read_text());result=saved['result']
        if (saved['identity']!=identity or saved['sha256']!=hashlib.sha256(
                json.dumps(result,sort_keys=True).encode()).hexdigest()
                or set(result)!={'tables','table_contexts','outside_text','narrative'}
                or len(result['tables'])!=len(result['table_contexts'])):
            raise RuntimeError('financial_document_structure_integrity_failed')
        return result
    soup=BeautifulSoup(html,'html.parser');narrative=soup.get_text(' ',strip=True)
    parser=_TableRowParser();parser.feed(financial_table_document(html,soup));parser.close()
    result={'tables':parser.tables,'table_contexts':parser.table_contexts,
            'outside_text':parser._outside_text,'narrative':narrative}
    if path is not None:atomic_json(path,{'identity':identity,'result':result,
        'sha256':hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()})
    return result


def summary_scope_heading(context, *, allow_statement_subheading=False):
    """Read explicit prefix/infix/suffix scope without discarding the entity owner."""
    pattern = (r"(?P<prefix>연결|별도|개별)?\s*요약\s*(?:(?P<infix>연결|별도|개별)(?:\s*\(\s*(?P<infix_alt>연결|별도|개별)\s*\))?|\(\s*(?P<paren_infix>연결|별도|개별)\s*\))?\s*"
               r"재무\s*(?:정보|제표|현황|상태표)"
               r"(?:\s*(?:[-:：]\s*|\(\s*)(?P<suffix>연결|별도|개별)\s*\)?)?")
    headings = [match for match in re.finditer(pattern, context)
                if not re.search(r"본\s*$", context[:match.start()])]
    boundaries = list(re.finditer(r"(?<!\d)[2345][.)]\s*(?:연결|별도|개별)?\s*재무제표", context))
    if boundaries and (not headings or boundaries[-1].start()>headings[-1].start()):
        return None
    subheadings = list(re.finditer(
        r"(?<![가-힣])[가나다라마바사아자차카타파하][.)]\s*(?P<basis>연결|별도|개별)?\s*재무제표(?!\s*(?:주석|에\s*대한|는))", context))
    if not headings and not allow_statement_subheading:
        return None
    choices = [(match, False) for match in headings] + [(match, True) for match in subheadings]
    if not choices:
        return None
    match, child = max(choices, key=lambda value: value[0].start())
    bases = ({match['basis']} if match['basis'] else set()) if child else {
        value for value in match.groupdict().values() if value is not None}
    # An unqualified summary may explicitly state its basis in the introduction.
    introduction = context[match.end():]
    declared = re.search(r"(?:는|은)\s*(연결|별도|개별)\s*재무제표", introduction)
    if declared:
        bases.add(declared[1])
    scopes = {"CFS" if value == "연결" else "OFS" for value in bases}
    subsidiary = bool(re.search(r"(?:종속|관계|피투자)(?:회사|기업)의\s*$", context[:match.start()])
                      or re.search(r"요약재무(?:정보|제표|현황)는.{0,25}(?:종속|관계|피투자)(?:회사|기업)인",
                                   clean(context[match.start():])))
    title_prefix = re.search(r"(?:K\s*-?\s*(?:IFRS|GAAP|GAPP)|한국채택국제회계기준)(?:\s*기준)?\s*$", context[:match.start()])
    standard_context = (title_prefix[0] if title_prefix else "") + context[match.start():]
    if child and headings and not re.search(r'KIFRS|KGAAP|KGAPP|한국채택국제회계기준',clean(standard_context)):
        standard_context = context[headings[-1].start():match.start()] + standard_context
    return {"scope": next(iter(scopes)) if len(scopes) == 1 else "OFS" if not scopes else None,
            "entity_scope": "subsidiary" if subsidiary else "issuer",
            "evidence": context, "heading": match[0], "start": match.start(),
            "standard_context": standard_context, "basis_explicit": bool(bases) or child}


def financial_records(frame):
    return json.loads(frame.to_json(orient='records',date_format='iso',double_precision=15))


def explicit_statement_year(header):
    """A named annual-year column, not a date range, interim, forecast or term alone."""
    compact=clean(header)
    match=re.fullmatch(r'((?:19|20)\d{2})(?:연도|년도|년)(?:제\d+기)?',compact)
    return int(match[1]) if match else None


def summary_column_standard(table, scope, context, table_id, primary, explicit):
    """Read the declared standard independently of compressed monetary rows."""
    if scope is None or (primary['consolidated_primary_present'] and not explicit):
        return {}
    result={}
    for row_id,row in enumerate(table):
        if not row or clean(row[0]) not in {'회계처리기준','회계기준'}:continue
        for column,value in enumerate(row[1:],start=1):
            if clean(value) not in {'KIFRS','한국채택국제회계기준'}:continue
            header=' '.join(r[column] for r in table[:row_id] if len(r)>column)
            year=explicit_statement_year(header)
            if year is None:continue
            result[(scope,year)]={'rule':'issuer_summary_explicit_column_standard_v1','scope':scope,
                'year':year,'summary_table_id':table_id,'column_id':column,'standard_row_id':row_id,
                'header_rows':table[:row_id],'raw_standard_cell':value,'statement_scope_evidence':context}
    return result


def statement_section_boundaries(context):
    return [match for match in re.finditer(
        r'(?<!\d)([2345])[.)]\s*(연\s*결\s*)?재\s*무\s*제\s*표(?!\s*작성)(\s*주\s*석)?',context)
        if (match[1],bool(match[2]),bool(match[3])) in {
            ('2',True,False),('3',True,True),('4',False,False),('5',False,True)}]


def annual_date_range(text):
    dates=re.findall(r'((?:19|20)\d{2})\s*(?:년|[./-])\s*(\d{1,2})\s*(?:월|[./-])\s*(\d{1,2})',text)
    if len(dates)!=2 or re.search(r'분기|반기|개월|잠정|가결산|설립\s*전|분할\s*전',text):return None
    try:
        start,end=(pd.Timestamp(*map(int,date)) for date in dates)
    except ValueError:return None
    if start.day!=1 or not end.is_month_end or end!=start+pd.DateOffset(years=1)-pd.Timedelta(days=1):return None
    return {'start':str(start.date()),'end':str(end.date()),'year':end.year,'text':text}


def noncalendar_fiscal_columns(parser):
    """Bind non-December year ends to explicit full-year issuer income periods."""
    ranges={};dated=[];context_ends={};scope=None;notes=False
    for table_id,(context,table) in enumerate(zip(parser.table_contexts,parser.tables)):
        declared=summary_scope_heading(context,allow_statement_subheading=scope is not None)
        if declared:
            scope=declared['scope'] if declared['entity_scope']=='issuer' else None;notes=False
        boundaries=statement_section_boundaries(context)
        if boundaries:
            last=boundaries[-1];scope='CFS' if last[2] else 'OFS';notes=bool(last[3])
            context=context[last.start():]
        if scope is None or notes:continue
        accounts={account_name(row[0]) for row in table if row}
        if len(accounts & {'revenue','operating_income','net_income','total_assets','total_liabilities','equity'})<2:continue
        terms=list(re.finditer(r'제\s*(\d+)\s*(?:\([^)]*\))?\s*기',context))
        for i,term in enumerate(terms):
            text=context[term.start():terms[i+1].start() if i+1<len(terms) else len(context)]
            period=annual_date_range(text)
            dates=re.findall(r'((?:19|20)\d{2})\s*(?:년|[./-])\s*(\d{1,2})\s*(?:월|[./-])\s*(\d{1,2})',text)
            if '재무상태표' in clean(context) and len(dates)==1:
                try:context_ends[(table_id,int(term[1]))]=str(pd.Timestamp(*map(int,dates[0])).date())
                except ValueError:pass
            if period and not period['end'].endswith('-12-31'):
                ranges.setdefault((scope,int(term[1])),[]).append({**period,'table_id':table_id,'fiscal_term':int(term[1])})
        first=next((i for i,row in enumerate(table) if row and account_name(row[0])),len(table))
        for column in range(1,max(map(len,table),default=0)):
            header=' '.join(row[column] for row in table[:first] if len(row)>column)
            period=annual_date_range(header)
            term_ids={int(t) for t in re.findall(r'제\s*(\d+)\s*(?:\([^)]*\))?\s*기',header)}
            if period and not period['end'].endswith('-12-31'):
                ranges.setdefault((scope,next(iter(term_ids)) if len(term_ids)==1 else None),[]).append(
                    {**period,'table_id':table_id,'column_id':column,
                     'fiscal_term':next(iter(term_ids)) if len(term_ids)==1 else None})
            dated.append((table_id,column,scope,header,term_ids))
    columns={}
    for table_id,column,scope,header,terms in dated:
        if re.search('분기|반기|개월|잠정|가결산|기초',header):continue
        candidates=[period for (basis,term),items in ranges.items() if basis==scope
                    and (term in terms or term is None or not terms) for period in items]
        own=annual_date_range(header)
        declared_ends={context_ends[(table_id,t)] for t in terms if (table_id,t) in context_ends}
        if declared_ends:candidates=[p for p in candidates if p['end'] in declared_ends]
        dates=re.findall(r'((?:19|20)\d{2})\s*(?:년|[./-])\s*(\d{1,2})\s*(?:월|[./-])\s*(\d{1,2})',header)
        if own:candidates=[p for p in candidates if (p['start'],p['end'])==(own['start'],own['end'])]
        elif dates:
            if len(dates)!=1:continue
            try:end=str(pd.Timestamp(*map(int,dates[0])).date())
            except ValueError:continue
            candidates=[p for p in candidates if p['end']==end]
        elif len(terms)!=1:continue
        else:
            explicit_years={int(y) for y in re.findall(r'(?:19|20)\d{2}',header)}
            if len(explicit_years)>1:continue
            if explicit_years:candidates=[p for p in candidates if p['year'] in explicit_years or
                p.get('fiscal_term') in terms and int(p['start'][:4]) in explicit_years]
        unique={(p['start'],p['end']) for p in candidates}
        if len(unique)==1:
            period=candidates[0]
            columns[(table_id,column)]={**period,'scope':scope,'terms':sorted(terms),
                'rule':'explicit_twelve_month_noncalendar_fiscal_period_v1'}
    return columns


def primary_annual_balances(parser, fiscal_columns=None):
    """Read scope/year/unit fingerprints from primary statements, never note tables."""
    fiscal_columns=noncalendar_fiscal_columns(parser) if fiscal_columns is None else fiscal_columns
    balances = {"total_assets", "total_liabilities", "equity"}
    scope, notes = None, False
    groups, consolidated = [], False
    for table_id, (context, table) in enumerate(zip(parser.table_contexts, parser.tables)):
        heading = clean(context)
        boundaries = statement_section_boundaries(context)
        if boundaries:
            scope = 'CFS' if boundaries[-1][2] else 'OFS'
            notes = bool(boundaries[-1][3])
        if scope is None or notes or '재무상태표' not in heading:
            continue
        positions = [(i,row,account_name(row[0]) or {
            '자산':'total_assets','부채':'total_liabilities','자본':'equity'}.get(clean(row[0])))
            for i,row in enumerate(table) if row]
        positions = [(i,row,name) for i,row,name in positions if name in balances]
        if len({name for _,_,name in positions})<3:
            continue
        consolidated |= scope == 'CFS'
        data_start=next((i for i,row in enumerate(table[:positions[0][0]])
            if clean(row[0]) not in {'과목','구분','기간','사업연도'} and any(
                re.fullmatch(r'(?:-?\d[\d,]*(?:\.\d+)?|\(\d[\d,]*(?:\.\d+)?\))',cell.strip())
                for cell in row[1:])),positions[0][0])
        headers=table[:data_start]
        units = {re.sub(r'\s+','',m) for m in re.findall(r"단위\s*[:：]\s*(백\s*만\s*원|천\s*원|원|KRW)", context+' '+' '.join(' '.join(row) for row in headers))}
        if len(units)!=1:
            continue
        scale = {'백만원':1_000_000,'천원':1_000,'원':1,'KRW':1}[units.pop()]
        terms=list(re.finditer(r'제\s*(\d+)\s*(?:\([^)]*\))?\s*(?:분기|반기|기)',context))
        periods={}
        for index,term in enumerate(terms):
            text=context[term.end():terms[index+1].start() if index+1<len(terms) else len(context)]
            dates=re.findall(r'((?:19|20)\d{2})\s*(?:년|[./-])\s*(\d{1,2})\s*(?:월|[./-])\s*(\d{1,2})',text)
            if len(dates)==1 and tuple(map(int,dates[0][1:]))==(12,31):
                periods.setdefault(int(term[1]),set()).add(int(dates[0][0]))
        for column in range(1,max(map(len,table))):
            header=' '.join(row[column] for row in headers if len(row)>column)
            years=set(re.findall(r'(?:19|20)\d{2}',header))
            term_ids={int(t) for t in re.findall(r'제\s*(\d+)\s*(?:\([^)]*\))?\s*기',header)}
            if re.search(r'분기|반기|기초|수정|조정|설립\s*전|분할\s*전|사업부|carve\s*out',header,re.I):
                continue
            if re.search(r'KGAAP|KGAPP|일반기업회계기준',clean(header)):
                continue
            fiscal=fiscal_columns.get((table_id,column))
            if fiscal and fiscal['scope']!=scope:fiscal=None
            if fiscal and fiscal['scope']==scope:
                year=fiscal['year']
            elif explicit_statement_year(header) is not None:
                year=explicit_statement_year(header)
            elif len(years)==1 and re.search(r'12\s*월',header):
                year=int(next(iter(years)))
            else:
                mapped={year for term in term_ids for year in periods.get(term,set())}
                if len(mapped)!=1 or years and {int(y) for y in years}!=mapped:
                    continue
                year=next(iter(mapped))
            facts={}
            for row_id,row,name in positions:
                if len(row)<=column:continue
                raw=row[column].strip().replace('−','-')
                if not re.fullmatch(r'(?:-?\d[\d,]*(?:\.\d+)?|\(\d[\d,]*(?:\.\d+)?\))',raw):continue
                value=float(raw.replace(',','').strip('()'))*(-1 if raw.startswith('(') else 1)
                facts.setdefault(name,[]).append({'account':name,'raw_amount':raw,'amount':value*scale,
                    'unit_multiplier':scale,'table_id':table_id,'row_id':row_id,'column_id':column,
                    'unit_context':context})
            if set(facts)==balances and all(len({r['amount'] for r in group})==1 for group in facts.values()):
                groups.append({'year':year,'fs_div':scope,'fiscal_terms':sorted(term_ids),
                               'facts':{name:group[0] for name,group in facts.items()}})
    return {'groups':groups,'consolidated_primary_present':consolidated}


def summary_column_basis(scope, explicit, positions, column, year, scale, primary):
    if explicit or not primary['consolidated_primary_present']:
        return scope, []
    balances={}
    for _,row,name in positions:
        if name not in {'total_assets','total_liabilities','equity'} or len(row)<=column:continue
        raw=row[column].strip().replace('−','-')
        if re.fullmatch(r'(?:-?\d[\d,]*(?:\.\d+)?|\(\d[\d,]*(?:\.\d+)?\))',raw):
            balances[name]=float(raw.replace(',','').strip('()'))*(-1 if raw.startswith('(') else 1)*scale
    if len(balances)!=3:
        return None, []
    matches=[group for group in primary['groups'] if group['year']==year and all(
        abs(balances[name]-fact['amount'])<=2*max(scale,fact['unit_multiplier'])
        for name,fact in group['facts'].items())]
    scopes={group['fs_div'] for group in matches}
    if len(scopes)!=1:
        return None, []
    return next(iter(scopes)), [group for group in matches]


def summary_unit_conflicts(primary, rows):
    """Detect identical primary balance cells printed under contradictory KRW units."""
    conflicts=[]
    for group in primary['groups']:
        year,scope=group['year'],group['fs_div'];matches={}
        for name,fact in group['facts'].items():
            value=fact['amount']/fact['unit_multiplier'];scale=fact['unit_multiplier']
            for candidate in rows:
                if (candidate['year']!=year or candidate['fs_div']!=scope or candidate['account_name_en']!=name
                        or candidate['unit_multiplier']==scale):continue
                raw=candidate['raw_amount'].replace(',','')
                number=float(raw.strip('()'))*(-1 if raw.startswith('(') else 1)
                if number==value and abs(candidate['amount']-fact['amount'])>2*max(scale,candidate['unit_multiplier']):
                    matches.setdefault(candidate['table_id'],[]).append({
                        'account':name,'raw_amount':fact['raw_amount'],'summary_unit_multiplier':candidate['unit_multiplier'],
                        'primary_unit_multiplier':scale,'primary_table_id':fact['table_id'],'primary_row_id':fact['row_id'],
                        'primary_column_id':fact['column_id'],'primary_unit_context':fact['unit_context']})
        for summary_id,evidence in matches.items():
            if len({r['account'] for r in evidence})>=2:
                conflicts.append({'summary_table_id':summary_id,'year':year,'fs_div':scope,'evidence':evidence})
    return conflicts


def income_period(table, first_account, row_id, column, year):
    """Explicit income ranges override annual balance-sheet headings, including wrapped rows."""
    period = None
    for index in range(first_account, row_id):
        cell = table[index][column] if len(table[index]) > column else ""
        if not re.search(r"(?:19|20)\d{2}.*[~～∼]", cell):
            continue
        period = " ".join(table[k][column] for k in range(index, min(index + 2, row_id))
                          if len(table[k]) > column)
    if period is None:
        return True, None
    dates = re.findall(r"((?:19|20)\d{2})\s*(?:[.\-/]|년)\s*(\d{1,2})\s*(?:[.\-/]|월)\s*(\d{1,2})", period)
    if len(dates) == 2:
        annual=annual_date_range(period)
        return (annual is not None and annual['year']==year), period
    months = re.search(r"(\d{4})년\s*(\d{1,2})월\s*[~～]\s*(?:(\d{4})년\s*)?(\d{1,2})월", period)
    if months:
        return (int(months[1]) == year and int(months[2]) == 1
                and (months[3] is None or int(months[3]) == year) and int(months[4]) == 12), period
    return False, period


def ifrs_preparation_statements(text, default_scope="OFS", table_note=False, fiscal_ends=()):
    """Bind affirmative preparation to its stated scope and fiscal period."""
    standard = r"한국채택국제회계기준(?:\([^)]*\))?"
    patterns = [
        r"(?<![가-힣])(?P<issuer>당사)의\s*요약재무정보는[^.。]{0,100}?"+standard+r"에\s*따라\s*작성\s*(?:되었|하였)",
        r"(?<![가-힣])(?P<basis>연결|별도|개별)?\s*재무제표\s*는[^.。]{0,180}?" + standard
        + r"\s*(?:작성기준)?\s*에\s*따라\s*작성\s*(?:되었|됐|하였|하고)",
        r"당사는\s*" + standard + r"을\s*적용하여\s*(?P<basis>연결|별도|개별)?\s*재무제표를\s*작성하고\s*있",
        r"당사는\s*" + standard + r"에\s*따라\s*(?P<basis>연결|별도|개별)?\s*재무제표를\s*작성하였",
        r"(?<![가-힣])(?P<issuer>연결실체|연결회사|당사|회사)는[^.。]{0,180}?" + standard
        + r"에\s*따라\s*(?P<basis>연결|별도|개별)?\s*재무제표를\s*작성하였",
        standard + r"에\s*따라\s*작성된\s*(?P<annual_year>(?:19|20)\d{2})년\s*12월\s*31일(?:자)?로\s*종료(?:하는|되는)\s*회계연도에\s*대한\s*(?:연차\s*)?(?P<basis>연결|별도|개별)?\s*재무제표",
    ]
    if table_note:
        patterns.append(standard + r"\s*(?:작성기준)?\s*에\s*따라\s*작성\s*(?:되었|됐|하였|하고)?")
        patterns.append(r"(?P<since_year>(?:19|20)\d{2})년\s*사업연도부터\s*" + standard
            + r"(?:을|를)?\s*적용한\s*(?P<basis>연결|별도|개별)?\s*감사보고서를\s*발행하였")
    result = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            prefix = text[max(0, match.start()-65):match.start()]
            issuer=match.groupdict().get('issuer')
            if issuer:
                other_entity=r'종속회사|종속기업|관계회사|피투자회사|비교회사'
                if (re.search(r'(?:'+other_entity+r')(?:인|의)?\s*$',prefix)
                        or re.search(other_entity,match[0])):
                    continue
            since_year=match.groupdict().get('since_year')
            if since_year and re.search(r'종속회사|관계회사|피투자회사|비교회사',prefix):
                continue
            annual_reference=match.groupdict().get('annual_year') is not None
            if annual_reference and re.search(r'(?:종속회사|종속기업|관계회사|피투자회사|비교회사)(?:인|의)?\s*$',prefix):
                continue
            if not annual_reference and re.search(r"(?:분기|반기|중간)(?:(?:연결|별도|개별)?재무제표는)?$",clean(prefix)):
                continue
            if re.search(r"차기|예정|채택할|적용할",prefix) or re.match(r"\s*(?:할|될|예정)",text[match.end():]):
                continue
            if re.search(r"분기|반기|중간|예정|적용할|차기|작성할", match[0]):
                continue
            dates = [tuple(map(int, date)) for date in re.findall(
                r"((?:19|20)\d{2})\s*(?:년|[./-])\s*(\d{1,2})\s*(?:월|[./-])\s*(\d{1,2})", match[0] if annual_reference else prefix + match[0])]
            if any(date[1:] != (12, 31) and str(pd.Timestamp(*date).date()) not in fiscal_ends for date in dates):
                continue
            basis = match.groupdict().get('basis')
            if basis:
                scope = "CFS" if basis == "연결" else "OFS"
            elif issuer in {'연결실체','연결회사'} or re.search(r"연결(?:(?:실체|회사)(?:의)?)?\s*$", prefix):
                scope = "CFS"
            else:
                scope = default_scope
            terms = set() if annual_reference else {int(term) for term in re.findall(r"제\s*(\d+)\s*기", prefix + match[0])}
            result.append({'scope': scope, 'text': prefix + match[0], 'terms': terms,
                           'years': {date[0] for date in dates},
                           **({'implicit_scope_annual_reference':True} if annual_reference and basis is None else {}),
                           **({'since_year':int(since_year)} if since_year else {})})
    return result


def preparation_for_year(statements, scope, year, header, term_years):
    terms = {int(term) for term in re.findall(r"제\s*(\d+)\s*(?:\([^)]*\))?\s*기", header)}
    for statement in statements:
        if statement['scope'] != scope or statement['years'] and year not in statement['years']:
            continue
        if statement.get('since_year') is not None and year < statement['since_year']:
            continue
        if statement['terms']:
            years = {int(value) for term in statement['terms'] for value in term_years.get(str(term),set())}
            if not (terms & statement['terms']) and year not in years:
                continue
        return statement['text']
    return None


def interim_preparation_declarations(text, scope, current):
    compact=re.sub(r'\s+','',text)
    bridge=r'(?:(?:[^.。!?])|(?<=\d)\.(?=\d)){0,160}?'
    interim_standard=r'(?:기업회계기준서|한국채택국제회계기준)제1034호(?:[\'"‘’「」(]*중간재무보고[\'"‘’「」)]*)?'
    standard=(r'(?:한국채택국제회계기준(?:\([^)]*\))?(?:'+interim_standard+r')?'
              r'|'+interim_standard+r')')
    pattern=(r'(?P<before>연결|별도|개별)?(?:요약)?(?:분기|반기|중간)(?:요약)?(?P<after>연결|별도|개별)?'
             r'재무제표(?:는|은)'+bridge+standard+r'에따라작성(?:되었|되는|하였)')
    applied=(r'(?P<before>연결|별도|개별)?재무제표(?:는|은)'+bridge+interim_standard
             +r'를적용하여작성하는요약중간(?P<after>연결|별도|개별)?재무제표')
    def matches_scope(bases):
        return not bases or {('CFS' if b=='연결' else 'OFS') for b in bases}=={scope}
    declarations=[]
    for match in sorted([*re.finditer(pattern,compact),*re.finditer(applied,compact)],key=lambda m:m.start()):
        if not matches_scope([b for b in [match['before'],match['after']] if b]):continue
        prefix=compact[max(0,match.start()-40):match.start()]
        group=re.search(r'(?:지배기업|지배회사)(?:및|와|과)(?:그)?종속(?:기업|회사)'
                        r'\(이하[\'"‘’“”]?(?:연결기업|연결회사|연결실체)[\'"‘’“”]?(?:이라|라)함\)의$',prefix)
        if scope=='CFS' and group:prefix=prefix[:group.start()]
        if re.search(r'종속회사|종속기업|관계회사|비교회사|예정|적용하지',prefix+match[0]):continue
        years={int(y) for y in re.findall(r'((?:19|20)\d{2})(?:년|[./-])\d{1,2}(?:월|[./-])\d{1,2}',match[0])}
        if years and years!={current}:continue
        declarations.append(match)
    return compact,declarations


def interim_annual_comparative_basis(text, scope, interim_years, published, fiscal_periods=(), interim_ends=(), interim_terms=()):
    """Bind a same-scope IFRS interim declaration to its named prior annual report."""
    if not interim_years:return {}
    current=max(interim_years)
    if current>pd.Timestamp(published).year:return {}
    compact,declarations=interim_preparation_declarations(text,scope,current)
    if not declarations:return {}
    def matches_scope(bases):
        return not bases or {('CFS' if b=='연결' else 'OFS') for b in bases}=={scope}
    references=[]
    reference=(r'(?P<year>(?:19|20)\d{2})(?:년|[./-])12(?:월|[./-])31일?(?:자)?로?종료(?:하는|되는)'
               r'회계(?:연도(?:에대한|의)?(?:연차)?|기간(?:에대한|의)?연차)(?P<basis>연결|별도|개별)?재무제표')
    for match in re.finditer(reference,compact):
        if int(match['year'])!=current-1 or not matches_scope([match['basis']] if match['basis'] else []):continue
        if re.search(r'종속회사|종속기업|관계회사|비교회사',compact[max(0,match.start()-80):match.start()]):continue
        references.append(match[0])
    # Resolve only an explicitly annual prior-year reference adjacent to this
    # declaration. Generic "prior statements" can refer to an interim period.
    relative=r'전기(?:말)?연차(?P<basis>연결|별도|개별)?재무제표'
    relative_references=[]
    for declaration in declarations:
        for match in re.finditer(relative,compact[declaration.end():declaration.end()+240]):
            if matches_scope([match['basis']] if match['basis'] else []):
                if fiscal_periods:relative_references.append(match[0])
                else:references.append(match[0]+' [resolved_from_interim_year='+str(current)+']')
    result={}
    if fiscal_periods and interim_ends:
        latest=max(pd.Timestamp(day) for day in interim_ends)
        pattern=(r'(?P<year>(?:19|20)\d{2})(?:년|[./-])(?P<month>\d{1,2})(?:월|[./-])'
                 r'(?P<day>\d{1,2})일?(?:자)?로?종료(?:하는|되는)회계(?:연도|기간)'
                 r'(?:에대한|의)?연차(?P<basis>연결|별도|개별)?재무제표')
        for match in re.finditer(pattern,compact):
            if not matches_scope([match['basis']] if match['basis'] else []):continue
            prefix=compact[max(0,match.start()-80):match.start()]
            if re.search('종속회사|종속기업|관계회사|비교회사|예정|작성할',prefix):continue
            try:end=pd.Timestamp(int(match['year']),int(match['month']),int(match['day']))
            except ValueError:continue
            periods=[p for p in fiscal_periods if p['end']==str(end.date())]
            if not periods or not end<latest<end+pd.DateOffset(years=1):continue
            result[(scope,end.year)]=json.dumps({'rule':'same_scope_ifrs_interim_explicit_fiscal_annual_v1',
                'scope':scope,'annual_year':end.year,'annual_period':periods[0],
                'interim_period_end':str(latest.date()),'ifrs_interim_declarations':[m[0] for m in declarations],
                'explicit_annual_reference':match[0]},ensure_ascii=False)
        prior_term=max(interim_terms)-1 if interim_terms else None
        prior=[p for p in fiscal_periods if prior_term in p.get('terms',[])
               and pd.Timestamp(p['end'])<latest<pd.Timestamp(p['end'])+pd.DateOffset(years=1)]
        if relative_references and len({p['end'] for p in prior})==1:
            period=prior[0]
            result[(scope,period['year'])]=json.dumps({'rule':'same_scope_ifrs_interim_explicit_fiscal_annual_v1',
                'scope':scope,'annual_year':period['year'],'annual_period':period,
                'interim_period_end':str(latest.date()),'interim_fiscal_term':prior_term+1,
                'annual_fiscal_term':prior_term,'ifrs_interim_declarations':[m[0] for m in declarations],
                'explicit_annual_reference':relative_references},ensure_ascii=False)
    if references:result[(scope,current-1)]=json.dumps({
        'rule':'same_scope_ifrs_interim_prior_annual_comparative_v1',
        'scope':scope,'interim_statement_year':current,'annual_year':current-1,
        'ifrs_interim_declarations':[m[0] for m in declarations],'explicit_annual_references':references,
        'source_kind':'annual_comparative_column_not_interim_amount_or_audit_opinion',
        'standard_reference':'https://www.ifrs.org/issued-standards/list-of-standards/ias-34-interim-financial-reporting/',
        'korean_standard_reference':'https://kasb.or.kr/front/board/View016001.do?seq=35659'},ensure_ascii=False)
    return result


def interim_comparative_balance_basis(text, scope, pair):
    """IAS 34.20(a) covers the prior closing balance, not full-year income."""
    if pair is None:return None
    current=pd.Timestamp(pair['interim_period_end'])
    _,declarations=interim_preparation_declarations(text,scope,current.year)
    verified=[]
    for declaration in declarations:
        bases=[b for b in [declaration['before'],declaration['after']] if b]
        if scope=='CFS' and '연결' not in bases:continue
        try:
            dates={str(pd.Timestamp(*map(int,d)).date()) for d in re.findall(
                r'((?:19|20)\d{2})(?:년|[./-])(\d{1,2})(?:월|[./-])(\d{1,2})',declaration[0])}
        except ValueError:continue
        if dates and dates!={pair['interim_period_end']}:continue
        verified.append(declaration[0])
    if not verified:return None
    return json.dumps({'rule':('same_document_ifrs_interim_prior_closing_balance_v2'
                              if pair.get('annual_period_origins') else 'same_table_ifrs_interim_prior_closing_balance_v1'),
        'scope':scope,**pair,'ifrs_interim_declarations':verified,
        'applies_to_accounts':['total_assets','total_liabilities','equity'],
        'source_kind':'prior_fiscal_closing_comparative_not_interim_or_full_year_income',
        'standard_reference':'https://www.ifrs.org/content/dam/ifrs/publications/amendments/english/2017/ifrs-practice-statement-2-making-materiality-judgements.pdf',
        'standard_paragraph':'IAS 34.20(a)','korean_standard_reference':'https://www.kasb.or.kr/front/board/ingAccountingList.do'},ensure_ascii=False)


def balance_caption_dates(text, *, annual=False):
    dates=re.findall(r'((?:19|20)\d{2})\s*(?:년|[./-])\s*(\d{1,2})\s*(?:월|[./-])\s*(\d{1,2})',text)
    try:days=[pd.Timestamp(*map(int,d)) for d in dates]
    except ValueError:return None
    if len(days)==1:return {'end':str(days[0].date())}
    if len(days)!=2:return None
    start,end=days
    if annual:
        if annual_date_range(text) is None:return None
    elif not (start.day==1 and end.is_month_end and start<end<start+pd.DateOffset(years=1)-pd.Timedelta(days=1)):
        return None
    return {'start':str(start.date()),'end':str(end.date())}


def summary_balance_period_columns(parser):
    """Keep dated issuer-summary columns for a three-balance primary-table join."""
    balances={'total_assets','total_liabilities','equity'};result=[];scope=None;explicit=False;non_ifrs=False
    for table_id,(context,table) in enumerate(zip(parser.table_contexts,parser.tables)):
        declared=summary_scope_heading(context)
        boundaries=statement_section_boundaries(context)
        if boundaries and (declared is None or boundaries[-1].start()>declared['start']):scope=None
        elif declared:
            scope=declared['scope'] if declared['entity_scope']=='issuer' else None
            explicit=declared['basis_explicit']
            standard=re.search('KIFRS|KGAAP|KGAPP|일반기업회계기준|한국채택국제회계기준',clean(declared['standard_context']))
            non_ifrs=bool(standard and standard[0] in {'KGAAP','KGAPP','일반기업회계기준'})
        if standalone_non_ifrs_caption(context):non_ifrs=True
        if scope is None or non_ifrs:continue
        first=next((i for i,row in enumerate(table) if row and clean(row[0]) not in {'구분','과목','기간','사업연도'}
                    and any(re.fullmatch(r'(?:-?\d[\d,]*(?:\.\d+)?|\(\d[\d,]*(?:\.\d+)?\))',cell.strip())
                            for cell in row[1:])),len(table))
        headers=table[:first]
        units={clean(unit) for unit in re.findall(r'단위\s*[:：]\s*(백\s*만\s*원|천\s*원|원|KRW)',
                                                context+' '+' '.join(' '.join(r) for r in headers))}
        if len(units)!=1:continue
        scale={'백만원':1_000_000,'천원':1_000,'원':1,'KRW':1}[units.pop()]
        for column in range(1,max(map(len,table),default=0)):
            header=' '.join(row[column] for row in headers if len(row)>column)
            if re.search('KGAAP|KGAPP|일반기업회계기준|기초|예상|예정|추정|잠정|가결산',clean(header)):continue
            terms={int(term) for term in re.findall(r'제\s*(\d+)\s*(?:\([^)]*\))?\s*기',header)}
            years={int(year) for year in re.findall(r'((?:19|20)\d{2})\s*년',header)}
            if len(terms)!=1:continue
            period=balance_caption_dates(header,annual=True) or balance_caption_dates(header)
            if period is None:
                month_ends=re.findall(r'((?:19|20)\d{2})\s*년\s*(\d{1,2})\s*월\s*말',header)
                if len(month_ends)==1:
                    try:period={'end':str((pd.Timestamp(int(month_ends[0][0]),int(month_ends[0][1]),1)+pd.offsets.MonthEnd()).date())}
                    except ValueError:continue
            interim=bool(re.search('분기|반기|당분|당반',header))
            if period is None and not (interim and len(years)==1):continue
            year=pd.Timestamp(period['end']).year if period else next(iter(years))
            facts={}
            for row_id,row in enumerate(table):
                if not row or len(row)<=column:continue
                name=account_name(row[0]);raw=row[column].strip().replace('−','-')
                if name not in balances or not re.fullmatch(r'(?:-?\d[\d,]*(?:\.\d+)?|\(\d[\d,]*(?:\.\d+)?\))',raw):continue
                number=float(raw.replace(',','').strip('()'))*(-1 if raw.startswith('(') else 1)
                facts.setdefault(name,[]).append({'account_name_en':name,'amount':number*scale,'raw_amount':raw,
                    'unit_multiplier':scale,'table_id':table_id,'row_id':row_id,'column_id':column})
            if set(facts)!=balances or any(len({r['amount'] for r in group})!=1 for group in facts.values()):continue
            result.append({'scope':scope if explicit else None,'entity_scope':'issuer','table_id':table_id,
                'column_id':column,'header':header,'fiscal_term':next(iter(terms)),'year':year,
                'end':period['end'] if period else None,'interim_label':interim,
                'facts':{name:group[0] for name,group in facts.items()}})
    return result


def matching_summary_periods(periods,scope,term,positions,column,scale):
    facts={}
    for _,row,name,_ in positions:
        if len(row)<=column:continue
        raw=row[column].strip().replace('−','-')
        if re.fullmatch(r'(?:-?\d[\d,]*(?:\.\d+)?|\(\d[\d,]*(?:\.\d+)?\))',raw):
            facts.setdefault(name,set()).add(float(raw.replace(',','').strip('()'))*(-1 if raw.startswith('(') else 1)*scale)
    if set(facts)!={'total_assets','total_liabilities','equity'} or any(len(v)!=1 for v in facts.values()):return []
    return [p for p in periods if p['scope'] in {None,scope} and p['fiscal_term']==term and all(
        abs(next(iter(facts[name]))-fact['amount'])<=2*max(scale,fact['unit_multiplier']) for name,fact in p['facts'].items())]


def summary_linked_interim_balance_pair(periods,scope,table_id,headers,positions,column,scale,term,year,text,published):
    annual=[p for p in matching_summary_periods(periods,scope,term,positions,column,scale)
            if not p['interim_label'] and p['end']==f'{year}-12-31']
    if not annual:return None
    _,declarations=interim_preparation_declarations(text,scope,year+1)
    dates=set()
    for declaration in declarations:
        if scope=='CFS' and '연결' not in {declaration['before'],declaration['after']}:continue
        for date in re.findall(r'((?:19|20)\d{2})(?:년|[./-])(\d{1,2})(?:월|[./-])(\d{1,2})',declaration[0]):
            try:dates.add(str(pd.Timestamp(*map(int,date)).date()))
            except ValueError:continue
    comparisons={}
    for current in range(1,max((len(r) for r in headers),default=0)):
        header=' '.join(r[current] for r in headers if len(r)>current)
        terms={int(t) for t in re.findall(r'제\s*(\d+)\s*(?:\([^)]*\))?\s*(?:[1-4]\s*분기|분기|반기|기)',header)}
        if terms!={term+1} or not re.search('분기|반기|당분|당반',header):continue
        if re.search('기초|수정|조정|KGAAP|KGAPP|일반기업회계기준',clean(header)):continue
        for origin in matching_summary_periods(periods,scope,term+1,positions,current,scale):
            end=origin['end']
            if end is None:
                candidates=[d for d in dates if int(d[:4])==origin['year']]
                if len(candidates)!=1:continue
                end=candidates[0]
            if not (pd.Timestamp(year,12,31)<pd.Timestamp(end)<pd.Timestamp(year+1,12,31)
                    and pd.Timestamp(end)<pd.Timestamp(published).tz_localize(None)):continue
            label=header+' '+origin['header']
            quarters={int(q) for q in re.findall(r'([1-4])\s*분기',label)}
            expected_months={3*q for q in quarters} | ({6} if '반기' in label else set())
            if expected_months and (len(expected_months)!=1 or pd.Timestamp(end).month not in expected_months
                                    or not pd.Timestamp(end).is_month_end):continue
            own=balance_caption_dates(header)
            if own and own['end']!=end:continue
            comparisons.setdefault((current,end),[]).append(origin)
    if len(comparisons)!=1:return None
    (current,end),origins=next(iter(comparisons.items()))
    return {'table_id':table_id,'annual_year':year,'annual_period_end':f'{year}-12-31',
        'annual_term':term,'annual_column':column,'annual_period_text':annual[0]['header'],
        'annual_header':' '.join(r[column] for r in headers if len(r)>column),
        'annual_period_origins':annual,'interim_period_end':end,'interim_term':term+1,
        'interim_column':current,'interim_period_text':origins[0]['header'],
        'interim_header':' '.join(r[current] for r in headers if len(r)>current),
        'interim_period_origins':origins}


def detailed_annual_accounts(parser, corp_code, receipt, published, source_url, digest, summary_basis=None, primary_terms=None,
                            basis_evidence=None, fiscal_columns=None):
    """Read named primary statements, never their arbitrary supporting note tables."""
    fiscal_columns=noncalendar_fiscal_columns(parser) if fiscal_columns is None else fiscal_columns
    summary_periods=summary_balance_period_columns(parser)
    regions = {"CFS": [], "OFS": []}
    candidates = []
    interim_years={'CFS':set(),'OFS':set()}
    interim_ends={'CFS':set(),'OFS':set()}
    interim_terms={'CFS':set(),'OFS':set()}
    rejected = Counter()
    scope, in_notes = None, False
    losses = {"영업손실": "operating_income", "당기순손실": "net_income",
              "연결당기순손실": "net_income"}
    def account(label):
        label = re.sub(r"^\s*(?:[IVX]+|\d+)[.)]\s*", "", label)
        name = clean(label)
        return account_name(label) or losses.get(name), name in losses
    entries = [*zip(parser.table_contexts, parser.tables), ("".join(parser._outside_text), [])]
    for table_id, (context, table) in enumerate(entries):
        boundaries = statement_section_boundaries(context)
        start=0
        for boundary in boundaries:
            if scope is not None:regions[scope].append(context[start:boundary.start()])
            scope="CFS" if boundary[2] else "OFS"
            in_notes=bool(boundary[3]);start=boundary.start()
        current_context=context[start:]
        if scope is None:
            continue
        regions[scope].append(current_context + " " + " ".join(" ".join(row) for row in table))
        if in_notes:
            continue
        titles = list(re.finditer(r"(연결)?(포괄손익계산서|손익계산서|재무상태표)", clean(current_context)))
        if not titles or (titles[-1][1] and scope != "CFS"):
            continue
        kind = "balance" if titles[-1][2] == "재무상태표" else "income"
        positions = [(i, row, *account(row[0])) for i, row in enumerate(table) if row]
        positions = [item for item in positions if item[2] and
                     ((item[2] in {"total_assets", "total_liabilities", "equity"}) == (kind == "balance"))]
        if len(positions) < 2:
            continue
        data_start=next((i for i,row in enumerate(table[:positions[0][0]])
            if clean(row[0]) not in {'과목','구분','기간','사업연도'} and any(
                re.fullmatch(r'(?:-?\d[\d,]*(?:\.\d+)?|\(\d[\d,]*(?:\.\d+)?\))',cell.strip())
                for cell in row[1:])),positions[0][0])
        headers = table[:data_start]
        heading = current_context + " " + " ".join(" ".join(row) for row in headers)
        units = re.findall(r"단위\s*[:：]\s*(백\s*만\s*원|천\s*원|원|KRW)", heading)
        units = {re.sub(r"\s+", "", value) for value in units}
        if len(units) != 1:
            rejected["krw_unit_unverified"] += 1
            continue
        scale = {"백만원": 1_000_000, "천원": 1_000, "원": 1, "KRW": 1}[units.pop()]
        column_headers={column:' '.join(row[column] for row in headers if len(row)>column)
                        for column in range(1,max(map(len,table)))}
        term_headers={}
        for header in column_headers.values():
            for term in re.findall(r'제\s*(\d+)\s*(?:\([^)]*\))?\s*(?:[1-4]\s*분기|분기|반기|기)',header):
                term_headers.setdefault(int(term),[]).append(header)
        terms = list(re.finditer(r"제\s*(\d+)\s*(?:\([^)]*\))?\s*(?:[1-4]\s*분기|분기|반기|기)", current_context))
        periods = {};table_interims=[]
        for index, term in enumerate(terms):
            text = current_context[term.start():terms[index+1].start() if index+1 < len(terms) else len(current_context)]
            stated_annual=annual_date_range(text) is not None
            point=balance_caption_dates(text,annual=True) if kind=='balance' else None
            stated_annual=stated_annual or bool(point and point['end'].endswith('-12-31'))
            interim_label=(re.search(r'분기|반기|당분|당반|[36]\s*개월',text)
                or not stated_annual and re.search(r'분기|반기|당분|당반|[36]\s*개월',' '.join(term_headers.get(int(term[1]),[]))))
            if interim_label or re.search(r"설립\s*전|분할\s*전|사업부|carve\s*out", text,re.I):
                if interim_label:
                    interim_terms[scope].add(int(term[1]))
                    interim_years[scope].update(int(y) for y in re.findall(
                        r'((?:19|20)\d{2})\s*(?:년|[./-])\s*\d{1,2}\s*(?:월|[./-])\s*\d{1,2}',text))
                    dates=re.findall(r'((?:19|20)\d{2})\s*(?:년|[./-])\s*(\d{1,2})\s*(?:월|[./-])\s*(\d{1,2})',text)
                    if dates:
                        try:interim_ends[scope].add(str(pd.Timestamp(*map(int,dates[-1])).date()))
                        except ValueError:pass
                    if kind=='balance':
                        period=balance_caption_dates(text)
                        if period and pd.Timestamp(period['end'])<pd.Timestamp(published).tz_localize(None):
                            table_interims.append({'term':int(term[1]),**period,'text':text})
                continue
            dates = [tuple(map(int, day)) for day in re.findall(
                r"((?:19|20)\d{2})\s*(?:년|[./-])\s*(\d{1,2})\s*(?:월|[./-])\s*(\d{1,2})", text)]
            if kind == "balance":
                period=balance_caption_dates(text,annual=True)
                valid=period is not None and period['end'].endswith('-12-31')
            else:
                valid = len(dates) == 2 and dates == [(dates[0][0], 1, 1), (dates[0][0], 12, 31)]
            if valid:
                periods.setdefault(int(term[1]), {})[dates[0][0]] = text
        for column in range(1, max(map(len, table))):
            header = " ".join(row[column] for row in headers if len(row) > column)
            if re.search(r'KGAAP|KGAPP|일반기업회계기준',clean(header)):
                rejected['explicit_gaap_not_ifrs']+=1
                continue
            if re.search(r'회계정책변경전',clean(header)):
                rejected['superseded_accounting_policy_column']+=1
                continue
            if re.search(r"분기|반기|당분|당반|기초|잠정|추정|예상|예정|계획|차기|미적용|적용하지|[36]\s*개월|누\s*적|설립\s*전|분할\s*전|사업부|carve\s*out", header,re.I):
                continue
            numbers = {int(m) for m in re.findall(r"제\s*(\d+)\s*(?:\([^)]*\))?\s*기", header)}
            if len(numbers) != 1:
                continue
            fiscal_term=numbers.pop()
            years = dict(periods.get(fiscal_term, {}))
            column_dates=[tuple(map(int,date)) for date in re.findall(
                r'((?:19|20)\d{2})\s*(?:년|[./-])\s*(\d{1,2})\s*(?:월|[./-])\s*(\d{1,2})',header)]
            column_period=balance_caption_dates(header,annual=True) if kind=='balance' else None
            column_annual=(column_period is not None and column_period['end'].endswith('-12-31') if kind=='balance'
                else len(column_dates)==2 and column_dates==[(column_dates[0][0],1,1),(column_dates[0][0],12,31)])
            if column_annual:years.setdefault(column_dates[0][0],header)
            period_origins=[]
            if not years and kind=='balance':
                period_origins=[p for p in matching_summary_periods(summary_periods,scope,fiscal_term,positions,column,scale)
                                if p['end'] and p['end'].endswith('-12-31') and not p['interim_label']]
                if len({p['end'] for p in period_origins})==1:
                    years={period_origins[0]['year']:period_origins[0]['header']}
            label_year=explicit_statement_year(header)
            if (not years and label_year is not None and (scope,label_year) in (summary_basis or {})
                    and not re.search('분기|반기|중간|개월',current_context)):
                years={label_year:header}
            fiscal=fiscal_columns.get((table_id,column))
            if fiscal and fiscal['scope']!=scope:fiscal=None
            if fiscal and fiscal['scope']==scope:years={fiscal['year']:fiscal['text']}
            if len(years) != 1:
                rejected["annual_period_unresolved_or_not_annual"] += 1
                continue
            year, period = next(iter(years.items()))
            expected_end=fiscal['end'] if fiscal else f'{year}-12-31'
            stated_period=balance_caption_dates(header,annual=True) if column_dates else None
            if kind=='income' and len(column_dates)==1:
                short_range=re.search(
                    r'((?:19|20)\d{2})\s*(?:년|[./-])\s*(\d{1,2})\s*(?:월|[./-])\s*(\d{1,2})'
                    r'\s*일?\s*[~～∼]\s*(\d{1,2})\s*(?:월|[./-])\s*(\d{1,2})\s*일?\s*(?:까지)?\s*$',header)
                if short_range:
                    start='-'.join(short_range.group(i) for i in (1,2,3))
                    end='-'.join(short_range.group(i) for i in (1,4,5))
                    stated_period=annual_date_range(start+'~'+end)
            if column_dates and (stated_period is None or stated_period['end']!=expected_end):
                rejected['column_period_conflicts_with_annual_statement']+=1
                continue
            known_years=(primary_terms or {}).get((scope,fiscal_term),set())
            if known_years and known_years!={year}:
                rejected['annual_period_fiscal_term_conflict']+=1
                continue
            if pd.Timestamp(fiscal['end'] if fiscal else f"{year}-12-31") >= pd.Timestamp(published).tz_localize(None):
                continue
            pair=None
            if kind=='balance':
                annual_end=pd.Timestamp(fiscal['end'] if fiscal else f'{year}-12-31')
                comparisons=[]
                for interim in table_interims:
                    if (interim['term']!=fiscal_term+1
                            or not annual_end<pd.Timestamp(interim['end'])<annual_end+pd.DateOffset(years=1)):continue
                    for current_column in range(1,max(map(len,table))):
                        current_header=' '.join(r[current_column] for r in headers if len(r)>current_column)
                        terms={int(t) for t in re.findall(r'제\s*(\d+)\s*(?:\([^)]*\))?\s*(?:[1-4]\s*분기|분기|반기|기)',current_header)}
                        if terms!={interim['term']} or re.search('기초|수정|조정|KGAAP|KGAPP|일반기업회계기준',clean(current_header)):continue
                        current_dates=re.findall(r'((?:19|20)\d{2})\s*(?:년|[./-])\s*(\d{1,2})\s*(?:월|[./-])\s*(\d{1,2})',current_header)
                        current_period=balance_caption_dates(current_header)
                        if current_dates and (current_period is None or current_period['end']!=interim['end']
                            or 'start' in current_period and 'start' in interim and current_period['start']!=interim['start']):continue
                        names={name for _,r,name,_ in positions if len(r)>current_column
                               and re.fullmatch(r'(?:-?\d[\d,]*(?:\.\d+)?|\(\d[\d,]*(?:\.\d+)?\))',r[current_column].strip())}
                        if names=={'total_assets','total_liabilities','equity'}:
                            comparisons.append({'interim_period_end':interim['end'],'interim_term':interim['term'],
                                'interim_column':current_column,'interim_period_text':interim['text'],
                                'interim_header':current_header})
                header_agrees=(not column_dates or column_period is not None and column_period['end']==str(annual_end.date()))
                if len(comparisons)==1 and header_agrees:
                    pair={'table_id':table_id,'annual_year':year,'annual_period_end':str(annual_end.date()),
                          'annual_term':fiscal_term,'annual_column':column,'annual_period_text':period,
                          'annual_header':header,**comparisons[0]}
                    if '회계정책변경후' in clean(header):
                        pair['annual_policy_selection']='explicit_after_accounting_policy_change'
                        pair['superseded_headers']=[{'column_id':j,'header':' '.join(r[j] for r in headers if len(r)>j)}
                            for j in range(1,max(map(len,table))) if '회계정책변경전' in clean(
                                ' '.join(r[j] for r in headers if len(r)>j))]
            candidates.append((scope, year, table_id, positions, column, scale, header, period,pair,period_origins,headers))
    # A dated, three-balance summary join can establish the current period;
    # only a separate explicit annual reference may then establish annual income.
    for index,candidate in enumerate(candidates):
        scope,year,table_id,positions,column,scale,header,period,pair,origins,headers=candidate
        if pair is not None or {p[2] for p in positions}!={'total_assets','total_liabilities','equity'}:continue
        terms={int(t) for t in re.findall(r'제\s*(\d+)\s*(?:\([^)]*\))?\s*기',header)}
        if len(terms)!=1:continue
        pair=summary_linked_interim_balance_pair(summary_periods,scope,table_id,headers,positions,column,scale,
            next(iter(terms)),year,' '.join(regions[scope]),published)
        if pair is None or interim_comparative_balance_basis(' '.join(regions[scope]),scope,pair) is None:continue
        candidates[index]=(*candidate[:8],pair,*candidate[9:])
        interim_years[scope].add(pd.Timestamp(pair['interim_period_end']).year)
        interim_ends[scope].add(pair['interim_period_end']);interim_terms[scope].add(pair['interim_term'])
    basis = {}
    for scope, blocks in regions.items():
        text = " ".join(blocks)
        scoped_fiscal=[p for p in fiscal_columns.values() if p['scope']==scope]
        statements = ifrs_preparation_statements(text,default_scope=scope,fiscal_ends=[p['end'] for p in scoped_fiscal])
        for candidate in candidates:
            if candidate[0] != scope:continue
            evidence = preparation_for_year(statements,scope,candidate[1],candidate[6],{})
            if evidence is not None:basis[(scope,candidate[1])] = evidence
        for key,evidence in interim_annual_comparative_basis(text,scope,interim_years[scope],published,
                scoped_fiscal,interim_ends[scope],interim_terms[scope]).items():
            basis.setdefault(key,evidence)
    for key,evidence in (summary_basis or {}).items():
        basis.setdefault(key,json.dumps(evidence,ensure_ascii=False))
    if basis_evidence is not None:
        basis_evidence.update(basis)
    rows = []
    for scope, year, table_id, positions, column, scale, header, period,pair,period_origins,headers in candidates:
        evidence = basis.get((scope, year))
        if evidence is None and re.search(r'[\[(]\s*(?:K\s*-?\s*IFRS|한국채택국제회계기준)\s*[\])]',header,re.I):
            evidence=json.dumps({'rule':'issuer_primary_explicit_column_standard_v1','scope':scope,
                'year':year,'table_id':table_id,'column_id':column,'header':header,'annual_period_text':period},ensure_ascii=False)
        if evidence is None and pair is not None:
            evidence=interim_comparative_balance_basis(' '.join(regions[scope]),scope,pair)
        if evidence is None:
            rejected["accounting_basis_unverified"] += 1
            continue
        for row_id, row, name, loss in positions:
            if len(row) <= column:
                continue
            raw = row[column].strip().replace("−", "-")
            if not re.fullmatch(r"(?:-?\d[\d,]*(?:\.\d+)?|\(\d[\d,]*(?:\.\d+)?\))", raw):
                continue
            value = float(raw.replace(",", "").strip("()")) * (-1 if raw.startswith("(") else 1)
            if loss:
                value = -abs(value)
            multiplier = 1 if name == "eps" else scale
            rows.append({"year": year, "account_name_en": name, "amount": value * multiplier,
                "corp_code": str(corp_code), "rcept_no": receipt, "report_code": "IPO_FINANCIAL_DISCLOSURE",
                "response_year": str(year), "fs_div": scope, "currency": "KRW", "raw_amount": raw,
                "raw_account_label": row[0], "loss_label_sign_applied": loss, "unit_multiplier": multiplier,
                "publication_verified": True, "published_at": str(published)[:10], "source_url": source_url,
                "source_sha256": digest, "source_kind": "ipo_disclosure", "table_id": table_id,
                "row_id": row_id, "column_id": column, "period_name": period,
                "accounting_standard": "K-IFRS", "accounting_standard_evidence": evidence,
                "entity_scope":"issuer", "statement_basis_explicit":True,
                "statement_scope_evidence":parser.table_contexts[table_id],
                "primary_balance_scope_evidence":[],
                "extraction_rule_id": "detailed_annual_period_and_basis_v1"})
            if period_origins:rows[-1]['period_source_evidence']=json.dumps(period_origins,ensure_ascii=False)
            fiscal=fiscal_columns.get((table_id,column))
            if fiscal and fiscal['scope']!=scope:fiscal=None
            if fiscal:
                rows[-1].update(period_start=fiscal['start'],period_end=fiscal['end'],
                    fiscal_period_evidence=json.dumps(fiscal,ensure_ascii=False))
    return rows, rejected


def parse_disclosure_financials(html, corp_code, receipt, published, source_url, diagnostics=None, *, document_cache=None):
    """Require issuer summary scope, annual periods, IFRS, explicit monetary units."""
    structure=financial_document_structure(html,document_cache)
    parser=SimpleNamespace(tables=structure['tables'],table_contexts=structure['table_contexts'],
                           _outside_text=structure['outside_text'])
    narrative=structure['narrative']
    fiscal_columns=noncalendar_fiscal_columns(parser)
    primary = primary_annual_balances(parser,fiscal_columns)
    scope, scale, rows = None, None, []
    entity_scope, scope_evidence = "issuer", None
    scope_explicit, resolved_scopes = False, {}
    standards = {}
    scope_standard = None
    term_years = {}
    rejected = Counter()
    # This is an affirmative preparation statement, not a generic IFRS mention.
    frameworks=ifrs_preparation_statements(narrative)
    primary_frameworks=None
    shared_framework_groups=set()
    summary_basis={}
    # A range's end year is not a fiscal-term identity. Use dated issuer
    # year-end balance columns, never a two-year income heading or note prose.
    term_proofs={}
    primary_terms={}
    for group in primary['groups']:
        for term in group['fiscal_terms']:
            key=str(term)
            term_years.setdefault(key,set()).add(str(group['year']))
            term_proofs.setdefault(key,[]).append(group)
            primary_terms.setdefault((group['fs_div'],term),set()).add(group['year'])
    digest = hashlib.sha256(html.encode()).hexdigest()
    for table_id, (context, table) in enumerate(zip(parser.table_contexts, parser.tables)):
        declared = summary_scope_heading(context, allow_statement_subheading=scope is not None and entity_scope == "issuer")
        # Ignore older headings before the last summary, but retain its owner.
        current_heading = clean(context[declared['start']:] if declared else context)
        boundary = bool(re.search(r"(?:연결대상|회계정책|중단영업|[2345](?:연결|별도|개별)?재무제표)", current_heading))
        if declared:
            scope = declared['scope']
            entity_scope = declared['entity_scope']
            scope_evidence = declared['evidence']
            scope_explicit = declared['basis_explicit']
            resolved_scopes = {}
            standards = {}
            scale = None
            declared_standard = re.search(r"KIFRS|KGAAP|KGAPP|일반기업회계기준|한국채택국제회계기준",clean(declared['standard_context']))
            scope_standard = ("K-GAAP" if declared_standard[0] in {"KGAAP","KGAPP","일반기업회계기준"} else "K-IFRS") if declared_standard else None
        if boundary:
            scope, scale, scope_standard = None, None, None
            entity_scope, scope_evidence = "issuer", None
            scope_explicit, resolved_scopes = False, {}
            standards = {}
        # A standalone framework caption starts a new basis even without a
        # repeated summary title. A mention inside a preceding note does not.
        if standalone_non_ifrs_caption(context):
            scope_standard='K-GAAP'
            standards={}
            resolved_scopes={}
        unit_text = context + " " + " ".join(" ".join(r) for r in table[:2])
        unit = re.search(r"단위\s*[:：]\s*(백\s*만\s*원|천\s*원|원|KRW)", unit_text)
        if unit:
            scale = {"백만원": 1_000_000, "천원": 1_000, "원": 1, "KRW": 1}[re.sub(r'\s+','',unit.group(1))]
        if entity_scope != "issuer":
            rejected['non_issuer_statement_scope'] += 1
            continue
        if scope is None or len(table) < 4:
            if len(table) >= 4:
                rejected['summary_scope_unresolved'] += 1
            continue
        if declared:
            summary_basis.update(summary_column_standard(table,scope,context,table_id,primary,scope_explicit))
        if scale is None:rejected['krw_unit_unverified'] += 1
        account_positions = [(i, r, account_name(r[0])) for i, r in enumerate(table) if r]
        account_positions = [(i, r, name) for i, r, name in account_positions if name]
        if len(account_positions) < 2:
            continue
        first_account = account_positions[0][0]
        data_start = next((i for i, row in enumerate(table[:first_account])
                           if any(re.fullmatch(r"\(?-?\d[\d,]*(?:\.\d+)?\)?", cell.strip())
                                  for cell in row[1:])
                           and clean(row[0]) not in {"구분", "기간", "사업연도"}), first_account)
        headers = table[:data_start]
        local_terms={}
        term_titles=list(re.finditer(r'제\s*(\d+)\s*(?:\([^)]*\))?\s*기',context))
        for index,term in enumerate(term_titles):
            title=context[term.end():term_titles[index+1].start() if index+1<len(term_titles) else len(context)]
            dates=re.findall(r'((?:19|20)\d{2})\s*(?:년|[./-])\s*(\d{1,2})\s*(?:월|[./-])\s*(\d{1,2})',title)
            if (len(dates)==1 and tuple(map(int,dates[0][1:]))==(12,31)
                    and not re.search('분기|반기|중간',title)):
                local_terms.setdefault(str(int(term[1])),set()).add(dates[0][0])
        adjacent = ""
        if table_id + 1 < len(parser.tables):
            adjacent = parser.table_contexts[table_id + 1].split("2. 연결재무제표")[0]
            following = parser.tables[table_id + 1]
            labelled_notes = bool(following) and all(
                len(row) == 1 and re.match(r"\s*(?:주\s*\d+[).]|\(주\s*\d+\))", row[0])
                for row in following)
            if len(following) <= 2 or labelled_notes:
                adjacent += " ".join(" ".join(r) for r in following)
        table_frameworks=ifrs_preparation_statements(adjacent,default_scope=scope,table_note=True)
        for column in range(1, max(map(len, table))):
            header = " ".join(r[column] for r in headers if len(r) > column)
            years = set(re.findall(r"(?:19|20)\d{2}", header))
            fiscal_evidence=None
            fiscal=fiscal_columns.get((table_id,column))
            if fiscal and fiscal['scope']==scope:
                years={str(fiscal['year'])}
                fiscal_evidence=json.dumps(fiscal,ensure_ascii=False)
            if not years:
                term = re.search(r"제\s*(\d+)\s*(?:\([^)]*\))?\s*기", header)
                key=str(int(term[1])) if term else None
                years = (term_years.get(key,set()) | local_terms.get(key,set())) if term else set()
                column_periods=[]
                if term:
                    for period_row,row in enumerate(table):
                        if len(row)<=column or clean(row[0]) not in {'구분','과목','기간','사업연도','손익현황'}:continue
                        cell=row[column]
                        ids={int(t) for t in re.findall(r'제\s*(\d+)\s*(?:\([^)]*\))?\s*기',cell)}
                        dates=[tuple(map(int,d)) for d in re.findall(
                            r'((?:19|20)\d{2})\s*(?:년|[./-])\s*(\d{1,2})\s*(?:월|[./-])\s*(\d{1,2})',cell)]
                        if (ids=={int(term[1])} and len(dates)==2
                                and dates==[(dates[0][0],1,1),(dates[0][0],12,31)]
                                and not re.search('분기|반기|예정|추정|예상|잠정|가결산|차기',cell)):
                            years=years | {str(dates[0][0])}
                            column_periods.append({'table_id':table_id,'row_id':period_row,'column_id':column,'text':cell})
                if term and len(years)==1:
                    fiscal_evidence=json.dumps({'rule':('issuer_primary_year_end_fiscal_term_v1' if key in term_proofs
                        else 'issuer_local_summary_year_end_fiscal_term_v1'),
                        'term':int(term[1]),'primary_balance_columns':term_proofs.get(key,[]),
                        'same_column_explicit_annual_periods':column_periods,
                        'local_summary_heading':context if key in local_terms else None},ensure_ascii=False)
            if len(years) != 1 or re.search(r"분기|반기|기초|잠정|추정|예상|가결산|수정|조정금액|차이금액", header):
                rejected['annual_period_unresolved_or_not_annual'] += 1
                continue
            if re.search(r"설립\s*전|분할\s*전|사업부|carve\s*out", header, re.I):
                continue
            if fiscal is None and re.search(r"\d{4}년\s*0?(?:[1-9]|10|11)월", header):
                continue
            calendar_dates = re.findall(r"\d{4}\s*[.\-/]\s*(\d{1,2})\s*[.\-/]\s*(\d{1,2})", header)
            if fiscal is None and calendar_dates and any((int(m), int(d)) != (12, 31) for m, d in calendar_dates):
                # Income headers with explicit full-year ranges are handled separately.
                if not income_period([["", header], ["", ""]], 0, 1, 1, int(next(iter(years))))[0]:
                    continue
                if not re.search(r"[~～]", header):
                    continue
            year = int(next(iter(years)))
            if pd.Timestamp(fiscal['end'] if fiscal else f"{year}-12-31") >= pd.Timestamp(published).tz_localize(None):
                continue
            column_scope, primary_scope_proof = resolved_scopes.get(year, (None, []))
            if column_scope is None:
                if scale is None and not scope_explicit and primary['consolidated_primary_present']:
                    column_scope,primary_scope_proof=None,[]
                else:
                    column_scope, primary_scope_proof = summary_column_basis(
                        scope,scope_explicit,account_positions,column,year,scale,primary)
            if column_scope is None:
                rejected['summary_primary_statement_basis_unverified'] += 1
                continue
            resolved_scopes[year]=(column_scope,primary_scope_proof)
            if re.search(r'KGAAP|KGAPP|일반기업회계기준',clean(header)):
                rejected['explicit_gaap_not_ifrs'] += 1
                continue
            if scope_standard == "K-GAAP" and "KIFRS" not in clean(header):
                rejected['explicit_gaap_not_ifrs'] += 1
                continue
            scoped_frameworks=(table_frameworks if column_scope==scope else
                               ifrs_preparation_statements(adjacent,default_scope=column_scope,table_note=True))
            local_evidence=(preparation_for_year(scoped_frameworks,column_scope,year,header,term_years)
                or preparation_for_year(ifrs_preparation_statements(
                    context[declared['start']:] if declared else context,default_scope=column_scope),
                    column_scope,year,header,term_years))
            if declared and local_evidence is None:
                parent=re.search(r'(?<![가-힣])당사의\s*요약재무정보는[^.。]{0,160}?작성\s*(?:되었|하였)',context[:declared['start']])
                if parent:
                    local_evidence=preparation_for_year(ifrs_preparation_statements(
                        parent[0],default_scope=column_scope),column_scope,year,header,term_years)
            framework_evidence=(local_evidence
                or preparation_for_year(frameworks,column_scope,year,header,term_years))
            if framework_evidence and any(statement.get('implicit_scope_annual_reference')
                    and preparation_for_year([statement],column_scope,year,header,term_years)==framework_evidence
                    for statement in frameworks):
                shared_framework_groups.add((year,column_scope))
            if (framework_evidence is None and (year,column_scope) not in standards
                    and 'KIFRS' not in clean(header) and scope_standard!='K-IFRS'):
                if primary_frameworks is None:
                    primary_frameworks={}
                    detailed_annual_accounts(parser,corp_code,receipt,published,source_url,digest,
                        primary_terms=primary_terms,basis_evidence=primary_frameworks,fiscal_columns=fiscal_columns)
                framework_evidence=primary_frameworks.get((column_scope,year))
                if framework_evidence is not None:shared_framework_groups.add((year,column_scope))
            elif framework_evidence is None and (year,column_scope) in shared_framework_groups:
                framework_evidence=primary_frameworks[(column_scope,year)]
            if framework_evidence is None:framework_evidence=standards.get((year,column_scope))
            if "KIFRS" in clean(header) or scope_standard == "K-IFRS" or framework_evidence:
                standards[(year,column_scope)]=framework_evidence or (scope_evidence if scope_standard=='K-IFRS' else header)
            if (year,column_scope) not in standards:
                rejected['accounting_basis_unverified'] += 1
                continue
            if local_evidence or 'KIFRS' in clean(header):
                summary_basis[(column_scope,year)]={
                    'rule':'issuer_summary_year_standard_bound_to_primary_v1',
                    'scope':column_scope,'year':year,'summary_table_id':table_id,
                    'column_id':column,'header_rows':headers,
                    'fiscal_term_year_evidence':fiscal_evidence,
                    'statement_scope_evidence':scope_evidence,
                    'standard_preparation_evidence':local_evidence or header}
            # A standard declaration is independent of a monetary unit. Only
            # the primary statement's own verified unit can supply its amounts.
            if scale is None:continue
            for row_id, row, account in account_positions:
                if len(row) <= column:
                    continue
                # Income rows can have a different period from balance-sheet columns.
                annual, period = income_period(table, 0, row_id, column, year)
                if account not in {"total_assets", "total_liabilities", "equity"}:
                    if not annual:
                        continue
                raw = row[column].strip().replace("−", "-")
                if not re.fullmatch(r"(?:-?\d[\d,]*(?:\.\d+)?|\(\d[\d,]*(?:\.\d+)?\))", raw):
                    continue
                amount = float(raw.replace(",", "").strip("()")) * (-1 if raw.startswith("(") else 1)
                amount *= 1 if account == "eps" else scale
                rows.append({"year": year, "account_name_en": account, "amount": amount,
                             "corp_code": str(corp_code), "rcept_no": receipt,
                             "report_code": "IPO_FINANCIAL_DISCLOSURE", "response_year": str(year),
                             "fs_div": column_scope, "currency": "KRW", "raw_amount": raw,
                             "unit_multiplier": 1 if account == "eps" else scale,
                             "publication_verified": True, "published_at": str(published)[:10],
                             "source_url": source_url, "source_sha256": digest, "source_kind": "ipo_disclosure",
                             "table_id": table_id, "row_id": row_id, "column_id": column,
                             "period_name": period or header, "accounting_standard": "K-IFRS",
                             "entity_scope": entity_scope, "statement_scope_evidence": scope_evidence,
                             "statement_basis_explicit":scope_explicit,
                             "primary_balance_scope_evidence":primary_scope_proof,
                             "extraction_rule_id": "summary_issuer_statement_scope_v2"})
                rows[-1]['fiscal_term_year_evidence']=fiscal_evidence
                if fiscal:
                    rows[-1].update(period_start=fiscal['start'],period_end=fiscal['end'],
                        fiscal_period_evidence=fiscal_evidence)
                rows[-1]["accounting_standard_evidence"] = standards[(year,column_scope)]
    detailed_rejections = Counter()
    unit_conflicts = summary_unit_conflicts(primary, rows)
    conflicting_tables = {item['summary_table_id'] for item in unit_conflicts}
    rows = [row for row in rows if row['table_id'] not in conflicting_tables]
    core = {'revenue','operating_income','total_liabilities','equity'}
    summary_groups = {}
    for row in rows:
        summary_groups.setdefault((row['year'],row['fs_div']),[]).append(row)
    detailed, detailed_rejections = detailed_annual_accounts(
        parser, corp_code, receipt, published, source_url, digest, summary_basis, primary_terms,
        fiscal_columns=fiscal_columns)
    primary_groups, amount_conflicts = {}, []
    disputed_units={(row['year'],row['fs_div']) for row in unit_conflicts}
    for row in detailed:
        primary_groups.setdefault((row['year'],row['fs_div']),[]).append(row)
    for key,group in primary_groups.items():
        if key in disputed_units:continue
        summary_group=summary_groups.get(key,[])
        incompatible=[]
        for summary_row in summary_group:
            if summary_row['account_name_en']=='eps':continue
            peers=[r for r in group if r['account_name_en']==summary_row['account_name_en']]
            if peers and not any(abs(summary_row['amount']-r['amount'])<=2*max(
                    summary_row['unit_multiplier'],r['unit_multiplier']) for r in peers):
                incompatible.append({'summary':summary_row,'primary':peers})
        if incompatible:
            amount_conflicts.append({'year':key[0],'fs_div':key[1],'evidence':incompatible})
            rows=[row for row in rows if (row['year'],row['fs_div'])!=key]
            continue
        summary_names={row['account_name_en'] for row in summary_group}
        primary_names={row['account_name_en'] for row in group}
        precise_primary=(key in shared_framework_groups and bool(summary_group) and core.issubset(primary_names)
            and summary_names.issubset(primary_names)
            and all(len({r['amount'] for r in group if r['account_name_en']==name})==1 for name in primary_names)
            and all(any(r['unit_multiplier']<=s['unit_multiplier'] for r in group
                        if r['account_name_en']==s['account_name_en']) for s in summary_group)
            and any(r['unit_multiplier']<s['unit_multiplier'] for s in summary_group for r in group
                    if r['account_name_en']==s['account_name_en']))
        if (not summary_group or not core.issubset({r['account_name_en'] for r in summary_group})
                and core.issubset(primary_names) or precise_primary):
            # Use one coherent statement group, not a blend of rounded/restated cells.
            rows=[row for row in rows if (row['year'],row['fs_div'])!=key]
            rows.extend(group)
        else:
            # Balance sheets and income statements are separate statements.
            # A newly verified partial summary must not suppress a complete
            # primary income statement from the same receipt, year and scope.
            for family in ({'total_assets','total_liabilities','equity'},
                           {'revenue','operating_income','net_income','eps'}):
                summary_family=[r for r in summary_group if r['account_name_en'] in family]
                primary_family=[r for r in group if r['account_name_en'] in family]
                summary_accounts={r['account_name_en'] for r in summary_family}
                primary_accounts={r['account_name_en'] for r in primary_family}
                required=({'total_assets','total_liabilities','equity'} if 'total_assets' in family
                          else {'revenue','operating_income','net_income'})
                family_precise=(bool(summary_family)
                    and summary_accounts.issubset(primary_accounts)
                    and all(any(r['unit_multiplier']<=s['unit_multiplier'] for r in primary_family
                                if r['account_name_en']==s['account_name_en']) for s in summary_family)
                    and any(r['unit_multiplier']<s['unit_multiplier'] for s in summary_family for r in primary_family
                            if r['account_name_en']==s['account_name_en']))
                if (required.issubset(primary_accounts) and (summary_accounts < primary_accounts or family_precise)
                        and all(len({r['amount'] for r in primary_family if r['account_name_en']==name})==1
                                for name in primary_accounts)):
                    rows=[r for r in rows if (r['year'],r['fs_div'])!=key or r['account_name_en'] not in family]
                    rows.extend(primary_family)
    if diagnostics is not None:
        diagnostics.update(tables=len(parser.tables), extracted_accounts=len(rows),
                           document_sha256=digest, source_url=source_url, receipt=receipt,
                           absence_confirmed=False,
                           reason="annual_accounts_extracted" if rows else "annual_summary_format_or_basis_unverified")
        diagnostics['rejected_table_or_column_counts'] = dict(rejected)
        diagnostics['detailed_statement_rejection_counts'] = dict(detailed_rejections)
        diagnostics['official_statement_unit_conflicts'] = unit_conflicts
        diagnostics['official_statement_amount_conflicts'] = amount_conflicts
        diagnostics['summary_primary_basis_unresolved_columns'] = rejected['summary_primary_statement_basis_unverified']
        diagnostics['selected_document_explicit_summary_absence'] = bool(
            not rows and re.search(r'요약\s*재무정보\s*(?:당사는\s*)?(?:증권신고서\s*(?:제출일|작성\s*기준일)\s*현재\s*)?해당사항\s*없', narrative[:2000]))
    return pd.DataFrame(rows)


def collect_disclosure_financials(dart, raw_dir, event_id, corp_code, corp_name, cutoff, lineage=None,
                                 diagnostics=None):
    lineage_path = Path(raw_dir) / "dart_disclosure_lineage.parquet"
    if lineage is None:
        if not lineage_path.exists():
            return pd.DataFrame(), "financial_event_lineage_missing"
        lineage = pd.read_parquet(lineage_path)
    required = {"event_id", "corp_code", "rcept_no", "rcept_dt", "filing_report_nm"}
    if not required.issubset(lineage):
        return pd.DataFrame(), "financial_event_lineage_missing"
    group = lineage[lineage.event_id.astype(str).eq(str(event_id))
                    & lineage.corp_code.astype(str).eq(str(corp_code))].copy()
    dates = pd.to_datetime(group.rcept_dt, errors="coerce")
    local_cutoff = pd.Timestamp(cutoff).tz_localize(None)
    group = group[(dates + pd.Timedelta(days=1) < local_cutoff)
                  & group.filing_report_nm.str.contains(r"투자설명서|증권신고서\(지분증권\)", na=False)]
    if group.empty:
        return pd.DataFrame(), "financial_pre_listing_disclosure_not_identified"
    # A terms-only correction is not a complete financial snapshot. Prefer the
    # latest prospectus; never silently substitute an arbitrary older report.
    latest_day = pd.to_datetime(group.rcept_dt).max()
    prospectuses = group[pd.to_datetime(group.rcept_dt).eq(latest_day)
                        & group.filing_report_nm.str.contains("투자설명서", na=False)]
    candidates = prospectuses if not prospectuses.empty else group
    chosen = candidates.sort_values(["rcept_dt", "rcept_no"]).iloc[-1]
    receipt = str(chosen.rcept_no)
    if not re.fullmatch(r"\d{14}", receipt):
        return pd.DataFrame(), "financial_disclosure_receipt_date_mismatch"
    from data.pipelines.financial_source_selection import resolve_financial_section
    payload = resolve_financial_section(dart,raw_dir,receipt,corp_name,str(corp_code),local_cutoff,
                                        pd.Timestamp(chosen.rcept_dt).date())
    if payload.get("status") != "official_section_downloaded":
        if diagnostics is not None:
            diagnostics.update(receipt=receipt, reason=payload.get('status'), absence_confirmed=False)
        return pd.DataFrame(), payload.get("status", "financial_source_unverified")
    actual_published=pd.Timestamp(payload['official_publication_date'])
    receipt=payload['receipt']
    if diagnostics is not None:
        diagnostics.update(receipt=receipt,actual_publication_date=str(actual_published.date()),
                           official_corp_code=payload['official_corp_code'],index_sha256=payload.get('index_sha256'),
                           document_sha256=payload['sha256'],source_url=payload['source_url'],absence_confirmed=False)
    if actual_published+pd.Timedelta(days=1)>=local_cutoff:
        return pd.DataFrame(),'financial_actual_publication_after_cutoff'
    identity = {'document_sha256': payload['sha256'], 'corp_code': str(corp_code),
                'published': str(actual_published.date()),'index_sha256':payload.get('index_sha256'),
                'document_structure':financial_document_structure_contract(),
                'parser_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    cache_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = Path(raw_dir) / 'dart_financial_parsed' / f'{receipt}_{cache_id}.json'
    if cache.exists():
        saved = json.loads(cache.read_text())
        records_hash = hashlib.sha256(json.dumps(saved['records'], sort_keys=True).encode()).hexdigest()
        if saved['identity'] != identity or records_hash != saved['records_sha256']:
            raise RuntimeError('parsed_financial_cache_integrity_failed')
        result = pd.DataFrame(saved['records'])
        if diagnostics is not None:
            diagnostics.update(saved['diagnostics'])
            if payload.get('selection_evidence'):diagnostics['financial_source_selection']=payload['selection_evidence']
        return result, saved['status']
    details = {}
    result = parse_disclosure_financials(payload["html"], corp_code, receipt,
                                        actual_published.date(), payload["source_url"], details,
                                        document_cache=Path(raw_dir)/'financial_document_geometry')
    if not result.empty:
        result['source_collected_at']=payload.get('collected_at')
        result['source_hash_scope']='original_official_section_html_utf8'
    details.update(official_corp_code=payload['official_corp_code'],actual_publication_date=str(actual_published.date()),
                   original_lineage_date=str(pd.Timestamp(chosen.rcept_dt).date()),
                   publication_date_corrected=payload.get('publication_date_corrected',False),
                   index_url=payload.get('index_url'),index_sha256=payload.get('index_sha256'))
    status = ("official_annual_financial_tables_found" if not result.empty else
              "official_statement_unit_conflict_requires_review" if details.get('official_statement_unit_conflicts') else
              "official_statement_amount_conflict_requires_review" if details.get('official_statement_amount_conflicts') else
              "summary_statement_basis_unverified" if details.get('summary_primary_basis_unresolved_columns') else
              "official_financial_section_without_tables" if '<table' not in payload['html'].lower() else
              "annual_financial_table_format_unhandled")
    records = financial_records(result)
    atomic_json(cache, {'identity': identity, 'records': records, 'diagnostics': details, 'status': status,
                       'records_sha256': hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()})
    if diagnostics is not None:
        diagnostics.update(details)
        if payload.get('selection_evidence'):diagnostics['financial_source_selection']=payload['selection_evidence']
    return result, status
