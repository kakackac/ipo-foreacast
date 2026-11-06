"""Issuer annual financial tables from the official pre-listing DART section."""
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from urllib.parse import urlencode, urlparse, parse_qs

import pandas as pd
import requests
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
    for block in re.split(r"var node\d+ = \{\};", index)[1:]:
        fields = dict(re.findall(r"node\d+\['(text|rcpNo|dcmNo|eleId|offset|length|dtd)'\]\s*=\s*\"([^\"]*)\"", block))
        if fields.get("rcpNo") == receipt and "재무에관한사항" in clean(fields.get("text", "")):
            if all(key in fields for key in ("rcpNo", "dcmNo", "eleId", "offset", "length", "dtd")):
                matches.append({key: fields[key] for key in ("rcpNo", "dcmNo", "eleId", "offset", "length", "dtd")})
    return matches[0] if len(matches) == 1 else None


def public_financial_section(dart, receipt, corp_name, cache_dir, corp_code=None, published=None):
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
        index,index_hash=public_text(dart,url,Path(cache_dir).parent/'dart_connection_sources',validate_index)
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
                lambda text: None if '<table' in text.lower() else (_ for _ in ()).throw(ValueError('financial_section_response_invalid')))
    except requests.RequestException as exc:
        return {"status": "financial_public_source_access_failed", "error_type": type(exc).__name__}
    except ValueError as exc:
        return {'status':str(exc)}
    if "재무" not in html or "<table" not in html.lower():
        return {"status": "financial_section_response_invalid"}
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
    "자산총계": "total_assets", "부채총계": "total_liabilities", "자본총계": "equity",
    "주당순이익원": "eps", "기본주당순이익원": "eps",
    "당분기순이익": "net_income", "당기순손익손실": "net_income",
    "영업이익영업손실": "operating_income", "당기순이익당기순손실": "net_income",
}


def account_name(value):
    return ACCOUNTS.get(clean(re.sub(r"^\s*(?:[IVX]+|\d+)[.)]\s*", "", value)))


def financial_table_document(html, soup=None):
    """Keep single-cell unit/heading tables in the following table's context."""
    soup = soup if soup is not None else BeautifulSoup(html, "html.parser")
    for table in list(soup.find_all("table")):
        text=table.get_text(" ",strip=True)
        cells=table.find_all(["td","th"])
        heading_only=(re.search(r"단위\s*[:：]",text)
                      and re.search(r"재무제표|재무상태표|손익계산서|제\s*\d+\s*기",text)
                      and not any(account_name(cell.get_text(" ",strip=True)) for cell in cells)
                      and not any(re.fullmatch(r"\(?-?\d[\d,]*(?:\.\d+)?\)?",cell.get_text(strip=True)) for cell in cells))
        if table.find("table") is not None:
            # Layout wrappers are not financial grids. Preserve their inner tables.
            table.unwrap()
        elif (heading_only or len(cells) == 1
              and re.search(r"단위\s*[:：]|요약\s*(?:연결|별도)?\s*재무", text)):
            paragraph = soup.new_tag("p")
            paragraph.string = table.get_text(" ", strip=True)
            table.replace_with(paragraph)
    return str(soup)


def income_period(table, first_account, row_id, column, year):
    """Explicit income ranges override annual balance-sheet headings, including wrapped rows."""
    period = None
    for index in range(first_account, row_id):
        cell = table[index][column] if len(table[index]) > column else ""
        if not re.search(r"(?:19|20)\d{2}.*[~～]", cell):
            continue
        period = " ".join(table[k][column] for k in range(index, min(index + 2, row_id))
                          if len(table[k]) > column)
    if period is None:
        return True, None
    dates = re.findall(r"((?:19|20)\d{2})\s*(?:[.\-/]|년)\s*(\d{1,2})\s*(?:[.\-/]|월)\s*(\d{1,2})", period)
    if len(dates) == 2:
        return ([tuple(map(int, d)) for d in dates] == [(year, 1, 1), (year, 12, 31)]), period
    months = re.search(r"(\d{4})년\s*(\d{1,2})월\s*[~～]\s*(?:(\d{4})년\s*)?(\d{1,2})월", period)
    if months:
        return (int(months[1]) == year and int(months[2]) == 1
                and (months[3] is None or int(months[3]) == year) and int(months[4]) == 12), period
    return False, period


def parse_disclosure_financials(html, corp_code, receipt, published, source_url, diagnostics=None):
    """Require issuer summary scope, annual periods, IFRS, explicit monetary units."""
    parser = _TableRowParser()
    soup = BeautifulSoup(html, "html.parser")
    narrative = soup.get_text(" ", strip=True)
    parser.feed(financial_table_document(html, soup))
    scope, scale, rows = None, None, []
    standards = set()
    scope_standard = None
    term_years = {}
    rejected = Counter()
    # This is an affirmative preparation statement, not a generic IFRS mention.
    frameworks={"CFS":None,"OFS":None}
    for framework in re.finditer(r"(?<![가-힣])(?P<basis>연결|별도|개별)?재무제표는[^.。]{0,180}?한국채택국제회계기준(?:\([^)]*\))?\s*(?:작성기준)?에\s*따라\s*작성(?:되었|됐|하였|하고)",narrative):
        frameworks['CFS' if framework['basis']=='연결' else 'OFS']=framework.group(0)
    for pattern in [r"당사는\s*한국채택국제회계기준(?:\([^)]*\))?을\s*적용하여\s*(?P<basis>연결)?재무제표를\s*작성하고\s*있",
                    r"당사는\s*한국채택국제회계기준에\s*따라\s*(?P<basis>연결)?재무제표를\s*작성하였"]:
        for framework in re.finditer(pattern,narrative):
            frameworks['CFS' if framework['basis']=='연결' else 'OFS']=framework.group(0)
    for table in parser.tables:
        for row in table:
            for cell in row:
                match = re.search(r"제\s*(\d+)\s*(?:\([^)]*\))?\s*기.*?((?:19|20)\d{2})년\s*12월\s*31일", cell)
                if match:
                    term_years.setdefault(str(int(match[1])), set()).add(match[2])
    digest = hashlib.sha256(html.encode()).hexdigest()
    for table_id, (context, table) in enumerate(zip(parser.table_contexts, parser.tables)):
        heading = clean(context)
        scope_headings = list(re.finditer(r"(?:연결|별도)?요약(?:연결|별도)?재무(?:정보|제표)", heading))
        if scope_headings:
            # A preceding CFS footnote must not override the next OFS heading.
            heading = heading[scope_headings[-1].start():]
        boundary = bool(re.search(r"(?:연결대상|회계정책|중단영업|[2345](?:연결)?재무제표)", heading))
        new_scope = False
        if ("요약연결재무정보" in heading or "요약연결재무제표" in heading
                or "연결요약재무정보" in heading or "연결요약재무제표" in heading):
            scope = "CFS"
            new_scope = True
        elif ("요약별도재무정보" in heading or "별도요약재무제표" in heading
              or "별도재무제표" in heading and not boundary):
            scope = "OFS"
            new_scope = True
        elif "요약재무정보" in heading or "요약재무제표" in heading:
            scope = "CFS" if "연결재무제표" in heading else "OFS"
            new_scope = True
        elif boundary:
            scope, scale = None, None
            standards = set()
            scope_standard = None
        if new_scope:
            standards = set()
            scale = None
            scope_standard = ("K-IFRS" if "KIFRS" in heading or "한국채택국제회계기준" in heading
                              else "K-GAAP" if "KGAAP" in heading else None)
        if boundary:
            scope, scale, scope_standard = None, None, None
            standards = set()
        unit_text = context + " " + " ".join(" ".join(r) for r in table[:2])
        unit = re.search(r"단위\s*[:：]\s*(백\s*만\s*원|천\s*원|원|KRW)", unit_text)
        if unit:
            scale = {"백만원": 1_000_000, "천원": 1_000, "원": 1, "KRW": 1}[re.sub(r'\s+','',unit.group(1))]
        if scope is None or scale is None or len(table) < 4:
            if len(table) >= 4:
                rejected['summary_scope_unresolved' if scope is None else 'krw_unit_unverified'] += 1
            continue
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
        adjacent = ""
        if table_id + 1 < len(parser.tables):
            adjacent = parser.table_contexts[table_id + 1].split("2. 연결재무제표")[0]
            following = parser.tables[table_id + 1]
            if len(following) <= 2:
                adjacent += " ".join(" ".join(r) for r in following)
        basis_note=clean(adjacent)
        table_ifrs=bool(re.search(r'(?:한국채택국제회계기준|KIFRS)(?:에따라작성|기준으로작성|기준)',basis_note)
                        and not re.search(r'예정|적용할|채택할|차기',basis_note))
        framework_evidence=frameworks.get(scope)
        for column in range(1, max(map(len, table))):
            header = " ".join(r[column] for r in headers if len(r) > column)
            years = set(re.findall(r"(?:19|20)\d{2}", header))
            if not years:
                term = re.search(r"제\s*(\d+)\s*기", header)
                years = term_years.get(str(int(term[1])), set()) if term else set()
            if len(years) != 1 or re.search(r"분기|반기|잠정|추정|예상|가결산|수정전|수정후", header):
                rejected['annual_period_unresolved_or_not_annual'] += 1
                continue
            if re.search(r"설립\s*전|분할\s*전|사업부|carve\s*out", header, re.I):
                continue
            if re.search(r"\d{4}년\s*0?(?:[1-9]|10|11)월", header):
                continue
            calendar_dates = re.findall(r"\d{4}\s*[.\-/]\s*(\d{1,2})\s*[.\-/]\s*(\d{1,2})", header)
            if calendar_dates and any((int(m), int(d)) != (12, 31) for m, d in calendar_dates):
                # Income headers with explicit full-year ranges are handled separately.
                if not income_period([["", header], ["", ""]], 0, 1, 1, int(next(iter(years))))[0]:
                    continue
                if not re.search(r"[~～]", header):
                    continue
            year = int(next(iter(years)))
            if pd.Timestamp(f"{year}-12-31") >= pd.Timestamp(published).tz_localize(None):
                continue
            if "KGAAP" in clean(header):
                rejected['explicit_gaap_not_ifrs'] += 1
                continue
            if scope_standard == "K-GAAP" and "KIFRS" not in clean(header):
                rejected['explicit_gaap_not_ifrs'] += 1
                continue
            if "KIFRS" in clean(header) or table_ifrs or scope_standard == "K-IFRS" or framework_evidence:
                standards.add(year)
            if year not in standards:
                rejected['accounting_basis_unverified'] += 1
                continue
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
                             "fs_div": scope, "currency": "KRW", "raw_amount": raw,
                             "unit_multiplier": 1 if account == "eps" else scale,
                             "publication_verified": True, "published_at": str(published)[:10],
                             "source_url": source_url, "source_sha256": digest, "source_kind": "ipo_disclosure",
                             "table_id": table_id, "row_id": row_id, "column_id": column,
                             "period_name": period or header, "accounting_standard": "K-IFRS"})
                rows[-1]["accounting_standard_evidence"] = framework_evidence or header + adjacent
    if diagnostics is not None:
        diagnostics.update(tables=len(parser.tables), extracted_accounts=len(rows),
                           document_sha256=digest, source_url=source_url, receipt=receipt,
                           absence_confirmed=False,
                           reason="annual_accounts_extracted" if rows else "annual_summary_format_or_basis_unverified")
        diagnostics['rejected_table_or_column_counts'] = dict(rejected)
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
    payload = public_financial_section(dart, receipt, corp_name, Path(raw_dir) / "dart_ipo_financial_sections",
                                       corp_code=corp_code, published=pd.Timestamp(chosen.rcept_dt).date())
    if payload.get("status") != "official_section_downloaded":
        if diagnostics is not None:
            diagnostics.update(receipt=receipt, reason=payload.get('status'), absence_confirmed=False)
        return pd.DataFrame(), payload.get("status", "financial_source_unverified")
    actual_published=pd.Timestamp(payload['official_publication_date'])
    if diagnostics is not None:
        diagnostics.update(receipt=receipt,actual_publication_date=str(actual_published.date()),
                           official_corp_code=payload['official_corp_code'],index_sha256=payload.get('index_sha256'),
                           document_sha256=payload['sha256'],source_url=payload['source_url'],absence_confirmed=False)
    if actual_published+pd.Timedelta(days=1)>=local_cutoff:
        return pd.DataFrame(),'financial_actual_publication_after_cutoff'
    identity = {'document_sha256': payload['sha256'], 'corp_code': str(corp_code),
                'published': str(actual_published.date()),'index_sha256':payload.get('index_sha256'),
                'parser_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    cache_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = Path(raw_dir) / 'dart_financial_parsed' / f'{receipt}_{cache_id}.json'
    if cache.exists():
        saved = json.loads(cache.read_text())
        records_hash = hashlib.sha256(json.dumps(saved['records'], sort_keys=True).encode()).hexdigest()
        if saved['identity'] != identity or records_hash != saved['records_sha256']:
            raise RuntimeError('parsed_financial_cache_integrity_failed')
        result = pd.DataFrame(saved['records'])
        if diagnostics is not None: diagnostics.update(saved['diagnostics'])
        return result, saved['status']
    details = {}
    result = parse_disclosure_financials(payload["html"], corp_code, receipt,
                                        actual_published.date(), payload["source_url"], details)
    if not result.empty:
        result['source_collected_at']=payload.get('collected_at')
        result['source_hash_scope']='original_official_section_html_utf8'
    details.update(official_corp_code=payload['official_corp_code'],actual_publication_date=str(actual_published.date()),
                   original_lineage_date=str(pd.Timestamp(chosen.rcept_dt).date()),
                   publication_date_corrected=payload.get('publication_date_corrected',False),
                   index_url=payload.get('index_url'),index_sha256=payload.get('index_sha256'))
    status = "official_annual_financial_tables_found" if not result.empty else "annual_financial_table_format_unhandled"
    records = json.loads(result.to_json(orient='records'))
    atomic_json(cache, {'identity': identity, 'records': records, 'diagnostics': details, 'status': status,
                       'records_sha256': hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()})
    if diagnostics is not None: diagnostics.update(details)
    return result, status
