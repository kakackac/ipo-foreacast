"""Issuer annual financial tables from the official pre-listing DART section."""
import hashlib
import json
import re
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests

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


def public_financial_section(dart, receipt, corp_name, cache_dir):
    """Read a bounded public section, never the first arbitrary viewer link."""
    path = Path(cache_dir) / f"{receipt}.json"
    if path.exists():
        payload = json.loads(path.read_text())
        if (payload.get("receipt") == receipt and issuer_name(payload.get("issuer", "")) == issuer_name(corp_name)
                and hashlib.sha256(payload["html"].encode()).hexdigest() == payload.get("sha256")):
            return payload
        raise RuntimeError("financial_disclosure_cache_integrity_failed")
    if not hasattr(dart, "session"):
        return {"status": "financial_public_transport_unavailable"}
    url = "https://dart.fss.or.kr/dsaf001/main.do?" + urlencode({"rcpNo": receipt})
    try:
        response = dart.session.get(url, timeout=30)
        response.raise_for_status()
        response.encoding = "utf-8"
        index = response.text
        title = re.search(r"<title>([^<]+)</title>", index, re.I)
        issuer = unescape(title.group(1)).split("/")[0].strip() if title else ""
        if issuer_name(issuer) != issuer_name(corp_name):
            return {"status": "official_document_issuer_mismatch"}
        node = select_financial_node(index, receipt)
        if node is None:
            return {"status": "financial_section_not_found"}
        section_url = "https://dart.fss.or.kr/report/viewer.do?" + urlencode(node)
        response = dart.session.get(section_url, timeout=30)
        response.raise_for_status()
        response.encoding = "utf-8"
        html = response.text
    except requests.RequestException as exc:
        return {"status": "financial_public_source_access_failed", "error_type": type(exc).__name__}
    if "재무" not in html or "<table" not in html.lower():
        return {"status": "financial_section_response_invalid"}
    payload = {"receipt": receipt, "issuer": issuer, "status": "official_section_downloaded",
               "html": html, "sha256": hashlib.sha256(html.encode()).hexdigest(),
               "source_url": section_url, "collected_at": datetime.now(timezone.utc).isoformat()}
    atomic_json(path, payload)
    return payload


ACCOUNTS = {
    "매출액": "revenue", "영업수익": "revenue", "수익매출액": "revenue",
    "영업이익": "operating_income", "영업이익손실": "operating_income", "영업손익": "operating_income",
    "당기순이익": "net_income", "당기순이익손실": "net_income", "당기순손익": "net_income",
    "자산총계": "total_assets", "부채총계": "total_liabilities", "자본총계": "equity",
    "주당순이익원": "eps", "기본주당순이익원": "eps",
}


def parse_disclosure_financials(html, corp_code, receipt, published, source_url):
    """Require issuer summary scope, annual periods, IFRS, explicit monetary units."""
    parser = _TableRowParser()
    parser.feed(html)
    scope, scale, rows = None, None, []
    standards = set()
    term_years = {}
    for table in parser.tables:
        for row in table:
            for cell in row:
                match = re.search(r"제\s*(\d+)\s*(?:\([^)]*\))?\s*기.*?((?:19|20)\d{2})년\s*12월\s*31일", cell)
                if match:
                    term_years.setdefault(match[1], set()).add(match[2])
    digest = hashlib.sha256(html.encode()).hexdigest()
    for table_id, (context, table) in enumerate(zip(parser.table_contexts, parser.tables)):
        heading = clean(context)
        if "요약연결재무정보" in heading:
            scope = "CFS"
            standards = set()
        elif "요약별도재무정보" in heading:
            scope = "OFS"
            standards = set()
        elif "요약재무정보" in heading:
            scope = "OFS"
            standards = set()
        elif context.strip() and re.search(r"(?:연결대상|회계정책|중단영업|재무제표주석|연결재무제표주석|[234]연결재무제표|4재무제표)", heading):
            scope, scale = None, None
            standards = set()
        unit_text = context + " " + " ".join(" ".join(r) for r in table[:2])
        unit = re.search(r"단위\s*[:：]\s*(백만원|천원|원|KRW)", unit_text)
        if unit:
            scale = {"백만원": 1_000_000, "천원": 1_000, "원": 1, "KRW": 1}[unit.group(1)]
        if scope is None or scale is None or len(table) < 4:
            continue
        account_positions = [(i, r, ACCOUNTS.get(clean(r[0]))) for i, r in enumerate(table) if r]
        account_positions = [(i, r, name) for i, r, name in account_positions if name]
        if len(account_positions) < 3:
            continue
        first_account = account_positions[0][0]
        headers = table[:first_account]
        adjacent = ""
        if table_id + 1 < len(parser.tables):
            adjacent = parser.table_contexts[table_id + 1].split("2. 연결재무제표")[0]
            following = parser.tables[table_id + 1]
            if len(following) <= 2:
                adjacent += " ".join(" ".join(r) for r in following)
        table_ifrs = "한국채택국제회계기준" in clean(adjacent) or "KIFRS" in clean(adjacent)
        for column in range(1, max(map(len, table))):
            header = " ".join(r[column] for r in headers if len(r) > column)
            years = set(re.findall(r"(?:19|20)\d{2}", header))
            if not years:
                term = re.search(r"제\s*(\d+)\s*기", header)
                years = term_years.get(term[1], set()) if term else set()
            if len(years) != 1 or re.search(r"분기|반기|잠정|추정|예상|가결산", header):
                continue
            if re.search(r"\d{4}년\s*(?:[1-9]|10|11)월", header):
                continue
            year = int(next(iter(years)))
            if pd.Timestamp(f"{year}-12-31") >= pd.Timestamp(published).tz_localize(None):
                continue
            if "KIFRS" in clean(header) or table_ifrs:
                standards.add(year)
            if year not in standards:
                continue
            for row_id, row, account in account_positions:
                if len(row) <= column:
                    continue
                # Income rows can have a different period from balance-sheet columns.
                period = None
                for preceding in table[first_account:row_id]:
                    if len(preceding) > column and re.search(r"\d{4}[.\-/]\d{2}[.\-/]\d{2}.*[~～]", preceding[column]):
                        period = preceding[column]
                if account not in {"total_assets", "total_liabilities", "equity"}:
                    if period is not None and not re.search(rf"{year}[.\-/]01[.\-/]01\s*[~～]\s*{year}[.\-/]12[.\-/]31", period):
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
    return pd.DataFrame(rows)


def collect_disclosure_financials(dart, raw_dir, event_id, corp_code, corp_name, cutoff, lineage=None):
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
    if not re.fullmatch(r"\d{14}", receipt) or pd.Timestamp(chosen.rcept_dt).strftime("%Y%m%d") != receipt[:8]:
        return pd.DataFrame(), "financial_disclosure_receipt_date_mismatch"
    payload = public_financial_section(dart, receipt, corp_name, Path(raw_dir) / "dart_ipo_financial_sections")
    if payload.get("status") != "official_section_downloaded":
        return pd.DataFrame(), payload.get("status", "financial_source_unverified")
    result = parse_disclosure_financials(payload["html"], corp_code, receipt,
                                        pd.Timestamp(chosen.rcept_dt).date(), payload["source_url"])
    return result, "official_annual_financial_tables_found" if not result.empty else "annual_financial_table_format_unhandled"
