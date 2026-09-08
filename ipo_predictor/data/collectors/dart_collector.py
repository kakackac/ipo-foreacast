"""
data/collectors/dart_collector.py
──────────────────────────────────
DART OpenAPI를 통해 공모주 관련 공시 데이터를 수집한다.

주요 수집 대상:
  1. 수요예측 결과 공시 → 기관 경쟁률, 의무보유확약 비율
  2. 증권신고서 → 공모가 밴드, 주관사, 발행 구조
  3. 재무제표 → PER 산출을 위한 EPS, 매출, 영업이익 등

실제 운영 시 고려 사항:
  - DART API는 일 10,000건 호출 제한이 있다.
  - 수요예측 결과 공시(pblntfNo)는 공모 기간 종료 후 D+2 영업일에 등록된다.
  - 투자설명서(prospectus) PDF는 별도 파싱 파이프라인이 필요하다.
"""

import logging
import json
import re
import time
import zipfile
from io import BytesIO
from datetime import date, timedelta
from html import unescape as html_unescape
from html.parser import HTMLParser
from typing import Optional

import requests
import pandas as pd

from config import DART_API_KEY, DART_BASE_URL, RAW_DIR

logger = logging.getLogger(__name__)


# ── 상수 ──────────────────────────────────────────────────────
REQUEST_DELAY = 0.3          # API 호출 간격 (초) — 속도 제한 회피
MAX_RETRIES   = 3
TIMEOUT       = 15
DEMAND_PARSER_VERSION = 6

FINAL_PRICE_LABEL_PATTERN = (
    r"(?:1\s*주당\s*)?(?:(?:확정|최종)\s*공모가(?:액|격)?|공모가(?:액|격)?\s*확정)"
)
PRICE_CONTEXT_EXCLUSIONS = r"희망|밴드|액면|총\s*공모|공모\s*총액|모집\s*총액|발행\s*총액|총\s*발행"


class _TableRowParser(HTMLParser):
    """DART 원문 HTML에서 표의 행·셀 경계를 보존한다."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows: list[str] = []
        self.tables: list[list[list[str]]] = []
        self.table_contexts: list[str] = []
        self._outside_text: list[str] = []
        self._current_context = ""
        self._table_depth = 0
        self._table_rows: list[list[str]] | None = None
        self._row: list[tuple[str, int, int]] | None = None
        self._cell: list[str] | None = None
        self._cell_rowspan = 1
        self._cell_colspan = 1
        self._pending_rowspans: dict[int, tuple[str, int]] = {}

    def handle_starttag(self, tag: str, attrs):
        tag = tag.lower()
        if tag == "table":
            if self._table_depth == 0:
                self._table_rows = []
                self._current_context = re.sub(r"\s+", " ", "".join(self._outside_text)).strip()
                self._outside_text = []
            self._table_depth += 1
        elif self._table_depth and tag == "tr":
            self._row = []
        elif self._table_depth and tag in {"td", "th"} and self._row is not None:
            self._cell = []
            attributes = dict(attrs)
            self._cell_rowspan = self._positive_span(attributes.get("rowspan"))
            self._cell_colspan = self._positive_span(attributes.get("colspan"))

    def handle_endtag(self, tag: str):
        tag = tag.lower()
        if self._table_depth and tag in {"td", "th"} and self._cell is not None and self._row is not None:
            value = re.sub(r"\s+", " ", "".join(self._cell)).strip()
            self._row.append((value, self._cell_rowspan, self._cell_colspan))
            self._cell = None
        elif self._table_depth and tag == "tr" and self._row is not None:
            if self._row:
                expanded = self._expand_row(self._row)
                self.rows.append(" | ".join(expanded))
                if self._table_rows is not None:
                    self._table_rows.append(expanded)
            self._row = None
        elif tag == "table" and self._table_depth:
            self._table_depth -= 1
            if self._table_depth == 0 and self._table_rows:
                self.tables.append(self._table_rows)
                self.table_contexts.append(self._current_context)
                self._table_rows = None
                self._pending_rowspans = {}

    def handle_data(self, data: str):
        if self._cell is not None:
            self._cell.append(data)
        elif not self._table_depth:
            self._outside_text.append(data)

    @staticmethod
    def _positive_span(value: str | None) -> int:
        try:
            return max(1, int(value or 1))
        except (TypeError, ValueError):
            return 1

    def _expand_row(self, cells: list[tuple[str, int, int]]) -> list[str]:
        """rowspan/colspan과 빈 셀을 보존한 직사각형 행을 만든다."""
        row: list[str] = []
        column = 0

        def append_pending() -> None:
            nonlocal column
            while column in self._pending_rowspans:
                value, remaining = self._pending_rowspans[column]
                row.append(value)
                if remaining <= 1:
                    del self._pending_rowspans[column]
                else:
                    self._pending_rowspans[column] = (value, remaining - 1)
                column += 1

        for value, rowspan, colspan in cells:
            append_pending()
            for _ in range(colspan):
                row.append(value)
                if rowspan > 1:
                    self._pending_rowspans[column] = (value, rowspan - 1)
                column += 1
        append_pending()
        return row


class DARTCollector:
    """DART OpenAPI 수집기"""

    def __init__(self, api_key: str = DART_API_KEY):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "IPO-Research/1.0"})

    @property
    def is_configured(self) -> bool:
        """실제 OpenDART 인증키가 설정되었는지 반환한다."""
        return bool(self.api_key and self.api_key != "YOUR_DART_API_KEY")

    # ── 저수준 API 호출 ────────────────────────────────────────

    def _get(self, endpoint: str, params: dict) -> dict:
        """재시도 포함 GET 요청"""
        url = f"{DART_BASE_URL}/{endpoint}.json"
        params["crtfc_key"] = self.api_key

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.session.get(url, params=params, timeout=TIMEOUT)
                resp.raise_for_status()
                data = resp.json()
                if data.get("status") == "000":
                    return data
                # 013/020 = 조회된 데이터 없음 (정상 결측)
                if data.get("status") in {"013", "020"}:
                    return {"status": data.get("status"), "list": []}
                logger.warning("DART API status=%s msg=%s", data.get("status"), data.get("message"))
                return data
            except requests.RequestException as e:
                logger.warning("DART API attempt %d failed: %s", attempt, e)
                if attempt < MAX_RETRIES:
                    time.sleep(attempt * 2)
        return {}

    def get_document_text(self, rcept_no: str) -> str:
        """DART 원문 ZIP을 내려받아 분석 가능한 평문으로 변환한다.

        OpenDART의 ``document.xml``은 이름과 달리 JSON/XBRL API가 아닌 ZIP
        바이너리 응답이다. 공시별 XML/HTML 조각을 모두 읽어 합치므로, 신고서
        정정본처럼 여러 파일로 구성된 원문도 같은 파서로 처리할 수 있다.
        """
        if not self.is_configured:
            raise RuntimeError("DART_API_KEY가 설정되지 않았습니다.")

        url = f"{DART_BASE_URL}/document.xml"
        try:
            response = self.session.get(
                url,
                params={"crtfc_key": self.api_key, "rcept_no": rcept_no},
                timeout=TIMEOUT,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"DART 원문 다운로드 실패 ({rcept_no}): {exc}") from exc

        content = response.content
        if not zipfile.is_zipfile(BytesIO(content)):
            message = content.decode("utf-8", errors="ignore")[:300]
            raise RuntimeError(f"DART 원문 ZIP 응답이 아닙니다 ({rcept_no}): {message}")

        fragments: list[str] = []
        with zipfile.ZipFile(BytesIO(content)) as archive:
            for name in archive.namelist():
                if name.endswith("/") or name.lower().endswith((".jpg", ".jpeg", ".gif", ".png", ".pdf")):
                    continue
                raw = archive.read(name)
                for encoding in ("utf-8", "cp949", "euc-kr"):
                    try:
                        fragments.append(raw.decode(encoding))
                        break
                    except UnicodeDecodeError:
                        continue

        # 후속 파서가 표의 같은 행인지 판단할 수 있도록 HTML/XML 경계는
        # 보존한다. 텍스트가 필요한 파서는 각자 _normalize_text를 호출한다.
        return " ".join(fragments)

    # ── 공시 목록 수집 ─────────────────────────────────────────

    def get_ipo_disclosure_list(
        self,
        start_date: str,   # "20150101"
        end_date:   str,   # "20241231"
        pblntf_ty: str = "C",  # C = 발행공시
        pblntf_detail_ty: str = "C001",  # 증권신고서(지분증권)
    ) -> pd.DataFrame:
        """증권신고서(지분증권) 목록을 날짜 범위로 수집한다.

        법인 고유번호 없이 OpenDART 목록 API를 호출할 때는 조회 기간이
        3개월로 제한되므로 긴 기간을 90일 단위로 나눈다.
        """
        all_records = []
        cursor = pd.Timestamp(start_date)
        final_date = pd.Timestamp(end_date)

        while cursor <= final_date:
            chunk_end = min(cursor + pd.Timedelta(days=89), final_date)
            page = 1
            while True:
                data = self._get("list", {
                    "bgn_de": cursor.strftime("%Y%m%d"),
                    "end_de": chunk_end.strftime("%Y%m%d"),
                    "pblntf_ty": pblntf_ty,
                    "pblntf_detail_ty": pblntf_detail_ty,
                    "page_no": page,
                    "page_count": 100,
                })
                items = data.get("list", [])
                if not items:
                    break
                all_records.extend(items)
                total_page = int(data.get("total_page", 1))
                if page >= total_page:
                    break
                page += 1
                time.sleep(REQUEST_DELAY)
            cursor = chunk_end + pd.Timedelta(days=1)

        if not all_records:
            return pd.DataFrame()

        df = pd.DataFrame(all_records).drop_duplicates("rcept_no", keep="last")
        # 공모 관련 공시만 필터 (제목에 '증권신고서(지분증권)' 포함)
        mask = df["report_nm"].str.contains("증권신고서.*지분증권|지분증권.*증권신고서", na=False)
        df = df[mask].copy()
        df["rcept_dt"] = pd.to_datetime(df["rcept_dt"], format="%Y%m%d")
        logger.info("IPO 공시 수집 완료: %d건 (%s ~ %s)", len(df), start_date, end_date)
        return df.reset_index(drop=True)

    def get_company_disclosure_list(
        self,
        corp_code: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """기업별 공시 목록을 반환한다.

        수요예측 결과는 신고서 정정본 또는 별도 공시 안에 들어갈 수 있어,
        공시 제목만으로 단정하지 않고 이 목록을 후보 탐색에 사용한다.
        """
        records = []
        page = 1
        while True:
            data = self._get("list", {
                "corp_code": corp_code,
                "bgn_de": start_date,
                "end_de": end_date,
                "page_no": page,
                "page_count": 100,
            })
            items = data.get("list", [])
            if not items:
                break
            records.extend(items)
            if page >= int(data.get("total_page", 1)):
                break
            page += 1
            time.sleep(REQUEST_DELAY)

        if not records:
            return pd.DataFrame()
        result = pd.DataFrame(records)
        result["rcept_dt"] = pd.to_datetime(result["rcept_dt"], format="%Y%m%d", errors="coerce")
        return result.sort_values("rcept_dt").reset_index(drop=True)

    def find_demand_forecast_disclosure(
        self,
        corp_code: str,
        start_date: str,
        end_date: str,
    ) -> Optional[str]:
        """수요예측 숫자를 포함할 가능성이 가장 큰 공시 접수번호를 찾는다."""
        record = self.find_demand_forecast_disclosure_record(corp_code, start_date, end_date)
        return None if record is None else str(record["rcept_no"])

    def find_demand_forecast_disclosure_record(
        self,
        corp_code: str,
        start_date: str,
        end_date: str,
    ) -> Optional[dict]:
        """수요예측 후보의 접수번호와 공개 시각을 함께 반환한다."""
        records = self.find_demand_forecast_disclosure_records(corp_code, start_date, end_date)
        return records[0] if records else None

    def find_demand_forecast_disclosure_records(
        self,
        corp_code: str,
        start_date: str,
        end_date: str,
    ) -> list[dict]:
        """수요예측·확약 값이 있을 수 있는 공시 계보 후보를 순서대로 반환한다.

        기관 수요예측 결과는 독립 제목 공시, 발행조건확정 신고서, 투자설명서
        중 어느 곳에나 들어갈 수 있다. 단일 제목을 "정답"으로 고르면 한 ZIP
        실패나 문서 형식 차이로 회사 전체의 값을 잃으므로, 제목 근거와 접수일을
        보존한 후보 목록을 반환한다.
        """
        disclosures = self.get_company_disclosure_list(corp_code, start_date, end_date)
        if disclosures.empty:
            return []

        title = disclosures["report_nm"].fillna("").astype(str)
        eligible = (
            title.str.contains("수요예측|발행조건확정|투자설명서|증권신고서", regex=True)
        )
        candidates = disclosures.loc[eligible].copy()
        if candidates.empty:
            return []
        candidate_title = candidates["report_nm"].fillna("").astype(str)
        candidates["candidate_score"] = (
            candidate_title.str.contains("수요예측", regex=False).astype(int) * 100
            + candidate_title.str.contains("발행조건확정", regex=False).astype(int) * 80
            + candidate_title.str.contains("기관투자자", regex=False).astype(int) * 40
            + candidate_title.str.contains("투자설명서", regex=False).astype(int) * 30
            + candidate_title.str.contains("정정", regex=False).astype(int) * 10
            + candidate_title.str.contains("증권신고서", regex=False).astype(int) * 5
        )
        candidates = candidates.sort_values(
            ["candidate_score", "rcept_dt", "rcept_no"], ascending=[False, False, False]
        ).drop_duplicates("rcept_no", keep="first")
        return [{
            "rcept_no": str(row.rcept_no),
            "rcept_dt": row.rcept_dt,
            "report_nm": row.report_nm,
            "candidate_score": int(row.candidate_score),
        } for row in candidates.itertuples(index=False)]

    # ── 수요예측 결과 파싱 ─────────────────────────────────────

    def get_demand_forecast(self, corp_code: str, rcept_no: str) -> dict:
        """
        수요예측 결과 공시에서 기관 경쟁률 및 의무보유확약 비율을 추출.

        DART 공시 XML에서 추출하는 필드:
          - 기관투자자 수요예측 참여 현황 (경쟁률)
          - 의무보유확약 기간별 비율

        실제 구현 시: DART XML API를 통해 공시 원문을 가져온 후
        특정 테이블 패턴을 정규식으로 파싱한다.
        """
        return self._parse_demand_forecast_html(self.get_document_text(rcept_no), corp_code)

    def _parse_demand_forecast_html(self, html: str, corp_code: str) -> dict:
        """
        수요예측 결과 HTML에서 경쟁률·확약 데이터 추출.

        DART 수요예측 결과 공시의 표준 테이블 구조:
        ┌─────────────────────┬──────────┬──────────────────────┐
        │ 구분                │ 건수     │ 신청주식수           │
        ├─────────────────────┼──────────┼──────────────────────┤
        │ 합계                │ XXX      │ XXX,XXX,XXX          │
        │ 확약없음            │ XXX      │ XXX,XXX,XXX          │
        │ 15일                │ XXX      │ XXX,XXX,XXX          │
        │ 1개월               │ XXX      │ XXX,XXX,XXX          │
        │ 3개월               │ XXX      │ XXX,XXX,XXX          │
        │ 6개월               │ XXX      │ XXX,XXX,XXX          │
        └─────────────────────┴──────────┴──────────────────────┘
        """
        result = {
            "corp_code":              corp_code,
            "institutional_demand_ratio": None,
            "demand_offering_price":  None,
            "demand_offering_price_context": None,
            "lockup_commitment_ratio": None,
            "lockup_6m_ratio":        None,
            "lockup_3m_ratio":        None,
            "lockup_1m_ratio":        None,
            "lockup_15d_ratio":       None,
            "lockup_none_ratio":      None,
            "institutional_demand_parse_method": None,
            "institutional_demand_evidence": None,
            "institutional_demand_rule_id": None,
            "institutional_demand_parser_validation_status": "value_not_found",
            "institutional_demand_structured_evidence": None,
            "institutional_demand_rejection_reason": None,
            "lockup_parse_method": None,
            "lockup_parse_evidence": None,
            "lockup_rule_id": None,
            "lockup_parser_validation_status": "value_not_found",
            "lockup_structured_evidence": None,
            "lockup_rejection_reason": None,
            "parse_success":          False,
        }

        if not html:
            return result
        text = self._normalize_text(html)

        demand_price = self._extract_offering_price_details(text)
        result["demand_offering_price"] = demand_price["offering_price"]
        result["demand_offering_price_context"] = demand_price["offering_price_audit_context"]

        # 기관별 열과 합계 열이 함께 있는 표에서는 헤더가 가리키는 합계 셀만
        # 사용한다. 마지막 숫자라는 위치 추정은 주석·재심의 기준을 오인할 수 있다.
        table_parser = _TableRowParser()
        table_parser.feed(html)
        table_parser.close()
        for table, heading in zip(table_parser.tables, table_parser.table_contexts):
            parsed_table = self._extract_aggregate_demand_ratio(table, heading)
            if parsed_table is not None:
                result.update(parsed_table)
                break

        if result["institutional_demand_ratio"] is None and re.search(
            r"(?:기관\s*(?:투자자)?|수요\s*예측).*?[0-9][0-9,]*(?:\.[0-9]+)?\s*(?::|：|대)\s*1\b",
            text,
        ):
            result["institutional_demand_parser_validation_status"] = "candidate_rejected"
            result["institutional_demand_rejection_reason"] = (
                "risk_or_planned_threshold_context"
                if re.search(r"재심의|예정|이하|리스크|기준", text)
                else "aggregate_scope_or_table_alignment_not_verified"
            )

        # 표가 아닌 문장에서는 기관 수요예측 라벨과 비율이 직접 연결된 경우만 허용한다.
        demand_blocks = self._extract_table_rows(html) + self._split_sentences(text)
        for block in demand_blocks if result["institutional_demand_ratio"] is None else []:
            normalized = re.sub(r"\s+", " ", block).strip()
            if re.search(r"비례\s*배정|일반\s*청약|개인\s*청약|재심의|예정|이하|리스크", normalized):
                continue
            if re.search(r"신규\s*상장\s*기업|기업\s*수|평균\s*공모|공모\s*규모", normalized):
                continue
            if len(re.findall(r"[0-9][0-9,]*(?:\.[0-9]+)?\s*(?::|：|대)\s*1\b", normalized)) != 1:
                continue
            match = re.search(
                r"기관\s*(?:투자자)?\s*(?:수요\s*예측\s*)?(?:유효\s*)?경쟁률"
                r"\s*(?:은|는|:|：)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(?::|：|대)\s*1\b",
                normalized,
            )
            if match:
                result["institutional_demand_ratio"] = float(match.group(1).replace(",", ""))
                result["institutional_demand_parse_method"] = "demand_ratio_same_table_row_or_sentence"
                result["institutional_demand_evidence"] = normalized[:300]
                result["institutional_demand_rule_id"] = "DART_DEMAND_DIRECT_LABEL_V1"
                result["institutional_demand_parser_validation_status"] = "structurally_verified"
                result["institutional_demand_structured_evidence"] = json.dumps({
                    "kind": "direct_label", "text": normalized[:1000],
                }, ensure_ascii=False)
                break

        lockup = self._extract_lockup_ratios_from_tables(html)
        for key in (
            "lockup_commitment_ratio", "lockup_6m_ratio", "lockup_3m_ratio", "lockup_1m_ratio",
            "lockup_15d_ratio", "lockup_none_ratio",
        ):
            result[key] = lockup.get(key)
        result["lockup_parse_method"] = lockup.get("parse_method")
        result["lockup_parse_evidence"] = lockup.get("evidence")
        result["lockup_rule_id"] = lockup.get("rule_id")
        result["lockup_parser_validation_status"] = lockup.get("parser_validation_status", "value_not_found")
        result["lockup_structured_evidence"] = lockup.get("structured_evidence")
        result["lockup_rejection_reason"] = lockup.get("rejection_reason")

        # 파싱 성공 여부만 플래그 설정 (실제 비율 계산은 수집된 데이터로)
        if result["institutional_demand_ratio"] is not None or any(
            result[field] is not None for field in (
                "lockup_commitment_ratio",
            )
        ):
            result["parse_success"] = True

        return result

    # ── 재무제표 수집 ──────────────────────────────────────────

    def get_financial_statements(
        self,
        corp_code: str,
        year:      int,
        report_code: str = "11011",  # 사업보고서
    ) -> pd.DataFrame:
        """
        DART 재무제표 API에서 손익계산서 + 재무상태표 핵심 항목 수집.

        report_code:
          11011 = 사업보고서 (연간)
          11012 = 반기보고서
          11013 = 1분기보고서
          11014 = 3분기보고서
        """
        data = self._get("fnlttSinglAcntAll", {
            "corp_code":   corp_code,
            "bsns_year":   str(year),
            "reprt_code":  report_code,
            "fs_div":      "CFS",  # CFS=연결, OFS=별도
        })

        items = data.get("list", [])
        if not items:
            # 연결 없으면 별도 재무제표 시도
            data = self._get("fnlttSinglAcntAll", {
                "corp_code":  corp_code,
                "bsns_year":  str(year),
                "reprt_code": report_code,
                "fs_div":     "OFS",
            })
            items = data.get("list", [])

        if not items:
            return pd.DataFrame()

        df = pd.DataFrame(items)
        df["year"] = year

        # 필요한 계정과목만 추출
        target_accounts = {
            "ifrs-full_Revenue":                    "revenue",
            "ifrs-full_OperatingIncomeLoss":        "operating_income",
            "ifrs-full_ProfitLoss":                 "net_income",
            "ifrs-full_Assets":                     "total_assets",
            "ifrs-full_Liabilities":                "total_liabilities",
            "ifrs-full_Equity":                     "equity",
            "ifrs-full_BasicEarningsLossPerShare":  "eps",
        }

        filtered = df[df["account_id"].isin(target_accounts.keys())].copy()
        filtered["account_name_en"] = filtered["account_id"].map(target_accounts)
        filtered["amount"] = pd.to_numeric(
            filtered["thstrm_amount"].astype(str).str.replace(",", ""), errors="coerce"
        )
        return filtered[["year", "account_name_en", "amount"]].dropna()

    def get_equity_offering_prices(
        self,
        corp_code: str,
        start_date: str,
        end_date: str,
    ) -> list[dict]:
        """지분증권 구조화 API의 모집(매출)가액을 접수번호별로 반환한다.

        ``estkRs``의 ``slprc``는 원문 표를 정규식으로 읽는 값이 아니라
        OpenDART가 구조화해 제공하는 모집(매출)가액이다. 원문 추출값과
        접수번호가 같은 경우에만 서로 대조한다.
        """
        data = self._get("estkRs", {
            "corp_code": corp_code,
            "bgn_de": start_date,
            "end_de": end_date,
        })
        records: list[dict] = []
        for item in self._walk_dicts(data):
            if "slprc" not in item:
                continue
            price = self._parse_money_value(item.get("slprc"))
            if price is None:
                continue
            records.append({
                "rcept_no": str(item.get("rcept_no", "")),
                "offering_price": price,
                "security_type": item.get("stksen"),
            })
        return records

    # ── 공모가 밴드 수집 ───────────────────────────────────────

    def get_offering_info(self, rcept_no: str) -> dict:
        """
        증권신고서에서 공모 구조 정보 추출.

        수집 항목:
          - 희망공모가 밴드 (하단, 상단)
          - 확정 공모가
          - 공모 주식수 (신주 / 구주 분리)
          - 상장 예정일
          - 주관사명
          - 최대주주 보호예수 기간
        """
        # 최종 발행조건 신고서에는 공모 구조뿐 아니라 대표주관사가 취합한
        # 기관 수요예측·의무보유확약 표가 함께 실릴 수 있다. 원문을 한 번만
        # 내려받아 두 파서를 적용해, 같은 접수번호의 통합 결과인지 검증한다.
        document_text = self.get_document_text(rcept_no)
        result = self._parse_offering_html(document_text, rcept_no)
        demand = self._parse_demand_forecast_html(document_text, "")
        result.update({
            key: value
            for key, value in demand.items()
            if key not in {"corp_code", "parse_success"}
        })
        # 공모 구조와 수요예측은 같은 원문을 읽지만, 어느 한쪽이 없다고 다른
        # 쪽의 파싱 성공 상태를 덮어쓰면 안 된다.
        result["dart_final_terms_demand_parse_success"] = demand["parse_success"]
        result["dart_final_terms_demand_parser_version"] = DEMAND_PARSER_VERSION
        return result

    def _parse_offering_html(self, html: str, rcept_no: str) -> dict:
        """공모 정보 HTML 파싱"""
        result = {
            "rcept_no":           rcept_no,
            "price_band_low":     None,
            "price_band_high":    None,
            "offering_price":     None,
            "offering_price_extracted_amount": None,
            "offering_price_review_status": "missing",
            "offering_price_finality": "unknown",
            "offering_price_parse_method": None,
            "offering_price_audit_context": None,
            "offering_price_range_warning": False,
            "new_shares":         None,
            "new_shares_parse_method": None,
            "new_shares_parse_evidence": None,
            "new_shares_parser_validation_status": "value_not_found",
            "secondary_shares":   None,
            "secondary_shares_parse_method": None,
            "secondary_shares_parse_evidence": None,
            "secondary_shares_parser_validation_status": "value_not_found",
            "total_post_listing_shares": None,
            "total_post_listing_shares_parse_method": None,
            "total_post_listing_shares_parse_evidence": None,
            "total_post_listing_shares_parser_validation_status": "value_not_found",
            "public_float_shares": None,
            "public_float_ratio_disclosed": None,
            "public_float_parse_method": None,
            "public_float_parse_evidence": None,
            "lead_underwriter":   None,
            "listing_date":       None,
            "major_shareholder_lockup_months": None,
            "risk_factor_count":  None,
            "parse_success":      False,
        }

        if not html:
            return result
        text = self._normalize_text(html)

        # 희망공모가 밴드 패턴
        band_pattern = (
            r"(?:희망\s*공모가(?:액)?|공모\s*희망가(?:액)?)\s*(?:밴드)?"
            r"\s*[:：(]?\s*([0-9][0-9,]*)\s*(?:원|KRW)?\s*[~～]"
            r"\s*([0-9][0-9,]*)\s*(?:원|KRW)"
        )
        bands = {(int(m[1].replace(",", "")), int(m[2].replace(",", "")))
                 for m in re.finditer(band_pattern, text)}
        if len(bands) == 1:
            low, high = next(iter(bands))
            if 0 < low < high:
                result.update(price_band_low=low, price_band_high=high,
                              price_band_validation_status="direct_currency_band")
        elif len(bands) > 1:
            result["price_band_validation_status"] = "conflicting_bands_review_required"

        price_details = self._extract_offering_price_details(
            text,
            table_rows=self._extract_table_rows(html),
        )
        result.update(price_details)

        # 공모 구조는 표 행 또는 라벨과 수량이 직접 연결된 문구만 승인한다.
        # 라벨 뒤 120자를 탐색하던 기존 규칙은 다른 표의 숫자를
        # 신주·구주 수량으로 오인할 수 있어 학습 승인에 사용하지 않는다.
        tables = self._extract_table_cells(html)
        result.update(self._extract_share_details(
            tables, text, "new_shares", r"신주\s*(?:모집|발행)(?:\s*주식수|\s*수량)?"
        ))
        result.update(self._extract_share_details(
            tables, text, "secondary_shares", r"(?:구주\s*매출|매출\s*주식수)(?:\s*주식수|\s*수량)?"
        ))
        for label, present, absent in (
            ("신주모집", "new_shares", "secondary_shares"),
            ("구주매출", "secondary_shares", "new_shares"),
        ):
            match = re.search(
                rf"{label}\s*([0-9][0-9,]*)\s*주\s*\(\s*공모주식(?:수)?의\s*100(?:\.0+)?\s*%\s*\)",
                text,
            )
            if match and result[present] == int(match[1].replace(",", "")) and result[absent] is None:
                result[absent] = 0
                result[f"{absent}_parser_validation_status"] = "structurally_verified"
                result[f"{absent}_parse_method"] = "explicit_all_offered_shares_v1"
                result[f"{absent}_parse_evidence"] = match[0]
            exclusive = re.search(rf"공모는\s*100\s*%\s*{label}(?:으)?로\s*진행", text)
            if exclusive and result[present] is not None and result[absent] is None:
                result[absent] = 0
                result[f"{absent}_parser_validation_status"] = "structurally_verified"
                result[f"{absent}_parse_method"] = "explicit_exclusive_offering_v1"
                result[f"{absent}_parse_evidence"] = exclusive[0]
        result.update(self._extract_share_details(
            tables,
            text,
            "total_post_listing_shares",
            r"(?:상장\s*예정\s*주식수|상장\s*후\s*총\s*발행주식수|발행\s*주식\s*총수)",
        ))
        public_float = self._extract_public_float_details(text)
        result.update(public_float)
        float_candidates = []
        for table in tables:
            if not table or not any("유통가능" in re.sub(r"\s+", "", " ".join(row)) for row in table):
                continue
            for row_index, row in enumerate(table):
                if row and re.fullmatch(r"합계|총계", re.sub(r"\s+", "", row[0])):
                    headers = table[:row_index]
                    columns = [c for c in range(len(row))
                               if any(c < len(h) and re.fullmatch(r"유통가능(?:물량|주식수)?", re.sub(r"\s+", "", h[c])) for h in headers)
                               and any(c < len(h) and re.fullmatch(r"(?:지분율|비율)(?:\(%\))?", re.sub(r"\s+", "", h[c])) for h in headers)]
                    if len(columns) == 1:
                        match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*%\s*", row[columns[0]])
                        if match and 0 <= float(match[1]) <= 100:
                            float_candidates.append((float(match[1]) / 100, table))
                if not any(re.fullmatch(r"유통가능물량소계", re.sub(r"\s+", "", cell)) for cell in row):
                    continue
                headers = table[:row_index]
                columns = [c for c in range(len(row))
                           if any(c < len(h) and re.fullmatch(r"공모후(?:기준)?", re.sub(r"\s+", "", h[c])) for h in headers)
                           and any(c < len(h) and re.fullmatch(r"(?:지분율|비율)\(%\)", re.sub(r"\s+", "", h[c])) for h in headers)]
                if len(columns) == 1:
                    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*%\s*", row[columns[0]])
                    if match and 0 <= float(match[1]) <= 100:
                        float_candidates.append((float(match[1]) / 100, table))
            for row in table:
                if not row or not re.fullmatch(r"상장(?:일|직후)유통가능", re.sub(r"\s+", "", row[0])):
                    continue
                ratios = [float(m[1]) / 100 for cell in row[1:]
                          if (m := re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*%\s*", cell))]
                if len(ratios) == 1 and 0 <= ratios[0] <= 1:
                    float_candidates.append((ratios[0], table))
        unique_ratios = {value for value, _ in float_candidates}
        if len(unique_ratios) == 1:
            result.update(public_float_ratio_disclosed=float_candidates[0][0],
                          public_float_parse_method="disclosed_public_float_ratio_direct_context",
                          public_float_parse_evidence=self._table_evidence(float_candidates[0][1]))
        elif len(unique_ratios) > 1:
            result.update(public_float_shares=None, public_float_ratio_disclosed=None,
                          public_float_parse_method="conflicting_float_ratios_review_required",
                          public_float_parse_evidence=json.dumps(sorted(unique_ratios)))

        # 대표 주관사
        underwriter_pattern = (
            r"(?:대표\s*주관\s*회사|대표주관회사|공동주관회사|주관회사|인수회사)"
            r"[^가-힣A-Za-z0-9]{0,20}([가-힣A-Za-z0-9&().\s·]+?증권)"
        )
        m = re.search(underwriter_pattern, text)
        if m:
            result["lead_underwriter"] = re.sub(r"\s+", " ", m.group(1)).strip()

        # 최대주주 의무보유기간
        lockup_pattern = r"최대주주.{0,120}?(\d+\s*년\s*\d+\s*개월|\d+\s*년|\d+\s*개월)"
        lockup_matches = [m.group(1) for m in re.finditer(lockup_pattern, text)]
        if lockup_matches:
            result["major_shareholder_lockup_months"] = max(
                self._parse_lockup_months(item) for item in lockup_matches
            )

        # 투자위험요소 항목 수
        risk_match = re.search(r"(?:투자위험요소|위험요소)(.{0,20000})", text)
        if risk_match:
            markers = re.findall(r"(?:^|\s)(?:[가-하]\.|[0-9]{1,2}\.|[①-⑳])", risk_match.group(1))
            if markers:
                result["risk_factor_count"] = min(len(markers), 60)

        # 상장일
        date_pattern = r"상장\s*예정일[^0-9]*(\d{4})\s*[.\-년]\s*(\d{1,2})\s*[.\-월]\s*(\d{1,2})"
        m = re.search(date_pattern, text)
        if m:
            y, mo, d = m.groups()
            try:
                result["listing_date"] = date(int(y), int(mo), int(d)).isoformat()
            except ValueError:
                pass

        result["parse_success"] = any(
            result[key] is not None
            for key in [
                "price_band_low",
                "price_band_high",
                "offering_price",
                "new_shares",
                "secondary_shares",
                "total_post_listing_shares",
                "public_float_shares",
                "public_float_ratio_disclosed",
                "lead_underwriter",
                "listing_date",
                "major_shareholder_lockup_months",
                "risk_factor_count",
            ]
        )

        return result

    @staticmethod
    def _normalize_text(raw_html: str) -> str:
        text = re.sub(r"<[^>]+>", " ", raw_html)
        text = html_unescape(text)
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def _extract_public_float_details(cls, text: str) -> dict:
        """상장 직후 실제 유통가능 물량을 공시 문맥에서만 읽는다.

        신주와 구주매출은 공모 물량일 뿐 기존주주의 즉시 유통 물량을 포함하지
        않는다. 따라서 두 값을 더해 유통가능 비율로 추정하지 않고, 신고서가
        직접 밝힌 주식 수 또는 전체 주식 대비 비율만 사용한다.
        """
        result = {
            "public_float_shares": None,
            "public_float_ratio_disclosed": None,
            "public_float_parse_method": None,
            "public_float_parse_evidence": None,
        }
        label = r"상장\s*(?:직후\s*)?유통\s*가능\s*(?:주식\s*)?(?:수|물량)|유통\s*가능\s*(?:주식\s*)?(?:수|물량)"
        # 문서에 유통가능물량을 설명한 뒤 다른 문단의 자기주식·일반투자자
        # 수량이 나오는 사례가 있다. 따라서 라벨 뒤 임의의 120자를 탐색하지
        # 않고, 라벨과 수량이 바로 이어지거나 "보호예수 제외"라는 계산식이
        # 같은 문맥에 직접 있는 경우만 승인한다.
        share_patterns = (
            rf"(?:{label})\s*(?:은|는|:|=)?\s*(?:공모\s*후\s*주식수\s*기준)?\s*([0-9][0-9,]*)\s*주",
            rf"(?:{label})[^.&]{{0,80}}보호예수\s*및\s*매도금지물량\s*을?\s*제외한\s*([0-9][0-9,]*)\s*주",
        )
        for pattern in share_patterns:
            share_match = re.search(pattern, text)
            if not share_match:
                continue
            shares = cls._parse_int(share_match.group(1))
            if shares is not None and shares > 0:
                result.update({
                    "public_float_shares": shares,
                    "public_float_parse_method": "disclosed_public_float_shares_direct_context",
                    "public_float_parse_evidence": share_match.group(0)[:300],
                })
                return result

        ratio_match = re.search(
            rf"(?:{label})\s*(?:은|는|:|=)?\s*(?:전체\s*(?:상장\s*)?주식\s*대비)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*%", text
        )
        if ratio_match:
            ratio = float(ratio_match.group(1).replace(",", "")) / 100
            if 0 <= ratio <= 1:
                result.update({
                    "public_float_ratio_disclosed": round(ratio, 6),
                    "public_float_parse_method": "disclosed_public_float_ratio_direct_context",
                    "public_float_parse_evidence": ratio_match.group(0)[:300],
                })
        return result

    @staticmethod
    def _extract_share_after(text: str, label_pattern: str) -> Optional[int]:
        pattern = rf"(?:{label_pattern})[^0-9]{{0,120}}([0-9,]+)\s*주"
        m = re.search(pattern, text)
        if not m:
            return None
        return DARTCollector._parse_int(m.group(1))

    @classmethod
    def _extract_share_details(
        cls,
        tables: list[list[list[str]]],
        text: str,
        field: str,
        label_pattern: str,
    ) -> dict:
        """공모 주식 수량을 명시적인 표 행/문구에서만 추출한다."""
        result = {
            field: None,
            f"{field}_parse_method": None,
            f"{field}_parse_evidence": None,
            f"{field}_parser_validation_status": "value_not_found",
        }
        value_pattern = re.compile(r"\s*([0-9][0-9,]*)\s*주\s*")
        label = re.compile(label_pattern)

        for table_index, table in enumerate(tables):
            for row_index, row in enumerate(table):
                for column_index, cell in enumerate(row):
                    if not label.fullmatch(cell.strip()):
                        continue
                    candidates: list[tuple[int, int]] = []
                    for value_column, candidate in enumerate(row[column_index + 1:], column_index + 1):
                        match = value_pattern.fullmatch(candidate)
                        if match:
                            candidates.append((value_column, int(match.group(1).replace(",", ""))))
                    if len(candidates) == 1:
                        value_column, value = candidates[0]
                        result.update({
                            field: value,
                            f"{field}_parse_method": "direct_table_row_v1",
                            f"{field}_parse_evidence": cls._table_evidence(
                                table, selected_row=row_index, label_column=column_index,
                                value_column=value_column, table_index=table_index,
                            ),
                            f"{field}_parser_validation_status": "structurally_verified",
                        })
                        return result

        direct = re.search(
            rf"(?:{label_pattern})\s*(?:은|는|:|=)?\s*([0-9][0-9,]*)\s*주",
            text,
        )
        if direct:
            result.update({
                field: int(direct.group(1).replace(",", "")),
                f"{field}_parse_method": "direct_label_text_v1",
                f"{field}_parse_evidence": direct.group(0),
                f"{field}_parser_validation_status": "structurally_verified",
            })
        return result

    @staticmethod
    def _extract_table_rows(raw_html: str) -> list[str]:
        parser = _TableRowParser()
        try:
            parser.feed(raw_html)
            parser.close()
        except Exception as exc:
            logger.debug("DART 표 행 파싱을 건너뜁니다: %s", exc)
            return []
        return parser.rows

    @staticmethod
    def _extract_table_cells(raw_html: str) -> list[list[list[str]]]:
        parser = _TableRowParser()
        try:
            parser.feed(raw_html)
            parser.close()
        except Exception as exc:
            logger.debug("DART 표 셀 파싱을 건너뜁니다: %s", exc)
            return []
        return parser.tables

    @staticmethod
    def _table_evidence(table: list[list[str]], **metadata) -> str:
        """재감사가 가능하도록 선택 셀과 주변 표 구조를 함께 직렬화한다."""
        # 중간에서 자른 JSON은 감사 시 다시 읽을 수 없다. Parquet 원장에는
        # 선택 좌표와 표 격자를 완전한 JSON으로 보존한다.
        return json.dumps({"table": table, **metadata}, ensure_ascii=False)

    @classmethod
    def _extract_aggregate_demand_ratio(cls, table: list[list[str]], heading: str = "") -> Optional[dict]:
        """기관 수요예측 표에서 명시적인 전체/합계 셀의 경쟁률만 승인한다."""
        if not table:
            return None
        table_text = " ".join(" ".join(row) for row in table)
        # A two-row summary can carry its institutional scope in the immediate heading.
        if len(table) == 2 and re.search(
            r"수요예측\s*결과\s*[①1.\s]*기관투자자\s*수요예측\s*참여내역\s*$", heading
        ):
            headers = [re.sub(r"\s+", "", c) for c in table[0]]
            if headers == ["참여건수(건)", "신청수량(주)", "단순경쟁률"] and len(table[1]) == 3:
                cells = [re.fullmatch(r"[0-9][0-9,]*(?:\.[0-9]+)?", c.strip()) for c in table[1]]
                if all(cells):
                    return {
                        "institutional_demand_ratio": float(table[1][2].replace(",", "")),
                        "institutional_demand_parse_method": "demand_ratio_institutional_summary",
                        "institutional_demand_evidence": " | ".join(table[1]),
                        "institutional_demand_rule_id": "DART_DEMAND_INSTITUTIONAL_SUMMARY_V1",
                        "institutional_demand_parser_validation_status": "structurally_verified",
                        "institutional_demand_structured_evidence": cls._table_evidence(
                            table, heading=heading, selected_row=1, selected_column=2,
                            scope="institutional_demand_participation_summary",
                        ),
                    }
        if not re.search(r"기관\s*(?:투자자)?|수요\s*예측", table_text):
            return None
        if re.search(
            r"신규\s*상장\s*기업|기업\s*수|평균\s*공모|공모\s*규모|재심의|예정|"
            r"일반\s*청약|개인\s*청약|비례\s*배정",
            table_text,
        ):
            return None

        # The competition-rate row supplies the unit even when cells omit ':1'.
        ratio_pattern = re.compile(r"([0-9][0-9,]*(?:\.[0-9]+)?)(?:\s*(?::|：|대)\s*1)?")
        ratio_rows = [
            (index, row) for index, row in enumerate(table)
            if re.search(r"(?:유효\s*|단순\s*)?경쟁률", " ".join(row))
        ]
        for row_index, row in ratio_rows:
            # 세로형 표: 합계/전체 행 자체에 경쟁률이 직접 기재된다.
            row_text = " | ".join(row)
            if re.search(r"합\s*계|총\s*계|전체", row_text):
                matches = [(index, ratio_pattern.fullmatch(cell.strip())) for index, cell in enumerate(row)]
                values = [(index, match) for index, match in matches if match]
                if len(values) == 1:
                    column_index, match = values[0]
                    return {
                        "institutional_demand_ratio": float(match.group(1).replace(",", "")),
                        "institutional_demand_parse_method": "demand_ratio_explicit_total_row",
                        "institutional_demand_evidence": row_text[:500],
                        "institutional_demand_rule_id": "DART_DEMAND_TOTAL_ROW_V1",
                        "institutional_demand_parser_validation_status": "structurally_verified",
                        "institutional_demand_structured_evidence": cls._table_evidence(
                            table, selected_row=row_index, selected_column=column_index,
                            scope="aggregate_total_row",
                        ),
                    }

            # 가로형 표: 경쟁률 행보다 앞선 헤더에서 합계/전체 열을 찾아 같은 열만 읽는다.
            header_rows = table[:row_index]
            aggregate_columns = {
                column_index
                for header in header_rows
                for column_index, cell in enumerate(header)
                if re.fullmatch(r"\s*(?:총\s*)?합\s*계\s*|\s*전\s*체\s*", cell)
            }
            valid = []
            for column_index in aggregate_columns:
                if column_index >= len(row):
                    continue
                match = ratio_pattern.fullmatch(row[column_index].strip())
                if match:
                    valid.append((column_index, match))
            if len(valid) != 1:
                continue
            column_index, match = valid[0]
            return {
                "institutional_demand_ratio": float(match.group(1).replace(",", "")),
                "institutional_demand_parse_method": "demand_ratio_explicit_total_column",
                "institutional_demand_evidence": row_text[:500],
                "institutional_demand_rule_id": "DART_DEMAND_TOTAL_COLUMN_V1",
                "institutional_demand_parser_validation_status": "structurally_verified",
                "institutional_demand_structured_evidence": cls._table_evidence(
                    table, selected_row=row_index, selected_column=column_index,
                    scope="aggregate_total_column",
                ),
            }
        return None

    @classmethod
    def _extract_lockup_ratios_from_tables(cls, raw_html: str) -> dict:
        """의무보유확약 표의 비율 열 또는 신청주식수 열만 사용한다.

        기존의 "6개월 뒤 첫 숫자" 방식은 다른 표의 번호·건수까지 가져올 수
        있었다. 이제 확약 표라는 문맥과 같은 행/열의 백분율 또는 신청주식수
        합계를 모두 확인할 때만 값을 만든다.
        """
        result = {
            "lockup_commitment_ratio": None,
            "lockup_6m_ratio": None, "lockup_3m_ratio": None,
            "lockup_1m_ratio": None, "lockup_15d_ratio": None,
            "lockup_none_ratio": None, "parse_method": None, "evidence": None,
            "rule_id": None, "parser_validation_status": "value_not_found",
            "structured_evidence": None,
            "rejection_reason": None,
        }
        label_map = (
            (r"6\s*개월", "lockup_6m_ratio"),
            (r"3\s*개월", "lockup_3m_ratio"),
            (r"1\s*개월", "lockup_1m_ratio"),
            (r"15\s*일", "lockup_15d_ratio"),
            (r"확약\s*없음|미확약", "lockup_none_ratio"),
        )
        normalized_document = cls._normalize_text(raw_html)
        direct_pattern = re.compile(
            r"의무\s*보유\s*확약\s*(?:비율|률)?\s*(?:은|는|:|：)?\s*"
            r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*%",
            flags=re.IGNORECASE,
        )
        # 평문 직접값은 기관 수요예측 결과와 확약 라벨이 같은 문장/표 행에
        # 함께 있을 때만 승인한다. 주변 250자 탐색은 다른 보호예수 표를 섞는다.
        direct_blocks = cls._extract_table_rows(raw_html) + cls._split_sentences(normalized_document)
        for block in direct_blocks:
            normalized = re.sub(r"\s+", " ", block).strip()
            direct_total = direct_pattern.search(normalized)
            if not direct_total:
                continue
            if not re.search(r"기관\s*(?:투자자)?", normalized) or not re.search(r"수요\s*예측", normalized):
                continue
            if re.search(r"최대\s*주주|기존\s*주주|임원|보유\s*주식", normalized):
                continue
            return {
                **result,
                "lockup_commitment_ratio": round(
                    float(direct_total.group(1).replace(",", "")) / 100, 6
                ),
                "parse_method": "lockup_direct_total_ratio",
                "evidence": normalized[:500],
                "rule_id": "DART_LOCKUP_DIRECT_AGGREGATE_V1",
                "parser_validation_status": "structurally_verified",
                "structured_evidence": json.dumps({
                    "kind": "direct_aggregate_label", "text": normalized[:1000],
                }, ensure_ascii=False),
            }
        table_parser = _TableRowParser()
        table_parser.feed(raw_html)
        table_parser.close()
        application_totals = set()
        for summary in table_parser.tables:
            if cls._extract_aggregate_demand_ratio(summary) is None:
                continue
            for i, row in enumerate(summary):
                if not row or re.sub(r"\s+", "", row[0]) not in ("수량", "신청수량"):
                    continue
                columns = {c for header in summary[:i] for c, cell in enumerate(header)
                           if re.fullmatch(r"합계|전체", re.sub(r"\s+", "", cell))}
                if len(columns) == 1:
                    c = next(iter(columns))
                    if c < len(row) and re.fullmatch(r"[0-9][0-9,]*", row[c]):
                        application_totals.add(int(row[c].replace(",", "")))
        application_total = next(iter(application_totals)) if len(application_totals) == 1 else None
        for table, context in zip(table_parser.tables, table_parser.table_contexts):
            table_text = " ".join(" ".join(row) for row in table)
            # Older reports put the institutional scope in a separate heading.
            if re.search(r"의무\s*보유\s*확약\s*(?:기관수\s*및\s*신청수량|신청내역)\s*$", context):
                header = table[0] if table else []
                quantity_columns = [i for i, cell in enumerate(header)
                                    if re.fullmatch(r"\s*(?:참여|신청)\s*수량\s*\((?:단위\s*:\s*)?주\)\s*", cell)]
                totals = [row for row in table if row and
                          re.fullmatch(r"\s*총\s*수량\s*대비\s*비율\s*(?:\(%\))?\s*", row[0])]
                if len(quantity_columns) == 1 and len(totals) == 1:
                    column = quantity_columns[0]
                    row = totals[0]
                    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*%\s*", row[column]) if column < len(row) else None
                    if match and 0 <= float(match[1]) <= 100:
                        result.update({
                            "lockup_commitment_ratio": float(match[1]) / 100,
                            "parse_method": "lockup_explicit_quantity_percentage",
                            "rule_id": "DART_LOCKUP_TOTAL_QUANTITY_PERCENT_V1",
                            "parser_validation_status": "structurally_verified",
                            "evidence": " | ".join(row),
                            "structured_evidence": cls._table_evidence(
                                table, heading=context, selected_column=column,
                                scope="institutional_application_quantity",
                            ),
                        })
                        return result
            if not re.search(r"기관\s*(?:투자자)?", table_text):
                continue
            aggregate_shares = cls._extract_aggregate_lockup_shares(table, application_total)
            if aggregate_shares is not None:
                result.update(aggregate_shares)
                return result
            if not re.search(r"의무\s*보유\s*확약|보유\s*확약|확약\s*기간", table_text):
                continue
            if re.search(r"최대\s*주주|기존\s*주주|주주\s*등|보유\s*주식", table_text):
                continue
            direct_total = re.search(
                r"의무\s*보유\s*확약\s*(?:비율|률)?\s*(?:은|는|:|：)?\s*"
                r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*%",
                table_text,
                flags=re.IGNORECASE,
            )
            if direct_total:
                result.update({
                    "lockup_commitment_ratio": round(float(direct_total.group(1).replace(",", "")) / 100, 6),
                    "parse_method": "lockup_direct_total_ratio",
                    "evidence": table_text[:500],
                    "rule_id": "DART_LOCKUP_TABLE_DIRECT_AGGREGATE_V1",
                    "parser_validation_status": "structurally_verified",
                    "structured_evidence": cls._table_evidence(table, scope="aggregate_direct_ratio"),
                })
                return result
            period_rows = [row for row in table if any(re.search(pattern, " ".join(row)) for pattern, _ in label_map)]
            if len(period_rows) < 2 or not re.search(r"의무\s*보유|보유\s*확약|확약\s*기간", table_text):
                continue

            direct_values: dict[str, float] = {}
            for row in period_rows:
                row_text = " | ".join(row)
                field = next((field for pattern, field in label_map if re.search(pattern, row_text)), None)
                percent = re.search(r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*%", row_text)
                if field and percent:
                    value = float(percent.group(1).replace(",", "")) / 100
                    if 0 <= value <= 1:
                        direct_values[field] = round(value, 6)
            if direct_values:
                result.update(direct_values)
                if all(field in direct_values for field in (
                    "lockup_6m_ratio", "lockup_3m_ratio", "lockup_1m_ratio", "lockup_15d_ratio",
                )):
                    result["lockup_commitment_ratio"] = round(sum(
                        direct_values[field] for field in (
                            "lockup_6m_ratio", "lockup_3m_ratio", "lockup_1m_ratio", "lockup_15d_ratio",
                        )
                    ), 6)
                    result["parse_method"] = "lockup_complete_periods_sum"
                    result["rule_id"] = "DART_LOCKUP_COMPLETE_PERIOD_PERCENT_SUM_V1"
                    result["parser_validation_status"] = "structurally_verified"
                else:
                    result["parse_method"] = "lockup_same_table_row_percent_audit_only"
                result["evidence"] = table_text[:500]
                result["structured_evidence"] = cls._table_evidence(table, scope="period_percent_rows")
                return result

            header_index = next((
                index for index, row in enumerate(table)
                if any(re.search(r"신청\s*주식\s*수|신청주식수", cell) for cell in row)
            ), None)
            if header_index is None:
                continue
            header = table[header_index]
            share_index = next(
                (index for index, cell in enumerate(header) if re.search(r"신청\s*주식\s*수|신청주식수", cell)),
                None,
            )
            if share_index is None:
                continue
            total_row = next((row for row in table[header_index + 1:] if re.search(r"합계|총계", " ".join(row))), None)
            if total_row is None or len(total_row) <= share_index:
                continue
            total = cls._parse_int(re.sub(r"[^0-9,]", "", total_row[share_index]))
            if not total:
                continue
            calculated: dict[str, float] = {}
            for row in period_rows:
                row_text = " | ".join(row)
                field = next((field for pattern, field in label_map if re.search(pattern, row_text)), None)
                if field is None or len(row) <= share_index:
                    continue
                shares = cls._parse_int(re.sub(r"[^0-9,]", "", row[share_index]))
                if shares is not None and 0 <= shares <= total:
                    calculated[field] = round(shares / total, 6)
            if calculated:
                result.update(calculated)
                if all(field in calculated for field in (
                    "lockup_6m_ratio", "lockup_3m_ratio", "lockup_1m_ratio", "lockup_15d_ratio",
                )):
                    result["lockup_commitment_ratio"] = round(sum(
                        calculated[field] for field in (
                            "lockup_6m_ratio", "lockup_3m_ratio", "lockup_1m_ratio", "lockup_15d_ratio",
                        )
                    ), 6)
                    result["parse_method"] = "lockup_complete_period_shares_sum"
                    result["rule_id"] = "DART_LOCKUP_COMPLETE_PERIOD_SHARE_SUM_V1"
                    result["parser_validation_status"] = "structurally_verified"
                else:
                    result["parse_method"] = "lockup_table_share_column_audit_only"
                result["evidence"] = table_text[:500]
                result["structured_evidence"] = cls._table_evidence(table, scope="period_share_rows")
                return result
        if re.search(r"의무\s*보유\s*확약.*?[0-9][0-9,]*(?:\.[0-9]+)?\s*%", normalized_document):
            result["parser_validation_status"] = "candidate_rejected"
            result["rejection_reason"] = "institutional_demand_scope_or_denominator_not_verified"
        return result

    @classmethod
    def _extract_aggregate_lockup_shares(cls, table: list[list[str]], application_total: Optional[int] = None) -> Optional[dict]:
        """Read the total quantity column, never a participant-count column."""
        text = " ".join(" ".join(row) for row in table)
        if re.search(r"최대\s*주주|기존\s*주주|보유\s*주식|매각제한", text):
            return None
        start = next((i for i, row in enumerate(table) if row and
                      re.fullmatch(r"\s*\d+\s*(?:개월|일)\s*확약\s*", row[0])), None)
        if start is None:
            return None
        headers = table[:start]
        columns = [c for c in range(max(map(len, headers), default=0))
                   if any(c < len(row) and re.fullmatch(r"\s*합\s*계\s*", row[c]) for row in headers)
                   and any(c < len(row) and re.fullmatch(r"\s*(?:신청\s*주식\s*수|수량)\s*", row[c]) for row in headers)]
        inferred_header = False
        if not columns and headers:
            # Some DART tables merge the total header through the unit row.
            # Recover its schema only when two explicit adjacent group schemas
            # agree and the total group has exactly the same width.
            units = [re.sub(r"\s+", "", c) for c in headers[-1]]
            total_columns = [c for c, value in enumerate(units) if value == "합계"]
            if len(total_columns) == 3:
                first = total_columns[0]
                schema = ["건수", "수량", "신청가격"]
                if (total_columns == list(range(first, first + 3)) and first >= 7
                        and units[first - 3:first] == schema
                        and units[first - 6:first - 3] == schema):
                    columns = [first + 1]
                    inferred_header = True
        if len(columns) != 1:
            return None
        column = columns[0]
        quantities = {}
        for i, row in enumerate(table[start:], start):
            if not row or column >= len(row):
                return None
            label = re.sub(r"\s+", "", row[0])
            if not re.fullmatch(r"\d+(?:개월|일)확약|미확약|확약없음|합계", label):
                return None
            if label in quantities or not re.fullmatch(r"[0-9][0-9,]*", row[column].strip()):
                return None
            quantities[label] = int(row[column].replace(",", ""))
        total = quantities.pop("합계", None)
        # A continued table can contain foreign groups plus the domestic+foreign
        # total. Anchor the merged unit to the separate demand summary, not just
        # the adjacent subgroup values (which do not cover domestic investors).
        if inferred_header and (application_total is None or total != application_total):
            return None
        none_labels = [label for label in ("미확약", "확약없음") if label in quantities]
        if len(none_labels) != 1:
            return None
        none = quantities.pop(none_labels[0])
        if total is None or total <= 0 or none is None or not quantities:
            return None
        committed = sum(quantities.values())
        if committed + none != total:
            return None
        return {
            "lockup_commitment_ratio": committed / total,
            "parse_method": "lockup_reconciled_aggregate_quantity",
            "rule_id": "DART_LOCKUP_RECONCILED_TOTAL_QUANTITY_V1",
            "parser_validation_status": "structurally_verified",
            "evidence": f"committed={committed}; noncommitted={none}; total={total}",
            "structured_evidence": cls._table_evidence(
                table, selected_column=column, committed=committed,
                noncommitted=none, total=total, scope="aggregate_quantity",
                inferred_total_unit_from_repeated_schema=inferred_header,
                independently_disclosed_application_total=application_total,
            ),
        }

    @staticmethod
    def _is_institutional_lockup_context(text: str, start: int, end: int) -> bool:
        """확약 수치가 기관 수요예측 결과인지 좁은 문맥에서 확인한다."""
        context = text[max(0, start - 250):min(len(text), end + 250)]
        if not re.search(r"기관\s*(?:투자자)?", context) or not re.search(
            r"수요\s*예측\s*(?:결과|경쟁률)", context
        ):
            return False
        return not re.search(r"최대\s*주주|기존\s*주주|임원|보유\s*주식", context)

    @staticmethod
    def _extract_offering_price_details(text: str, table_rows: Optional[list[str]] = None) -> dict:
        """공모가와 추출 근거를 함께 반환한다.

        금액 범위는 삭제 기준이 아니다. ``원`` 또는 ``KRW`` 단위까지 있는
        값만 자동 확인하고, 단위 없는 숫자는 표 번호일 가능성이 있어
        감사 로그에서 사람이 판단할 수 있도록 ``needs_review``로 격리한다.
        """
        result = {
            "offering_price": None,
            "offering_price_extracted_amount": None,
            "offering_price_review_status": "missing",
            "offering_price_finality": "unknown",
            "offering_price_parse_method": None,
            "offering_price_audit_context": None,
            "offering_price_range_warning": False,
        }
        sources = [("final_price_table_row", row) for row in (table_rows or [])]
        sources.extend(("final_price_same_sentence", sentence) for sentence in DARTCollector._split_sentences(text))

        # 자동 승인: 확정가 라벨과 통화 단위 금액이 같은 표 행 또는 같은
        # 문장에 직접 이어진 경우만 허용한다. 넓은 임의 길이 탐색은 하지 않는다.
        for method, context in sources:
            value = DARTCollector._extract_direct_price_with_currency(context)
            if value is not None:
                result.update({
                    "offering_price": value,
                    "offering_price_extracted_amount": value,
                    "offering_price_review_status": "verified_currency_unit",
                    "offering_price_finality": "confirmed_price_language",
                    "offering_price_parse_method": method,
                    "offering_price_audit_context": context,
                    "offering_price_range_warning": value < 100 or value > 10_000_000,
                })
                return result

        # 초기 신고서의 정형 문구는 금액 후보를 만들지 않고 별도 상태로 남긴다.
        for _, context in sources:
            if re.search(
                FINAL_PRICE_LABEL_PATTERN
                + r"[^.!?。;]{0,100}(?:최종\s*)?결정할\s*예정|"
                + FINAL_PRICE_LABEL_PATTERN
                + r"[^.!?。;]{0,100}정정(?:증권)?신고서.{0,40}?제출할\s*예정",
                context,
                flags=re.IGNORECASE,
            ):
                result.update({
                    "offering_price_review_status": "preliminary_price_language",
                    "offering_price_finality": "preliminary_price_language",
                    "offering_price_parse_method": "preliminary_price_statement",
                    "offering_price_audit_context": context,
                })
                return result

        # 단위 없는 직접 연결 숫자는 감사 후보로만 남긴다. 1주당의 "1"과
        # 희망밴드·액면가·총액 문맥은 후보에서도 제외한다.
        for _, context in sources:
            value = DARTCollector._extract_direct_price_without_currency(context)
            if value is not None:
                result.update({
                    "offering_price_extracted_amount": value,
                    "offering_price_review_status": "needs_review_no_currency_unit",
                    "offering_price_parse_method": "direct_price_without_currency_unit",
                    "offering_price_audit_context": context,
                    "offering_price_range_warning": value < 100 or value > 10_000_000,
                })
                return result
        return result

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        # 기관 수요예측 경쟁률은 `850.00 : 1`처럼 소수점을 자주 포함한다.
        # 숫자 사이의 `.`를 문장 끝으로 분리하면 경쟁률과 `: 1`이 갈라져
        # 실제 DART 원문 값을 놓치므로, 숫자 뒤 마침표는 보존한다.
        return [
            sentence.strip()
            for sentence in re.split(r"[!?。;]|(?<!\d)\.", text)
            if sentence.strip()
        ]

    @staticmethod
    def _extract_direct_price_with_currency(context: str) -> Optional[int]:
        pattern = re.compile(
            FINAL_PRICE_LABEL_PATTERN
            + r"(?P<bridge>[^.!?。;]{0,28}?)(?P<amount>[1-9][0-9,]*)\s*(?:원|KRW)",
            flags=re.IGNORECASE,
        )
        for match in pattern.finditer(context):
            bridge = match.group("bridge")
            if "~" in bridge or "～" in bridge or re.search(PRICE_CONTEXT_EXCLUSIONS, bridge):
                continue
            value = DARTCollector._parse_int(match.group("amount"))
            if value is not None:
                return value
        return None

    @staticmethod
    def _extract_direct_price_without_currency(context: str) -> Optional[int]:
        pattern = re.compile(
            FINAL_PRICE_LABEL_PATTERN
            + r"(?P<bridge>[^.!?。;]{0,28}?)(?P<amount>[1-9][0-9,]*)(?!\s*(?:원|KRW))",
            flags=re.IGNORECASE,
        )
        for match in pattern.finditer(context):
            bridge = match.group("bridge")
            suffix = context[match.end("amount"):]
            if (
                "~" in bridge
                or "～" in bridge
                or re.search(PRICE_CONTEXT_EXCLUSIONS, bridge)
                or re.match(r"\s*주당", suffix)
            ):
                continue
            value = DARTCollector._parse_int(match.group("amount"))
            if value is not None:
                return value
        return None

    @staticmethod
    def _parse_int(value: str) -> Optional[int]:
        try:
            return int(str(value).replace(",", ""))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_money_value(value: object) -> Optional[int]:
        match = re.search(r"([0-9][0-9,]*)", str(value or ""))
        return DARTCollector._parse_int(match.group(1)) if match else None

    @staticmethod
    def _walk_dicts(value: object):
        """OpenDART의 그룹형 JSON 응답을 평탄화한다."""
        if isinstance(value, dict):
            yield value
            for child in value.values():
                yield from DARTCollector._walk_dicts(child)
        elif isinstance(value, list):
            for child in value:
                yield from DARTCollector._walk_dicts(child)

    @staticmethod
    def _parse_lockup_months(text: str) -> int:
        years = re.findall(r"(\d+)\s*년", text)
        months = re.findall(r"(\d+)\s*개월", text)
        total = sum(int(y) * 12 for y in years) + sum(int(m) for m in months)
        return min(total, 36)

    # ── 배치 수집 ─────────────────────────────────────────────

    def collect_full_history(
        self,
        start_year: int = 2015,
        end_year:   int = 2024,
    ) -> pd.DataFrame:
        """
        전체 히스토리 수집 메인 함수.
        각 연도별로 공시 목록 → 상세 데이터를 수집해 DataFrame으로 반환.

        실제 운영 시 진행상황을 체크포인트 파일에 저장해
        중단 후 재시작 시 이어서 수집할 수 있도록 한다.
        """
        records = []
        checkpoint_path = RAW_DIR / "collect_checkpoint.csv"

        # 이미 수집된 rcept_no 로드 (중복 방지)
        collected_nos = set()
        if checkpoint_path.exists():
            done = pd.read_csv(checkpoint_path)
            collected_nos = set(done["rcept_no"].tolist())
            logger.info("체크포인트 로드: %d건 이미 수집됨", len(collected_nos))

        for year in range(start_year, end_year + 1):
            start = f"{year}0101"
            end   = f"{year}1231"
            logger.info("수집 중: %d년", year)

            disc_list = self.get_ipo_disclosure_list(start, end)
            if disc_list.empty:
                continue

            for _, row in disc_list.iterrows():
                rcept_no  = row["rcept_no"]
                corp_code = row["corp_code"]

                if rcept_no in collected_nos:
                    continue

                # 공모 기본 정보
                offering = self.get_offering_info(rcept_no)
                time.sleep(REQUEST_DELAY)

                # 수요예측 결과 (별도 공시 번호가 있으므로 연계 필요)
                # NOTE: 실제 구현 시 공시 목록에서 "수요예측결과" 공시를
                #       corp_code 기준으로 조회해 연계한다.
                demand = self.get_demand_forecast(corp_code, rcept_no)
                time.sleep(REQUEST_DELAY)

                record = {
                    "rcept_no":   rcept_no,
                    "corp_code":  corp_code,
                    "corp_name":  row.get("corp_name", ""),
                    "rcept_dt":   row["rcept_dt"],
                    **offering,
                    **{k: v for k, v in demand.items() if k != "corp_code"},
                }
                records.append(record)

                # 체크포인트 저장 (10건마다)
                if len(records) % 10 == 0:
                    pd.DataFrame(records).to_csv(checkpoint_path, index=False)

            logger.info("%d년 완료: 누적 %d건", year, len(records))

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        out_path = RAW_DIR / "dart_ipo_raw.parquet"
        df.to_parquet(out_path, index=False)
        logger.info("수집 완료: %d건 → %s", len(df), out_path)
        return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    collector = DARTCollector()
    df = collector.collect_full_history(start_year=2020, end_year=2024)
    print(df.head())
    print(f"\n수집 완료: {len(df)}건")
