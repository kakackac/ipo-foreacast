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
import re
import time
import zipfile
from io import BytesIO
from datetime import date, timedelta
from html import unescape as html_unescape
from typing import Optional

import requests
import pandas as pd

from config import DART_API_KEY, DART_BASE_URL, RAW_DIR

logger = logging.getLogger(__name__)


# ── 상수 ──────────────────────────────────────────────────────
REQUEST_DELAY = 0.3          # API 호출 간격 (초) — 속도 제한 회피
MAX_RETRIES   = 3
TIMEOUT       = 15


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

        return self._normalize_text(" ".join(fragments))

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
        disclosures = self.get_company_disclosure_list(corp_code, start_date, end_date)
        if disclosures.empty:
            return None

        title = disclosures["report_nm"].fillna("")
        score = (
            title.str.contains("수요예측", regex=False).astype(int) * 10
            + title.str.contains("기관투자자", regex=False).astype(int) * 4
            + title.str.contains("투자설명서", regex=False).astype(int) * 2
            + title.str.contains("증권신고서", regex=False).astype(int)
        )
        candidates = disclosures.assign(_score=score).sort_values(["_score", "rcept_dt"], ascending=[False, False])
        best = candidates.iloc[0]
        # 단순 증권신고서만 있는 경우에는 같은 원문을 수요예측 결과로 오인하지
        # 않는다. 수요예측 또는 투자설명서 단서가 있는 경우만 원문 파싱한다.
        return str(best["rcept_no"]) if int(best["_score"]) >= 2 else None

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
            "lockup_6m_ratio":        None,
            "lockup_3m_ratio":        None,
            "lockup_1m_ratio":        None,
            "lockup_15d_ratio":       None,
            "lockup_none_ratio":      None,
            "parse_success":          False,
        }

        if not html:
            return result
        text = self._normalize_text(html)

        demand_price = self._extract_offering_price_details(text)
        result["demand_offering_price"] = demand_price["offering_price"]
        result["demand_offering_price_context"] = demand_price["offering_price_audit_context"]

        # 경쟁률 패턴: "XXX : 1" 또는 "XXX대 1"
        ratio_pattern = r"경쟁률[^0-9]*([0-9,]+(?:\.[0-9]+)?)\s*(?::|대)\s*1"
        m = re.search(ratio_pattern, text)
        if m:
            result["institutional_demand_ratio"] = float(m.group(1).replace(",", ""))

        # 확약 비율 파싱 — 각 행의 신청주식수를 추출해 합계 대비 비율 계산
        total_shares = self._extract_share_after(text, "합계")
        lockup_shares = {
            "lockup_6m_ratio":   self._extract_share_after(text, "6개월"),
            "lockup_3m_ratio":   self._extract_share_after(text, "3개월"),
            "lockup_1m_ratio":   self._extract_share_after(text, "1개월"),
            "lockup_15d_ratio":  self._extract_share_after(text, "15일"),
            "lockup_none_ratio": self._extract_share_after(text, "확약없음|미확약"),
        }
        if total_shares and total_shares > 0:
            for key, shares in lockup_shares.items():
                if shares is not None:
                    result[key] = round(min(max(shares / total_shares, 0), 1), 6)

        # 파싱 성공 여부만 플래그 설정 (실제 비율 계산은 수집된 데이터로)
        if result["institutional_demand_ratio"] is not None or total_shares:
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
        return self._parse_offering_html(self.get_document_text(rcept_no), rcept_no)

    def _parse_offering_html(self, html: str, rcept_no: str) -> dict:
        """공모 정보 HTML 파싱"""
        result = {
            "rcept_no":           rcept_no,
            "price_band_low":     None,
            "price_band_high":    None,
            "offering_price":     None,
            "offering_price_extracted_amount": None,
            "offering_price_review_status": "missing",
            "offering_price_parse_method": None,
            "offering_price_audit_context": None,
            "offering_price_range_warning": False,
            "new_shares":         None,
            "secondary_shares":   None,
            "total_post_listing_shares": None,
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
        band_pattern = r"희망\s*공모가[^0-9]*([0-9,]+)\s*[~～]\s*([0-9,]+)"
        m = re.search(band_pattern, text)
        if m:
            result["price_band_low"]  = int(m.group(1).replace(",", ""))
            result["price_band_high"] = int(m.group(2).replace(",", ""))

        price_details = self._extract_offering_price_details(text)
        result.update(price_details)

        # 공모 구조
        result["new_shares"] = self._extract_share_after(text, "신주모집|신주발행|모집주식수")
        result["secondary_shares"] = self._extract_share_after(text, "구주매출|매출주식수")
        result["total_post_listing_shares"] = self._extract_share_after(
            text,
            "상장예정주식수|상장\\s*예정\\s*주식수|상장\\s*후\\s*총\\s*발행주식수|발행주식총수",
        )

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

    @staticmethod
    def _extract_share_after(text: str, label_pattern: str) -> Optional[int]:
        pattern = rf"(?:{label_pattern})[^0-9]{{0,120}}([0-9,]+)\s*주"
        m = re.search(pattern, text)
        if not m:
            return None
        return DARTCollector._parse_int(m.group(1))

    @staticmethod
    def _extract_offering_price_details(text: str) -> dict:
        """공모가와 추출 근거를 함께 반환한다.

        금액 범위는 삭제 기준이 아니다. ``원`` 또는 ``KRW`` 단위까지 있는
        값만 자동 확인하고, 단위 없는 숫자는 표 번호일 가능성이 있어
        감사 로그에서 사람이 판단할 수 있도록 ``needs_review``로 격리한다.
        """
        result = {
            "offering_price": None,
            "offering_price_extracted_amount": None,
            "offering_price_review_status": "missing",
            "offering_price_parse_method": None,
            "offering_price_audit_context": None,
            "offering_price_range_warning": False,
        }
        context_pattern = r"(?:1\s*주당\s*)?(?:확정|최종)\s*공모가(?:액)?|공모가\s*확정"
        money_pattern = r"(?<![0-9])([1-9][0-9,]*)\s*(?:원|KRW)"
        numeric_pattern = r"(?<![0-9])([0-9][0-9,]*)(?![0-9])"
        for match in re.finditer(context_pattern, text, flags=re.IGNORECASE):
            # 확정 공모가 문구 바로 뒤의 좁은 문맥만 사용한다. 멀리 떨어진
            # 공모총액·발행금액 같은 다른 원화 금액을 가져오는 일을 줄인다.
            context = text[max(0, match.start() - 80):match.end() + 140]
            price_context = text[match.end():match.end() + 140]
            money_match = re.search(money_pattern, price_context, flags=re.IGNORECASE)
            if money_match:
                value = DARTCollector._parse_int(money_match.group(1))
                if value is not None:
                    result.update({
                        "offering_price": value,
                        "offering_price_extracted_amount": value,
                        "offering_price_review_status": "verified_currency_unit",
                        "offering_price_parse_method": "final_price_with_currency_unit",
                        "offering_price_audit_context": context,
                        "offering_price_range_warning": value < 100 or value > 10_000_000,
                    })
                    return result

            numeric_match = re.search(numeric_pattern, price_context)
            if numeric_match:
                value = DARTCollector._parse_int(numeric_match.group(1))
                result.update({
                    "offering_price_extracted_amount": value,
                    "offering_price_review_status": "needs_review_no_currency_unit",
                    "offering_price_parse_method": "unverified_numeric_candidate",
                    "offering_price_audit_context": context,
                    "offering_price_range_warning": bool(value is not None and (value < 100 or value > 10_000_000)),
                })
                return result
        return result

    @staticmethod
    def _parse_int(value: str) -> Optional[int]:
        try:
            return int(str(value).replace(",", ""))
        except (TypeError, ValueError):
            return None

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
