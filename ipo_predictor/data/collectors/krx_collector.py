"""KRX OpenAPI를 이용해 IPO 사후 실적과 시장 데이터를 수집한다.

승인된 일별 API만 사용한다. KRX 웹사이트 로그인이나 개인 ID/비밀번호는
사용하지 않으며, API 키는 ``KRX_API_KEY`` 환경변수로만 전달한다.
"""

import hashlib
import json
import logging
import re
import time
from datetime import date, datetime, timedelta
from io import StringIO
from typing import Any, Optional

import pandas as pd
import requests

from config import KRX_API_KEY, KRX_OPENAPI_BASE_URL, RAW_DIR

logger = logging.getLogger(__name__)

REQUEST_DELAY = 0.1
MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 1.0
# KRX 일반 인증키의 일일 호출 한도보다 여유를 두고 중단한다.
MAX_REQUESTS_PER_RUN = 8_000
KIND_LISTING_URL = "https://kind.krx.co.kr/listinvstg/listingcompany.do"
KIND_LISTING_PAGE_URL = (
    "https://kind.krx.co.kr/listinvstg/listingcompany.do?method=searchListingTypeMain"
)

MARKETS = {
    "KOSPI": {
        "daily_price": "sto/stk_bydd_trd",
        "basic_info": "sto/stk_isu_base_info",
        "index": "idx/kospi_dd_trd",
        "index_names": {"KOSPI", "코스피"},
    },
    "KOSDAQ": {
        "daily_price": "sto/ksq_bydd_trd",
        "basic_info": "sto/ksq_isu_base_info",
        "index": "idx/kosdaq_dd_trd",
        "index_names": {"KOSDAQ", "코스닥"},
    },
}


class KRXCollector:
    """KRX OpenAPI 일별 데이터 수집기."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = KRX_OPENAPI_BASE_URL,
        session: requests.Session | None = None,
        request_delay: float = REQUEST_DELAY,
    ):
        self.api_key = (api_key if api_key is not None else KRX_API_KEY).strip()
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.request_delay = request_delay
        self.request_count = 0
        self.official_listing_requests: list[dict[str, Any]] = []

    @property
    def is_configured(self) -> bool:
        """KRX OpenAPI 키가 준비됐는지 반환한다."""
        return bool(self.api_key)

    def _get_daily_records(self, endpoint: str, bas_dd: str) -> list[dict[str, Any]]:
        """한 기준일의 KRX OpenAPI 응답을 표준 레코드 목록으로 바꾼다."""
        if not self.is_configured:
            raise RuntimeError("KRX_API_KEY를 설정한 뒤 KRX OpenAPI 수집을 실행하세요.")
        if self.request_count >= MAX_REQUESTS_PER_RUN:
            raise RuntimeError(
                f"이번 실행의 KRX OpenAPI 요청이 {MAX_REQUESTS_PER_RUN:,}건에 도달했습니다. "
                "기간을 나누어 다시 실행하세요."
            )

        payload = None
        last_error: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.session.get(
                    f"{self.base_url}/{endpoint}",
                    params={"basDd": bas_dd},
                    headers={"AUTH_KEY": self.api_key, "Accept": "application/json"},
                    timeout=30,
                )
                self.request_count += 1
                response.raise_for_status()
                payload = response.json()
                break
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt == MAX_RETRIES:
                    break
                delay = RETRY_BACKOFF_SECONDS * attempt
                logger.warning(
                    "KRX OpenAPI 일시 오류, %.1f초 후 재시도 (%d/%d): %s %s (%s)",
                    delay, attempt, MAX_RETRIES, endpoint, bas_dd, exc,
                )
                time.sleep(delay)

        if payload is None:
            raise RuntimeError(
                f"KRX OpenAPI 요청 실패 ({endpoint}, {bas_dd}, {MAX_RETRIES}회 재시도): {last_error}"
            ) from last_error

        if self.request_delay:
            time.sleep(self.request_delay)

        if not isinstance(payload, dict):
            raise RuntimeError(f"KRX OpenAPI 응답 형식이 올바르지 않습니다 ({endpoint}, {bas_dd}).")
        records = payload.get("OutBlock_1", payload.get("output", []))
        if not isinstance(records, list):
            logger.warning("KRX OpenAPI 응답에 데이터 목록이 없습니다: %s %s", endpoint, bas_dd)
            return []
        return records

    @staticmethod
    def _to_number(value: object) -> Optional[float]:
        if value is None or pd.isna(value):
            return None
        text = str(value).replace(",", "").strip()
        if text in {"", "-", "N/A"}:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    @staticmethod
    def _as_bas_dd(value: str | date | datetime | pd.Timestamp) -> str:
        return pd.Timestamp(value).strftime("%Y%m%d")

    @staticmethod
    def _normalise_ticker(value: object) -> str:
        return str(value or "").strip().zfill(6)

    @classmethod
    def _short_issue_code(cls, value: object) -> str:
        """KRX 표준종목코드 또는 거래코드를 6자리 거래코드로 통일한다.

        종목기본정보의 ``ISU_CD``는 ``KR7365550003``처럼 12자리 ISIN이고,
        일별매매정보의 ``ISU_CD``는 ``365550``처럼 거래코드다. ISIN의
        4~9번째 문자가 거래코드이므로 두 응답을 비교하기 전에 변환한다.
        """
        code = str(value or "").strip().upper()
        if len(code) == 12 and code.startswith("KR"):
            return code[3:9]
        return cls._normalise_ticker(code)

    @staticmethod
    def _normalise_index_name(value: object) -> str:
        return str(value or "").replace(" ", "").upper()

    @staticmethod
    def _normalise_market(market: str | None) -> str | None:
        if market is None:
            return None
        normalised = str(market).upper().strip()
        return normalised if normalised in MARKETS else None

    @staticmethod
    def _normalise_company_name(value: object) -> str:
        """외국기업 Reg.S 표기 차이를 제외하고 회사명을 비교한다."""
        name = re.sub(r"[^0-9A-Za-z가-힣]", "", str(value or "")).upper()
        return name.replace("REGS", "")

    # ── 시장 지수 ─────────────────────────────────────────────

    def get_index_ohlcv(self, index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """KOSPI 또는 KOSDAQ의 일별 종가를 공식 OpenAPI에서 수집한다.

        KRX 일별 API는 하나의 기준일(``basDd``)만 받으므로 영업일별로
        조회한다. 휴장일은 빈 응답으로 건너뛴다.
        """
        market = "KOSPI" if str(index_code) == "1" else "KOSDAQ"
        metadata = MARKETS[market]
        start = pd.Timestamp(start_date)
        end = min(pd.Timestamp(end_date), pd.Timestamp.today().normalize())
        if start > end:
            return pd.DataFrame(columns=["date", "index_code", "close"])

        records: list[dict[str, Any]] = []
        target_names = {self._normalise_index_name(name) for name in metadata["index_names"]}
        for bas_dd in pd.bdate_range(start, end):
            rows = self._get_daily_records(metadata["index"], self._as_bas_dd(bas_dd))
            row = next(
                (item for item in rows if self._normalise_index_name(item.get("IDX_NM")) in target_names),
                None,
            )
            if row is None:
                continue
            records.append({
                "date": pd.Timestamp(row.get("BAS_DD", bas_dd)),
                "index_code": market,
                "close": self._to_number(row.get("CLSPRC_IDX")),
            })

        frame = pd.DataFrame(records, columns=["date", "index_code", "close"])
        if frame.empty:
            logger.warning("%s 지수 데이터가 없습니다: %s~%s", market, start_date, end_date)
            return frame
        return frame.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)

    def get_market_momentum(
        self,
        listing_date: date,
        index: str = "KOSPI",
        windows: list[int] = [5, 20, 60],
    ) -> dict[str, float]:
        """상장 직전 거래일 종가를 기준으로 시장 모멘텀을 계산한다."""
        start = listing_date - timedelta(days=max(windows) * 2)
        end = listing_date - timedelta(days=1)
        index_code = "1" if index.upper() == "KOSPI" else "2"
        frame = self.get_index_ohlcv(index_code, self._as_bas_dd(start), self._as_bas_dd(end))
        if frame.empty:
            return {f"{index.lower()}_momentum_{window}d": None for window in windows}

        closes = frame["close"].to_numpy()
        result = {}
        for window in windows:
            result[f"{index.lower()}_momentum_{window}d"] = (
                round(closes[-1] / closes[-1 - window] - 1, 6) if len(closes) > window else None
            )
        return result

    # ── 상장 종목과 상장일 가격 ─────────────────────────────────

    def get_listing_day_price(
        self,
        ticker: str,
        listing_date: str,
        isu_cd: Optional[str] = None,
        market: str | None = None,
        corp_name: str | None = None,
    ) -> dict[str, Any]:
        """상장일 시가·정규장 종가를 수집한다.

        ``isu_cd``가 있으면 KRX 표준 종목코드로 우선 비교한다. 시장을 알 수
        없을 때만 KOSPI와 KOSDAQ를 모두 조회해, API 호출을 최소화한다.
        """
        selected_market = self._normalise_market(market)
        # KIND 시장값이 맞더라도 과거 코드·시장 변경 사례를 감사하기 위해, 첫
        # 시장에서 못 찾은 경우에만 반대 시장을 추가로 확인한다.
        markets = [selected_market] if selected_market else list(MARKETS)
        if selected_market:
            markets.extend(name for name in MARKETS if name != selected_market)
        normalized_ticker = self._normalise_ticker(ticker)
        normalized_isu_cd = str(isu_cd or "").strip().upper()
        expected_short_code = self._short_issue_code(isu_cd or ticker)
        normalized_name = self._normalise_company_name(corp_name)
        rows_by_market: dict[str, list[dict[str, Any]]] = {}

        for market_name in markets:
            rows = self._get_daily_records(
                MARKETS[market_name]["daily_price"], self._as_bas_dd(listing_date)
            )
            rows_by_market[market_name] = rows
            for row in rows:
                row_isu_cd = str(row.get("ISU_CD", "")).strip().upper()
                row_ticker = self._normalise_ticker(row.get("ISU_SRT_CD"))
                if normalized_isu_cd and row_isu_cd == normalized_isu_cd:
                    method = "krx_standard_code"
                    if selected_market and market_name != selected_market:
                        method = "krx_standard_code_market_mismatch_recovered"
                    return self._price_record(
                        ticker, listing_date, isu_cd, row, market_name, method, rows_by_market
                    )
                if (
                    row_ticker == normalized_ticker
                    or self._short_issue_code(row_isu_cd) == expected_short_code
                ):
                    method = "ticker_or_short_issue_code"
                    if selected_market and market_name != selected_market:
                        method = "ticker_or_short_issue_code_market_mismatch_recovered"
                    return self._price_record(
                        ticker, listing_date, isu_cd or row_isu_cd, row, market_name, method, rows_by_market
                    )

        if normalized_name:
            for market_name, rows in rows_by_market.items():
                name_match = next(
                    (
                        row for row in rows
                        if self._normalise_company_name(row.get("ISU_NM")) == normalized_name
                    ),
                    None,
                )
                if name_match is not None:
                    method = "company_name_fallback"
                    if selected_market and market_name != selected_market:
                        method = "company_name_market_mismatch_recovered"
                    logger.info("회사명으로 상장일 가격을 매칭했습니다: %s (%s)", corp_name, method)
                    return self._price_record(
                        ticker, listing_date, isu_cd, name_match, market_name, method, rows_by_market
                    )

        row_count = sum(len(rows) for rows in rows_by_market.values())
        if row_count == 0:
            failure_reason = "daily_price_api_response_empty"
        elif normalized_name:
            failure_reason = "daily_rows_code_and_company_name_unmatched"
        else:
            failure_reason = "daily_rows_code_unmatched_company_name_unavailable"
        logger.warning(
            "상장일 가격 미매칭 | 종목=%s | 일자=%s | 사유=%s | 조회시장=%s | 응답행=%d",
            ticker, listing_date, failure_reason, ",".join(rows_by_market), row_count,
        )
        return {
            "ticker": ticker,
            "isu_cd": isu_cd,
            "listing_date": self._as_bas_dd(listing_date),
            "market": selected_market,
            "open_price": None,
            "close_price": None,
            "high_price": None,
            "low_price": None,
            "volume": None,
            "price_match_status": "unmatched",
            "price_match_method": None,
            "price_failure_reason": failure_reason,
            "price_markets_queried": ",".join(rows_by_market),
            "price_api_rows_returned": row_count,
            "price_raw_response_evidence": self._raw_price_response_evidence(rows_by_market),
            "price_matched_ticker": None,
            "price_matched_isu_cd": None,
            "price_matched_corp_name": None,
        }

    def _price_record(
        self,
        ticker: str,
        listing_date: str,
        isu_cd: str | None,
        row: dict[str, Any],
        market: str,
        match_method: str,
        rows_by_market: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        return {
            "ticker": ticker,
            "isu_cd": isu_cd,
            "listing_date": self._as_bas_dd(listing_date),
            "market": market,
            "open_price": self._to_number(row.get("TDD_OPNPRC")),
            "close_price": self._to_number(row.get("TDD_CLSPRC")),
            "high_price": self._to_number(row.get("TDD_HGPRC")),
            "low_price": self._to_number(row.get("TDD_LWPRC")),
            "volume": self._to_number(row.get("ACC_TRDVOL")),
            "price_match_status": "matched",
            "price_match_method": match_method,
            "price_failure_reason": None,
            "price_markets_queried": ",".join(rows_by_market),
            "price_api_rows_returned": sum(len(rows) for rows in rows_by_market.values()),
            "price_raw_response_evidence": self._raw_price_response_evidence(rows_by_market),
            "price_matched_ticker": row.get("ISU_SRT_CD"),
            "price_matched_isu_cd": row.get("ISU_CD"),
            "price_matched_corp_name": row.get("ISU_NM"),
        }

    @staticmethod
    def _raw_price_response_evidence(rows_by_market: dict[str, list[dict[str, Any]]]) -> str:
        """일별 원시 응답을 재현 가능하게 요약한다.

        일자별 전체 시세 행을 이벤트마다 중복 저장하지 않는다. 대신 실제로
        매칭에 사용한 KRX 응답의 시장별 행 수, SHA-256 해시, 식별 열 표본을
        보관해 같은 응답인지와 코드·회사명 대조 근거를 감사할 수 있게 한다.
        """
        evidence: dict[str, dict[str, Any]] = {}
        for market, rows in rows_by_market.items():
            raw = json.dumps(rows, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
            evidence[market] = {
                "row_count": len(rows),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "identity_sample": [
                    {
                        "ISU_CD": row.get("ISU_CD"),
                        "ISU_SRT_CD": row.get("ISU_SRT_CD"),
                        "ISU_NM": row.get("ISU_NM"),
                    }
                    for row in rows[:20]
                ],
            }
        return json.dumps(evidence, ensure_ascii=False, sort_keys=True)

    def get_legacy_list_dd_candidates(self, start_date: str, end_date: str) -> pd.DataFrame:
        """종목기본정보 ``LIST_DD`` 기반의 이전 후보 목록을 반환한다.

        이것은 공식 IPO 모집단이 아니다. 기준일 시점 상장 종목만 포함하므로
        상장폐지 이력 누락 가능성이 있다. 새 이벤트 마스터와의 행 단위 비교를
        위한 ``legacy_list_dd_candidate`` 원천으로만 보존한다.
        """
        start = pd.Timestamp(start_date)
        end = min(pd.Timestamp(end_date), pd.Timestamp.today().normalize())
        if start > end:
            return pd.DataFrame()

        frames = []
        for market, metadata in MARKETS.items():
            records = self._get_daily_records(metadata["basic_info"], self._as_bas_dd(end))
            if not records:
                continue
            frame = pd.DataFrame(records).rename(columns={
                "ISU_CD": "isu_cd",
                "ISU_SRT_CD": "ticker",
                "ISU_ABBRV": "corp_name",
                "LIST_DD": "listing_date",
                "SECT_TP_NM": "sector",
            })
            if "ticker" not in frame:
                continue
            frame["market"] = market
            frames.append(frame)

        if not frames:
            return pd.DataFrame()
        frame = pd.concat(frames, ignore_index=True)
        frame["ticker"] = frame["ticker"].map(self._normalise_ticker)
        frame["listing_date"] = pd.to_datetime(frame["listing_date"], errors="coerce")
        frame = frame[(frame["listing_date"] >= start) & (frame["listing_date"] <= end)].copy()
        if frame.empty:
            return frame

        counts = frame.groupby("listing_date").size().rename("same_day_ipo_count")
        frame = frame.merge(counts, on="listing_date", how="left")
        frame["legacy_source"] = "krx_openapi_issue_master_list_dd"
        frame["legacy_candidate_type"] = "legacy_list_dd_candidate"
        columns = [
            "ticker", "isu_cd", "corp_name", "listing_date", "market", "sector",
            "same_day_ipo_count", "legacy_source", "legacy_candidate_type",
        ]
        return frame.reindex(columns=columns).drop_duplicates(["ticker", "listing_date"]).reset_index(drop=True)

    # 이전 호출부와의 호환성은 유지하되, 파이프라인은 이 메서드를 사용하지 않는다.
    def get_ipo_calendar(self, start_date: str, end_date: str) -> pd.DataFrame:
        return self.get_legacy_list_dd_candidates(start_date, end_date)

    @staticmethod
    def _kind_text(value: object) -> str | None:
        if value is None or pd.isna(value):
            return None
        text = re.sub(r"\s+", " ", str(value)).strip()
        return text if text and text not in {"-", "nan", "None"} else None

    @classmethod
    def _classify_official_listing_event(cls, row: pd.Series) -> dict[str, str | bool]:
        """KIND의 공식 상장유형·증권구분을 우선해 보수적으로 분류한다."""
        name = cls._kind_text(row.get("corp_name")) or ""
        listing_type = cls._kind_text(row.get("listing_type")) or ""
        security_type = cls._kind_text(row.get("security_type")) or ""
        stock_type = cls._kind_text(row.get("stock_type")) or ""
        country = cls._kind_text(row.get("country")) or ""
        normalized = re.sub(r"\s+", "", name).upper()
        security_text = " ".join([security_type, stock_type]).upper()

        def result(
            category: str,
            reason: str,
            confidence: str,
            review: bool = False,
            offering_type: str = "review_required",
            retail_eligibility_status: str = "review_required",
        ) -> dict[str, str | bool]:
            return {
                "event_class": category,
                "offering_type": offering_type,
                # KIND의 신규상장 표만으로 일반청약 가능 여부를 확정하지 않는다.
                # 실제 일반청약 결과 공지가 연결된 뒤에만 True로 승격한다.
                "retail_subscription_eligibility_status": retail_eligibility_status,
                "classification_reason": reason,
                "classification_confidence": confidence,
                "classification_review_required": review,
            }

        if "재상장" in listing_type:
            return result("relisting", f"KIND 상장유형={listing_type}", "high", offering_type="relisting")
        if "이전상장" in listing_type:
            # KIND 공개 목록은 이전 시장을 항상 주지 않으므로 KONEX 여부를 추측하지 않는다.
            return result(
                "unclassified_review", "KIND 이전상장; 이전 시장 미확인", "medium", True,
                offering_type="market_transfer",
            )
        if any(token in security_text for token in ("ETF", "ETN", "수익증권", "펀드", "ELW")):
            return result(
                "ineligible_product", f"KIND 증권구분/주식종류={security_text}", "high",
                offering_type="fund_or_exchange_traded_product",
                retail_eligibility_status="not_eligible_product",
            )
        # '메리츠'처럼 '리츠'를 포함하는 일반·스팩 종목을 리츠로 오인하지 않도록
        # 스팩 판정을 리츠 문자열보다 먼저 한다.
        if "스팩" in name or "SPAC" in normalized:
            return result(
                "spac_ipo", "종목명 스팩/SPAC 단서", "high", offering_type="spac_ipo",
                retail_eligibility_status="candidate_requires_official_notice",
            )
        if "리츠" in normalized or "REIT" in normalized or "부동산투자회사" in security_text:
            return result(
                "ineligible_product", "리츠·부동산투자회사", "high", offering_type="reit",
                retail_eligibility_status="not_eligible_product",
            )
        if re.search(r"(?:우|우B|우C|우선)$", normalized) or "우선" in stock_type:
            return result(
                "preferred_or_class_share", "종목명 또는 주식종류의 우선·종류주 단서", "medium", True,
                offering_type="preferred_or_class_share",
            )
        if country and country not in {"대한민국", "국내"}:
            return result(
                "foreign_listing", f"KIND 국적={country}", "high",
                offering_type="foreign_common_stock_listing",
                retail_eligibility_status="candidate_requires_official_notice",
            )
        if listing_type == "신규상장" and security_type in {"주권", "일반주권", ""}:
            return result(
                "general_ipo", "KIND 신규상장·주권; 추가 공시 정합 대기", "medium", True,
                offering_type="common_stock_ipo",
                retail_eligibility_status="candidate_requires_official_notice",
            )
        return result("unclassified_review", "공식 분류 필드가 부족하거나 규칙 미포함", "low", True)

    @staticmethod
    def _kind_column(frame: pd.DataFrame, *candidates: str) -> pd.Series:
        for candidate in candidates:
            if candidate in frame.columns:
                return frame[candidate]
        return pd.Series([None] * len(frame), index=frame.index)

    def get_official_listing_events(self, start_date: str, end_date: str) -> pd.DataFrame:
        """KIND 공식 신규상장기업현황을 이벤트 마스터 원천으로 수집한다.

        KIND는 KRX가 운영하는 공식 상장공시 채널이다. 이 공개 결과는
        상장유형·증권구분·업종·국적·상장주선인을 제공하며, 종목기본정보의
        현재 스냅샷을 과거 IPO 모집단으로 쓰는 문제를 피한다.
        """
        start = pd.Timestamp(start_date).normalize()
        end = min(pd.Timestamp(end_date).normalize(), pd.Timestamp.today().normalize())
        columns = [
            "event_id", "ticker", "krx_standard_code", "corp_name", "market", "listing_date",
            "security_type", "stock_type", "listing_type", "offering_price", "offering_shares",
            "lead_underwriter", "industry_name", "industry_code", "country", "face_value",
            "offering_amount", "event_class", "classification_reason", "classification_confidence",
            "classification_review_required", "offering_type", "retail_subscription_eligibility_status",
            "source_name", "source_url", "source_request_id",
            "collected_at", "verification_status", "listing_segment",
        ]
        if start > end:
            return pd.DataFrame(columns=columns)

        payload = {
            "method": "searchListingTypeSub",
            "forward": "listingtype_down",
            "currentPageSize": "3000",
            "pageIndex": "1",
            "marketType": "",
            "country": "",
            "industry": "",
            "listTypeArrStr": "01|02|03|04|05",
            "secuGrpArrStr": "ST|FS|MF|SC|RT|IF|DR",
            "choicTypeArrStr": "01|02|03|05",
            "fromDate": start.strftime("%Y-%m-%d"),
            "toDate": end.strftime("%Y-%m-%d"),
        }
        request_id = f"kind_listing_{start:%Y%m%d}_{end:%Y%m%d}"
        attempt = {
            "request_id": request_id,
            "source": "KIND_official_listing_company",
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "cache_used": False,
            "status": "started",
            "response_rows": 0,
        }
        try:
            response = self.session.post(KIND_LISTING_URL, data=payload, timeout=30)
            response.raise_for_status()
            html = response.content.decode("euc-kr", errors="replace")
            tables = pd.read_html(StringIO(html))
            table = next((item for item in tables if "회사명" in item.columns and "상장일" in item.columns), None)
            if table is None:
                raise RuntimeError("KIND 신규상장 결과 표를 찾지 못했습니다.")
            frame = table.copy()
            result = pd.DataFrame({
                "corp_name": self._kind_column(frame, "회사명").map(self._kind_text),
                "ticker": self._kind_column(frame, "종목코드").map(self._kind_text),
                "listing_date": pd.to_datetime(self._kind_column(frame, "상장일"), errors="coerce"),
                "listing_type": self._kind_column(frame, "상장유형").map(self._kind_text),
                "security_type": self._kind_column(frame, "증권구분").map(self._kind_text),
                "stock_type": self._kind_column(frame, "주식종류").map(self._kind_text),
                "industry_name": self._kind_column(frame, "업종", "업종명").map(self._kind_text),
                "country": self._kind_column(frame, "국적").map(self._kind_text),
                "lead_underwriter": self._kind_column(
                    frame,
                    "상장주선인/지정자문인",
                    "상장주선인/ 지정자문인",
                    "상장주선인(지정자문인)",
                    "상장주선인",
                ).map(self._kind_text),
                "face_value": self._kind_column(frame, "액면가 (원)", "최초 액면가").map(self._to_number),
                "offering_price": self._kind_column(frame, "공모가 (원)", "공모가").map(self._to_number),
                "offering_amount": self._kind_column(frame, "공모금액 (천원)", "공모금액").map(self._to_number),
                "offering_shares": self._kind_column(frame, "최초상장주식수 (주)", "공모주식수").map(self._to_number),
            })
            result = result.dropna(subset=["corp_name", "listing_date"]).copy()
            result["ticker"] = result["ticker"].fillna("").astype(str).str.strip()
            result["krx_standard_code"] = None
            result["market"] = None
            result["industry_code"] = None
            result["listing_segment"] = None
            classified = result.apply(self._classify_official_listing_event, axis=1, result_type="expand")
            result = pd.concat([result, classified], axis=1)
            result["source_name"] = "KRX_KIND_new_listing_company"
            result["source_url"] = KIND_LISTING_PAGE_URL
            result["source_request_id"] = request_id
            result["collected_at"] = pd.Timestamp.now(tz="Asia/Seoul")
            result["verification_status"] = "official_source_pending_krx_code_enrichment"
            result["event_id"] = (
                "krx_kind|" + result["ticker"].fillna("") + "|" +
                result["listing_date"].dt.strftime("%Y%m%d") + "|" + result["corp_name"].fillna("")
            )
            result = result.drop_duplicates("event_id", keep="last").sort_values("listing_date")
            attempt.update(status="success", response_rows=int(len(result)))
            return result.reindex(columns=columns).reset_index(drop=True)
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            attempt.update(status="failed", error_type=type(exc).__name__, error_message=str(exc)[:500])
            raise RuntimeError(f"KIND 공식 신규상장 수집 실패 ({start:%Y-%m-%d}~{end:%Y-%m-%d}): {exc}") from exc
        finally:
            attempt["finished_at"] = pd.Timestamp.now(tz="Asia/Seoul").isoformat()
            self.official_listing_requests.append(attempt)

    def get_listing_segment_info(self, ticker: str, as_of_date: str | None = None) -> dict[str, Any]:
        """종목기본정보의 소속부 값을 반환한다.

        ``SECT_TP_*``는 실제 산업 분류가 아니라 코스닥 소속부 성격의 값이므로
        ``industry_*`` 필드로 사용하지 않는다. 산업명은 KIND 신규상장 원천의
        별도 ``업종`` 열에서 보존한다.
        """
        bas_dd = self._as_bas_dd(as_of_date or date.today())
        normalized_ticker = self._normalise_ticker(ticker)
        for market, metadata in MARKETS.items():
            for row in self._get_daily_records(metadata["basic_info"], bas_dd):
                if self._normalise_ticker(row.get("ISU_SRT_CD")) == normalized_ticker:
                    return {
                        "ticker": ticker,
                        "market": market,
                        "listing_segment_code": row.get("SECT_TP_CD"),
                        "listing_segment": row.get("SECT_TP_NM"),
                    }
        return {"ticker": ticker, "listing_segment_code": None, "listing_segment": None}

    # 외부 호출 호환성만 유지한다. 새 파이프라인은 이 값을 산업으로 해석하지 않는다.
    def get_sector_info(self, ticker: str, as_of_date: str | None = None) -> dict[str, Any]:
        return self.get_listing_segment_info(ticker, as_of_date)

    # ── 저장용 편의 메서드 ──────────────────────────────────────

    def collect_market_data(self, start_year: int = 2015, end_year: int = 2024) -> dict[str, pd.DataFrame]:
        """KOSPI·KOSDAQ 일별 지수를 원본 저장소에 기록한다."""
        kospi = self.get_index_ohlcv("1", f"{start_year}0101", f"{end_year}1231")
        kosdaq = self.get_index_ohlcv("2", f"{start_year}0101", f"{end_year}1231")
        if not kospi.empty:
            kospi.to_parquet(RAW_DIR / "kospi_index.parquet", index=False)
        if not kosdaq.empty:
            kosdaq.to_parquet(RAW_DIR / "kosdaq_index.parquet", index=False)
        return {"kospi": kospi, "kosdaq": kosdaq}

    def collect_ipo_prices(self, ipo_list: pd.DataFrame) -> pd.DataFrame:
        """IPO 목록의 상장일 시가·종가를 일별 API로 저장한다."""
        records = []
        for row in ipo_list.itertuples(index=False):
            records.append(self.get_listing_day_price(
                ticker=str(row.ticker),
                listing_date=self._as_bas_dd(row.listing_date),
                isu_cd=getattr(row, "isu_cd", None),
                market=getattr(row, "market", None),
                corp_name=getattr(row, "corp_name", None),
            ))
        frame = pd.DataFrame(records)
        frame.to_parquet(RAW_DIR / "ipo_listing_prices.parquet", index=False)
        return frame
