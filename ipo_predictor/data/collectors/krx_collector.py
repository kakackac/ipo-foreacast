"""KRX OpenAPI를 이용해 IPO 사후 실적과 시장 데이터를 수집한다.

승인된 일별 API만 사용한다. KRX 웹사이트 로그인이나 개인 ID/비밀번호는
사용하지 않으며, API 키는 ``KRX_API_KEY`` 환경변수로만 전달한다.
"""

import logging
import time
from datetime import date, datetime, timedelta
from typing import Any, Optional

import pandas as pd
import requests

from config import KRX_API_KEY, KRX_OPENAPI_BASE_URL, RAW_DIR

logger = logging.getLogger(__name__)

REQUEST_DELAY = 0.1
# KRX 일반 인증키의 일일 호출 한도보다 여유를 두고 중단한다.
MAX_REQUESTS_PER_RUN = 8_000

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
        except (requests.RequestException, ValueError) as exc:
            raise RuntimeError(f"KRX OpenAPI 요청 실패 ({endpoint}, {bas_dd}): {exc}") from exc

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

    @staticmethod
    def _normalise_index_name(value: object) -> str:
        return str(value or "").replace(" ", "").upper()

    @staticmethod
    def _normalise_market(market: str | None) -> str | None:
        if market is None:
            return None
        normalised = str(market).upper().strip()
        return normalised if normalised in MARKETS else None

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
            return {f"{index.lower()}_momentum_{window}d": 0.0 for window in windows}

        closes = frame["close"].to_numpy()
        result = {}
        for window in windows:
            result[f"{index.lower()}_momentum_{window}d"] = (
                round(closes[-1] / closes[-1 - window] - 1, 6) if len(closes) > window else 0.0
            )
        return result

    # ── 상장 종목과 상장일 가격 ─────────────────────────────────

    def get_listing_day_price(
        self,
        ticker: str,
        listing_date: str,
        isu_cd: Optional[str] = None,
        market: str | None = None,
    ) -> dict[str, Any]:
        """상장일 시가·정규장 종가를 수집한다.

        ``isu_cd``가 있으면 KRX 표준 종목코드로 우선 비교한다. 시장을 알 수
        없을 때만 KOSPI와 KOSDAQ를 모두 조회해, API 호출을 최소화한다.
        """
        selected_market = self._normalise_market(market)
        markets = [selected_market] if selected_market else list(MARKETS)
        normalized_ticker = self._normalise_ticker(ticker)
        normalized_isu_cd = str(isu_cd or "").strip().upper()

        for market_name in markets:
            rows = self._get_daily_records(
                MARKETS[market_name]["daily_price"], self._as_bas_dd(listing_date)
            )
            for row in rows:
                row_isu_cd = str(row.get("ISU_CD", "")).strip().upper()
                row_ticker = self._normalise_ticker(row.get("ISU_SRT_CD"))
                if normalized_isu_cd and row_isu_cd == normalized_isu_cd:
                    return self._price_record(ticker, listing_date, isu_cd, row, market_name)
                if row_ticker == normalized_ticker:
                    return self._price_record(ticker, listing_date, isu_cd or row_isu_cd, row, market_name)

        logger.warning("상장일 가격을 찾지 못했습니다: %s (%s)", ticker, listing_date)
        return {
            "ticker": ticker,
            "isu_cd": isu_cd,
            "listing_date": self._as_bas_dd(listing_date),
            "open_price": None,
            "close_price": None,
            "high_price": None,
            "low_price": None,
            "volume": None,
        }

    def _price_record(
        self,
        ticker: str,
        listing_date: str,
        isu_cd: str | None,
        row: dict[str, Any],
        market: str,
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
        }

    def get_ipo_calendar(self, start_date: str, end_date: str) -> pd.DataFrame:
        """기간 내 신규 상장 종목을 KRX 종목기본정보에서 찾는다.

        이 API는 기준일 시점의 상장 종목을 돌려준다. 따라서 파이프라인은
        연도별 마지막 유효 기준일을 조회하고 ``LIST_DD``로 해당 연도 IPO를
        추린 뒤, DART 증권신고서와 정합해 비공모 상장을 제거한다.
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
        columns = ["ticker", "isu_cd", "corp_name", "listing_date", "market", "sector", "same_day_ipo_count"]
        return frame.reindex(columns=columns).drop_duplicates(["ticker", "listing_date"]).reset_index(drop=True)

    def get_sector_info(self, ticker: str, as_of_date: str | None = None) -> dict[str, Any]:
        """현재 또는 지정 기준일의 종목 업종 정보를 반환한다."""
        bas_dd = self._as_bas_dd(as_of_date or date.today())
        normalized_ticker = self._normalise_ticker(ticker)
        for market, metadata in MARKETS.items():
            for row in self._get_daily_records(metadata["basic_info"], bas_dd):
                if self._normalise_ticker(row.get("ISU_SRT_CD")) == normalized_ticker:
                    return {
                        "ticker": ticker,
                        "market": market,
                        "sector_code": row.get("SECT_TP_CD"),
                        "sector_name": row.get("SECT_TP_NM"),
                    }
        return {"ticker": ticker, "sector_code": None, "sector_name": None}

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
            ))
        frame = pd.DataFrame(records)
        frame.to_parquet(RAW_DIR / "ipo_listing_prices.parquet", index=False)
        return frame
