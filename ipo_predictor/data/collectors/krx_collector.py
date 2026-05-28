"""
data/collectors/krx_collector.py
─────────────────────────────────
KRX(한국거래소) 데이터 수집기.

수집 대상:
  1. KOSPI / KOSDAQ 일별 지수 (시장 모멘텀 계산용)
  2. 신규 상장 종목 일별 OHLCV (시초가·종가 수익률 계산용)
  3. IPO 일정 (상장 예정일 목록)

KRX는 공식 API가 없어 정보데이터시스템(data.krx.co.kr)의
AJAX 엔드포인트를 사용한다.
실 운영 시 User-Agent 및 Referer 헤더 설정이 필수이다.
"""

import time
import logging
from datetime import date, timedelta
from typing import Optional

import requests
import pandas as pd

from config import KRX_BASE_URL, RAW_DIR

logger = logging.getLogger(__name__)

REQUEST_DELAY = 0.5
HEADERS = {
    "User-Agent":  "Mozilla/5.0 (compatible; IPO-Research/1.0)",
    "Referer":     "http://data.krx.co.kr/",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
}


class KRXCollector:
    """KRX 정보데이터시스템 수집기"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def _post(self, params: dict) -> dict:
        try:
            resp = self.session.post(KRX_BASE_URL, data=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning("KRX 요청 실패: %s", e)
            return {}

    # ── 지수 데이터 ────────────────────────────────────────────

    def get_index_ohlcv(
        self,
        index_code: str,   # "1" = KOSPI, "2" = KOSDAQ
        start_date: str,   # "20150101"
        end_date:   str,
    ) -> pd.DataFrame:
        """
        KOSPI / KOSDAQ 일별 종가 수집.
        KRX bld 코드: MDCSTAT00301 (지수 시계열)
        """
        params = {
            "bld":      "dbms/MDC/STAT/standard/MDCSTAT00301",
            "indIdx":   index_code,
            "indIdx2":  "001" if index_code == "1" else "001",
            "strtDd":   start_date,
            "endDd":    end_date,
            "share":    "2",
            "money":    "3",
            "csvxls_isNo": "false",
        }
        data = self._post(params)
        items = data.get("output", [])
        if not items:
            logger.warning("지수 데이터 없음: %s %s~%s", index_code, start_date, end_date)
            return pd.DataFrame()

        df = pd.DataFrame(items)
        df = df.rename(columns={
            "TRD_DD":   "date",
            "CLSPRC_IDX": "close",
            "OPNPRC_IDX": "open",
            "HGPRC_IDX":  "high",
            "LWPRC_IDX":  "low",
        })
        df["date"]  = pd.to_datetime(df["date"].str.replace("/", "-"))
        df["close"] = pd.to_numeric(df["close"].str.replace(",", ""), errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)
        df["index_code"] = "KOSPI" if index_code == "1" else "KOSDAQ"
        return df[["date", "index_code", "close"]]

    def get_market_momentum(
        self,
        listing_date: date,
        index: str = "KOSPI",
        windows: list[int] = [5, 20, 60],
    ) -> dict[str, float]:
        """
        특정 상장일 기준 지수 모멘텀 계산.
        listing_date 이전 영업일의 종가를 기준으로 windows일 수익률 반환.
        """
        # 충분한 기간의 데이터 로드
        start = (listing_date - timedelta(days=max(windows) * 2)).strftime("%Y%m%d")
        end   = (listing_date - timedelta(days=1)).strftime("%Y%m%d")

        idx_code = "1" if index == "KOSPI" else "2"
        df = self.get_index_ohlcv(idx_code, start, end)
        if df.empty:
            return {f"{index.lower()}_momentum_{w}d": 0.0 for w in windows}

        closes = df["close"].values
        result = {}
        for w in windows:
            if len(closes) > w:
                ret = closes[-1] / closes[-1 - w] - 1
            else:
                ret = 0.0
            result[f"{index.lower()}_momentum_{w}d"] = round(ret, 6)
        return result

    # ── 신규 상장 종목 가격 ────────────────────────────────────

    def get_listing_day_price(
        self,
        ticker:       str,
        listing_date: str,   # "20230310"
    ) -> dict:
        """
        상장 당일 시초가 / 종가 수집.
        수익률 = (시초가 / 공모가 - 1) 은 feature_engineer에서 계산한다.

        KRX bld: dbms/MDC/STAT/standard/MDCSTAT01501
        """
        params = {
            "bld":       "dbms/MDC/STAT/standard/MDCSTAT01501",
            "isuCd":     ticker,
            "strtDd":    listing_date,
            "endDd":     listing_date,
            "adjStkPrc": "1",
            "csvxls_isNo": "false",
        }
        data = self._post(params)
        items = data.get("output", [])
        if not items:
            return {"ticker": ticker, "open_price": None, "close_price": None}

        row = items[0]
        return {
            "ticker":      ticker,
            "listing_date": listing_date,
            "open_price":  self._parse_num(row.get("TDD_OPNPRC")),
            "close_price": self._parse_num(row.get("TDD_CLSPRC")),
            "high_price":  self._parse_num(row.get("TDD_HGPRC")),
            "low_price":   self._parse_num(row.get("TDD_LWPRC")),
            "volume":      self._parse_num(row.get("ACC_TRDVOL")),
        }

    def get_ipo_calendar(
        self,
        start_date: str,
        end_date:   str,
    ) -> pd.DataFrame:
        """
        KRX 신규 상장 일정 수집.
        반환: ticker, corp_name, listing_date, market (KOSPI/KOSDAQ)
        """
        params = {
            "bld":       "dbms/MDC/STAT/standard/MDCSTAT03901",
            "strtDd":    start_date,
            "endDd":     end_date,
            "csvxls_isNo": "false",
        }
        data = self._post(params)
        items = data.get("output", [])
        if not items:
            return pd.DataFrame()

        df = pd.DataFrame(items)
        df = df.rename(columns={
            "ISU_CD":      "ticker",
            "ISU_ABBRV":   "corp_name",
            "LIST_DD":     "listing_date",
            "MKT_TP_NM":   "market",
            "SECT_TP_NM":  "sector",
        })
        df["listing_date"] = pd.to_datetime(df["listing_date"].str.replace("/", "-"))

        # 동일 상장일 종목 수 계산
        date_counts = df.groupby("listing_date").size().reset_index(name="same_day_ipo_count")
        df = df.merge(date_counts, on="listing_date")

        logger.info("IPO 캘린더 수집: %d건", len(df))
        return df.reset_index(drop=True)

    # ── 섹터 분류 ─────────────────────────────────────────────

    def get_sector_info(self, ticker: str) -> dict:
        """
        종목 섹터(업종) 코드 및 명칭 조회.
        동일 섹터 최근 IPO 수익률 계산에 사용.
        """
        params = {
            "bld":       "dbms/MDC/STAT/standard/MDCSTAT03901",
            "isuCd":     ticker,
            "csvxls_isNo": "false",
        }
        data = self._post(params)
        items = data.get("output", [])
        if not items:
            return {"ticker": ticker, "sector_code": None, "sector_name": None}
        row = items[0]
        return {
            "ticker":       ticker,
            "sector_code":  row.get("SECT_TP_CD"),
            "sector_name":  row.get("SECT_TP_NM"),
        }

    # ── 전체 히스토리 수집 ─────────────────────────────────────

    def collect_market_data(
        self,
        start_year: int = 2015,
        end_year:   int = 2024,
    ) -> dict[str, pd.DataFrame]:
        """
        KOSPI·KOSDAQ 전체 히스토리 지수 데이터 수집.
        """
        start = f"{start_year}0101"
        end   = f"{end_year}1231"

        logger.info("KOSPI 지수 수집 중...")
        kospi = self.get_index_ohlcv("1", start, end)

        logger.info("KOSDAQ 지수 수집 중...")
        kosdaq = self.get_index_ohlcv("2", start, end)

        if not kospi.empty:
            kospi.to_parquet(RAW_DIR / "kospi_index.parquet", index=False)
        if not kosdaq.empty:
            kosdaq.to_parquet(RAW_DIR / "kosdaq_index.parquet", index=False)

        logger.info("시장 지수 저장 완료")
        return {"kospi": kospi, "kosdaq": kosdaq}

    def collect_ipo_prices(
        self,
        ipo_list: pd.DataFrame,  # listing_date, ticker 컬럼 필요
    ) -> pd.DataFrame:
        """
        IPO 목록의 상장일 가격 수집.
        ipo_list: get_ipo_calendar() 결과
        """
        records = []
        total = len(ipo_list)

        for i, row in enumerate(ipo_list.itertuples(), 1):
            ticker = row.ticker
            dt = row.listing_date.strftime("%Y%m%d")
            price_data = self.get_listing_day_price(ticker, dt)
            records.append(price_data)

            if i % 20 == 0:
                logger.info("가격 수집 %d / %d", i, total)
            time.sleep(REQUEST_DELAY)

        df = pd.DataFrame(records)
        out_path = RAW_DIR / "ipo_listing_prices.parquet"
        df.to_parquet(out_path, index=False)
        logger.info("상장일 가격 저장: %d건 → %s", len(df), out_path)
        return df

    @staticmethod
    def _parse_num(val: Optional[str]) -> Optional[float]:
        if val is None:
            return None
        try:
            return float(str(val).replace(",", ""))
        except ValueError:
            return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    collector = KRXCollector()

    # 테스트: 2024년 IPO 캘린더
    cal = collector.get_ipo_calendar("20240101", "20241231")
    print(cal.head(10))
    print(f"\n2024년 상장 종목: {len(cal)}건")
