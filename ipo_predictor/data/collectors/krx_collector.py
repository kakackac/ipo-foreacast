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

from config import KRX_BASE_URL, KRX_ID, KRX_PASSWORD, RAW_DIR

logger = logging.getLogger(__name__)

REQUEST_DELAY = 0.5
HEADERS = {
    "User-Agent":  "Mozilla/5.0 (compatible; IPO-Research/1.0)",
    "Referer":     "https://data.krx.co.kr/contents/MDC/MDI/outerLoader/index.cmd",
    "X-Requested-With": "XMLHttpRequest",
}


class KRXCollector:
    """KRX 정보데이터시스템 수집기"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self._authenticated = False

    @property
    def is_configured(self) -> bool:
        """현재 KRX 데이터 시스템이 요구하는 로그인 정보의 설정 여부."""
        return bool(KRX_ID and KRX_PASSWORD)

    def _authenticate(self) -> None:
        """KRX 세션을 초기화하고 환경변수 자격증명으로 로그인한다."""
        if self._authenticated:
            return
        if not self.is_configured:
            raise RuntimeError("KRX_ID와 KRX_PASSWORD를 설정한 뒤 KRX 수집을 실행하세요.")
        user_agent = HEADERS["User-Agent"]
        try:
            login_page = "https://data.krx.co.kr/contents/MDC/COMS/client/MDCCOMS001.cmd"
            login_view = "https://data.krx.co.kr/contents/MDC/COMS/client/view/login.jsp?site=mdc"
            login_url = "https://data.krx.co.kr/contents/MDC/COMS/client/MDCCOMS001D1.cmd"
            self.session.get(login_page, headers={"User-Agent": user_agent}, timeout=15)
            self.session.get(login_view, headers={"User-Agent": user_agent, "Referer": login_page}, timeout=15)
            response = self.session.post(
                login_url,
                data={"mbrNm": "", "telNo": "", "di": "", "certType": "", "mbrId": KRX_ID, "pw": KRX_PASSWORD},
                headers={"User-Agent": user_agent, "Referer": login_page},
                timeout=15,
            )
            response.raise_for_status()
            if response.json().get("_error_code") != "CD001":
                raise RuntimeError("KRX 로그인에 실패했습니다. KRX_ID와 KRX_PASSWORD를 확인하세요.")
            self._authenticated = True
        except (requests.RequestException, ValueError) as exc:
            raise RuntimeError(f"KRX 로그인 실패: {exc}") from exc

    def _post(self, params: dict) -> dict:
        try:
            self._authenticate()
            resp = self.session.post(KRX_BASE_URL, data=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"KRX 요청 실패: {e}") from e

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
        isu_cd: Optional[str] = None,
    ) -> dict:
        """
        상장 당일 시초가 / 종가 수집.
        수익률 = (시초가 / 공모가 - 1) 은 feature_engineer에서 계산한다.

        KRX bld: dbms/MDC/STAT/standard/MDCSTAT01701
        """
        params = {
            "bld":       "dbms/MDC/STAT/standard/MDCSTAT01701",
            # KRX 조회에는 6자리 단축코드가 아니라 12자리 ISU_CD가 필요한
            # 경우가 있다. 캘린더에서 받은 ISU_CD를 우선 사용한다.
            "isuCd":     isu_cd or ticker,
            "strtDd":    listing_date,
            "endDd":     listing_date,
            "adjStkPrc": "1",
        }
        data = self._post(params)
        items = data.get("output", [])
        if not items:
            return {"ticker": ticker, "open_price": None, "close_price": None}

        row = items[0]
        return {
            "ticker":      ticker,
            "isu_cd":      isu_cd,
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
        KRX 상장종목 기본정보에서 기간 내 신규 상장 종목을 추린다.

        KRX는 이력형 IPO 달력 API를 별도로 공개하지 않으므로, 상장일
        (LIST_DD)이 담긴 공식 기본정보를 기준으로 한다. 현재 상장 중인
        종목을 우선 확보하며, 이후 DART 증권신고서와 정합해 ETF/ETN 등
        비공모 상장을 제거한다.
        반환: ticker, corp_name, listing_date, market (KOSPI/KOSDAQ)
        """
        params = {
            "bld":       "dbms/MDC/STAT/standard/MDCSTAT01901",
            "mktId":     "ALL",
        }
        data = self._post(params)
        items = data.get("OutBlock_1", [])
        if not items:
            return pd.DataFrame()

        df = pd.DataFrame(items)
        df = df.rename(columns={
            "ISU_CD":      "isu_cd",
            "ISU_SRT_CD":  "ticker",
            "ISU_ABBRV":   "corp_name",
            "LIST_DD":     "listing_date",
            "MKT_TP_NM":   "market",
            "SECT_TP_NM":  "sector",
        })
        if "ticker" not in df.columns:
            # 구형 응답에는 단축코드가 없을 수 있다. 이 경우에도 ISU_CD를
            # 보존해 가격 조회가 가능하도록 한다.
            df["ticker"] = df["isu_cd"]
        df["ticker"] = df["ticker"].astype(str).str.strip().str.zfill(6)
        df["listing_date"] = pd.to_datetime(df["listing_date"].astype(str).str.replace("/", "-"))
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        df = df[(df["listing_date"] >= start) & (df["listing_date"] <= end)].copy()

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
            "bld":       "dbms/MDC/STAT/standard/MDCSTAT01901",
            "mktId":     "ALL",
        }
        data = self._post(params)
        items = data.get("OutBlock_1", [])
        if not items:
            return {"ticker": ticker, "sector_code": None, "sector_name": None}
        normalized = str(ticker).zfill(6)
        row = next(
            (item for item in items if str(item.get("ISU_SRT_CD", "")).zfill(6) == normalized),
            None,
        )
        if row is None:
            return {"ticker": ticker, "sector_code": None, "sector_name": None}
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
            ticker = str(row.ticker)
            isu_cd = getattr(row, "isu_cd", None)
            dt = row.listing_date.strftime("%Y%m%d")
            price_data = self.get_listing_day_price(ticker, dt, isu_cd=isu_cd)
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
