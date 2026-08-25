"""과거 IPO의 DART 공시, KRX 실적, 시장 데이터를 하나의 학습셋으로 만든다."""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from config import PROC_DIR, RAW_DIR
from data.collectors.dart_collector import DARTCollector
from data.collectors.krx_collector import KRXCollector
from data.processors.feature_engineer import FeatureEngineer

logger = logging.getLogger(__name__)

MAX_FILING_TO_LISTING_DAYS = 400


class HistoricalIPOPipeline:
    """실제 원천 데이터 수집부터 ``features_all.parquet`` 생성까지 담당한다."""

    def __init__(
        self,
        dart_collector: DARTCollector | None = None,
        krx_collector: KRXCollector | None = None,
        raw_dir: Path = RAW_DIR,
        processed_dir: Path = PROC_DIR,
    ):
        self.dart = dart_collector or DARTCollector()
        self.krx = krx_collector or KRXCollector()
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.manual_dir = self.raw_dir.parent / "manual"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.manual_dir.mkdir(parents=True, exist_ok=True)
        self._document_failures = self._load_cached_frame("dart_document_failures.parquet")

    def run(self, start_year: int, end_year: int, feature_set: str = "phase2") -> dict[str, Any]:
        """수집 결과와 데이터 품질 요약을 반환하고 산출물을 디스크에 저장한다."""
        if start_year > end_year:
            raise ValueError("start_year는 end_year보다 클 수 없습니다.")
        if getattr(self.dart, "is_configured", True) is False:
            raise RuntimeError("DART_API_KEY를 설정한 뒤 실제 수집을 실행하세요.")
        if getattr(self.krx, "is_configured", True) is False:
            raise RuntimeError("KRX_API_KEY를 설정한 뒤 KRX OpenAPI 수집을 실행하세요.")

        calendar = self._collect_calendar(start_year, end_year)
        if calendar.empty:
            raise RuntimeError("KRX 상장 캘린더를 받지 못했습니다. KRX 응답 형식과 네트워크를 확인하세요.")
        calendar.to_parquet(self.raw_dir / "krx_ipo_calendar.parquet", index=False)

        prices = self._collect_listing_prices(calendar)
        prices.to_parquet(self.raw_dir / "ipo_listing_prices.parquet", index=False)
        krx_ipo = self._attach_prices(calendar, prices)

        kospi = self._collect_index_with_cache("1", start_year, end_year, "kospi_index.parquet")
        kosdaq = self._collect_index_with_cache("2", start_year, end_year, "kosdaq_index.parquet")
        kospi.to_parquet(self.raw_dir / "kospi_index.parquet", index=False)
        kosdaq.to_parquet(self.raw_dir / "kosdaq_index.parquet", index=False)

        dart_ipo, financials = self._collect_dart_records(calendar, start_year, end_year)
        self._document_failures.to_parquet(self.raw_dir / "dart_document_failures.parquet", index=False)
        dart_ipo = self._apply_offering_price_overrides(dart_ipo)
        dart_ipo.to_parquet(self.raw_dir / "dart_ipo_raw.parquet", index=False)
        financials.to_parquet(self.raw_dir / "dart_financials.parquet", index=False)
        if dart_ipo.empty:
            raise RuntimeError("KRX 상장 종목과 연결된 DART 증권신고서를 찾지 못했습니다.")

        price_audit = self._build_offering_price_audit(dart_ipo)
        price_audit.to_parquet(self.raw_dir / "dart_offering_price_audit.parquet", index=False)
        verified_statuses = {
            "verified_currency_unit",
            "verified_text_and_structured",
            "verified_structured_api",
            "manual_verified",
        }
        review_queue = price_audit[
            ~price_audit["offering_price_review_status"].fillna("missing").isin(verified_statuses)
        ].copy()
        review_queue.to_parquet(self.raw_dir / "dart_offering_price_review_queue.parquet", index=False)

        engineer = FeatureEngineer(feature_set=feature_set)
        features = engineer.build_features(dart_ipo, krx_ipo, kospi, kosdaq)
        features.to_parquet(self.processed_dir / "features_all.parquet", index=False)

        summary = self._build_summary(calendar, dart_ipo, prices, features)
        with open(self.processed_dir / "data_collection_summary.json", "w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2, default=str)
        logger.info("실제 데이터 파이프라인 완료: %d개 학습 행", len(features))
        return summary

    def _collect_calendar(self, start_year: int, end_year: int) -> pd.DataFrame:
        cached = self._load_cached_frame("krx_ipo_calendar.parquet")
        if not cached.empty and "listing_date" in cached:
            cached = cached.copy()
            cached["listing_date"] = pd.to_datetime(cached["listing_date"], errors="coerce")
            cached = cached[
                cached["listing_date"].dt.year.between(start_year, end_year, inclusive="both")
            ]
        cached_years = set(cached["listing_date"].dropna().dt.year) if not cached.empty else set()
        frames = [cached] if not cached.empty else []
        for year in range(start_year, end_year + 1):
            # 완료된 과거 연도는 불변 이력으로 재사용한다. 마지막 연도는
            # 신규 상장을 반영하기 위해 매 실행마다 갱신한다.
            if year < end_year and year in cached_years:
                continue
            frame = self.krx.get_ipo_calendar(f"{year}0101", f"{year}1231")
            if not frame.empty:
                frames.append(frame)
        if not frames:
            return pd.DataFrame()
        calendar = pd.concat(frames, ignore_index=True)
        calendar["listing_date"] = pd.to_datetime(calendar["listing_date"], errors="coerce")
        return calendar.drop_duplicates(subset=["ticker", "listing_date"], keep="last")

    def _collect_listing_prices(self, calendar: pd.DataFrame) -> pd.DataFrame:
        cached = self._load_cached_frame("ipo_listing_prices.parquet")
        if not cached.empty:
            cached = cached.copy()
            cached["listing_date"] = pd.to_datetime(cached["listing_date"], errors="coerce")
            cached["_cache_key"] = self._listing_key(cached)
            cached = cached.drop_duplicates("_cache_key", keep="last")

        expected = calendar.copy()
        expected["listing_date"] = pd.to_datetime(expected["listing_date"], errors="coerce")
        expected["_cache_key"] = self._listing_key(expected)
        cached_by_key = cached.set_index("_cache_key") if not cached.empty else pd.DataFrame()
        records = []
        reused = 0
        for _, row in expected.iterrows():
            cache_key = row["_cache_key"]
            if not cached.empty and cache_key in cached_by_key.index:
                previous = cached_by_key.loc[cache_key]
                if pd.notna(previous.get("open_price")) and pd.notna(previous.get("close_price")):
                    records.append(previous.to_dict())
                    reused += 1
                    continue
            listing_date = pd.Timestamp(row["listing_date"]).strftime("%Y%m%d")
            ticker = str(row["ticker"])
            isu_cd = row.get("isu_cd")
            market = row.get("market")
            corp_name = row.get("corp_name")
            record = self.krx.get_listing_day_price(
                ticker, listing_date, isu_cd=isu_cd, market=market, corp_name=corp_name
            )
            records.append(record)
        if reused:
            logger.info("KRX 상장일 가격 캐시 재사용: %d건", reused)
        prices = pd.DataFrame(records)
        # 캐시 행은 Timestamp, 방금 받은 API 행은 YYYYMMDD 문자열일 수 있다.
        # Parquet은 같은 열의 혼합 자료형을 저장할 수 없으므로 여기서 통일한다.
        prices["listing_date"] = pd.to_datetime(prices["listing_date"], errors="coerce")
        return prices.drop_duplicates(["ticker", "listing_date"], keep="last")

    def _collect_index_with_cache(
        self, index_code: str, start_year: int, end_year: int, filename: str
    ) -> pd.DataFrame:
        cached = self._load_cached_frame(filename)
        if not cached.empty and "date" in cached:
            cached = cached.copy()
            cached["date"] = pd.to_datetime(cached["date"], errors="coerce")
            cached = cached.dropna(subset=["date"])
        start = pd.Timestamp(f"{start_year}0101")
        end = min(pd.Timestamp(f"{end_year}1231"), pd.Timestamp.today().normalize())
        if not cached.empty:
            covered = cached[(cached["date"] >= start) & (cached["date"] <= end)]
            if not covered.empty:
                next_date = covered["date"].max() + pd.Timedelta(days=1)
                if next_date > start:
                    start = next_date
                logger.info("KRX %s 지수 캐시 재사용: %d건", "KOSPI" if index_code == "1" else "KOSDAQ", len(covered))
        fresh = self.krx.get_index_ohlcv(index_code, start.strftime("%Y%m%d"), end.strftime("%Y%m%d"))
        combined = pd.concat([cached, fresh], ignore_index=True) if not cached.empty else fresh
        if combined.empty:
            return combined
        return combined.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)

    def _load_cached_frame(self, filename: str) -> pd.DataFrame:
        path = self.raw_dir / filename
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_parquet(path)
        except (OSError, ValueError) as exc:
            logger.warning("캐시를 읽지 못해 새로 수집합니다 (%s): %s", filename, exc)
            return pd.DataFrame()

    @staticmethod
    def _listing_key(frame: pd.DataFrame) -> pd.Series:
        return frame["ticker"].astype(str).str.strip() + "|" + pd.to_datetime(
            frame["listing_date"], errors="coerce"
        ).dt.strftime("%Y%m%d")

    @staticmethod
    def _attach_prices(calendar: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        if prices.empty:
            return calendar.copy()
        right = prices.copy()
        if "listing_date" in right.columns:
            right["listing_date"] = pd.to_datetime(right["listing_date"], errors="coerce")
        left = calendar.copy()
        left["listing_date"] = pd.to_datetime(left["listing_date"], errors="coerce")
        return left.merge(right, on=["ticker", "listing_date"], how="left", suffixes=("", "_price"))

    def _collect_dart_records(
        self,
        calendar: pd.DataFrame,
        start_year: int,
        end_year: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        disclosures = []
        for year in range(start_year, end_year + 1):
            frame = self.dart.get_ipo_disclosure_list(f"{year}0101", f"{year}1231")
            if not frame.empty:
                disclosures.append(frame)
        if not disclosures:
            return pd.DataFrame(), pd.DataFrame()

        filings = pd.concat(disclosures, ignore_index=True)
        filings["corp_name_clean"] = filings["corp_name"].map(self._clean_name)
        filings["is_correction"] = filings["report_nm"].fillna("").str.contains("정정", regex=False)
        cached_records = self._load_cached_frame("dart_ipo_raw.parquet")
        cached_by_receipt = {}
        if not cached_records.empty and "rcept_no" in cached_records:
            cached_by_receipt = {
                str(row.rcept_no): row._asdict()
                for row in cached_records.drop_duplicates("rcept_no", keep="last").itertuples(index=False)
            }
        calendar = calendar.copy()
        calendar["corp_name_clean"] = calendar["corp_name"].map(self._clean_name)
        calendar["listing_date"] = pd.to_datetime(calendar["listing_date"], errors="coerce")

        records: list[dict[str, Any]] = []
        financial_rows: list[pd.DataFrame] = []
        for listing in calendar.itertuples(index=False):
            candidates = filings[filings["corp_name_clean"] == listing.corp_name_clean].copy()
            candidates = candidates[candidates["rcept_dt"] <= listing.listing_date]
            candidates = candidates[
                (listing.listing_date - candidates["rcept_dt"]).dt.days.between(0, MAX_FILING_TO_LISTING_DAYS)
            ]
            if candidates.empty:
                continue
            # 같은 날짜의 신고서가 여러 개면 정정 신고서를 우선한다. 날짜가
            # 더 늦은 신고서는 정정 여부와 관계없이 최신 공시가 우선이다.
            filing = candidates.sort_values(
                ["rcept_dt", "is_correction", "rcept_no"], ascending=[True, True, True]
            ).iloc[-1]
            cached = cached_by_receipt.get(str(filing.rcept_no))
            # 구조화 slprc 대조가 끝난 행만 재사용한다. 이전 버전의 캐시는
            # 이번 한 번 다시 검사해 공모가 검증 근거를 보완한다.
            if cached is not None and cached.get("structured_price_check") is not None:
                records.append(cached)
                continue
            if str(filing.rcept_no) in set(self._document_failures.get("rcept_no", pd.Series(dtype=str)).astype(str)):
                logger.info("원문 부재로 기록된 신고서는 다시 호출하지 않습니다: %s", filing.rcept_no)
                continue
            try:
                offering = self.dart.get_offering_info(str(filing.rcept_no))
            except RuntimeError as exc:
                if "<status>014</status>" in str(exc):
                    self._record_document_failure(filing, "014_file_not_found")
                logger.warning("신고서 원문 파싱 실패 (%s): %s", filing.corp_name, exc)
                continue

            structured_prices = self.dart.get_equity_offering_prices(
                str(filing.corp_code),
                pd.Timestamp(filing.rcept_dt).strftime("%Y%m%d"),
                pd.Timestamp(listing.listing_date).strftime("%Y%m%d"),
            )
            structured = next(
                (item for item in structured_prices if item["rcept_no"] == str(filing.rcept_no)), None
            )
            is_final_price_disclosure = "발행조건확정" in str(filing.report_nm)
            offering = self._reconcile_structured_offering_price(
                offering, structured, is_final_price_disclosure
            )

            demand_rcept_no = None
            demand = {}
            try:
                demand_rcept_no = self.dart.find_demand_forecast_disclosure(
                    str(filing.corp_code),
                    pd.Timestamp(filing.rcept_dt).strftime("%Y%m%d"),
                    pd.Timestamp(listing.listing_date).strftime("%Y%m%d"),
                )
                if demand_rcept_no:
                    demand = self.dart.get_demand_forecast(str(filing.corp_code), demand_rcept_no)
            except RuntimeError as exc:
                logger.warning("수요예측 원문 파싱 실패 (%s): %s", filing.corp_name, exc)

            financial_summary, collected_financials = self._collect_financials(
                str(filing.corp_code), pd.Timestamp(listing.listing_date)
            )
            if not collected_financials.empty:
                financial_rows.append(collected_financials)
            records.append({
                "rcept_no": str(filing.rcept_no),
                "corp_code": str(filing.corp_code),
                "corp_name": filing.corp_name,
                "rcept_dt": filing.rcept_dt,
                "filing_report_nm": filing.report_nm,
                "filing_is_correction": bool(filing.is_correction),
                "filing_is_final_price_disclosure": is_final_price_disclosure,
                "filing_candidate_count": len(candidates),
                "demand_rcept_no": demand_rcept_no,
                **offering,
                **{key: value for key, value in demand.items() if key != "corp_code"},
                **self._compare_offering_sources(offering, demand),
                **financial_summary,
            })

        financials = pd.concat(financial_rows, ignore_index=True) if financial_rows else pd.DataFrame(
            columns=["corp_code", "listing_date", "year", "account_name_en", "amount"]
        )
        return pd.DataFrame(records), financials

    @staticmethod
    def _reconcile_structured_offering_price(
        offering: dict[str, Any], structured: dict[str, Any] | None, is_final_price_disclosure: bool = True
    ) -> dict[str, Any]:
        """원문 공모가와 DART 구조화 모집가액을 접수번호 단위로 대조한다."""
        result = offering.copy()
        result["dart_structured_offering_price"] = None
        result["dart_structured_security_type"] = None
        result["structured_price_check"] = "not_available"
        if structured is None:
            return result

        structured_price = structured["offering_price"]
        result["dart_structured_offering_price"] = structured_price
        result["dart_structured_security_type"] = structured.get("security_type")
        if not is_final_price_disclosure:
            result["structured_price_check"] = "structured_price_unverified_report_type"
            return result
        text_price = result.get("offering_price")
        if text_price is None:
            result["offering_price"] = structured_price
            result["offering_price_extracted_amount"] = structured_price
            result["offering_price_review_status"] = "verified_structured_api"
            result["offering_price_parse_method"] = "dart_estkRs_slprc"
            result["structured_price_check"] = "structured_price_used"
        elif float(text_price) == float(structured_price):
            result["offering_price_review_status"] = "verified_text_and_structured"
            result["structured_price_check"] = "matches_structured_price"
        else:
            result["offering_price_review_status"] = "needs_review_structured_mismatch"
            result["structured_price_check"] = "mismatch_with_structured_price"
        return result

    def _record_document_failure(self, filing: pd.Series, reason: str) -> None:
        record = pd.DataFrame([{
            "rcept_no": str(filing.rcept_no),
            "corp_name": filing.corp_name,
            "rcept_dt": filing.rcept_dt,
            "reason": reason,
            "recorded_at": pd.Timestamp.now(),
        }])
        self._document_failures = pd.concat([self._document_failures, record], ignore_index=True)
        self._document_failures = self._document_failures.drop_duplicates("rcept_no", keep="last")

    @staticmethod
    def _compare_offering_sources(offering: dict[str, Any], demand: dict[str, Any]) -> dict[str, Any]:
        """신고서·희망밴드·수요예측 원문의 공모가를 대조한다."""
        price = pd.to_numeric(pd.Series([offering.get("offering_price")]), errors="coerce").iloc[0]
        demand_price = pd.to_numeric(pd.Series([demand.get("demand_offering_price")]), errors="coerce").iloc[0]
        low = pd.to_numeric(pd.Series([offering.get("price_band_low")]), errors="coerce").iloc[0]
        high = pd.to_numeric(pd.Series([offering.get("price_band_high")]), errors="coerce").iloc[0]

        if pd.isna(price) or pd.isna(low) or pd.isna(high):
            band_check = "not_available"
        elif low <= price <= high:
            band_check = "within_price_band"
        elif price > high:
            band_check = "above_price_band"
        else:
            band_check = "below_price_band"

        if pd.isna(price) or pd.isna(demand_price):
            demand_check = "not_available"
        elif price == demand_price:
            demand_check = "matches_demand_disclosure"
        else:
            demand_check = "mismatch_with_demand_disclosure"

        status = offering.get("offering_price_review_status", "missing")
        if status == "verified_currency_unit" and demand_check == "mismatch_with_demand_disclosure":
            status = "needs_review_source_mismatch"
        return {
            "price_band_check": band_check,
            "demand_price_check": demand_check,
            "offering_price_review_status": status,
        }

    def _apply_offering_price_overrides(self, dart_ipo: pd.DataFrame) -> pd.DataFrame:
        """사람이 원문을 확인해 승인한 공모가만 학습용 값으로 반영한다."""
        if dart_ipo.empty:
            return dart_ipo
        override_path = self.manual_dir / "offering_price_overrides.csv"
        if not override_path.exists():
            return dart_ipo

        overrides = pd.read_csv(override_path, dtype={"rcept_no": str})
        required = {"rcept_no", "offering_price", "decision"}
        missing = required - set(overrides.columns)
        if missing:
            raise RuntimeError(
                f"공모가 검토 파일에 필요한 열이 없습니다: {', '.join(sorted(missing))}"
            )
        overrides["decision"] = overrides["decision"].fillna("").str.strip().str.lower()
        overrides["offering_price"] = pd.to_numeric(overrides["offering_price"], errors="coerce")
        approved = overrides[
            (overrides["decision"] == "verified") & overrides["offering_price"].notna()
        ].drop_duplicates("rcept_no", keep="last")
        if approved.empty:
            return dart_ipo

        result = dart_ipo.copy()
        result["rcept_no"] = result["rcept_no"].astype(str)
        approved = approved.set_index("rcept_no")
        for index, row in result.iterrows():
            override = approved.loc[row["rcept_no"]] if row["rcept_no"] in approved.index else None
            if override is None:
                continue
            result.at[index, "offering_price"] = override["offering_price"]
            result.at[index, "offering_price_extracted_amount"] = override["offering_price"]
            result.at[index, "offering_price_review_status"] = "manual_verified"
            result.at[index, "offering_price_parse_method"] = "manual_audit_override"
            result.at[index, "offering_price_audit_context"] = str(override.get("note", "manual verification"))
            result.at[index, "offering_price_range_warning"] = bool(
                override["offering_price"] < 100 or override["offering_price"] > 10_000_000
            )
        logger.info("원문 검토로 확정 공모가 %d건을 반영했습니다.", len(approved))
        return result

    @staticmethod
    def _build_offering_price_audit(dart_ipo: pd.DataFrame) -> pd.DataFrame:
        columns = [
            "corp_name", "rcept_no", "rcept_dt", "filing_report_nm", "filing_is_correction",
            "filing_is_final_price_disclosure",
            "filing_candidate_count", "demand_rcept_no", "offering_price",
            "offering_price_extracted_amount", "offering_price_review_status",
            "offering_price_parse_method", "offering_price_range_warning",
            "offering_price_audit_context", "price_band_low", "price_band_high",
            "price_band_check", "dart_structured_offering_price", "dart_structured_security_type",
            "structured_price_check", "demand_offering_price", "demand_price_check",
            "demand_offering_price_context",
        ]
        return dart_ipo.reindex(columns=columns).copy()

    def _collect_financials(self, corp_code: str, listing_date: pd.Timestamp) -> tuple[dict[str, Any], pd.DataFrame]:
        frames = []
        # 상장 직전 시점에 공개돼 있던 최근 3개 사업연도만 사용한다.
        for year in range(listing_date.year - 1, listing_date.year - 4, -1):
            frame = self.dart.get_financial_statements(corp_code, year)
            if not frame.empty:
                frame = frame.copy()
                frame["corp_code"] = corp_code
                frame["listing_date"] = listing_date
                frames.append(frame)
        if not frames:
            return {}, pd.DataFrame()

        financials = pd.concat(frames, ignore_index=True)
        latest_year = int(financials["year"].max())
        latest = financials[financials["year"] == latest_year].drop_duplicates("account_name_en", keep="first")
        summary = latest.set_index("account_name_en")["amount"].to_dict()
        summary["financial_as_of_year"] = latest_year

        revenue_history = financials[financials["account_name_en"] == "revenue"].sort_values("year")
        if len(revenue_history) >= 2:
            old = revenue_history.iloc[0]
            recent = revenue_history.iloc[-1]
            years = int(recent.year - old.year)
            if years > 0 and old.amount > 0 and recent.amount > 0:
                summary["revenue_growth_3y"] = (recent.amount / old.amount) ** (1 / years) - 1
            summary["revenue_3y_ago"] = old.amount
        return summary, financials

    @staticmethod
    def _clean_name(value: object) -> str:
        text = str(value or "")
        text = text.replace("주식회사", "").replace("㈜", "").replace("(주)", "").replace("(株)", "")
        return "".join(char for char in text.upper() if char.isalnum())

    @staticmethod
    def _build_summary(
        calendar: pd.DataFrame,
        dart_ipo: pd.DataFrame,
        prices: pd.DataFrame,
        features: pd.DataFrame,
    ) -> dict[str, Any]:
        offering_price = pd.to_numeric(
            dart_ipo.get("offering_price", pd.Series(dtype=float)), errors="coerce"
        )
        expected_price_range = offering_price.between(100, 10_000_000)
        price_status = dart_ipo.get("offering_price_review_status", pd.Series(dtype=str)).fillna("missing")
        verified_statuses = {
            "verified_currency_unit",
            "verified_text_and_structured",
            "verified_structured_api",
            "manual_verified",
        }
        price_band_low = pd.to_numeric(
            dart_ipo.get("price_band_low", pd.Series(dtype=float)), errors="coerce"
        )
        price_band_high = pd.to_numeric(
            dart_ipo.get("price_band_high", pd.Series(dtype=float)), errors="coerce"
        )
        open_return = pd.to_numeric(
            features.get("open_return_pct", pd.Series(dtype=float)), errors="coerce"
        )
        return {
            "calendar_rows": len(calendar),
            "dart_matched_rows": len(dart_ipo),
            "listing_price_rows": len(prices),
            "listing_open_price_rows": int(prices.get("open_price", pd.Series(dtype=float)).notna().sum()),
            "listing_close_price_rows": int(prices.get("close_price", pd.Series(dtype=float)).notna().sum()),
            "feature_rows": len(features),
            "open_target_rows": int(features.get("open_return_pct", pd.Series(dtype=float)).notna().sum()),
            "close_target_rows": int(features.get("close_return_pct", pd.Series(dtype=float)).notna().sum()),
            "offering_price_rows": int(offering_price.notna().sum()),
            "offering_price_within_expected_range_rows": int(expected_price_range.sum()),
            "offering_price_range_warning_rows": int((offering_price.notna() & ~expected_price_range).sum()),
            "offering_price_needs_review_rows": int((~price_status.isin(verified_statuses)).sum()),
            "offering_price_manual_verified_rows": int((price_status == "manual_verified").sum()),
            "price_band_rows": int((price_band_low.notna() & price_band_high.notna()).sum()),
            "demand_ratio_rows": int(dart_ipo.get("institutional_demand_ratio", pd.Series(dtype=float)).notna().sum()),
            "lockup_rows": int(dart_ipo.get("lockup_6m_ratio", pd.Series(dtype=float)).notna().sum()),
            "financial_revenue_rows": int(dart_ipo.get("revenue", pd.Series(dtype=float)).notna().sum()),
            "extreme_open_return_rows": int((open_return.abs() > 200).sum()),
            "source": "OpenDART + KRX OpenAPI",
        }
