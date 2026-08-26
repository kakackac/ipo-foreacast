"""과거 IPO의 DART 공시, KRX 실적, 시장 데이터를 하나의 학습셋으로 만든다."""

import json
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from config import PROC_DIR, RAW_DIR
from data.collectors.dart_collector import DARTCollector
from data.collectors.krx_collector import KRXCollector
from data.processors.feature_engineer import FeatureEngineer

logger = logging.getLogger(__name__)

MAX_FILING_TO_LISTING_DAYS = 400
STRUCTURED_PRICE_CHECK_VERSION = 3
OFFERING_PRICE_PARSER_VERSION = 2
DART_LINEAGE_VERSION = 1
DOCUMENT_RETRY_AFTER_DAYS = 7


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
        self._demand_document_failures = self._load_cached_frame("dart_demand_document_failures.parquet")
        self._lineage_rows: list[dict[str, Any]] = []

    def run(self, start_year: int, end_year: int, feature_set: str = "phase2") -> dict[str, Any]:
        """수집 결과와 데이터 품질 요약을 반환하고 산출물을 디스크에 저장한다."""
        if start_year > end_year:
            raise ValueError("start_year는 end_year보다 클 수 없습니다.")
        if getattr(self.dart, "is_configured", True) is False:
            raise RuntimeError("DART_API_KEY를 설정한 뒤 실제 수집을 실행하세요.")
        if getattr(self.krx, "is_configured", True) is False:
            raise RuntimeError("KRX_API_KEY를 설정한 뒤 KRX OpenAPI 수집을 실행하세요.")

        calendar, event_manifest = self.collect_official_event_master(start_year, end_year)
        if calendar.empty:
            raise RuntimeError("KRX 공식 신규상장 이벤트를 받지 못했습니다. KIND 응답과 네트워크를 확인하세요.")

        prices = self._collect_listing_prices(calendar)
        prices.to_parquet(self.raw_dir / "ipo_listing_prices.parquet", index=False)
        krx_ipo = self._attach_prices(calendar, prices)
        # 상장일 가격 조회에서 확인된 KRX 표준코드·시장 정보를 이벤트 마스터에
        # 되돌려 다음 실행의 이벤트 식별과 정합에 재사용한다.
        krx_ipo.to_parquet(self.raw_dir / "krx_official_event_master.parquet", index=False)

        kospi = self._collect_index_with_cache("1", start_year, end_year, "kospi_index.parquet")
        kosdaq = self._collect_index_with_cache("2", start_year, end_year, "kosdaq_index.parquet")
        kospi.to_parquet(self.raw_dir / "kospi_index.parquet", index=False)
        kosdaq.to_parquet(self.raw_dir / "kosdaq_index.parquet", index=False)

        dart_ipo, financials = self._collect_dart_records(calendar, start_year, end_year)
        self._document_failures.to_parquet(self.raw_dir / "dart_document_failures.parquet", index=False)
        self._demand_document_failures.to_parquet(
            self.raw_dir / "dart_demand_document_failures.parquet", index=False
        )
        lineage = pd.DataFrame(self._lineage_rows)
        lineage.to_parquet(self.raw_dir / "dart_disclosure_lineage.parquet", index=False)
        self._build_document_failure_audit().to_parquet(
            self.raw_dir / "dart_document_failure_audit.parquet", index=False
        )
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
        engineer.build_feature_observations(features).to_parquet(
            self.processed_dir / "feature_observations.parquet", index=False
        )
        time_audit = self._build_feature_time_audit(features)
        time_audit.to_parquet(self.processed_dir / "feature_time_validation.parquet", index=False)

        summary = self._build_summary(calendar, dart_ipo, prices, features)
        summary["event_master_source"] = "KRX_KIND_new_listing_company"
        summary["event_class_counts"] = {
            str(key): int(value)
            for key, value in calendar.get("event_class", pd.Series(dtype=str)).value_counts(dropna=False).items()
        }
        summary["legacy_list_dd_candidate_rows"] = event_manifest["legacy_candidate_rows"]
        summary["event_master_manifest"] = event_manifest["path"]
        summary["future_information_violations"] = int(time_audit["is_future_information"].sum())
        summary["feature_time_validation_rows"] = int(len(time_audit))
        with open(self.processed_dir / "data_collection_summary.json", "w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2, default=str)
        logger.info("실제 데이터 파이프라인 완료: %d개 학습 행", len(features))
        return summary

    def collect_official_event_master(
        self, start_year: int, end_year: int, force_refresh: bool = False
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """KIND 공식 신규상장 이벤트 마스터와 실행 매니페스트를 만든다.

        레거시 ``LIST_DD`` 산출물은 삭제하지 않고 별도 파일로 보존한다. 이
        메서드는 DART·가격·지수 수집을 호출하지 않으므로 2026년 공식 이벤트
        저장 여부를 독립적으로 검증하는 데도 사용한다.
        """
        run_id = f"krx_event_master_{pd.Timestamp.now(tz='Asia/Seoul'):%Y%m%dT%H%M%S}_{uuid4().hex[:8]}"
        manifest: dict[str, Any] = {
            "run_id": run_id,
            "started_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
            "requested_start_year": start_year,
            "requested_end_year": end_year,
            "source": "KRX_KIND_new_listing_company",
            "force_refresh": force_refresh,
            "cache_used_years": [],
            "fetched_years": [],
            "yearly_rows": {},
            "status": "started",
        }
        manifest_dir = self.raw_dir / "collection_manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / f"{run_id}.json"

        # 기존 파일은 종목기본정보 LIST_DD 후보일 수 있다. 이름을 바꾸지 않고
        # 별도 보존해 공식 이벤트와 비교 가능하게 만든다.
        legacy = self._load_cached_frame("legacy_list_dd_candidate.parquet")
        if legacy.empty:
            prior_calendar = self._load_cached_frame("krx_ipo_calendar.parquet")
            if not prior_calendar.empty:
                legacy = prior_calendar.copy()
                legacy["legacy_candidate_type"] = "legacy_list_dd_candidate"
                legacy["legacy_source"] = "krx_openapi_issue_master_list_dd"
        if not legacy.empty:
            legacy.to_parquet(self.raw_dir / "legacy_list_dd_candidate.parquet", index=False)
        manifest["legacy_candidate_rows"] = int(len(legacy))

        cached = self._load_cached_frame("krx_official_event_master.parquet")
        if not cached.empty and "listing_date" in cached:
            cached = cached.copy()
            cached["listing_date"] = pd.to_datetime(cached["listing_date"], errors="coerce")
            cached = cached[
                cached["listing_date"].dt.year.between(start_year, end_year, inclusive="both")
            ]
        cached_years = set(cached["listing_date"].dropna().dt.year) if not cached.empty else set()
        frames = [cached] if not cached.empty else []
        try:
            for year in range(start_year, end_year + 1):
                # 과거 공식 스냅샷은 재사용하지만, 마지막 요청 연도는 매번 공식
                # KIND에서 다시 읽어 2026 누락을 재현 가능하게 검증한다.
                if not force_refresh and year < end_year and year in cached_years:
                    manifest["cache_used_years"].append(year)
                    continue
                end = min(pd.Timestamp(f"{year}1231"), pd.Timestamp.today().normalize())
                frame = self.krx.get_official_listing_events(f"{year}0101", end.strftime("%Y%m%d"))
                manifest["fetched_years"].append(year)
                manifest["yearly_rows"][str(year)] = int(len(frame))
                if not frame.empty:
                    frames.append(frame)
            if not frames:
                calendar = pd.DataFrame()
            else:
                calendar = pd.concat(frames, ignore_index=True)
                calendar["listing_date"] = pd.to_datetime(calendar["listing_date"], errors="coerce")
                calendar = calendar.drop_duplicates(subset=["event_id"], keep="last")
                calendar["same_day_ipo_count"] = calendar.groupby("listing_date")["event_id"].transform("size")
                calendar = calendar.sort_values("listing_date").reset_index(drop=True)
                calendar.to_parquet(self.raw_dir / "krx_official_event_master.parquet", index=False)
                self._write_legacy_comparison(calendar, legacy)
            manifest["official_event_rows"] = int(len(calendar))
            manifest["official_listing_requests"] = getattr(self.krx, "official_listing_requests", [])
            manifest["status"] = "success"
            return calendar, {"path": str(manifest_path), **manifest}
        except Exception as exc:
            manifest["status"] = "failed"
            manifest["error_type"] = type(exc).__name__
            manifest["error_message"] = str(exc)[:500]
            manifest["official_listing_requests"] = getattr(self.krx, "official_listing_requests", [])
            raise
        finally:
            manifest["finished_at"] = pd.Timestamp.now(tz="Asia/Seoul").isoformat()
            manifest["official_listing_requests"] = getattr(self.krx, "official_listing_requests", [])
            manifest["path"] = str(manifest_path)
            with open(manifest_path, "w", encoding="utf-8") as file:
                json.dump(manifest, file, ensure_ascii=False, indent=2, default=str)
            with open(self.raw_dir / "latest_collection_manifest.json", "w", encoding="utf-8") as file:
                json.dump(manifest, file, ensure_ascii=False, indent=2, default=str)

    def _write_legacy_comparison(self, official: pd.DataFrame, legacy: pd.DataFrame) -> None:
        """공식 이벤트와 기존 LIST_DD 후보의 행 단위 대조를 저장한다."""
        def key(frame: pd.DataFrame) -> pd.Series:
            names = frame["corp_name"].fillna("").astype(str).str.replace(r"[^0-9A-Za-z가-힣]", "", regex=True).str.upper()
            dates = pd.to_datetime(frame["listing_date"], errors="coerce").dt.strftime("%Y%m%d").fillna("")
            return names + "|" + dates

        left = official[["event_id", "corp_name", "listing_date", "event_class"]].copy()
        left["event_key"] = key(left)
        right = legacy.reindex(columns=["corp_name", "listing_date"]).copy()
        right["event_key"] = key(right)
        right["legacy_present"] = True
        comparison = left.merge(right[["event_key", "legacy_present"]], on="event_key", how="outer", indicator=True)
        comparison["comparison_status"] = comparison["_merge"].map({
            "both": "matched", "left_only": "official_only", "right_only": "legacy_only",
        })
        comparison = comparison.drop(columns="_merge")
        comparison.to_parquet(self.raw_dir / "krx_official_vs_legacy_comparison.parquet", index=False)


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
        merged = left.merge(right, on=["ticker", "listing_date"], how="left", suffixes=("", "_price"))
        if "market_price" in merged.columns:
            merged["market"] = merged.get("market", pd.Series(index=merged.index, dtype=object)).combine_first(
                merged["market_price"]
            )
        if "krx_standard_code" in merged.columns and "isu_cd" in merged.columns:
            merged["krx_standard_code"] = merged["krx_standard_code"].combine_first(merged["isu_cd"])
        elif "isu_cd" in merged.columns:
            merged["krx_standard_code"] = merged["isu_cd"]
        if "verification_status" in merged.columns:
            enriched = merged.get("krx_standard_code", pd.Series(index=merged.index, dtype=object)).notna()
            merged.loc[enriched, "verification_status"] = "official_source_krx_code_enriched"
        return merged

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
        filings["is_final_conditions"] = filings["report_nm"].fillna("").str.contains("발행조건확정", regex=False)
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
            event_id = str(getattr(listing, "event_id", ""))
            if candidates.empty:
                self._lineage_rows.append({
                    "event_id": event_id,
                    "ticker": getattr(listing, "ticker", None),
                    "krx_standard_code": getattr(listing, "krx_standard_code", None),
                    "listing_date": listing.listing_date,
                    "corp_name": listing.corp_name,
                    "lineage_status": "actual_related_disclosure_not_found",
                    "lineage_version": DART_LINEAGE_VERSION,
                    "source_name": "OpenDART_list_C_C001",
                    "recorded_at": pd.Timestamp.now(tz="Asia/Seoul"),
                })
                continue

            # estkRs의 날짜 기준은 해당 신고서의 "최초접수일"이다. 발행조건
            # 확정본의 접수일만 시작점으로 쓰면 초기 신고서로 묶인 구조화 값을
            # 놓칠 수 있으므로 IPO 연결 허용 구간 전체를 조회한다. 다만 공모가
            # 승인에는 아래에서 선택한 동일 접수번호만 사용한다.
            candidates = candidates.sort_values(
                ["is_final_conditions", "rcept_dt", "is_correction", "rcept_no"],
                ascending=[False, False, False, False],
            ).reset_index(drop=True)
            corp_codes = candidates["corp_code"].dropna().astype(str).unique()
            if len(corp_codes) != 1:
                for _, candidate in candidates.iterrows():
                    self._lineage_rows.append({
                        "event_id": event_id,
                        "ticker": getattr(listing, "ticker", None),
                        "krx_standard_code": getattr(listing, "krx_standard_code", None),
                        "listing_date": listing.listing_date,
                        "corp_name": candidate.corp_name,
                        "corp_code": str(candidate.corp_code),
                        "rcept_no": str(candidate.rcept_no),
                        "rcept_dt": candidate.rcept_dt,
                        "filing_report_nm": candidate.report_nm,
                        "match_method": "corp_name_bootstrap_ambiguous_corp_code",
                        "lineage_validation_status": "review_required_multiple_dart_corp_codes",
                        "source_name": "OpenDART_list_C_C001",
                        "source_url": "https://opendart.fss.or.kr/api/list.json",
                        "lineage_version": DART_LINEAGE_VERSION,
                        "attempt_status": "not_used_ambiguous_corp_code",
                        "recorded_at": pd.Timestamp.now(tz="Asia/Seoul"),
                    })
                continue
            structured_start = pd.Timestamp(listing.listing_date) - pd.Timedelta(
                days=MAX_FILING_TO_LISTING_DAYS
            )
            try:
                structured_prices = self.dart.get_equity_offering_prices(
                    str(candidates.iloc[0].corp_code),
                    structured_start.strftime("%Y%m%d"),
                    pd.Timestamp(listing.listing_date).strftime("%Y%m%d"),
                )
            except RuntimeError as exc:
                structured_prices = []
                logger.warning("DART 구조화 지분증권 조회 실패 (%s): %s", listing.corp_name, exc)

            lineage_entries: list[dict[str, Any]] = []
            for rank, candidate in candidates.iterrows():
                entry = {
                    "event_id": event_id,
                    "ticker": getattr(listing, "ticker", None),
                    "krx_standard_code": getattr(listing, "krx_standard_code", None),
                    "listing_date": listing.listing_date,
                    "corp_name": candidate.corp_name,
                    "corp_code": str(candidate.corp_code),
                    "rcept_no": str(candidate.rcept_no),
                    "rcept_dt": candidate.rcept_dt,
                    "filing_report_nm": candidate.report_nm,
                    "is_correction": bool(candidate.is_correction),
                    "is_final_conditions": bool(candidate.is_final_conditions),
                    "selection_rank": rank + 1,
                    "match_method": "corp_name_bootstrap_unique_dart_corp_code_then_receipt_lineage",
                    "lineage_validation_status": "unique_dart_corp_code_date_bounded_candidate",
                    "source_name": "OpenDART_list_C_C001",
                    "source_url": "https://opendart.fss.or.kr/api/list.json",
                    "lineage_version": DART_LINEAGE_VERSION,
                    "attempt_status": "not_attempted",
                    "recorded_at": pd.Timestamp.now(tz="Asia/Seoul"),
                }
                lineage_entries.append(entry)
                self._lineage_rows.append(entry)

            filing = None
            offering: dict[str, Any] | None = None
            for rank, candidate in candidates.iterrows():
                entry = lineage_entries[rank]
                cached = cached_by_receipt.get(str(candidate.rcept_no))
                if (
                    cached is not None
                    and cached.get("structured_price_check_version") == STRUCTURED_PRICE_CHECK_VERSION
                    and cached.get("offering_price_parser_version") == OFFERING_PRICE_PARSER_VERSION
                ):
                    filing = candidate
                    offering = cached
                    entry["attempt_status"] = "cached_verified_record"
                    break
                if not self._should_retry_document(str(candidate.rcept_no)):
                    entry["attempt_status"] = "retry_deferred"
                    continue
                try:
                    offering = self.dart.get_offering_info(str(candidate.rcept_no))
                    filing = candidate
                    entry["attempt_status"] = "document_parsed"
                    break
                except RuntimeError as exc:
                    if "<status>014</status>" in str(exc):
                        has_structured = any(
                            item.get("rcept_no") == str(candidate.rcept_no) for item in structured_prices
                        )
                        reason = "structured_value_zip_missing" if has_structured else "zip_file_missing_retry_required"
                        self._record_document_failure(
                            candidate, reason, listing=listing, candidate_count=len(candidates),
                            structured_value_present=has_structured,
                        )
                        entry["attempt_status"] = reason
                        continue
                    self._record_document_failure(
                        candidate, "document_parse_retry_required", listing=listing,
                        candidate_count=len(candidates), structured_value_present=False,
                    )
                    entry["attempt_status"] = "document_parse_retry_required"
                    logger.warning("신고서 원문 파싱 실패 (%s): %s", candidate.corp_name, exc)
            if filing is None or offering is None:
                continue

            structured = next(
                (item for item in structured_prices if item["rcept_no"] == str(filing.rcept_no)), None
            )
            is_final_price_report = "발행조건확정" in str(filing.report_nm)
            is_final_price_disclosure = bool(
                is_final_price_report
                and offering.get("offering_price_finality") == "confirmed_price_language"
            )
            offering = self._reconcile_structured_offering_price(
                offering,
                structured,
                is_final_price_disclosure,
                structured_record_count=len(structured_prices),
            )
            offering["structured_price_check_version"] = STRUCTURED_PRICE_CHECK_VERSION
            offering["offering_price_parser_version"] = OFFERING_PRICE_PARSER_VERSION

            demand_rcept_no = None
            demand = {}
            try:
                demand_rcept_no = self.dart.find_demand_forecast_disclosure(
                    str(filing.corp_code),
                    pd.Timestamp(filing.rcept_dt).strftime("%Y%m%d"),
                    pd.Timestamp(listing.listing_date).strftime("%Y%m%d"),
                )
                if demand_rcept_no:
                    if self._should_retry_demand_document(demand_rcept_no):
                        demand = self.dart.get_demand_forecast(str(filing.corp_code), demand_rcept_no)
                    else:
                        logger.info(
                            "수요예측 원문 재시도 보류 (%s, %s)", filing.corp_name, demand_rcept_no
                        )
            except RuntimeError as exc:
                if demand_rcept_no:
                    self._record_demand_document_failure(
                        rcept_no=demand_rcept_no,
                        corp_code=str(filing.corp_code),
                        corp_name=filing.corp_name,
                        event_id=event_id,
                        listing_date=listing.listing_date,
                        error=exc,
                    )
                logger.warning("수요예측 원문 파싱 실패 (%s): %s", filing.corp_name, exc)

            financial_summary, collected_financials = self._collect_financials(
                str(filing.corp_code), pd.Timestamp(listing.listing_date)
            )
            if not collected_financials.empty:
                financial_rows.append(collected_financials)
            # fnlttSinglAcntAll은 사업연도 기준 값만 주고 해당 값이 상장 전에
            # 공개됐는지 판별할 접수일을 주지 않는다. 공개시각 계보를 수집하기
            # 전에는 재무 수치를 모델 피처로 넣지 않아 미래 정보 누출을 막는다.
            financial_model_values: dict[str, Any] = {}
            financial_time_validation_status = "publication_time_unverified_excluded"
            records.append({
                "event_id": event_id,
                "ticker": getattr(listing, "ticker", None),
                "krx_standard_code": getattr(listing, "krx_standard_code", None),
                "event_class": getattr(listing, "event_class", "unclassified_review"),
                "industry_name": getattr(listing, "industry_name", None),
                "listing_segment": getattr(listing, "listing_segment", None),
                "rcept_no": str(filing.rcept_no),
                "corp_code": str(filing.corp_code),
                "corp_name": filing.corp_name,
                "rcept_dt": filing.rcept_dt,
                "filing_report_nm": filing.report_nm,
                "filing_is_correction": bool(filing.is_correction),
                "filing_is_final_price_report": is_final_price_report,
                "filing_is_final_price_disclosure": is_final_price_disclosure,
                "filing_candidate_count": len(candidates),
                "lineage_version": DART_LINEAGE_VERSION,
                "feature_available_at": filing.rcept_dt,
                "demand_rcept_no": demand_rcept_no,
                **offering,
                # 이 행은 현재 KRX 상장 이벤트에 맞춰 수집한 공시다. 원문에
                # 기재된 상장예정일은 별도 보존하고, 병합 키는 실제 상장일을 쓴다.
                "disclosed_listing_date": offering.get("listing_date"),
                "listing_date": listing.listing_date,
                **{key: value for key, value in demand.items() if key != "corp_code"},
                **self._compare_offering_sources(offering, demand),
                "financial_as_of_year": financial_summary.get("financial_as_of_year"),
                "financial_time_validation_status": financial_time_validation_status,
                **financial_model_values,
            })

        financials = pd.concat(financial_rows, ignore_index=True) if financial_rows else pd.DataFrame(
            columns=["corp_code", "listing_date", "year", "account_name_en", "amount"]
        )
        return pd.DataFrame(records), financials

    @staticmethod
    def _reconcile_structured_offering_price(
        offering: dict[str, Any],
        structured: dict[str, Any] | None,
        is_final_price_disclosure: bool = True,
        structured_record_count: int = 0,
    ) -> dict[str, Any]:
        """원문 공모가와 DART 구조화 모집가액을 접수번호 단위로 대조한다."""
        result = offering.copy()
        result["dart_structured_offering_price"] = None
        result["dart_structured_security_type"] = None
        result["structured_price_record_count"] = structured_record_count
        result["structured_price_check"] = (
            "source_no_result" if structured_record_count == 0 else "no_matching_receipt"
        )
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

    def _should_retry_document(self, rcept_no: str) -> bool:
        """014는 영구 실패가 아니다. 버전 변경 또는 재시도 기한 후 다시 시도한다."""
        failures = self._document_failures
        if failures.empty or "rcept_no" not in failures:
            return True
        prior = failures[failures["rcept_no"].astype(str) == str(rcept_no)].copy()
        if prior.empty:
            return True
        if "lineage_version" not in prior or prior["lineage_version"].isna().all():
            return True
        latest = prior.sort_values("recorded_at").iloc[-1]
        if latest.get("lineage_version") != DART_LINEAGE_VERSION:
            return True
        recorded_at = pd.to_datetime(latest.get("recorded_at"), errors="coerce")
        if pd.isna(recorded_at):
            return True
        if recorded_at.tzinfo is None:
            recorded_at = recorded_at.tz_localize("Asia/Seoul")
        return pd.Timestamp.now(tz="Asia/Seoul") - recorded_at >= pd.Timedelta(days=DOCUMENT_RETRY_AFTER_DAYS)

    def _record_document_failure(
        self,
        filing: pd.Series,
        reason: str,
        *,
        listing: Any,
        candidate_count: int,
        structured_value_present: bool,
    ) -> None:
        """접수번호 실패 이력을 누적한다. 회사 전체 실패로 덮어쓰지 않는다."""
        prior = self._document_failures
        attempts = 0
        if not prior.empty and "rcept_no" in prior:
            attempts = int((prior["rcept_no"].astype(str) == str(filing.rcept_no)).sum())
        record = pd.DataFrame([{
            "rcept_no": str(filing.rcept_no),
            "corp_code": str(filing.corp_code),
            "corp_name": filing.corp_name,
            "rcept_dt": filing.rcept_dt,
            "filing_report_nm": filing.report_nm,
            "event_id": getattr(listing, "event_id", None),
            "ticker": getattr(listing, "ticker", None),
            "listing_date": getattr(listing, "listing_date", None),
            "candidate_count": candidate_count,
            "reason": reason,
            "failure_classification": reason,
            "structured_value_present": structured_value_present,
            "retriable": True,
            "attempt_number": attempts + 1,
            "lineage_version": DART_LINEAGE_VERSION,
            "recorded_at": pd.Timestamp.now(tz="Asia/Seoul"),
        }])
        self._document_failures = pd.concat([self._document_failures, record], ignore_index=True)

    def _build_document_failure_audit(self) -> pd.DataFrame:
        """기존 014와 새 실패 이력을 원인별 재감사 가능한 표로 정규화한다."""
        failures = self._document_failures.copy()
        if failures.empty:
            return failures
        if "failure_classification" not in failures:
            failures["failure_classification"] = None
        legacy = failures["failure_classification"].isna()
        failures.loc[legacy, "failure_classification"] = "recheck_required_legacy_metadata_incomplete"
        if "retriable" not in failures:
            failures["retriable"] = True
        failures["retriable"] = failures["retriable"].map(
            lambda value: True if pd.isna(value) else bool(value)
        ).astype(bool)
        failures["audit_status"] = failures["failure_classification"].map({
            "structured_value_zip_missing": "recoverable_via_structured_lineage_review",
            "zip_file_missing_retry_required": "retry_or_alternate_candidate_required",
            "document_parse_retry_required": "retry_or_parser_review_required",
            "source_document_now_available_parser_retry": "parser_retry_required",
            "document_response_empty_retry_required": "retry_required",
            "dart_zip_unavailable_recheck_web_lineage": "web_disclosure_or_lineage_review_required",
            "document_request_retry_required": "retry_required",
            "recheck_required_legacy_metadata_incomplete": "re_audit_required",
        }).fillna("review_required")
        return failures

    def _should_retry_demand_document(self, rcept_no: str) -> bool:
        """수요예측 원문도 014를 영구 제외하지 않고 접수번호별로 재시도한다."""
        failures = self._demand_document_failures
        if failures.empty or "rcept_no" not in failures:
            return True
        prior = failures[failures["rcept_no"].astype(str) == str(rcept_no)].copy()
        if prior.empty:
            return True
        recorded_at = pd.to_datetime(prior["recorded_at"], errors="coerce").max()
        if pd.isna(recorded_at):
            return True
        if recorded_at.tzinfo is None:
            recorded_at = recorded_at.tz_localize("Asia/Seoul")
        return pd.Timestamp.now(tz="Asia/Seoul") - recorded_at >= pd.Timedelta(days=DOCUMENT_RETRY_AFTER_DAYS)

    def _record_demand_document_failure(
        self,
        *,
        rcept_no: str,
        corp_code: str,
        corp_name: str,
        event_id: str,
        listing_date: object,
        error: RuntimeError,
    ) -> None:
        """수요예측 원문 실패를 별도 이력으로 저장한다."""
        is_zip_missing = "<status>014</status>" in str(error)
        prior = self._demand_document_failures
        attempts = 0
        if not prior.empty and "rcept_no" in prior:
            attempts = int((prior["rcept_no"].astype(str) == str(rcept_no)).sum())
        record = pd.DataFrame([{
            "rcept_no": str(rcept_no),
            "corp_code": corp_code,
            "corp_name": corp_name,
            "event_id": event_id,
            "listing_date": listing_date,
            "reason": "zip_file_missing_retry_required" if is_zip_missing else "document_request_retry_required",
            "retriable": True,
            "attempt_number": attempts + 1,
            "recorded_at": pd.Timestamp.now(tz="Asia/Seoul"),
        }])
        self._demand_document_failures = pd.concat([prior, record], ignore_index=True)

    def audit_document_failures(self) -> pd.DataFrame:
        """기존 원문 실패 접수번호를 접수번호 단위로 다시 감사한다.

        014는 회사 전체가 아니라 해당 ZIP 원문만 없다는 의미이므로, 이 감사는
        접수번호마다 원문 재조회 결과를 이력으로 추가한다. 대체 후보 선택은
        ``run``의 공시 계보 단계가 담당한다.
        """
        if getattr(self.dart, "is_configured", True) is False:
            raise RuntimeError("DART_API_KEY를 설정한 뒤 원문 실패 재감사를 실행하세요.")
        failures = self._document_failures.copy()
        if failures.empty or "rcept_no" not in failures:
            audit = self._build_document_failure_audit()
            audit.to_parquet(self.raw_dir / "dart_document_failure_audit.parquet", index=False)
            return audit

        latest = failures.dropna(subset=["rcept_no"]).copy()
        if "recorded_at" in latest:
            latest = latest.sort_values("recorded_at").drop_duplicates("rcept_no", keep="last")
        for failure in latest.itertuples(index=False):
            rcept_no = str(failure.rcept_no)
            try:
                text = self.dart.get_document_text(rcept_no)
                reason = "source_document_now_available_parser_retry"
                retriable = True
                if not str(text).strip():
                    reason = "document_response_empty_retry_required"
            except RuntimeError as exc:
                if "<status>014</status>" in str(exc):
                    reason = "dart_zip_unavailable_recheck_web_lineage"
                    retriable = False
                else:
                    reason = "document_request_retry_required"
                    retriable = True
            self._document_failures = pd.concat([self._document_failures, pd.DataFrame([{
                "rcept_no": rcept_no,
                "corp_code": getattr(failure, "corp_code", None),
                "corp_name": getattr(failure, "corp_name", None),
                "rcept_dt": getattr(failure, "rcept_dt", None),
                "filing_report_nm": getattr(failure, "filing_report_nm", None),
                "event_id": getattr(failure, "event_id", None),
                "ticker": getattr(failure, "ticker", None),
                "listing_date": getattr(failure, "listing_date", None),
                "candidate_count": getattr(failure, "candidate_count", None),
                "reason": reason,
                "failure_classification": reason,
                "structured_value_present": False,
                "retriable": retriable,
                "attempt_number": int((self._document_failures["rcept_no"].astype(str) == rcept_no).sum()) + 1,
                "lineage_version": DART_LINEAGE_VERSION,
                "recorded_at": pd.Timestamp.now(tz="Asia/Seoul"),
            }])], ignore_index=True)
        self._document_failures.to_parquet(self.raw_dir / "dart_document_failures.parquet", index=False)
        audit = self._build_document_failure_audit()
        audit.to_parquet(self.raw_dir / "dart_document_failure_audit.parquet", index=False)
        return audit

    @staticmethod
    def _build_feature_time_audit(features: pd.DataFrame) -> pd.DataFrame:
        """상장 전 정보만 피처로 사용했는지 행 단위로 검사한다."""
        columns = [
            "event_id", "corp_name", "listing_date", "feature_available_at",
            "is_future_information", "time_validation_status",
        ]
        audit = features.reindex(columns=columns[:-2]).copy()
        listing_date = pd.to_datetime(audit["listing_date"], errors="coerce")
        available_at = pd.to_datetime(audit["feature_available_at"], errors="coerce")
        audit["is_future_information"] = (
            available_at.notna() & listing_date.notna() & (available_at > listing_date)
        )
        audit["time_validation_status"] = "missing_feature_available_at_review_required"
        audit.loc[available_at.notna() & listing_date.notna() & ~audit["is_future_information"], "time_validation_status"] = (
            "pre_listing_or_same_day"
        )
        audit.loc[audit["is_future_information"], "time_validation_status"] = "future_information_blocked"
        return audit

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
            "filing_is_final_price_report", "filing_is_final_price_disclosure",
            "filing_candidate_count", "demand_rcept_no", "offering_price",
            "offering_price_extracted_amount", "offering_price_review_status",
            "offering_price_finality", "offering_price_parse_method", "offering_price_range_warning",
            "offering_price_parser_version",
            "offering_price_audit_context", "price_band_low", "price_band_high",
            "price_band_check", "dart_structured_offering_price", "dart_structured_security_type",
            "structured_price_check", "structured_price_record_count", "structured_price_check_version",
            "demand_offering_price", "demand_price_check",
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
