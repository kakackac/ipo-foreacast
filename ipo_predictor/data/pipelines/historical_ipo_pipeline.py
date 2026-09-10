"""과거 IPO의 DART 공시, KRX 실적, 시장 데이터를 하나의 학습셋으로 만든다."""

import json
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from config import PROC_DIR, RAW_DIR
from data.collectors.dart_collector import DEMAND_PARSER_VERSION, DARTCollector
from data.collectors.krx_collector import KRXCollector
from data.collectors.institutional_result_collector import (
    INSTITUTIONAL_RESULT_COLUMNS,
    LOCKUP_FIELDS,
    OfficialInstitutionalResultCollector,
)
from data.collectors.underwriter_registry import (
    OFFICIAL_UNDERWRITER_REGISTRY,
    build_underwriter_priorities,
    build_underwriter_source_readiness,
    normalize_underwriter,
)
from data.processors.feature_engineer import FeatureEngineer
from features.model_profiles import MODEL_PROFILES, build_stage_dataset, stage_readiness_by_offering_type
from features.source_contracts import (
    DART_STRUCTURAL_AGGREGATE_STATUS,
    OFFICIAL_UNDERWRITER_AGGREGATE_STATUS,
)

logger = logging.getLogger(__name__)

MAX_FILING_TO_LISTING_DAYS = 400
STRUCTURED_PRICE_CHECK_VERSION = 3
OFFERING_PRICE_PARSER_VERSION = 6
DART_LINEAGE_VERSION = 1
DART_DEMAND_PARSER_VERSION = DEMAND_PARSER_VERSION
DART_FINAL_TERMS_DEMAND_PARSER_VERSION = DEMAND_PARSER_VERSION
DART_INSTITUTIONAL_DATA_CONTRACT_VERSION = 1
MAX_DEMAND_DOCUMENT_CANDIDATES = 8
DOCUMENT_RETRY_AFTER_DAYS = 7
OFFICIAL_SOURCE_RESOLUTION_COLUMNS = [
    "event_id", "feature_name", "resolution_status", "checked_at", "checked_sources",
    "reviewed_by", "note",
]
UNDERWRITER_INSTITUTIONAL_REVIEW_QUEUE_COLUMNS = [
    "event_id", "event_class", "offering_type", "ticker", "corp_name", "lead_underwriter",
    "market", "listing_date", "offering_price", "event_source_url", "public_discovery_url",
    "collection_policy", "review_status", "review_note", "notice_url",
    "notice_title", "published_at", "source_offering_price", "notice_underwriter",
    "aggregate_scope_verification",
]


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
        self._demand_document_cache = self._load_cached_frame("dart_demand_document_cache.parquet")
        self._offering_document_cache = self._load_cached_frame("dart_offering_document_cache.parquet")
        self._demand_document_cache_updates: list[dict[str, Any]] = []
        self._offering_document_cache_updates: list[dict[str, Any]] = []
        self._lineage_rows: list[dict[str, Any]] = []

    def run(
        self,
        start_year: int,
        end_year: int,
        feature_set: str = "phase2",
        include_dart_demand_audit: bool = True,
    ) -> dict[str, Any]:
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
        listing_price_audit = self._build_listing_price_audit(calendar, prices)
        listing_price_audit.to_parquet(self.raw_dir / "krx_listing_price_audit.parquet", index=False)
        krx_ipo = self._attach_prices(calendar, prices)
        krx_ipo = self._attach_price_resolution(krx_ipo, listing_price_audit)
        # 상장일 가격 조회에서 확인된 KRX 표준코드·시장 정보를 이벤트 마스터에
        # 되돌려 다음 실행의 이벤트 식별과 정합에 재사용한다.
        krx_ipo.to_parquet(self.raw_dir / "krx_official_event_master.parquet", index=False)
        underwriter_priorities = build_underwriter_priorities(krx_ipo)
        underwriter_priorities.to_parquet(
            self.raw_dir / "official_underwriter_priorities.parquet", index=False
        )
        source_readiness = build_underwriter_source_readiness(krx_ipo)
        source_readiness.to_parquet(
            self.raw_dir / "official_underwriter_source_readiness.parquet", index=False
        )

        kospi = self._collect_index_with_cache("1", start_year, end_year, "kospi_index.parquet")
        kosdaq = self._collect_index_with_cache("2", start_year, end_year, "kosdaq_index.parquet")
        kospi.to_parquet(self.raw_dir / "kospi_index.parquet", index=False)
        kosdaq.to_parquet(self.raw_dir / "kosdaq_index.parquet", index=False)

        dart_ipo, financials = self._collect_dart_records(
            calendar,
            start_year,
            end_year,
            include_dart_demand_audit=include_dart_demand_audit,
        )
        self._flush_document_cache_updates()
        underwriter_results = self._collect_official_underwriter_institutional_results(krx_ipo)
        underwriter_results.to_parquet(
            self.raw_dir / "official_underwriter_institutional_results.parquet", index=False
        )
        dart_ipo = self._merge_official_underwriter_institutional_results(
            dart_ipo, underwriter_results
        )
        self._document_failures.to_parquet(self.raw_dir / "dart_document_failures.parquet", index=False)
        self._demand_document_failures.to_parquet(
            self.raw_dir / "dart_demand_document_failures.parquet", index=False
        )
        self._demand_document_cache.to_parquet(
            self.raw_dir / "dart_demand_document_cache.parquet", index=False
        )
        self._offering_document_cache.to_parquet(
            self.raw_dir / "dart_offering_document_cache.parquet", index=False
        )
        self._build_institutional_extraction_audit().to_parquet(
            self.raw_dir / "dart_institutional_extraction_audit.parquet", index=False
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
        observations = engineer.build_feature_observations(features)
        source_resolutions = self._load_official_source_resolutions()
        source_resolutions.to_parquet(self.raw_dir / "official_source_resolutions.parquet", index=False)
        observations = self._apply_official_source_resolutions(observations, source_resolutions)
        observations.to_parquet(
            self.processed_dir / "feature_observations.parquet", index=False
        )
        feature_coverage = self._build_feature_coverage_audit(observations)
        feature_coverage.to_parquet(
            self.processed_dir / "feature_coverage_audit.parquet", index=False
        )
        time_audit = self._build_feature_time_audit(features, observations)
        time_audit.to_parquet(self.processed_dir / "feature_time_validation.parquet", index=False)
        stage_summary = self._write_stage_datasets(features, time_audit)

        summary = self._build_summary(calendar, dart_ipo, prices, features)
        summary["event_master_source"] = "KRX_KIND_new_listing_company"
        summary["event_class_counts"] = {
            str(key): int(value)
            for key, value in calendar.get("event_class", pd.Series(dtype=str)).value_counts(dropna=False).items()
        }
        summary["legacy_list_dd_candidate_rows"] = event_manifest["legacy_candidate_rows"]
        summary["event_master_manifest"] = event_manifest["path"]
        summary["listing_price_unmatched_rows"] = int(
            listing_price_audit.get("price_match_status", pd.Series(dtype=object)).eq("unmatched").sum()
        )
        summary["listing_price_failure_reasons"] = {
            str(reason): int(count)
            for reason, count in listing_price_audit.get(
                "price_failure_reason", pd.Series(dtype=object)
            ).dropna().value_counts().items()
        }
        summary["listing_price_resolution_counts"] = {
            str(status): int(count)
            for status, count in listing_price_audit.get(
                "price_resolution_status", pd.Series(dtype=object)
            ).value_counts(dropna=False).items()
        }
        verified_price_rows = listing_price_audit.get(
            "price_resolution_status", pd.Series(dtype=object)
        ).eq("official_price_verified")
        summary["listing_price_target_eligible_rows"] = int(verified_price_rows.sum())
        summary["listing_price_target_blocked_rows"] = int((~verified_price_rows).sum())
        summary["future_information_violations"] = int(time_audit["is_future_information"].sum())
        summary["feature_time_validation_rows"] = int(len(time_audit))
        summary["feature_coverage"] = {
            str(row.feature_name): {
                "observed_rows": int(row.observed_rows),
                "coverage_rate": float(row.coverage_rate),
                "human_review_required_rows": int(row.human_review_required_rows),
            }
            for row in feature_coverage.itertuples(index=False)
        }
        summary["official_underwriter_institutional_notice_rows"] = int(len(underwriter_results))
        summary["official_underwriter_verified_aggregate_bundle_rows"] = int(
            (underwriter_results.get("validation_status", pd.Series(dtype=str)) ==
             "verified_official_underwriter_aggregate_bundle").sum()
        )
        summary["official_underwriter_priority_rows"] = int(len(underwriter_priorities))
        summary["official_underwriter_priority_coverage"] = round(
            float(underwriter_priorities.get("coverage_ratio", pd.Series(dtype=float)).sum()), 4
        )
        summary["model_stage_readiness"] = stage_summary
        with open(self.processed_dir / "data_collection_summary.json", "w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2, default=str)
        logger.info("실제 데이터 파이프라인 완료: %d개 학습 행", len(features))
        return summary

    def _build_institutional_extraction_audit(self) -> pd.DataFrame:
        """문서별 후보·승인·거절 근거를 모델 피처와 분리해 보존한다."""
        frames: list[pd.DataFrame] = []
        for source_kind, cache, version_column in (
            ("demand_lineage", self._demand_document_cache, "demand_parser_version"),
            ("final_terms", self._offering_document_cache, "dart_final_terms_demand_parser_version"),
        ):
            if cache.empty or "rcept_no" not in cache.columns:
                continue
            frame = cache.copy()
            frame["document_source_kind"] = source_kind
            frame["parser_version"] = pd.to_numeric(
                frame.get(version_column, pd.Series(index=frame.index, dtype=float)), errors="coerce"
            )
            frame = frame[frame["parser_version"].eq(DEMAND_PARSER_VERSION)]
            if frame.empty:
                continue
            frames.append(frame)
        columns = [
            "document_source_kind", "rcept_no", "corp_code", "corp_name", "rcept_dt", "report_nm",
            "parser_version", "parsed_at", "institutional_demand_ratio",
            "institutional_demand_rule_id", "institutional_demand_parser_validation_status",
            "institutional_demand_rejection_reason", "institutional_demand_evidence",
            "institutional_demand_structured_evidence", "lockup_commitment_ratio", "lockup_rule_id",
            "lockup_parser_validation_status", "lockup_rejection_reason", "lockup_parse_evidence",
            "lockup_structured_evidence", "demand_offering_price", "demand_offering_price_context",
        ]
        if not frames:
            return pd.DataFrame(columns=columns)
        audit = pd.concat(frames, ignore_index=True, sort=False)
        for column in columns:
            if column not in audit.columns:
                audit[column] = pd.NA
        return audit[columns].sort_values(
            ["rcept_dt", "rcept_no", "document_source_kind"], na_position="last"
        ).drop_duplicates(["rcept_no", "document_source_kind"], keep="last").reset_index(drop=True)

    def _write_stage_datasets(
        self, features: pd.DataFrame, feature_time_audit: pd.DataFrame
    ) -> dict[str, dict[str, Any]]:
        """동일 원천 데이터로 세 공개 단계의 후보 표와 유형별 준비도를 함께 저장한다."""
        stage_dir = self.processed_dir / "model_stage_datasets"
        stage_dir.mkdir(parents=True, exist_ok=True)
        summary: dict[str, dict[str, Any]] = {}
        for stage_name in MODEL_PROFILES:
            dataset = build_stage_dataset(features, stage_name, feature_time_audit)
            dataset.to_parquet(stage_dir / f"{stage_name}.parquet", index=False)
            profile = MODEL_PROFILES[stage_name]
            summary[stage_name] = {
                "description": profile.description,
                "required_features": list(profile.feature_names),
                "event_rows": int(len(dataset)),
                "verified_offering_price_rows": int(dataset["stage_offering_price_verified"].sum()),
                "dual_target_rows": int(dataset["stage_dual_target_ready"].sum()),
                "feature_complete_rows": int(dataset["stage_features_complete"].sum()),
                "critical_feature_complete_rows": int(
                    dataset["stage_critical_features_complete"].sum()
                ),
                "time_valid_rows": int(dataset["stage_time_valid"].sum()),
                "source_valid_rows": int(dataset["stage_source_valid"].sum()),
                "model_candidate_rows": int(dataset["stage_model_candidate"].sum()),
                "offering_type_breakdown": stage_readiness_by_offering_type(dataset),
            }
        with open(self.processed_dir / "model_stage_readiness.json", "w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2, default=str)
        return summary

    def _collect_official_underwriter_institutional_results(self, events: pd.DataFrame) -> pd.DataFrame:
        """DART 최종 문서가 비었을 때 쓸 기관 수요예측 보조 원천을 수집한다.

        원장에는 대표주관사의 공개 결과 문서 URL과 게시일, 공모가, 통합 범위
        확인값만 기록한다. 개인청약·증권사별 경쟁률·비례배정 수치는 읽지 않는다.
        """
        source_path = self.manual_dir / "underwriter_institutional_sources.csv"
        if not source_path.exists():
            return pd.DataFrame(columns=INSTITUTIONAL_RESULT_COLUMNS)
        sources = pd.read_csv(source_path, dtype=str).fillna("")
        if sources.empty:
            return pd.DataFrame(columns=INSTITUTIONAL_RESULT_COLUMNS)
        collector = OfficialInstitutionalResultCollector()
        cached = self._load_cached_frame("official_underwriter_institutional_results.parquet")
        requested_ids = sources.apply(
            lambda row: collector.notice_id(
                str(row["event_id"]), str(row["notice_url"]), str(row.get("source_version", "initial"))
            ), axis=1,
        )
        reusable_statuses = {
            "verified_official_underwriter_aggregate_bundle",
            "official_notice_aggregate_bundle_review_required",
            "official_notice_incomplete_institutional_bundle",
            "official_notice_no_supported_institutional_bundle",
            "source_document_type_not_supported",
        }
        required_columns = {
            "notice_id", "validation_status", "event_context_validation_status",
            "source_document_sha256", "available_at",
        }
        reusable = (
            cached[cached["notice_id"].astype(str).isin(set(requested_ids))
                   & cached["validation_status"].isin(reusable_statuses)]
            if not cached.empty and required_columns.issubset(cached.columns)
            else pd.DataFrame(columns=INSTITUTIONAL_RESULT_COLUMNS)
        )
        reusable_ids = set(reusable.get("notice_id", pd.Series(dtype=str)).astype(str))
        pending = sources.loc[~requested_ids.isin(reusable_ids)].copy()
        event_contexts = {
            str(row.event_id): row._asdict()
            for row in events.itertuples(index=False)
            if pd.notna(getattr(row, "event_id", None))
        }
        fresh = collector.collect_sources(pending, event_contexts) if not pending.empty else pd.DataFrame(
            columns=INSTITUTIONAL_RESULT_COLUMNS
        )
        return pd.concat([reusable, fresh], ignore_index=True).reindex(
            columns=INSTITUTIONAL_RESULT_COLUMNS
        ).drop_duplicates("notice_id", keep="last")

    def prepare_underwriter_institutional_review_queue(self) -> pd.DataFrame:
        """대표주관사 기관 수요예측 결과 URL 검토 대기열을 만든다.

        이 메서드는 주관사 사이트를 탐색하거나 URL을 추측하지 않는다. 이미
        저장된 KRX 이벤트 마스터에서 지원 주관사의 IPO 후보만 골라,
        공식 안내 페이지와 확인해야 할 이벤트 메타데이터를 함께 제공한다.
        """
        event_path = self.raw_dir / "krx_official_event_master.parquet"
        if not event_path.exists():
            raise RuntimeError(
                "공식 KRX 이벤트 마스터가 없습니다. 먼저 collect-events 또는 collect를 실행하세요."
            )
        events = pd.read_parquet(event_path)
        if events.empty:
            return pd.DataFrame(columns=UNDERWRITER_INSTITUTIONAL_REVIEW_QUEUE_COLUMNS)
        candidates = events.copy()
        candidates["normalized_underwriter"] = candidates.get(
            "lead_underwriter", pd.Series(index=candidates.index, dtype=object)
        ).map(normalize_underwriter)
        candidates = candidates[candidates["normalized_underwriter"].isin(OFFICIAL_UNDERWRITER_REGISTRY)].copy()
        candidates = candidates[candidates.get("event_class", pd.Series(index=candidates.index)).isin({
            "general_ipo", "spac_ipo", "foreign_listing",
        })].copy()

        source_path = self.manual_dir / "underwriter_institutional_sources.csv"
        already_linked: set[str] = set()
        if source_path.exists():
            linked = pd.read_csv(source_path, dtype=str).fillna("")
            if "event_id" in linked.columns and "notice_url" in linked.columns:
                already_linked = set(linked[linked["notice_url"].astype(str).str.strip().ne("")]["event_id"])

        records: list[dict[str, object]] = []
        for event in candidates.itertuples(index=False):
            config = OFFICIAL_UNDERWRITER_REGISTRY[str(event.normalized_underwriter)]
            event_id = str(getattr(event, "event_id", ""))
            records.append({
                "event_id": event_id,
                "event_class": getattr(event, "event_class", None),
                "offering_type": getattr(event, "offering_type", None),
                "ticker": getattr(event, "ticker", None),
                "corp_name": getattr(event, "corp_name", None),
                "lead_underwriter": event.normalized_underwriter,
                "market": getattr(event, "market", None),
                "listing_date": getattr(event, "listing_date", None),
                "offering_price": getattr(event, "offering_price", None),
                "event_source_url": getattr(event, "source_url", None),
                "public_discovery_url": config["public_discovery_url"],
                "collection_policy": config["collection_policy"],
                "review_status": (
                    "official_institutional_url_already_linked" if event_id in already_linked
                    else "official_institutional_url_required"
                ),
                "review_note": "대표주관사의 통합 기관 수요예측·확약 결과 문서만 연결합니다.",
                "notice_url": None,
                "notice_title": None,
                "published_at": None,
                "source_offering_price": None,
                "notice_underwriter": event.normalized_underwriter,
                "aggregate_scope_verification": None,
            })
        queue = pd.DataFrame(records, columns=UNDERWRITER_INSTITUTIONAL_REVIEW_QUEUE_COLUMNS)
        queue = queue.sort_values(["listing_date", "corp_name"], na_position="last").reset_index(drop=True)
        queue.to_parquet(self.raw_dir / "official_underwriter_institutional_review_queue.parquet", index=False)
        queue.to_csv(self.manual_dir / "underwriter_institutional_review_queue.csv", index=False, encoding="utf-8-sig")
        return queue

    def audit_underwriter_source_readiness(self) -> pd.DataFrame:
        """공식 주관사 결과 원천의 표본 감사 상태를 이벤트 규모와 함께 저장한다.

        공개 경로가 있다는 사실과 기관 경쟁률·확약 값을 공개적으로 제공한다는
        사실은 다르다. 이 감사는 후자가 검증되기 전 자동 수집을 차단한다.
        """
        event_path = self.raw_dir / "krx_official_event_master.parquet"
        if not event_path.exists():
            raise RuntimeError(
                "공식 KRX 이벤트 마스터가 없습니다. 먼저 collect-events 또는 collect를 실행하세요."
            )
        readiness = build_underwriter_source_readiness(pd.read_parquet(event_path))
        readiness.to_parquet(
            self.raw_dir / "official_underwriter_source_readiness.parquet", index=False
        )
        readiness.to_csv(
            self.manual_dir / "official_underwriter_source_readiness.csv",
            index=False,
            encoding="utf-8-sig",
        )
        return readiness

    def _load_official_source_resolutions(self) -> pd.DataFrame:
        """공식 원천을 확인한 뒤 확정한 결측 사유만 관측 원장에 반영한다."""
        path = self.manual_dir / "official_source_resolutions.csv"
        if not path.exists():
            return pd.DataFrame(columns=OFFICIAL_SOURCE_RESOLUTION_COLUMNS)
        resolutions = pd.read_csv(path, dtype=str).fillna("")
        missing = {"event_id", "feature_name", "resolution_status", "checked_at"} - set(resolutions.columns)
        if missing:
            raise ValueError(f"공식 원천 결측 원장에 필요한 열이 없습니다: {', '.join(sorted(missing))}")
        allowed = {
            "official_source_not_published", "not_yet_published", "parser_failed", "source_access_failed",
        }
        invalid = set(resolutions["resolution_status"]) - allowed
        if invalid:
            raise ValueError(f"지원하지 않는 공식 원천 결측 상태입니다: {', '.join(sorted(invalid))}")
        return resolutions.reindex(columns=OFFICIAL_SOURCE_RESOLUTION_COLUMNS)

    @staticmethod
    def _apply_official_source_resolutions(
        observations: pd.DataFrame, resolutions: pd.DataFrame
    ) -> pd.DataFrame:
        """값을 만들지 않고, 확인된 결측 사유의 근거만 피처 관측에 붙인다."""
        if observations.empty or resolutions.empty:
            return observations
        latest = resolutions.copy()
        latest["checked_at"] = pd.to_datetime(latest["checked_at"], errors="coerce")
        latest = latest.sort_values("checked_at").drop_duplicates(["event_id", "feature_name"], keep="last")
        latest = latest.rename(columns={
            "resolution_status": "resolved_missing_reason",
            "checked_sources": "resolution_source_reference",
            "checked_at": "resolution_checked_at",
        })
        result = observations.merge(
            latest[[
                "event_id", "feature_name", "resolved_missing_reason",
                "resolution_source_reference", "resolution_checked_at",
            ]],
            on=["event_id", "feature_name"], how="left",
        )
        mask = result["is_missing"] & result["resolved_missing_reason"].notna()
        result.loc[mask, "missing_reason"] = result.loc[mask, "resolved_missing_reason"]
        source_reference = result["resolution_source_reference"].replace("", pd.NA)
        result.loc[mask & source_reference.notna(), "source_reference"] = source_reference[mask & source_reference.notna()]
        result.loc[mask, "collected_at"] = result.loc[mask, "resolution_checked_at"]
        result.loc[mask, "validation_status"] = result.loc[mask, "resolved_missing_reason"]
        result.loc[mask, "human_review_required"] = True
        return result.drop(columns=[
            "resolved_missing_reason", "resolution_source_reference", "resolution_checked_at",
        ])

    @staticmethod
    def _merge_official_underwriter_institutional_results(
        dart_ipo: pd.DataFrame, underwriter_results: pd.DataFrame
    ) -> pd.DataFrame:
        """공식 주관사 결과를 필드별 최신 검증 문서로 보완한다.

        기관 경쟁률과 통합 확약은 같은 IPO의 공식 원천·상장 전 공개·공모가
        정합을 각각 만족하면 서로 다른 문서에서 올 수 있다. 한 문서에 두 값이
        모두 없다는 이유로 이미 검증된 값을 버리지 않으며, 각 필드의 원문 URL과
        공개 시각은 독립적으로 보존한다.
        """
        if dart_ipo.empty or underwriter_results.empty:
            return dart_ipo
        required = {"event_id", "event_context_validation_status", "validation_status"}
        if not required.issubset(underwriter_results.columns):
            return dart_ipo
        candidates = underwriter_results.copy()
        scope = candidates.get("aggregate_scope_verification", pd.Series(index=candidates.index, dtype=object))
        bundle_verified = (
            candidates["validation_status"].eq("verified_official_underwriter_aggregate_bundle")
            & candidates.get("institutional_demand_ratio", pd.Series(index=candidates.index)).notna()
            & candidates.get("lockup_commitment_ratio", pd.Series(index=candidates.index)).notna()
        )
        candidates = candidates[
            candidates["event_context_validation_status"].eq("verified_event_context")
            & (scope.eq("manual_verified_aggregate_institutional") | bundle_verified)
        ].copy()
        if candidates.empty:
            return dart_ipo
        candidates["published_at"] = pd.to_datetime(candidates.get("published_at"), errors="coerce")
        candidates["collected_at"] = pd.to_datetime(candidates.get("collected_at"), errors="coerce")

        def latest_for(field: str, prefix: str) -> pd.DataFrame:
            available = candidates[candidates.get(field, pd.Series(index=candidates.index)).notna()].copy()
            if available.empty:
                return pd.DataFrame(columns=["event_id"])
            available = available.sort_values(["published_at", "collected_at"], na_position="first")
            latest = available.groupby("event_id", dropna=False).tail(1).copy()
            return latest.rename(columns={
                field: f"underwriter_{prefix}_value",
                "notice_url": f"underwriter_{prefix}_source_url",
                "available_at": f"underwriter_{prefix}_available_at",
                f"{prefix}_evidence": f"underwriter_{prefix}_evidence",
            })

        institutional = latest_for("institutional_demand_ratio", "institutional")
        lockup = latest_for("lockup_commitment_ratio", "lockup")
        result = dart_ipo.copy()
        if not institutional.empty:
            result = result.merge(institutional.reindex(columns=[
                "event_id", "underwriter_institutional_value", "underwriter_institutional_source_url",
                "underwriter_institutional_available_at", "underwriter_institutional_evidence",
            ]), on="event_id", how="left")
        if not lockup.empty:
            result = result.merge(lockup.reindex(columns=[
                "event_id", "underwriter_lockup_value", "underwriter_lockup_source_url",
                "underwriter_lockup_available_at", "underwriter_lockup_evidence",
            ]), on="event_id", how="left")

        institutional_missing = result.get(
            "institutional_demand_ratio", pd.Series(index=result.index, dtype=float)
        ).isna()
        if "underwriter_institutional_value" in result:
            use = institutional_missing & result["underwriter_institutional_value"].notna()
            result.loc[use, "institutional_demand_ratio"] = result.loc[use, "underwriter_institutional_value"]
            result.loc[use, "institutional_source_url"] = result.loc[use, "underwriter_institutional_source_url"]
            result.loc[use, "institutional_available_at"] = result.loc[use, "underwriter_institutional_available_at"]
            result.loc[use, "institutional_demand_evidence"] = result.loc[use, "underwriter_institutional_evidence"]
            result.loc[use, "institutional_validation_status"] = OFFICIAL_UNDERWRITER_AGGREGATE_STATUS
        lockup_missing = result.get("lockup_commitment_ratio", pd.Series(index=result.index, dtype=float)).isna()
        if "underwriter_lockup_value" in result:
            use = lockup_missing & result["underwriter_lockup_value"].notna()
            result.loc[use, "lockup_commitment_ratio"] = result.loc[use, "underwriter_lockup_value"]
            result.loc[use, "lockup_source_url"] = result.loc[use, "underwriter_lockup_source_url"]
            result.loc[use, "lockup_available_at"] = result.loc[use, "underwriter_lockup_available_at"]
            result.loc[use, "lockup_parse_evidence"] = result.loc[use, "underwriter_lockup_evidence"]
            result.loc[use, "lockup_validation_status"] = OFFICIAL_UNDERWRITER_AGGREGATE_STATUS
        return result.drop(columns=[column for column in result.columns if column.startswith("underwriter_")])

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
            "classification_cache_repaired_rows": 0,
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
                prior_offering_type = calendar.get(
                    "offering_type", pd.Series(index=calendar.index, dtype=object)
                ).copy()
                calendar = self._refresh_official_event_classification(calendar)
                repaired = (
                    prior_offering_type.fillna("").astype(str).str.strip()
                    != calendar["offering_type"].fillna("").astype(str).str.strip()
                )
                manifest["classification_cache_repaired_rows"] = int(repaired.sum())
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

    def _refresh_official_event_classification(self, calendar: pd.DataFrame) -> pd.DataFrame:
        """공식 이벤트 캐시의 누락된 유형 분류를 원천 필드에서 복구한다."""
        reclassify = getattr(self.krx, "reclassify_official_listing_events", None)
        if callable(reclassify):
            return reclassify(calendar)

        # 테스트용 또는 이전 호환 수집기는 분류 함수를 제공하지 않을 수 있다.
        # 그 경우에도 이미 보유한 event_class에서 확정 가능한 최소 유형만 채워
        # 빈 값이 학습 준비도 집계에서 review_required로 뭉개지지 않게 한다.
        result = calendar.copy()
        if "offering_type" not in result:
            result["offering_type"] = pd.NA
        fallback = result.get("event_class", pd.Series(index=result.index, dtype=object)).map({
            "general_ipo": "common_stock_ipo",
            "spac_ipo": "spac_ipo",
            "foreign_listing": "foreign_common_stock_listing",
            "relisting": "relisting",
        })
        missing = result["offering_type"].isna() | result["offering_type"].astype(str).str.strip().eq("")
        result.loc[missing, "offering_type"] = fallback[missing].fillna("review_required")
        return result

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
                    cached_record = previous.to_dict()
                    has_raw_evidence = pd.notna(cached_record.get("price_raw_response_evidence"))
                    has_direct_match = cached_record.get("price_match_status") == "matched"
                    if has_raw_evidence and has_direct_match:
                        records.append(cached_record)
                        reused += 1
                        continue
                    # 이전 코드가 남긴 값만 있는 캐시는 신뢰 상태를 승격하지
                    # 않는다. 이번 실행에서 KRX 일별 원시 응답을 다시 받아
                    # 이벤트 식별자와 재매칭한 뒤에만 타깃으로 쓸 수 있다.
                    logger.info("KRX 상장일 가격 캐시 재감사: %s", cache_key)
            listing_date = pd.Timestamp(row["listing_date"]).strftime("%Y%m%d")
            ticker = str(row["ticker"])
            isu_cd = row.get("isu_cd")
            market = row.get("market")
            corp_name = row.get("corp_name")
            try:
                record = self.krx.get_listing_day_price(
                    ticker, listing_date, isu_cd=isu_cd, market=market, corp_name=corp_name
                )
            except (RuntimeError, ConnectionError, TimeoutError):
                self._checkpoint_listing_prices(cached, records)
                logger.error("KRX 가격 수집 중단: 완료한 %d행을 저장했습니다. 실패 행은 재실행 대상입니다.", len(records))
                raise
            records.append(record)
            if len(records) % 25 == 0:
                self._checkpoint_listing_prices(cached, records)
        if reused:
            logger.info("KRX 상장일 가격 캐시 재사용: %d건", reused)
        prices = pd.DataFrame(records)
        # 캐시 행은 Timestamp, 방금 받은 API 행은 YYYYMMDD 문자열일 수 있다.
        # Parquet은 같은 열의 혼합 자료형을 저장할 수 없으므로 여기서 통일한다.
        prices["listing_date"] = pd.to_datetime(prices["listing_date"], errors="coerce")
        return prices.drop_duplicates(["ticker", "listing_date"], keep="last")

    def _checkpoint_listing_prices(self, cached: pd.DataFrame, records: list[dict]) -> None:
        """Preserve previous dates and atomically checkpoint completed requests only."""
        if not records:
            return
        combined = pd.concat([cached, pd.DataFrame(records)], ignore_index=True)
        combined = combined.drop(columns="_cache_key", errors="ignore")
        combined["listing_date"] = pd.to_datetime(combined["listing_date"], errors="coerce")
        combined = combined.drop_duplicates(["ticker", "listing_date"], keep="last")
        target = self.raw_dir / "ipo_listing_prices.parquet"
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        try:
            combined.to_parquet(temporary, index=False)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)

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
        # 이벤트 마스터는 이전 실행에서 이미 가격·시장 보강 열을 가질 수 있다.
        # 다시 merge할 때 같은 이름을 쓰면 pandas가 중복 열을 만들므로, 가격
        # 원천을 임시 이름으로 붙인 뒤 아래에서 단일 정식 열로 합친다.
        price_columns = [
            "isu_cd", "market", "open_price", "close_price", "high_price", "low_price", "volume",
        ]
        available_price_columns = [column for column in price_columns if column in prices.columns]
        right = prices.reindex(columns=["ticker", "listing_date", *available_price_columns]).copy()
        if "listing_date" in right.columns:
            right["listing_date"] = pd.to_datetime(right["listing_date"], errors="coerce")
        right = right.rename(columns={column: f"{column}_from_price" for column in available_price_columns})
        left = calendar.copy()
        left["listing_date"] = pd.to_datetime(left["listing_date"], errors="coerce")
        merged = left.merge(right, on=["ticker", "listing_date"], how="left")
        for column in price_columns:
            source_column = f"{column}_from_price"
            if source_column not in merged.columns:
                continue
            existing = merged[column] if column in merged.columns else pd.Series(index=merged.index, dtype=object)
            # 이번 가격 조회값이 있으면 사용하고, 해당 종목의 가격 조회가 실패한
            # 경우에만 이전 실행에서 보강한 값을 유지한다.
            source_value = merged[source_column]
            merged[column] = source_value.where(source_value.notna(), existing)
        source_isu_code = merged.get("isu_cd_from_price")
        if "krx_standard_code" in merged.columns and source_isu_code is not None:
            merged["krx_standard_code"] = source_isu_code.combine_first(merged["krx_standard_code"])
        elif source_isu_code is not None:
            merged["krx_standard_code"] = source_isu_code
        # 과거 버전이 남긴 ``market_price``는 이미 ``market``에 반영했거나
        # 새 가격 원천으로 대체했으므로 저장 스키마에서는 제거한다.
        helper_columns = [column for column in merged.columns if column.endswith("_from_price")]
        if "market_price" in merged.columns:
            helper_columns.append("market_price")
        merged = merged.drop(columns=helper_columns)
        if "verification_status" in merged.columns:
            enriched = merged.get("krx_standard_code", pd.Series(index=merged.index, dtype=object)).notna()
            merged.loc[enriched, "verification_status"] = "official_source_krx_code_enriched"
        return merged

    @staticmethod
    def _build_listing_price_audit(calendar: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        """KRX 상장일 가격 매칭 결과를 실패 원인까지 보존한다."""
        event_columns = [
            "event_id", "ticker", "corp_name", "listing_date", "market", "krx_standard_code",
            "event_class", "offering_type",
        ]
        price_columns = [
            "ticker", "listing_date", "isu_cd", "market", "open_price", "close_price",
            "price_match_status", "price_match_method", "price_failure_reason",
            "price_markets_queried", "price_api_rows_returned",
            "price_raw_response_evidence", "price_matched_ticker", "price_matched_isu_cd",
            "price_matched_corp_name",
        ]
        events = calendar.reindex(columns=event_columns).copy()
        events["listing_date"] = pd.to_datetime(events["listing_date"], errors="coerce")
        result = prices.reindex(columns=price_columns).copy()
        result["listing_date"] = pd.to_datetime(result["listing_date"], errors="coerce")
        result = result.rename(columns={
            "isu_cd": "price_isu_cd", "market": "price_market",
            "open_price": "listing_open_price", "close_price": "listing_close_price",
        })
        audit = events.merge(result, on=["ticker", "listing_date"], how="left")
        audit["price_match_status"] = audit["price_match_status"].fillna("unknown_legacy_cache")
        audit["price_raw_response_verified"] = (
            audit["price_raw_response_evidence"].notna()
            & audit["price_raw_response_evidence"].astype(str).str.len().gt(0)
        )
        non_target = audit.get("event_class", pd.Series("", index=audit.index)).isin({
            "relisting", "unclassified_review", "preferred_or_class_share", "ineligible_product",
        })
        direct_match = (
            audit["price_match_status"].eq("matched")
            & audit["price_raw_response_verified"]
        )
        api_empty = audit["price_failure_reason"].eq("daily_price_api_response_empty")
        legacy_value = (
            ~audit["price_raw_response_verified"]
            & (audit["listing_open_price"].notna() | audit["listing_close_price"].notna())
        )
        audit["price_resolution_status"] = "historical_identifier_review_required"
        audit.loc[non_target, "price_resolution_status"] = "excluded_non_target_event"
        audit.loc[~non_target & direct_match, "price_resolution_status"] = "official_price_verified"
        audit.loc[~non_target & api_empty & audit["price_raw_response_verified"], "price_resolution_status"] = (
            "official_price_unconfirmed"
        )
        audit.loc[~non_target & legacy_value, "price_resolution_status"] = (
            "historical_price_cache_reaudit_required"
        )
        audit["historical_identifier_rematch_status"] = audit["price_match_method"].fillna(
            "all_identifier_methods_unmatched"
        )
        audit["human_review_required"] = audit["price_resolution_status"].isin({
            "historical_identifier_review_required", "historical_price_cache_reaudit_required",
        })
        return audit

    @staticmethod
    def _attach_price_resolution(krx_ipo: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
        """가격 감사의 최종 상태를 이벤트 마스터에 연결한다.

        가격 원시값과 검증 상태를 함께 보존해야 피처 단계에서 검증되지 않은
        캐시·식별자 불일치 가격을 타깃으로 사용하지 않을 수 있다.
        """
        columns = [
            "event_id", "price_resolution_status", "price_match_status",
            "price_match_method", "price_failure_reason", "price_raw_response_verified",
            "human_review_required",
        ]
        resolution = audit.reindex(columns=columns).drop_duplicates("event_id", keep="last")
        result = krx_ipo.drop(columns=columns[1:], errors="ignore")
        return result.merge(resolution, on="event_id", how="left")

    def _queue_document_cache(self, cache_kind: str, record: dict[str, Any]) -> None:
        """원문 파싱 캐시를 실행 종료 시 한 번에 저장하도록 누적한다."""
        if cache_kind == "demand":
            self._demand_document_cache_updates.append(record)
        elif cache_kind == "offering":
            self._offering_document_cache_updates.append(record)
        else:
            raise ValueError(f"지원하지 않는 문서 캐시 종류입니다: {cache_kind}")

    def _flush_document_cache_updates(self) -> None:
        """신규 문서 캐시를 기존 파일 캐시와 한 번만 병합한다."""
        cache_specs = (
            ("_demand_document_cache", "_demand_document_cache_updates"),
            ("_offering_document_cache", "_offering_document_cache_updates"),
        )
        for cache_attr, updates_attr in cache_specs:
            updates = getattr(self, updates_attr)
            if not updates:
                continue
            cached = getattr(self, cache_attr)
            rows = [] if cached.empty else cached.to_dict(orient="records")
            rows.extend(updates)
            setattr(self, cache_attr, pd.DataFrame.from_records(rows))
            updates.clear()

    @staticmethod
    def _select_dart_final_terms_demand(
        offering_documents: list[tuple[pd.Series, dict[str, Any]]],
        expected_offering_price: Any = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """최종 발행조건 문서에서 구조 검증된 기관 피처를 값별로 고른다.

        공동주관사별 공지를 합산하지 않는다. 발행회사와 대표주관사가 DART에
        제출한 최종 발행조건 문서의 기관 수요예측 표가 전체 기관 기준으로
        직접 기재된 경우만 사용한다. 경쟁률과 통합 확약은 같은 IPO의 서로
        다른 최신 공시에도 존재할 수 있으므로 값별 출처를 독립 보존한다.
        """
        result: dict[str, Any] = {}
        metadata = {
            "institutional_rcept_no": None,
            "institutional_rcept_dt": None,
            "lockup_rcept_no": None,
            "lockup_rcept_dt": None,
        }
        lockup_fields = LOCKUP_FIELDS
        expected_price = pd.to_numeric(
            pd.Series([expected_offering_price]), errors="coerce"
        ).iloc[0]
        ordered = sorted(
            offering_documents,
            key=lambda item: (
                pd.to_datetime(getattr(item[0], "rcept_dt", None), errors="coerce").value
                if not pd.isna(pd.to_datetime(getattr(item[0], "rcept_dt", None), errors="coerce")) else -1,
                str(getattr(item[0], "rcept_no", "")),
            ),
            reverse=True,
        )
        for candidate, document in ordered:
            if not bool(getattr(candidate, "is_final_conditions", False)):
                continue
            if document.get("dart_final_terms_demand_parser_version") != DART_FINAL_TERMS_DEMAND_PARSER_VERSION:
                continue
            if pd.notna(expected_price):
                document_price = pd.to_numeric(pd.Series([
                    document.get("demand_offering_price", document.get("offering_price"))
                ]), errors="coerce").iloc[0]
                if pd.isna(document_price) or document_price != expected_price:
                    continue
            receipt = str(candidate.rcept_no)
            receipt_date = candidate.rcept_dt
            if (
                result.get("institutional_demand_ratio") is None
                and document.get("institutional_demand_ratio") is not None
                and document.get("institutional_demand_parser_validation_status") == "structurally_verified"
                and bool(document.get("institutional_demand_rule_id"))
            ):
                for field in (
                    "institutional_demand_ratio", "institutional_demand_parse_method",
                    "institutional_demand_evidence", "institutional_demand_rule_id",
                    "institutional_demand_parser_validation_status",
                    "institutional_demand_structured_evidence",
                ):
                    result[field] = document.get(field)
                result["institutional_source_scope"] = "dart_final_terms_aggregate_institutional"
                metadata["institutional_rcept_no"] = receipt
                metadata["institutional_rcept_dt"] = receipt_date
            source_url = f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={receipt}"
            if metadata["institutional_rcept_no"] == receipt:
                result["institutional_source_url"] = source_url
            if (
                result.get("lockup_commitment_ratio") is None
                and document.get("lockup_commitment_ratio") is not None
                and document.get("lockup_parser_validation_status") == "structurally_verified"
                and bool(document.get("lockup_rule_id"))
            ):
                for field in (
                    *lockup_fields, "lockup_none_ratio", "lockup_parse_method", "lockup_parse_evidence",
                    "lockup_rule_id", "lockup_parser_validation_status", "lockup_structured_evidence",
                ):
                    result[field] = document.get(field)
                result["lockup_source_scope"] = "dart_final_terms_aggregate_institutional"
                result["lockup_source_url"] = source_url
                metadata["lockup_rcept_no"] = receipt
                metadata["lockup_rcept_dt"] = receipt_date
            if result.get("institutional_demand_ratio") is not None and result.get("lockup_commitment_ratio") is not None:
                break
        return result, metadata

    @staticmethod
    def _select_dart_lineage_demand(
        candidate_documents: list[tuple[dict[str, Any], dict[str, Any]]],
        expected_offering_price: Any,
        listing_date: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """DART 공시 계보에서 기관 경쟁률·통합 확약을 각각 최신 원문으로 고른다.

        두 값은 같은 IPO의 공식 DART 문서라는 공통 조건을 만족해야 하지만,
        법정 공시 양식상 반드시 같은 접수번호에 함께 실릴 필요는 없다. 각 값은
        회사 고유번호로 연결된 상장 전 문서, 공모가 일치, 명시적 기관 범위라는
        조건을 독립적으로 통과해야 한다.
        """
        result: dict[str, Any] = {}
        metadata = {
            "institutional_rcept_no": None,
            "institutional_rcept_dt": None,
            "lockup_rcept_no": None,
            "lockup_rcept_dt": None,
        }
        expected_price = pd.to_numeric(pd.Series([expected_offering_price]), errors="coerce").iloc[0]
        listing_timestamp = pd.to_datetime(listing_date, errors="coerce")
        def candidate_sort_key(item: tuple[dict[str, Any], dict[str, Any]]) -> tuple[int, int, str]:
            timestamp = pd.to_datetime(item[0].get("rcept_dt"), errors="coerce")
            timestamp_value = -1 if pd.isna(timestamp) else int(timestamp.value)
            return timestamp_value, int(item[0].get("candidate_score") or 0), str(item[0].get("rcept_no", ""))

        ordered = sorted(candidate_documents, key=candidate_sort_key, reverse=True)
        for candidate, document in ordered:
            receipt = str(candidate.get("rcept_no", "")).strip()
            receipt_date = pd.to_datetime(candidate.get("rcept_dt"), errors="coerce")
            if not receipt or pd.isna(receipt_date) or (not pd.isna(listing_timestamp) and receipt_date >= listing_timestamp):
                continue
            document_price = pd.to_numeric(pd.Series([
                document.get("demand_offering_price", document.get("offering_price"))
            ]), errors="coerce").iloc[0]
            if pd.isna(expected_price) or pd.isna(document_price) or document_price != expected_price:
                continue
            source_url = f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={receipt}"
            if (
                result.get("institutional_demand_ratio") is None
                and document.get("institutional_demand_ratio") is not None
                and document.get("institutional_demand_parser_validation_status") == "structurally_verified"
                and bool(document.get("institutional_demand_rule_id"))
            ):
                result.update({
                    "institutional_demand_ratio": document["institutional_demand_ratio"],
                    "institutional_demand_parse_method": document.get("institutional_demand_parse_method"),
                    "institutional_demand_evidence": document.get("institutional_demand_evidence"),
                    "institutional_demand_rule_id": document.get("institutional_demand_rule_id"),
                    "institutional_demand_parser_validation_status": document.get(
                        "institutional_demand_parser_validation_status"
                    ),
                    "institutional_demand_structured_evidence": document.get(
                        "institutional_demand_structured_evidence"
                    ),
                    "institutional_source_scope": "dart_lineage_aggregate_institutional",
                    "institutional_source_url": source_url,
                })
                metadata["institutional_rcept_no"] = receipt
                metadata["institutional_rcept_dt"] = receipt_date
            if (
                result.get("lockup_commitment_ratio") is None
                and document.get("lockup_commitment_ratio") is not None
                and document.get("lockup_parser_validation_status") == "structurally_verified"
                and bool(document.get("lockup_rule_id"))
            ):
                for field in (
                    *LOCKUP_FIELDS, "lockup_none_ratio", "lockup_parse_method", "lockup_parse_evidence",
                    "lockup_rule_id", "lockup_parser_validation_status", "lockup_structured_evidence",
                ):
                    result[field] = document.get(field)
                result.update({
                    "lockup_source_scope": "dart_lineage_aggregate_institutional",
                    "lockup_source_url": source_url,
                })
                metadata["lockup_rcept_no"] = receipt
                metadata["lockup_rcept_dt"] = receipt_date
        return result, metadata

    def _collect_dart_records(
        self,
        calendar: pd.DataFrame,
        start_year: int,
        end_year: int,
        *,
        include_dart_demand_audit: bool,
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
        offering_cache_by_receipt: dict[str, dict[str, Any]] = {}
        reusable_offering_columns = {
            "rcept_no", "offering_price_parser_version", "structured_price_check_version",
            "dart_final_terms_demand_parser_version",
        }
        for frame in (self._offering_document_cache, cached_records):
            if frame.empty or not reusable_offering_columns.issubset(frame.columns):
                continue
            reusable_offerings = frame[
                frame["offering_price_parser_version"].eq(OFFERING_PRICE_PARSER_VERSION)
                & frame["structured_price_check_version"].eq(STRUCTURED_PRICE_CHECK_VERSION)
                & frame["dart_final_terms_demand_parser_version"].eq(
                    DART_FINAL_TERMS_DEMAND_PARSER_VERSION
                )
            ].drop_duplicates("rcept_no", keep="last")
            offering_cache_by_receipt.update({
                str(row.rcept_no): row._asdict() for row in reusable_offerings.itertuples(index=False)
            })
        demand_cache_by_receipt: dict[str, dict[str, Any]] = {}
        if not self._demand_document_cache.empty and {
            "rcept_no", "demand_parser_version",
        }.issubset(self._demand_document_cache.columns):
            reusable_demand = self._demand_document_cache[
                self._demand_document_cache["demand_parser_version"].eq(DART_DEMAND_PARSER_VERSION)
            ].drop_duplicates("rcept_no", keep="last")
            demand_cache_by_receipt = {
                str(row.rcept_no): row._asdict() for row in reusable_demand.itertuples(index=False)
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

            offering_documents: list[tuple[pd.Series, dict[str, Any]]] = []
            for rank, candidate in candidates.iterrows():
                entry = lineage_entries[rank]
                receipt = str(candidate.rcept_no)
                cached = offering_cache_by_receipt.get(receipt) or cached_by_receipt.get(receipt)
                if cached is not None and (
                    cached.get("structured_price_check_version") == STRUCTURED_PRICE_CHECK_VERSION
                    and cached.get("offering_price_parser_version") == OFFERING_PRICE_PARSER_VERSION
                    and cached.get("dart_final_terms_demand_parser_version")
                    == DART_FINAL_TERMS_DEMAND_PARSER_VERSION
                ):
                    offering_documents.append((candidate, cached))
                    entry["attempt_status"] = "cached_verified_record"
                    continue
                if not self._should_retry_document(receipt):
                    entry["attempt_status"] = "retry_deferred"
                    continue
                try:
                    parsed_offering = self.dart.get_offering_info(receipt)
                    parsed_offering["structured_price_check_version"] = STRUCTURED_PRICE_CHECK_VERSION
                    parsed_offering["offering_price_parser_version"] = OFFERING_PRICE_PARSER_VERSION
                    parsed_offering["dart_final_terms_demand_parser_version"] = (
                        DART_FINAL_TERMS_DEMAND_PARSER_VERSION
                    )
                    self._queue_document_cache("offering", parsed_offering)
                    offering_cache_by_receipt[receipt] = parsed_offering
                    offering_documents.append((candidate, parsed_offering))
                    entry["attempt_status"] = "document_parsed"
                except RuntimeError as exc:
                    if "<status>014</status>" in str(exc):
                        has_structured = any(
                            item.get("rcept_no") == receipt for item in structured_prices
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
            if not offering_documents:
                continue

            # 최종 확정 공모가의 승인 문서는 최신 우선순위로 선택하되, 희망
            # 밴드·공모 구조는 같은 계보의 다른 신고서에만 있는 경우가 있어
            # 값별로 가장 최신의 실제 원문 값을 보완한다.
            filing, offering = offering_documents[0]

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
            # 신주·구주는 같은 문서의 구조 검증을 동시에 통과한
            # 경우만 파생 비율에 쓴다. 서로 다른 정정본의 값을
            # 조합하면 실제 공시에 없는 공모 구조가 만들어진다.
            share_fields = (
                "new_shares", "new_shares_parse_method", "new_shares_parse_evidence",
                "new_shares_parser_validation_status", "secondary_shares",
                "secondary_shares_parse_method", "secondary_shares_parse_evidence",
                "secondary_shares_parser_validation_status", "total_post_listing_shares",
                "total_post_listing_shares_parse_method",
                "total_post_listing_shares_parse_evidence",
                "total_post_listing_shares_parser_validation_status",
            )
            share_source = next((
                (candidate, document) for candidate, document in offering_documents
                if document.get("new_shares_parser_validation_status") == "structurally_verified"
                and document.get("secondary_shares_parser_validation_status") == "structurally_verified"
            ), None)
            if share_source is not None:
                source_candidate, source_document = share_source
                for field in share_fields:
                    offering[field] = source_document.get(field)
                offering["offering_structure_rcept_no"] = str(source_candidate.rcept_no)
                offering["offering_structure_rcept_dt"] = source_candidate.rcept_dt

            float_fields = (
                "public_float_shares", "public_float_ratio_disclosed",
                "public_float_parse_method", "public_float_parse_evidence",
                "total_post_listing_shares", "total_post_listing_shares_parse_method",
                "total_post_listing_shares_parse_evidence",
                "total_post_listing_shares_parser_validation_status",
            )
            float_source = next((
                (candidate, document) for candidate, document in offering_documents
                if document.get("public_float_parse_method") ==
                    "disclosed_public_float_ratio_direct_context"
                or (
                    document.get("public_float_parse_method") ==
                        "disclosed_public_float_shares_direct_context"
                    and document.get("total_post_listing_shares_parser_validation_status") ==
                        "structurally_verified"
                )
            ), None)
            if float_source is not None:
                source_candidate, source_document = float_source
                for field in float_fields:
                    offering[field] = source_document.get(field)
                offering["public_float_rcept_no"] = str(source_candidate.rcept_no)
                offering["public_float_rcept_dt"] = source_candidate.rcept_dt

            field_source_groups = {
                "price_band": ("price_band_low", "price_band_high"),
                "governance_structure": (
                    "lead_underwriter", "major_shareholder_lockup_months", "risk_factor_count",
                ),
            }
            for source_group, fields in field_source_groups.items():
                source_candidate = next(
                    (candidate for candidate, document in offering_documents if any(
                        document.get(field) is not None for field in fields
                    )),
                    None,
                )
                if source_candidate is None:
                    continue
                source_document = next(
                    document for candidate, document in offering_documents
                    if str(candidate.rcept_no) == str(source_candidate.rcept_no)
                )
                for field in fields:
                    if offering.get(field) is None and source_document.get(field) is not None:
                        offering[field] = source_document[field]
                offering[f"{source_group}_rcept_no"] = str(source_candidate.rcept_no)
                offering[f"{source_group}_rcept_dt"] = source_candidate.rcept_dt

            # DART 최종 발행조건 문서가 통합 기관 수요예측·확약 표를 직접
            # 제공하면 이것이 1차 모델 원천이다. 공동주관사별 숫자를 모으거나
            # 합산하지 않는다.
            verified_demand, verified_demand_metadata = self._select_dart_final_terms_demand(
                offering_documents, offering.get("offering_price")
            )
            institutional_rcept_no = verified_demand_metadata["institutional_rcept_no"]
            institutional_rcept_dt = verified_demand_metadata["institutional_rcept_dt"]
            lockup_rcept_no = verified_demand_metadata["lockup_rcept_no"]
            lockup_rcept_dt = verified_demand_metadata["lockup_rcept_dt"]

            demand_candidates: list[dict[str, Any]] = []
            # 최종 발행조건 문서에 두 값이 모두 실린다는 가정을 두지 않는다.
            # 같은 IPO의 상장 전 DART 계보에서 경쟁률·통합 확약을 각각 찾되,
            # 각 값은 공모가와 시점까지 독립적으로 검증한다.
            if include_dart_demand_audit:
                try:
                    demand_records_method = getattr(
                        self.dart, "find_demand_forecast_disclosure_records", None
                    )
                    search_start = pd.Timestamp(listing.listing_date) - pd.Timedelta(
                        days=MAX_FILING_TO_LISTING_DAYS
                    )
                    if callable(demand_records_method):
                        demand_candidates.extend(demand_records_method(
                            str(filing.corp_code),
                            search_start.strftime("%Y%m%d"),
                            pd.Timestamp(listing.listing_date).strftime("%Y%m%d"),
                        ))
                    else:
                        demand_record_method = getattr(
                            self.dart, "find_demand_forecast_disclosure_record", None
                        )
                        if callable(demand_record_method):
                            record = demand_record_method(
                                str(filing.corp_code), search_start.strftime("%Y%m%d"),
                                pd.Timestamp(listing.listing_date).strftime("%Y%m%d"),
                            )
                            if record:
                                demand_candidates.append(record)
                        else:
                            legacy_demand_method = getattr(
                                self.dart, "find_demand_forecast_disclosure", None
                            )
                            if callable(legacy_demand_method):
                                receipt = legacy_demand_method(
                                    str(filing.corp_code), search_start.strftime("%Y%m%d"),
                                    pd.Timestamp(listing.listing_date).strftime("%Y%m%d"),
                                )
                                if receipt:
                                    demand_candidates.append({"rcept_no": str(receipt)})
                except RuntimeError as exc:
                    logger.warning("수요예측 공시 계보 조회 실패 (%s): %s", filing.corp_name, exc)

                # 목록 API 제목이 누락·축약된 경우에도 이미 연결한 증권신고서 계보는
                # 수요예측 결과를 담을 수 있다. 후보를 합치되 접수번호별로 한 번만 읽는다.
                demand_candidates.extend({
                    "rcept_no": str(candidate.rcept_no),
                    "rcept_dt": candidate.rcept_dt,
                    "report_nm": candidate.report_nm,
                    "candidate_score": 80 if bool(candidate.is_final_conditions) else 5,
                } for candidate in candidates.itertuples(index=False))
            deduped_demand_candidates: list[dict[str, Any]] = []
            seen_demand_receipts: set[str] = set()
            for candidate in demand_candidates:
                receipt = str(candidate.get("rcept_no", "")).strip()
                if not receipt or receipt in seen_demand_receipts:
                    continue
                seen_demand_receipts.add(receipt)
                deduped_demand_candidates.append(candidate)
            demand_candidates = deduped_demand_candidates[:MAX_DEMAND_DOCUMENT_CANDIDATES]

            demand_candidate_documents: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for rank, demand_candidate in enumerate(demand_candidates, start=1):
                candidate_receipt = str(demand_candidate["rcept_no"])
                candidate_dt = demand_candidate.get("rcept_dt")
                self._lineage_rows.append({
                    "event_id": event_id,
                    "ticker": getattr(listing, "ticker", None),
                    "krx_standard_code": getattr(listing, "krx_standard_code", None),
                    "listing_date": listing.listing_date,
                    "corp_name": filing.corp_name,
                    "corp_code": str(filing.corp_code),
                    "rcept_no": candidate_receipt,
                    "rcept_dt": candidate_dt,
                    "filing_report_nm": demand_candidate.get("report_nm"),
                    "lineage_role": "demand_forecast_candidate",
                    "selection_rank": rank,
                    "match_method": "dart_corp_code_pre_listing_demand_lineage",
                    "lineage_validation_status": "pre_listing_demand_candidate",
                    "source_name": "OpenDART_list",
                    "source_url": "https://opendart.fss.or.kr/api/list.json",
                    "lineage_version": DART_LINEAGE_VERSION,
                    "attempt_status": "not_attempted",
                    "recorded_at": pd.Timestamp.now(tz="Asia/Seoul"),
                })
                try:
                    cached_demand = demand_cache_by_receipt.get(candidate_receipt)
                    if (
                        cached_demand is not None
                        and cached_demand.get("demand_parser_version") == DART_DEMAND_PARSER_VERSION
                    ):
                        parsed_demand = cached_demand
                        self._lineage_rows[-1]["attempt_status"] = "cached_parser_v2"
                    elif not self._should_retry_demand_document(candidate_receipt):
                        self._lineage_rows[-1]["attempt_status"] = "retry_deferred"
                        logger.info("수요예측 원문 재시도 보류 (%s, %s)", filing.corp_name, candidate_receipt)
                        continue
                    else:
                        parsed_demand = self.dart.get_demand_forecast(str(filing.corp_code), candidate_receipt)
                        cache_record = {
                            "rcept_no": candidate_receipt,
                            "corp_code": str(filing.corp_code),
                            "corp_name": filing.corp_name,
                            "rcept_dt": candidate_dt,
                            "report_nm": demand_candidate.get("report_nm"),
                            "demand_parser_version": DART_DEMAND_PARSER_VERSION,
                            "parsed_at": pd.Timestamp.now(tz="Asia/Seoul"),
                            **{key: value for key, value in parsed_demand.items() if key != "corp_code"},
                        }
                        self._queue_document_cache("demand", cache_record)
                        demand_cache_by_receipt[candidate_receipt] = cache_record
                        self._lineage_rows[-1]["attempt_status"] = "document_parsed"
                except RuntimeError as exc:
                    self._record_demand_document_failure(
                        rcept_no=candidate_receipt,
                        corp_code=str(filing.corp_code),
                        corp_name=filing.corp_name,
                        event_id=event_id,
                        listing_date=listing.listing_date,
                        error=exc,
                    )
                    self._lineage_rows[-1]["attempt_status"] = "document_failure_recorded"
                    logger.warning("수요예측 원문 파싱 실패 (%s, %s): %s", filing.corp_name, candidate_receipt, exc)
                    continue

                demand_candidate_documents.append((demand_candidate, parsed_demand))

            lineage_demand, lineage_metadata = self._select_dart_lineage_demand(
                demand_candidate_documents, offering.get("offering_price"), listing.listing_date
            )
            if verified_demand.get("institutional_demand_ratio") is None and lineage_demand.get(
                "institutional_demand_ratio"
            ) is not None:
                for field in (
                    "institutional_demand_ratio", "institutional_demand_parse_method",
                    "institutional_demand_evidence", "institutional_demand_rule_id",
                    "institutional_demand_parser_validation_status",
                    "institutional_demand_structured_evidence", "institutional_source_scope",
                    "institutional_source_url",
                ):
                    verified_demand[field] = lineage_demand.get(field)
                institutional_rcept_no = lineage_metadata["institutional_rcept_no"]
                institutional_rcept_dt = lineage_metadata["institutional_rcept_dt"]
            if verified_demand.get("lockup_commitment_ratio") is None and lineage_demand.get(
                "lockup_commitment_ratio"
            ) is not None:
                for field in (
                    *LOCKUP_FIELDS, "lockup_none_ratio", "lockup_parse_method", "lockup_parse_evidence",
                    "lockup_rule_id", "lockup_parser_validation_status", "lockup_structured_evidence",
                    "lockup_source_scope", "lockup_source_url",
                ):
                    verified_demand[field] = lineage_demand.get(field)
                lockup_rcept_no = lineage_metadata["lockup_rcept_no"]
                lockup_rcept_dt = lineage_metadata["lockup_rcept_dt"]

            demand_rcept_no = institutional_rcept_no or lockup_rcept_no
            demand_rcept_dt = institutional_rcept_dt or lockup_rcept_dt

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
                "demand_rcept_dt": demand_rcept_dt,
                "institutional_rcept_no": institutional_rcept_no,
                "institutional_rcept_dt": institutional_rcept_dt,
                "lockup_rcept_no": lockup_rcept_no,
                "lockup_rcept_dt": lockup_rcept_dt,
                "institutional_available_at": institutional_rcept_dt,
                "lockup_available_at": lockup_rcept_dt,
                "institutional_data_contract_version": DART_INSTITUTIONAL_DATA_CONTRACT_VERSION,
                "institutional_source_url": verified_demand.get("institutional_source_url"),
                "lockup_source_url": verified_demand.get("lockup_source_url"),
                "institutional_validation_status": (
                    DART_STRUCTURAL_AGGREGATE_STATUS
                    if (
                        verified_demand.get("institutional_demand_ratio") is not None
                        and verified_demand.get("institutional_demand_parser_validation_status")
                        == "structurally_verified"
                        and bool(verified_demand.get("institutional_demand_rule_id"))
                    )
                    else "dart_aggregate_value_not_verified"
                ),
                "lockup_validation_status": (
                    DART_STRUCTURAL_AGGREGATE_STATUS
                    if (
                        verified_demand.get("lockup_commitment_ratio") is not None
                        and verified_demand.get("lockup_parser_validation_status")
                        == "structurally_verified"
                        and bool(verified_demand.get("lockup_rule_id"))
                    )
                    else "dart_aggregate_value_not_verified"
                ),
                **offering,
                # 이 행은 현재 KRX 상장 이벤트에 맞춰 수집한 공시다. 원문에
                # 기재된 상장예정일은 별도 보존하고, 병합 키는 실제 상장일을 쓴다.
                "disclosed_listing_date": offering.get("listing_date"),
                "listing_date": listing.listing_date,
                **verified_demand,
                **self._compare_offering_sources(offering, verified_demand),
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
        return self._should_retry_document_failure(self._demand_document_failures, rcept_no)

    @staticmethod
    def _should_retry_document_failure(failures: pd.DataFrame, rcept_no: str) -> bool:
        """같은 ZIP 실패는 TTL 이후에만 재시도해 호출 낭비를 막는다."""
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
    def _build_feature_coverage_audit(observations: pd.DataFrame) -> pd.DataFrame:
        """피처별 원시 충족률과 결측·검증 상태를 감사 가능한 집계로 만든다."""
        columns = [
            "feature_name", "total_rows", "observed_rows", "missing_rows", "coverage_rate",
            "human_review_required_rows", "source_reference_rows", "available_at_rows",
            "missing_reason_counts", "validation_status_counts",
        ]
        if observations.empty or "feature_name" not in observations.columns:
            return pd.DataFrame(columns=columns)

        rows: list[dict[str, object]] = []
        for feature_name, group in observations.groupby("feature_name", dropna=False):
            missing = group.get("is_missing", pd.Series(True, index=group.index)).fillna(True).astype(bool)
            review_required = group.get(
                "human_review_required", pd.Series(False, index=group.index)
            ).fillna(False).astype(bool)
            missing_reasons = group.loc[missing, "missing_reason"].fillna("unspecified").astype(str)
            validation = group.get(
                "validation_status", pd.Series("unspecified", index=group.index)
            ).fillna("unspecified").astype(str)
            source_reference = group.get(
                "source_reference", pd.Series(index=group.index, dtype=object)
            )
            available_at = group.get("available_at", pd.Series(index=group.index, dtype=object))
            total = int(len(group))
            observed = int((~missing).sum())
            rows.append({
                "feature_name": str(feature_name),
                "total_rows": total,
                "observed_rows": observed,
                "missing_rows": int(missing.sum()),
                "coverage_rate": round(observed / total, 4) if total else 0.0,
                "human_review_required_rows": int(review_required.sum()),
                "source_reference_rows": int(source_reference.notna().sum()),
                "available_at_rows": int(available_at.notna().sum()),
                "missing_reason_counts": json.dumps(
                    {str(key): int(value) for key, value in missing_reasons.value_counts().items()},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "validation_status_counts": json.dumps(
                    {str(key): int(value) for key, value in validation.value_counts().items()},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            })
        return pd.DataFrame(rows, columns=columns).sort_values("feature_name").reset_index(drop=True)

    @staticmethod
    def _build_feature_time_audit(
        features: pd.DataFrame, observations: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """각 피처의 실제 공개 시각이 상장 전인지 검사한다."""
        if observations is None or observations.empty:
            audit = features.reindex(columns=[
                "event_id", "corp_name", "listing_date", "feature_available_at",
            ]).copy()
            audit["feature_name"] = "all_features_legacy"
            audit["available_at"] = audit["feature_available_at"]
        else:
            audit = observations.reindex(columns=[
                "event_id", "corp_name", "listing_date", "feature_name", "available_at", "is_missing",
            ]).copy()
            audit["feature_available_at"] = audit["available_at"]
        listing_date = pd.to_datetime(audit["listing_date"], errors="coerce")
        available_at = pd.to_datetime(audit["available_at"], errors="coerce")
        # DART/KIND의 날짜 필드는 공시 시각을 제공하지 않는다. 상장 당일 값은
        # 오전 9시 이전 공개를 증명할 수 없으므로 상장 전 예측에는 사용하지 않는다.
        audit["is_future_information"] = (
            available_at.notna() & listing_date.notna() & (available_at >= listing_date)
        )
        audit["time_validation_status"] = "missing_feature_available_at_review_required"
        audit.loc[available_at.notna() & listing_date.notna() & ~audit["is_future_information"], "time_validation_status"] = (
            "pre_listing_verified"
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
            "price_band_rcept_no", "price_band_rcept_dt",
            "offering_structure_rcept_no", "offering_structure_rcept_dt",
            "price_band_check", "dart_structured_offering_price", "dart_structured_security_type",
            "structured_price_check", "structured_price_record_count", "structured_price_check_version",
            "institutional_rcept_no", "institutional_rcept_dt",
            "lockup_rcept_no", "lockup_rcept_dt", "institutional_demand_ratio",
            "institutional_demand_parse_method", "institutional_demand_evidence",
            "lockup_commitment_ratio", "lockup_parse_method", "lockup_parse_evidence",
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
        institutional_statuses = dart_ipo.get(
            "institutional_validation_status", pd.Series(dtype=str)
        ).fillna("missing").value_counts().to_dict()
        lockup_statuses = dart_ipo.get(
            "lockup_validation_status", pd.Series(dtype=str)
        ).fillna("missing").value_counts().to_dict()
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
            "lockup_rows": int(dart_ipo.get("lockup_commitment_ratio", pd.Series(dtype=float)).notna().sum()),
            "institutional_validation_status_counts": institutional_statuses,
            "lockup_validation_status_counts": lockup_statuses,
            "financial_revenue_rows": int(dart_ipo.get("revenue", pd.Series(dtype=float)).notna().sum()),
            "extreme_open_return_rows": int((open_return.abs() > 200).sum()),
            "source": "OpenDART + KRX OpenAPI",
        }
