"""주관사 공식 공지의 청약·수요예측 결과를 감사 가능하게 수집한다.

개인 계정, 로그인 화면, 비공식 집계 사이트는 사용하지 않는다. 입력 URL은
KRX 이벤트 ID와 연결된 공개 공지여야 하며, 자동 승인 가능한 값은 통합
경쟁률처럼 전체 참여 증권사를 포함한다고 원문이 명시한 경우로 제한한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import re
from typing import Any
from urllib.parse import urlparse

from bs4 import BeautifulSoup
import pandas as pd
import requests

from data.collectors.underwriter_registry import normalize_underwriter, official_hosts

OFFICIAL_UNDERWRITER_HOSTS = official_hosts()

RESULT_COLUMNS = [
    "notice_id", "notice_version_id", "source_version", "revision_of_notice_id", "is_correction",
    "event_id", "corp_name", "lead_underwriter", "notice_title", "notice_url",
    "source_host", "source_type", "published_at", "available_at", "collected_at",
    "source_document_sha256", "source_offering_price", "subscription_start", "subscription_end",
    "event_listing_date", "event_offering_price", "event_context_validation_status",
    "retail_subscription_ratio", "retail_ratio_scope", "institutional_demand_ratio",
    "lockup_ratio", "parse_evidence", "validation_status", "missing_reason",
    "human_review_required",
]


@dataclass(frozen=True)
class OfficialNoticeSource:
    event_id: str
    corp_name: str
    lead_underwriter: str
    notice_url: str
    source_type: str = "public_notice"
    published_at: str | None = None
    source_version: str = "initial"
    revision_of_notice_id: str | None = None
    is_correction: bool = False
    notice_title: str | None = None
    source_offering_price: str | None = None
    subscription_start: str | None = None
    subscription_end: str | None = None


class OfficialUnderwriterCollector:
    """공개된 주관사 공지 URL만 읽어 청약 결과를 추출하는 수집기."""

    def __init__(self, session: requests.Session | None = None):
        self.session = session or requests.Session()

    @staticmethod
    def notice_id(event_id: str, notice_url: str, source_version: str = "initial") -> str:
        return sha256(f"{event_id}|{notice_url}|{source_version}".encode("utf-8")).hexdigest()

    @staticmethod
    def _ratio(value: str) -> float:
        return float(value.replace(",", ""))

    @staticmethod
    def _direct_ratio(blocks: list[str], labels: tuple[str, ...]) -> tuple[float | None, str | None]:
        """같은 표 행 또는 문단에서 라벨과 ``숫자 : 1``이 맞닿은 값만 찾는다."""
        for block in blocks:
            normalized = re.sub(r"\s+", " ", block).strip()
            for label in labels:
                match = re.search(
                    rf"{label}\s*(?:은|는|:|：)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(?::|：|대)\s*1\b",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if match:
                    return OfficialUnderwriterCollector._ratio(match.group(1)), match.group(0)[:240]
        return None, None

    @staticmethod
    def _direct_percent(blocks: list[str], labels: tuple[str, ...]) -> tuple[float | None, str | None]:
        """같은 표 행 또는 문단에서 라벨과 백분율이 직접 연결된 값만 찾는다."""
        for block in blocks:
            normalized = re.sub(r"\s+", " ", block).strip()
            for label in labels:
                match = re.search(
                    rf"{label}\s*(?:은|는|:|：)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*%",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if match:
                    return OfficialUnderwriterCollector._ratio(match.group(1)) / 100, match.group(0)[:240]
        return None, None

    @staticmethod
    def _as_blocks(content: bytes, content_type: str) -> tuple[list[str] | None, str | None]:
        """HTML 또는 공개 PDF에서 원문 구조를 최대한 보존한 텍스트 블록을 만든다."""
        content_type = content_type.lower()
        if "pdf" in content_type or content.startswith(b"%PDF"):
            try:
                from pypdf import PdfReader
            except ImportError:
                return None, "pdf_parser_not_installed"
            try:
                text = "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(content)).pages)
            except Exception:
                return None, "pdf_text_extract_failed"
            return [line.strip() for line in text.splitlines() if line.strip()], None
        if "html" not in content_type:
            return None, "source_document_type_not_supported"
        charset_match = re.search(r"charset\s*=\s*([A-Za-z0-9_-]+)", content_type, re.IGNORECASE)
        encoding = charset_match.group(1) if charset_match else "utf-8"
        try:
            decoded = content.decode(encoding, errors="replace")
        except LookupError:
            decoded = content.decode("utf-8", errors="replace")
        soup = BeautifulSoup(decoded, "html.parser")
        blocks = [element.get_text(" ", strip=True) for element in soup.select("tr, p, li")]
        blocks.extend(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
        return list(dict.fromkeys(block for block in blocks if block)), None

    @staticmethod
    def _as_timestamp(value: object) -> pd.Timestamp | None:
        if value is None or str(value).strip() == "":
            return None
        timestamp = pd.to_datetime(value, errors="coerce")
        return None if pd.isna(timestamp) else timestamp

    @staticmethod
    def _as_number(value: object) -> float | None:
        number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return None if pd.isna(number) else float(number)

    @staticmethod
    def _clean_name(value: object) -> str:
        return re.sub(r"[^0-9A-Z가-힣]", "", str(value or "").upper().replace("주식회사", ""))

    def _validate_event_context(
        self, source: OfficialNoticeSource, event_context: dict[str, Any] | None
    ) -> tuple[str, dict[str, object]]:
        """공지 메타데이터가 KRX 이벤트·상장 전 시점과 맞는지 검사한다."""
        details: dict[str, object] = {
            "event_listing_date": None,
            "event_offering_price": None,
        }
        if event_context is None:
            return "not_checked_no_event_context", details
        expected_name = self._clean_name(event_context.get("corp_name"))
        source_name = self._clean_name(source.corp_name)
        expected_underwriter = normalize_underwriter(event_context.get("lead_underwriter"))
        source_underwriter = normalize_underwriter(source.lead_underwriter)
        listing_date = self._as_timestamp(event_context.get("listing_date"))
        event_price = self._as_number(event_context.get("offering_price"))
        published_at = self._as_timestamp(source.published_at)
        subscription_end = self._as_timestamp(source.subscription_end)
        source_price = self._as_number(source.source_offering_price)
        details["event_listing_date"] = listing_date
        details["event_offering_price"] = event_price
        if not expected_name or source_name != expected_name:
            return "needs_review_corp_name_mismatch", details
        if expected_underwriter and source_underwriter != expected_underwriter:
            return "needs_review_underwriter_mismatch", details
        if listing_date is None or published_at is None:
            return "needs_review_missing_notice_published_at", details
        if published_at >= listing_date:
            return "needs_review_notice_not_pre_listing", details
        if subscription_end is None:
            return "needs_review_missing_subscription_period", details
        if subscription_end >= listing_date:
            return "needs_review_subscription_period_not_pre_listing", details
        if event_price is not None and source_price is None:
            return "needs_review_missing_offering_price_crosscheck", details
        if event_price is not None and source_price != event_price:
            return "needs_review_offering_price_mismatch", details
        return "verified_event_context", details

    def collect_notice(
        self, source: OfficialNoticeSource, event_context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """한 공식 공지를 추출하고, 원문 근거와 승인 상태를 함께 반환한다."""
        host = (urlparse(source.notice_url).hostname or "").lower()
        record: dict[str, Any] = {
            "notice_id": self.notice_id(source.event_id, source.notice_url, source.source_version),
            "notice_version_id": None,
            "source_version": source.source_version,
            "revision_of_notice_id": source.revision_of_notice_id,
            "is_correction": source.is_correction,
            "event_id": source.event_id,
            "corp_name": source.corp_name,
            "lead_underwriter": source.lead_underwriter,
            "notice_title": source.notice_title,
            "notice_url": source.notice_url,
            "source_host": host,
            "source_type": source.source_type,
            "published_at": source.published_at,
            "available_at": source.published_at,
            "collected_at": pd.Timestamp.now(tz="Asia/Seoul"),
            "source_document_sha256": None,
            "source_offering_price": source.source_offering_price,
            "subscription_start": source.subscription_start,
            "subscription_end": source.subscription_end,
            "event_listing_date": None,
            "event_offering_price": None,
            "event_context_validation_status": "not_checked_no_event_context",
            "retail_subscription_ratio": None,
            "retail_ratio_scope": None,
            "institutional_demand_ratio": None,
            "lockup_ratio": None,
            "parse_evidence": None,
            "validation_status": "needs_review",
            "missing_reason": None,
            "human_review_required": True,
        }
        context_status, context_details = self._validate_event_context(source, event_context)
        record.update(context_details, event_context_validation_status=context_status)
        if host not in OFFICIAL_UNDERWRITER_HOSTS:
            record.update(
                validation_status="rejected_non_official_underwriter_host",
                missing_reason="source_host_not_in_official_registry",
            )
            return record
        if urlparse(source.notice_url).scheme != "https":
            record.update(
                validation_status="rejected_non_https_source",
                missing_reason="source_requires_https",
            )
            return record
        try:
            response = self.session.get(
                source.notice_url,
                headers={"Accept": "text/html,application/xhtml+xml"},
                timeout=30,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            record.update(
                validation_status="source_access_failed",
                missing_reason=f"{type(exc).__name__}_retry_required",
            )
            return record

        content_hash = sha256(response.content).hexdigest()
        record["source_document_sha256"] = content_hash
        record["notice_version_id"] = sha256(
            f"{record['notice_id']}|{content_hash}".encode("utf-8")
        ).hexdigest()

        blocks, document_error = self._as_blocks(
            response.content, response.headers.get("Content-Type", "")
        )
        if blocks is None:
            record.update(
                validation_status="source_document_type_not_supported",
                missing_reason=document_error,
            )
            return record

        retail, retail_evidence = self._direct_ratio(
            blocks, ("통합경쟁률", "통합 경쟁률", "일반청약 경쟁률", "일반 경쟁률", "비례배정 경쟁률")
        )
        institutional, institutional_evidence = self._direct_ratio(
            blocks, ("기관투자자 경쟁률", "기관 수요예측 경쟁률", "수요예측 경쟁률")
        )
        lockup, lockup_evidence = self._direct_percent(
            blocks, ("의무보유확약 비율", "의무보유 확약 비율", "확약비율")
        )
        evidence = retail_evidence or institutional_evidence or lockup_evidence
        scope = "integrated" if retail_evidence and "통합" in retail_evidence else None
        if retail is not None and scope is None:
            scope = "underwriter_only_or_unknown"
        record.update(
            retail_subscription_ratio=retail,
            retail_ratio_scope=scope,
            institutional_demand_ratio=institutional,
            lockup_ratio=lockup,
            parse_evidence=evidence,
        )
        context_is_valid = context_status in {"verified_event_context", "not_checked_no_event_context"}
        if retail is not None and scope == "integrated" and context_is_valid:
            record.update(validation_status="official_notice_integrated_retail_ratio", human_review_required=False)
        elif retail is not None and scope == "integrated":
            record.update(validation_status="official_notice_event_link_review_required")
        elif any(value is not None for value in (retail, institutional, lockup)):
            record.update(validation_status="official_notice_value_requires_scope_review")
        else:
            record.update(
                validation_status="official_notice_no_supported_value",
                missing_reason="supported_labels_not_found",
            )
        return record

    @staticmethod
    def _source_value(row: object, field: str, default: object = None) -> object:
        value = getattr(row, field, default)
        return default if pd.isna(value) or str(value).strip() == "" else value

    def collect_sources(
        self, sources: pd.DataFrame, event_contexts: dict[str, dict[str, Any]] | None = None
    ) -> pd.DataFrame:
        required = {"event_id", "corp_name", "lead_underwriter", "notice_url"}
        missing = required - set(sources.columns)
        if missing:
            raise ValueError(f"주관사 공식 공지 입력 파일에 필요한 열이 없습니다: {', '.join(sorted(missing))}")
        records = []
        for row in sources.itertuples(index=False):
            records.append(self.collect_notice(OfficialNoticeSource(
                event_id=str(self._source_value(row, "event_id", "")),
                corp_name=str(self._source_value(row, "corp_name", "")),
                lead_underwriter=str(self._source_value(row, "lead_underwriter", "")),
                notice_url=str(self._source_value(row, "notice_url", "")),
                source_type=str(self._source_value(row, "source_type", "public_notice")),
                published_at=self._source_value(row, "published_at"),
                source_version=str(self._source_value(row, "source_version", "initial")),
                revision_of_notice_id=self._source_value(row, "revision_of_notice_id"),
                is_correction=str(self._source_value(row, "is_correction", "false")).lower() == "true",
                notice_title=self._source_value(row, "notice_title"),
                source_offering_price=self._source_value(row, "source_offering_price"),
                subscription_start=self._source_value(row, "subscription_start"),
                subscription_end=self._source_value(row, "subscription_end"),
            ), (event_contexts or {}).get(str(self._source_value(row, "event_id", "")))))
        return pd.DataFrame(records, columns=RESULT_COLUMNS)
