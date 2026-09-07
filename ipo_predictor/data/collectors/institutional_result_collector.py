"""대표주관사 공식 수요예측 결과의 기관 통합값만 보조 수집한다.

개인 일반청약 경쟁률, 증권사별 경쟁률, 비례배정 경쟁률은 이 모듈의 대상이
아니다. DART 최종 발행조건 문서에 통합 결과가 없을 때만 사용하는 보조 원천이며,
기관 경쟁률과 기간별 확약이 같은 공식 문서에서 함께 검증될 때만 반환한다.
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
LOCKUP_FIELDS = (
    "lockup_6m_ratio", "lockup_3m_ratio", "lockup_1m_ratio", "lockup_15d_ratio",
)
INSTITUTIONAL_RESULT_COLUMNS = [
    "notice_id", "notice_version_id", "source_version", "revision_of_notice_id", "is_correction",
    "event_id", "corp_name", "lead_underwriter", "notice_underwriter", "notice_title", "notice_url",
    "source_host", "source_type", "published_at", "available_at", "collected_at",
    "source_document_sha256", "source_offering_price", "event_listing_date", "event_offering_price",
    "event_context_validation_status", "aggregate_scope_verification", "institutional_demand_ratio",
    *LOCKUP_FIELDS, "lockup_none_ratio", "institutional_evidence", "lockup_evidence",
    "parse_evidence", "validation_status", "missing_reason", "human_review_required",
]


@dataclass(frozen=True)
class OfficialInstitutionalResultSource:
    event_id: str
    corp_name: str
    lead_underwriter: str
    notice_url: str
    source_type: str = "official_institutional_result"
    published_at: str | None = None
    source_version: str = "initial"
    revision_of_notice_id: str | None = None
    is_correction: bool = False
    notice_title: str | None = None
    source_offering_price: str | None = None
    notice_underwriter: str | None = None
    aggregate_scope_verification: str | None = None


class OfficialInstitutionalResultCollector:
    """대표주관사의 공개 통합 수요예측 결과 문서를 엄격하게 읽는다."""

    def __init__(self, session: requests.Session | None = None):
        self.session = session or requests.Session()

    @staticmethod
    def notice_id(event_id: str, notice_url: str, source_version: str = "initial") -> str:
        return sha256(f"{event_id}|{notice_url}|{source_version}".encode("utf-8")).hexdigest()

    @staticmethod
    def _clean_name(value: object) -> str:
        return re.sub(r"[^0-9A-Z가-힣]", "", str(value or "").upper().replace("주식회사", ""))

    @staticmethod
    def _as_timestamp(value: object) -> pd.Timestamp | None:
        timestamp = pd.to_datetime(value, errors="coerce")
        return None if pd.isna(timestamp) else timestamp

    @staticmethod
    def _as_number(value: object) -> float | None:
        number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return None if pd.isna(number) else float(number)

    @staticmethod
    def _as_blocks(content: bytes, content_type: str) -> tuple[list[str] | None, str | None]:
        content_type = content_type.lower()
        if "pdf" in content_type or content.startswith(b"%PDF"):
            try:
                from pypdf import PdfReader
                text = "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(content)).pages)
            except ImportError:
                return None, "pdf_parser_not_installed"
            except Exception:
                return None, "pdf_text_extract_failed"
            return [line.strip() for line in text.splitlines() if line.strip()], None
        if "html" not in content_type:
            return None, "source_document_type_not_supported"
        charset = re.search(r"charset\s*=\s*([A-Za-z0-9_-]+)", content_type, re.IGNORECASE)
        encoding = charset.group(1) if charset else "utf-8"
        try:
            html = content.decode(encoding, errors="replace")
        except LookupError:
            html = content.decode("utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")
        blocks = [element.get_text(" ", strip=True) for element in soup.select("tr, p, li")]
        blocks.extend(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
        return list(dict.fromkeys(block for block in blocks if block)), None

    @staticmethod
    def _direct_ratio(blocks: list[str]) -> tuple[float | None, str | None]:
        labels = ("기관투자자 수요예측 경쟁률", "기관 수요예측 경쟁률", "기관투자자 경쟁률")
        excluded = re.compile(r"일반\s*청약|개인\s*청약|비례\s*배정", re.IGNORECASE)
        for block in blocks:
            normalized = re.sub(r"\s+", " ", block).strip()
            if excluded.search(normalized):
                continue
            for label in labels:
                match = re.search(
                    rf"{label}\s*(?:은|는|:|：)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(?::|：|대)\s*1\b",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if match:
                    return float(match.group(1).replace(",", "")), match.group(0)[:300]
        return None, None

    @staticmethod
    def _lockup_bundle(blocks: list[str]) -> tuple[dict[str, float | None], str | None]:
        values: dict[str, float | None] = {
            "lockup_6m_ratio": None,
            "lockup_3m_ratio": None,
            "lockup_1m_ratio": None,
            "lockup_15d_ratio": None,
            "lockup_none_ratio": None,
        }
        if not any(re.search(r"의무\s*보유\s*확약|확약\s*비율", block, re.IGNORECASE) for block in blocks):
            return values, None
        periods = (
            (r"6\s*개월", "lockup_6m_ratio"),
            (r"3\s*개월", "lockup_3m_ratio"),
            (r"1\s*개월", "lockup_1m_ratio"),
            (r"15\s*일", "lockup_15d_ratio"),
            (r"확약\s*없음|미확약", "lockup_none_ratio"),
        )
        evidence: list[str] = []
        for block in blocks:
            normalized = re.sub(r"\s+", " ", block).strip()
            for label, field in periods:
                if values[field] is not None:
                    continue
                match = re.search(
                    rf"(?:{label})[^%]{{0,80}}?(?P<ratio>[0-9][0-9,]*(?:\.[0-9]+)?)\s*%",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if match:
                    values[field] = float(match.group("ratio").replace(",", "")) / 100
                    evidence.append(normalized[:300])
        return values, " | ".join(dict.fromkeys(evidence))[:1000] or None

    def _validate_event_context(
        self, source: OfficialInstitutionalResultSource, event: dict[str, Any] | None
    ) -> tuple[str, dict[str, object]]:
        details: dict[str, object] = {"event_listing_date": None, "event_offering_price": None}
        if event is None:
            return "not_checked_no_event_context", details
        if self._clean_name(source.corp_name) != self._clean_name(event.get("corp_name")):
            return "needs_review_corp_name_mismatch", details
        expected_underwriter = normalize_underwriter(event.get("lead_underwriter"))
        notice_underwriter = normalize_underwriter(source.notice_underwriter or source.lead_underwriter)
        if expected_underwriter and notice_underwriter != expected_underwriter:
            return "needs_review_not_lead_underwriter", details
        listing_date = self._as_timestamp(event.get("listing_date"))
        published_at = self._as_timestamp(source.published_at)
        source_price = self._as_number(source.source_offering_price)
        event_price = self._as_number(event.get("offering_price"))
        details.update(event_listing_date=listing_date, event_offering_price=event_price)
        if listing_date is None or published_at is None:
            return "needs_review_missing_notice_published_at", details
        if published_at >= listing_date:
            return "needs_review_notice_not_pre_listing", details
        if event_price is not None and source_price != event_price:
            return "needs_review_offering_price_mismatch", details
        return "verified_event_context", details

    def collect_notice(
        self, source: OfficialInstitutionalResultSource, event_context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        host = (urlparse(source.notice_url).hostname or "").lower()
        record: dict[str, Any] = {
            "notice_id": self.notice_id(source.event_id, source.notice_url, source.source_version),
            "notice_version_id": None, "source_version": source.source_version,
            "revision_of_notice_id": source.revision_of_notice_id, "is_correction": source.is_correction,
            "event_id": source.event_id, "corp_name": source.corp_name,
            "lead_underwriter": source.lead_underwriter,
            "notice_underwriter": source.notice_underwriter or source.lead_underwriter,
            "notice_title": source.notice_title, "notice_url": source.notice_url, "source_host": host,
            "source_type": source.source_type, "published_at": source.published_at,
            "available_at": source.published_at, "collected_at": pd.Timestamp.now(tz="Asia/Seoul"),
            "source_document_sha256": None, "source_offering_price": source.source_offering_price,
            "event_listing_date": None, "event_offering_price": None,
            "event_context_validation_status": "not_checked_no_event_context",
            "aggregate_scope_verification": source.aggregate_scope_verification,
            "institutional_demand_ratio": None, "lockup_6m_ratio": None, "lockup_3m_ratio": None,
            "lockup_1m_ratio": None, "lockup_15d_ratio": None, "lockup_none_ratio": None,
            "institutional_evidence": None, "lockup_evidence": None, "parse_evidence": None,
            "validation_status": "needs_review", "missing_reason": None, "human_review_required": True,
        }
        context_status, context_details = self._validate_event_context(source, event_context)
        record.update(context_details, event_context_validation_status=context_status)
        if host not in OFFICIAL_UNDERWRITER_HOSTS:
            record.update(validation_status="rejected_non_official_underwriter_host")
            return record
        if urlparse(source.notice_url).scheme != "https":
            record.update(validation_status="rejected_non_https_source")
            return record
        try:
            response = self.session.get(source.notice_url, headers={"Accept": "text/html,application/pdf"}, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            record.update(validation_status="source_access_failed", missing_reason=f"{type(exc).__name__}_retry_required")
            return record
        record["source_document_sha256"] = sha256(response.content).hexdigest()
        record["notice_version_id"] = sha256(
            f"{record['notice_id']}|{record['source_document_sha256']}".encode("utf-8")
        ).hexdigest()
        blocks, error = self._as_blocks(response.content, response.headers.get("Content-Type", ""))
        if blocks is None:
            record.update(validation_status="source_document_type_not_supported", missing_reason=error)
            return record
        ratio, ratio_evidence = self._direct_ratio(blocks)
        lockup, lockup_evidence = self._lockup_bundle(blocks)
        record.update(
            institutional_demand_ratio=ratio,
            institutional_evidence=ratio_evidence,
            lockup_evidence=lockup_evidence,
            parse_evidence=" | ".join(value for value in (ratio_evidence, lockup_evidence) if value)[:1200] or None,
            **lockup,
        )
        complete_bundle = ratio is not None and all(lockup[field] is not None for field in LOCKUP_FIELDS)
        scope_verified = source.aggregate_scope_verification == "manual_verified_aggregate_institutional"
        if complete_bundle and context_status == "verified_event_context" and scope_verified:
            record.update(validation_status="verified_official_underwriter_aggregate_bundle", human_review_required=False)
        elif complete_bundle:
            record.update(validation_status="official_notice_aggregate_bundle_review_required")
        elif ratio is not None or any(lockup[field] is not None for field in LOCKUP_FIELDS):
            record.update(validation_status="official_notice_incomplete_institutional_bundle")
        else:
            record.update(validation_status="official_notice_no_supported_institutional_bundle")
        return record

    @staticmethod
    def _value(row: object, field: str, default: object = None) -> object:
        value = getattr(row, field, default)
        return default if pd.isna(value) or str(value).strip() == "" else value

    def collect_sources(
        self, sources: pd.DataFrame, event_contexts: dict[str, dict[str, Any]] | None = None
    ) -> pd.DataFrame:
        required = {"event_id", "corp_name", "lead_underwriter", "notice_url"}
        missing = required - set(sources.columns)
        if missing:
            raise ValueError(f"기관 수요예측 공식 공지 원장에 필요한 열이 없습니다: {', '.join(sorted(missing))}")
        records = []
        for row in sources.itertuples(index=False):
            event_id = str(self._value(row, "event_id", ""))
            records.append(self.collect_notice(OfficialInstitutionalResultSource(
                event_id=event_id,
                corp_name=str(self._value(row, "corp_name", "")),
                lead_underwriter=str(self._value(row, "lead_underwriter", "")),
                notice_url=str(self._value(row, "notice_url", "")),
                source_type=str(self._value(row, "source_type", "official_institutional_result")),
                published_at=self._value(row, "published_at"),
                source_version=str(self._value(row, "source_version", "initial")),
                revision_of_notice_id=self._value(row, "revision_of_notice_id"),
                is_correction=str(self._value(row, "is_correction", "false")).lower() == "true",
                notice_title=self._value(row, "notice_title"),
                source_offering_price=self._value(row, "source_offering_price"),
                notice_underwriter=self._value(row, "notice_underwriter"),
                aggregate_scope_verification=self._value(row, "aggregate_scope_verification"),
            ), (event_contexts or {}).get(event_id)))
        return pd.DataFrame(records, columns=INSTITUTIONAL_RESULT_COLUMNS)
