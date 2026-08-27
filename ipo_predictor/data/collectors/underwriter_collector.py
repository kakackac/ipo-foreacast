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


OFFICIAL_UNDERWRITER_HOSTS = {
    "securities.koreainvestment.com": "한국투자증권",
    "www.kbsec.com": "KB증권",
    "fdata.kbsec.com": "KB증권",
    "securities.miraeasset.com": "미래에셋증권",
    "www.nhqv.com": "NH투자증권",
    "securities.nhqv.com": "NH투자증권",
}

RESULT_COLUMNS = [
    "notice_id", "event_id", "corp_name", "lead_underwriter", "notice_url",
    "source_host", "source_type", "published_at", "collected_at",
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


class OfficialUnderwriterCollector:
    """공개된 주관사 공지 URL만 읽어 청약 결과를 추출하는 수집기."""

    def __init__(self, session: requests.Session | None = None):
        self.session = session or requests.Session()

    @staticmethod
    def notice_id(event_id: str, notice_url: str) -> str:
        return sha256(f"{event_id}|{notice_url}".encode("utf-8")).hexdigest()

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

    def collect_notice(self, source: OfficialNoticeSource) -> dict[str, Any]:
        """한 공식 공지를 추출하고, 원문 근거와 승인 상태를 함께 반환한다."""
        host = (urlparse(source.notice_url).hostname or "").lower()
        record: dict[str, Any] = {
            "notice_id": self.notice_id(source.event_id, source.notice_url),
            "event_id": source.event_id,
            "corp_name": source.corp_name,
            "lead_underwriter": source.lead_underwriter,
            "notice_url": source.notice_url,
            "source_host": host,
            "source_type": source.source_type,
            "published_at": source.published_at,
            "collected_at": pd.Timestamp.now(tz="Asia/Seoul"),
            "retail_subscription_ratio": None,
            "retail_ratio_scope": None,
            "institutional_demand_ratio": None,
            "lockup_ratio": None,
            "parse_evidence": None,
            "validation_status": "needs_review",
            "missing_reason": None,
            "human_review_required": True,
        }
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
                validation_status="source_request_retry_required",
                missing_reason=type(exc).__name__,
            )
            return record

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
        if retail is not None and scope == "integrated":
            record.update(validation_status="official_notice_integrated_retail_ratio", human_review_required=False)
        elif any(value is not None for value in (retail, institutional, lockup)):
            record.update(validation_status="official_notice_value_requires_scope_review")
        else:
            record.update(
                validation_status="official_notice_no_supported_value",
                missing_reason="supported_labels_not_found",
            )
        return record

    def collect_sources(self, sources: pd.DataFrame) -> pd.DataFrame:
        required = {"event_id", "corp_name", "lead_underwriter", "notice_url"}
        missing = required - set(sources.columns)
        if missing:
            raise ValueError(f"주관사 공식 공지 입력 파일에 필요한 열이 없습니다: {', '.join(sorted(missing))}")
        records = []
        for row in sources.itertuples(index=False):
            records.append(self.collect_notice(OfficialNoticeSource(
                event_id=str(getattr(row, "event_id")),
                corp_name=str(getattr(row, "corp_name")),
                lead_underwriter=str(getattr(row, "lead_underwriter")),
                notice_url=str(getattr(row, "notice_url")),
                source_type=str(getattr(row, "source_type", "public_notice")),
                published_at=getattr(row, "published_at", None),
            )))
        return pd.DataFrame(records, columns=RESULT_COLUMNS)
