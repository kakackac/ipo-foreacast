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
    "notice_underwriter", "retail_participating_brokers", "scope_verification_status",
    "retail_subscription_ratio", "retail_ratio_scope", "retail_subscribed_shares",
    "retail_allocation_shares", "retail_raw_values_evidence", "institutional_demand_ratio",
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
    notice_underwriter: str | None = None
    retail_participating_brokers: str | None = None
    scope_verification_status: str | None = None


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
    def _direct_share(blocks: list[str], labels: tuple[str, ...]) -> tuple[int | None, str | None]:
        """같은 표 행 또는 문단에서 일반청약 주식 수를 읽는다.

        단위가 ``주``로 명시된 값만 허용한다. 금액·비례경쟁률·배정 건수는
        전체 경쟁률 재구성의 분자/분모로 사용할 수 없으므로 여기서 제외한다.
        """
        for block in blocks:
            normalized = re.sub(r"\s+", " ", block).strip()
            for label in labels:
                match = re.search(
                    rf"{label}\s*(?:은|는|:|：)?\s*([0-9][0-9,]*)\s*주\b",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if match:
                    return int(match.group(1).replace(",", "")), match.group(0)[:240]
        return None, None

    @staticmethod
    def _normalize_broker_set(value: object) -> tuple[str, ...]:
        """사람이 공식 원문에서 확인한 참여 증권사 목록을 정규화한다."""
        items = re.split(r"[,/|;·]", str(value or ""))
        brokers = [normalize_underwriter(item) for item in items]
        return tuple(sorted({broker for broker in brokers if broker}))

    @staticmethod
    def _retail_ratio_scope(blocks: list[str], evidence: str | None) -> str | None:
        """전체 참여 증권사 범위가 원문에서 명시된 경우만 통합값으로 인정한다.

        ``통합경쟁률``은 한 주관사 내부의 균등·비례 청약을 합친 의미로도
        쓰일 수 있다. 따라서 단어 자체가 아니라 같은 행/문단의 전체 참여사
        범위 문구까지 확인해야 모델 입력으로 안전하다.
        """
        if not evidence:
            return None
        all_participant_patterns = (
            r"(?:전체|모든)\s*(?:참여\s*)?(?:증권사|주관사|청약\s*취급처)",
            r"(?:전체|총)\s*일반\s*(?:공모\s*)?청약",
            r"일반\s*(?:공모\s*)?청약\s*(?:전체|총)",
            r"통합\s*일반\s*(?:공모\s*)?청약",
            r"(?:공동\s*주관사|대표\s*주관사).{0,40}(?:합산|합계|전체)",
            r"(?:합산|합계).{0,40}(?:전체|모든)\s*(?:증권사|주관사|청약\s*취급처)",
        )
        normalized_evidence = re.sub(r"\s+", " ", evidence).strip()
        for block in blocks:
            normalized_block = re.sub(r"\s+", " ", block).strip()
            if normalized_evidence not in normalized_block:
                continue
            if any(re.search(pattern, normalized_block, flags=re.IGNORECASE) for pattern in all_participant_patterns):
                return "integrated_all_participants"
        return None

    @staticmethod
    def _is_sole_retail_intake_scope(blocks: list[str], evidence: str | None) -> bool:
        """단독 주관·단일 일반청약 접수처가 원문에서 직접 확인되는지 검사한다."""
        if not evidence:
            return False
        sole_patterns = (
            r"단독\s*주관", r"단일\s*(?:일반\s*)?청약\s*접수처",
            r"유일한\s*(?:일반\s*)?청약\s*접수처",
        )
        normalized_evidence = re.sub(r"\s+", " ", evidence).strip()
        return any(
            normalized_evidence in re.sub(r"\s+", " ", block).strip()
            and any(re.search(pattern, block, flags=re.IGNORECASE) for pattern in sole_patterns)
            for block in blocks
        )

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
        source_underwriter = normalize_underwriter(source.notice_underwriter or source.lead_underwriter)
        listing_date = self._as_timestamp(event_context.get("listing_date"))
        event_price = self._as_number(event_context.get("offering_price"))
        published_at = self._as_timestamp(source.published_at)
        subscription_end = self._as_timestamp(source.subscription_end)
        source_price = self._as_number(source.source_offering_price)
        details["event_listing_date"] = listing_date
        details["event_offering_price"] = event_price
        if not expected_name or source_name != expected_name:
            return "needs_review_corp_name_mismatch", details
        participating_brokers = self._normalize_broker_set(source.retail_participating_brokers)
        if expected_underwriter and source_underwriter != expected_underwriter:
            # 공동주관사 원시 수치 재구성에는 대표주관사 외 공식 문서도 필요하다.
            # 다만 사람이 공식 원문으로 확인한 참여사 목록에 게시 주관사가 없으면
            # 이벤트 연결을 승인하지 않는다.
            if source_underwriter not in participating_brokers:
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
            "notice_underwriter": source.notice_underwriter or source.lead_underwriter,
            "retail_participating_brokers": source.retail_participating_brokers,
            "scope_verification_status": source.scope_verification_status,
            "retail_subscription_ratio": None,
            "retail_ratio_scope": None,
            "retail_subscribed_shares": None,
            "retail_allocation_shares": None,
            "retail_raw_values_evidence": None,
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
            blocks, (
                "공동주관사 전체 일반청약 경쟁률", "전체 일반청약 경쟁률",
                "일반청약 전체 경쟁률", "통합 일반청약 경쟁률", "통합경쟁률",
                "통합 경쟁률", "일반청약 경쟁률", "일반 경쟁률", "비례배정 경쟁률",
            )
        )
        subscribed_shares, subscribed_evidence = self._direct_share(
            blocks, (
                r"일반\s*(?:공모\s*)?청약\s*(?:신청\s*)?주식\s*수",
                r"일반\s*(?:공모\s*)?청약\s*청약\s*주식\s*수",
            )
        )
        allocation_shares, allocation_evidence = self._direct_share(
            blocks, (
                r"일반\s*(?:공모\s*)?청약\s*(?:배정\s*)?(?:주식\s*)?(?:수|물량)",
                r"일반\s*(?:공모\s*)?청약\s*배정\s*주식\s*수",
            )
        )
        institutional, institutional_evidence = self._direct_ratio(
            blocks, ("기관투자자 경쟁률", "기관 수요예측 경쟁률", "수요예측 경쟁률")
        )
        lockup, lockup_evidence = self._direct_percent(
            blocks, ("의무보유확약 비율", "의무보유 확약 비율", "확약비율")
        )
        evidence = retail_evidence or subscribed_evidence or allocation_evidence or institutional_evidence or lockup_evidence
        scope = self._retail_ratio_scope(blocks, retail_evidence)
        sole_scope = self._is_sole_retail_intake_scope(blocks, retail_evidence)
        if retail is not None and scope is None and sole_scope:
            scope = "sole_retail_intake_broker"
        if retail is not None and scope is None:
            scope = "underwriter_only_or_unknown"
        record.update(
            retail_subscription_ratio=retail,
            retail_ratio_scope=scope,
            retail_subscribed_shares=subscribed_shares,
            retail_allocation_shares=allocation_shares,
            retail_raw_values_evidence=" | ".join(
                evidence for evidence in (subscribed_evidence, allocation_evidence) if evidence
            ) or None,
            institutional_demand_ratio=institutional,
            lockup_ratio=lockup,
            parse_evidence=evidence,
        )
        context_is_valid = context_status in {"verified_event_context", "not_checked_no_event_context"}
        if retail is not None and scope == "integrated_all_participants" and context_is_valid:
            record.update(validation_status="official_notice_integrated_retail_ratio", human_review_required=False)
        elif retail is not None and scope == "sole_retail_intake_broker" and context_is_valid:
            record.update(validation_status="official_notice_single_retail_intake_ratio", human_review_required=False)
        elif retail is not None and scope == "integrated_all_participants":
            record.update(validation_status="official_notice_event_link_review_required")
        elif retail is not None and scope == "sole_retail_intake_broker":
            record.update(validation_status="official_notice_event_link_review_required")
        elif subscribed_shares is not None and allocation_shares is not None:
            record.update(
                validation_status="official_notice_raw_retail_components_collected",
                missing_reason="requires_all_participants_reconstruction",
            )
        elif any(value is not None for value in (retail, institutional, lockup)):
            record.update(validation_status="official_notice_value_requires_scope_review")
        else:
            record.update(
                validation_status="official_notice_no_supported_value",
                missing_reason="supported_labels_not_found",
            )
        return record

    def resolve_reconstructed_retail_ratios(self, records: pd.DataFrame) -> pd.DataFrame:
        """모든 참여 증권사의 원시 수치가 있을 때만 전체 경쟁률을 재구성한다.

        이 메서드는 증권사별 경쟁률을 평균내지 않는다. 각 공식 문서에서 파싱한
        ``일반청약 청약주식수``와 ``일반청약 배정주식수/물량``을 합산해
        ``합계 청약주식수 / 합계 배정주식수``로 계산한다. 참여사 목록은 공식
        원문을 확인한 사람이 URL 원장에만 기록하며, 숫자는 원문 파서 결과만
        사용한다.
        """
        if records.empty:
            return records.reindex(columns=RESULT_COLUMNS)
        result = records.copy().reindex(columns=RESULT_COLUMNS)
        direct_approved = {
            "official_notice_integrated_retail_ratio",
            "official_notice_single_retail_intake_ratio",
        }
        generated: list[dict[str, Any]] = []
        for event_id, group in result.groupby("event_id", dropna=False):
            if group["validation_status"].isin(direct_approved).any():
                continue
            components = group[
                group["validation_status"].eq("official_notice_raw_retail_components_collected")
                & group["event_context_validation_status"].eq("verified_event_context")
            ].copy()
            if components.empty:
                continue
            scope_statuses = set(components["scope_verification_status"].fillna(""))
            if scope_statuses != {"manual_verified_official_source"}:
                continue
            broker_sets = {
                self._normalize_broker_set(value)
                for value in components["retail_participating_brokers"]
            }
            if len(broker_sets) != 1:
                continue
            expected_brokers = next(iter(broker_sets))
            actual_brokers = components["notice_underwriter"].map(normalize_underwriter)
            if (
                not expected_brokers
                or actual_brokers.isna().any()
                or set(actual_brokers) != set(expected_brokers)
                or actual_brokers.duplicated().any()
            ):
                continue
            subscribed = pd.to_numeric(components["retail_subscribed_shares"], errors="coerce")
            allocation = pd.to_numeric(components["retail_allocation_shares"], errors="coerce")
            if subscribed.isna().any() or allocation.isna().any() or (allocation <= 0).any():
                continue
            total_allocation = float(allocation.sum())
            if total_allocation <= 0:
                continue
            ratio = float(subscribed.sum() / total_allocation)
            representative = components.sort_values("collected_at").iloc[-1].to_dict()
            source_urls = " | ".join(sorted(components["notice_url"].dropna().astype(str).unique()))
            raw_evidence = " | ".join(
                evidence for evidence in components["retail_raw_values_evidence"].dropna().astype(str)
                if evidence
            )[:1000]
            component_ids = "|".join(sorted(components["notice_id"].dropna().astype(str)))
            generated.append({
                **representative,
                "notice_id": self.notice_id(str(event_id), component_ids, "reconstructed_all_participants"),
                "notice_version_id": sha256(component_ids.encode("utf-8")).hexdigest(),
                "source_version": "reconstructed_all_participants",
                "notice_url": source_urls,
                "source_type": "official_notice_reconstructed",
                "source_document_sha256": sha256(component_ids.encode("utf-8")).hexdigest(),
                "retail_subscription_ratio": ratio,
                "retail_ratio_scope": "reconstructed_all_participants",
                "retail_subscribed_shares": int(subscribed.sum()),
                "retail_allocation_shares": int(total_allocation),
                "retail_raw_values_evidence": raw_evidence or None,
                "parse_evidence": "all_participant_raw_shares_reconstruction",
                "validation_status": "official_notice_reconstructed_retail_ratio",
                "missing_reason": None,
                "human_review_required": False,
            })
        if generated:
            # pandas가 전부 결측인 감사 열을 병합하면서 내는 dtype 경고를 피하고,
            # 원시 문서 행과 계산 행을 같은 감사 스키마로 명시적으로 재구성한다.
            result = pd.DataFrame(
                [*result.to_dict("records"), *generated], columns=RESULT_COLUMNS
            )
        return result.reindex(columns=RESULT_COLUMNS)

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
                notice_underwriter=self._source_value(row, "notice_underwriter"),
                retail_participating_brokers=self._source_value(row, "retail_participating_brokers"),
                scope_verification_status=self._source_value(row, "scope_verification_status"),
            ), (event_contexts or {}).get(str(self._source_value(row, "event_id", "")))))
        return pd.DataFrame(records, columns=RESULT_COLUMNS)
