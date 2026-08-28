"""공식 주관사 원천의 허용 범위와 우선순위를 관리한다."""

from __future__ import annotations

import re

import pandas as pd


OFFICIAL_UNDERWRITER_REGISTRY = {
    "한국투자증권": {
        "aliases": ("한국투자증권", "한국투자"),
        "hosts": ("securities.koreainvestment.com",),
        "public_discovery_url": "https://securities.koreainvestment.com/pro_help/7327.html",
        "document_formats": "public_html",
        "automatic_url_discovery": False,
        "collection_policy": "manual_url_only_current_or_authenticated_ratio_screen",
    },
    "미래에셋증권": {
        "aliases": ("미래에셋증권", "미래에셋"),
        "hosts": ("securities.miraeasset.com",),
        "public_discovery_url": "https://securities.miraeasset.com/public/mw/guide/html/notice01.html",
        "document_formats": "public_html_or_pdf",
        "automatic_url_discovery": False,
        "collection_policy": "manual_url_only_current_or_authenticated_ratio_screen",
    },
    "NH투자증권": {
        "aliases": ("NH투자증권", "엔에이치투자증권", "NH"),
        "hosts": ("www.nhqv.com", "securities.nhqv.com"),
        "public_discovery_url": "https://www.nhqv.com/",
        "document_formats": "public_html_or_pdf",
        "automatic_url_discovery": False,
        "collection_policy": "manual_url_only_public_result_route_not_yet_verified",
    },
    "KB증권": {
        "aliases": ("KB증권", "KB"),
        "hosts": ("www.kbsec.com", "fdata.kbsec.com"),
        "public_discovery_url": "https://www.kbsec.com/go.able?linkcd=m02070000",
        "document_formats": "public_html_or_pdf",
        "automatic_url_discovery": False,
        "collection_policy": "manual_url_only_public_notice_pdf_scope_review_required",
    },
}


def normalize_underwriter(value: object) -> str | None:
    """KRX 표기의 법인 접미사를 제거해 등록된 공식 주관사명으로 통일한다."""
    text = re.sub(r"[\s㈜()]", "", str(value or "")).upper()
    text = text.replace("주식회사", "")
    if not text or text == "해당없음":
        return None
    for canonical, config in OFFICIAL_UNDERWRITER_REGISTRY.items():
        aliases = (canonical, *config["aliases"])
        for alias in aliases:
            normalized_alias = re.sub(r"[\s㈜()]", "", alias).upper().replace("주식회사", "")
            if normalized_alias and normalized_alias in text:
                return canonical
    return None


def official_hosts() -> dict[str, str]:
    """허용된 공개 도메인과 해당 주관사의 역색인을 반환한다."""
    return {
        host: underwriter
        for underwriter, config in OFFICIAL_UNDERWRITER_REGISTRY.items()
        for host in config["hosts"]
    }


def build_underwriter_priorities(events: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """일반 IPO 이벤트 비중으로 공식 수집 확대 순서를 만든다."""
    columns = [
        "priority", "lead_underwriter", "general_ipo_event_count", "coverage_ratio",
        "public_discovery_url", "document_formats", "automatic_url_discovery", "collection_policy", "supported",
    ]
    if events.empty or "lead_underwriter" not in events.columns:
        return pd.DataFrame(columns=columns)
    candidates = events.copy()
    if "event_class" in candidates.columns:
        candidates = candidates[candidates["event_class"].eq("general_ipo")]
    total_general_ipo_events = len(candidates)
    candidates["lead_underwriter"] = candidates["lead_underwriter"].map(normalize_underwriter)
    candidates = candidates.dropna(subset=["lead_underwriter"])
    counts = candidates["lead_underwriter"].value_counts().rename_axis("lead_underwriter").reset_index(name="general_ipo_event_count")
    records: list[dict[str, object]] = []
    for row in counts.itertuples(index=False):
        config = OFFICIAL_UNDERWRITER_REGISTRY.get(row.lead_underwriter)
        if config is None:
            continue
        records.append({
            "lead_underwriter": row.lead_underwriter,
            "general_ipo_event_count": int(row.general_ipo_event_count),
            "coverage_ratio": round(
                int(row.general_ipo_event_count) / total_general_ipo_events, 6
            ) if total_general_ipo_events else 0.0,
            "public_discovery_url": config["public_discovery_url"],
            "document_formats": config["document_formats"],
            "automatic_url_discovery": bool(config["automatic_url_discovery"]),
            "collection_policy": config["collection_policy"],
            "supported": True,
        })
    result = pd.DataFrame(records, columns=columns[1:])
    if result.empty:
        return pd.DataFrame(columns=columns)
    result = result.sort_values("general_ipo_event_count", ascending=False).head(top_n).reset_index(drop=True)
    result.insert(0, "priority", range(1, len(result) + 1))
    return result
