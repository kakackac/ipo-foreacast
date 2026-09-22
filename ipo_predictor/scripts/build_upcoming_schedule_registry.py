"""Build a source-checked offering schedule registry, separate from model datasets."""
import argparse
import hashlib
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_dart_subscription_calendar import replay
from scripts.audit_upcoming_disclosures import parse_viewer, summary_node


def normalize(value):
    return re.sub(r"\s+", "", value)


def parse_date(value):
    match = re.fullmatch(r"\s*(\d{4})(?:년\s*|[.\-/])(\d{1,2})(?:월\s*|[.\-/])(\d{1,2})일?\s*", value)
    if not match:
        raise ValueError("Unknown date notation")
    return date(*map(int, match.groups())).isoformat()


def labeled_row(soup, labels):
    matches = []
    for table_index, table in enumerate(soup.find_all("table")):
        rows = [r for r in table.find_all("tr") if r.find_parent("table") is table]
        for index, row in enumerate(rows):
            cells = row.find_all(["th", "td"], recursive=False)
            headers = [normalize(c.get_text(" ", strip=True)) for c in cells]
            if headers != labels:
                continue
            if index + 1 >= len(rows):
                raise ValueError("Table data row missing")
            values = rows[index + 1].find_all(["th", "td"], recursive=False)
            if len(values) != len(labels) or any(c.get("colspan", "1") != "1" or c.get("rowspan", "1") != "1" for c in [*cells, *values]):
                raise ValueError("Ambiguous table spans")
            matches.append({"table_index": table_index, "row_index": index + 1,
                            "headers": headers, "raw_values": [c.get_text(" ", strip=True) for c in values]})
    if len(matches) != 1:
        raise ValueError("Table absent or ambiguous")
    return matches[0]


def parse_summary(content):
    soup = BeautifulSoup(content, "html.parser")
    schedule = labeled_row(soup, ["청약기일", "납입기일", "청약공고일", "배정공고일", "배정기준일"])
    offering = labeled_row(soup, ["증권의종류", "증권수량", "액면가액", "모집(매출)가액", "모집(매출)총액", "모집(매출)방법"])
    dates = re.split(r"[~～]", schedule["raw_values"][0])
    if len(dates) != 2:
        raise ValueError("Subscription interval missing")
    start, end = map(parse_date, dates)
    if start > end:
        raise ValueError("Reversed subscription interval")
    method, security = offering["raw_values"][-1], offering["raw_values"][0]
    if normalize(method) != "일반공모" or normalize(security) != "보통주":
        raise ValueError("Not a public common-stock offering")
    return {"subscription_start": start, "subscription_end": end,
            "payment_date": parse_date(schedule["raw_values"][1]), "security_type": security,
            "offering_method": method, "schedule_evidence": schedule, "offering_table_evidence": offering,
            "offering_table_amount_raw": offering["raw_values"][3],
            "final_offering_price": None, "final_price_status": "not_verified_from_summary_table",
            "listing_date": None, "listing_date_status": "not_verified_by_this_registry"}


def build(calendar_path, disclosure_path, reference):
    calendar = replay(calendar_path, reference)
    events = calendar["events"]
    source = Path(disclosure_path)
    report = json.loads((source / "report.json").read_text())
    rows, rejected, seen = [], [], set()
    for entry in report["rows"]:
        try:
            corp, receipt = entry["corp_code"], entry["rcept_no"]
            key = (corp, receipt)
            if key in seen:
                raise ValueError("Duplicate registry identity")
            seen.add(key)
            viewer = (source / f"{receipt}_viewer.html").read_bytes()
            summary = (source / f"{receipt}_summary.html").read_bytes()
            for content, field in ((viewer, "viewer_sha256"), (summary, "summary_sha256")):
                if hashlib.sha256(content).hexdigest() != entry[field]:
                    raise ValueError("Source checksum mismatch")
            as_of = datetime.fromisoformat(entry["collected_at"]).date().isoformat()
            lineage = parse_viewer(viewer, receipt, corp, as_of)
            if lineage["status"] != "calendar_receipt_latest_in_viewer_family" or summary_node(viewer, receipt) != entry["summary_request"]:
                raise ValueError("Source lineage changed")
            parsed = parse_summary(summary)
            official_events = [e for e in events if e["corp_code"] == corp and e["rcept_no"] == receipt]
            starts = {e["subscription_date"] for e in official_events if e["event_type"] == "start"}
            ends = {e["subscription_date"] for e in official_events if e["event_type"] == "end"}
            if starts != {parsed["subscription_start"]} or ends != {parsed["subscription_end"]}:
                raise ValueError("Calendar and registration schedule disagree")
            rows.append({"candidate_id": f"dart_offering:{corp}:{receipt}", "corp_code": corp,
                "corp_name": official_events[0]["corp_name"], "rcept_no": receipt, **parsed,
                "schedule_validation_status": "calendar_and_registration_schedule_agree",
                "offering_class": "public_common_stock_offering_ipo_subtype_pending",
                "source_url": f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={receipt}",
                "summary_sha256": entry["summary_sha256"], "source_collected_at": entry["collected_at"],
                "schedule_candidate_eligible": True, "model_eligible": False,
                "model_blockers": ["ipo_subtype_not_verified", "listing_date_not_verified", "final_price_not_verified", "stage_features_not_verified"]})
        except (ValueError, KeyError, OSError) as exc:
            rejected.append({"reference_name": entry.get("reference_name"), "status": "needs_review", "error_type": type(exc).__name__})
    return {"registry_version": 1, "created_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
            "scope": "twenty_reference_offerings_not_all_market", "rows": rows, "rejected": rejected,
            "historical_training_data_modified": False, "deployment_authorized": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calendar", type=Path, required=True)
    parser.add_argument("--disclosures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads((Path(__file__).resolve().parents[1] / "data/manual/screenshot_schedule_reference.json").read_text())
    result = build(args.calendar, args.disclosures, reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"schedule_verified": len(result["rows"]), "rejected": len(result["rejected"]), "model_approved": 0}))
    sys.exit(1 if result["rejected"] else 0)
