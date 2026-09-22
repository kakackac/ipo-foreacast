"""Public KIND offering candidates, never automatically approved model inputs."""
import hashlib
import json
import re
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

URL = "https://kind.krx.co.kr/listinvstg/pubofrprogcom.do"
HEADERS = ["시장구분", "회사명", "신고서제출일", "수요예측일정", "청약일정", "납입일",
           "확정공모가", "공모금액(백만원)", "상장예정일", "상장주선인"]
KST = ZoneInfo("Asia/Seoul")


def text(cell):
    return " ".join(cell.get_text(" ", strip=True).split())


def schedule_range(raw):
    if raw.strip() in ("", "-", "미정"):
        return None, None, "not_announced_in_source"
    match = re.fullmatch(r"(\d{4}-\d{2}-\d{2})\s*~\s*(\d{4}-\d{2}-\d{2})", raw)
    if match:
        try:
            start, end = (date.fromisoformat(v) for v in match.groups())
            if start <= end:
                return start.isoformat(), end.isoformat(), "parsed_official_schedule"
        except ValueError:
            pass
    return None, None, "schedule_review_required"


def classify_schedule(row, as_of):
    start, end, status = schedule_range(row["subscription_schedule"])
    row.update(subscription_start=start, subscription_end=end, subscription_schedule_status=status)
    demand_start, demand_end, demand_status = schedule_range(row["demand_schedule"])
    row.update(demand_start=demand_start, demand_end=demand_end, demand_schedule_status=demand_status)
    row["subscription_state"] = (
        "review_required" if status == "schedule_review_required" else
        "date_unconfirmed" if start is None else
        "subscription_upcoming" if date.fromisoformat(start) > as_of else
        "subscription_open" if date.fromisoformat(end) >= as_of else "subscription_period_elapsed")
    row["listing_state"] = (
        "date_unconfirmed" if row["listing_date"] is None else
        "listing_upcoming" if date.fromisoformat(row["listing_date"]) > as_of else
        "listing_scheduled_today" if date.fromisoformat(row["listing_date"]) == as_of else "listing_date_elapsed_unconfirmed")
    # Elapsed planned dates do not establish completed listing or cancellation.
    row["service_schedule_candidate"] = (row["subscription_state"] != "subscription_period_elapsed"
        or row["listing_state"] != "listing_date_elapsed_unconfirmed")
    row["subscription_d_day"] = None if start is None else (date.fromisoformat(start) - as_of).days
    row["schedule_as_of"] = as_of.isoformat()
    return row


def parse_candidates(export_html, listing_html, market, collected_at):
    export = BeautifulSoup(export_html, "html.parser")
    listing = BeautifulSoup(listing_html, "html.parser")
    table = export.find("table")
    if table is None or [text(c) for c in table.select("th")] != HEADERS:
        raise ValueError("KIND export schema changed")
    identities = {}
    for row in listing.select("tr[onclick]"):
        match = re.fullmatch(r"fnDetailView\('([0-9]{14})'\)", row["onclick"])
        cells = row.find_all("td", recursive=False)
        if not match or len(cells) != 9:
            raise ValueError("KIND identity schema changed")
        key = tuple(text(c) for c in cells)
        if key in identities:
            raise ValueError("Ambiguous KIND candidate")
        identities[key] = match.group(1)
    total = re.search(r"전체\s*([\d,]+)\s*건", listing.get_text(" ", strip=True))
    if total is None or int(total.group(1).replace(",", "")) != len(identities):
        raise ValueError("KIND pagination incomplete; narrow the submission-date range")
    results = []
    for row in table.select("tr"):
        cells = row.find_all("td", recursive=False)
        if not cells:
            continue
        if len(cells) == 1 and not identities:
            continue
        values = [text(c) for c in cells]
        if len(values) != 10 or tuple(values[1:]) not in identities:
            raise ValueError("KIND export/list identity mismatch")
        expected = "코스닥" if market == "KOSDAQ" else "유가증권"
        if expected not in values[0]:
            raise ValueError("KIND market mismatch")
        process_id = identities.pop(tuple(values[1:]))
        scheduled = None
        if values[8] not in ("", "-", "미정"):
            scheduled = date.fromisoformat(values[8]).isoformat()
        price = None
        if values[6] not in ("", "-"):
            if not re.fullmatch(r"[\d,]+", values[6]):
                raise ValueError("Unknown offering-price format")
            price = int(values[6].replace(",", ""))
        results.append({"candidate_id": f"kind_offering:{process_id}", "kind_process_id": process_id,
            "corp_name": values[1], "market": market, "submission_date": values[2],
            "demand_schedule": values[3], "subscription_schedule": values[4],
            "listing_date": scheduled, "offering_price_candidate": price,
            "lead_underwriter": values[9], "collected_at": collected_at,
            "source_url": URL + "?method=searchPubofrProgComMain",
            "detail_url": "https://kind.krx.co.kr/listinvstg/pubofrprogcomdetail.do?method=searchProgComDetailMain&bzProcsNo=" + process_id,
            "verification_status": "official_candidate_requires_event_and_dart_link",
            "model_eligible": False})
    if identities:
        raise ValueError("KIND export missing candidate rows")
    as_of = datetime.fromisoformat(collected_at).astimezone(KST).date()
    return [classify_schedule(row, as_of) for row in results]


class UpcomingIPOCollector:
    def __init__(self, session=None):
        self.session = session or requests.Session()

    def collect(self, start, end, output):
        start, end = date.fromisoformat(start), date.fromisoformat(end)
        now = datetime.now(KST)
        if start > end or end > now.date():
            raise ValueError("Invalid submission-date window")
        destination = Path(output) / now.strftime("%Y%m%dT%H%M%S%f")
        destination.mkdir(parents=True, exist_ok=False)
        manifest = {"started_at": now.isoformat(), "submission_start": str(start), "submission_end": str(end),
                    "status": "started", "requests": [], "model_eligible": False}
        results = []
        try:
            for market, code in (("KOSPI", "1"), ("KOSDAQ", "2")):
                documents = {}
                for form in ("sub", "down"):
                    request = {"market": market, "form": form, "status": "started"}
                    manifest["requests"].append(request)
                    response = self.session.post(URL, data={"method": "searchPubofrProgComSub",
                        "forward": f"pubofrprogcom_{form}", "marketType": code, "currentPageSize": "100",
                        "pageIndex": "1", "fromDate": str(start), "toDate": str(end)}, timeout=30)
                    response.raise_for_status()
                    content = response.content
                    (destination / f"{market}_{form}.html").write_bytes(content)
                    request.update(status="received", sha256=hashlib.sha256(content).hexdigest())
                    documents[form] = content
                rows = parse_candidates(documents["down"], documents["sub"], market, datetime.now(KST).isoformat())
                for row in rows:
                    row["schedule_state"] = ("date_unconfirmed" if row["listing_date"] is None else
                        "upcoming" if date.fromisoformat(row["listing_date"]) > now.date() else "listing_today_or_past")
                results.extend(rows)
            if len({r["candidate_id"] for r in results}) != len(results):
                raise ValueError("Duplicate process identity across markets")
            (destination / "candidates.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            manifest.update(status="complete", rows=len(results), upcoming=sum(r["schedule_state"] == "upcoming" for r in results),
                upcoming_definition="listing_date_after_collection_date_only_not_all_offerings",
                subscription_upcoming=sum(r["subscription_state"] == "subscription_upcoming" for r in results),
                subscription_open=sum(r["subscription_state"] == "subscription_open" for r in results),
                service_schedule_candidates=sum(r["service_schedule_candidate"] for r in results),
                completeness="queried_submission_window_only")
        except Exception as exc:
            # Never serialize request objects, headers, cookies or exception messages.
            manifest.update(status="failed", error_type=type(exc).__name__)
            raise
        finally:
            manifest["finished_at"] = datetime.now(KST).isoformat()
            (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"path": str(destination), **manifest}
