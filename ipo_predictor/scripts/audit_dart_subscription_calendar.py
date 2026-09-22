"""Audit public DART subscription calendars; these also contain non-IPO offerings."""
import argparse
import hashlib
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

URL = "https://dart.fss.or.kr/dsac008/main.do"


def compare_reference(events, reference):
    result = []
    for item in reference["companies"]:
        name = item.get("official_name_candidate", item["name"]).replace(" ", "")
        matches = [r for r in events if r["corp_name"].replace(" ", "") == name
                   and (not item.get("official_corp_code_candidate") or r["corp_code"] == item["official_corp_code_candidate"])]
        result.append({"name": item["name"], "matches": matches,
                       "match_basis": "reviewed_alias_candidate" if "official_name_candidate" in item else "exact_name_candidate",
                       "model_eligible": False})
    return result


def parse_calendar(content, year, month):
    soup = BeautifulSoup(content, "html.parser")
    for key, expected in (("year", year), ("month", month)):
        selected = soup.select_one(f"select#{key} option[selected]")
        if selected is None or int(selected.get("value", "0")) != expected:
            raise ValueError("DART returned another calendar period")
    cells = soup.select("li.day:not(.other-month)")
    if not cells:
        raise ValueError("Calendar structure missing; not an empty official result")
    rows = []
    for cell in cells:
        day = cell.select_one("div.date")
        if day is None:
            raise ValueError("Calendar day missing")
        day = date(year, month, int(day.get_text(strip=True))).isoformat()
        for link in cell.select('a[href*="rcpNo="]'):
            receipt = parse_qs(urlparse(link["href"]).query).get("rcpNo", [""])[0]
            company = link.select_one("div.gm")
            codes = [] if company is None else [c for c in company.get("class", []) if re.fullmatch(r"\d{8}", c)]
            if not re.fullmatch(r"\d{14}", receipt) or len(codes) != 1:
                raise ValueError("Calendar disclosure identity missing")
            for badge in company.select("span"):
                badge.decompose()
            value = company.get_text(" ", strip=True)
            match = re.fullmatch(r"(.+?)\s*\[(시작|종료)\]", value)
            if not match:
                raise ValueError("Calendar event type unknown")
            rows.append({"corp_name": match.group(1).strip(), "corp_code": codes[0], "rcept_no": receipt,
                "subscription_date": day, "event_type": "start" if match.group(2) == "시작" else "end",
                "source_url": "https://dart.fss.or.kr" + link["href"],
                "classification": "equity_subscription_ipo_status_unverified", "model_eligible": False})
    return rows


def collect(year, months, output, reference):
    destination = Path(output) / datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%dT%H%M%S%f")
    destination.mkdir(parents=True, exist_ok=False)
    report = {"source": URL, "requests": [], "events": [], "status": "started"}
    try:
        for month in months:
            response = requests.get(URL, params={"year": year, "month": month}, timeout=30)
            response.raise_for_status()
            (destination / f"{year}_{month:02d}.html").write_bytes(response.content)
            report["requests"].append({"year": year, "month": month, "sha256": hashlib.sha256(response.content).hexdigest()})
            report["events"].extend(parse_calendar(response.content, year, month))
        report["comparison"] = compare_reference(report["events"], reference)
        report.update(status="complete", matched_reference_names=sum(bool(r["matches"]) for r in report["comparison"]))
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        report["collected_at"] = datetime.now(ZoneInfo("Asia/Seoul")).isoformat()
        (destination / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"path": str(destination), "status": report["status"], "calendar_events": len(report["events"]), "matched_reference_names": report["matched_reference_names"]}


def replay(folder, reference):
    folder = Path(folder)
    report = json.loads((folder / "report.json").read_text())
    if report["status"] != "complete":
        raise ValueError("Cannot replay incomplete audit")
    events = []
    for request in report["requests"]:
        content = (folder / f"{request['year']}_{request['month']:02d}.html").read_bytes()
        if hashlib.sha256(content).hexdigest() != request["sha256"]:
            raise ValueError("Calendar evidence hash mismatch")
        events.extend(parse_calendar(content, request["year"], request["month"]))
    return {"comparison": compare_reference(events, reference), "events": events, "model_eligible": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--months", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads((Path(__file__).resolve().parents[1] / "data/manual/screenshot_schedule_reference.json").read_text())
    try:
        print(json.dumps(collect(args.year, args.months, args.output, reference), ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}), file=sys.stderr)
        sys.exit(1)
