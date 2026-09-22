"""Public DART viewer lineage audit. Does not approve IPO/model features."""
import argparse
import hashlib
import json
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path
from urllib.parse import parse_qs
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_dart_subscription_calendar import replay

BASE = "https://dart.fss.or.kr"


def parse_viewer(content, receipt, corp_code, as_of):
    soup = BeautifulSoup(content, "html.parser")
    identities = set()
    for element in soup.select("[onclick]"):
        match = re.search(r"openCorpInfoNew\('([0-9]{8})'", element["onclick"])
        if match:
            identities.add(match.group(1))
    if identities != {corp_code}:
        raise ValueError("Viewer corporate identity mismatch")
    family = []
    selected = None
    for option in soup.select("select#family option"):
        values = parse_qs(option.get("value", ""))
        rcp = values.get("rcpNo", [""])[0]
        if not re.fullmatch(r"\d{14}", rcp):
            continue
        label = " ".join(option.get_text(" ", strip=True).split())
        match = re.match(r"(\d{4})\.(\d{2})\.(\d{2})", label)
        if not match:
            raise ValueError("Missing family disclosure date")
        published = date(*map(int, match.groups())).isoformat()
        family.append({"rcept_no": rcp, "published_date": published, "label": label,
                       "equity_registration": "증권신고서(지분증권)" in label})
        if option.has_attr("selected"):
            selected = rcp
    if selected != receipt or not any(r["rcept_no"] == receipt and r["equity_registration"] for r in family):
        raise ValueError("Calendar receipt is not selected equity registration")
    eligible = [r for r in family if r["equity_registration"] and r["published_date"] <= as_of]
    if not eligible:
        raise ValueError("No eligible registration")
    latest_day = max(r["published_date"] for r in eligible)
    latest = sorted({r["rcept_no"] for r in eligible if r["published_date"] == latest_day})
    return {"corp_code": corp_code, "rcept_no": receipt, "family": family,
            "latest_registration_candidates": latest,
            "status": "calendar_receipt_latest_in_viewer_family" if latest == [receipt] else "newer_or_ambiguous_registration_review",
            "model_eligible": False}


def summary_node(content, receipt):
    soup = BeautifulSoup(content, "html.parser")
    nodes = []
    for script in soup.find_all("script"):
        # Extract only string-literal metadata; never execute page JavaScript.
        for block in re.split(r"var\s+node\d+\s*=\s*\{\s*\}\s*;", script.get_text()):
            pairs = re.findall(r"node\d+\['(text|rcpNo|dcmNo|eleId|offset|length|dtd)'\]\s*=\s*\"([^\"]*)\"\s*;", block)
            if len(pairs) != len({key for key, _ in pairs}):
                continue
            node = dict(pairs)
            if re.sub(r"\s", "", node.get("text", "")) != "요약정보":
                continue
            if node.get("rcpNo") != receipt or node.get("dtd") not in ("dart3.xsd", "dart4.xsd"):
                raise ValueError("Summary receipt/schema mismatch")
            for key in ("dcmNo", "eleId", "offset", "length"):
                if not re.fullmatch(r"\d+", node.get(key, "")):
                    raise ValueError("Invalid summary parameters")
            nodes.append({key: value for key, value in node.items() if key != "text"})
    if len(nodes) != 1:
        raise ValueError("Unique summary section not found")
    return nodes[0]


def run(calendar, output, reference):
    candidates = replay(calendar, reference)["comparison"]
    root = Path(output) / datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%dT%H%M%S%f")
    root.mkdir(parents=True, exist_ok=False)
    report = {"source": "public_DART_viewer", "rows": [], "model_eligible": False}
    session = requests.Session()
    for candidate in candidates:
        starts = {(r["corp_code"], r["rcept_no"]): r for r in candidate["matches"] if r["event_type"] == "start"}
        entry = {"reference_name": candidate["name"], "model_eligible": False}
        report["rows"].append(entry)
        try:
            if len(starts) != 1:
                raise ValueError("Calendar identity ambiguous")
            (corp, receipt), event = next(iter(starts.items()))
            entry.update(corp_code=corp, calendar_receipt=receipt, subscription_start=event["subscription_date"])
            response = session.get(BASE + "/dsaf001/main.do", params={"rcpNo": receipt}, timeout=30)
            response.raise_for_status()
            content = response.content
            (root / f"{receipt}_viewer.html").write_bytes(content)
            entry["viewer_sha256"] = hashlib.sha256(content).hexdigest()
            entry.update(parse_viewer(content, receipt, corp, datetime.now(ZoneInfo("Asia/Seoul")).date().isoformat()))
            if entry["status"] == "calendar_receipt_latest_in_viewer_family":
                params = summary_node(content, receipt)
                response = session.get(BASE + "/report/viewer.do", params=params, timeout=30)
                response.raise_for_status()
                body = response.content
                parsed = BeautifulSoup(body, "html.parser")
                if not parsed.find("table") or len(parsed.get_text(strip=True)) < 200:
                    raise ValueError("Summary body unavailable")
                (root / f"{receipt}_summary.html").write_bytes(body)
                entry.update(summary_sha256=hashlib.sha256(body).hexdigest(), summary_request=params,
                             summary_status="collected_requires_ipo_and_feature_verification")
                paragraphs = [" ".join(p.get_text(" ", strip=True).split()) for p in parsed.select("p")]
                entry["listing_context_candidates"] = [p for p in paragraphs if any(w in p for w in ("신규상장", "기업공개", "기업인수목적", "이전상장"))][:5]
        except Exception as exc:
            entry.update(status="review_required", error_type=type(exc).__name__)
        finally:
            entry["collected_at"] = datetime.now(ZoneInfo("Asia/Seoul")).isoformat()
            (root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            time.sleep(.3)
    return {"path": str(root), "rows": len(report["rows"]),
            "latest_family_verified": sum(r["status"] == "calendar_receipt_latest_in_viewer_family" for r in report["rows"]),
            "summaries_collected": sum("summary_sha256" in r for r in report["rows"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calendar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads((Path(__file__).resolve().parents[1] / "data/manual/screenshot_schedule_reference.json").read_text())
    print(json.dumps(run(args.calendar, args.output, reference), ensure_ascii=False))
