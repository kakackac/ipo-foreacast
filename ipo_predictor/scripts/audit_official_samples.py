"""Download bounded public DART samples and retain reproducible parser evidence.

This audit does not change production data or approve model training.
"""
import hashlib
import argparse
import math
import json
import re
import sys
import time
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, parse_qs, urlparse


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.collectors.dart_collector import DARTCollector

SAMPLES = {
    "cosmecca_flat_lockup_2016": "20161014000121",
    "incross_flat_lockup_2016": "20161018000162",
    "clio_flat_lockup_2016": "20161027000349",
    "parksystems_flat_lockup_2015": "20151208000179",
    "hct_flat_lockup_conflict_2016": "20161004000323",
    "daelim_allocated_lockup_2016": "20160318000684",
    "fusion_allocated_lockup_2016": "20161209000102",
    "daesung_2022": "20220808000054",
    "nbt_2021": "20210111000425",
    "sensorview_2023": "20230706000211",
    "ns_shopping_2015": "20150210000401",
    "wiseitech_2020": "20200128000055",
    "rznomics_2025": "20251205000229",
    "sensorview_prospectus_2023": "20230706000212",
    "rznomics_prospectus_2025": "20251205000237",
    "rznomics_risks_2025": "20251205000237",
    "ns_shopping_final_2015": "20150312000834",
    "melcon_2026": "20260929000191",
    "est_2026": "20261021000363",
    "sk14_2026": "20261014000196",
    "est_risks_2026": "20261021000370",
    "melcon_risks_2026": "20260929000194",
    "maint_prospectus_2015": "20150331001027",
    "dypiston_prospectus_2017": "20171127000116",
    "maxst_prospectus_2021": "20210715000154",
    "yas_correction_2017": "20170926000244",
    "airbusan_risks_2018": "20181217000216",
    "koreacenter_risks_2019": "20191120000496",
    "atgen_lockup_2015": "20151013000034",
    "curiox_lockup_2023": "20230731000302",
    "nrb_lockup_2025": "20250716000293",
    "futuretech_lockup_2015": "20151118000030",
    "paxnet_lockup_2016": "20160714000305",
    "giantstep_lockup_2021": "20210312000380",
    "voronoi_lockup_zero_2022": "20220613000198",
    "sinsin_quantity_conflict_2017": "20170214000585",
    "atgen_annual_financial_2015": "20151013000033",
    "ajnet_annual_financial_2015": "20150807000011",
    "sunbeauty_lockup_conflict_2021": "20210114000349",
    "vc_annual_financial_2022": "20220214000382",
    "hansolcnp_float_2016": "20160118000101",
    "me2on_terms_2016": "20160923000059",
    "hims_terms_2017": "20170707000203",
    "biodyne_terms_2021": "20210304000632",
    "milli_terms_2023": "20230915000280",
    "contec_terms_2023": "20231030000344",
    "pinkfong_terms_2025": "20251105000307",
    "surplus_final_2017": "20170116000302",
    "straffic_final_2017": "20171122000164",
    "ctk_final_2017": "20171124000037",
    "ecomaister_final_2018": "20180302000254",
    "haitai_terms_2016": "20160426000421",
    "sunic_terms_2017": "20170907000115",
    "iljin_terms_conflict_2021": "20210823000374",
    "iljin_prospectus_conflict_2021": "20210823000386",
    "design_malformed_rate_2018": "20181106000423",
    "gnbs_independent_lockup_2021": "20211018000277",
    "qrt_independent_quantity_2022": "20221021000286",
    "woorison_terms_band_2016": "20160714000393",
    "woorison_prospectus_band_omitted_2016": "20160714000401",
    "madup_terms_ambiguous_band_2026": "20260622000382",
    "madup_later_prospectus_band_2026": "20260622000384",
    "genoray_numbered_float_subtotal_2018": "20180515000471",
    "mekics_adjacent_post_float_2015": "20151202000452",
    "lemon_initial_float_subtotal_2020": "20200218000962",
}


def reuse_verified_original(url, path):
    cache=path.parent.parent/'dart_connection_sources'/(hashlib.sha256(url.encode()).hexdigest()+'.json')
    if not cache.exists():
        receipt=parse_qs(urlparse(url).query).get('rcpNo',[''])[0]
        financial=path.parent.parent/'dart_ipo_financial_sections'/(receipt+'.json')
        if re.fullmatch(r'\d{14}',receipt) and financial.exists():
            original=json.loads(financial.read_text())
            if original.get('source_url')==url and hashlib.sha256(original['html'].encode()).hexdigest()==original.get('sha256'):
                path.write_text(original['html'],encoding='utf-8')
                return True
        return False
    original=json.loads(cache.read_text())
    if original.get('url')!=url or hashlib.sha256(original['text'].encode()).hexdigest()!=original.get('sha256'):
        raise ValueError('official_sample_cached_original_integrity_failed')
    path.write_text(original['text'],encoding='utf-8')
    return True


def select_document(index, receipt, section=None):
    nodes = []
    for block in re.split(r"var node\d+ = \{\};", index)[1:]:
        fields = dict(re.findall(r"node\d+\['(text|rcpNo|dcmNo|eleId|offset|length|dtd)'\]\s*=\s*\"([^\"]*)\"", block))
        if fields.get("rcpNo") == receipt and "dcmNo" in fields:
            nodes.append(fields)
    body = next((n for n in nodes if "본문" in re.sub(r"\s+", "", n.get("text", ""))), None)
    offering = next((n for n in nodes if "제1부" in n.get("text", "")), None)
    selected = (next((n for n in nodes if n.get("text") == section), None)
                if section else offering or body or (nodes[0] if len(nodes) == 1 else None))
    if selected is None:
        return None
    return {k: selected[k] for k in ("rcpNo", "dcmNo", "eleId", "offset", "length", "dtd")}


def download(url, path):
    temporary = path.with_suffix(".part")
    subprocess.run([
        "curl", "--compressed", "--max-time", "180", "--fail", "--silent", "--show-error",
        "--output", str(temporary), url,
    ], check=True, timeout=185)
    temporary.replace(path)
    time.sleep(0.5)


def verify_record(record, reference):
    """Compare to reviewed values; never regenerate expectations from parser output."""
    problems = []
    for key in ("receipt", "source_url", "sha256"):
        if record.get(key) != reference.get(key):
            problems.append(f"source_mismatch:{key}")
    values = {**record.get("offering", {}), **record.get("demand", {}), **record.get("financial", {})}
    def evidence_agrees(actual, expected):
        if isinstance(expected,dict):
            return isinstance(actual,dict) and all(
                key in actual and evidence_agrees(actual[key],value) for key,value in expected.items())
        return actual == expected
    for field, expected in reference.get('expected_evidence',{}).items():
        actual=values.get(field)
        actual=json.loads(actual) if isinstance(actual,str) else actual
        if not evidence_agrees(actual,expected):
            problems.append('evidence_mismatch:'+field)
    basis=reference.get('financial_context',{}).get('expected_fs_div')
    if basis:
        evidence=json.loads(record.get('financial',{}).get('financial_feature_provenance','{}'))
        if not evidence or any(value.get('fs_div')!=basis for value in evidence.values()):
            problems.append('financial_basis_mismatch')
    for key, expected in reference["expected"].items():
        if key not in values:
            problems.append(f"missing_output:{key}")
            continue
        actual = values[key]
        agrees = actual is None if expected is None else actual == expected if isinstance(expected,str) else (
            isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)
        )
        if not agrees:
            problems.append(f"value_mismatch:{key}")
    return problems


def main():
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--cached-only", action="store_true")
    arguments.add_argument("--sample", choices=SAMPLES, action="append")
    args = arguments.parse_args()
    output = Path(__file__).resolve().parents[1] / "data/raw/official_sample_audit"
    output.mkdir(parents=True, exist_ok=True)
    collector = DARTCollector(api_key="unused_public_audit")
    reference_path = Path(__file__).with_name("official_sample_expectations.json")
    references = {r["sample"]: r for r in json.loads(reference_path.read_text())["samples"]}
    records = []
    for name, receipt in SAMPLES.items():
        if args.sample and name not in args.sample:
            continue
        print("auditing", name, flush=True)
        main_url = "https://dart.fss.or.kr/dsaf001/main.do?" + urlencode({"rcpNo": receipt})
        main_path = output / f"{name}_index.html"
        if not main_path.exists():
            reuse_verified_original(main_url,main_path)
        if not main_path.exists():
            if args.cached_only:
                records.append({"sample": name, "status": "cache_missing"})
                continue
            try:
                download(main_url, main_path)
            except (subprocess.SubprocessError, OSError) as exc:
                records.append({"sample": name, "receipt": receipt,
                                "status": "source_access_failed", "error_type": type(exc).__name__})
                continue
        index = main_path.read_text(encoding="utf-8")
        section = {"rznomics_prospectus_2025": "I. 모집 또는 매출에 관한 일반사항",
                   "rznomics_risks_2025": "III. 투자위험요소",
                   "est_risks_2026": "III. 투자위험요소",
                   "melcon_risks_2026": "III. 투자위험요소",
                   "maint_prospectus_2015": "Ⅰ. 모집 또는 매출에 관한 일반사항",
                   "dypiston_prospectus_2017": "Ⅰ. 모집 또는 매출에 관한 일반사항",
                   "maxst_prospectus_2021": "Ⅰ. 모집 또는 매출에 관한 일반사항"}.get(name)
        if name in {'airbusan_risks_2018','koreacenter_risks_2019'}:
            section='Ⅲ. 투자위험요소'
        section=references.get(name,{}).get('section_title') or section
        parameters = select_document(index, receipt, section)
        if not parameters:
            records.append({"sample": name, "receipt": receipt, "status": "viewer_link_not_found"})
            continue
        url = "https://dart.fss.or.kr/report/viewer.do?" + urlencode(parameters)
        body_path = output / (f"{name}_section_{parameters['eleId']}.html" if section else f"{name}_body.html")
        if not body_path.exists():
            reuse_verified_original(url,body_path)
        if not body_path.exists():
            if args.cached_only:
                records.append({"sample": name, "status": "cache_missing"})
                continue
            try:
                download(url, body_path)
            except (subprocess.SubprocessError, OSError) as exc:
                records.append({"sample": name, "receipt": receipt,
                                "status": "source_access_failed", "error_type": type(exc).__name__})
                continue
        body = body_path.read_text(encoding="utf-8")
        reference = references.get(name)
        view_evidence=None
        if reference and reference.get('extraction_view')=='explicit_after_correction_v1':
            from data.pipelines.correction_views import after_correction_view
            body,view_evidence=after_correction_view(body)
        elif reference and reference.get('extraction_view')=='normalized_after_correction_v2':
            from data.pipelines.stage_snapshots import amendment_view
            body,view_evidence=amendment_view(body)
        record = {
            "sample": name, "receipt": receipt, "source_url": url,
            "document_path": str(body_path), "selection": parameters,
            "sha256": hashlib.sha256(body_path.read_bytes()).hexdigest(),
            "audited_at": datetime.now(timezone.utc).isoformat(),
            "parser_sha256": hashlib.sha256(Path(sys.modules[DARTCollector.__module__].__file__).read_bytes()).hexdigest(),
            "float_parser_sha256": hashlib.sha256((Path(__file__).resolve().parents[1] / "data/collectors/float_tables.py").read_bytes()).hexdigest(),
            "expectations_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            "status": "parsed_not_independently_approved",
            "view_evidence": view_evidence,
        }
        if reference and reference.get('financial_context'):
            from data.pipelines.disclosure_financials import parse_disclosure_financials
            from data.pipelines.financial_history import summarize_asof
            from data.pipelines.disclosure_connections import viewer_identity
            context=reference['financial_context']
            record['financial_identity']=viewer_identity(index,receipt,context['corp_code'],context['published_at'])
            frame=parse_disclosure_financials(body,context['corp_code'],receipt,context['published_at'],url)
            record['financial']=summarize_asof(frame,context['corp_code'],context['cutoff'])
            record['financial_accounts']=json.loads(frame.to_json(orient='records'))
            record['parser_sha256']=hashlib.sha256(Path(sys.modules[parse_disclosure_financials.__module__].__file__).read_bytes()).hexdigest()
        else:
            record['offering']=collector._parse_offering_html(body,receipt)
            record['demand']=collector._parse_demand_forecast_html(body,name)
        record["comparison_failures"] = verify_record(record, reference) if reference else ["reference_missing"]
        record["status"] = "sample_passed" if not record["comparison_failures"] else "sample_failed"
        records.append(record)
    suffix = "" if not args.sample else "_" + hashlib.sha256("|".join(sorted(args.sample)).encode()).hexdigest()[:12]
    (output / f"results{suffix}.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "samples_requested": len(records),
        "samples_passed": sum(r["status"] == "sample_passed" for r in records),
        "all_reference_samples_checked": {r["sample"] for r in records} == set(references),
        "checked_fields": sum(len(references[r["sample"]]["expected"]) for r in records if r["status"] == "sample_passed"),
        "full_collection_authorized": False,
        "remaining_gate": "Authenticated ZIP transport, event linkage and time/source contracts not validated by this public HTML audit.",
    }
    (output / f"summary{suffix}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for record in records:
        print(record["sample"], {k: record.get("demand", {}).get(k) for k in ["institutional_demand_ratio", "lockup_commitment_ratio"]})
    print(json.dumps(summary, ensure_ascii=False))
    return 1 if any(r["status"] != "sample_passed" for r in records) else 0


if __name__ == "__main__":
    sys.exit(main())
