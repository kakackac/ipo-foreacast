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
    "brils_printed_initial_float_2026": "20260916000240",
    "jlk_post_total_float_2019": "20191129001530",
    "beautyskin_horizontal_float_2023": "20230712000262",
    "sigetronics_post_total_float_2023": "20230721000288",
    "ivim_post_total_float_2024": "20240723000182",
    "irondevice_post_total_float_2024": "20240906000325",
    "lsasia_excluded_total_conflict_float_2016": "20160907000314",
    "almac_original_percent_conflict_float_2023": "20230616000539",
    "opennol_preferred_note_float_2023": "20230620000341",
    "qualitas_initial_conversion_note_float_2023": "20231017000137",
    "ecoprobm_original_percent_conflict_float_2019": "20190220000799",
    "uracle_original_percent_conflict_float_2024": "20240805000137",
    "boratr_ordinary_initial_float_2017": "20170526000344",
    "raphas_listing_initial_float_2019": "20191030000166",
    "saebit_offering_initial_float_2022": "20220726000371",
    "rp_offering_initial_float_2022": "20220919000288",
    "easycare_bound_share_unit_2019": "20190308000278",
    "cowin_bound_share_unit_2019": "20190723000079",
    "knj_bound_share_unit_2019": "20191015000138",
    "newpower_subtotal_lockup_2016": "20161118000332",
    "sjgroup_header_lockup_2019": "20191105000199",
    "taewoong_header_lockup_2019": "20191125000235",
    "aplus_header_lockup_2020": "20201109000184",
    "hana_header_lockup_2020": "20201112000090",
    "jitech_composite_lockup_2022": "20221025000142",
    "samyang_continued_lockup_2025": "20250114000182",
    "lgcns_continued_lockup_2025": "20250117000578",
    "daehan_continued_lockup_2025": "20250721000037",
    "noeul_multiheader_dash_lockup_2022": "20220218001132",
    "webs_count_scope_lockup_2015": "20150810000107",
    "cherry_malformed_total_lockup_2017": "20171120000164",
    "unitechno_dash_lockup_2016": "20160902000083",
    "socar_total_alias_lockup_2022": "20220809000316",
    "modelsolution_subtotal_lockup_2022": "20220923000080",
    "seculetter_total_alias_lockup_2023": "20230811001361",
    "hyundaihims_total_alias_lockup_2024": "20240116000207",
    "sunic_threshold_float_2017": "20170908000030",
    "cng_malformed_quantity_float_2018": "20180115000154",
    "cntus_option_scope_float_2021": "20210115000208",
    "dayone_listing_schedule_float_2025": "20250114000477",
    "woorison_group_float_2016": "20160714000401",
    "paxnet_column_float_2016": "20160714000305",
    "zaigle_total_float_2016": "20160824000350",
    "ymt_class_float_2017": "20170414001551",
    "younghwa_inline_float_2017": "20171016000066",
    "jitech_group_float_2022": "20221025000142",
    "strad_group_float_2026": "20260617000171",
    "syswork_group_float_2017": "20171213000011",
    "alrico_ordinary_float_2018": "20180131000083",
    "scm_ordinary_float_2020": "20200605000211",
    "solum_column_scope_float_2021": "20210120000141",
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


def audit_band_origins():
    """Replay reviewed band-source connections from cached official originals."""
    import pandas as pd
    from data.pipelines.reconciled_dataset import pin_connections
    from data.pipelines.stage_snapshots import read_document, document_analysis, stage_documents, values_from_document
    from data.pipelines.band_source_resolver import resolve_offering_band, currency_band, finalized_band_overview
    from data.pipelines.financial_history import atomic_json

    root=Path(__file__).resolve().parents[1]
    raw=root/'data/raw';processed=root/'data/processed'
    source,_=pin_connections(processed)
    rows=pd.read_parquet(source/'connection_results.parquet').set_index('event_id')
    references=json.loads(Path(__file__).with_name('official_sample_expectations.json').read_text())['band_origin_samples']
    collector=DARTCollector(api_key='unused');checks=[]
    for sample in references:
        row=rows.loc[sample['event_id']];record=json.loads(row.record);proof=json.loads(row.proof)
        family={m['rcept_no']:m for bundle in (proof,proof.get('supplementary',{})) for m in bundle.get('family',[])}
        def load(m):return read_document(collector,raw,m,str(record['corp_code']),True)
        analyse=lambda m,b,c:document_analysis(collector,b,m['rcept_no'],c,raw)
        post=stage_documents(list(family.values()),load,record['listing_date'],analyse=analyse)['post_demand']
        if post['status']!='historical_document_selected':raise ValueError('sample_post_document_not_selected')
        values,_=values_from_document(collector,post,str(record['corp_code']),'post_demand')
        main=resolve_offering_band(raw,family[record['rcept_no']],proof,record['offering_price'],record['listing_date'],collector)
        expected=tuple(sample['expected_band'])
        if (currency_band(main)!=expected or currency_band(post['analysis']['offering'])!=expected
                or main['price_band_rcept_no']!=sample['band_receipt']
                or post['metadata']['rcept_no']!=sample['complete_receipt']
                or post['proof']['result_field_sources']['price_band_low']['receipt']!=sample['band_receipt']
                or not math.isclose(values['offering_price_band_position'],sample['expected_post_position'],abs_tol=1e-12)):
            raise ValueError('official_band_origin_sample_mismatch:'+sample['company'])
        checks.append({'company':sample['company'],'event_id':sample['event_id'],
            'band':list(expected),'band_receipt':sample['band_receipt'],
            'post_receipt':post['metadata']['rcept_no'],'main_evidence':json.loads(main['price_band_resolution_evidence']),
            'post_band_origin':post['proof']['result_field_sources']['price_band_low']})
    transitions=[]
    for sample in json.loads(Path(__file__).with_name('official_sample_expectations.json').read_text()).get('band_transition_samples',[]):
        matched=rows[rows.record.map(lambda record:json.loads(record)['rcept_no']==sample['notice_receipt'])]
        if len(matched)!=1:raise ValueError('official_transition_sample_event_not_unique')
        row=matched.iloc[0];record=json.loads(row.record);proof=json.loads(row.proof)
        family={m['rcept_no']:m for bundle in (proof,proof.get('supplementary',{})) for m in bundle.get('family',[])}
        resolved=resolve_offering_band(raw,family[record['rcept_no']],proof,record['offering_price'],record['listing_date'],collector)
        history=json.loads(resolved['price_band_resolution_evidence'])['historical_band']
        item=next((c for c in history['checks'] if c['receipt']==sample['notice_receipt']),None)
        if not item or item.get('status')!='explicit_planned_range_finalized':
            raise ValueError('official_transition_sample_not_verified:'+sample['company'])
        evidence=item['finalization_overview'];pair=tuple(sample['expected_band'])
        if currency_band(resolved)!=pair or evidence['confirmed_price']!=sample['expected_price']:
            raise ValueError('official_transition_value_mismatch:'+sample['company'])
        notice,notice_origin=read_document(collector,raw,family[sample['notice_receipt']],str(record['corp_code']),True)
        full_meta=family[evidence['complete_receipt']]
        full,full_origin=read_document(collector,raw,full_meta,str(record['corp_code']),True)
        confirmed=collector._parse_offering_html(full,full_meta['rcept_no'])
        repeated=finalized_band_overview(notice,family[sample['notice_receipt']],notice_origin,
                                       (full_meta,full,full_origin),pair,confirmed)
        if repeated!=evidence:raise ValueError('official_transition_evidence_not_reproduced')
        transitions.append({'company':sample['company'],'notice_receipt':sample['notice_receipt'],
            'expected_band':list(pair),'expected_price':sample['expected_price'],'evidence':evidence,
            'scope':'band_continuity_only_not_institutional_or_stage_selection_approval'})
    result={'samples_checked':len(checks),'checks':checks,'transition_samples_checked':len(transitions),
        'transition_checks':transitions,'cached_only':True,'passed':True,
        'training_executed':False,'deployment_authorized':False}
    atomic_json(processed/'band_origin_sample_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in {'checks','transition_checks'}},ensure_ascii=False))


def main():
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--cached-only", action="store_true")
    arguments.add_argument("--sample", choices=SAMPLES, action="append")
    arguments.add_argument("--band-origins", action="store_true", help="Replay the reviewed cached band-origin cases only")
    args = arguments.parse_args()
    if args.band_origins:
        if args.sample or not args.cached_only:arguments.error('--band-origins requires --cached-only without --sample')
        audit_band_origins();return
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
