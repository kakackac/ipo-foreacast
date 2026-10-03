"""Receipt-bound annual financial collection and historical feature repair."""
import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from features.source_contracts import DART_ANNUAL_FINANCIAL_STATUS

FINANCIAL_STATUS = DART_ANNUAL_FINANCIAL_STATUS
CACHE_VERSION = 1


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     suffix=".tmp", delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, allow_nan=False, indent=2)
        temporary = handle.name
    os.replace(temporary, path)


def get_annual_history(dart, corp_code, listing_date, cache_dir, cached_only=False):
    """Cache only completed requests; old amount-only caches cannot supply evidence."""
    corp_code = str(corp_code)
    if not re.fullmatch(r"\d{8}", corp_code):
        raise ValueError("invalid_financial_corp_code")
    listing = pd.Timestamp(listing_date)
    frames = []
    now = datetime.now(timezone.utc)
    for year in range(listing.year - 1, max(2014, listing.year - 5), -1):
        path = Path(cache_dir) / f"{corp_code}_{year}_11011_v{CACHE_VERSION}.json"
        cached = None
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                body = json.dumps(payload["records"], sort_keys=True, allow_nan=False).encode()
                age = now - datetime.fromisoformat(payload["collected_at"])
                if (payload["schema_version"] == CACHE_VERSION
                        and payload["corp_code"] == corp_code and payload["year"] == year
                        and timedelta(0) <= age < timedelta(days=7)
                        and hashlib.sha256(body).hexdigest() == payload["sha256"]):
                    cached = pd.DataFrame(payload["records"])
            except (ValueError, KeyError, TypeError):
                pass
        if cached is not None:
            frames.append(cached)
            continue
        if cached_only:
            continue
        frame = dart.get_financial_statements(corp_code, year).copy()
        if not frame.empty:
            frame["publication_verified"] = False
            frame["published_at"] = None
            frame["missing_reason"] = "receipt_metadata_missing"
            if {"rcept_no", "corp_code", "report_code", "response_year"}.issubset(frame):
                for receipt, group in frame.groupby("rcept_no"):
                    receipt = str(receipt)
                    if not re.fullmatch(r"\d{14}", receipt):
                        continue
                    if not (group.corp_code.eq(corp_code).all() and group.report_code.eq("11011").all()
                            and group.response_year.eq(str(year)).all()):
                        frame.loc[group.index, "missing_reason"] = "financial_response_identity_mismatch"
                        continue
                    if not hasattr(dart, "get_company_disclosure_list"):
                        continue
                    day = receipt[:8]
                    disclosures = dart.get_company_disclosure_list(corp_code, day, day)
                    matched = disclosures[
                        disclosures.rcept_no.astype(str).eq(receipt)
                        & disclosures.corp_code.astype(str).eq(corp_code)
                    ] if {"rcept_no", "corp_code", "rcept_dt", "report_nm"}.issubset(disclosures) else pd.DataFrame()
                    valid = []
                    for report in matched.itertuples(index=False):
                        published = pd.to_datetime(report.rcept_dt, errors="coerce")
                        title = str(report.report_nm)
                        if (pd.notna(published) and published.strftime("%Y%m%d") == day
                                and "사업보고서" in title and re.search(rf"\({year}\.\d{{2}}\)", title)):
                            valid.append(published.strftime("%Y-%m-%d"))
                    if len(set(valid)) == 1:
                        frame.loc[group.index, "publication_verified"] = True
                        frame.loc[group.index, "published_at"] = valid[0]
                        frame.loc[group.index, "missing_reason"] = None
                    else:
                        frame.loc[group.index, "missing_reason"] = "annual_receipt_not_verified_in_official_list"
        records = json.loads(frame.to_json(orient="records", date_format="iso"))
        payload = {"schema_version": CACHE_VERSION, "corp_code": corp_code, "year": year,
                   "collected_at": now.isoformat(), "records": records,
                   "sha256": hashlib.sha256(json.dumps(records, sort_keys=True, allow_nan=False).encode()).hexdigest()}
        atomic_json(path, payload)
        frames.append(frame)
    return pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(
        not f.empty for f in frames) else pd.DataFrame()


def summarize_asof(financials, corp_code, cutoff):
    """Select published annual values without combining currencies or statement bases."""
    metadata = {"year", "account_name_en", "amount", "corp_code", "rcept_no", "report_code",
                "response_year", "fs_div", "currency", "published_at", "publication_verified"}
    if financials.empty:
        return {"financial_time_validation_status": "annual_structured_values_unavailable",
                "financial_feature_provenance": "{}"}
    if not metadata.issubset(financials):
        return {"financial_time_validation_status": "receipt_metadata_missing",
                "financial_feature_provenance": "{}"}
    cutoff = pd.Timestamp(cutoff)
    cutoff = cutoff.tz_localize("Asia/Seoul") if cutoff.tzinfo is None else cutoff.tz_convert("Asia/Seoul")
    frame = financials.copy()
    dates = pd.to_datetime(frame.published_at, errors="coerce", utc=True)
    # Public dates have no intraday publication time: permit only from the next local day.
    frame["available_at"] = dates.dt.tz_convert("Asia/Seoul").dt.normalize() + pd.Timedelta(days=1)
    frame["amount"] = pd.to_numeric(frame.amount, errors="coerce")
    frame = frame[frame.publication_verified.eq(True) & frame.corp_code.astype(str).eq(str(corp_code))
                  & frame.currency.eq("KRW") & frame.report_code.isin(["11011", "IPO_FINANCIAL_DISCLOSURE"])
                  & frame.fs_div.isin(["CFS", "OFS"]) & (frame.available_at < cutoff)
                  & frame.amount.map(lambda v: pd.notna(v) and math.isfinite(v))]
    yearly = {}
    for year, group in frame.groupby("year"):
        if not group.response_year.eq(str(int(year))).all():
            continue
        candidates = []
        for (receipt, basis), source in group.groupby(["rcept_no", "fs_div"]):
            names = set(source.account_name_en)
            complete = {"revenue", "operating_income", "total_liabilities", "equity"}.issubset(names)
            candidates.append((complete, source.available_at.max(), basis == "CFS", receipt, source))
        group = sorted(candidates, key=lambda x: x[:4])[-1][4]
        accounts = {}
        for name, entries in group.groupby("account_name_en"):
            if entries.amount.nunique() == 1:
                accounts[name] = entries.iloc[0]
        if accounts:
            assets, liabilities, equity = (accounts.get(x) for x in ("total_assets", "total_liabilities", "equity"))
            if assets is not None and liabilities is not None and equity is not None:
                scale = max(float(x.get("unit_multiplier", 1)) for x in (assets, liabilities, equity))
                if abs(assets.amount - liabilities.amount - equity.amount) > max(2*scale, abs(assets.amount)*1e-6):
                    for name in ("total_assets", "total_liabilities", "equity"):
                        accounts.pop(name, None)
            yearly[int(year)] = accounts
    if not yearly:
        return {"financial_time_validation_status": "no_verified_pre_cutoff_krw_annual_values",
                "financial_feature_provenance": "{}"}
    latest_year = max(yearly)
    latest = yearly[latest_year]
    result = {name: float(row.amount) for name, row in latest.items()}
    provenance = {}

    def add_evidence(feature, entries):
        provenance[feature] = {
            "source_reference": "|".join(sorted({str(x.rcept_no) for x in entries})),
            "available_at": max(x.available_at for x in entries).isoformat(),
            "validation_status": FINANCIAL_STATUS,
            "fs_div": entries[0].fs_div, "currency": "KRW",
            "source_urls": sorted({str(x.get("source_url")) for x in entries if pd.notna(x.get("source_url"))}),
            "source_sha256": sorted({str(x.get("source_sha256")) for x in entries if pd.notna(x.get("source_sha256"))}),
        }

    for name, row in latest.items():
        add_evidence(name, [row])
    for feature, numerator, denominator in [("operating_margin", "operating_income", "revenue"),
                                             ("debt_ratio", "total_liabilities", "equity")]:
        a, b = latest.get(numerator), latest.get(denominator)
        if a is not None and b is not None and b.amount > 0:
            result[feature] = float(a.amount / b.amount)
            add_evidence(feature, [a, b])
    result["revenue_growth_3y"] = None
    old = yearly.get(latest_year - 3, {}).get("revenue")
    recent = latest.get("revenue")
    if (old is not None and recent is not None and old.amount > 0 and recent.amount > 0
            and old.fs_div == recent.fs_div):
        result["revenue_growth_3y"] = float((recent.amount / old.amount) ** (1 / 3) - 1)
        add_evidence("revenue_growth_3y", [old, recent])
    result.update(financial_as_of_year=latest_year, financial_time_validation_status=FINANCIAL_STATUS,
                  financial_available_at=max(x.available_at for x in latest.values()).isoformat(),
                  financial_feature_provenance=json.dumps(provenance, ensure_ascii=False))
    return result


def run_repair(dart, raw_dir, processed_dir, start_year=2015, end_year=2026, dry_run=False, event_limit=None,
               documents_only=False):
    """Repair financial features without recollecting IPO documents or KRX prices."""
    from data.processors.feature_engineer import FeatureEngineer
    raw_dir, processed_dir = Path(raw_dir), Path(processed_dir)
    features_path = processed_dir / "features_all.parquet"
    original_hash = hashlib.sha256(features_path.read_bytes()).hexdigest()
    features = pd.read_parquet(features_path)
    dates = pd.to_datetime(features.listing_date, errors="coerce")
    targets = features[features.event_class.eq("general_ipo") & features.market.isin(["KOSPI", "KOSDAQ"])
                       & dates.dt.year.between(start_year, end_year)].copy()
    raw_records = pd.read_parquet(raw_dir / "dart_ipo_raw.parquet")
    identities = raw_records[["event_id", "corp_code"]].drop_duplicates()
    if identities.event_id.duplicated().any():
        raise ValueError("financial_event_corp_identity_ambiguous")
    targets = targets.drop(columns=["corp_code"], errors="ignore").merge(identities, on="event_id", how="left", validate="one_to_one")
    targets = targets.sort_values(["listing_date", "event_id"])
    if event_limit is not None:
        if event_limit < 1:
            raise ValueError("financial_event_limit_must_be_positive")
        # Span years rather than validating only adjacent IPOs.
        positions = sorted({round(i * (len(targets)-1) / max(1, event_limit-1))
                            for i in range(min(event_limit, len(targets)))})
        targets = targets.iloc[positions]
    requests = {(str(r.corp_code), y) for r in targets.itertuples(index=False)
                if re.fullmatch(r"\d{8}", str(r.corp_code))
                for y in range(pd.Timestamp(r.listing_date).year - 1,
                               max(2014, pd.Timestamp(r.listing_date).year - 5), -1)}
    plan = {"events": len(targets), "unique_annual_requests": len(requests),
            "request_count_is_not_total_http_calls": True,
            "needs_api": None if documents_only else "DART", "public_dart_documents": True,
            "krx_recollection_required": False,
            "legacy_amount_only_cache_rows": len(pd.read_parquet(raw_dir / "dart_financials.parquet")),
            "training_executed": False}
    if dry_run:
        destination = processed_dir / "financial_collection_plan.json"
        atomic_json(destination, {**plan, "input_sha256": original_hash,
                                  "requests": [{"corp_code": c, "business_year": y} for c, y in sorted(requests)],
                                  "event_ids": targets.event_id.astype(str).tolist()})
        return {**plan, "path": str(destination)}
    if not dart.is_configured and not documents_only:
        raise RuntimeError("DART_API_KEY is required for financial receipt recovery")
    summaries, all_rows, resolutions = {}, [], []
    for row in targets.itertuples(index=False):
        if not re.fullmatch(r"\d{8}", str(row.corp_code)):
            summaries[row.event_id] = {"financial_time_validation_status": "dart_corp_code_missing",
                                       "financial_feature_provenance": "{}"}
            continue
        history = get_annual_history(dart, row.corp_code, row.listing_date, raw_dir / "dart_financial_sources",
                                     cached_only=documents_only)
        cutoff = pd.Timestamp(row.listing_date).normalize() - pd.Timedelta(hours=3)
        summary = summarize_asof(history, row.corp_code, cutoff)
        structured_status = summary["financial_time_validation_status"]
        document_status = "not_required"
        if not {"operating_margin", "debt_ratio"}.issubset(summary):
            from data.pipelines.disclosure_financials import collect_disclosure_financials
            extra, document_status = collect_disclosure_financials(
                dart, raw_dir, row.event_id, row.corp_code, row.corp_name, cutoff)
            if not extra.empty:
                history = pd.concat([history, extra], ignore_index=True) if not history.empty else extra
                summary = summarize_asof(history, row.corp_code, cutoff)
            elif summary["financial_time_validation_status"] != FINANCIAL_STATUS:
                summary["financial_time_validation_status"] = document_status
        resolutions.append({"event_id": row.event_id, "structured_status": structured_status,
                            "disclosure_status": document_status,
                            "result_status": summary["financial_time_validation_status"]})
        if not history.empty:
            history["event_id"] = row.event_id
            all_rows.append(history)
        summaries[row.event_id] = summary
    repaired = features.copy()
    owned_values = ["revenue", "operating_income", "net_income", "total_assets", "total_liabilities",
                    "equity", "eps", "revenue_growth_3y", "operating_margin", "debt_ratio",
                    "offering_per", "per_vs_sector_median"]
    selected = repaired.event_id.isin(summaries)
    for column in owned_values:
        if column not in repaired:
            repaired[column] = float("nan")
        repaired.loc[selected, column] = float("nan")
    for event, summary in summaries.items():
        for key, value in summary.items():
            if key not in repaired:
                repaired[key] = pd.Series(index=repaired.index, dtype=object)
            repaired.loc[repaired.event_id.eq(event), key] = value
    engineer = FeatureEngineer(feature_set="phase2")
    repaired = engineer._calc_financial_features(repaired)
    repaired = engineer._calc_valuation_features(repaired)
    observations = engineer.build_feature_observations(repaired)
    if hashlib.sha256(features_path.read_bytes()).hexdigest() != original_hash:
        raise RuntimeError("financial_repair_inputs_changed")
    fingerprint = hashlib.sha256((original_hash + pd.DataFrame(summaries).to_json()
                                  + Path(__file__).read_text()
                                  + (Path(__file__).with_name("disclosure_financials.py")).read_text()).encode()).hexdigest()[:20]
    root = processed_dir / "financial_repairs" / fingerprint
    if root.exists():
        manifest = json.loads((root / "manifest.json").read_text())
        for name, digest in manifest["output_sha256"].items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError("financial_repair_existing_output_changed")
        return {**manifest, "path": str(root), "reused": True}
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".financial.", dir=root.parent))
    outputs = {"features_all.parquet": repaired, "feature_observations.parquet": observations,
               "source_resolutions.parquet": pd.DataFrame(resolutions),
               "financial_sources.parquet": pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()}
    for name, frame in outputs.items():
        frame.to_parquet(staging / name, index=False)
    report = {**plan, "input_sha256": original_hash,
              "status_counts": pd.Series([s["financial_time_validation_status"] for s in summaries.values()]).value_counts().to_dict(),
              "output_sha256": {name: hashlib.sha256((staging / name).read_bytes()).hexdigest() for name in outputs},
              "production_files_replaced": False, "asof_policy": "listing_eve_2100_core_v1",
              "disclosure_resolution_counts": pd.Series([r["disclosure_status"] for r in resolutions]).value_counts().to_dict(),
              "financial_missing_counts": repaired.loc[selected, owned_values].isna().sum().to_dict()}
    atomic_json(staging / "manifest.json", report)
    staging.rename(root)
    return {**report, "path": str(root), "reused": False}
