"""Replay NS Shopping's price selection with saved official sources, without network."""
import hashlib
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from config import RAW_DIR, PROC_DIR
from data.collectors.dart_collector import DARTCollector
from data.pipelines.historical_ipo_pipeline import HistoricalIPOPipeline


def main():
    refs = json.loads(Path(__file__).with_name("official_sample_expectations.json").read_text())
    refs = {row["sample"]: row for row in refs["samples"]}
    originals = {}
    for name in ("ns_shopping_2015", "ns_shopping_final_2015"):
        content = (RAW_DIR / "official_sample_audit" / f"{name}_body.html").read_bytes()
        if hashlib.sha256(content).hexdigest() != refs[name]["sha256"]:
            raise RuntimeError("Official sample hash mismatch")
        originals[refs[name]["receipt"]] = content.decode("utf-8")
    events = pd.read_parquet(RAW_DIR / "krx_official_event_master.parquet")
    events = events[events["corp_name"].eq("엔에스쇼핑")]
    if len(events) != 1:
        raise RuntimeError("Official event not unique")
    lineage = pd.read_parquet(RAW_DIR / "dart_disclosure_lineage.parquet")
    lineage = lineage[lineage["event_id"].eq(events.iloc[0]["event_id"])].copy()
    lineage = lineage.dropna(subset=["rcept_no"]).drop_duplicates("rcept_no")
    filings = lineage[lineage["filing_report_nm"].str.contains("증권신고서", na=False)].rename(
        columns={"filing_report_nm": "report_nm"})
    candidates = lineage.rename(columns={"filing_report_nm": "report_nm"})[
        ["rcept_no", "rcept_dt", "report_nm"]].to_dict("records")

    class LocalSources(DARTCollector):
        def get_document_text(self, receipt, **kwargs):
            if receipt not in originals:
                raise RuntimeError("local_audit_source_not_available")
            return originals[receipt]

        def get_ipo_disclosure_list(self, start, end):
            return filings.copy()

        def find_demand_forecast_disclosure_records(self, corp_code, start, end):
            return candidates

        def get_equity_offering_prices(self, *args):
            return []

        def get_financial_statements(self, *args):
            return pd.DataFrame()

    collector = LocalSources("", None)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        pipeline = HistoricalIPOPipeline(dart_collector=collector,
            raw_dir=root / "raw", processed_dir=root / "processed")
        rows, _ = pipeline._collect_dart_records(events, 2015, 2015, include_dart_demand_audit=True)
        if len(rows) != 1 or rows.iloc[0]["offering_price"] != 235000:
            raise RuntimeError("Official confirmed price did not reach output")
        if rows.iloc[0]["rcept_no"] != "20150312000834":
            raise RuntimeError("Wrong price source selected")
        PROC_DIR.mkdir(parents=True, exist_ok=True)
        output = PROC_DIR / "ns_price_lineage_sample.parquet"
        rows.to_parquet(output, index=False)
        saved = pd.read_parquet(output)
        if saved.iloc[0]["offering_price"] != 235000:
            raise RuntimeError("Saved price differs")
    print(json.dumps({"receipt": saved.iloc[0]["rcept_no"], "offering_price": 235000,
        "status": "official_source_replay_passed", "production_dataset_updated": False,
        "model_training_authorized": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
