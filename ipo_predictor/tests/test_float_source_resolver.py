import unittest
import tempfile
import json
import subprocess
import hashlib
from pathlib import Path
from unittest.mock import patch,Mock
import pandas as pd
from data.pipelines.float_source_resolver import resolve_public_float,refresh_cached_float
from data.pipelines.data_delivery import digest
from data.pipelines.financial_history import atomic_json
from data.pipelines.historical_reconciliation import connection_contract


class FloatSourceResolverTests(unittest.TestCase):
    def test_field_refresh_preserves_other_values_and_reuses_completed_version(self):
        repo=Path(__file__).parents[2]
        previous=connection_contract()
        for field,path in [('float','float_source_resolver.py'),('float_tables','../collectors/float_tables.py')]:
            source='ipo_predictor/data/pipelines/'+path if field=='float' else 'ipo_predictor/data/collectors/float_tables.py'
            previous[field]=hashlib.sha256(subprocess.check_output(['git','show','82839b0:'+source],cwd=repo)).hexdigest()
        with tempfile.TemporaryDirectory() as root:
            root=Path(root).resolve();processed=root/'processed';parent=processed/'historical_reconciliations'/'parent';parent.mkdir(parents=True)
            metadata={'rcept_no':'20240201000001','rcept_dt':'2024-02-01'}
            record={'corp_code':'00000001','rcept_no':metadata['rcept_no'],'listing_date':'2024-03-01','offering_price':10000,'institutional_demand_ratio':123}
            proof={'corp_code':'00000001','sources':[]}
            frame=pd.DataFrame([{'event_id':'e','corp_name':'Issuer','status':'official_sources_connected','record':json.dumps(record),'proof':json.dumps(proof)}])
            frame.to_parquet(parent/'connection_results.parquet',index=False)
            pd.DataFrame([{'event_id':'e'}]).to_parquet(parent/'events.parquet',index=False)
            atomic_json(parent/'manifest.json',{'inputs':{'inputs':{'pinned':'data'},'contract':previous},
                'output_sha256':{name:digest(parent/name) for name in ['connection_results.parquet','events.parquet']}})
            pointer=processed/'current_historical_source_connections.json';atomic_json(pointer,{'manifest':str((parent/'manifest.json').relative_to(processed)),'sha256':digest(parent/'manifest.json')})
            inputs={'input_sha256':{'pinned':'data'},'events':pd.DataFrame([{'event_id':'e','event_class':'general_ipo'}])}
            dart=Mock();dart._parse_offering_html.return_value={'offering_price':10000,'public_float_parse_method':'disclosed_public_float_ratio_direct_context','public_float_ratio_disclosed':.4}
            with patch('data.pipelines.historical_reconciliation.load_inputs',return_value=inputs),patch('data.pipelines.lockup_reconciliation.documents',return_value=[(metadata,{'source_url':'official'},'body')]):
                result=refresh_cached_float(root/'raw',processed,dart)
                self.assertIsNotNone(result)
                updated=json.loads(pd.read_parquet(Path(result['path'])/'connection_results.parquet').iloc[0].record)
                self.assertEqual(updated['public_float_ratio_disclosed'],.4)
                self.assertEqual(updated['institutional_demand_ratio'],123)
                self.assertEqual(updated['offering_price'],10000)
                before=digest(pointer)
                self.assertIsNone(refresh_cached_float(root/'raw',processed,dart))
                self.assertEqual(digest(pointer),before);self.assertEqual(dart._parse_offering_html.call_count,1)

    def document(self, receipt, date, ratio=None, conflict=False, supplementary=False, price=22500):
        return ({"rcept_no": receipt, "rcept_dt": date, "supplementary": supplementary}, {
            "offering_price": price, "public_float_ratio_disclosed": ratio,
            "public_float_parse_method": "conflicting_float_ratios_review_required" if conflict
            else "disclosed_public_float_ratio_direct_context",
        })

    def test_latest_conflict_does_not_restore_old_value(self):
        documents = [self.document("20251114001150", "2025-11-14", .3523),
                     self.document("20251205000229", "2025-12-05", conflict=True)]
        result = resolve_public_float(documents, 22500, "2025-12-18")
        self.assertIsNone(result["public_float_ratio_disclosed"])
        self.assertEqual(result["public_float_resolution_status"], "latest_conflict_blocks_older_fallback")

    def test_later_price_linked_prospectus_resolves_conflict(self):
        documents = [self.document("20251205000229", "2025-12-05", conflict=True),
                     self.document("20251205000237", "2025-12-05", .3527, supplementary=True)]
        result = resolve_public_float(documents, 22500, "2025-12-18")
        self.assertEqual(result["public_float_ratio_disclosed"], .3527)
        self.assertEqual(result["public_float_rcept_no"], "20251205000237")

    def test_wrong_price_cannot_restore_old_value(self):
        documents = [self.document("20251114001150", "2025-11-14", .3523),
                     self.document("20251205000237", "2025-12-05", .3527,
                                   supplementary=True, price=17000)]
        self.assertIsNone(resolve_public_float(documents, 22500, "2025-12-18")["public_float_ratio_disclosed"])

    def test_listing_day_source_is_not_used(self):
        result = resolve_public_float([self.document("20251218000237", "2025-12-18", .3527)],
                                      22500, "2025-12-18")
        self.assertIsNone(result["public_float_ratio_disclosed"])
