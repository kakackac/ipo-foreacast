import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd

from data.pipelines.data_delivery import digest
from data.pipelines.historical_financial_reconciliation import unresolved_event_ids, connected_lineage


class FinancialReconciliationTargetTests(unittest.TestCase):
    def test_latest_holds_include_previously_approved_rows_instead_of_legacy_subset(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);f=p/'historical_financial_reconciliations/latest';f.mkdir(parents=True)
            pd.DataFrame([{'event_id':'newly_held','missing_category':'accounting_basis_unverified'},
                          {'event_id':'recovered','missing_category':'not_missing'}]).to_parquet(f/'resolutions.parquet',index=False)
            (f/'manifest.json').write_text('{}')
            r=p/'reconciliation_runs/latest';r.mkdir(parents=True)
            (r/'manifest.json').write_text(json.dumps({'financial':str(f),'input_sha256':{'financial':digest(f/'manifest.json')}}))
            (p/'current_reconciliation_run.json').write_text(json.dumps({'manifest':'reconciliation_runs/latest/manifest.json','sha256':digest(r/'manifest.json')}))
            fallback=pd.DataFrame([{'event_id':'recovered','missing_category':'old_hold'}])
            with patch('data.pipelines.historical_financial_reconciliation.checked_artifact'):
                held,known,previous=unresolved_event_ids(p,fallback)
            self.assertEqual(held,{'newly_held'});self.assertEqual(known,{'newly_held','recovered'})
            self.assertEqual(previous.set_index('event_id').loc['newly_held','missing_category'],'accounting_basis_unverified')

    def test_targeted_financial_path_uses_verified_family_dates_and_corporate_identity(self):
        proof={'corp_code':'00000001','family':[{'rcept_no':'20150101000001','rcept_dt':'2015-02-03','report_nm':'증권신고서(지분증권)'}]}
        c=pd.DataFrame([{'event_id':'e','status':'official_sources_connected','record':json.dumps({'corp_code':'00000001'}),'proof':json.dumps(proof)}])
        result=connected_lineage(c)
        self.assertEqual(result.iloc[0].rcept_dt,'2015-02-03')
        c['record']=json.dumps({'corp_code':'00000002'})
        with self.assertRaisesRegex(RuntimeError,'corp_code_conflict'):connected_lineage(c)
