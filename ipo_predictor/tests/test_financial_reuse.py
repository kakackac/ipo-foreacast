import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from data.pipelines.data_delivery import digest, checked_artifact
from data.pipelines.financial_history import atomic_json
from data.pipelines.historical_financial_reconciliation import reuse_financials, reuse_inputs
from data.pipelines.historical_financial_reconciliation import run
from tests.reconciliation_fixtures import components, publish_fixture


class FinancialReuseTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.processed=(Path(self.temporary.name)/'processed').resolve()
        self.paths=components(self.processed,'old')
        self.signature={'future_annual_cache_present':False,'projection':'unchanged','code':{'parser':'fixed'}}
        self.configure()

    def configure(self,eligible=True):
        report=json.loads((self.paths['financial']/'manifest.json').read_text())
        report.update(reuse_eligible=eligible,reuse_inputs=self.signature)
        atomic_json(self.paths['financial']/'manifest.json',report)
        dataset=json.loads((self.paths['dataset']/'manifest.json').read_text())
        dataset['inputs']['financial']=digest(self.paths['financial']/'manifest.json')
        atomic_json(self.paths['dataset']/'manifest.json',dataset)
        publish_fixture(self.processed,self.paths)

    def test_unchanged_source_reuses_the_exact_financial_version(self):
        result=reuse_financials(self.processed,self.signature,digest(self.paths['sources']/'manifest.json'))
        self.assertEqual(Path(result['path']),self.paths['financial'])
        self.assertTrue(result['reused']);self.assertEqual(result['reused_events'],1)

    def test_nonfinancial_source_change_rebinds_without_changing_financial_files(self):
        result=reuse_financials(self.processed,self.signature,'new-source-manifest-hash')
        root=Path(result['path']);report=checked_artifact(root)
        self.assertNotEqual(root,self.paths['financial'])
        self.assertEqual(report['connection_manifest'],'new-source-manifest-hash')
        for name in report['output_sha256']:
            self.assertEqual((root/name).read_bytes(),(self.paths['financial']/name).read_bytes())
        again=reuse_financials(self.processed,self.signature,'new-source-manifest-hash')
        self.assertEqual(result['version'],again['version'])

    def test_changed_projection_or_code_cannot_reuse(self):
        for changed in [{**self.signature,'projection':'different'},
                        {**self.signature,'code':{'parser':'new'}}]:
            self.assertIsNone(reuse_financials(self.processed,changed,'source'))

    def test_failure_resolutions_and_legacy_versions_are_not_reusable(self):
        self.configure(eligible=False)
        self.assertIsNone(reuse_financials(self.processed,self.signature,'source'))
        self.configure()
        report=json.loads((self.paths['financial']/'manifest.json').read_text())
        report.pop('reuse_inputs');atomic_json(self.paths['financial']/'manifest.json',report)
        dataset=json.loads((self.paths['dataset']/'manifest.json').read_text())
        dataset['inputs']['financial']=digest(self.paths['financial']/'manifest.json')
        atomic_json(self.paths['dataset']/'manifest.json',dataset);publish_fixture(self.processed,self.paths)
        self.assertIsNone(reuse_financials(self.processed,self.signature,'source'))

    def test_future_annual_cache_cannot_be_frozen_as_a_reusable_result(self):
        self.signature={**self.signature,'future_annual_cache_present':True};self.configure()
        self.assertIsNone(reuse_financials(self.processed,self.signature,'source'))

    def test_corrupt_financial_outputs_raise_instead_of_becoming_a_cache_hit(self):
        (self.paths['financial']/'payload.json').write_text('corrupt')
        with self.assertRaisesRegex(RuntimeError,'artifact_integrity_failed'):
            reuse_financials(self.processed,self.signature,'source')

    def projection(self,raw,*,corp='00000001',day='2024-03-01',receipt='20240201000001',floating=.3):
        targets=pd.DataFrame([{'event_id':'event','corp_name':'Issuer','listing_date':day,'float_share_ratio':floating}])
        records=pd.DataFrame([{'event_id':'event','corp_code':corp,'float_share_ratio':floating}]).set_index('event_id')
        lineage=pd.DataFrame([{'event_id':'event','corp_code':corp,'rcept_no':receipt,'rcept_dt':'2024-02-01'}])
        return reuse_inputs(raw,targets,records,lineage,{'base':'stable'},{'parser':'stable'})

    def test_projection_ignores_float_but_tracks_issuer_cutoff_and_lineage(self):
        raw=Path(self.temporary.name)/'raw'
        baseline=self.projection(raw)
        self.assertEqual(baseline,self.projection(raw,floating=.8))
        for change in [{'corp':'00000002'},{'day':'2024-03-02'},{'receipt':'20240202000001'}]:
            self.assertNotEqual(baseline,self.projection(raw,**change))

    def test_raw_cache_addition_edit_and_removal_change_the_fingerprint(self):
        raw=Path(self.temporary.name)/'raw';baseline=self.projection(raw)
        path=raw/'dart_connection_sources'/'source.json';atomic_json(path,{'html':'original'})
        added=self.projection(raw);self.assertNotEqual(added,baseline)
        atomic_json(path,{'html':'corrected'});self.assertNotEqual(added,self.projection(raw))
        path.unlink();self.assertEqual(baseline,self.projection(raw))

    def test_future_collected_at_is_an_explicit_reuse_guard(self):
        raw=Path(self.temporary.name)/'raw'
        atomic_json(raw/'dart_financial_sources'/'annual.json',{
            'collected_at':(pd.Timestamp.now(tz='UTC')+pd.Timedelta(days=1)).isoformat()})
        self.assertTrue(self.projection(raw)['future_annual_cache_present'])

    def test_real_runner_pins_approval_contract_and_runtime_before_cache_reuse(self):
        source=self.paths['sources'];base=self.processed/'base';base.mkdir()
        pd.DataFrame([{'event_id':'event','missing_category':'not_missing'}]).to_parquet(base/'source_resolutions.parquet')
        pd.DataFrame([{'event_id':'event','status':'official_sources_connected',
            'record':json.dumps({'corp_code':'00000001'}),
            'proof':json.dumps({'corp_code':'00000001','family':[{'rcept_no':'20240201000001',
                'rcept_dt':'2024-02-01','report_nm':'투자설명서'}]})}]).to_parquet(source/'connection_results.parquet')
        report=json.loads((source/'manifest.json').read_text())
        report['output_sha256']['connection_results.parquet']=digest(source/'connection_results.parquet')
        atomic_json(source/'manifest.json',report)
        atomic_json(self.processed/'current_historical_source_connections.json',{
            'manifest':str((source/'manifest.json').relative_to(self.processed)),'sha256':digest(source/'manifest.json')})
        inputs={'base':base,'input_sha256':{'fixture':'old'},
            'events':pd.DataFrame([{'event_id':'event','corp_name':'Issuer','listing_date':'2024-03-01','event_class':'general_ipo'}]),
            'records':pd.DataFrame([{'event_id':'event','corp_code':'00000001'}])}
        def inspect(processed,signature,connection):
            self.assertEqual(signature['code']['source_contracts'],digest(Path(__file__).parents[1]/'features/source_contracts.py'))
            self.assertEqual(signature['code']['pandas_version'],pd.__version__)
            return {'path':'cache-hit','version':'cached','reused_events':1}
        with patch('data.pipelines.historical_financial_reconciliation.load_inputs',return_value=inputs), \
                patch('data.pipelines.historical_financial_reconciliation.reuse_financials',side_effect=inspect), \
                patch('data.pipelines.historical_financial_reconciliation.collect_disclosure_financials') as collect:
            result=run(Path(self.temporary.name)/'raw',self.processed)
        self.assertEqual(result['path'],'cache-hit');collect.assert_not_called()
