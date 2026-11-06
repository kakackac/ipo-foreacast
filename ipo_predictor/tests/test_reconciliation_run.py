import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from data.pipelines.reconciliation_run import run


class ReconciliationRunTests(unittest.TestCase):
    def test_financial_access_failure_cannot_replace_complete_dataset(self):
        with tempfile.TemporaryDirectory() as root, \
                patch('data.pipelines.reconciliation_run.collect_connections',return_value={
                    'population':1,'status_counts':{'official_sources_connected':1},'path':'source'}), \
                patch('data.pipelines.reconciliation_run.reconcile_financials',return_value={
                    'path':'failed-finance','missing_categories':{'source_access_failed':1}}), \
                patch('data.pipelines.reconciliation_run.build_dataset') as build:
            result=run(Path(root)/'raw',Path(root)/'processed',dart=Mock())
            self.assertEqual(result['status'],'financial_sources_incomplete')
            build.assert_not_called()

    def test_identical_complete_runs_keep_one_version_and_pointer(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            for name in ('source','financial','dataset'):
                (root/name).mkdir();(root/name/'manifest.json').write_text('{}')
            with patch('data.pipelines.reconciliation_run.collect_connections',return_value={
                    'population':1,'status_counts':{'official_sources_connected':1},'path':str(root/'source')}), \
                    patch('data.pipelines.reconciliation_run.reconcile_financials',return_value={'path':str(root/'financial')}), \
                    patch('data.pipelines.reconciliation_run.build_dataset',return_value={'path':str(root/'dataset')}) as build:
                first=run(root/'raw',root/'processed',dart=Mock())
                pointer=(root/'processed/current_reconciliation_run.json').read_bytes()
                second=run(root/'raw',root/'processed',dart=Mock())
            self.assertEqual(first['version'],second['version'])
            self.assertEqual(pointer,(root/'processed/current_reconciliation_run.json').read_bytes())
            self.assertEqual(build.call_args.args[2],str(root/'financial'))

    def test_holds_do_not_start_finance_or_publish(self):
        with tempfile.TemporaryDirectory() as root, \
                patch('data.pipelines.reconciliation_run.collect_connections',return_value={
                    'population':2,'status_counts':{'official_sources_connected':1,'review_required':1},'path':'held'}), \
                patch('data.pipelines.reconciliation_run.reconcile_financials') as financial:
            result=run(Path(root)/'raw',Path(root)/'processed',dart=Mock())
            self.assertEqual(result['status'],'source_holds_require_resolution')
            financial.assert_not_called()
            self.assertFalse((Path(root)/'processed/current_reconciliation_run.json').exists())

    def test_failure_keeps_last_good_run_and_restores_transport(self):
        with tempfile.TemporaryDirectory() as root, \
                patch('data.pipelines.reconciliation_run.collect_connections',return_value={
                    'population':1,'status_counts':{'official_sources_connected':1},'path':'source'}), \
                patch('data.pipelines.reconciliation_run.reconcile_financials',side_effect=RuntimeError('interrupted')):
            processed=Path(root)/'processed';processed.mkdir()
            pointer=processed/'current_reconciliation_run.json';pointer.write_text('{"last":"good"}')
            dart=Mock();request=dart.session.request
            with self.assertRaisesRegex(RuntimeError,'interrupted'):
                run(Path(root)/'raw',processed,dart=dart,cached_only=True)
            self.assertEqual(json.loads(pointer.read_text()),{'last':'good'})
            self.assertIs(dart.session.request,request)
