import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from data.pipelines.reconciliation_run import run
from data.pipelines.financial_history import atomic_json
from data.pipelines.reconciliation_publication import SOURCE, DATASET, RUN, committed_reconciliation
from tests.reconciliation_fixtures import components, references, publish_fixture


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
            root=Path(root).resolve()
            processed=root/'processed';paths=components(processed,'test')
            refs=references(processed,paths)
            def source(*args,**kwargs):
                atomic_json(processed/SOURCE,refs[SOURCE])
                return {'population':1,'status_counts':{'official_sources_connected':1},'path':str(paths['sources'])}
            def dataset(*args,**kwargs):
                atomic_json(processed/DATASET,refs[DATASET])
                return {'path':str(paths['dataset'])}
            with patch('data.pipelines.band_source_resolver.refresh_cached_band',return_value=None), \
                    patch('data.pipelines.float_source_resolver.refresh_cached_float',return_value=None), \
                    patch('data.pipelines.lockup_reconciliation.run',return_value=None), \
                    patch('data.pipelines.reconciliation_run.collect_connections',side_effect=source), \
                    patch('data.pipelines.reconciliation_run.reconcile_financials',return_value={'path':str(paths['financial'])}), \
                    patch('data.pipelines.reconciliation_run.build_dataset',side_effect=dataset) as build:
                first=run(root/'raw',root/'processed',dart=Mock(),include_stages=False)
                pointer=(root/'processed/current_reconciliation_run.json').read_bytes()
                second=run(root/'raw',root/'processed',dart=Mock(),include_stages=False)
            self.assertEqual(first['version'],second['version'])
            self.assertEqual(pointer,(root/'processed/current_reconciliation_run.json').read_bytes())
            self.assertEqual(build.call_args.args[2],str(paths['financial']))

    def test_phase_failures_keep_every_previously_complete_reference(self):
        for phase in ('sources','financials','dataset','source_hold','financial_hold'):
            with self.subTest(phase=phase),tempfile.TemporaryDirectory() as root:
                root=Path(root).resolve();processed=root/'processed'
                old=components(processed,'old');new=components(processed,'new')
                publish_fixture(processed,old)
                before={name:(processed/name).read_bytes() for name in (SOURCE,DATASET,RUN)}
                refs=references(processed,new)
                def source(*args,**kwargs):
                    atomic_json(processed/SOURCE,refs[SOURCE])
                    self.assertEqual(committed_reconciliation(processed)['paths'],old)
                    if phase=='sources':raise RuntimeError('injected_phase_failure')
                    return {'population':1,'status_counts':{'review_required':1} if phase=='source_hold'
                            else {'official_sources_connected':1},'path':str(new['sources'])}
                def financial(*args,**kwargs):
                    if phase=='financials':raise RuntimeError('injected_phase_failure')
                    return {'path':str(new['financial']),
                            'missing_categories':{'source_access_failed':1} if phase=='financial_hold' else {}}
                def dataset(*args,**kwargs):
                    atomic_json(processed/DATASET,refs[DATASET])
                    if phase=='dataset':raise RuntimeError('injected_phase_failure')
                    return {'path':str(new['dataset'])}
                with patch('data.pipelines.band_source_resolver.refresh_cached_band',return_value=None), \
                        patch('data.pipelines.float_source_resolver.refresh_cached_float',return_value=None), \
                        patch('data.pipelines.lockup_reconciliation.run',return_value=None), \
                        patch('data.pipelines.reconciliation_run.collect_connections',side_effect=source), \
                        patch('data.pipelines.reconciliation_run.reconcile_financials',side_effect=financial), \
                        patch('data.pipelines.reconciliation_run.build_dataset',side_effect=dataset):
                    if phase.endswith('_hold'):
                        self.assertFalse(run(root/'raw',processed,dart=Mock())['final_pointer_replaced'])
                    else:
                        with self.assertRaisesRegex(RuntimeError,'injected_phase_failure'):
                            run(root/'raw',processed,dart=Mock())
                self.assertEqual({name:(processed/name).read_bytes() for name in before},before)
                self.assertEqual(committed_reconciliation(processed)['paths'],old)

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
