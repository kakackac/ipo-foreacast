import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import pandas as pd

from data.pipelines.data_delivery import digest
from data.pipelines.financial_history import atomic_json
from data.pipelines import reconciliation_publication as publication
from data.pipelines.reconciliation_run import run
from tests.reconciliation_fixtures import components, references, complete_run, publish_fixture


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.processed = (Path(self.temporary.name) / 'processed').resolve()
        self.old = components(self.processed, 'old')
        self.new = components(self.processed, 'new')
        publish_fixture(self.processed, self.old)
        self.before = self.pointers()

    def pointers(self):
        return {name: (self.processed / name).read_bytes() for name in publication.ALIASES}

    def stage(self, paths=None):
        paths = paths or self.new
        for name, value in references(self.processed, paths).items():
            atomic_json(self.processed / name, value)
        atomic_json(self.processed / publication.RUN, complete_run(self.processed, paths))

    def test_candidates_are_private_and_producer_reads_see_their_own_source(self):
        from data.pipelines.reconciled_dataset import pin_connections
        with publication.ReconciliationPublication(self.processed):
            self.stage()
            self.assertEqual(self.pointers(), self.before)
            self.assertEqual(pin_connections(self.processed)[0], self.new['sources'])
            self.assertEqual(publication.committed_reconciliation(self.processed)['paths'], self.old)
        self.assertEqual(self.pointers(), self.before)

    def test_abort_restores_context_and_preserves_all_normal_references(self):
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            with publication.ReconciliationPublication(self.processed):
                self.stage()
                raise RuntimeError('interrupted')
        self.assertEqual(self.pointers(), self.before)
        self.assertEqual(publication.reference_path(self.processed / publication.SOURCE), self.processed / publication.SOURCE)
        self.assertEqual(json.loads((self.processed / publication.JOURNAL).read_text())['state'], 'aborted_before_commit')

    def test_atomic_commit_and_identical_replay_keep_one_release(self):
        for _ in range(2):
            with publication.ReconciliationPublication(self.processed) as transaction:
                self.stage()
                transaction.commit()
            result = publication.committed_reconciliation(self.processed, verify_outputs=True)
            self.assertEqual(result['paths'], self.new)
            self.assertEqual(result['references'], {name: json.loads(value) for name, value in self.pointers().items()})
            if _ == 0:
                first = self.pointers()
            else:
                self.assertEqual(self.pointers(), first)

    def test_committed_reader_does_not_consume_independent_legacy_aliases(self):
        for name in (publication.SOURCE, publication.DATASET):
            atomic_json(self.processed / name, {'manifest': 'uncommitted/or/malformed', 'sha256': 'bad'})
        result = publication.committed_reconciliation(self.processed)
        self.assertEqual(result['paths'], self.old)
        self.assertEqual(result['pointer_sha256'], digest(self.processed / publication.RUN))

    def test_stage_entry_pins_source_from_complete_run_not_uncommitted_alias(self):
        from data.pipelines import stage_snapshots
        atomic_json(self.processed / publication.SOURCE, references(self.processed, self.new)[publication.SOURCE])
        expected = references(self.processed, self.old)[publication.SOURCE]
        def inspect(processed, pointer):
            self.assertEqual(pointer, expected)
            raise RuntimeError('complete_run_pinned')
        with patch('data.pipelines.stage_snapshots.pin_connections', side_effect=inspect), \
                self.assertRaisesRegex(RuntimeError, 'complete_run_pinned'):
            stage_snapshots._run(Path(self.temporary.name) / 'raw', self.processed,
                                 cached_only=True, event_limit=None, event_ids=None, dart=Mock())

    def test_actual_stage_checkpoint_pins_source_approval_contract(self):
        from data.pipelines import stage_snapshots
        source,financial,dataset=(self.old[key] for key in ('sources','financial','dataset'))
        pd.DataFrame([{'event_id':'event','record':json.dumps({'corp_code':'00000001'}),'proof':'{}'}]).to_parquet(source/'connection_results.parquet')
        report=json.loads((source/'manifest.json').read_text())
        report['output_sha256']['connection_results.parquet']=digest(source/'connection_results.parquet')
        atomic_json(source/'manifest.json',report)
        pd.DataFrame(columns=['event_id']).to_parquet(financial/'accounts.parquet')
        report=json.loads((financial/'manifest.json').read_text())
        base=Path('ipo_predictor/data/pipelines')
        report.update(connection_manifest=digest(source/'manifest.json'),code={
            'parser':digest(base/'disclosure_financials.py'),'summary':digest(base/'financial_history.py'),
            'source_selection':digest(base/'financial_source_selection.py')})
        report['output_sha256']['accounts.parquet']=digest(financial/'accounts.parquet')
        atomic_json(financial/'manifest.json',report)
        pd.DataFrame([{'event_id':'event','offering_price':10000,'open_return_pct':10,
                       'close_return_pct':20,'listing_date':pd.Timestamp('2024-03-01')}]).to_parquet(dataset/'features_all.parquet')
        pd.DataFrame({'date':[pd.Timestamp('2024-02-01')],'close':[100]}).to_parquet(dataset/'kospi_index_snapshot.parquet')
        report=json.loads((dataset/'manifest.json').read_text())
        report['inputs'].update(sources=digest(source/'manifest.json'),financial=digest(financial/'manifest.json'))
        for name in ['features_all.parquet','kospi_index_snapshot.parquet']:
            report['output_sha256'][name]=digest(dataset/name)
        atomic_json(dataset/'manifest.json',report);publish_fixture(self.processed,self.old)
        def inspect(event,identity,*args):
            self.assertEqual(identity['dependency_sha256']['source_contracts'],
                             digest(Path('ipo_predictor/features/source_contracts.py')))
            raise RuntimeError('approval_contract_dependency_verified')
        with patch('data.pipelines.stage_snapshots.stage_checkpoint_identity',side_effect=inspect), \
                self.assertRaisesRegex(RuntimeError,'approval_contract_dependency_verified'):
            stage_snapshots._run(Path(self.temporary.name)/'raw',self.processed,
                                 cached_only=True,event_limit=None,event_ids=None,dart=Mock())

    def test_mixed_candidate_aliases_cannot_commit(self):
        with self.assertRaisesRegex(RuntimeError, 'candidate_aliases_not_connected'):
            with publication.ReconciliationPublication(self.processed) as transaction:
                self.stage()
                atomic_json(self.processed / publication.SOURCE, references(self.processed, self.old)[publication.SOURCE])
                transaction.commit()
        self.assertEqual(self.pointers(), self.before)

    def test_changed_output_cannot_commit(self):
        with self.assertRaisesRegex(RuntimeError, 'artifact_integrity_failed'):
            with publication.ReconciliationPublication(self.processed) as transaction:
                self.stage()
                (self.new['dataset'] / 'payload.json').write_text('changed')
                transaction.commit()
        self.assertEqual(self.pointers(), self.before)

    def test_failure_before_atomic_commit_keeps_old_run(self):
        direct = publication._direct_json
        def fail(path, value):
            if Path(path) == self.processed / publication.RUN:
                raise OSError('injected_before_commit')
            direct(path, value)
        with self.assertRaises(OSError), publication.ReconciliationPublication(self.processed) as transaction:
            self.stage()
            with patch.object(publication, '_direct_json', side_effect=fail):
                transaction.commit()
        self.assertFalse(transaction.committed)
        self.assertEqual(self.pointers(), self.before)

    def test_post_commit_alias_failure_keeps_new_complete_graph_and_recovers(self):
        direct = publication._direct_json
        def fail(path, value):
            if Path(path) == self.processed / publication.SOURCE:
                raise OSError('injected_alias_failure')
            direct(path, value)
        with self.assertRaises(OSError), publication.ReconciliationPublication(self.processed) as transaction:
            self.stage()
            with patch.object(publication, '_direct_json', side_effect=fail):
                transaction.commit()
        self.assertTrue(transaction.committed)
        self.assertEqual(publication.committed_reconciliation(self.processed)['paths'], self.new)
        self.assertEqual(publication.recover_publication(self.processed), 'complete')
        self.assertEqual(publication.committed_reconciliation(self.processed)['references'],
                         {name: json.loads(value) for name, value in self.pointers().items()})

    def test_live_writer_lock_prevents_recovery_or_new_work(self):
        journal = self.processed / publication.JOURNAL
        atomic_json(journal, {'state': 'preparing', 'live_writer': True})
        before = journal.read_bytes()
        with (self.processed / 'reconciliation.lock').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(run(Path(self.temporary.name) / 'raw', self.processed, dart=Mock())['status'], 'already_running')
        self.assertEqual(journal.read_bytes(), before)

    def test_real_process_death_recovers_each_commit_boundary(self):
        program = '''
import json,os,signal,sys
from pathlib import Path
from unittest.mock import patch
from data.pipelines import reconciliation_publication as p
from data.pipelines.financial_history import atomic_json
from tests.reconciliation_fixtures import references,complete_run
root=Path(sys.argv[1]);boundary=sys.argv[2]
paths={k:root/f/'new' for k,f in [('sources','historical_reconciliations'),('financial','historical_financial_reconciliations'),('dataset','reconciled_datasets')]}
with p.ReconciliationPublication(root) as transaction:
    for name,value in references(root,paths).items():atomic_json(root/name,value)
    atomic_json(root/p.RUN,complete_run(root,paths))
    if boundary=='before':os.kill(os.getpid(),signal.SIGKILL)
    direct=p._direct_json
    def interrupt(path,value):
        direct(path,value)
        if Path(path)==root/(p.RUN if boundary=='commit' else p.SOURCE):os.kill(os.getpid(),signal.SIGKILL)
    with patch.object(p,'_direct_json',side_effect=interrupt):transaction.commit()
'''
        environment = {'PATH': os.defpath, 'PYTHONPATH': str(Path(__file__).resolve().parents[1])}
        for boundary in ('before', 'commit', 'mirror'):
            with self.subTest(boundary=boundary):
                publish_fixture(self.processed, self.old)
                result = subprocess.run([sys.executable, '-c', program, str(self.processed), boundary],
                                        env=environment, capture_output=True, timeout=30)
                self.assertEqual(result.returncode, -9, result.stderr.decode())
                expected = self.old if boundary == 'before' else self.new
                self.assertEqual(publication.committed_reconciliation(self.processed)['paths'], expected)
                state = publication.recover_publication(self.processed)
                self.assertEqual(state, 'aborted_before_commit' if boundary == 'before' else 'complete')
                self.assertEqual(publication.committed_reconciliation(self.processed)['references'],
                                 {name: json.loads(value) for name, value in self.pointers().items()})

    def test_other_stores_and_regular_files_are_not_redirected(self):
        with publication.ReconciliationPublication(self.processed):
            for path in (self.processed / 'other.json', self.processed / 'nested' / publication.SOURCE):
                atomic_json(path, {'regular': True})
                self.assertEqual(json.loads(path.read_text()), {'regular': True})

    def test_nested_publications_are_rejected_without_leaking_context(self):
        with publication.ReconciliationPublication(self.processed), self.assertRaisesRegex(RuntimeError, 'nested'):
            with publication.ReconciliationPublication(self.processed):
                pass
        self.assertEqual(publication.reference_path(self.processed / publication.RUN), self.processed / publication.RUN)
