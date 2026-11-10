import fcntl
import json
from pathlib import Path
import tempfile
import unittest
import os
import subprocess
import sys
from contextlib import ExitStack
from unittest.mock import Mock, patch
import pandas as pd

from data.pipelines import reconciliation_publication as publication
from data.pipelines.reconciliation_run import run
from data.pipelines.financial_history import atomic_json
from tests.reconciliation_fixtures import components, references, publish_fixture, complete_run
from data.pipelines.data_delivery import digest
from data.pipelines import stage_snapshots


def stage_fixture(processed, paths, tag):
    from tests.test_stage_model_inputs import StageModelInputTests
    finance=paths['financial'];pd.DataFrame({'fixture':[tag]}).to_parquet(finance/'accounts.parquet')
    report=json.loads((finance/'manifest.json').read_text())
    report['output_sha256']['accounts.parquet']=digest(finance/'accounts.parquet')
    atomic_json(finance/'manifest.json',report)
    dataset=paths['dataset'];report=json.loads((dataset/'manifest.json').read_text())
    report['inputs']['financial']=digest(finance/'manifest.json');atomic_json(dataset/'manifest.json',report)
    frame,audit=StageModelInputTests().fixture();inputs=stage_snapshots.build_stage_model_inputs(frame,audit)
    root=processed/'stage_snapshots'/tag;root.mkdir(parents=True)
    outputs={'stages.parquet':frame,'observations.parquet':audit,
             'milestones.parquet':pd.DataFrame({'fixture':[tag]}),'source_attempts.parquet':pd.DataFrame({'fixture':[tag]})}
    outputs.update({name+'_inputs.parquet':value for name,value in inputs.items()})
    for name,value in outputs.items():value.to_parquet(root/name,index=False)
    report={'version':tag,'scope':'all_connected_general_ipo','events':1,'policy':stage_snapshots.POLICY,
        'input_policy':stage_snapshots.INPUT_POLICY,'training_executed':False,'deployment_authorized':False,
        'input_profiles':{s:{'features':list(stage_snapshots.get_model_profile(s).feature_names),
            'critical_features':list(stage_snapshots.get_model_profile(s).critical_features)} for s in inputs},
        'inputs':{'builder':digest(stage_snapshots.__file__),'dataset':digest(dataset/'manifest.json'),
            'sources':digest(paths['sources']/'manifest.json'),'accounts':digest(finance/'accounts.parquet'),
            'dependency_sha256':{key:digest(file) for key,file in stage_snapshots.stage_dependency_paths().items()},
            'parser':digest(Path(stage_snapshots.__file__).parents[1]/'collectors/dart_collector.py'),
            'financial_document_structure':stage_snapshots.financial_document_structure_contract()},
        'stages':{s:{'rows':1,'ready_rows':1,'status_counts':{'historical_document_selected':1}} for s in inputs},
        'output_sha256':{name:digest(root/name) for name in outputs}}
    atomic_json(root/'manifest.json',report)
    return {**paths,'stages':root}


class CompleteCollectionReleaseTests(unittest.TestCase):
    def patch_phases(self, processed, paths, stage_effect=None):
        stack=ExitStack();refs=references(processed,paths)
        def source(*args,**kwargs):
            atomic_json(processed/publication.SOURCE,refs[publication.SOURCE])
            return {'population':1,'status_counts':{'official_sources_connected':1},'path':str(paths['sources'])}
        def dataset(*args,**kwargs):
            atomic_json(processed/publication.DATASET,refs[publication.DATASET]);return {'path':str(paths['dataset'])}
        for name in ['band_source_resolver.refresh_cached_band','float_source_resolver.refresh_cached_float','lockup_reconciliation.run']:
            stack.enter_context(patch('data.pipelines.'+name,return_value=None))
        stack.enter_context(patch('data.pipelines.reconciliation_run.collect_connections',side_effect=source))
        stack.enter_context(patch('data.pipelines.reconciliation_run.reconcile_financials',return_value={'path':str(paths['financial'])}))
        stack.enter_context(patch('data.pipelines.reconciliation_run.build_dataset',side_effect=dataset))
        stack.enter_context(patch('data.pipelines.reconciliation_run.build_stages',side_effect=stage_effect,
                                 return_value={'path':str(paths['stages'])}))
        return stack

    def test_full_default_run_and_replay_pin_both_stages_in_one_committed_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();processed=root/'processed'
            paths=stage_fixture(processed,components(processed,'new'),'new')
            with self.patch_phases(processed,paths):
                first=run(root/'raw',processed,dart=Mock())
                saved={name:(processed/name).read_bytes() for name in publication.WORKING_ALIASES}
                second=run(root/'raw',processed,dart=Mock())
            self.assertEqual(first['version'],second['version'])
            self.assertEqual(saved,{name:(processed/name).read_bytes() for name in saved})
            release=publication.committed_reconciliation(processed,verify_outputs=True)
            self.assertEqual(release['paths'],paths)
            self.assertEqual(release['run']['scope'],publication.COMPLETE_SCOPE)
            self.assertEqual(release['references'][publication.STAGES],references(processed,paths)[publication.STAGES])
            atomic_json(processed/publication.STAGES,{'manifest':'uncommitted','sha256':'bad'})
            for stage in ('pre_demand','post_demand'):
                frame,_,_=stage_snapshots.load_stage_model_inputs(processed,stage)
                self.assertEqual(len(frame),1)
                self.assertEqual(frame.attrs['stage_snapshot_reference'],release['references'][publication.STAGES])
            with self.assertRaisesRegex(RuntimeError,'reference_not_in_complete_release'):
                stage_snapshots.load_stage_model_inputs(processed,'pre_demand',reference={'manifest':'other','sha256':'bad'})

    def test_bad_stage_parent_population_or_scope_never_replaces_previous_release(self):
        for change in ('parent','population','scope','missing_output'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as directory:
                root=Path(directory).resolve();processed=root/'processed'
                old=stage_fixture(processed,components(processed,'old'),'old')
                new=stage_fixture(processed,components(processed,'new'),'new');publish_fixture(processed,old)
                before={n:(processed/n).read_bytes() for n in publication.WORKING_ALIASES}
                report=json.loads((new['stages']/'manifest.json').read_text())
                if change=='parent':report['inputs']['dataset']='other'
                elif change=='population':report['events']=2
                elif change=='scope':report['scope']='limited_sample'
                else:report['output_sha256'].pop('milestones.parquet')
                atomic_json(new['stages']/'manifest.json',report)
                with self.patch_phases(processed,new),self.assertRaises(RuntimeError):
                    run(root/'raw',processed,dart=Mock())
                self.assertEqual(before,{n:(processed/n).read_bytes() for n in before})

    def test_complete_release_cannot_be_downgraded_by_base_only_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();processed=root/'processed'
            old=stage_fixture(processed,components(processed,'old'),'old')
            new=stage_fixture(processed,components(processed,'new'),'new');publish_fixture(processed,old)
            before=(processed/publication.RUN).read_bytes()
            with self.patch_phases(processed,new),self.assertRaisesRegex(RuntimeError,'cannot_downgrade'):
                run(root/'raw',processed,dart=Mock(),include_stages=False)
            self.assertEqual(before,(processed/publication.RUN).read_bytes())

    def test_post_commit_stage_alias_failure_still_serves_new_inputs_and_recovers(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();processed=root/'processed'
            old=stage_fixture(processed,components(processed,'old'),'old')
            new=stage_fixture(processed,components(processed,'new'),'new');publish_fixture(processed,old)
            direct=publication._direct_json
            def fail(path,value):
                if Path(path)==processed/publication.STAGES:raise OSError('stage_alias_interrupted')
                return direct(path,value)
            with self.patch_phases(processed,new),patch.object(publication,'_direct_json',side_effect=fail), \
                    self.assertRaisesRegex(OSError,'stage_alias_interrupted'):
                run(root/'raw',processed,dart=Mock())
            self.assertEqual(publication.committed_reconciliation(processed)['paths'],new)
            for stage in ('pre_demand','post_demand'):
                frame,_,_=stage_snapshots.load_stage_model_inputs(processed,stage)
                self.assertEqual(frame.attrs['stage_snapshot_reference'],references(processed,new)[publication.STAGES])
            self.assertEqual(publication.recover_publication(processed),'complete')
            self.assertEqual(json.loads((processed/publication.STAGES).read_text()),references(processed,new)[publication.STAGES])

    def test_wrong_saved_candidate_flag_is_rejected_before_atomic_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();processed=root/'processed'
            old=stage_fixture(processed,components(processed,'old'),'old')
            new=stage_fixture(processed,components(processed,'new'),'new');publish_fixture(processed,old)
            before={name:(processed/name).read_bytes() for name in publication.WORKING_ALIASES}
            file=new['stages']/'pre_demand_inputs.parquet'
            table=pd.read_parquet(file);table['stage_model_candidate']=False;table.to_parquet(file,index=False)
            report=json.loads((new['stages']/'manifest.json').read_text());report['output_sha256'][file.name]=digest(file)
            atomic_json(new['stages']/'manifest.json',report)
            with self.assertRaisesRegex(RuntimeError,'materialization_or_candidate_changed'), \
                    publication.ReconciliationPublication(processed) as transaction:
                for name,value in references(processed,new).items():atomic_json(processed/name,value)
                atomic_json(processed/publication.RUN,complete_run(processed,new))
                transaction.commit()
            self.assertEqual(before,{name:(processed/name).read_bytes() for name in before})

    def test_real_process_death_keeps_a_readable_complete_stage_graph(self):
        program='''
import os,signal,sys
from pathlib import Path
from unittest.mock import patch
from data.pipelines import reconciliation_publication as p
from data.pipelines.financial_history import atomic_json
from tests.reconciliation_fixtures import references,complete_run
root=Path(sys.argv[1]);boundary=sys.argv[2]
paths={k:root/f/'new' for k,f in [('sources','historical_reconciliations'),
    ('financial','historical_financial_reconciliations'),('dataset','reconciled_datasets'),('stages','stage_snapshots')]}
with p.ReconciliationPublication(root) as transaction:
    for name,value in references(root,paths).items():atomic_json(root/name,value)
    atomic_json(root/p.RUN,complete_run(root,paths))
    if boundary=='before':os.kill(os.getpid(),signal.SIGKILL)
    direct=p._direct_json
    def interrupt(path,value):
        if boundary=='stage_mirror' and Path(path)==root/p.STAGES:os.kill(os.getpid(),signal.SIGKILL)
        direct(path,value)
        if boundary=='commit' and Path(path)==root/p.RUN:os.kill(os.getpid(),signal.SIGKILL)
    with patch.object(p,'_direct_json',side_effect=interrupt):transaction.commit()
'''
        with tempfile.TemporaryDirectory() as directory:
            processed=Path(directory).resolve()/'processed'
            old=stage_fixture(processed,components(processed,'old'),'old')
            new=stage_fixture(processed,components(processed,'new'),'new')
            environment={'PATH':os.defpath,'PYTHONPATH':str(Path(__file__).resolve().parents[1])}
            for boundary in ('before','commit','stage_mirror'):
                with self.subTest(boundary=boundary):
                    publish_fixture(processed,old)
                    result=subprocess.run([sys.executable,'-c',program,str(processed),boundary],
                        env=environment,capture_output=True,timeout=30)
                    self.assertEqual(result.returncode,-9,result.stderr.decode())
                    expected=old if boundary=='before' else new
                    self.assertEqual(publication.committed_reconciliation(processed)['paths'],expected)
                    for stage in ('pre_demand','post_demand'):
                        frame,_,_=stage_snapshots.load_stage_model_inputs(processed,stage)
                        self.assertEqual(frame.attrs['stage_snapshot_reference'],references(processed,expected)[publication.STAGES])
                    with (processed/'reconciliation.lock').open('a') as lock:
                        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        publication.recover_publication(processed)
                    self.assertEqual(json.loads((processed/publication.STAGES).read_text()),references(processed,expected)[publication.STAGES])

    def test_default_run_does_not_publish_sources_before_stage_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();processed=root/'processed'
            old=components(processed,'old');new=components(processed,'new')
            publish_fixture(processed,old)
            atomic_json(processed/'current_stage_snapshots.json',{'last':'complete'})
            names=(*publication.ALIASES,'current_stage_snapshots.json')
            before={name:(processed/name).read_bytes() for name in names}
            refs=references(processed,new)
            def source(*args,**kwargs):
                atomic_json(processed/publication.SOURCE,refs[publication.SOURCE])
                return {'population':1,'status_counts':{'official_sources_connected':1},'path':str(new['sources'])}
            def dataset(*args,**kwargs):
                atomic_json(processed/publication.DATASET,refs[publication.DATASET])
                return {'path':str(new['dataset'])}
            def fail(*args,**kwargs):
                self.assertEqual(before,{name:(processed/name).read_bytes() for name in names})
                self.assertEqual(publication.committed_reconciliation(processed)['paths'],old)
                candidate=publication.committed_reconciliation(processed,reference=kwargs['release_reference'])
                self.assertEqual(candidate['paths'],new)
                raise RuntimeError('injected_stage_failure')
            with patch('data.pipelines.band_source_resolver.refresh_cached_band',return_value=None), \
                    patch('data.pipelines.float_source_resolver.refresh_cached_float',return_value=None), \
                    patch('data.pipelines.lockup_reconciliation.run',return_value=None), \
                    patch('data.pipelines.reconciliation_run.collect_connections',side_effect=source), \
                    patch('data.pipelines.reconciliation_run.reconcile_financials',return_value={'path':str(new['financial'])}), \
                    patch('data.pipelines.reconciliation_run.build_dataset',side_effect=dataset), \
                    patch('data.pipelines.reconciliation_run.build_stages',side_effect=fail,create=True), \
                    self.assertRaisesRegex(RuntimeError,'injected_stage_failure'):
                run(root/'raw',processed,dart=Mock())
            self.assertEqual(before,{name:(processed/name).read_bytes() for name in names})

    def test_stage_writer_lock_prevents_partial_update_before_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);processed=root/'processed';processed.mkdir()
            with (processed/'stage_snapshots.lock').open('a') as lock, \
                    patch('data.pipelines.reconciliation_run.collect_connections') as collect:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                result=run(root/'raw',processed,dart=Mock())
            self.assertEqual(result['status'],'already_running')
            collect.assert_not_called()
            self.assertFalse((processed/publication.RUN).exists())


if __name__=='__main__':unittest.main()
