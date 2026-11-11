"""One locked ingestion-to-ledger run; no model or deployment side effects."""
import fcntl
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path

from data.collectors.dart_collector import DARTCollector
from data.pipelines.data_delivery import digest
from data.pipelines.financial_history import atomic_json
from data.pipelines.historical_reconciliation import collect_connections
from data.pipelines.historical_financial_reconciliation import run as reconcile_financials
from data.pipelines.reconciled_dataset import run as build_dataset
from data.pipelines.reconciliation_publication import ReconciliationPublication, BASE_SCOPE, COMPLETE_SCOPE, STAGES


def build_stages(raw, processed, *, cached_only, dart, release_reference):
    from data.pipelines.stage_snapshots import _run
    return _run(raw,processed,cached_only=cached_only,event_limit=None,event_ids=None,
                dart=dart,release_reference=release_reference)


def run(raw_dir, processed_dir, *, cached_only=False, dart=None, include_stages=True):
    raw, processed = Path(raw_dir).resolve(), Path(processed_dir).resolve()
    processed.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        try:
            for name in ('reconciliation.lock','stage_snapshots.lock') if include_stages else ('reconciliation.lock',):
                lock=stack.enter_context((processed/name).open('a'))
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {'status':'already_running', 'training_executed':False}
        dart = dart or DARTCollector()
        original_request = dart.session.request
        def deny_network(*args, **kwargs):
            raise RuntimeError('reconciliation_cached_only_source_missing')
        if cached_only:
            dart.session.request = deny_network
        phase = 'sources'
        publication = None
        try:
            # A field-scoped migration may reuse the unchanged, verified IPO
            # identity and price graph instead of re-collecting every source.
            from data.pipelines.lockup_reconciliation import run as repair_lockup
            from data.pipelines.float_source_resolver import refresh_cached_float
            from data.pipelines.band_source_resolver import refresh_cached_band
            with ReconciliationPublication(processed) as publication:
                refresh_cached_band(raw, processed, dart=dart)
                refresh_cached_float(raw, processed, dart=dart)
                repair_lockup(raw, processed, dart=dart)
                sources = collect_connections(raw, processed, dart=dart, cached_only=cached_only)
                if sources['status_counts'] != {'official_sources_connected':sources['population']}:
                    atomic_json(processed/'reconciliation_last_attempt.json', {
                        'status':'source_holds_require_resolution', 'phase':phase, 'training_executed':False})
                    return {'status':'source_holds_require_resolution', 'sources':sources['path'],
                            'training_executed':False, 'final_pointer_replaced':False}
                phase = 'financials'
                financial = reconcile_financials(raw, processed, dart=dart)
                if financial.get('missing_categories',{}).get('source_access_failed',0):
                    atomic_json(processed/'reconciliation_last_attempt.json', {
                        'status':'financial_sources_incomplete', 'phase':phase, 'training_executed':False})
                    return {'status':'financial_sources_incomplete', 'sources':sources['path'],
                            'financial':financial['path'], 'final_pointer_replaced':False, 'training_executed':False}
                phase = 'dataset'
                dataset = build_dataset(raw, processed, financial['path'])
                identity = {'sources':digest(Path(sources['path'])/'manifest.json'),
                            'financial':digest(Path(financial['path'])/'manifest.json'),
                            'dataset':digest(Path(dataset['path'])/'manifest.json'),
                            'driver':digest(__file__),
                            'publication':digest(Path(__file__).with_name('reconciliation_publication.py'))}
                def prepare(stage_path=None):
                    version=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
                    destination=processed/'reconciliation_runs'/version
                    destination.mkdir(parents=True,exist_ok=True);manifest=destination/'manifest.json'
                    expected={'status':'complete_with_recorded_feature_holds','version':version,
                        'input_sha256':dict(identity),'sources':sources['path'],'financial':financial['path'],
                        'dataset':dataset['path'],'training_executed':False,'deployment_authorized':False,
                        'scope':COMPLETE_SCOPE if stage_path is not None else BASE_SCOPE}
                    if stage_path is not None:expected['stages']=stage_path
                    if not manifest.exists():atomic_json(manifest,expected)
                    if json.loads(manifest.read_text())!=expected:
                        raise RuntimeError('reconciliation_run_manifest_integrity_failed')
                    pointer={'manifest':str(manifest.relative_to(processed)),'sha256':digest(manifest)}
                    atomic_json(processed/'current_reconciliation_run.json',pointer)
                    return destination,manifest,version,pointer
                destination,manifest,version,base_reference=prepare()
                if include_stages:
                    phase='stages'
                    stages=build_stages(raw,processed,cached_only=cached_only,dart=dart,
                                        release_reference=base_reference)
                    stage_path=Path(stages['path'])
                    stage_reference={'manifest':str((stage_path/'manifest.json').relative_to(processed)),
                                     'sha256':digest(stage_path/'manifest.json')}
                    atomic_json(processed/STAGES,stage_reference)
                    identity['stages']=stage_reference['sha256']
                    destination,manifest,version,_=prepare(str(stage_path))
                phase = 'publication'
                publication.commit()
            atomic_json(processed/'reconciliation_last_attempt.json', {
                'status':'complete', 'version':version, 'training_executed':False})
            return {**json.loads(manifest.read_text()), 'path':str(destination)}
        except BaseException as exc:
            atomic_json(processed/'reconciliation_last_attempt.json', {
                'status':'interrupted_or_failed', 'phase':phase, 'error_type':type(exc).__name__,
                    'training_executed':False,
                    'committed_before_interruption':bool(publication and publication.committed),
                    'last_success_pointer_not_replaced_by_failure':not bool(publication and publication.committed)})
            raise
        finally:
            dart.session.request = original_request
