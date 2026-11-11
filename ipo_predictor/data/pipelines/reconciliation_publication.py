"""Keep candidate aliases private and commit one verified reconciliation release."""
from contextvars import ContextVar
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile


SOURCE = 'current_historical_source_connections.json'
DATASET = 'current_reconciled_data.json'
RUN = 'current_reconciliation_run.json'
STAGES = 'current_stage_snapshots.json'
ALIASES = (SOURCE, DATASET, RUN)
WORKING_ALIASES = (*ALIASES, STAGES)
BASE_SCOPE = 'stored_general_ipo_listing_eve_not_product_stage_or_live_scheduler'
COMPLETE_SCOPE = 'stored_general_ipo_pre_post_inputs_not_live_scheduler'
JOURNAL = 'reconciliation_publication.json'
_working = ContextVar('reconciliation_working_references', default=None)
logger = logging.getLogger(__name__)


def reference_path(path):
    path = Path(path)
    transaction = _working.get()
    if (transaction is not None and path.name in WORKING_ALIASES
            and path.parent.resolve() == transaction.processed):
        return transaction.directory / path.name
    return path


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _direct_json(path, value):
    from data.pipelines.financial_history import atomic_json
    token = _working.set(None)
    try:
        atomic_json(path, value)
        with Path(path).open('rb') as handle:
            os.fsync(handle.fileno())
        descriptor = os.open(Path(path).parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        _working.reset(token)


def committed_reconciliation(processed_dir, *, reference=None, verify_outputs=False):
    """Pin a complete graph using only the atomic run pointer, never its aliases."""
    processed = Path(processed_dir).resolve()
    payload = (processed / RUN).read_bytes() if reference is None else None
    pointer = reference if reference is not None else json.loads(payload)
    manifest = (processed / pointer['manifest']).resolve()
    if (not manifest.is_relative_to(processed / 'reconciliation_runs')
            or manifest.name != 'manifest.json' or _digest(manifest) != pointer['sha256']):
        raise RuntimeError('committed_reconciliation_pointer_invalid')
    run = json.loads(manifest.read_text())
    if (run.get('status') != 'complete_with_recorded_feature_holds'
            or run.get('version') != manifest.parent.name
            or run.get('scope') != (COMPLETE_SCOPE if 'stages' in run else BASE_SCOPE)
            or run.get('training_executed') is not False
            or run.get('deployment_authorized') is not False):
        raise RuntimeError('committed_reconciliation_status_invalid')
    folders = {'sources': 'historical_reconciliations',
               'financial': 'historical_financial_reconciliations', 'dataset': 'reconciled_datasets'}
    paths, reports = {}, {}
    for key, folder in folders.items():
        root = Path(run[key]).resolve()
        if (not root.is_relative_to(processed / folder)
                or _digest(root / 'manifest.json') != run['input_sha256'][key]):
            raise RuntimeError('committed_reconciliation_component_invalid')
        paths[key] = root
        reports[key] = json.loads((root / 'manifest.json').read_text())
        if verify_outputs:
            from data.pipelines.data_delivery import checked_artifact
            checked_artifact(root)
    source, financial, dataset = (reports[key] for key in folders)
    if (source['inputs']['inputs'] != financial['inputs']
            or financial['inputs'] != dataset['inputs']['inputs']
            or financial['connection_manifest'] != run['input_sha256']['sources']
            or dataset['inputs']['sources'] != run['input_sha256']['sources']
            or dataset['inputs']['financial'] != run['input_sha256']['financial']
            or financial['scope'] != 'all_connected_general_ipo_financial_sections'):
        raise RuntimeError('committed_reconciliation_components_not_connected')
    validation = dataset['validation']
    population = source.get('population', 0)
    if (not isinstance(population, int) or population <= 0
            or source.get('status_counts') != {'official_sources_connected': population}
            or financial.get('population') != population or financial.get('processed') != population
            or dataset.get('rows', 0) != validation.get('rows') or validation.get('rows', 0) < population
            or validation.get('observations', 0) <= 0):
        raise RuntimeError('committed_reconciliation_population_invalid')
    if (validation.get('status') != 'snapshot_valid_with_recorded_holds'
            or any(validation.get(key) != 0 for key in
                   ('time_violations', 'value_violations', 'non_general_model_observations'))):
        raise RuntimeError('committed_reconciliation_dataset_invalid')
    if 'stages' in run:
        root=Path(run['stages']).resolve()
        if (not root.is_relative_to(processed/'stage_snapshots')
                or _digest(root/'manifest.json')!=run['input_sha256'].get('stages')):
            raise RuntimeError('committed_reconciliation_stage_invalid')
        stage=json.loads((root/'manifest.json').read_text())
        from data.pipelines.stage_snapshots import INPUT_POLICY
        from features.model_profiles import get_model_profile
        profiles={name:{'features':list(get_model_profile(name).feature_names),
                        'critical_features':list(get_model_profile(name).critical_features)}
                  for name in ('pre_demand','post_demand')}
        required={'stages.parquet','observations.parquet','milestones.parquet','source_attempts.parquet',
                  'pre_demand_inputs.parquet','post_demand_inputs.parquet'}
        if (stage.get('scope')!='all_connected_general_ipo' or stage.get('events')!=population
                or stage.get('input_policy')!=INPUT_POLICY or stage.get('input_profiles')!=profiles
                or stage.get('version')!=root.name or stage.get('training_executed') is not False
                or stage.get('deployment_authorized') is not False
                or set(stage.get('output_sha256',{}))!=required
                or stage.get('inputs',{}).get('dataset')!=run['input_sha256']['dataset']
                or stage['inputs'].get('sources')!=run['input_sha256']['sources']
                or stage['inputs'].get('accounts')!=_digest(paths['financial']/'accounts.parquet')
                or set(stage.get('stages',{}))!={'pre_demand','post_demand'}
                or any(stage['stages'][name].get('rows')!=population
                       or not isinstance(stage['stages'][name].get('ready_rows'),int)
                       or not 0<=stage['stages'][name]['ready_rows']<=population for name in stage['stages'])):
            raise RuntimeError('committed_reconciliation_stage_not_connected')
        if verify_outputs:
            from data.pipelines.data_delivery import checked_artifact
            checked_artifact(root)
        paths['stages']=root;reports['stages']=stage
    references = {RUN: pointer}
    for alias, key in [(SOURCE, 'sources'), (DATASET, 'dataset')]:
        references[alias] = {'manifest': str((paths[key] / 'manifest.json').relative_to(processed)),
                             'sha256': run['input_sha256'][key]}
    references[DATASET].update(training_authorized=False, deployment_authorized=False)
    if 'stages' in paths:
        references[STAGES]={'manifest':str((paths['stages']/'manifest.json').relative_to(processed)),
                            'sha256':run['input_sha256']['stages']}
    return {'run': run, 'pointer': pointer, 'manifest': manifest, 'paths': paths,
            'pointer_sha256': hashlib.sha256(payload).hexdigest() if payload is not None else None,
            'references': references, 'reports': reports}


def _sync_aliases(processed, release):
    for name in (SOURCE, DATASET, STAGES):
        if name not in release['references']:continue
        path, value = processed / name, release['references'][name]
        if not path.exists() or json.loads(path.read_text()) != value:
            _direct_json(path, value)


def recover_publication(processed_dir):
    """Called only after acquiring the writer lock; an old journal is not liveness."""
    processed = Path(processed_dir).resolve()
    journal = processed / JOURNAL
    if not journal.exists():
        return None
    saved = json.loads(journal.read_text())
    if saved['state'] not in {'preparing', 'committed_pending_alias_sync'}:
        return saved['state']
    candidate = (processed / saved['candidate']).resolve()
    if not candidate.is_relative_to(processed / 'reconciliation_publications'):
        raise RuntimeError('reconciliation_publication_candidate_outside_store')
    root_pointer, candidate_pointer = processed / RUN, candidate / RUN
    current = json.loads(root_pointer.read_text()) if root_pointer.exists() else None
    proposed = json.loads(candidate_pointer.read_text()) if candidate_pointer.exists() else None
    commit_hash = saved.get('commit_candidate_sha256')
    if commit_hash is not None and candidate_pointer.exists() and _digest(candidate_pointer) != commit_hash:
        raise RuntimeError('reconciliation_publication_candidate_pointer_changed')
    if commit_hash is not None and proposed is not None and current == proposed:
        release = committed_reconciliation(processed, verify_outputs=True)
        _sync_aliases(processed, release)
        state = 'complete'
    elif root_pointer.exists():
        if _digest(root_pointer) != saved['previous_run_sha256']:
            raise RuntimeError('reconciliation_publication_unexpected_committed_run')
        state = 'aborted_before_commit'
    else:
        if saved['previous_run_sha256'] is not None:
            raise RuntimeError('reconciliation_publication_previous_run_missing')
        state = 'aborted_before_commit'
    _direct_json(journal, {**saved, 'state': state, 'recovered': True})
    return state


class ReconciliationPublication:
    def __init__(self, processed_dir):
        self.processed = Path(processed_dir).resolve()
        self.directory = None
        self.committed = False
        self.token = None

    def __enter__(self):
        if _working.get() is not None:
            raise RuntimeError('nested_reconciliation_publication')
        self.recovery = recover_publication(self.processed)
        root = self.processed / 'reconciliation_publications'
        root.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix='candidate_', dir=root))
        self.previous = (self.processed / RUN).read_bytes() if (self.processed / RUN).exists() else None
        for name in WORKING_ALIASES:
            path = self.processed / name
            if path.exists():
                _direct_json(self.directory / name, json.loads(path.read_text()))
        self.journal = {'state': 'preparing', 'candidate': str(self.directory.relative_to(self.processed)),
                        'previous_run_sha256': hashlib.sha256(self.previous).hexdigest() if self.previous is not None else None}
        _direct_json(self.processed / JOURNAL, self.journal)
        self.token = _working.set(self)
        return self

    def commit(self):
        pointer = json.loads((self.directory / RUN).read_text())
        release = committed_reconciliation(self.processed, reference=pointer, verify_outputs=True)
        for name in (SOURCE, DATASET, STAGES):
            if name not in release['references']:continue
            actual = json.loads((self.directory / name).read_text())
            if actual != release['references'][name]:
                raise RuntimeError('reconciliation_candidate_aliases_not_connected')
        if STAGES in release['references']:
            from data.pipelines.stage_snapshots import load_stage_model_inputs
            for stage in ('pre_demand','post_demand'):
                load_stage_model_inputs(self.processed,stage,reference=release['references'][STAGES],
                                        reconciliation_reference=pointer)
        root = self.processed / RUN
        current = root.read_bytes() if root.exists() else None
        if current != self.previous:
            raise RuntimeError('reconciliation_committed_run_changed_during_build')
        if self.previous is not None:
            previous=committed_reconciliation(self.processed)
            if 'stages' in previous['paths'] and 'stages' not in release['paths']:
                raise RuntimeError('complete_collection_cannot_downgrade_to_base_only')
        self.journal['commit_candidate_sha256'] = _digest(self.directory / RUN)
        _direct_json(self.processed / JOURNAL, self.journal)
        try:
            if current is None or json.loads(current) != pointer:
                _direct_json(root, pointer)
            self.committed = True
        except BaseException:
            self.committed = root.exists() and json.loads(root.read_text()) == pointer
            raise
        _direct_json(self.processed / JOURNAL, {**self.journal, 'state': 'committed_pending_alias_sync'})
        _sync_aliases(self.processed, release)
        _direct_json(self.processed / JOURNAL, {**self.journal, 'state': 'complete'})

    def __exit__(self, error_type, error, traceback):
        _working.reset(self.token)
        if not self.committed or error_type is not None:
            state = 'committed_pending_alias_sync' if self.committed else 'aborted_before_commit'
            try:
                _direct_json(self.processed / JOURNAL, {**self.journal, 'state': state})
            except OSError:
                logger.warning('Reconciliation publication journal update failed; recovery will inspect the committed pointer.')
        return False
