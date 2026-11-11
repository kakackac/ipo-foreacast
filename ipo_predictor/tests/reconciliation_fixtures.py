import hashlib
import json
from pathlib import Path

from data.pipelines.data_delivery import digest
from data.pipelines.financial_history import atomic_json
from data.pipelines.reconciliation_publication import SOURCE, DATASET, RUN, STAGES, BASE_SCOPE, COMPLETE_SCOPE


def components(processed, tag):
    from data.pipelines.historical_reconciliation import connection_contract
    paths = {key: Path(processed) / folder / tag for key, folder in
             [('sources', 'historical_reconciliations'), ('financial', 'historical_financial_reconciliations'),
              ('dataset', 'reconciled_datasets')]}
    def save(key, report):
        root = paths[key]
        atomic_json(root / 'payload.json', {'fixture': tag, 'component': key})
        atomic_json(root / 'manifest.json', {**report, 'version': tag,
                    'output_sha256': {'payload.json': digest(root / 'payload.json')}})
    inputs = {'fixture': tag}
    save('sources', {'inputs': {'inputs': inputs, 'contract': connection_contract()}, 'population': 1,
                     'status_counts': {'official_sources_connected': 1}})
    save('financial', {'inputs': inputs, 'connection_manifest': digest(paths['sources'] / 'manifest.json'),
                      'scope': 'all_connected_general_ipo_financial_sections', 'population': 1, 'processed': 1})
    save('dataset', {'inputs': {'inputs': inputs, 'sources': digest(paths['sources'] / 'manifest.json'),
                               'financial': digest(paths['financial'] / 'manifest.json')},
                     'rows': 1, 'validation': {'status': 'snapshot_valid_with_recorded_holds', 'rows': 1,
                        'observations': 1, 'time_violations': 0, 'value_violations': 0, 'non_general_model_observations': 0}})
    return paths


def references(processed, paths):
    return {alias: {'manifest': str((paths[key] / 'manifest.json').relative_to(processed)),
                     'sha256': digest(paths[key] / 'manifest.json'),
                     **({'training_authorized': False, 'deployment_authorized': False} if alias == DATASET else {})}
            for alias, key in [(SOURCE, 'sources'), (DATASET, 'dataset'), *([(STAGES,'stages')] if 'stages' in paths else [])]}


def complete_run(processed, paths):
    hashes = {key: digest(path / 'manifest.json') for key, path in paths.items()}
    version = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:20]
    manifest = processed / 'reconciliation_runs' / version / 'manifest.json'
    atomic_json(manifest, {'version': version, 'status': 'complete_with_recorded_feature_holds',
        'input_sha256': hashes, **{key: str(path) for key, path in paths.items()},
        'training_executed': False, 'deployment_authorized': False,
        'scope': COMPLETE_SCOPE if 'stages' in paths else BASE_SCOPE})
    return {'manifest': str(manifest.relative_to(processed)), 'sha256': digest(manifest)}


def publish_fixture(processed, paths):
    for alias, value in references(processed, paths).items():
        atomic_json(processed / alias, value)
    atomic_json(processed / RUN, complete_run(processed, paths))
