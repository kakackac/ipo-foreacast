"""Frozen, nested chronological development and research-only model artifacts."""
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data.pipelines.data_delivery import checked_artifact, current_financial_repair, digest
from data.pipelines.financial_history import atomic_json
from scripts.finalize_listing_eve_experiment import paired_improvement

CORE = ['institutional_demand_ratio', 'kospi_momentum_5d', 'kospi_momentum_20d', 'recent_ipo_avg_return_all']
FEATURE_SETS = {
    'core': CORE,
    'financial': CORE + ['operating_margin', 'debt_ratio'],
    'all': CORE + ['lockup_commitment_ratio', 'operating_margin', 'debt_ratio'],
}
TARGETS = ['open_return_pct', 'close_return_pct']
BASELINES = ['median', 'mean', 'zero']
POLICY = {
    'version': 1, 'scope': 'general_ipo_listing_eve_research_only',
    'development_end_exclusive': '2026-01-01', 'minimum_train': 100, 'minimum_validation': 20,
    'optional_minimum_observations': 20, 'optional_minimum_fraction': .2,
    'selection': 'macro_year_mae_with_five_percent_improvement_required_over_best_constant',
    'outer_selection': 'use_only_completed_earlier_validation_years; median_if_none',
    'feature_sets': FEATURE_SETS, 'baselines': BASELINES,
    'estimators': {
        'gb_huber': {'loss': 'huber', 'alpha': .9, 'n_estimators': 120, 'max_depth': 2,
                     'min_samples_leaf': 10, 'learning_rate': .05, 'random_state': 42},
        'gb_squared': {'loss': 'squared_error', 'n_estimators': 100, 'max_depth': 2,
                       'min_samples_leaf': 10, 'learning_rate': .05, 'random_state': 42},
        'ridge': {'alpha': 100.},
    },
    'excluded_features': {'revenue_growth_3y': 'only_8_of_390_verified_rows_in_prior_data_audit',
                          'retail_subscription_ratio': 'outside_product_policy'},
    'preprocessing': 'fit_training_only_medians_missing_indicators_and_scaling',
    'uncertainty': 'nested_development_absolute_error_percentiles_are_references_not_prediction_intervals',
    'exposed_2026': 'not_read_for_selection_fitting_or_scoring',
    'challenger': 'retain_best_learned_candidate_even_if_constant_policy_is_selected',
    'deployment_authorized': False,
}


def candidates():
    return BASELINES + [f'{model}:{group}' for model in POLICY['estimators'] for group in FEATURE_SETS]


def listing_days(series):
    return pd.to_datetime(series, utc=True, errors='raise').dt.tz_convert('Asia/Seoul').dt.normalize()


def numeric(frame, fields):
    if any(name not in frame for name in fields):
        raise ValueError('required_model_columns_missing')
    values = frame[fields].copy()
    if values.map(lambda v: isinstance(v, (bool, np.bool_))).any().any():
        raise ValueError('boolean_is_not_a_numeric_feature')
    values = values.apply(pd.to_numeric, errors='raise').astype(float)
    if np.isinf(values.to_numpy()).any():
        raise ValueError('infinite_model_feature')
    if values[CORE].isna().any().any():
        raise ValueError('required_core_observation_missing')
    if (values.institutional_demand_ratio < 0).any():
        raise ValueError('negative_institutional_ratio')
    if 'lockup_commitment_ratio' in fields and not values.lockup_commitment_ratio.dropna().between(0, 1).all():
        raise ValueError('lockup_requires_fraction_not_percent')
    if 'debt_ratio' in fields and (values.debt_ratio.dropna() < 0).any():
        raise ValueError('negative_debt_ratio')
    return values


def build_model(frame, candidate):
    requested = CORE if candidate in BASELINES else FEATURE_SETS[candidate.split(':')[1]]
    values = numeric(frame, requested)
    threshold = max(POLICY['optional_minimum_observations'], math.ceil(len(frame) * POLICY['optional_minimum_fraction']))
    fields = [f for f in requested if f in CORE or values[f].notna().sum() >= threshold]
    preprocessing = ColumnTransformer([
        ('values', SimpleImputer(strategy='median'), fields),
        ('missing', MissingIndicator(features='all'), fields),
    ], remainder='drop')
    if candidate in BASELINES:
        estimator = DummyRegressor(strategy='constant', constant=0) if candidate == 'zero' else DummyRegressor(strategy=candidate)
    elif candidate.startswith('ridge:'):
        estimator = Ridge(**POLICY['estimators']['ridge'])
    else:
        estimator = GradientBoostingRegressor(**POLICY['estimators'][candidate.split(':')[0]])
    return Pipeline([('preprocessing', preprocessing), ('scale', StandardScaler()), ('estimator', estimator)]), fields


def annual_folds(frame):
    dates = listing_days(frame.listing_date)
    if dates.isna().any() or (dates >= pd.Timestamp(POLICY['development_end_exclusive'], tz='Asia/Seoul')).any():
        raise ValueError('exposed_or_invalid_listing_date_in_development')
    if frame.event_id.isna().any() or frame.event_id.duplicated().any():
        raise ValueError('invalid_development_identity')
    folds = []
    for year in sorted(dates.dt.year.unique()):
        training = frame[dates.dt.year < year]
        validation = frame[dates.dt.year == year]
        if len(training) >= POLICY['minimum_train'] and len(validation) >= POLICY['minimum_validation']:
            prediction_at = pd.to_datetime(validation.prediction_at, utc=True, errors='raise').min()
            outcomes_at = listing_days(training.listing_date).max() + pd.Timedelta(days=1)
            if outcomes_at >= prediction_at:
                raise ValueError('training_outcomes_not_available_at_validation_cutoff')
            folds.append((int(year), training, validation))
    if len(folds) < 2:
        raise ValueError('insufficient_chronological_folds')
    return folds


def select_candidate(prior_predictions):
    if prior_predictions.empty:
        return 'median'
    errors = prior_predictions.assign(error=(prior_predictions.prediction - prior_predictions.actual).abs())
    ranking = errors.groupby(['candidate', 'year']).error.mean().groupby('candidate').mean()
    baseline = min(BASELINES, key=lambda c: (ranking[c], BASELINES.index(c)))
    learned = [c for c in candidates() if c not in BASELINES]
    best = min(learned, key=lambda c: (ranking[c], candidates().index(c)))
    return best if ranking[best] < .95 * ranking[baseline] else baseline


def score(frame):
    error = frame.prediction - frame.actual
    return {'n': len(frame), 'mae_pp': float(error.abs().mean()),
            'median_ae_pp': float(error.abs().median()), 'rmse_pp': float(np.sqrt((error**2).mean())),
            'direction_accuracy': float(((frame.prediction > 0) == (frame.actual > 0)).mean())}


def validate_development(frame, evidence):
    annual_folds(frame)
    numeric(frame, FEATURE_SETS['all'])
    for target in TARGETS:
        if not np.isfinite(pd.to_numeric(frame[target], errors='raise')).all():
            raise ValueError('invalid_training_target')
    listing = listing_days(frame.listing_date)
    cutoff = pd.to_datetime(frame.prediction_at, utc=True, errors='raise')
    if not (cutoff == listing - pd.Timedelta(hours=3)).all():
        raise ValueError('unsupported_prediction_cutoff')
    selected_evidence = evidence[evidence.event_id.isin(frame.event_id)]
    if selected_evidence.duplicated(['event_id', 'feature']).any():
        raise ValueError('duplicate_source_evidence')
    proof = selected_evidence[selected_evidence.used.eq(True)].merge(
        frame[['event_id', 'prediction_at']], on='event_id', how='left', validate='many_to_one')
    when = pd.to_datetime(proof.available_at, utc=True, errors='coerce')
    limits = pd.to_datetime(proof.prediction_at, utc=True, errors='coerce')
    if when.isna().any() or (when >= limits).any():
        raise ValueError('late_or_missing_source_evidence')
    for _, row in frame.iterrows():
        expected = [f for f in FEATURE_SETS['all'] if pd.notna(row[f])]
        observed = set(proof.loc[proof.event_id.eq(row.event_id), 'feature'])
        if not set(expected).issubset(observed):
            raise ValueError('observed_feature_without_accepted_evidence')


def nested_development(frame):
    records, decisions, fitted_fields = [], [], []
    for year, training, validation in annual_folds(frame):
        for target in TARGETS:
            prior = pd.DataFrame([r for r in records if r['target'] == target])
            selected = select_candidate(prior)
            decisions.append({'year': year, 'target': target, 'selected': selected,
                              'selection_validation_years': sorted(prior.year.unique().tolist()) if len(prior) else [],
                              'train_rows': len(training), 'validation_rows': len(validation)})
            for candidate in candidates():
                pipeline, fields = build_model(training, candidate)
                pipeline.fit(numeric(training, fields), training[target])
                prediction = pipeline.predict(numeric(validation, fields))
                fitted_fields.append({'year': year, 'target': target, 'candidate': candidate, 'fields': fields})
                for row, value in zip(validation.itertuples(index=False), prediction):
                    records.append({'event_id': row.event_id, 'listing_date': str(row.listing_date), 'year': year,
                                    'target': target, 'candidate': candidate, 'actual': float(getattr(row, target)),
                                    'prediction': float(value), 'selected_before_year': candidate == selected})
    return pd.DataFrame(records), decisions, fitted_fields


def verify_run(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, expected in manifest['output_sha256'].items():
        path = (root / name).resolve()
        if path.parent != root or digest(path) != expected:
            raise RuntimeError('research_model_artifact_changed')
    return manifest


def load_model(root, target, challenger=False):
    if target not in TARGETS:
        raise ValueError('unsupported_target')
    root = Path(root)
    manifest = verify_run(root)
    if manifest['sklearn_version'] != sklearn.__version__:
        raise RuntimeError('model_runtime_version_mismatch')
    # Only trusted local artifacts are supported; hashes do not authenticate an untrusted publisher.
    name = f'{target}_challenger.joblib' if challenger else f'{target}.joblib'
    if name not in manifest['output_sha256']:
        name = f'{target}.joblib'
    if name not in manifest['output_sha256']:
        raise RuntimeError('model_file_not_bound_to_manifest')
    result = joblib.load(root / name)
    if result.get('target') != target:
        raise RuntimeError('model_target_contract_mismatch')
    return result


def predict(payload, frame):
    values = numeric(frame, payload['fields'])
    point = payload['pipeline'].predict(values)
    if not np.isfinite(point).all():
        raise ValueError('nonfinite_model_prediction')
    return pd.DataFrame({'pred_return_pct': point,
        'historical_error_p90_reference_pp': payload['historical_error_p90_reference_pp'],
        'uncertainty_calibrated': False, 'deployment_authorized': False}, index=frame.index)


def train(processed_dir, model_dir, research_only=False):
    if not research_only:
        raise PermissionError('explicit_research_training_permission_required')
    processed, parent = Path(processed_dir), Path(model_dir)
    pointer_path = processed / 'current_research_data.json'
    pointer_hash = digest(pointer_path)
    current_financial_repair(processed)
    if digest(pointer_path) != pointer_hash:
        raise RuntimeError('model_data_pointer_changed')
    pointer = json.loads(pointer_path.read_text())
    delivery = json.loads((processed / pointer['manifest']).read_text())
    source = processed / delivery['research_dataset']
    manifest = checked_artifact(source)
    frame = pd.read_parquet(source / 'development.parquet').sort_values(['listing_date', 'event_id'])
    evidence = pd.read_parquet(source / 'evidence.parquet')
    validate_development(frame, evidence)
    frozen = {'policy': POLICY, 'data_version': delivery['version'],
              'development_sha256': manifest['output_sha256']['development.parquet'],
              'evidence_sha256': manifest['output_sha256']['evidence.parquet'],
              'sklearn_version': sklearn.__version__, 'numpy_version': np.__version__,
              'pandas_version': pd.__version__, 'code_sha256': digest(__file__)}
    run_id = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()[:20]
    root = parent / run_id
    if root.exists():
        result = verify_run(root)
        return {**result, 'path': str(root), 'reused': True}
    root.mkdir(parents=True, exist_ok=False)
    atomic_json(root / 'frozen_policy.json', frozen)
    atomic_json(root / 'training_started.json', {'started_at': datetime.now(timezone.utc).isoformat(),
                                                'research_only': True, 'deployment_authorized': False})
    try:
        predictions, decisions, fields = nested_development(frame)
        all_metrics, by_year, selected_models, comparisons = [], [], {}, {}
        selected_metrics, model_files = {}, []
        for target in TARGETS:
            target_rows = predictions[predictions.target.eq(target)]
            nested = target_rows[target_rows.selected_before_year]
            selected_metrics[target] = score(nested)
            comparisons[target] = {}
            for baseline in BASELINES:
                constant = target_rows[target_rows.candidate.eq(baseline)].set_index('event_id').reindex(nested.event_id)
                comparisons[target][baseline] = paired_improvement(nested.actual, nested.prediction,
                                                                   constant.prediction, nested.listing_date)
            for candidate, rows in target_rows.groupby('candidate'):
                all_metrics.append({'target': target, 'candidate': candidate, **score(rows)})
                for year, subset in rows.groupby('year'):
                    by_year.append({'target': target, 'candidate': candidate, 'year': int(year), **score(subset)})
            chosen = select_candidate(target_rows)
            pipeline, active = build_model(frame, chosen)
            pipeline.fit(numeric(frame, active), frame[target])
            payload = {'pipeline': pipeline, 'fields': active, 'target': target,
                       'candidate': chosen, 'data_version': delivery['version'],
                       'historical_error_p90_reference_pp': float((nested.prediction - nested.actual).abs().quantile(.9)),
                       'uncertainty_calibrated': False, 'scope': POLICY['scope']}
            filename = root / f'{target}.joblib'
            joblib.dump(payload, filename)
            model_files.append(filename.name)
            loaded = joblib.load(filename)
            before, after = predict(payload, frame.head(3)), predict(loaded, frame.head(3))
            np.testing.assert_allclose(before.pred_return_pct, after.pred_return_pct, atol=1e-12, rtol=0)
            selected_models[target] = {'candidate': chosen, 'fields': active, 'training_rows': len(frame),
                'training_last_listing_date': str(frame.listing_date.max()), 'serialization_round_trip_passed': True,
                'is_learned_estimator': chosen not in BASELINES,
                'historical_error_p90_reference_pp': payload['historical_error_p90_reference_pp']}
            learned = target_rows[~target_rows.candidate.isin(BASELINES)].copy()
            learned['absolute_error'] = (learned.prediction - learned.actual).abs()
            ranking = learned.groupby(['candidate', 'year']).absolute_error.mean().groupby('candidate').mean()
            best_learned = min(ranking.index, key=lambda c: (ranking[c], candidates().index(c)))
            selected_models[target]['best_learned_candidate'] = best_learned
            if best_learned != chosen:
                challenger_pipeline, challenger_fields = build_model(frame, best_learned)
                challenger_pipeline.fit(numeric(frame, challenger_fields), frame[target])
                challenger = {**payload, 'pipeline': challenger_pipeline, 'fields': challenger_fields,
                              'candidate': best_learned, 'selection_role': 'challenger_not_selected_policy'}
                challenger_file = root / f'{target}_challenger.joblib'
                joblib.dump(challenger, challenger_file)
                model_files.append(challenger_file.name)
                restored = joblib.load(challenger_file)
                np.testing.assert_allclose(predict(challenger, frame.head(3)).pred_return_pct,
                                           predict(restored, frame.head(3)).pred_return_pct, atol=1e-12, rtol=0)
        predictions.to_parquet(root / 'development_predictions.parquet', index=False)
        report = {'run_id': run_id, 'scope': POLICY['scope'], 'training_rows': len(frame),
            'previously_exposed_2026_scored': False, 'previously_exposed_2026_used_for_fit': False,
            'fresh_independent_test_available': False,
            'nested_selection_metrics': selected_metrics, 'paired_improvements': comparisons,
            'candidate_selection_diagnostics_not_independent_test': all_metrics, 'year_metrics': by_year,
            'yearly_selection_decisions': decisions, 'fold_feature_contracts': fields,
            'selected_models': selected_models,
            'retrospective_development_signal': {target: all(comparisons[target][b]['day_cluster_bootstrap_95_interval_pp'][0] > 0
                                                           for b in ['mean', 'median']) for target in TARGETS},
            'uncertainty_warning': 'Historical errors are not calibrated prediction intervals or guaranteed coverage.',
            'release_blockers': ['fresh_prospective_evaluation_required', 'point_in_time_live_input_adapter_required',
                                 'general_ipo_listing_eve_only_not_pre_demand', 'historical_sample_selection_bias'],
            'training_executed': True, 'deployment_authorized': False}
        atomic_json(root / 'report.json', report)
        if digest(pointer_path) != pointer_hash or checked_artifact(source) != manifest:
            raise RuntimeError('model_training_inputs_changed')
        files = ['frozen_policy.json', 'training_started.json', 'report.json', 'development_predictions.parquet']
        files += model_files
        output = {'run_id': run_id, 'data_version': delivery['version'], 'sklearn_version': sklearn.__version__,
                  'output_sha256': {name: digest(root / name) for name in files},
                  'training_executed': True, 'deployment_authorized': False,
                  'completed_at': datetime.now(timezone.utc).isoformat(), 'report': report}
        atomic_json(root / 'manifest.json', output)
        atomic_json(parent / 'current_candidate.json', {'run_id': run_id, 'manifest_sha256': digest(root / 'manifest.json'),
                                                       'deployment_authorized': False})
        return {**output, 'path': str(root), 'reused': False}
    except Exception as exc:
        atomic_json(root / 'failed.json', {'error_type': type(exc).__name__, 'deployment_authorized': False})
        raise
