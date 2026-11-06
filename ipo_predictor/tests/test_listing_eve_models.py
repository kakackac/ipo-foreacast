import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from data.pipelines.data_delivery import digest
from data.pipelines.financial_history import atomic_json
from models.training.listing_eve import (
    BASELINES, CORE, FEATURE_SETS, TARGETS, annual_folds, build_model, candidates,
    load_model, nested_development, numeric, predict, select_candidate, train, validate_development,
)


def frame():
    dates = pd.to_datetime(['2020-03-01']*60 + ['2021-03-01']*60 + ['2022-03-01']*25 + ['2023-03-01']*25)
    f = pd.DataFrame({'event_id': [f'e{i}' for i in range(len(dates))], 'listing_date': dates})
    f['prediction_at'] = (dates.tz_localize('UTC') - pd.Timedelta(hours=12)).astype(str)
    for name in FEATURE_SETS['all']:
        f[name] = 1.
    f['lockup_commitment_ratio'] = .2
    f['operating_margin'] = np.nan
    for name in TARGETS:
        f[name] = np.arange(len(f), dtype=float)
    return f


class ListingEveModelTests(unittest.TestCase):
    def test_preprocessing_and_optional_feature_selection_fit_training_only(self):
        f = frame().head(60)
        f.loc[:29, 'operating_margin'] = 2.
        pipeline, fields = build_model(f, 'ridge:all')
        pipeline.fit(numeric(f, fields), f.open_return_pct)
        values = pipeline.named_steps['preprocessing'].named_transformers_['values']
        self.assertEqual(values.statistics_[fields.index('operating_margin')], 2.)
        validation = f.head(2).copy()
        validation.operating_margin = [10000., np.nan]
        before = values.statistics_.copy()
        pipeline.predict(numeric(validation, fields))
        np.testing.assert_array_equal(before, values.statistics_)
        self.assertNotIn('operating_margin', build_model(frame().head(60), 'ridge:all')[1])

    def test_invalid_units_missing_core_and_infinite_values_rejected(self):
        for field, value in [('institutional_demand_ratio', None), ('lockup_commitment_ratio', 17.),
                             ('debt_ratio', -1.), ('operating_margin', np.inf), ('kospi_momentum_5d', True)]:
            f = frame().head(1).copy()
            f[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                numeric(f, FEATURE_SETS['all'])

    def test_folds_are_chronological_and_never_include_2026(self):
        f = frame()
        for year, training, validation in annual_folds(f):
            self.assertLess(training.listing_date.max(), validation.listing_date.min())
            self.assertEqual(validation.listing_date.dt.year.unique().tolist(), [year])
        f.loc[f.index[-1], 'listing_date'] = pd.Timestamp('2026-01-01')
        with self.assertRaises(ValueError):
            annual_folds(f)

    def test_kst_aware_and_calendar_only_listing_dates_have_identical_folds(self):
        plain = frame()
        aware = plain.copy()
        aware['listing_date'] = aware.listing_date.dt.tz_localize('Asia/Seoul')
        self.assertEqual([(year, len(t), len(v)) for year, t, v in annual_folds(plain)],
                         [(year, len(t), len(v)) for year, t, v in annual_folds(aware)])

    def test_selection_requires_improvement_over_best_constant(self):
        records = []
        for candidate in candidates():
            error = 10 if candidate in BASELINES else 9.6
            records.append({'candidate': candidate, 'year': 2022, 'actual': 0., 'prediction': error})
        self.assertEqual(select_candidate(pd.DataFrame(records)), 'median')
        records[-1]['prediction'] = 9.
        self.assertEqual(select_candidate(pd.DataFrame(records)), candidates()[-1])
        self.assertEqual(select_candidate(pd.DataFrame()), 'median')

    def test_nested_selection_never_looks_at_current_validation_labels(self):
        f = frame()
        with patch('models.training.listing_eve.candidates', return_value=['median', 'mean', 'zero', 'ridge:core']):
            # Restrict model family only for speed, retaining the same chronological selection.
            predictions, decisions, _ = nested_development(f)
            changed = f.copy()
            changed.loc[changed.listing_date.dt.year.eq(2023), TARGETS] += 10000
            _, changed_decisions, _ = nested_development(changed)
        self.assertEqual(decisions, changed_decisions)
        self.assertTrue(all(max(d['selection_validation_years'], default=0) < d['year'] for d in decisions))
        for target in TARGETS:
            selected = predictions[predictions.target.eq(target) & predictions.selected_before_year]
            self.assertFalse(selected.event_id.duplicated().any())

    def test_source_evidence_and_exact_cutoff_are_required(self):
        f = frame()
        records = [{'event_id': r.event_id, 'feature': name, 'used': True, 'available_at': '2010-01-01'}
                   for r in f.itertuples() for name in FEATURE_SETS['all'] if pd.notna(getattr(r, name))]
        evidence = pd.DataFrame(records)
        validate_development(f, evidence)
        evidence.loc[0, 'available_at'] = '2027-01-01'
        with self.assertRaises(ValueError):
            validate_development(f, evidence)
        with self.assertRaises(ValueError):
            validate_development(f, pd.DataFrame(records[1:]))

    def test_saved_model_round_trip_and_tampering_block(self):
        f = frame()
        pipeline, fields = build_model(f, 'ridge:core')
        pipeline.fit(numeric(f, fields), f.open_return_pct)
        payload = {'pipeline': pipeline, 'fields': fields, 'historical_error_p90_reference_pp': 10., 'target': 'open_return_pct'}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'open_return_pct.joblib'
            joblib.dump(payload, path)
            import sklearn
            (root / 'manifest.json').write_text(json.dumps({'sklearn_version': sklearn.__version__,
                                                          'output_sha256': {path.name: digest(path)}}))
            loaded = load_model(root, 'open_return_pct')
            result = predict(loaded, f.head(2))
            self.assertFalse(result.uncertainty_calibrated.any())
            self.assertFalse(result.deployment_authorized.any())
            path.write_bytes(b'altered')
            with patch('models.training.listing_eve.joblib.load') as unsafe_load, self.assertRaises(RuntimeError):
                load_model(root, 'open_return_pct')
            unsafe_load.assert_not_called()

    def test_unlisted_model_file_is_never_unpickled(self):
        import sklearn
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'manifest.json').write_text(json.dumps({'sklearn_version': sklearn.__version__, 'output_sha256': {}}))
            with patch('models.training.listing_eve.joblib.load') as unsafe_load, self.assertRaises(RuntimeError):
                load_model(root, 'open_return_pct')
            unsafe_load.assert_not_called()

    def test_training_needs_explicit_research_permission(self):
        with self.assertRaises(PermissionError):
            train('/missing', '/missing')

    def test_full_training_reuses_artifacts_and_does_not_decode_exposed_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            processed = Path(directory) / 'processed'
            source = processed / 'research'
            source.mkdir(parents=True)
            f = frame()
            f.to_parquet(source / 'development.parquet', index=False)
            proof = [{'event_id': r.event_id, 'feature': name, 'used': True, 'available_at': '2010-01-01'}
                     for r in f.itertuples() for name in FEATURE_SETS['all'] if pd.notna(getattr(r, name))]
            pd.DataFrame(proof).to_parquet(source / 'evidence.parquet', index=False)
            (source / 'dataset.parquet').write_bytes(b'exposed_data_must_not_be_decoded')
            atomic_json(source / 'manifest.json', {'output_sha256': {name: digest(source / name) for name in
                        ['development.parquet', 'evidence.parquet', 'dataset.parquet']}})
            atomic_json(processed / 'delivery.json', {'version': 'fixture', 'research_dataset': 'research'})
            atomic_json(processed / 'current_research_data.json', {'manifest': 'delivery.json'})
            original = pd.read_parquet

            def guarded(path, *args, **kwargs):
                if Path(path).name == 'dataset.parquet':
                    raise AssertionError('exposed labels were read')
                return original(path, *args, **kwargs)

            with patch('models.training.listing_eve.current_financial_repair'), \
                 patch('models.training.listing_eve.pd.read_parquet', side_effect=guarded), \
                 patch('models.training.listing_eve.candidates', return_value=['median', 'mean', 'zero', 'ridge:core']):
                result = train(processed, Path(directory) / 'models', research_only=True)
                self.assertTrue(result['training_executed'])
                self.assertFalse(result['deployment_authorized'])
                self.assertTrue(train(processed, Path(directory) / 'models', research_only=True)['reused'])
            for target in TARGETS:
                payload = load_model(result['path'], target)
                self.assertTrue(np.isfinite(predict(payload, f.head(1)).pred_return_pct).all())
