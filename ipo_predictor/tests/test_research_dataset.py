import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data.pipelines.research_dataset import build_frames, run


class ResearchDatasetTests(unittest.TestCase):
    def setUp(self):
        self.features = pd.DataFrame([{'event_id': 'e', 'event_class': 'general_ipo', 'market': 'KOSDAQ',
            'listing_date': '2020-02-10', 'offering_price': 1000, 'offering_price_review_status': 'verified_currency_unit',
            'price_target_validation_status': 'official_price_verified', 'open_return_pct': 10, 'close_return_pct': -5,
            'institutional_demand_ratio': 100, 'lockup_commitment_ratio': None, 'rcept_no': '20200110000001'}])
        self.observations = pd.DataFrame([{'event_id': 'e', 'feature_name': 'institutional_demand_ratio',
            'raw_value': 100, 'is_missing': False, 'human_review_required': False,
            'validation_status': 'verified_dart_structural_aggregate_v1', 'source_reference': '20200110000001',
            'available_at': '2020-01-10'}])
        self.index = pd.DataFrame({'date': pd.date_range('2020-01-01', periods=45), 'close': range(100, 145)})
        self.lineage = pd.DataFrame([{'event_id': 'e', 'rcept_no': '20200110000001', 'rcept_dt': '2020-01-10'}])

    def build(self):
        return build_frames(self.features, self.observations, self.index, self.lineage)

    def test_complete_build_preserves_optional_missing_and_cutoff(self):
        data, excluded, evidence = self.build()
        self.assertEqual(len(data), 1)
        self.assertTrue(excluded.empty)
        self.assertTrue(data.lockup_commitment_ratio__missing.iloc[0])
        self.assertTrue(pd.isna(data.lockup_commitment_ratio.iloc[0]))
        cutoff = pd.Timestamp(data.prediction_at.iloc[0])
        self.assertEqual(cutoff, pd.Timestamp('2020-02-09T21:00:00+09:00'))
        self.assertTrue(all(pd.Timestamp(x) < cutoff for x in evidence.loc[evidence.used, 'available_at']))
        before = data.kospi_momentum_5d.iloc[0]
        self.index.loc[self.index.date.ge('2020-02-09'), 'close'] = 99999
        self.assertEqual(self.build()[0].kospi_momentum_5d.iloc[0], before)

    def test_required_future_and_wrong_receipt_are_rejected(self):
        self.observations.loc[0, 'available_at'] = '2020-02-10'
        self.assertEqual(len(self.build()[0]), 0)
        self.observations.loc[0, 'available_at'] = '2020-01-10'
        self.lineage.loc[0, 'event_id'] = 'another'
        self.assertEqual(len(self.build()[0]), 0)

    def test_duplicate_observation_is_not_silently_selected(self):
        self.observations = pd.concat([self.observations, self.observations])
        with self.assertRaises(ValueError):
            self.build()

    def test_files_reused_and_modification_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            raw, processed = Path(folder) / 'raw', Path(folder) / 'processed'
            raw.mkdir()
            processed.mkdir()
            self.features.to_parquet(processed / 'features_all.parquet')
            self.observations.to_parquet(processed / 'feature_observations.parquet')
            self.index.to_parquet(raw / 'kospi_index.parquet')
            self.lineage.to_parquet(raw / 'dart_disclosure_lineage.parquet')
            first = run(raw, processed)
            self.assertEqual(first['development_rows'], 1)
            self.assertTrue(run(raw, processed)['reused'])
            (Path(first['path']) / 'dataset.parquet').write_bytes(b'changed')
            with self.assertRaises(RuntimeError):
                run(raw, processed)
