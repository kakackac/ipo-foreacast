import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data.pipelines.data_delivery import current_financial_repair, digest, publish


class DataDeliveryTests(unittest.TestCase):
    def artifacts(self, folder, available='2020-01-01'):
        processed = Path(folder)
        pd.DataFrame({'event_id': ['e']}).to_parquet(processed / 'features_all.parquet')
        repair = processed / 'financial_repairs' / 'one'
        research = processed / 'research_datasets' / 'one'
        repair.mkdir(parents=True)
        research.mkdir(parents=True)
        pd.DataFrame({'event_id': ['e']}).to_parquet(repair / 'features_all.parquet')
        pd.DataFrame({'feature': ['x']}).to_parquet(repair / 'feature_observations.parquet')
        output = {name: digest(repair / name) for name in ['features_all.parquet', 'feature_observations.parquet']}
        (repair / 'manifest.json').write_text(json.dumps({'input_sha256': digest(processed / 'features_all.parquet'),
                                                        'output_sha256': output, 'status_counts': {'verified': 1}}))
        pd.DataFrame({'event_id': ['e'], 'listing_date': ['2020-02-02'], 'prediction_at': ['2020-02-01'],
                      'x': [1]}).to_parquet(research / 'dataset.parquet')
        pd.DataFrame({'event_id': ['e'], 'used': [True], 'available_at': [available]}).to_parquet(research / 'evidence.parquet')
        (research / 'manifest.json').write_text(json.dumps({
            'source_sha256': {'features': output['features_all.parquet'], 'observations': output['feature_observations.parquet']},
            'features': ['x'], 'output_sha256': {n: digest(research / n) for n in ['dataset.parquet', 'evidence.parquet']}}))
        return processed, repair, research

    def test_publish_reuses_version_without_replacing_production_or_approving_training(self):
        with tempfile.TemporaryDirectory() as folder:
            processed, repair, research = self.artifacts(folder)
            before = digest(processed / 'features_all.parquet')
            result = publish(processed, repair, research)
            self.assertEqual(current_financial_repair(processed), repair.resolve())
            self.assertEqual(result, publish(processed, repair, research))
            self.assertEqual(result['time_violations'], 0)
            self.assertFalse(result['deployment_authorized'])
            self.assertEqual(digest(processed / 'features_all.parquet'), before)

    def test_modified_output_invalidates_current_pointer(self):
        with tempfile.TemporaryDirectory() as folder:
            processed, repair, research = self.artifacts(folder)
            publish(processed, repair, research)
            (research / 'dataset.parquet').write_bytes(b'changed')
            with self.assertRaises(RuntimeError):
                current_financial_repair(processed)

    def test_future_evidence_cannot_be_published(self):
        with tempfile.TemporaryDirectory() as folder:
            processed, repair, research = self.artifacts(folder, available='2020-02-02')
            with self.assertRaisesRegex(RuntimeError, 'future_or_missing'):
                publish(processed, repair, research)
            self.assertFalse((processed / 'current_research_data.json').exists())
