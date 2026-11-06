import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from api import server
from api.release_gate import verify_release


class ReleaseGateTests(unittest.TestCase):
    def test_artifacts_must_match_reviewed_release(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = {'approved': True, 'prediction_stage': 'post_demand', 'release_id': 'test',
                        'reviewed_by': 'test', 'validation_report': 'test report', 'artifact_sha256': {}}
            for target in ('open', 'close'):
                for suffix in ('.pkl', '_meta.json'):
                    name = f'post_demand_{target}_v1{suffix}'
                    (root / name).write_bytes(b'fixture')
                    manifest['artifact_sha256'][name] = hashlib.sha256(b'fixture').hexdigest()
            path = root / 'production_release.json'
            path.write_text(json.dumps(manifest))
            self.assertTrue(verify_release(root, 'post_demand'))
            (root / 'post_demand_open_v1.pkl').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                verify_release(root, 'post_demand')
            path.write_text('[]')
            with self.assertRaises(ValueError):
                verify_release(root, 'post_demand')

    def test_missing_release_never_trains_demo_and_returns_503(self):
        request = {'institutional_demand_ratio': 100, 'lockup_commitment_ratio': .1,
                   'offering_price_band_position': 1}
        with tempfile.TemporaryDirectory() as folder, patch('config.MODEL_DIR', Path(folder)), \
             patch('models.baseline.gradient_boost_model.IPOPriceModel.fit') as training, patch.object(server, '_models', {'opening': object()}):
            client = TestClient(server.app)
            self.assertEqual(client.get('/health').status_code, 200)
            for url, body in [('/predict', request), ('/predict/batch', {'items': [request]})]:
                self.assertEqual(client.post(url, json=body).status_code, 503)
            self.assertEqual(client.get('/ready').status_code, 503)
            self.assertEqual(client.get('/model/info').status_code, 503)
            training.assert_not_called()
            self.assertEqual(server._models, {})

    def test_missing_market_inputs_are_not_fabricated(self):
        features = server.IPOFeatures(institutional_demand_ratio=100, lockup_commitment_ratio=.1,
                                      offering_price_band_position=1)
        values = server._feature_dict(features)
        for name in ('kospi_momentum_5d', 'kospi_momentum_20d', 'recent_ipo_avg_return_sector', 'recent_ipo_avg_return_all'):
            self.assertIsNone(values[name])
