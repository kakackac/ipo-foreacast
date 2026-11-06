import unittest
from dataclasses import replace
import pandas as pd

from features.model_profiles import MODEL_PROFILES, _stage_time_valid
from features.prediction_time_contract import validate_prediction_times


class PredictionTimeContractTests(unittest.TestCase):
    def setUp(self):
        self.profile = replace(MODEL_PROFILES['post_demand'], feature_names=('institutional_demand_ratio',))
        self.frame = pd.DataFrame([{'event_id': 'a', 'institutional_demand_ratio': 100,
            'listing_date': '2026-10-10', 'prediction_at': '2026-10-09T10:00:00+09:00',
            'prediction_time_source': 'research_snapshot:a', 'demand_result_available_at': '2026-10-08T12:00:00+09:00',
            'demand_result_time_source': 'official_receipt:a'}])
        self.audit = pd.DataFrame([{'event_id': 'a', 'feature_name': 'institutional_demand_ratio',
            'available_at': '2026-10-08T12:00:00+09:00', 'time_validation_status': 'pre_listing_verified', 'is_missing': False}])

    def test_verified_stage_and_each_feature_cutoff(self):
        self.assertTrue(validate_prediction_times(self.frame, self.profile, self.audit).iloc[0])
        self.audit.loc[0, 'available_at'] = '2026-10-09T11:00:00+09:00'
        self.assertFalse(validate_prediction_times(self.frame, self.profile, self.audit).iloc[0])

    def test_pre_listing_flag_does_not_prove_prediction_time(self):
        self.frame = self.frame.drop(columns=['prediction_at'])
        self.assertFalse(validate_prediction_times(self.frame, self.profile, self.audit).iloc[0])

    def test_stage_boundary_date_only_and_duplicate_evidence(self):
        pre = replace(self.profile, name='pre_demand')
        self.assertFalse(validate_prediction_times(self.frame, pre, self.audit).iloc[0])
        self.frame.loc[0, 'demand_result_available_at'] = '2026-10-09'
        self.assertFalse(validate_prediction_times(self.frame, self.profile, self.audit).iloc[0])
        self.frame.loc[0, 'demand_result_available_at'] = '2026-10-08'
        self.assertFalse(validate_prediction_times(self.frame, self.profile, pd.concat([self.audit, self.audit])).iloc[0])

    def test_all_missing_audit_cannot_approve_observed_feature(self):
        self.audit['is_missing'] = True
        self.assertFalse(_stage_time_valid(self.frame, self.profile, self.audit).iloc[0])
        self.assertFalse(validate_prediction_times(self.frame, self.profile, self.audit).iloc[0])
