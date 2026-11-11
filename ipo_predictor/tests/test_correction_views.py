import unittest

import pandas as pd

from data.pipelines.correction_views import after_correction_view
from data.pipelines.reconciled_dataset import mask_unapproved
from features.model_profiles import stage_source_valid,get_model_profile


class CorrectionViewsTests(unittest.TestCase):
    def test_inline_markers_keep_only_explicit_after_tables(self):
        body='<p><span>[정정 전]</span>old 100원<span>[정정 후]</span>new 200원</p><table><tr><td>after</td></tr></table>'
        body+='<p>[정정 전]</p><table><tr><td>old</td></tr></table><p>[정정 후]</p><p>new</p>'
        view,evidence=after_correction_view(body)
        self.assertNotIn('100원',view);self.assertNotIn('<td>old</td>',view)
        self.assertIn('200원',view);self.assertIn('<td>after</td>',view)
        self.assertEqual(evidence['pair_count'],2)

    def test_ambiguous_or_unpaired_markers_are_not_guessed(self):
        for body in ['<p>[정정 후]</p>200원','<p>[정정 전]</p>100원',
                     '<table><tr><td>[정정 전]</td><td>[정정 후]</td></tr></table>',
                     '<p>본문에서 [정정 후]라는 표현을 참고</p>']:
            with self.assertRaises(ValueError):after_correction_view(body)

    def test_raw_candidates_survive_but_unapproved_and_future_values_do_not_enter_models(self):
        features=pd.DataFrame([dict(event_id='e',listing_date='2024-01-10',underwriter_tier=1,
            institutional_demand_ratio=100,institutional_validation_status='verified_dart_structural_aggregate_v1',
            offering_price_band_position=1,price_band_validation_status='needs_review')])
        observed=pd.DataFrame([
            dict(event_id='e',feature_name='underwriter_tier',raw_value=1,human_review_required=False,available_at='2024-01-01'),
            dict(event_id='e',feature_name='institutional_demand_ratio',raw_value=100,human_review_required=False,available_at='2024-01-10'),
            dict(event_id='e',feature_name='offering_price_band_position',raw_value=1,human_review_required=True,available_at='2024-01-01')])
        result,ledger=mask_unapproved(features,observed)
        self.assertTrue(result[['underwriter_tier','institutional_demand_ratio','offering_price_band_position']].isna().all().all())
        self.assertEqual(ledger.raw_value.tolist(),[1,100,1])
        self.assertFalse(ledger.model_eligible.any())

    def test_optional_null_needs_no_label_but_legacy_heuristic_value_is_blocked(self):
        frame=pd.DataFrame([{'major_shareholder_lockup_months':None,'risk_factor_count':None}])
        self.assertTrue(stage_source_valid(frame,get_model_profile('pre_demand')).iloc[0])
        frame.loc[0,'risk_factor_count']=12
        self.assertFalse(stage_source_valid(frame,get_model_profile('pre_demand')).iloc[0])
