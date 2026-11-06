import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.parse import urlencode

import pandas as pd
import requests

from data.pipelines.event_disclosures import FIELDS, _bundle_scalars, connect_event
from data.processors.feature_engineer import FeatureEngineer


class EventDisclosureBundleTests(unittest.TestCase):
    primary_meta = {'rcept_no': '20200101000001', 'rcept_dt': '2020-01-01', 'report_nm': 'terms'}
    extra_meta = {'rcept_no': '20200102000001', 'rcept_dt': '2020-01-02', 'report_nm': 'prospectus'}

    def document(self, **fields):
        return {'offering_price': 1000, 'offering_price_review_status': 'verified_currency_unit',
                'offering_price_finality': 'confirmed_price_language', **fields}

    def structure(self, new=None, secondary=None, **fields):
        result = dict(fields)
        for field, value in [('new_shares', new), ('secondary_shares', secondary)]:
            result[field] = value
            result[field + '_parser_validation_status'] = 'structurally_verified' if value is not None else 'value_not_found'
            result[field + '_parse_method'] = 'direct_table_row_v1' if value is not None else None
            result[field + '_parse_evidence'] = str(value) if value is not None else None
        return result

    def connect(self, primary, extra=None, extra_meta=None, failure=None, primary_scope=None):
        extra_meta = extra_meta or self.extra_meta
        event = SimpleNamespace(ticker='123456', market='KOSDAQ', listing_date='2020-02-01', offering_price=1000)
        company_url = 'https://dart.fss.or.kr/dsae001/selectPopup.ax?' + urlencode({'selectKey': '12345678'})
        primary_url = 'https://dart.fss.or.kr/dsaf001/main.do?rcpNo=' + self.primary_meta['rcept_no']
        extra_url = 'https://dart.fss.or.kr/dsaf001/main.do?rcpNo=' + extra_meta['rcept_no']
        proof = {'corp_code': '12345678', 'family': [self.primary_meta], 'source_url': primary_url, 'sections': []}
        if primary_scope is not None: proof['content_scope']=primary_scope
        extra_proof = {'corp_code': '12345678', 'family': [extra_meta], 'source_url': extra_url, 'sections': []}
        dart = Mock()
        dart.session.get.side_effect = AssertionError('network is forbidden')
        dart._parse_offering_html.side_effect = lambda body, receipt: dict(primary if body == 'primary' else extra)
        dart._parse_demand_forecast_html.return_value = {}
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder) / 'dart_connection_sources'
            cache.mkdir()
            for url in (company_url, primary_url, extra_url):
                payload = {'url': url, 'text': 'synthetic', 'sha256': hashlib.sha256(b'synthetic').hexdigest(),
                           'collected_at': '2026-01-01T00:00:00+00:00'}
                (cache / (hashlib.sha256(url.encode()).hexdigest() + '.json')).write_text(json.dumps(payload))
            with patch('data.pipelines.event_disclosures.public_text', return_value=('synthetic', 'hash')), \
                 patch('data.pipelines.event_disclosures.company_identity', return_value={}), \
                 patch('data.pipelines.event_disclosures.official_family_document', return_value=(self.primary_meta, 'primary', proof)), \
                 patch('data.pipelines.event_disclosures.official_related_prospectus') as supplement:
                supplement.side_effect = failure or (ValueError('no_pre_cutoff_related_prospectus') if extra is None else None)
                supplement.return_value = (extra_meta, 'extra', extra_proof)
                result = connect_event(dart, folder, event, [{'corp_code': '12345678', **self.primary_meta}], cached_only=True)
        dart.session.get.assert_not_called()
        self.assertEqual(len(result), 6)
        self.assertEqual(set(result[0]), set(FIELDS))
        json.dumps(result[2])
        for bundle in ('resolved_float', 'resolved_structure'):
            json.dumps(result[2][bundle], allow_nan=False)
        return result

    def test_complete_supplement_replaces_pair_and_metadata_atomically(self):
        primary = self.document(**self.structure(new=100))
        extra = self.document(**self.structure(new=200, secondary=50))
        values, _, proof, raw, _, _ = self.connect(primary, extra)
        self.assertEqual((values['new_shares'], values['secondary_shares']), (200, 50))
        bundle = proof['resolved_structure']
        self.assertEqual(bundle['offering_structure_rcept_no'], self.extra_meta['rcept_no'])
        self.assertEqual(bundle['new_shares_parse_evidence'], '200')
        self.assertEqual(bundle['secondary_shares_parse_evidence'], '50')
        self.assertEqual(bundle['available_at'], '2020-01-03T00:00:00+09:00')
        self.assertEqual(proof['field_sources']['new_shares'], proof['field_sources']['secondary_shares'])
        self.assertEqual(raw['new_shares'], 100)

    def test_partial_correction_uses_same_day_latest_related_prospectus_price(self):
        from data.pipelines.historical_event_recovery import source_record
        primary={'offering_price':None,'offering_price_review_status':'missing',
                 'offering_price_finality':'unknown','lockup_commitment_ratio':.9}
        extra=self.document(institutional_demand_ratio=600,institutional_demand_parser_validation_status='structurally_verified',
                            institutional_demand_rule_id='aggregate')
        values,meta,proof,raw,_,_=self.connect(primary,extra)
        self.assertEqual(values['offering_price'],1000)
        self.assertEqual(proof['price_resolution_status'],'latest_related_prospectus_price_verified')
        self.assertEqual(proof['field_sources']['offering_price']['receipt'],self.extra_meta['rcept_no'])
        self.assertIsNone(values['lockup_commitment_ratio'])
        event=SimpleNamespace(event_id='one',ticker='123456',corp_name='Historical',listing_date='2020-02-01')
        record=source_record(event,values,meta,proof,raw)
        self.assertEqual(record['offering_price_rcept_no'],self.extra_meta['rcept_no'])
        self.assertEqual(record['institutional_rcept_no'],self.extra_meta['rcept_no'])
        self.assertEqual(record['offering_price_review_status'],'verified_currency_unit')

    def test_partial_correction_does_not_use_older_or_unconfirmed_price(self):
        primary={'offering_price':None,'offering_price_review_status':'missing','offering_price_finality':'unknown'}
        for extra,metadata in [(self.document(),{**self.extra_meta,'rcept_dt':'2019-12-31'}),
                               ({**self.document(),'offering_price_finality':'preliminary_price_language'},self.extra_meta)]:
            with self.assertRaises(ValueError):self.connect(primary,extra,metadata)
        with self.assertRaisesRegex(ValueError,'krx_mismatch'):
            self.connect(self.document(offering_price=999),self.document())

    def test_record_never_reselects_a_different_institutional_source(self):
        from data.pipelines.historical_event_recovery import source_record
        primary=self.document(lockup_commitment_ratio=.1,lockup_parser_validation_status='structurally_verified',lockup_rule_id='aggregate')
        extra=self.document(lockup_commitment_ratio=.3,lockup_parser_validation_status='structurally_verified',lockup_rule_id='aggregate')
        values,meta,proof,raw,_,_=self.connect(primary,extra)
        event=SimpleNamespace(event_id='one',ticker='123456',corp_name='Historical',listing_date='2020-02-01')
        record=source_record(event,values,meta,proof,raw)
        self.assertEqual(record['lockup_commitment_ratio'],values['lockup_commitment_ratio'])
        self.assertEqual(record['lockup_rcept_no'],self.extra_meta['rcept_no'])

    def test_notice_before_after_price_is_not_the_complete_current_statement(self):
        values,_,proof,_,_,_=self.connect(self.document(offering_price=500),self.document(),
            primary_scope='partial_or_terms_notice')
        self.assertEqual(values['offering_price'],1000)
        self.assertEqual(proof['field_sources']['offering_price']['receipt'],self.extra_meta['rcept_no'])
        with self.assertRaises(ValueError):
            self.connect(self.document(),primary_scope='partial_or_terms_notice')

    def test_incomplete_documents_never_form_cross_source_pair(self):
        values, _, proof, _, _, _ = self.connect(
            self.document(**self.structure(new=100)), self.document(**self.structure(secondary=50)))
        self.assertIsNone(values['new_shares'])
        self.assertIsNone(values['secondary_shares'])
        self.assertEqual(proof['resolved_structure']['offering_structure_resolution_status'], 'latest_partial_structure_requires_review')
        self.assertNotIn('secondary_shares', proof['field_sources'])

    def test_new_partial_or_unverified_structure_cannot_revive_old_pair(self):
        for fields in [self.structure(secondary=50),
                       self.structure(new=200, secondary=50) | {'secondary_shares_parser_validation_status': 'candidate_rejected'},
                       self.structure(new=200, secondary=50) | {'new_shares_parse_method': None}]:
            with self.subTest(fields=fields):
                values, _, proof, _, _, _ = self.connect(
                    self.document(**self.structure(new=100, secondary=0)), self.document(**fields))
                self.assertEqual((values['new_shares'], values['secondary_shares']), (None, None))
                self.assertIsNone(proof['resolved_structure']['offering_structure_rcept_no'])

    def test_newer_complete_structure_wins_and_older_supplement_does_not(self):
        primary = self.document(**self.structure(new=100, secondary=0))
        extra = self.document(**self.structure(new=200, secondary=50))
        for extra_meta, expected in [(self.extra_meta, (200, 50)),
                                     ({**self.extra_meta, 'rcept_dt': '2019-12-30'}, (100, 0))]:
            with self.subTest(extra_meta=extra_meta):
                values, _, _, _, _, _ = self.connect(primary, extra, extra_meta)
                self.assertEqual((values['new_shares'], values['secondary_shares']), expected)

    def test_newer_float_ratio_replaces_old_value_and_provenance(self):
        primary = self.document(public_float_ratio_disclosed=.2, public_float_parse_method='disclosed_public_float_ratio_direct_context')
        extra = self.document(public_float_ratio_disclosed=.3, public_float_parse_method='disclosed_public_float_ratio_direct_context')
        values, _, proof, _, _, _ = self.connect(primary, extra)
        self.assertEqual(values['public_float_ratio_disclosed'], .3)
        bundle = proof['resolved_float']
        self.assertEqual(bundle['public_float_rcept_no'], self.extra_meta['rcept_no'])
        self.assertEqual(bundle['public_float_rcept_dt'], '2020-01-02T00:00:00')
        self.assertEqual(proof['field_sources']['public_float_ratio_disclosed']['source_url'], bundle['source_url'])

    def test_latest_float_conflict_clears_all_old_value_and_denominator_fields(self):
        primary = self.document(public_float_ratio_disclosed=.2, public_float_shares=100,
                                public_float_parse_method='disclosed_public_float_ratio_direct_context',
                                total_post_listing_shares=500, total_post_listing_shares_parser_validation_status='structurally_verified')
        extra = self.document(public_float_parse_method='conflicting_float_ratios_review_required', public_float_parse_evidence='[0.3, 0.4]')
        values, _, proof, raw, _, _ = self.connect(primary, extra)
        self.assertIsNone(values['public_float_ratio_disclosed'])
        bundle = proof['resolved_float']
        self.assertEqual(bundle['public_float_resolution_status'], 'latest_conflict_blocks_older_fallback')
        for field in ('public_float_shares', 'public_float_ratio_disclosed', 'total_post_listing_shares',
                      'total_post_listing_shares_parser_validation_status', 'public_float_rcept_no', 'source_url', 'available_at'):
            self.assertIsNone(bundle[field], field)
        self.assertEqual(bundle['public_float_parse_evidence'], '[0.3, 0.4]')
        self.assertEqual(raw['public_float_ratio_disclosed'], .2)

    def test_shares_only_float_preserves_matching_denominator_and_metadata(self):
        primary = self.document(public_float_ratio_disclosed=.2, public_float_parse_method='disclosed_public_float_ratio_direct_context',
                                total_post_listing_shares=1000)
        extra = self.document(public_float_shares=100, public_float_parse_method='disclosed_public_float_shares_direct_context',
                              public_float_parse_evidence='100 shares', total_post_listing_shares=200,
                              total_post_listing_shares_parse_method='direct_table_row_v1',
                              total_post_listing_shares_parse_evidence='200 shares',
                              total_post_listing_shares_parser_validation_status='structurally_verified')
        values, _, proof, _, _, _ = self.connect(primary, extra)
        self.assertIsNone(values['public_float_ratio_disclosed'])
        bundle = proof['resolved_float']
        self.assertEqual(bundle['public_float_shares'], 100)
        self.assertEqual(bundle['total_post_listing_shares'], 200)
        self.assertEqual(bundle['total_post_listing_shares_parse_evidence'], '200 shares')
        self.assertEqual(bundle['public_float_rcept_no'], self.extra_meta['rcept_no'])
        features = FeatureEngineer('phase2')._calc_supply_structure_features(pd.DataFrame([bundle]))
        self.assertEqual(features.iloc[0]['float_share_ratio'], .5)

    def test_unverified_new_float_denominator_blocks_old_fallback(self):
        primary = self.document(public_float_ratio_disclosed=.2, public_float_parse_method='disclosed_public_float_ratio_direct_context')
        extra = self.document(public_float_shares=100, public_float_parse_method='disclosed_public_float_shares_direct_context',
                              total_post_listing_shares=200, total_post_listing_shares_parser_validation_status='value_not_found')
        values, _, proof, _, _, _ = self.connect(primary, extra)
        self.assertIsNone(values['public_float_ratio_disclosed'])
        self.assertEqual(proof['resolved_float']['public_float_resolution_status'], 'latest_source_denominator_unverified')

    def test_no_supplement_or_transport_failure_resolves_primary_bundles(self):
        primary = self.document(**self.structure(new=100, secondary=0), public_float_ratio_disclosed=.2,
                                public_float_parse_method='disclosed_public_float_ratio_direct_context')
        for failure in [None, requests.Timeout('synthetic transport failure')]:
            with self.subTest(failure=failure):
                values, _, proof, _, _, _ = self.connect(primary, failure=failure)
                self.assertEqual(values['public_float_ratio_disclosed'], .2)
                self.assertEqual(proof['resolved_float']['public_float_rcept_no'], self.primary_meta['rcept_no'])
                self.assertEqual(values['secondary_shares'], 0)

    def test_after_cutoff_or_price_mismatched_supplement_cannot_supply_bundles(self):
        primary = self.document(**self.structure(new=100, secondary=0))
        extra = self.document(**self.structure(new=200, secondary=50), public_float_ratio_disclosed=.3,
                              public_float_parse_method='disclosed_public_float_ratio_direct_context')
        for document, metadata in [(extra, {**self.extra_meta, 'rcept_dt': '2020-01-31'})]:
            with self.subTest(metadata=metadata, price=document['offering_price']):
                values, _, proof, _, _, _ = self.connect(primary, document, metadata)
                self.assertEqual((values['new_shares'], values['secondary_shares']), (100, 0))
                self.assertIsNone(values['public_float_ratio_disclosed'])
                self.assertNotIn('supplementary', proof)
        with self.assertRaisesRegex(ValueError,'krx_mismatch'):
            self.connect(primary,{**extra,'offering_price':999})

    def test_latest_complete_institutional_source_wins_without_mutating_raw_values(self):
        primary = self.document(institutional_demand_ratio=500, institutional_demand_parser_validation_status='candidate_rejected',
                                institutional_demand_rule_id='rejected', lockup_commitment_ratio=.1,
                                lockup_parser_validation_status='structurally_verified', lockup_rule_id='aggregate')
        extra = self.document(institutional_demand_ratio=600, institutional_demand_parser_validation_status='structurally_verified',
                              institutional_demand_rule_id='aggregate', lockup_commitment_ratio=.3,
                              lockup_parser_validation_status='structurally_verified', lockup_rule_id='aggregate')
        values, _, proof, raw, _, _ = self.connect(primary, extra)
        self.assertEqual(values['institutional_demand_ratio'], 600)
        self.assertEqual(raw['institutional_demand_ratio'], 500)
        self.assertEqual(raw['institutional_demand_parser_validation_status'], 'candidate_rejected')
        self.assertEqual(values['lockup_commitment_ratio'], .3)
        self.assertEqual(proof['field_sources']['institutional_demand_ratio']['receipt'], self.extra_meta['rcept_no'])
        primary['institutional_demand_parser_validation_status'] = 'structurally_verified'
        values, _, proof, _, _, _ = self.connect(primary, extra)
        self.assertEqual(values['institutional_demand_ratio'], 600)
        self.assertEqual(proof['field_sources']['institutional_demand_ratio']['receipt'],self.extra_meta['rcept_no'])

    def test_bundle_scalar_normalization_is_strict_json_and_preserves_explicit_nulls(self):
        bundle = _bundle_scalars({'date': pd.Timestamp('2020-01-02'), 'missing': pd.NA,
                                  'not_a_time': pd.NaT, 'nan': float('nan'), 'inf': float('inf'),
                                  'negative_inf': float('-inf'), 'zero': 0, 'fraction': .5})
        self.assertEqual(bundle['date'], '2020-01-02T00:00:00')
        self.assertEqual(bundle['zero'], 0)
        self.assertEqual(bundle['fraction'], .5)
        for field in ('missing', 'not_a_time', 'nan', 'inf', 'negative_inf'):
            self.assertIsNone(bundle[field], field)
        self.assertEqual(json.loads(json.dumps(bundle, allow_nan=False)), bundle)

    def test_bundle_denominator_nonfinite_metadata_is_explicit_null(self):
        primary = self.document(**self.structure(new=100, secondary=0), public_float_ratio_disclosed=.2,
                                public_float_parse_method='disclosed_public_float_ratio_direct_context',
                                total_post_listing_shares=float('inf'),
                                total_post_listing_shares_parse_evidence=pd.NA)
        _, _, proof, _, _, _ = self.connect(primary)
        self.assertIsNone(proof['resolved_float']['total_post_listing_shares'])
        self.assertIsNone(proof['resolved_float']['total_post_listing_shares_parse_evidence'])

    def test_structure_cannot_overwrite_independently_selected_float_denominator(self):
        primary = self.document(**self.structure(new=100, secondary=0),
                                public_float_shares=100, public_float_parse_method='disclosed_public_float_shares_direct_context',
                                total_post_listing_shares=200, total_post_listing_shares_parser_validation_status='structurally_verified')
        extra = self.document(**self.structure(new=200, secondary=50), total_post_listing_shares=1000)
        _, _, proof, _, _, _ = self.connect(primary, extra)
        self.assertNotIn('total_post_listing_shares', proof['resolved_structure'])
        self.assertEqual(proof['resolved_structure']['offering_structure_rcept_no'], self.extra_meta['rcept_no'])
        self.assertEqual(proof['resolved_float']['public_float_rcept_no'], self.primary_meta['rcept_no'])
        for bundles in [(proof['resolved_float'], proof['resolved_structure']),
                        (proof['resolved_structure'], proof['resolved_float'])]:
            merged = {**bundles[0], **bundles[1]}
            features = FeatureEngineer('phase2')._calc_supply_structure_features(pd.DataFrame([merged]))
            self.assertEqual(features.iloc[0]['float_share_ratio'], .5)


if __name__ == '__main__':
    unittest.main()
