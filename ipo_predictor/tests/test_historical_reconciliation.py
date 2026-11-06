import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from data.pipelines.historical_reconciliation import collect_connections
from data.pipelines.historical_financial_reconciliation import missing_category


class HistoricalReconciliationTests(unittest.TestCase):
    def inputs(self):
        return {'events':pd.DataFrame([dict(event_id='e',ticker='000001',listing_date='2020-01-10',
                    offering_price=10000,corp_name='Issuer',event_class='general_ipo')]),
                'records':pd.DataFrame([dict(event_id='e',corp_code='01234567',rcept_no='20200101000001')]),
                'features':pd.DataFrame([dict(event_id='e',offering_price_review_status='needs_review')]),
                'input_sha256':{'base':'pinned'}}

    def result(self):
        return {'status':'official_sources_connected','sources':[], 'record':{'event_id':'e'},'proof':{}}

    def test_success_reuses_source_checked_checkpoint_without_new_network(self):
        with tempfile.TemporaryDirectory() as root, patch('data.pipelines.historical_reconciliation.load_inputs',return_value=self.inputs()), \
                patch('data.pipelines.historical_reconciliation.capture_sources') as capture, \
                patch('data.pipelines.historical_reconciliation.connect_one',return_value=self.result()) as connect:
            first=collect_connections(Path(root)/'raw',Path(root)/'processed',dart=object())
            second=collect_connections(Path(root)/'raw',Path(root)/'processed',dart=object())
            self.assertEqual(connect.call_count,1)
            self.assertEqual(first['version'],second['version'])
            self.assertGreaterEqual(capture.call_count,2)
            self.assertFalse(first['training_executed'])

    def test_partial_sample_does_not_publish_full_pointer_and_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as root, patch('data.pipelines.historical_reconciliation.load_inputs',return_value=self.inputs()), \
                patch('data.pipelines.historical_reconciliation.capture_sources'), \
                patch('data.pipelines.historical_reconciliation.connect_one',return_value=self.result()):
            raw,processed=Path(root)/'raw',Path(root)/'processed'
            collect_connections(raw,processed,dart=object(),event_limit=1)
            self.assertFalse((processed/'current_historical_source_connections.json').exists())
            checkpoint=next((raw/'historical_reconciliation_connections').glob('*.json'))
            body=json.loads(checkpoint.read_text());body['result']['record']['tampered']=True
            checkpoint.write_text(json.dumps(body))
            with self.assertRaisesRegex(RuntimeError,'checkpoint_integrity_failed'):
                collect_connections(raw,processed,dart=object())

    def test_source_failure_does_not_permanently_exclude_the_company(self):
        with tempfile.TemporaryDirectory() as root, patch('data.pipelines.historical_reconciliation.load_inputs',return_value=self.inputs()), \
                patch('data.pipelines.historical_reconciliation.capture_sources'), \
                patch('data.pipelines.historical_reconciliation.connect_one',side_effect=[ValueError('official_source_not_cached'),self.result()]) as connect:
            raw,processed=Path(root)/'raw',Path(root)/'processed'
            first=collect_connections(raw,processed,dart=object())
            second=collect_connections(raw,processed,dart=object())
            self.assertEqual(first['status_counts'],{'pending_source_download':1})
            self.assertEqual(second['status_counts'],{'official_sources_connected':1})
            self.assertEqual(connect.call_count,2)

    def test_missing_financial_summary_is_not_issuer_wide_nonpublication(self):
        self.assertEqual(missing_category('annual_financial_table_format_unhandled',
                         {'selected_document_explicit_summary_absence':True}),
                         'summary_explicitly_absent_detailed_statements_not_supported')
        self.assertEqual(missing_category('financial_public_source_access_failed',{}),'source_access_failed')
        self.assertEqual(missing_category('annual_financial_table_format_unhandled',
                         {'rejected_table_or_column_counts':{'accounting_basis_unverified':2}}),
                         'accounting_basis_unverified')

    def test_failed_full_batch_preserves_last_complete_pointer(self):
        with tempfile.TemporaryDirectory() as root, patch('data.pipelines.historical_reconciliation.load_inputs',return_value=self.inputs()), \
                patch('data.pipelines.historical_reconciliation.capture_sources'), \
                patch('data.pipelines.historical_reconciliation.connect_one',return_value=self.result()):
            raw,processed=Path(root)/'raw',Path(root)/'processed'
            collect_connections(raw,processed,dart=object())
            pointer=processed/'current_historical_source_connections.json'
            previous=pointer.read_bytes()
            with patch('data.pipelines.historical_reconciliation.connection_contract',return_value={'changed':'parser'}), \
                    patch('data.pipelines.historical_reconciliation.connect_one',side_effect=ValueError('official_source_not_cached')):
                failed=collect_connections(raw,processed,dart=object())
            self.assertEqual(failed['status_counts'],{'pending_source_download':1})
            self.assertEqual(pointer.read_bytes(),previous)

    def test_partial_correction_is_resolved_in_normal_connection_path(self):
        from data.pipelines.historical_reconciliation import connect_one
        event=next(self.inputs()['events'].itertuples(index=False))
        record=self.inputs()['records'].iloc[0]
        with patch('data.pipelines.historical_reconciliation.connect_event',side_effect=ValueError('price_source_predates_latest_partial_correction')), \
                patch('data.pipelines.correction_views.connect_corrected_notice',return_value=(self.result(),{})) as corrected:
            self.assertEqual(connect_one(object(),Path('/tmp'),event,record,True),self.result())
            corrected.assert_called_once()
