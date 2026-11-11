import unittest
import pandas as pd
from scripts.build_historical_observation_ledger import bind_event, select_asof


class HistoricalLedgerTests(unittest.TestCase):
    def test_publication_uses_official_date_not_receipt_prefix(self):
        rows = pd.DataFrame([{'event_id': 'e', 'corp_code': '01234567', 'rcept_no': '20220103000001',
                              'rcept_dt': '2022-01-04', 'listing_date': '2022-01-10'}])
        content = "<title>변경회사/투자설명서/2022.01.04</title>openCorpInfoNew('01234567', 'a')"
        self.assertEqual(bind_event({'receipt': '20220103000001'}, rows, content)['published_date'], '2022-01-04')
        with self.assertRaises(ValueError):
            bind_event({'receipt': '20220103000001'}, rows, content.replace('2022.01.04', '2022.01.05'))

    def test_corporate_receipt_and_listing_identity(self):
        rows = pd.DataFrame([{'event_id': 'event', 'corp_code': '01234567', 'rcept_no': '20220103000001',
                              'rcept_dt': '2022-01-03', 'listing_date': '2022-01-10'}])
        record = {'receipt': '20220103000001'}
        result = bind_event(record, rows, "openCorpInfoNew('01234567', 'a')")
        self.assertEqual(result['available_at'], '2022-01-04T00:00:00+09:00')
        with self.assertRaises(ValueError):
            bind_event(record, rows, "openCorpInfoNew('11111111', 'a')")
        rows.loc[0, 'listing_date'] = '2022-01-04'
        with self.assertRaises(ValueError):
            bind_event(record, rows, "openCorpInfoNew('01234567', 'a')")

    def test_asof_never_uses_future_or_backfills_missing_correction(self):
        def row(day, value, key):
            return {'event_id': 'event', 'feature_name': 'x', 'available_at': f'2022-01-{day}T00:00:00+09:00',
                    'value': value, 'observation_id': key}
        rows = [row('04', 100, 'a'), row('06', 200, 'b')]
        self.assertEqual(select_asof(rows, 'event', 'x', '2022-01-05T12:00:00+09:00')['value'], 100)
        rows.append(row('08', None, 'c'))
        self.assertIsNone(select_asof(rows, 'event', 'x', '2022-01-09T12:00:00+09:00')['value'])
        rows.append(row('06', 300, 'd'))
        self.assertIsNone(select_asof(rows, 'event', 'x', '2022-01-07T12:00:00+09:00')['value'])
        with self.assertRaises(ValueError):
            select_asof(rows, 'event', 'x', '2022-01-07')
