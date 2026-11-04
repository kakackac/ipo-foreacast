import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import pandas as pd

from data.pipelines.historical_ipo_pipeline import HistoricalIPOPipeline


class SnapshotPriceTests(unittest.TestCase):
    def event(self, ticker='123450', day='2020-01-02'):
        return {'event_id': ticker+'|'+day, 'ticker': ticker, 'listing_date': day,
                'market': 'KOSDAQ', 'corp_name': '테스트', 'event_class': 'general_ipo'}

    def price(self, ticker='123450'):
        return {'ticker': ticker, 'listing_date': '20200102', 'open_price': 100, 'close_price': 110,
                'price_match_status': 'matched', 'price_match_method': 'ticker', 'market': 'KOSDAQ',
                'price_raw_response_evidence': '{}', 'price_matched_ticker': ticker}

    def test_completed_cache_survives_partial_failure_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); snapshot = root/'events.parquet'
            pd.DataFrame([self.event(),self.event('234560')]).to_parquet(snapshot,index=False)
            before = snapshot.read_bytes()
            collector = Mock(is_configured=True)
            collector.get_listing_day_price.side_effect = [self.price(), RuntimeError('secret')]
            pipeline = HistoricalIPOPipeline(krx_collector=collector,raw_dir=root/'out',processed_dir=root/'proc')
            first = pipeline.collect_snapshot_prices(snapshot)
            self.assertEqual(first['status'],'partial_failure_retryable')
            self.assertNotIn('secret',str(first))
            collector.get_listing_day_price.side_effect = [self.price('234560')]
            second = pipeline.collect_snapshot_prices(snapshot)
            self.assertEqual(second['status'],'complete')
            self.assertEqual(second['validation_counts'],{'official_price_verified':2})
            self.assertEqual(collector.get_listing_day_price.call_count,3)
            self.assertEqual(snapshot.read_bytes(),before)
            pipeline.collect_snapshot_prices(snapshot)
            self.assertEqual(collector.get_listing_day_price.call_count,3)

    def test_future_and_unsupported_market_do_not_call_api(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); snapshot=root/'events.parquet'
            pd.DataFrame([self.event(day='2099-01-01'),dict(self.event(),market='KONEX')]).to_parquet(snapshot,index=False)
            collector=Mock(is_configured=False)
            pipeline=HistoricalIPOPipeline(krx_collector=collector,raw_dir=root/'out',processed_dir=root/'proc')
            self.assertEqual(pipeline.collect_snapshot_prices(snapshot)['result_rows'],0)
            collector.get_listing_day_price.assert_not_called()


if __name__=='__main__': unittest.main()
