import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data.pipelines.data_delivery import digest
from data.pipelines.event_delivery import merge_frames, run, current_event_delivery
from data.pipelines.historical_ipo_pipeline import HistoricalIPOPipeline


class EventDeliveryTests(unittest.TestCase):
    def frames(self):
        common = {'market':'KOSDAQ','listing_type':'신규상장','security_type':'주권',
                  'country':'대한민국','offering_price':2000,'price_resolution_status':'official_price_verified'}
        base = pd.DataFrame([dict(common,event_id='old',ticker='123450',corp_name='기존',listing_date='2026-01-01',
                                  krx_standard_code='KR7123456000')])
        snapshot = pd.DataFrame([dict(common,event_id='new',ticker='234560',corp_name='신규',listing_date='2026-02-02',
                                      krx_standard_code='234560',open_price=4000.,close_price=3000.)])
        audit = pd.DataFrame([{'event_id':'new','ticker':'234560','listing_date':'2026-02-02','price_market':'KOSDAQ',
             'price_match_status':'matched','price_matched_ticker':None,'price_matched_isu_cd':'234560',
             'listing_open_price':4000.,'listing_close_price':3000.,
             'price_raw_response_evidence':json.dumps({'KOSDAQ':{'row_count':100,'sha256':'a'*64}})}])
        return base,snapshot,audit

    def test_union_and_standard_code_are_not_confused_with_trading_code(self):
        base,snapshot,audit = self.frames()
        merged,added,links = merge_frames(base,snapshot,audit)
        self.assertEqual(len(merged),2)
        self.assertEqual(added.event_id.tolist(),['new'])
        self.assertTrue(pd.isna(added.krx_standard_code.iloc[0]))
        self.assertEqual(added.legacy_standard_code_value.iloc[0],'234560')
        self.assertFalse(links.model_eligible.any())
        prices = snapshot[['ticker','listing_date','market','open_price','close_price']].assign(isu_cd='234560')
        attached = HistoricalIPOPipeline._attach_prices(snapshot,prices)
        self.assertTrue(pd.isna(attached.krx_standard_code.iloc[0]))
        # A genuine ISIN must survive a short-code daily response.
        snapshot['krx_standard_code']='KR7234560000'
        self.assertEqual(HistoricalIPOPipeline._attach_prices(snapshot,prices).krx_standard_code.iloc[0],'KR7234560000')

    def test_name_discovery_never_approves_a_dart_identity(self):
        base,snapshot,audit = self.frames()
        registry = {'rows':[{'corp_name':'신규','corp_code':'12345678','rcept_no':'20260101000001',
                            'subscription_end':'2026-01-15'}]}
        links=merge_frames(base,snapshot,audit,registry)[2]
        self.assertIn('candidate_only',links.link_status.iloc[0])
        self.assertFalse(links.model_eligible.iloc[0])

    def test_price_code_market_value_and_population_conflicts_fail_closed(self):
        for field,value in [('price_matched_isu_cd','999999'),('price_market','KOSPI'),('listing_open_price',100.)]:
            base,snapshot,audit=self.frames();audit[field]=value
            with self.assertRaises(ValueError):merge_frames(base,snapshot,audit)
        base,snapshot,audit=self.frames()
        with self.assertRaises(ValueError):merge_frames(base,pd.concat([snapshot,snapshot]),audit)

    def test_stable_event_id_and_isin_survive_an_updated_display_name(self):
        base,snapshot,audit=self.frames()
        snapshot['ticker']='123450';snapshot['listing_date']='2026-01-01'
        audit['ticker']='123450';audit['listing_date']='2026-01-01';audit['price_matched_isu_cd']='123450'
        merged,added,_=merge_frames(base,snapshot,audit)
        self.assertEqual(merged.event_id.tolist(),['old'])
        self.assertEqual(merged.krx_standard_code.iloc[0],'KR7123456000')
        self.assertEqual(len(added),0)

    def test_version_roundtrip_and_source_or_artifact_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);raw=root/'raw';processed=root/'processed';prices=root/'prices'
            raw.mkdir();processed.mkdir();prices.mkdir()
            base,snapshot,audit=self.frames()
            base.to_parquet(raw/'krx_official_event_master.parquet',index=False)
            path=root/'snapshot.parquet';snapshot.to_parquet(path,index=False)
            snapshot.to_parquet(prices/'events_with_prices.parquet',index=False)
            audit.to_parquet(prices/'krx_listing_price_audit.parquet',index=False)
            manifest={'status':'complete','snapshot_sha256':digest(path),'result_rows':1,'due_events':1,
                      'started_at':'2026-02-02T21:00:00+09:00','finished_at':'2026-02-02T21:01:00+09:00'}
            (prices/'snapshot_prices_manifest.json').write_text(json.dumps(manifest))
            first=run(raw,processed,path,prices);second=run(raw,processed,path,prices)
            self.assertEqual(first,second)
            self.assertEqual(current_event_delivery(processed)[1]['added_rows'],1)
            (Path(first['path'])/'events.parquet').write_bytes(b'tampered')
            with self.assertRaises(RuntimeError):current_event_delivery(processed)

    def test_regular_session_prices_are_not_accepted_before_2100(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);raw=root/'raw';processed=root/'processed';prices=root/'prices'
            for p in (raw,processed,prices):p.mkdir()
            base,snapshot,audit=self.frames();base.to_parquet(raw/'krx_official_event_master.parquet',index=False)
            path=root/'snapshot.parquet';snapshot.to_parquet(path,index=False)
            snapshot.to_parquet(prices/'events_with_prices.parquet',index=False);audit.to_parquet(prices/'krx_listing_price_audit.parquet',index=False)
            (prices/'snapshot_prices_manifest.json').write_text(json.dumps({'status':'complete','snapshot_sha256':digest(path),
                'result_rows':1,'due_events':1,'finished_at':'2026-02-02T20:59:00+09:00'}))
            with self.assertRaisesRegex(ValueError,'before_listing_day_2100'):run(raw,processed,path,prices)

    def test_known_official_event_remains_in_master_without_a_collected_price(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);raw=root/'raw';processed=root/'processed';prices=root/'prices'
            for p in (raw,processed,prices):p.mkdir()
            base,snapshot,audit=self.frames();base.to_parquet(raw/'krx_official_event_master.parquet',index=False)
            pending=snapshot.copy();pending['event_id']='pending';pending['ticker']='345670';pending['listing_date']='2026-02-03'
            path=root/'snapshot.parquet';pd.concat([snapshot,pending]).to_parquet(path,index=False)
            snapshot.to_parquet(prices/'events_with_prices.parquet',index=False);audit.to_parquet(prices/'krx_listing_price_audit.parquet',index=False)
            (prices/'snapshot_prices_manifest.json').write_text(json.dumps({'status':'complete','snapshot_sha256':digest(path),
                'result_rows':1,'due_events':1,'started_at':'2026-02-02T21:00:00+09:00','finished_at':'2026-02-02T21:01:00+09:00'}))
            result=run(raw,processed,path,prices)
            self.assertEqual(result['rows'],3)
            frame=pd.read_parquet(Path(result['path'])/'events.parquet')
            row=frame[frame.event_id.eq('pending')].iloc[0]
            self.assertTrue(pd.isna(row.open_price))
            self.assertEqual(row.price_resolution_status,'pending_collection')
            self.assertFalse(row.model_eligible)


if __name__=='__main__':unittest.main()
