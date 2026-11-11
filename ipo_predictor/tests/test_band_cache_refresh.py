import hashlib
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from data.pipelines.band_source_resolver import band_retry_dependencies, refresh_cached_band, unchanged_band_rules
from data.pipelines.data_delivery import digest
from data.pipelines.financial_history import atomic_json


class BandCacheRefreshTests(unittest.TestCase):
    def test_policy_reuse_rejects_changed_extraction_even_with_a_pinned_old_source(self):
        from data.pipelines.historical_reconciliation import connection_contract
        import data.pipelines.band_source_resolver as bands
        path=Path(bands.__file__);old=path.read_text();sha=digest(path);contract=connection_contract()
        read=Path.read_text
        def changed(p,*args,**kwargs):
            source=read(p,*args,**kwargs)
            return source.replace('historical_band_later_range_conflict','changed_range_approval_rule') if p==path else source
        with patch('data.pipelines.lockup_reconciliation.parser_snapshot',return_value=old):
            self.assertTrue(unchanged_band_rules(Path('/tmp'),{'amendment':{'repair':sha}},contract,contract))
            with patch.object(Path,'read_text',new=changed):
                self.assertFalse(unchanged_band_rules(Path('/tmp'),{'amendment':{'repair':sha}},contract,contract))

    def fixture(self, root, missing=1):
        root=root.resolve()
        raw=root/'raw';processed=root/'processed';parent=processed/'historical_reconciliations/base'
        parent.mkdir(parents=True);raw.mkdir()
        contract=dict(band='b'*64,adapter='a'*64,record='c'*64,resolver='d'*64)
        rows=[];resolved={}
        for i in range(1,missing+2):
            receipt='2024010200000'+str(i)
            url='https://dart.fss.or.kr/dsaf001/main.do?rcpNo='+receipt
            record={'rcept_no':receipt,'listing_date':'2024-02-01','offering_price':100,
                'price_band_low':80 if i==1 else None,'price_band_high':120 if i==1 else None,
                'price_band_resolution_status':'verified_currency_band_source_selected' if i==1 else 'no_verified_currency_band_in_admitted_documents',
                'price_band_resolution_evidence':json.dumps({'historical_band':None if i==1 else {
                    'reason':'official_source_not_cached','failed_url':url}})}
            proof={'family':[{'rcept_no':receipt,'rcept_dt':'2024-01-02'}],
                   'sources':[],'corp_code':'00000001'}
            rows.append({'event_id':str(i),'corp_name':'Company '+str(i),'status':'official_sources_connected',
                         'reason':None,'record':json.dumps(record),'proof':json.dumps(proof)})
            if i>1:
                text='synthetic unit-test original '+str(i)
                atomic_json(raw/'dart_connection_sources'/(hashlib.sha256(url.encode()).hexdigest()+'.json'),
                    {'url':url,'text':text,'sha256':hashlib.sha256(text.encode()).hexdigest(),'collected_at':'2026-11-08'})
                resolved[receipt]={'price_band_low':80,'price_band_high':120,
                    'price_band_parse_method':'direct_currency_range_v1',
                    'price_band_parse_evidence':'["희망공모가액 80원 ~ 120원"]',
                    'price_band_validation_status':'direct_currency_band',
                    'price_band_rcept_no':receipt,'price_band_rcept_dt':'2024-01-02',
                    'price_band_source_url':url,'price_band_source_sha256':'e'*64,
                    'price_band_resolution_status':'verified_currency_band_source_selected',
                    'price_band_resolution_evidence':'{}'}
        rows=pd.DataFrame(rows)
        rows.to_parquet(parent/'connection_results.parquet',index=False)
        events=pd.DataFrame({'event_id':rows.event_id,'event_class':'general_ipo'})
        events.to_parquet(parent/'events.parquet',index=False)
        report={'version':'base','inputs':{'contract':contract,'inputs':{'sample':'stable'}},'replay_contract':contract}
        atomic_json(parent/'manifest.json',report)
        atomic_json(processed/'current_historical_source_connections.json',{'manifest':'historical_reconciliations/base/manifest.json',
            'sha256':digest(parent/'manifest.json')})
        return raw,processed,parent,report,contract,rows,events,resolved

    def patches(self, stack, parent, report, contract, events):
        stack.enter_context(patch('data.pipelines.reconciled_dataset.pin_connections',return_value=(parent,report)))
        stack.enter_context(patch('data.pipelines.historical_reconciliation.connection_contract',return_value=contract))
        stack.enter_context(patch('data.pipelines.historical_reconciliation.load_inputs',return_value={
            'input_sha256':{'sample':'stable'},'events':events}))
        network=stack.enter_context(patch('requests.sessions.Session.request',side_effect=AssertionError('unexpected_network')))
        return network

    def test_new_original_retries_only_held_event_without_parser_change_then_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw,p,parent,report,contract,rows,events,resolved=self.fixture(Path(temporary))
            self.assertEqual(set(band_retry_dependencies(raw,rows)),{'2'})
            with ExitStack() as stack:
                network=self.patches(stack,parent,report,contract,events)
                resolve=stack.enter_context(patch('data.pipelines.band_source_resolver.resolve_offering_band',
                    side_effect=lambda raw,m,*args,**kwargs:resolved[m['rcept_no']]))
                result=refresh_cached_band(raw,p)
                self.assertEqual(resolve.call_count,1);network.assert_not_called()
            self.assertEqual(result['band_field_reprocessing']['reparsed_events'],1)
            self.assertEqual(result['band_field_reprocessing']['unchanged_events_reused'],1)
            after=pd.read_parquet(Path(result['path'])/'connection_results.parquet')
            self.assertEqual(after.loc[0,'record'],rows.loc[0,'record'])
            self.assertEqual(after.loc[0,'proof'],rows.loc[0,'proof'])
            self.assertEqual(json.loads(after.loc[1,'record'])['offering_price'],100)
            before=(p/'current_historical_source_connections.json').read_bytes()
            with ExitStack() as stack:
                network=self.patches(stack,Path(result['path']),result,contract,events)
                resolve=stack.enter_context(patch('data.pipelines.band_source_resolver.resolve_offering_band'))
                self.assertIsNone(refresh_cached_band(raw,p));resolve.assert_not_called();network.assert_not_called()
            self.assertEqual((p/'current_historical_source_connections.json').read_bytes(),before)

    def test_partial_failure_preserves_current_version_and_resumes_completed_event(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw,p,parent,report,contract,rows,events,resolved=self.fixture(Path(temporary),missing=2)
            pointer=p/'current_historical_source_connections.json';before=pointer.read_bytes()
            original=digest(parent/'connection_results.parquet')
            counter=0
            def fail_after_first(raw,m,*args,**kwargs):
                nonlocal counter
                counter+=1
                if counter==2:raise RuntimeError('simulated_interruption')
                return resolved[m['rcept_no']]
            with ExitStack() as stack:
                self.patches(stack,parent,report,contract,events)
                stack.enter_context(patch('data.pipelines.band_source_resolver.resolve_offering_band',side_effect=fail_after_first))
                with self.assertRaisesRegex(RuntimeError,'simulated_interruption'):refresh_cached_band(raw,p)
            self.assertEqual(pointer.read_bytes(),before)
            self.assertEqual(digest(parent/'connection_results.parquet'),original)
            with ExitStack() as stack:
                network=self.patches(stack,parent,report,contract,events)
                resolve=stack.enter_context(patch('data.pipelines.band_source_resolver.resolve_offering_band',
                    side_effect=lambda raw,m,*args,**kwargs:resolved[m['rcept_no']]))
                result=refresh_cached_band(raw,p)
                self.assertEqual(resolve.call_count,1);network.assert_not_called()
            self.assertEqual(result['band_field_reprocessing']['reparsed_events'],2)

    def test_missing_file_is_not_a_retry_input_but_corrupt_or_foreign_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw,p,parent,report,contract,rows,events,resolved=self.fixture(Path(temporary))
            row=json.loads(rows.loc[1,'record']);url=json.loads(row['price_band_resolution_evidence'])['historical_band']['failed_url']
            path=raw/'dart_connection_sources'/(hashlib.sha256(url.encode()).hexdigest()+'.json')
            original=path.read_text();path.unlink()
            self.assertEqual(band_retry_dependencies(raw,rows),{})
            payload=json.loads(original);payload['text']='changed without matching hash';atomic_json(path,payload)
            with self.assertRaisesRegex(RuntimeError,'cache_integrity'):band_retry_dependencies(raw,rows)
            row['price_band_resolution_evidence']=json.dumps({'historical_band':{
                'reason':'official_source_not_cached','failed_url':'https://example.invalid/private'}})
            rows.loc[1,'record']=json.dumps(row)
            with self.assertRaisesRegex(RuntimeError,'outside_verified_family'):band_retry_dependencies(raw,rows)
