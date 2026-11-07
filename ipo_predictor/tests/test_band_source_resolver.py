import json
import unittest
from copy import deepcopy
from unittest.mock import patch

from data.collectors.dart_collector import DARTCollector
from data.pipelines.band_source_resolver import currency_band, select_notice_band, verified_after_view, historical_indicative_band
from data.pipelines.stage_snapshots import result_day_document, values_from_document


class BandSourceTests(unittest.TestCase):
    def test_added_note_band_requires_reproducible_empty_before_proof(self):
        from data.pipelines.stage_snapshots import added_note_after_view
        body=('<table><tr><td>항목</td><td>정정 전</td><td>정정 후</td></tr>'
            '<tr><td>희망공모가</td><td>-</td><td>주1)</td></tr></table>'
            '<p>주1)</p><p>■ 정정 후</p><p>희망공모가액 80원 ~ 120원</p>')
        _,view=added_note_after_view(body)
        view['strict_section_rejection']='correction_after_without_matching_before'
        self.assertTrue(verified_after_view(view))
        notice=self.notice();notice['view']=view
        self.assertEqual(currency_band(select_notice_band(self.document(),[notice],100)[0]),(80,120))
        for key,value in [('pair_count',0),('original_sha256','not_a_hash'),('empty_before_notes',{}),
                          ('strict_section_rejection',None)]:
            changed=deepcopy(view);changed[key]=value
            self.assertFalse(verified_after_view(changed))
        changed=deepcopy(view);changed['empty_before_notes']['1']['cells'][1]='unknown'
        self.assertFalse(verified_after_view(changed))

    def history(self, mutate=None, fault=None):
        pre={'rcept_no':'20240201000001','rcept_dt':'2024-02-01','report_nm':'투자설명서'}
        post={'rcept_no':'20240210000001','rcept_dt':'2024-02-10','report_nm':'[정정]투자설명서'}
        schedule='<table><tr><td>수요예측 일시</td><td>2024년 02월 08일 ~ 02월 09일</td></tr></table>'
        records={pre['rcept_no']:[pre,schedule+'<p>희망공모가액 80원 ~ 120원</p>',
                    {'offering':self.document(80,120)}],
                 post['rcept_no']:[post,schedule+'<p>확정 공모가 100원</p>',{'offering':self.document()}]}
        # The indicative range is valid even though its document has no final price.
        records[pre['rcept_no']][2]['offering']['offering_price']=None
        records[pre['rcept_no']][2]['offering']['offering_price_finality']='preliminary_price_language'
        if mutate:mutate(records,pre,post,schedule)
        def load(m):
            if fault and m==pre:raise fault
            return records[m['rcept_no']][1],{'receipt':m['rcept_no'],'corp_code':'00000001',
                'content_scope':'partial_terms_notice' if '발행조건확정' in m['report_nm'] else 'full',
                'related_receipts':[post['rcept_no']],'source_url':'official/'+m['rcept_no'],
                'published_at':m['rcept_dt'],'available_at':m['rcept_dt']+'T00:00:00+09:00',
                'index_sha256':'a'*64,'sections':[{'url':'section/'+m['rcept_no'],'sha256':'b'*64}]}
        analyse=lambda m,b,c:records[m['rcept_no']][2]
        return historical_indicative_band((post,*load(post)),[r[0] for r in records.values()],load,analyse,'2024-02-11T00:01:00+09:00')

    def test_indicative_range_and_final_price_have_independent_receipts(self):
        donor,proof=self.history()
        self.assertEqual(currency_band(donor['document']),(80,120))
        self.assertEqual(donor['metadata']['rcept_no'],'20240201000001')
        self.assertEqual(proof['status'],'verified_historical_indicative_band')
        self.assertEqual(len(proof['checks']),2)

    def test_historical_range_conflict_or_ambiguity_blocks_older_range(self):
        for value in [self.document(90,130),{'price_band_validation_status':'conflicting_bands_review_required'}]:
            def change(records,pre,post,schedule):records[post['rcept_no']][2]['offering']=value
            donor,proof=self.history(change)
            self.assertIsNone(donor)
            self.assertIn(proof['reason'],{'historical_band_later_range_conflict','historical_band_later_range_ambiguous'})

    def test_latest_pre_period_document_not_older_matching_range_is_required(self):
        def change(records,pre,post,schedule):
            newer={**pre,'rcept_no':'20240205000001','rcept_dt':'2024-02-05'}
            records[newer['rcept_no']]=[newer,schedule,{'offering':self.document()}]
        donor,proof=self.history(change)
        self.assertIsNone(donor);self.assertEqual(proof['reason'],'historical_band_latest_pre_period_range_unverified')

    def test_historical_range_requires_same_round_and_no_withdrawal(self):
        def changed_schedule(records,pre,post,schedule):records[pre['rcept_no']][1]=schedule.replace('08일','07일')
        def withdrawal(records,pre,post,schedule):
            m={'rcept_no':'20240205000001','rcept_dt':'2024-02-05','report_nm':'철회신고서'}
            records[m['rcept_no']]=[m,'',{}]
        for mutate,reason in [(changed_schedule,'historical_band_demand_round_conflict'),
                              (withdrawal,'historical_band_withdrawal_crosses_lineage')]:
            donor,proof=self.history(mutate)
            self.assertIsNone(donor);self.assertEqual(proof['reason'],reason)

    def test_unresolved_after_notice_does_not_become_absent_official_band(self):
        def change(records,pre,post,schedule):
            m={'rcept_no':'20240209000001','rcept_dt':'2024-02-09','report_nm':'[발행조건확정]증권신고서(지분증권)'}
            records[m['rcept_no']]=[m,'<p>unresolved amendment</p>',{'offering':self.document()}]
        donor,proof=self.history(change)
        self.assertIsNone(donor);self.assertEqual(proof['status'],'historical_band_review_required')

    def test_missing_or_inaccessible_history_remains_review_not_nonpublication(self):
        from requests import ConnectionError
        from data.pipelines.stage_snapshots import StageSourceMissing
        for error,reason in [(StageSourceMissing('https://dart.fss.or.kr/source'),'official_source_not_cached'),
                             (ConnectionError('transport failed'),'source_access_failed')]:
            donor,proof=self.history(fault=error)
            self.assertIsNone(donor);self.assertEqual(proof['reason'],reason)
            self.assertEqual(proof['status'],'historical_band_review_required')

    def test_band_cache_contract_includes_document_and_amendment_selection(self):
        import ast
        from pathlib import Path
        from data.pipelines.historical_reconciliation import connection_contract
        before=connection_contract()
        read=Path.read_text
        for filename,name in [('stage_snapshots.py','amendment_view'),
                              ('stage_snapshots.py','added_note_after_view'),
                              ('lockup_reconciliation.py','documents'),
                              ('lockup_reconciliation.py','original_body')]:
            def changed_source(path,*args,**kwargs):
                text=read(path,*args,**kwargs)
                if path.name==filename:
                    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
                    lines=text.splitlines(keepends=True);lines.insert(node.lineno,'    pass\n');text=''.join(lines)
                return text
            with self.subTest(dependency=name),patch.object(Path,'read_text',new=changed_source):
                after=connection_contract()
                self.assertNotEqual(before['band'],after['band'])
                self.assertEqual({k:v for k,v in before.items() if k!='band'},
                                 {k:v for k,v in after.items() if k!='band'})

    def test_notice_parser_ambiguity_is_not_a_verified_official_band_conflict(self):
        full=self.document(80,120);notice=self.notice()
        notice['document']['price_band_validation_status']='conflicting_bands_review_required'
        selected,donor,checks=select_notice_band(full,[notice],100,complete_receipt='20240201000002')
        self.assertIs(selected,full);self.assertIsNone(donor)
        self.assertEqual(checks[0]['status'],'later_complete_band_retained_notice_parser_ambiguous')
        for receipt in [None,'20240201000000']:
            selected,_,_=select_notice_band(full,[notice],100,complete_receipt=receipt)
            self.assertIsNone(currency_band(selected))

    def document(self, low=None, high=None):
        doc={'offering_price':100,'offering_price_finality':'confirmed_price_language',
             'offering_price_review_status':'verified_currency_unit'}
        if low is not None:
            doc.update(price_band_low=low,price_band_high=high,price_band_validation_status='direct_currency_band',
                price_band_parse_method='direct_currency_range_v1',
                price_band_parse_evidence=json.dumps([f'희망공모가액 {low}원 ~ {high}원']))
        return doc

    def notice(self, receipt='20240201000001', low=80, high=120):
        return {'receipt':receipt,'document':self.document(low,high),'origin':{},'linked':True,
                'view':{'rule':'explicit_marker_before_after_sections_v1'}}

    def test_missing_full_band_uses_one_verified_after_notice_pair(self):
        full=self.document();notice=self.notice()
        selected,donor,checks=select_notice_band(full,[notice],100)
        self.assertEqual(currency_band(selected),(80,120))
        self.assertIs(donor,notice)
        self.assertNotIn('price_band_low',full)
        self.assertEqual(selected['offering_price'],100)
        self.assertEqual(checks[0]['status'],'verified_after_notice_fills_absent_complete_band')

    def test_same_pair_keeps_full_origin_but_real_conflicts_are_held(self):
        full=self.document(80,120)
        selected,donor,_=select_notice_band(full,[self.notice()],100)
        self.assertIs(selected,full);self.assertIsNone(donor)
        for changed in [self.document(85,120),{**full,'price_band_validation_status':'conflicting_bands_review_required'}]:
            selected,donor,_=select_notice_band(changed,[self.notice()],100)
            self.assertIsNone(selected['price_band_low']);self.assertIsNone(selected['price_band_high'])
            self.assertEqual(selected['price_band_validation_status'],'conflicting_bands_review_required')

    def test_latest_verified_component_wins_without_mixing_two_pairs(self):
        old=self.notice(low=70,high=110)
        latest=self.notice('20240201000002',90,130)
        selected,donor,_=select_notice_band(self.document(),[old,latest],100)
        self.assertEqual(currency_band(selected),(90,130));self.assertIs(donor,latest)
        latest['document']['price_band_validation_status']='conflicting_bands_review_required'
        self.assertIsNone(currency_band(select_notice_band(self.document(),[old,latest],100)[0]))

    def test_no_donation_without_link_after_view_and_matching_confirmed_prices(self):
        for change in ['unlinked','no_view','horizontal_only','notice_price','complete_price','planned','currency']:
            with self.subTest(change=change):
                full=self.document();notice=self.notice()
                if change=='unlinked':notice['linked']=False
                if change=='no_view':notice['view']=None
                if change=='horizontal_only':notice['view']['rule']='horizontal_after_cells_consistency_only_v1'
                if change=='notice_price':notice['document']['offering_price']=101
                if change=='complete_price':full['offering_price']=101
                if change=='planned':notice['document']['offering_price_finality']='planned'
                if change=='currency':notice['document']['offering_price_review_status']='unverified'
                selected,donor,_=select_notice_band(full,[notice],100)
                self.assertIsNone(currency_band(selected));self.assertIsNone(donor)

    def test_band_requires_finite_positive_paired_currency_evidence(self):
        for field,value in [('price_band_low',0),('price_band_low',float('inf')),
                            ('price_band_high',79),('price_band_parse_method','guessed'),
                            ('price_band_parse_evidence','["80~120"]'),
                            ('price_band_parse_evidence','["희망공모가액 81원 ~ 120원"]')]:
            with self.subTest(field=field,value=value):
                changed=self.document(80,120);changed[field]=value
                self.assertIsNone(currency_band(changed))

    def test_stage_derived_band_retains_both_component_origins(self):
        early={'rcept_no':'20240201000001','rcept_dt':'2024-02-01','report_nm':'증권신고서(지분증권)'}
        full={'rcept_no':'20240201000002','rcept_dt':'2024-02-01','report_nm':'투자설명서'}
        bodies={early['rcept_no']:'<p>[정정 전]</p><p>공모가액(예정) 90원</p>'
            '<p>[정정 후]</p><p>확정 공모가 100원</p><p>희망공모가액 80원 ~ 120원</p>',
            full['rcept_no']:'<p>확정 공모가 100원</p>'}
        def load(meta):
            receipt=meta['rcept_no']
            return bodies[receipt],{'receipt':receipt,'corp_code':'00000001',
                'source_url':'https://dart.fss.or.kr/dsaf001/main.do?rcpNo='+receipt,
                'index_sha256':'a'*64,'published_at':'2024-02-01','available_at':'2024-02-02T00:00:00+09:00',
                'sections':[{'url':'official/'+receipt,'sha256':'b'*64}],
                'related_receipts':[early['rcept_no'],full['rcept_no']],
                'content_scope':'full' if receipt==full['rcept_no'] else 'partial_or_terms_notice'}
        meta,body,proof,analysis=result_day_document(early,[early,full],load)
        self.assertEqual(meta,full)
        self.assertEqual(currency_band(analysis['offering']),(80,120))
        sources=proof['result_field_sources']
        self.assertEqual(sources['price_band_low']['receipt'],early['rcept_no'])
        components=sources['offering_price_band_position']['component_sources']
        self.assertEqual(components['band']['receipt'],early['rcept_no'])
        self.assertEqual(components['price']['receipt'],full['rcept_no'])
        selected={'html':body,'metadata':meta,'proof':proof,'analysis':analysis}
        post,_=values_from_document(DARTCollector(api_key='unused'),deepcopy(selected),'00000001','post_demand')
        pre,_=values_from_document(DARTCollector(api_key='unused'),deepcopy(selected),'00000001','pre_demand')
        self.assertEqual(post['offering_price_band_position'],.5)
        self.assertNotIn('offering_price_band_position',pre)
        self.assertNotIn('offering_price',pre)
        bodies[full['rcept_no']]+='<p>희망공모가액 90원 ~ 110원</p>'
        _,_,held,analysis=result_day_document(early,[early,full],load)
        self.assertIsNone(currency_band(analysis['offering']))
        self.assertEqual(analysis['offering']['offering_price'],100)
        self.assertIn('price_band',held['held_result_fields'])
        self.assertNotIn('offering_price_band_position',held['result_field_sources'])
