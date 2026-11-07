import json
import unittest
from copy import deepcopy
from unittest.mock import patch

from data.collectors.dart_collector import DARTCollector
from data.pipelines.band_source_resolver import currency_band, select_notice_band
from data.pipelines.stage_snapshots import result_day_document, values_from_document


class BandSourceTests(unittest.TestCase):
    def test_band_cache_contract_includes_document_and_amendment_selection(self):
        import ast
        from pathlib import Path
        from data.pipelines.historical_reconciliation import connection_contract
        before=connection_contract()
        read=Path.read_text
        for filename,name in [('stage_snapshots.py','amendment_view'),
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
