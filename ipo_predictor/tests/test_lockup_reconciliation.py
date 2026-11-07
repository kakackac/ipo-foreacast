import json
import hashlib
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import pandas as pd

from data.collectors.dart_collector import DARTCollector
from data.pipelines.lockup_reconciliation import unchanged_other_parser_logic, reprocess_record, reprocess_institutional_record, run, documents, parser_snapshot, METHODS, institutional_extraction_reusable
from data.pipelines.data_delivery import digest


class LockupRepairTests(unittest.TestCase):
    def flat_outcome(self, quantity='1,000', total='100', percent='10.0'):
        return ('<p>수요예측 결과 기관투자자 수요예측 참여내역</p><table>'
            '<tr><td>참여건수(건)</td><td>신청수량(주)</td><td>단순경쟁률</td></tr>'
            f'<tr><td>100</td><td>{quantity}</td><td>10:1</td></tr></table>'
            '<h3>의무보유확약 기관수 및 신청수량</h3><table>'
            '<tr><td>구분</td><td>기관수(개)</td><td>신청수량(주)</td><td>비율(%)</td></tr>'
            '<tr><td>1개월 확약</td><td>1</td><td>100</td><td>10.0</td></tr>'
            f'<tr><td>합계</td><td>1</td><td>{total}</td><td>{percent}</td></tr></table>')

    def test_flat_committed_total_prefers_explicit_percentage_without_noncommitted_rows(self):
        parsed=DARTCollector._extract_lockup_ratios_from_tables(self.flat_outcome())
        self.assertEqual(parsed['lockup_commitment_ratio'],.1)
        proof=json.loads(parsed['structured_evidence'])
        self.assertEqual(proof['selected_column'],2)
        self.assertEqual(proof['percentage_column'],3)
        self.assertTrue(proof['printed_aggregate_preferred'])
        # Missing/incomplete duration detail does not invalidate an explicit aggregate.
        partial=self.flat_outcome().replace('<td>1개월 확약</td><td>1</td><td>100</td><td>10.0</td>',
                                           '<td>1개월 확약</td><td>1</td><td>50</td><td>5.0</td>')
        self.assertEqual(DARTCollector._extract_lockup_ratios_from_tables(partial)['lockup_commitment_ratio'],.1)

    def test_flat_committed_quantity_requires_independent_scope_and_preserves_percentage_conflicts(self):
        html=self.flat_outcome()
        for original,replacement in [('신청수량(주)','배정수량(주)'),
                ('의무보유확약 기관수 및 신청수량','국내기관 의무보유확약 기관수 및 신청수량'),
                ('1,000','1,2,000')]:
            with self.subTest(replacement=replacement):
                parsed=DARTCollector._extract_lockup_ratios_from_tables(html.replace(original,replacement))
                self.assertIsNone(parsed['lockup_commitment_ratio'])
        parsed=DARTCollector._extract_lockup_ratios_from_tables(self.flat_outcome(percent='11.0'))
        self.assertIsNone(parsed['lockup_commitment_ratio'])
        self.assertEqual(parsed['rejection_reason'],'lockup_disclosed_percent_quantity_conflict')
        malformed=DARTCollector._extract_lockup_ratios_from_tables(self.flat_outcome(percent='10,0'))
        self.assertIsNone(malformed['lockup_commitment_ratio'])
        self.assertEqual(malformed['rejection_reason'],'lockup_flat_percentage_cell_unverified')

    def test_flat_total_with_percentage_row_uses_quantity_not_count(self):
        html=self.flat_outcome().replace('<td>비율(%)</td>','<td>비고</td>')
        html=html.replace('<td>100</td><td>10.0</td></tr></table>',
            '<td>100</td><td>-</td></tr><tr><td>총 수량 대비 비율(%)</td><td>1.0</td><td>10.0</td><td>-</td></tr></table>')
        parsed=DARTCollector._extract_lockup_ratios_from_tables(html)
        self.assertEqual(parsed['lockup_commitment_ratio'],.1)
        self.assertEqual(json.loads(parsed['structured_evidence'])['percentage_column'],2)

    def test_original_header_variants_require_independent_quantity_and_keep_raw_cells(self):
        for header in ['신청수랑(주)','신청수량㈜']:
            html=self.flat_outcome().replace('<td>기관수(개)</td><td>신청수량(주)</td>',
                                            '<td>기관수(개)</td><td>'+header+'</td>')
            parsed=DARTCollector._extract_lockup_ratios_from_tables(html)
            self.assertEqual(parsed['lockup_commitment_ratio'],.1)
            proof=json.loads(parsed['structured_evidence'])
            self.assertEqual(proof['table'][0][2],header)
            self.assertEqual(proof['header_unit_normalizations'][0]['original'],header)
            self.assertTrue(proof['independent_application_sources'])
            for bad in [html[html.index('<h3>'):],html.replace('1,000','1,100'),html.replace(header,'신청수랑(원)')]:
                self.assertIsNone(DARTCollector._extract_lockup_ratios_from_tables(bad)['lockup_commitment_ratio'])

    def test_committed_subtotal_before_noncommitted_is_not_the_whole_total(self):
        summary=self.flat_outcome().split('<h3>',1)[0]
        table=[['구분','기관수(개)','신청수량(주)'],['1개월 확약','1','100'],
               ['합계','1','100'],['총 건수 대비율','1.0%','10.0%'],['미확약','99','900']]
        def html(rows):return summary+'<h3>의무보유 확약 기관수 및 신청수량</h3><table>'+''.join(
            '<tr>'+''.join('<td>'+cell+'</td>' for cell in row)+'</tr>' for row in rows)+'</table>'
        parsed=DARTCollector._extract_lockup_ratios_from_tables(html(table))
        self.assertEqual(parsed['lockup_commitment_ratio'],.1)
        proof=json.loads(parsed['structured_evidence'])
        self.assertEqual(proof['subtotal_noncommitted_reconciliation']['noncommitted'],900)
        self.assertTrue(proof['aggregate_quantity_check']['subtotal_before_noncommitted'])
        for changed in [table[:1]+table[2:],table[:-1]+[['미확약','99','901']],
                        table[:2]+[table[-1]]+table[2:-1]]:
            self.assertIsNone(DARTCollector._extract_lockup_ratios_from_tables(html(changed))['lockup_commitment_ratio'])

    def test_independent_demand_dash_components_require_paired_counts_and_exact_partition(self):
        summary=('<p>수요예측 결과 기관투자자 수요예측 참여내역</p><table>'
            '<tr><td>구분</td><td>국내기관투자자</td><td>해외기관투자자</td><td>합계</td></tr>'
            '<tr><td>건수</td><td>100</td><td>-</td><td>100</td></tr>'
            '<tr><td>수량</td><td>1,000</td><td>-</td><td>1,000</td></tr></table>')
        outcome='<h3>'+self.flat_outcome().split('<h3>',1)[1]
        outcome=outcome.replace('신청수량(주)','신청수랑(주)')
        parsed=DARTCollector._extract_lockup_ratios_from_tables(summary+outcome)
        self.assertEqual(parsed['lockup_commitment_ratio'],.1)
        proof=json.loads(parsed['structured_evidence'])
        self.assertEqual(proof['independent_application_sources'][0]['paired_dash_components_reconciled_as_zero'],[2])
        for changed in [summary.replace('<td>100</td><td>-</td><td>100</td>',
                                          '<td>100</td><td>1</td><td>101</td>'),
                        summary.replace('<td>1,000</td><td>-</td><td>1,000</td>',
                                        '<td>1,000</td><td>-</td><td>1,100</td>')]:
            self.assertIsNone(DARTCollector._extract_lockup_ratios_from_tables(changed+outcome)['lockup_commitment_ratio'])

    def test_allocation_quantity_is_a_different_basis_not_missing_official_data(self):
        html=self.flat_outcome().replace('<td>신청수량(주)</td><td>비율(%)</td>',
                                         '<td>배정수량(주)</td><td>비율(%)</td>')
        parsed=DARTCollector._extract_lockup_ratios_from_tables(html)
        self.assertIsNone(parsed['lockup_commitment_ratio'])
        self.assertEqual(parsed['rejection_reason'],'lockup_allocation_quantity_scope_not_application_demand')

    def test_identical_flat_table_without_its_separate_denominator_keeps_composition_contract(self):
        from data.pipelines.stage_snapshots import same_verified_lockup_table
        complete=self.flat_outcome()
        without_summary='<h3>'+complete.split('<h3>',1)[1]
        dart=DARTCollector(api_key='unused')
        good=dart._parse_demand_forecast_html(complete,'test')
        missing=dart._parse_demand_forecast_html(without_summary,'test')
        self.assertIsNone(missing['lockup_commitment_ratio'])
        self.assertTrue(same_verified_lockup_table(good,missing))
        self.assertEqual(json.loads(missing['lockup_structured_evidence'])['failure_cause'],
                         'independent_application_quantity_unavailable')

    def test_institutional_reuse_includes_selection_and_direct_label_logic(self):
        before='DEMAND_PARSER_VERSION=23\nclass DARTCollector:\n def _extract_lockup_ratios_from_tables(self):return 1\n def _extract_aggregate_demand_ratio(self):return 10\n def _parse_demand_forecast_html(self):return 20\n'
        after=before.replace('=23','=24').replace('return 1\n','return 2\n')
        old={'repair':'a','amendment_view':'b','outcome_table_equivalence':'c','demand_rounding':'d','parser':'old'}
        current={**old,'parser':'new'}
        self.assertTrue(institutional_extraction_reusable(before,after,old,current))
        for key in ['repair','amendment_view','outcome_table_equivalence','demand_rounding']:
            with self.subTest(key=key):
                self.assertFalse(institutional_extraction_reusable(before,after,old,{**current,key:'changed'}))
        self.assertFalse(institutional_extraction_reusable(before,after.replace('return 20','return 21'),old,current))
        self.assertFalse(institutional_extraction_reusable(before,after.replace('return 10','return 11'),old,current))
        self.assertFalse(institutional_extraction_reusable(None,after,old,current))
    def test_aggregate_cells_with_both_count_and_shares_use_explicit_share_unit(self):
        table=[['구분','국내기관투자자','해외기관투자자','합계'],
            ['6개월 확약','1건 / 100주','-','1건 / 100주'],
            ['미확약','9건 / 900주','0건 / 0주','9건 / 900주'],
            ['합계','10건 / 1,000주','0건 / 0주','10건 / 1,000주']]
        result=DARTCollector._extract_aggregate_lockup_shares(table,1000)
        self.assertIsNotNone(result)
        self.assertEqual(result['lockup_commitment_ratio'],.1)
        proof=json.loads(result['structured_evidence'])
        self.assertTrue(proof['explicit_count_and_share_cell_units'])
        self.assertEqual(proof['table'][1][-1],'1건 / 100주')
        for text in ['1건 / 100원','1주 / 100건','100','1건 / 100백만주']:
            changed=json.loads(json.dumps(table));changed[1][-1]=text
            self.assertIsNone(DARTCollector._extract_aggregate_lockup_shares(changed,1000))
        self.assertIsNone(DARTCollector._extract_aggregate_lockup_shares(table,1001))

    def test_explicit_composite_share_cells_allow_literal_zero_not_a_bare_positive_number(self):
        table=[['구분','국내기관투자자','해외기관투자자','합계'],
            ['6개월 확약','0','0','0'],['3개월 확약','1건 / 100주','0','1건 / 100주'],
            ['미확약','9건 / 900주','0건 / 0주','9건 / 900주'],
            ['합계','10건 / 1,000주','0건 / 0주','10건 / 1,000주']]
        parsed=DARTCollector._extract_aggregate_lockup_shares(table,1000)
        self.assertEqual(parsed['lockup_commitment_ratio'],.1)
        self.assertEqual(json.loads(parsed['structured_evidence'])['table'][1][-1],'0')
        for value in ['100','1건 / 1,0주','1주 / 100건','1건 / 100원']:
            changed=json.loads(json.dumps(table));changed[2][-1]=value
            self.assertIsNone(DARTCollector._extract_aggregate_lockup_shares(changed,1000))
        changed=json.loads(json.dumps(table));changed[2][-1]='1.0건 / 100주'
        parsed=DARTCollector._extract_aggregate_lockup_shares(changed,1000)
        self.assertEqual(parsed['lockup_commitment_ratio'],.1)
        self.assertEqual(json.loads(parsed['structured_evidence'])['unparsed_participant_count_cells'][0][
            'raw_count_token'],'1.0')

    def test_empty_outcome_cells_are_not_an_unresolved_numeric_conflict_or_zero(self):
        html='<h3>기관 수요예측 의무보유확약 신청내역</h3><table><tr><td>확약기간</td><td>신청수량(주)</td></tr>'
        html+=''.join(f'<tr><td>{label}</td><td></td></tr>' for label in ['6개월','3개월','미확약','합계'])+'</table>'
        parsed=DARTCollector._extract_lockup_ratios_from_tables(html)
        self.assertIsNone(parsed['lockup_commitment_ratio'])
        self.assertEqual(parsed['parser_validation_status'],'value_not_found')
        self.assertEqual(parsed['rejection_reason'],'selected_document_lockup_outcome_cells_empty')
        self.assertNotEqual(DARTCollector._extract_lockup_ratios_from_tables(html.replace('<td></td>','<td>-</td>'))[
            'rejection_reason'],'selected_document_lockup_outcome_cells_empty')

    def test_dash_only_result_form_is_unreported_not_zero_or_numeric_conflict(self):
        html='<h3>기관 수요예측 의무보유확약 신청내역</h3><table><tr><td>구분</td><td>기관수(개)</td><td>신청수량(주)</td></tr>'
        html+=''.join(f'<tr><td>{label}</td><td>-</td><td>-</td></tr>' for label in ['3개월 확약','1개월 확약','합계'])+'</table>'
        result=DARTCollector._extract_lockup_ratios_from_tables(html)
        self.assertIsNone(result['lockup_commitment_ratio'])
        self.assertEqual(result['parser_validation_status'],'value_not_found')
        self.assertEqual(result['rejection_reason'],'selected_document_lockup_outcome_dash_only')
        mixed=html.replace('<td>3개월 확약</td><td>-</td>','<td>3개월 확약</td><td>3</td>')
        self.assertEqual(DARTCollector._extract_lockup_ratios_from_tables(mixed)['parser_validation_status'],'candidate_rejected')

    def test_multiheader_dash_only_outcomes_are_separated_from_headers(self):
        table=[['구분','국내기관투자자','해외기관투자자','합계'],
            ['구분','수량','수량','수량'],['6개월 확약','-','-','-'],
            ['1개월 확약','-','-','-'],['미확약','-','-','-'],['총 합계','-','-','-']]
        html='<h3>기관 수요예측 의무보유확약 신청내역</h3><table>'+''.join(
            '<tr>'+''.join('<td>'+s+'</td>' for s in row)+'</tr>' for row in table)+'</table>'
        parsed=DARTCollector._extract_lockup_ratios_from_tables(html)
        self.assertIsNone(parsed['lockup_commitment_ratio'])
        self.assertEqual(parsed['parser_validation_status'],'value_not_found')
        self.assertEqual(parsed['rejection_reason'],'selected_document_lockup_outcome_dash_only')
        self.assertEqual(json.loads(parsed['structured_evidence'])['candidate_tables'][0]['table'],table)

    def test_adjacent_domestic_foreign_tables_require_row_and_column_quantity_count_reconciliation(self):
        summary=self.flat_outcome().split('<h3>',1)[0].replace('<td>100</td><td>1,000</td>',
                                                             '<td>10</td><td>1,000</td>')
        domestic=[['구분','국내기관투자자','국내기관투자자','국내기관투자자'],
            ['구분','건수','수량','신청가격'],['1개월 확약','1','100','10000'],
            ['미확약','8','800','10000'],['합계','9','900','10000']]
        foreign=[['구분','외국기관투자자','외국기관투자자','외국기관투자자','합계','합계','합계'],
            ['구분','거래실적 유','거래실적 유','거래실적 유','합계','합계','합계'],
            ['1개월 확약','-','-','-','1','100','10000'],
            ['미확약','1','100','10000','9','900','10000'],
            ['합계','1','100','10000','10','1000','10000']]
        def html(left,right):return summary+'<h3>기관 수요예측 의무보유확약 신청내역</h3>'+''.join(
            '<table>'+''.join('<tr>'+''.join('<td>'+s+'</td>' for s in row)+'</tr>' for row in table)+'</table>'
            for table in (left,right))
        parsed=DARTCollector._extract_lockup_ratios_from_tables(html(domestic,foreign))
        self.assertEqual(parsed['lockup_commitment_ratio'],.1)
        proof=json.loads(parsed['structured_evidence'])
        self.assertEqual(proof['table'],foreign)
        self.assertEqual(proof['continuation_schema_source']['table'],domestic)
        captioned=html(domestic,foreign).replace('</table><table>',
            '</table><table><tr><td>(단위: 건, 주)</td></tr></table><table>')
        parsed=DARTCollector._extract_lockup_ratios_from_tables(captioned)
        self.assertEqual(parsed['lockup_commitment_ratio'],.1)
        self.assertEqual(json.loads(parsed['structured_evidence'])['continuation_schema_source'][
            'intervening_unit_captions'][0]['table'],[['(단위: 건, 주)']])
        self.assertIsNone(DARTCollector._extract_lockup_ratios_from_tables(captioned.replace(
            '(단위: 건, 주)','다른 수요예측 결과'))['lockup_commitment_ratio'])
        for row,column,value in [(2,4,'2'),(3,5,'901'),(4,5,'1001')]:
            changed=json.loads(json.dumps(foreign));changed[row][column]=value
            self.assertIsNone(DARTCollector._extract_lockup_ratios_from_tables(html(domestic,changed))['lockup_commitment_ratio'])
        changed=json.loads(json.dumps(domestic));changed[1][2]='신청가격'
        self.assertIsNone(DARTCollector._extract_lockup_ratios_from_tables(html(changed,foreign))['lockup_commitment_ratio'])
        self.assertIsNone(DARTCollector._extract_lockup_ratios_from_tables(html(domestic,foreign[1:]))['lockup_commitment_ratio'])

    def full_scope_lockup_table(self, total_label='총합계'):
        return [['구분','국내기관투자자','국내기관투자자','해외기관투자자','해외기관투자자','합계','합계'],
                ['구분','건수','수량','건수','수량','건수','수량'],
                ['1개월 확약','1','90','1','10','2','100'],
                ['미확약','8','800','1','100','9','900'],
                [total_label,'9','890','2','110','11','1000']]

    def test_all_institution_total_alias_is_verified_by_its_own_complete_quantity_partition(self):
        for label in ['계','총합계','총계','총합']:
            table=self.full_scope_lockup_table(label)
            result=DARTCollector._extract_aggregate_lockup_shares(table)
            self.assertAlmostEqual(result['lockup_commitment_ratio'],.1)
            self.assertTrue(json.loads(result['structured_evidence'])['all_institution_domestic_foreign_headers'])
            table[-1][-1]='1001'
            self.assertIsNone(DARTCollector._extract_aggregate_lockup_shares(table))
        table=self.full_scope_lockup_table();table[0]=[cell.replace('해외기관투자자','국내기관투자자') for cell in table[0]]
        self.assertIsNone(DARTCollector._extract_aggregate_lockup_shares(table))
        table=self.full_scope_lockup_table();table[0]=['구분','합계','합계','국내기관투자자','국내기관투자자','해외기관투자자','해외기관투자자']
        self.assertIsNone(DARTCollector._extract_aggregate_lockup_shares(table))

    def test_explicit_commitment_share_subtotal_and_percentage_are_not_counts(self):
        table=self.full_scope_lockup_table('합계')
        table.extend([['확약 소계','1','90','1','10','2','100'],
                      ['확약 비중','11.11%','10.11%','50%','9.09%','18.18%','10.0%']])
        result=DARTCollector._extract_disclosed_aggregate_lockup_ratio(table)
        self.assertEqual(result['lockup_commitment_ratio'],.1)
        self.assertEqual(json.loads(result['structured_evidence'])['selected_column'],6)
        html='<h3>기관 수요예측 의무보유확약 기관수 및 신청수량</h3><table>'+''.join(
            '<tr>'+''.join('<td>'+s+'</td>' for s in r)+'</tr>' for r in table)+'</table>'
        self.assertEqual(DARTCollector._extract_lockup_ratios_from_tables(html)['lockup_commitment_ratio'],.1)
        broken=json.loads(json.dumps(table));broken[-2][-1]='101'
        self.assertIsNone(DARTCollector._extract_disclosed_aggregate_lockup_ratio(broken))
        broken=json.loads(json.dumps(table));broken[-1][-1]='11.0%'
        self.assertIsNone(DARTCollector._extract_disclosed_aggregate_lockup_ratio(broken))
        table[0]=[cell.replace('해외기관투자자','국내기관투자자') for cell in table[0]]
        self.assertIsNone(DARTCollector._extract_disclosed_aggregate_lockup_ratio(table))

    def test_subtotal_and_grand_total_coexist_without_changing_the_whole_basis(self):
        summary=self.flat_outcome().split('<h3>',1)[0]
        table=self.full_scope_lockup_table('총합계')
        table.insert(3,['합계','1','90','1','10','2','100'])
        table.insert(4,['총참여건수및신청수량대비비율','11.11%','10.11%','50%','9.09%','18.18%','10.0%'])
        html=summary+'<h3>기관 수요예측 의무보유확약 신청내역</h3><table>'+''.join(
            '<tr>'+''.join('<td>'+s+'</td>' for s in row)+'</tr>' for row in table)+'</table>'
        parsed=DARTCollector._extract_lockup_ratios_from_tables(html)
        self.assertEqual(parsed['lockup_commitment_ratio'],.1)
        self.assertEqual(json.loads(parsed['structured_evidence'])['aggregate_quantity_check']['status'],'compatible')

    def test_same_original_section_but_changed_after_view_does_not_reuse_old_lockup(self):
        from data.collectors.dart_collector import DEMAND_PARSER_VERSION
        metadata={'rcept_no':'20240201000001','rcept_dt':'2024-02-01'}
        section=[{'url':'official','sha256':'a'*64}]
        previous={'view':{'selected_sha256':'old'}};current={'view':{'selected_sha256':'new'}}
        proof={'lockup_reprocessing':{'parser_version':DEMAND_PARSER_VERSION,'attempts':[
            {'receipt':metadata['rcept_no'],'published':metadata['rcept_dt'],'sections':section,
             'parser_status':'structurally_verified','field_notice_selection':previous}]}}
        record={'listing_date':'2024-03-01','lockup_parser_version':DEMAND_PARSER_VERSION,'lockup_commitment_ratio':.1}
        dart=DARTCollector(api_key='unused')
        with patch('data.pipelines.lockup_reconciliation.documents',return_value=[(metadata,
                {'sections':section,'source_url':'official','field_notice_selection':current},'new after view')]), \
                patch.object(dart,'_extract_lockup_ratios_from_tables',return_value={
                    'lockup_commitment_ratio':.2,'parser_validation_status':'structurally_verified','rule_id':'verified'}) as parse:
            updated,evidence=reprocess_record(Path('/tmp'),record,proof,dart,allow_reuse=True)
        parse.assert_called_once()
        self.assertEqual(updated['lockup_commitment_ratio'],.2)
        self.assertEqual(evidence['lockup_reprocessing']['attempts'][0]['field_notice_selection'],current)

    def test_same_day_confirmed_after_notice_supplies_unpopulated_full_document(self):
        primary={'rcept_no':'20240201000001','rcept_dt':'2024-02-01'}
        full={'rcept_no':'20240201000002','rcept_dt':'2024-02-01'}
        other={'source_url':'https://dart.fss.or.kr/dsaf001/main.do?rcpNo='+full['rcept_no'],
               'family':[full],'corp_code':'00000001','sections':[{'sha256':'b'*64}]}
        proof={'family':[primary],'content_scope':'partial_or_terms_notice','corp_code':'00000001',
            'supplementary_status':'official_related_prospectus_verified','supplementary':other,
            'sections':[{'sha256':'a'*64}]}
        record={'rcept_no':primary['rcept_no'],'corp_code':'00000001','offering_price':10000}
        table=('<h3>수요예측 결과 기관투자자 수요예측 참여내역</h3><table>'
            '<tr><td>참여건수(건)</td><td>신청수량(주)</td><td>단순경쟁률</td></tr>'
            '<tr><td>100</td><td>1000</td><td>30:1</td></tr></table>')
        after='<p>확정 공모가 10,000원</p>'+table
        notice='<p>[정정 전]</p>'+table.replace('30:1','10:1')+'<p>[정정 후]</p>'+after
        def bodies(raw,bundle):return notice if bundle is proof else 'unpopulated official prospectus'
        with patch('data.pipelines.lockup_reconciliation.original_body',side_effect=bodies):
            selected=documents(Path('/tmp'),record,proof,field='institutional_demand_ratio')
        self.assertEqual(len(selected),1)
        self.assertEqual(selected[0][0],primary)
        self.assertNotIn('10:1',selected[0][2])
        self.assertEqual(DARTCollector(api_key='unused')._parse_demand_forecast_html(
            selected[0][2],'00000001')['institutional_demand_ratio'],30)
        self.assertEqual(selected[0][1]['field_notice_selection']['price'],10000)
        for change in ['price','corp','later_full','full_conflict','no_after_boundary']:
            updated=json.loads(json.dumps(proof));changed=json.loads(json.dumps(record))
            full_body='unpopulated official prospectus';notice_body=notice
            if change=='price':changed['offering_price']=11000
            if change=='corp':updated['supplementary']['corp_code']='00000002'
            if change=='later_full':updated['supplementary']['family'][0]['rcept_dt']='2024-02-02'
            if change=='full_conflict':full_body=after.replace('30:1','20:1')
            if change=='no_after_boundary':notice_body=after
            with self.subTest(change=change),patch('data.pipelines.lockup_reconciliation.original_body',
                    side_effect=lambda raw,bundle:notice_body if bundle is updated else full_body):
                selected=documents(Path('/tmp'),changed,updated,field='institutional_demand_ratio')
                self.assertEqual(selected[0][0]['rcept_no'],full['rcept_no'])
                self.assertNotIn('field_notice_selection',selected[0][1])

    def test_actual_lockup_conflict_in_full_document_blocks_notice_fallback(self):
        primary={'rcept_no':'20240201000001','rcept_dt':'2024-02-01'}
        full={'rcept_no':'20240201000002','rcept_dt':'2024-02-01'}
        extra={'source_url':'https://dart.fss.or.kr/dsaf001/main.do?rcpNo='+full['rcept_no'],
            'family':[full],'corp_code':'00000001','sections':[]}
        proof={'family':[primary],'content_scope':'partial_or_terms_notice','corp_code':'00000001',
            'supplementary_status':'official_related_prospectus_verified','supplementary':extra,'sections':[]}
        parsed={'lockup_commitment_ratio':.1,'lockup_parser_validation_status':'structurally_verified'}
        rejected={'lockup_commitment_ratio':None,'lockup_parser_validation_status':'candidate_rejected',
                  'lockup_rejection_reason':'lockup_independent_application_quantity_mismatch'}
        with patch('data.pipelines.lockup_reconciliation.original_body',return_value='official'), \
                patch('data.pipelines.stage_snapshots.amendment_view',return_value=('after',{})), \
                patch.object(DARTCollector,'_parse_offering_html',return_value={'offering_price':10000,
                    'offering_price_review_status':'verified_currency_unit','offering_price_finality':'confirmed_price_language'}), \
                patch.object(DARTCollector,'_parse_demand_forecast_html',side_effect=[parsed,rejected]):
            selected=documents(Path('/tmp'),{'rcept_no':primary['rcept_no'],'corp_code':'00000001',
                'offering_price':10000},proof,field='lockup_commitment_ratio')
        self.assertEqual(selected[0][0],full)

    def test_official_notice_conflict_is_held_not_silently_reapproved_from_full_copy(self):
        from data.collectors.dart_collector import DEMAND_PARSER_VERSION
        metadata={'rcept_no':'20240201000002','rcept_dt':'2024-02-01'}
        conflict={'field':'institutional_demand_ratio','notice_value':1471.01,
            'full_document_conflicts':[{'value':1471.04}]}
        bundle={'source_url':'official','sections':[],'field_notice_conflict':conflict}
        record={'corp_code':'00000001','listing_date':'2024-03-01',
            'institutional_demand_ratio':1471.04,'lockup_commitment_ratio':.2}
        dart=DARTCollector(api_key='unused')
        with patch('data.pipelines.lockup_reconciliation.documents',return_value=[(metadata,bundle,'body')]), \
                patch.object(dart,'_parse_demand_forecast_html',return_value={
                    'institutional_demand_ratio':1471.04,'institutional_demand_parser_validation_status':'structurally_verified',
                    'institutional_demand_rule_id':'original_total_cell'}):
            updated,proof=reprocess_institutional_record(Path('/tmp'),record,{},dart)
        self.assertIsNone(updated['institutional_demand_ratio'])
        self.assertEqual(updated['institutional_demand_rejection_reason'],'same_day_official_institutional_conflict')
        self.assertEqual(updated['lockup_commitment_ratio'],.2)
        self.assertEqual(proof['institutional_reprocessing']['attempts'][0]['field_notice_conflict'],conflict)
        bundle['field_notice_conflict']={'field':'lockup_commitment_ratio','notice_value':.1}
        record.update(lockup_parser_version=DEMAND_PARSER_VERSION)
        old={'lockup_reprocessing':{'parser_version':DEMAND_PARSER_VERSION,'attempts':[
            {'receipt':metadata['rcept_no'],'published':metadata['rcept_dt'],'sections':[],
             'parser_status':'structurally_verified'}]}}
        with patch('data.pipelines.lockup_reconciliation.documents',return_value=[(metadata,bundle,'body')]), \
                patch.object(dart,'_extract_lockup_ratios_from_tables',return_value={
                    'lockup_commitment_ratio':.2,'parser_validation_status':'structurally_verified','rule_id':'original_total'}) as parse:
            updated,proof=reprocess_record(Path('/tmp'),record,old,dart,allow_reuse=True)
        parse.assert_called_once()
        self.assertIsNone(updated['lockup_commitment_ratio'])
        self.assertEqual(updated['lockup_rejection_reason'],'same_day_official_lockup_conflict')

    def aggregate_html(self, quantity, committed, printed, periods=''):
        return ('<p>수요예측 결과 기관투자자 수요예측 참여내역</p><table>'
                '<tr><td>참여건수(건)</td><td>신청수량(주)</td><td>단순경쟁률</td></tr>'
                f'<tr><td>100</td><td>{quantity}</td><td>10:1</td></tr></table>'
                '<p>(다) 의무보유 확약 신청내역</p><table><tr><td>구분</td><td>신청수량(주)</td></tr>'
                f'{periods}<tr><td>총합계</td><td>{committed}</td></tr>'
                f'<tr><td>총 신청수량 대비 비율(%)</td><td>{printed}%</td></tr></table>')

    def test_direct_total_does_not_require_duration_components(self):
        for periods in ['', '<tr><td>1개월</td><td>50</td></tr>']:
            result=DARTCollector._extract_lockup_ratios_from_tables(self.aggregate_html(1000,100,'10',periods))
            self.assertEqual(result['lockup_commitment_ratio'],.1)
            check=json.loads(result['structured_evidence'])['aggregate_quantity_check']
            self.assertEqual(check['status'],'compatible')
            if periods:
                self.assertEqual(check['warnings'],['period_breakdown_differs_from_explicit_aggregate'])

    def test_reported_percentage_can_be_truncated_not_only_rounded(self):
        result=DARTCollector._extract_lockup_ratios_from_tables(self.aggregate_html(100000000,10458930,'10.45'))
        self.assertEqual(result['lockup_commitment_ratio'],.1045)
        check=json.loads(result['structured_evidence'])['aggregate_quantity_check']
        self.assertEqual(check['compatible_display_modes'],['positive_truncation'])
        self.assertAlmostEqual(check['computed_fraction'],.1045893)
        conflict=DARTCollector._extract_lockup_ratios_from_tables(self.aggregate_html(100000000,10460000,'10.45'))
        self.assertIsNone(conflict['lockup_commitment_ratio'])
        self.assertEqual(conflict['rejection_reason'],'lockup_disclosed_percent_quantity_conflict')

    def test_week_duration_and_noncommitted_totals_are_audit_not_required_components(self):
        html=self.aggregate_html(1000,100,'10','<tr><td>1주일</td><td>100</td></tr>')
        result=DARTCollector._extract_lockup_ratios_from_tables(html)
        self.assertNotIn('warnings',json.loads(result['structured_evidence'])['aggregate_quantity_check'])
        html=html.replace('<td>총합계</td><td>100</td>',
                          '<td>미확약</td><td>900</td></tr><tr><td>총합계</td><td>1000</td>')
        result=DARTCollector._extract_lockup_ratios_from_tables(html)
        self.assertEqual(result['lockup_commitment_ratio'],.1)

    def test_field_scoped_parser_reuse_requires_archived_hash_and_unchanged_other_methods(self):
        before='DEMAND_PARSER_VERSION=18\nclass DARTCollector:\n def _extract_lockup_ratios_from_tables(self):return 1\n def _extract_aggregate_demand_ratio(self):return 10\n'
        after=before.replace('=18','=19').replace('return 1\n','return 2\n')
        methods=METHODS-{'_extract_aggregate_demand_ratio'}
        self.assertTrue(unchanged_other_parser_logic(before,after,methods))
        self.assertFalse(unchanged_other_parser_logic(before,after.replace('return 10','return 20'),methods))
        with tempfile.TemporaryDirectory() as root:
            raw=Path(root);sha=hashlib.sha256(before.encode()).hexdigest()
            self.assertIsNone(parser_snapshot(raw,sha))
            parser_snapshot(raw,sha,before)
            self.assertEqual(parser_snapshot(raw,sha),before)
            path=raw/'parser_source_snapshots'/(sha+'.json')
            path.write_text(json.dumps({'source_sha256':sha,'source':after}))
            with self.assertRaisesRegex(RuntimeError,'snapshot_integrity_failed'):
                parser_snapshot(raw,sha)

    def test_explicit_lockup_percentage_must_not_conflict_with_its_quantities(self):
        body='<p>(12) 수요예측 결과 통보 ① 기관투자자 수요예측 참여내역</p><table><tr><td>참여건수(건)</td><td>신청수량(주)</td><td>단순경쟁률</td></tr><tr><td>1507</td><td>1,838,477,000</td><td>1,431:1</td></tr></table>'
        body+='<p>(다) 의무보유 확약 신청내역</p><table><tr><td>구분</td><td>신청수량(주)</td></tr><tr><td>15일</td><td>120,656,000</td></tr><tr><td>1개월</td><td>131,210,000</td></tr><tr><td>3개월</td><td>75,965,000</td></tr><tr><td>총합계</td><td>327,831,000</td></tr><tr><td>총 신청수량 대비 비율(%)</td><td>17.88%</td></tr></table>'
        result=DARTCollector._extract_lockup_ratios_from_tables(body)
        self.assertIsNone(result['lockup_commitment_ratio'])
        self.assertEqual(result['rejection_reason'],'lockup_disclosed_percent_quantity_conflict')
        rounded=DARTCollector._extract_lockup_ratios_from_tables(body.replace('17.88%','17.83%'))
        self.assertEqual(rounded['lockup_commitment_ratio'],.1783)

    def test_institutional_field_repair_keeps_price_and_lockup_provenance(self):
        record={'corp_code':'00000001','listing_date':'2024-03-01','offering_price':10000,
                'institutional_demand_ratio':None,'lockup_commitment_ratio':.2,'institutional_data_contract_version':2}
        metadata={'rcept_no':'20240201000001','rcept_dt':'2024-02-01'}
        bundle={'source_url':'official','sections':[{'sha256':'a'*64}]}
        parsed={'institutional_demand_ratio':40.88,'institutional_demand_parser_validation_status':'structurally_verified',
                'institutional_demand_rule_id':'DART_DEMAND_INSTITUTIONAL_SUMMARY_V3'}
        dart=DARTCollector(api_key='unused')
        with patch('data.pipelines.lockup_reconciliation.documents',return_value=[(metadata,bundle,'html')]), \
                patch.object(dart,'_parse_demand_forecast_html',return_value=parsed):
            updated,proof=reprocess_institutional_record(Path('/tmp'),record,{},dart)
        self.assertEqual(updated['institutional_demand_ratio'],40.88)
        self.assertEqual(updated['lockup_commitment_ratio'],.2)
        self.assertEqual(updated['offering_price'],10000)
        self.assertEqual(updated['institutional_data_contract_version'],2)
        self.assertEqual(proof['field_sources']['institutional_demand_ratio']['receipt'],metadata['rcept_no'])
    def test_same_parser_and_same_admitted_original_do_not_parse_again(self):
        from data.collectors.dart_collector import DEMAND_PARSER_VERSION
        metadata={'rcept_no':'20240201000001','rcept_dt':'2024-02-01'}
        section=[{'url':'official','sha256':'a'*64}]
        proof={'lockup_reprocessing':{'parser_version':DEMAND_PARSER_VERSION,'attempts':[
            {'receipt':metadata['rcept_no'],'published':metadata['rcept_dt'],'sections':section,
             'parser_status':'structurally_verified'}]}}
        record={'listing_date':'2024-03-01','lockup_parser_version':DEMAND_PARSER_VERSION,'lockup_commitment_ratio':.1}
        dart=DARTCollector(api_key='unused')
        with patch('data.pipelines.lockup_reconciliation.documents',return_value=[(metadata,{'sections':section},'original')]), \
                patch.object(dart,'_extract_lockup_ratios_from_tables') as parse:
            updated,evidence=reprocess_record(Path('/tmp'),record,proof,dart,allow_reuse=True)
        parse.assert_not_called()
        self.assertEqual(updated,record)
        self.assertTrue(evidence['lockup_reprocessing']['extraction_reused'])

    def test_full_section_without_confirmed_price_is_not_readmitted(self):
        primary={'rcept_no':'20181116000143','rcept_dt':'2018-11-16'}
        companion={'rcept_no':'20181116000018','rcept_dt':'2018-11-16'}
        proof={'family':[primary],'content_scope':'full_offering_sections',
            'price_resolution_status':'latest_related_prospectus_price_verified',
            'supplementary_status':'official_related_prospectus_verified',
            'supplementary':{'family':[companion],'source_url':'https://dart.fss.or.kr/dsaf001/main.do?rcpNo=20181116000018'}}
        with patch('data.pipelines.lockup_reconciliation.original_body',return_value='cached') as read:
            result=documents(Path('/tmp'),{'rcept_no':primary['rcept_no']},proof)
        self.assertEqual([x[0]['rcept_no'] for x in result],[companion['rcept_no']])
        read.assert_called_once_with(Path('/tmp'),proof['supplementary'])

    def test_interrupted_field_batch_resumes_without_reprocessing_completed_events(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root).resolve();processed=root/'processed';raw=root/'raw'
            parent=processed/'historical_reconciliations/parent';parent.mkdir(parents=True)
            baseline=b'baseline';parser=Path(__file__).parents[1]/'data/collectors/dart_collector.py'
            contract={'parser':digest(parser),'connection':'unchanged'}
            report={'inputs':{'inputs':{'base':'x'},'contract':{
                **contract,'parser':hashlib.sha256(baseline).hexdigest()}},'population':2}
            (parent/'manifest.json').write_text(json.dumps(report))
            events=pd.DataFrame([{'event_id':e,'event_class':'general_ipo'} for e in ['e1','e2']])
            events.to_parquet(parent/'events.parquet',index=False)
            pd.DataFrame([{'event_id':e,'corp_name':e,'status':'official_sources_connected','reason':None,
                'record':json.dumps({'event_id':e,'lockup_commitment_ratio':None}), 'proof':'{"sources":[]}'}
                for e in ['e1','e2']]).to_parquet(parent/'connection_results.parquet',index=False)
            pointer=processed/'current_historical_source_connections.json';pointer.write_text('{"last":"good"}')
            original=pointer.read_bytes()
            def repaired(raw,record,proof,dart,**kwargs):
                return {**record,'lockup_commitment_ratio':.1,'lockup_rcept_no':'receipt','lockup_rule_id':'rule'}, {
                    **proof,'lockup_reprocessing':{'attempts':[]}}
            def interrupted(raw,record,proof,dart,**kwargs):
                if record['event_id']=='e2':raise RuntimeError('interrupted')
                return repaired(raw,record,proof,dart)
            with patch('data.pipelines.lockup_reconciliation.pin_connections',return_value=(parent,report)), \
                    patch('data.pipelines.lockup_reconciliation.load_inputs',return_value={'events':events,'input_sha256':{'base':'x'}}), \
                    patch('data.pipelines.lockup_reconciliation.connection_contract',return_value=contract), \
                    patch('data.pipelines.lockup_reconciliation.subprocess.check_output',return_value=baseline), \
                    patch('data.pipelines.lockup_reconciliation.reprocess_institutional_record',side_effect=lambda raw,record,proof,dart:(record,proof)), \
                    patch('data.pipelines.lockup_reconciliation.unchanged_other_parser_logic',return_value=True):
                with patch('data.pipelines.lockup_reconciliation.reprocess_record',side_effect=interrupted):
                    with self.assertRaisesRegex(RuntimeError,'interrupted'):run(raw,processed,dart=object())
                self.assertEqual(pointer.read_bytes(),original)
                with patch('data.pipelines.lockup_reconciliation.reprocess_record',side_effect=repaired) as repair:
                    result=run(raw,processed,dart=object())
                self.assertEqual(repair.call_count,1)
                self.assertEqual(result['lockup_field_reprocessing']['processed'],2)

    def table(self, none_first=True, total_label='합계'):
        rows=[['구분','국내 기관투자자','국내 기관투자자','합계','합계'],
              ['구분','건수','수량','건수','수량']]
        none=['미확약','90','900','90','900']
        period=['6개월 확약','10','100','10','100']
        return rows+([none,period] if none_first else [period,none])+[[total_label,'100','1000','100','1000']]

    def test_noncommitment_row_before_period_rows_is_not_lost(self):
        for first in [True,False]:
            result=DARTCollector._extract_aggregate_lockup_shares(self.table(first),1000)
            self.assertEqual(result['lockup_commitment_ratio'],.1)
            self.assertTrue(json.loads(result['structured_evidence'])['row_order_independent'])

    def test_standalone_outcome_uses_shares_not_count_percentage(self):
        table=[['구분','참여건수(건)','비중','신청수량(주)'],
               ['1개월 확약','12','5.7%','10,279,000'],
               ['미확약','197','94.3%','112,881,000'],['합계','209','100%','123,160,000']]
        result=DARTCollector._extract_aggregate_lockup_shares(table,heading='③ 의무보유확약 신청내역')
        self.assertAlmostEqual(result['lockup_commitment_ratio'],10279000/123160000)
        self.assertNotEqual(result['lockup_commitment_ratio'],.057)
        self.assertIsNone(DARTCollector._extract_aggregate_lockup_shares(table,heading='주주 보호예수'))

    def test_zero_is_proved_by_complete_quantity_reconciliation_not_dash_alone(self):
        table=self.table()
        table[3][-1]='-';table[2][-1]='1000'
        table.insert(4,['확약 합계','-','-','-','-'])
        table.insert(5,['총 참여건수 및 수량 대비 비율(%)','-','-','-','-'])
        result=DARTCollector._extract_aggregate_lockup_shares(table,1000)
        self.assertEqual(result['lockup_commitment_ratio'],0)
        table[2][-1]='900'
        self.assertIsNone(DARTCollector._extract_aggregate_lockup_shares(table,1000))

    def test_actual_outcome_is_distinguished_from_regulatory_rules_only(self):
        html='<h3>의무보유확약 신청내역</h3><table><tr><td>구분</td><td>참여건수(건)</td><td>비중</td><td>신청수량(주)</td></tr><tr><td>1개월 확약</td><td>1</td><td>50%</td><td>100</td></tr><tr><td>미확약</td><td>1</td><td>50%</td><td>900</td></tr><tr><td>합계</td><td>2</td><td>100%</td><td>1000</td></tr></table>'
        parsed=DARTCollector._extract_lockup_ratios_from_tables(html)
        self.assertEqual(parsed['lockup_commitment_ratio'],.1)
        rules=DARTCollector._extract_lockup_ratios_from_tables('<p>기관투자자 의무보유확약 우대 배정 10%</p>')
        self.assertIsNone(rules['lockup_commitment_ratio'])
        self.assertEqual(rules['rejection_reason'],'selected_document_lockup_rules_without_identified_outcome')

    def test_direct_quantity_percentage_is_not_lost_when_only_commitment_periods_are_shown(self):
        for label in ['총 신청수량 대비 비율','총 신청건수 및 신청수량 대비 비율(%)']:
            html=f'<h3>(다) 의무보유 확약 기관수 및 신청 수량</h3><table><tr><td>구분</td><td>신청건수(건)</td><td>신청수량(주)</td></tr><tr><td>1개월</td><td>31</td><td>60,078,000</td></tr><tr><td>합계</td><td>40</td><td>75,782,000</td></tr><tr><td>{label}</td><td>4.76%</td><td>5.19%</td></tr></table>'
            parsed=DARTCollector._extract_lockup_ratios_from_tables(html)
            self.assertEqual(parsed['lockup_commitment_ratio'],.0519)

    def test_dash_in_subgroup_table_requires_independent_total(self):
        table=self.table();table.insert(4,['1개월 확약','-','-','-','-'])
        self.assertIsNone(DARTCollector._extract_aggregate_lockup_shares(table))
        self.assertEqual(DARTCollector._extract_aggregate_lockup_shares(table,1000)['lockup_commitment_ratio'],.1)

    def test_total_alias_requires_independent_quantity_not_counts(self):
        self.assertIsNone(DARTCollector._extract_aggregate_lockup_shares(self.table(total_label='총 합계')))
        self.assertEqual(DARTCollector._extract_aggregate_lockup_shares(
            self.table(total_label='총 합계'),1000)['lockup_commitment_ratio'],.1)
        self.assertIsNone(DARTCollector._extract_aggregate_lockup_shares(self.table(),100))

    def test_official_percentage_selects_quantity_column_not_participant_count(self):
        table=self.table(False)
        table.insert(3,['총 참여건수 및 수량 대비 비율(%)','10%','20%','1.16%','2.63%'])
        self.assertEqual(DARTCollector._extract_disclosed_aggregate_lockup_ratio(table)['lockup_commitment_ratio'],.0263)

    def test_dashes_and_misaligned_cells_do_not_bypass_reconciliation(self):
        table=self.table();table[3][4]='-'
        self.assertIsNone(DARTCollector._extract_aggregate_lockup_shares(table,1000))
        table=self.table();table[3][4]='10';table[3].append('100')
        self.assertIsNone(DARTCollector._extract_aggregate_lockup_shares(table,1000))

    def test_ast_guard_allows_only_named_field_methods(self):
        before='DEMAND_PARSER_VERSION=13\nclass DARTCollector:\n def _extract_lockup_ratios_from_tables(self):return 1\n def price(self):return 100\n'
        after=before.replace('=13','=14').replace('return 1\n','return 2\n')
        self.assertTrue(unchanged_other_parser_logic(before,after))
        self.assertFalse(unchanged_other_parser_logic(before,after.replace('return 100','return 99')))

    def test_field_repair_keeps_other_values_and_blocks_newer_rejection(self):
        record={'event_id':'e','listing_date':'2024-03-01','offering_price':10000,
                'lockup_commitment_ratio':.2,'institutional_demand_ratio':100}
        docs=[({'rcept_no':'20240201000002','rcept_dt':'2024-02-01'},
               {'source_url':'official','sections':[{'sha256':'a'*64}]},'html'),
              ({'rcept_no':'20240101000001','rcept_dt':'2024-01-01'}, {},'older')]
        dart=DARTCollector(api_key='unused')
        rejected={'lockup_commitment_ratio':None,'parser_validation_status':'candidate_rejected','rule_id':None}
        with patch('data.pipelines.lockup_reconciliation.documents',return_value=docs), \
                patch.object(dart,'_extract_lockup_ratios_from_tables',return_value=rejected) as parse:
            updated,proof=reprocess_record(Path('/tmp'),record,{},dart)
        self.assertIsNone(updated['lockup_commitment_ratio'])
        self.assertEqual(updated['offering_price'],10000)
        self.assertEqual(updated['institutional_demand_ratio'],100)
        self.assertEqual(parse.call_count,1)
