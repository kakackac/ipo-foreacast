import json
import hashlib
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import pandas as pd

from data.collectors.dart_collector import DARTCollector
from data.pipelines.lockup_reconciliation import unchanged_other_parser_logic, reprocess_record, reprocess_institutional_record, run, documents
from data.pipelines.data_delivery import digest


class LockupRepairTests(unittest.TestCase):
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
