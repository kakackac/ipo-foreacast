import json
import unittest
from data.collectors.dart_collector import DARTCollector


class InstitutionalParticipationSummaryTests(unittest.TestCase):
    def quantity_result(self, table, heading='수요예측 결과 (가) 수요예측 참여 내역', total='1000'):
        html='<p>'+heading+'</p><table>'+''.join('<tr>'+''.join(
            '<td>'+c+'</td>' for c in row)+'</tr>' for row in table)+'</table>'
        html+='<p>의무보유 확약 신청내역</p><table><tr><td>구분</td><td>신청수량(주)</td></tr>'
        html+='<tr><td>3개월 확약</td><td>100</td></tr><tr><td>미확약</td><td>900</td></tr>'
        html+='<tr><td>총합계</td><td>'+total+'</td></tr></table>'
        return DARTCollector(api_key='unused')._parse_demand_forecast_html(html,'00000001')

    def test_independent_participation_quantity_does_not_need_a_rate(self):
        table=[['구분','국내 기관투자자','해외 기관투자자','합계'],
            ['건수','8','2','10'],['수량','800','200','1000'],['경쟁률','8','2','-']]
        for cell in ['-', '', '771,73:1']:
            table[-1][-1]=cell
            parsed=self.quantity_result(table)
            self.assertIsNone(parsed['institutional_demand_ratio'])
            self.assertEqual(parsed['lockup_commitment_ratio'],.1)
            proof=json.loads(parsed['lockup_structured_evidence'])
            source=proof['independent_application_sources'][0]
            self.assertEqual(source['application_quantity'],1000)
            self.assertEqual(source['table'][-1][-1],cell)
            self.assertEqual(source['rule_id'],'DART_INDEPENDENT_APPLICATION_QUANTITY_V1')

    def test_independent_summary_quantity_does_not_need_a_rate(self):
        table=[['참여건수(건)','신청수량(주)','단순경쟁률'],['10','1000','-']]
        parsed=self.quantity_result(table,heading='기관투자자 수요예측 참여내역')
        self.assertIsNone(parsed['institutional_demand_ratio'])
        self.assertEqual(parsed['lockup_commitment_ratio'],.1)

    def test_independent_quantity_rejects_partial_scope_wrong_units_and_totals(self):
        table=[['구분','국내 기관투자자','해외 기관투자자','합계'],
            ['건수','8','2','10'],['수량','800','200','1000'],['경쟁률','8','2','-']]
        for change in ['domestic_only','retail','planned','allocation','money','scaled_unit',
                       'bad_grouping','quantity_sum','count_sum','duplicate_total']:
            with self.subTest(change=change):
                changed=json.loads(json.dumps(table));heading='수요예측 결과 (가) 수요예측 참여 내역'
                if change=='domestic_only':heading='국내 기관투자자 수요예측 참여내역'
                if change=='retail':heading='일반청약 참여내역'
                if change=='planned':heading+=' 예정'
                if change=='allocation':changed[2][0]='배정수량'
                if change=='money':changed[2][0]='신청금액(원)'
                if change=='scaled_unit':changed[2][0]='신청수량(백만주)'
                if change=='bad_grouping':changed[2][-1]='1,00'
                if change=='quantity_sum':changed[2][1]='801'
                if change=='count_sum':changed[1][1]='9'
                if change=='duplicate_total':changed[0][1]='합계'
                parsed=self.quantity_result(changed,heading=heading)
                self.assertIsNone(parsed['lockup_commitment_ratio'])

    def test_independent_quantity_must_match_lockup_outcome_total(self):
        table=[['참여건수(건)','신청수량(주)','단순경쟁률'],['10','1001','-']]
        parsed=self.quantity_result(table,heading='기관투자자 수요예측 참여내역')
        self.assertIsNone(parsed['lockup_commitment_ratio'])
        self.assertEqual(parsed['lockup_rejection_reason'],'lockup_independent_application_quantity_mismatch')

    def test_malformed_competition_does_not_discard_independently_valid_commitment_quantities(self):
        table=[['구분','국내 기관투자자','해외 기관투자자','합계'],
            ['수량','800','200','1000'],['경쟁률','8','2','771,73:1']]
        html='<p>기관 수요예측 결과</p><table>'+''.join('<tr>'+''.join(
            '<td>'+c+'</td>' for c in row)+'</tr>' for row in table)+'</table>'
        html+='<p>의무보유 확약 신청내역</p><table><tr><td>구분</td><td>신청수량(주)</td></tr>'
        html+='<tr><td>3개월 확약</td><td>100</td></tr><tr><td>미확약</td><td>900</td></tr>'
        html+='<tr><td>합계</td><td>1000</td></tr></table>'
        result=DARTCollector(api_key='unused')._parse_demand_forecast_html(html,'00000001')
        self.assertIsNone(result['institutional_demand_ratio'])
        self.assertEqual(result['lockup_commitment_ratio'],.1)
        self.assertEqual(result['lockup_parser_validation_status'],'structurally_verified')

    def test_malformed_comma_is_not_removed_or_guessed_as_a_decimal(self):
        for cell in ['771,73 : 1','1.134.34:1','1,152.850,023:1']:
            table=[['구분','국내 기관투자자','해외 기관투자자','합계'],
                ['경쟁률','10','20',cell]]
            parsed=DARTCollector._extract_aggregate_demand_ratio(table)
            self.assertIsNone(parsed['institutional_demand_ratio'])
            self.assertEqual(parsed['institutional_demand_rejection_reason'],
                'institutional_aggregate_numeric_grouping_invalid')
            body='<p>기관 수요예측 결과</p><table>'+''.join('<tr>'+''.join(
                '<td>'+c+'</td>' for c in row)+'</tr>' for row in table)+'</table>'
            result=DARTCollector(api_key='unused')._parse_demand_forecast_html(body,'00000001')
            self.assertEqual(result['institutional_demand_rejection_reason'],
                'institutional_aggregate_numeric_grouping_invalid')
            self.assertIsNone(result['institutional_demand_ratio'])
        for cell,expected in [('771.73',771.73),('1,471.04:1',1471.04),('1471.04:1',1471.04)]:
            table=[['구분','국내 기관투자자','해외 기관투자자','합계'],['경쟁률','10','20',cell]]
            self.assertEqual(DARTCollector._extract_aggregate_demand_ratio(table)['institutional_demand_ratio'],expected)
        result=DARTCollector(api_key='unused')._parse_demand_forecast_html(
            '<p>기관 수요예측 경쟁률 771,73 : 1</p>','00000001')
        self.assertIsNone(result['institutional_demand_ratio'])

    def test_official_competition_label_spelling_variant_keeps_explicit_total_cell(self):
        for label in ['경쟁률','경쟁율']:
            table=[['구분','국내 기관투자자','외국 기관투자자','합계'],
                ['건수','100','20','120'],['수량','10000','2000','12000'],
                [label,'10','2','12.34']]
            parsed=DARTCollector._extract_aggregate_demand_ratio(table)
            self.assertEqual(parsed['institutional_demand_ratio'],12.34)
            self.assertEqual(json.loads(parsed['institutional_demand_structured_evidence'])['selected_column'],3)
            table[-1][-1]='-'
            self.assertIsNone(DARTCollector._extract_aggregate_demand_ratio(table))
        summary=[['참여건수(건)','신청수량(주)','단순경쟁율'],['100','12000','12.34']]
        self.assertEqual(DARTCollector._extract_aggregate_demand_ratio(summary,
            '수요예측 결과 기관투자자 수요예측 참여내역')['institutional_demand_ratio'],12.34)

    def test_participant_footnote_is_not_a_second_numeric_value(self):
        heading='(13) 수요예측 결과 ① 기관투자자 수요예측 참여내역'
        table=[['참여건수(건)','신청수량(주)','단순경쟁률'],
               ['506 (주1)','418,689,000','73.1']]
        parsed=DARTCollector._extract_aggregate_demand_ratio(table,heading)
        self.assertEqual(parsed['institutional_demand_ratio'],73.1)
        evidence=json.loads(parsed['institutional_demand_structured_evidence'])
        self.assertEqual(evidence['application_count'],506)
        self.assertEqual(evidence['application_quantity'],418689000)
        self.assertEqual(evidence['table'][1][0],'506 (주1)')
        for invalid in ['506 29','506 (국내 29)','506 (주1) 29','506-29','(주1) 506']:
            table[1][0]=invalid
            self.assertIsNone(DARTCollector._extract_aggregate_demand_ratio(table,heading))
        table[1][0]='506 (주1)';table[1][2]='73.1 (주2)'
        self.assertIsNone(DARTCollector._extract_aggregate_demand_ratio(table,heading))

    def test_actual_summary_heading_and_header_variants(self):
        cases = [
            ('(13) 수요예측 결과 통보 1) 기관투자자 수요예측 참여내역',
             ['참여건수(건)', '신청수량(주)', '단순경쟁률'], ['445', '516,374,000', '153.08:1'], 153.08),
            ('(13) 수요예측 결과 ① 기관투자자 수요예측 참가내역',
             ['참가건수(건)', '참가수량(주)', '단순경쟁률'], ['379', '767,795,100', '345.68 : 1'], 345.68),
            ('(13) 수요예측 결과 ① 기관투자자 수요예측 참여내역',
             ['구분', '참여건수(건)', '참여주식수(주)', '단순경쟁률'],
             ['국내외 기관투자자', '512', '285,927,000', '326.77 : 1'], 326.77),
            ('(12) 수요예측 결과 통보 (가) 기관투자자 수요예측 참여 내역',
             ['참여건수(건)', '참여수량(주)', '단순 경쟁률(:1)'], ['244', '121,277,827', '179.61'], 179.61),
            ('(13) 기관투자자 수요예측 참여 내역 ① 기관투자자 수요예측 참여내역',
             ['참여건수(건)', '신청수량(주)', '단순경쟁률'], ['104', '29,841,000', '39.79 : 1'], 39.79),
        ]
        for heading, header, row, expected in cases:
            with self.subTest(heading=heading):
                result=DARTCollector._extract_aggregate_demand_ratio([header, row], heading)
                self.assertEqual(result['institutional_demand_ratio'], expected)
                self.assertEqual(result['institutional_demand_rule_id'], 'DART_DEMAND_INSTITUTIONAL_SUMMARY_V4')
                proof=json.loads(result['institutional_demand_structured_evidence'])
                self.assertGreater(proof['application_quantity'], 0)

    def test_immediate_institutional_heading_not_earlier_retail_regulations(self):
        table=[['구분','운용사(집합) 공모','운용사(고유)','투자일임사','합계'],
               ['건수','18','289','728','2274'], ['수량','22462000','173192000','576462000','1873827000'],
               ['경쟁률','12.48','96.22','320.26','1041.02']]
        heading='일반청약 및 개인청약은 별도입니다. (13) 수요예측 결과 (가) 수요예측 참여 내역'
        result=DARTCollector._extract_aggregate_demand_ratio(table, heading)
        self.assertEqual(result['institutional_demand_ratio'],1041.02)
        table[-1][-1]='-'
        self.assertIsNone(DARTCollector._extract_aggregate_demand_ratio(table,heading))

    def test_nearby_result_word_is_not_scope_and_unobserved_units_are_not_shares(self):
        table=[['참여건수(건)','신청수량(주)','단순경쟁률'],['21','1,245,000','3.77 :1']]
        for heading in ['수요예측 결과 통보 후 일반청약 참여내역',
                        '국내 기관투자자 수요예측 참여내역', '해외 기관투자자 수요예측 참여내역',
                        '기관투자자 수요예측 참여내역 예정']:
            self.assertIsNone(DARTCollector._extract_aggregate_demand_ratio(table,heading))
        heading='수요예측 결과 통보① 기관투자자 수요예측 참여내역'
        for quantity in ['배정수량(주)','신청금액(원)','신청수량(백만주)']:
            table[0][1]=quantity
            self.assertIsNone(DARTCollector._extract_aggregate_demand_ratio(table,heading))

    def test_result_notice_and_parenthesized_heading_keep_aggregate_scope(self):
        for heading in ['(12) 수요예측 결과 통보① 기관투자자 수요예측 참여내역',
                        '(12) 수요예측 결과(가) 기관투자자 수요예측 참여내역',
                        '(9) 수요예측 결과 및 배정결과 통보 ① 수요예측 참여내역']:
            table=[['참여건수(건)','신청수량(주)','단순 경쟁률'],['442','100,002,000','40.88:1']]
            parsed=DARTCollector._extract_aggregate_demand_ratio(table,heading)
            self.assertIsNotNone(parsed)
            self.assertEqual(parsed['institutional_demand_ratio'],40.88)
            self.assertEqual(json.loads(parsed['institutional_demand_structured_evidence'])['application_quantity'],100002000)

    def test_explicit_units_and_domestic_foreign_total_not_domestic_only(self):
        heading='수요예측 결과 통보① 기관투자자 수요예측 참여내역'
        table=[['참여건수','신청수량','단순경쟁률'],['21건','1,245,000주','3.77 :1']]
        self.assertEqual(DARTCollector._extract_aggregate_demand_ratio(table,heading)['institutional_demand_ratio'],3.77)
        table=[['구분','참여건수(건)','신청수량(주)','단순경쟁률'],['국내외 기관투자자','554','5,801,400,000','455.1:1']]
        self.assertEqual(DARTCollector._extract_aggregate_demand_ratio(table,heading)['institutional_demand_ratio'],455.1)
        table[1][0]='국내 기관투자자'
        self.assertIsNone(DARTCollector._extract_aggregate_demand_ratio(table,heading))

    def test_lockup_total_conflict_is_not_filled_from_a_matching_internal_subtotal(self):
        html='<h3>수요예측 결과 통보① 기관투자자 수요예측 참여내역</h3><table><tr><td>참여건수(건)</td><td>신청수량(주)</td><td>단순경쟁률</td></tr><tr><td>442</td><td>100,002,000</td><td>40.88:1</td></tr></table><h3>의무보유확약 신청내역</h3><table><tr><td>구분</td><td>참여건수(건)</td><td>신청수량(주)</td></tr><tr><td>1개월 확약</td><td>1</td><td>60,000</td></tr><tr><td>미확약</td><td>441</td><td>101,942,000</td></tr><tr><td>합계</td><td>442</td><td>102,002,000</td></tr></table>'
        parsed=DARTCollector._extract_lockup_ratios_from_tables(html)
        self.assertIsNone(parsed['lockup_commitment_ratio'])
        self.assertEqual(parsed['rejection_reason'],'lockup_independent_application_quantity_mismatch')
