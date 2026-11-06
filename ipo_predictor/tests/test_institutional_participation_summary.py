import json
import unittest
from data.collectors.dart_collector import DARTCollector


class InstitutionalParticipationSummaryTests(unittest.TestCase):
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
