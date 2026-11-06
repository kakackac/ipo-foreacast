import json
import unittest
from data.collectors.dart_collector import DARTCollector


class InstitutionalParticipationSummaryTests(unittest.TestCase):
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
