import unittest
from data.collectors.float_tables import ownership_float
from data.collectors.dart_collector import DARTCollector
from data.processors.feature_engineer import FeatureEngineer
import pandas as pd


class FloatTablesTests(unittest.TestCase):
    def test_only_quantity_column_can_use_adjacent_post_offering_percentage(self):
        table=[['구분','성명','주식수','공모후 지분율','매도금지기간'],
               ['유통가능물량 소계','유통가능물량 소계','3,332,040','62.15%','-'],
               ['합계','합계','5,361,500','100.0%','-']]
        result=ownership_float([table])
        self.assertEqual(result['public_float_shares'],3332040)
        self.assertEqual(result['total_post_listing_shares'],5361500)
        self.assertAlmostEqual(result['public_float_ratio_disclosed'],.6215)
        table[0][3]='공모전 지분율'
        self.assertEqual(ownership_float([table]),{})
        table[0][3]='지분율'
        self.assertEqual(ownership_float([table]),{})
        table[0][3]='공모후 지분율';table[0][2]='공모전 주식수'
        self.assertEqual(ownership_float([table]),{})

    def test_initial_float_subtotal_uses_post_not_pre_offering_quantity(self):
        table=[['구분','공모 전','공모 전','공모 후','공모 후'],
               ['구분','주식수','지분율','주식수','지분율'],
               ['상장직후 유통가능물량 소계','9,160,050','30.13%','12,710,050','36.84%'],
               ['합계','30,400,000','100.00%','34,500,000','100.00%']]
        result=ownership_float([table])
        self.assertEqual(result['public_float_shares'],12710050)
        self.assertEqual(result['total_post_listing_shares'],34500000)
        self.assertAlmostEqual(result['public_float_ratio_disclosed'],.3684)
        table[2][0]='상장 6개월 후 유통가능물량 소계'
        self.assertEqual(ownership_float([table]),{})

    def test_float_subtotal_reference_is_not_a_different_scope(self):
        table=[['구분','구분','공모 후 주식수(주)','지분율(%)','보호예수기간'],
               ['유통제한물량','유통제한물량 소계 (1)','1,606,762','39.9%','-'],
               ['유통가능물량','벤처금융','204,746','5.1%','-'],
               ['유통가능물량','기타주주','2,217,192','55.0%','-'],
               ['유통가능물량','유통가능물량 소계 (2)','2,421,938','60.1%','-'],
               ['상장예정주식수 합계 (1+2)','상장예정주식수 합계 (1+2)','4,028,700','100.0%','-']]
        for label in ['유통가능물량 소계 (2)','유통가능물량 소계 (주2)','소계 (2)']:
            with self.subTest(label=label):
                table[4][1]=label
                result=ownership_float([table])
                self.assertEqual(result['public_float_shares'],2421938)
                self.assertEqual(result['total_post_listing_shares'],4028700)
                self.assertAlmostEqual(result['public_float_ratio_disclosed'],.601)
                self.assertEqual(result['total_post_listing_shares_parse_method'],'same_table_post_offering_scope_v1')
        table[4][1]='유통가능물량 소계 (2년 후)'
        self.assertEqual(ownership_float([table]),{})

    def test_numbered_investor_subtotal_is_not_entire_float(self):
        table=[['구분','구분','구분','공모 후 주식수','지분율'],
               ['유통가능물량','벤처금융','소계 (2)','10','10%'],
               ['합계','합계','합계','100','100%']]
        self.assertEqual(ownership_float([table]),{})

    def test_inline_percentage_before_share_quantity_is_not_discarded(self):
        text='당사의 상장예정주식 수 48,052,977주 중 약 39.10%에 해당하는 18,790,687주는 상장 직후 유통가능물량입니다.'
        result=ownership_float([[[text]]])
        self.assertAlmostEqual(result['public_float_ratio_disclosed'],.3910)
        wrong=text.replace('39.10%','49.10%')
        self.assertEqual(ownership_float([[[wrong]]])['public_float_basis_status'],'same_table_candidates_conflict')
        table=[['구분','주주명','주식의 종류','유통가능물량','유통가능물량','매각제한물량'],
               ['구분','주주명','주식의 종류','주식수','지분율','주식수'],
               ['합계','합계','보통주','18,702,287','38.92%','29,350,690']]
        result=ownership_float([table,[[text]]])
        self.assertEqual(result['public_float_basis_status'],'same_table_candidates_conflict')

    def test_different_printed_precision_is_not_itself_a_float_conflict(self):
        text='당사의 상장예정주식수 23,666,666주 중 42.47%에 해당하는 10,051,128주는 상장 직후 유통가능물량입니다.'
        table=[['구분','주주명','주식의 종류','유통가능물량','유통가능물량','매각제한물량'],
               ['구분','주주명','주식의 종류','주식수','지분율','주식수'],
               ['합계','합계','보통주','10,051,128','42.5%','13,615,538']]
        self.assertAlmostEqual(ownership_float([table,[[text]]])['public_float_ratio_disclosed'],.4247)

    def test_coarse_initial_percentage_is_checked_at_its_actual_precision(self):
        for total,n,pct in [(66165000,13277441,'20'),(48898070,20276708,'41.5'),
                            (22000000,5520000,'25.1'),(9551420,2592062,'27.13')]:
            text=f'당사의 상장예정주식수 {total}주 중 약 {pct}%에 해당하는 {n}주는 상장 직후 유통가능물량입니다.'
            self.assertAlmostEqual(ownership_float([[[text]]])['public_float_ratio_disclosed'],float(pct)/100)
        coarse='당사의 상장예정주식수 10000주 중 20%에 해당하는 2007주는 상장 직후 유통가능물량입니다.'
        precise=[['구분','공모후 주식수','공모후 주식수','유통가능물량','유통가능물량'],
                 ['구분','주식수','지분율','주식수','지분율'],['합계','10000','100%','2007','20.07%']]
        self.assertAlmostEqual(ownership_float([precise,[[coarse]]])['public_float_ratio_disclosed'],.2007)

    def test_explicit_initial_float_clause_is_bound_to_same_issuer_total(self):
        text='당사의 상장예정주식수 3,460,340주 중 보호예수 대상은 2,083,000주입니다. 상장예정주식수의 39.8%에 해당하는 1,377,340주는 상장 직후 유통가능물량입니다.'
        result=ownership_float([[[text]]])
        self.assertEqual(result['public_float_shares'],1377340)
        self.assertEqual(result['total_post_listing_shares'],3460340)
        self.assertAlmostEqual(result['public_float_ratio_disclosed'],.398)
        self.assertEqual(ownership_float([[[text.replace('당사의','비교회사의')]]]),{})
        excluded='회사의 상장예정주식수 6,212,799주 중 3,996,472주(공모 후, 63.8%)를 제외하고는 상장 직후 유통가능 물량입니다.'
        self.assertEqual(ownership_float([[[excluded]]]),{})
        future=text.replace('상장 직후','보호예수 종료 후')
        self.assertEqual(ownership_float([[[future]]]),{})
        wrong=text.replace('39.8%','49.8%')
        self.assertEqual(ownership_float([[[wrong]]])['public_float_parse_method'],'conflicting_float_ratios_review_required')

    def test_invalid_ratio_is_missing_not_clipped_into_valid_observation(self):
        result=FeatureEngineer('phase2')._calc_supply_structure_features(pd.DataFrame({
            'public_float_shares':[200], 'total_post_listing_shares':[100],
            'public_float_parse_method':['disclosed_public_float_shares_direct_context'],
            'total_post_listing_shares_parser_validation_status':['structurally_verified'],
            'total_post_listing_shares_parse_method':['same_table_post_offering_scope_v1']}))
        self.assertTrue(pd.isna(result.iloc[0].float_share_ratio))
        self.assertEqual(result.iloc[0].float_share_validation_status,'public_float_not_structurally_verified')
    def test_post_offering_column_not_pre_offering(self):
        table=[['구분','공모전 주식수','공모전 주식수','공모후 주식수','공모후 주식수'],
               ['구분','주식수','지분율','주식수','지분율'],
               ['유통가능물량 소계','24,005,000','48.01%','27,920,600','53.62%'],
               ['총계','50,000,000','100.00%','52,070,000','100.00%']]
        result=ownership_float([table])
        self.assertEqual(result['total_post_listing_shares'],52070000)
        self.assertEqual(result['public_float_shares'],27920600)
        self.assertAlmostEqual(result['public_float_ratio_disclosed'],.5362)

    def test_explicit_listed_scope_not_rcps_outstanding_label(self):
        table=[['구분','주식수','공모 후 지분율'],
               ['유통가능주식수(2)','2,396,023주','9.97%'],
               ['상장 후 총 발행주식수(1+2)','24,031,048주','100.00%']]
        html='<p>발행주식총수 1,423,848주</p><table>'+''.join('<tr>'+''.join('<td>'+s+'</td>' for s in r)+'</tr>' for r in table)+'</table>'
        parsed=DARTCollector(api_key='unused')._parse_offering_html(html,'20191120000496')
        self.assertEqual(parsed['total_post_listing_shares'],24031048)
        self.assertAlmostEqual(parsed['public_float_ratio_disclosed'],.0997)

    def test_ambiguous_or_inconsistent_pair_not_approved(self):
        table=[['구분','주식수','공모후 지분율'],['유통가능주식수(2)','20주','20%'],
               ['상장 후 총 발행주식수(1+2)','100주','100%']]
        other=[r[:] for r in table];other[1]=['유통가능주식수(2)','30주','30%']
        self.assertEqual(ownership_float([table,other])['public_float_basis_status'],'same_table_candidates_conflict')
        other[1]=['유통가능주식수(2)','120주','30%']
        self.assertEqual(ownership_float([other]),{})

    def test_investor_subtotal_is_not_the_entire_float(self):
        table=[['구분','구분','구분','공모후 주식수','공모후 지분율'],
               ['유통가능물량','전문투자자','소계','10','10%'],
               ['유통가능물량','유통가능물량 소계','소계','40','40%'],
               ['합계','합계','합계','100','100%']]
        self.assertEqual(ownership_float([table])['public_float_shares'],40)

    def test_initial_group_subtotal_can_have_ordinary_share_class_between_labels(self):
        table=[['구분','종류','구분','공모후 주식수','공모후 지분율'],
               ['유통제한물량','보통주','소계','44,717,113','73.12%'],
               ['상장직후유통가능물량주)','보통주','기존주주','50,000','0.08%'],
               ['상장직후유통가능물량주)','보통주','공모주주','16,390,377','26.80%'],
               ['상장직후유통가능물량주)','보통주','소계','16,440,377','26.88%'],
               ['합계','합계','합계','61,157,490','100.00%']]
        result=ownership_float([table])
        self.assertEqual(result['public_float_shares'],16440377)
        self.assertEqual(result['total_post_listing_shares'],61157490)
        self.assertAlmostEqual(result['public_float_ratio_disclosed'],.2688)
        table[4][1]='벤처금융'
        self.assertEqual(ownership_float([table]),{})

    def test_post_prefixed_quantity_column_is_not_confused_with_locked_quantity(self):
        table=[['구분','공모전','공모전','공모후','공모후','공모후유통가능물량','공모후매도금지물량'],
               ['구분','주식수','지분율','주식수','지분율','공모후유통가능물량','공모후매도금지물량'],
               ['합계','8,225,421','100%','11,077,521','100%','5,768,451','5,309,070']]
        result=ownership_float([table])
        self.assertEqual(result['public_float_shares'],5768451)
        self.assertEqual(result['total_post_listing_shares'],11077521)
        self.assertIsNone(result['public_float_ratio_disclosed'])
        table[0][5]=table[1][5]='상장6개월후유통가능물량'
        self.assertEqual(ownership_float([table]),{})

    def test_inline_quantity_percent_has_its_own_percentage_cell(self):
        table=[['구분','공모후','공모후','공모후 유통가능물량','공모후 매도금지물량'],
               ['구분','주식수','지분율','공모후 유통가능물량','공모후 매도금지물량'],
               ['합계','5,345,090','100.0%','2,301,570(43.06%)','3,043,520(56.94%)']]
        result=ownership_float([table])
        self.assertEqual(result['public_float_shares'],2301570)
        self.assertAlmostEqual(result['public_float_ratio_disclosed'],.4306)
        table[2][3]='2,301,570(53.06%)'
        self.assertEqual(ownership_float([table])['public_float_basis_status'],'same_table_candidates_conflict')

    def test_group_share_subtotal_is_not_each_investors_quantity(self):
        table=[['구분','구분','구분','공모후주식수','공모후지분율'],
               ['유통가능주식수','공모주주','A','20','20%'],
               ['유통가능주식수','기존주주','B','30','30%'],
               ['유통가능주식수','소계','소계','50','50%'],
               ['총합계','총합계','총합계','100','100%']]
        self.assertEqual(ownership_float([table])['public_float_shares'],50)
        table[3]=['유통가능주식수','벤처금융','소계','10','10%']
        self.assertEqual(ownership_float([table]),{})

    def test_named_float_total_accepts_explicit_listed_total_not_future_total(self):
        table=[['구분','구분','주식종류','공모후주식수','공모후지분율'],
               ['유통가능 주식 합계','유통가능 주식 합계','보통주','40','40%'],
               ['상장후 총 발행주식 합계','상장후 총 발행주식 합계','보통주','100','100%']]
        self.assertEqual(ownership_float([table])['public_float_shares'],40)
        table[1][0]=table[1][1]='상장 1년 후 유통가능 주식 합계'
        self.assertEqual(ownership_float([table]),{})

    def test_positive_preferred_or_convertible_amount_cannot_define_listed_denominator(self):
        for kind in ['전환상환우선주','우선주','전환사채']:
            table=[['구분','종류','구분','공모후주식수','공모후지분율'],
                   ['유통제한물량',kind,'A','10','10%'],
                   ['유통가능물량','보통주','소계','40','40%'],
                   ['합계','합계','합계','100','100%']]
            self.assertEqual(ownership_float([table]),{})
            table[1][3]='-'
            self.assertEqual(ownership_float([table])['public_float_shares'],40)

    def test_malformed_grouped_quantity_is_not_normalized_into_valid_number(self):
        table=[['구분','공모후주식수','공모후지분율'],
               ['유통가능 주식 합계','4,0','40%'], ['합계','100','100%']]
        self.assertEqual(ownership_float([table]),{})

    def test_multiple_post_exercise_bases_are_not_selected_by_column_order(self):
        table=[['구분','공모후','공모후','공모후','공모후'],
               ['구분','행사전주식수','지분율','행사후주식수','지분율'],
               ['유통가능 소계','40','40%','60','50%'],['합계','100','100%','120','100%']]
        self.assertEqual(ownership_float([table]),{})

    def test_shareholder_subset_total_is_not_the_issuers_post_total(self):
        table=[['주주명','공모후','공모후','공모후유통가능물량','공모후유통가능물량'],
               ['주주명','보유주식','지분율','주식수','지분율'],
               ['합계','100','20%','50','10%']]
        self.assertEqual(ownership_float([table]),{})

    def test_explicit_voluntary_adjustment_does_not_silently_approve_unadjusted_float(self):
        base=[['구분','공모후','공모후','유통가능물량','유통가능물량'],
              ['구분','주식수','지분율','주식수','지분율'],['합계','100','100%','50','50%']]
        adjusted=[['구분','유통가능주식수','유통가능주식수비율',
                   '자발적매각제한확약후유통가능주식수','자발적매각제한확약후유통가능주식수비율'],
                  ['상장일 유통가능','50','50%','40','40%']]
        result=ownership_float([base,adjusted])
        self.assertEqual(result['public_float_basis_status'],'same_table_candidates_conflict')
        self.assertIn('explicit_initial_float_basis_adjustment_requires_reconciliation_v2',result['public_float_parse_evidence'])

    def test_direct_float_label_can_span_two_cells_before_quantity(self):
        table=[['구분','성명','주식수','지분율','보호예수기간'],
               ['유통가능주식수(2)','유통가능주식수(2)','5,558,932','31.8%','-'],
               ['상장후총발행주식수(1+2)','상장후총발행주식수(1+2)','17,500,000','100%','-']]
        result=ownership_float([table])
        self.assertEqual(result['public_float_shares'],5558932)
        self.assertEqual(result['public_float_ratio_disclosed'],.318)

    def test_total_bare_100_is_dated_quantity_basis_only_with_percentage_header(self):
        table=[['구분','공모후','공모후','유통가능물량','매도금지물량'],
               ['구분','주식수','지분율','유통가능물량','매도금지물량'],
               ['합계','1000','100.0','300','700']]
        result=ownership_float([table])
        self.assertEqual(result['public_float_shares'],300)
        self.assertEqual(result['total_post_listing_shares'],1000)
        self.assertIsNone(result['public_float_ratio_disclosed'])
        table[1][2]='보유수량'
        self.assertEqual(ownership_float([table]),{})

    def test_explicit_listed_total_must_not_use_diluted_basis_percentage(self):
        table=[['구분','공모후 주주명부','공모후 주주명부','유통가능물량','유통가능물량'],
               ['구분','주식수','지분율','주식수','지분율'],
               ['보통주(상장예정주식수)','33,846,192','95.01%','10,052,575','28.22%'],
               ['상환전환우선주(투자자)','1,777,568','4.99%','888,784','2.49%'],
               ['희석가능주식합계','35,623,760','100%','10,941,359','30.71%']]
        result=ownership_float([table])
        self.assertEqual(result['public_float_shares'],10052575)
        self.assertEqual(result['total_post_listing_shares'],33846192)
        self.assertIsNone(result['public_float_ratio_disclosed'])
        self.assertIn('28.22%',result['public_float_parse_evidence'])
        text='당사의 상장예정주식수 33,846,192주 중 29.7%에 해당하는 10,052,575주는 상장 직후 유통가능물량입니다.'
        self.assertAlmostEqual(ownership_float([table,[[text]]])['public_float_ratio_disclosed'],.297)

    def test_offering_percentage_note_in_quantity_header_is_not_percentage_column(self):
        table=[['구분','공모후(공모비율20%)','공모후(공모비율20%)','유통가능주식수','유통불가능주식수'],
               ['구분','소유주식수','비율','유통가능주식수','유통불가능주식수'],
               ['합계','19,062,968','100%','7,375,898','11,687,070']]
        result=ownership_float([table])
        self.assertEqual(result['total_post_listing_shares'],19062968)
        self.assertEqual(result['public_float_shares'],7375898)

    def test_malformed_header_cannot_contradict_independently_bound_complete_grid(self):
        table=[['구분','공모후','공모후','기간','기간','기간','기간','기간','기간'],
               ['구분','공모후','공모후','보유주식','보유주식','매각제한물량','매각제한물량','유통가능물량','유통가능물량'],
               ['구분','주식수','지분율','주식수','지분율','주식수','지분율','주식수','지분율'],
               ['합계','80','100%','100','100%','70','70%','30','30%']]
        text='당사의 상장예정주식수 100주 중 30%에 해당하는 30주는 상장 직후 유통가능물량입니다.'
        result=ownership_float([table,[[text]]])
        self.assertEqual(result['total_post_listing_shares'],100)
        self.assertEqual(result['public_float_ratio_disclosed'],.3)
        self.assertIn('resolve_bad_header',result['public_float_parse_evidence'])
        self.assertEqual(ownership_float([table])['public_float_basis_status'],'same_table_candidates_conflict')
        table[3][5]='69'
        self.assertEqual(ownership_float([table,[[text]]])['public_float_basis_status'],'same_table_candidates_conflict')

    def test_explicit_ordinary_totals_are_not_excluded_due_to_other_preferred_rows(self):
        table=[['구분','구분','주식의종류','공모후주식수','지분율'],
               ['유통제한물량','주주','우선주','516,750','5.52%'],
               ['유통가능물량','유통가능물량소계','보통주','2,633,480','28.13%'],
               ['합계','합계','보통주','8,843,825','94.48%'],
               ['합계','합계','우선주','516,750','5.52%'],
               ['합계','합계','전체주식수','9,360,575','100%']]
        result=ownership_float([table])
        self.assertEqual(result['public_float_shares'],2633480)
        self.assertEqual(result['total_post_listing_shares'],8843825)
        self.assertIsNone(result['public_float_ratio_disclosed'])
        self.assertIn('28.13%',result['public_float_parse_evidence'])
        table[2][2]='우선주'
        self.assertEqual(ownership_float([table]),{})

    def test_float_column_uses_same_ordinary_total_not_all_issued_class_total(self):
        table=[['구분','종류','공모후','공모후','유통가능물량','유통가능물량'],
               ['구분','종류','주식수','지분율','주식수','지분율'],
               ['합계','보통주','11,026,646','93.91%','4,086,746','34.81%'],
               ['합계','우선주','714,828','6.09%','324,783','2.77%'],
               ['합계','계','11,741,474','100%','4,411,529','37.57%']]
        result=ownership_float([table])
        self.assertEqual(result['public_float_shares'],4086746)
        self.assertEqual(result['total_post_listing_shares'],11026646)
        self.assertIsNone(result['public_float_ratio_disclosed'])

    def test_separate_column_totals_of_100_percent_are_not_entire_issuer_float(self):
        table=[['구분','공모후기준','공모후기준','유통불가능','유통불가능','유통가능','유통가능'],
               ['구분','주식수','지분율','주식수','지분율','주식수','지분율'],
               ['합계','100','100%','60','100%','40','100%']]
        result=ownership_float([table])
        self.assertEqual(result['public_float_shares'],40)
        self.assertEqual(result['total_post_listing_shares'],100)
        self.assertIsNone(result['public_float_ratio_disclosed'])
        self.assertIn('within_float_and_locked_columns',result['public_float_parse_evidence'])
        table[2][3]='59'
        self.assertEqual(ownership_float([table])['public_float_basis_status'],'same_table_candidates_conflict')
