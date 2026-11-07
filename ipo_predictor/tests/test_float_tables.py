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
