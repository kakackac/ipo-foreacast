import unittest
from data.collectors.float_tables import ownership_float
from data.collectors.dart_collector import DARTCollector
from data.processors.feature_engineer import FeatureEngineer
import pandas as pd


class FloatTablesTests(unittest.TestCase):
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
