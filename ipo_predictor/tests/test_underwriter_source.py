import unittest

from data.pipelines.underwriter_source import parse_underwriters
from features.source_contracts import DART_UNDERWRITER_TIER_STATUS


def table(broker='한국투자증권',role='대표'):
    return ('<table><tr><td colspan="2">인수인</td><td>증권의 종류</td><td>인수수량</td><td>인수방법</td></tr>'
            f'<tr><td>{role}</td><td>{broker}</td><td>기명식보통주</td><td>1,200,000</td><td>총액인수</td></tr></table>')


class UnderwriterSourceTests(unittest.TestCase):
    def test_explicit_role_schema_and_quantity_are_required(self):
        result=parse_underwriters(table())
        self.assertEqual(result['underwriter_validation_status'],DART_UNDERWRITER_TIER_STATUS)
        self.assertEqual(result['underwriter_tier'],1)
        self.assertIsNone(parse_underwriters(table(role='과거 주관사'))['underwriter_tier'])
        self.assertIsNone(parse_underwriters(table().replace('1,200,000','예정'))['underwriter_tier'])

    def test_legacy_official_alias_and_unknown_broker_are_not_guessed(self):
        self.assertEqual(parse_underwriters(table('엔에이치투자증권㈜'))['underwriter_tier'],1)
        self.assertIsNone(parse_underwriters(table('미래에셋대우'))['underwriter_tier'])
        self.assertIsNone(parse_underwriters(table('임의 회사'))['underwriter_tier'])

    def test_conflicting_tables_do_not_use_first_match(self):
        result=parse_underwriters(table()+table('키움증권'))
        self.assertEqual(result['underwriter_validation_status'],'conflicting_official_underwriter_tables')
        self.assertIsNone(result['underwriter_tier'])
