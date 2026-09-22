import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import build_upcoming_schedule_registry as registry


SUMMARY = '''<table><tr><th>청약기일</th><th>납입기일</th><th>청약공고일</th><th>배정공고일</th><th>배정기준일</th></tr>
<tr><td>2026년 10월 01일 ~ 2026.10.02</td><td>2026.10.07</td><td>-</td><td>-</td><td>-</td></tr></table>
<table><tr><th>증권의 종류</th><th>증권수량</th><th>액면가액</th><th>모집(매출) 가액</th><th>모집(매출) 총액</th><th>모집(매출) 방법</th></tr>
<tr><td>보통주</td><td>100주</td><td>500원</td><td>10,700원</td><td>1,070,000원</td><td>일반공모</td></tr></table>'''


class ScheduleRegistryTests(unittest.TestCase):
    def test_schedule_does_not_approve_price_or_listing(self):
        result = registry.parse_summary(SUMMARY)
        self.assertEqual(result['subscription_start'], '2026-10-01')
        self.assertEqual(result['subscription_end'], '2026-10-02')
        self.assertEqual(result['offering_table_amount_raw'], '10,700원')
        self.assertIsNone(result['final_offering_price'])
        self.assertIsNone(result['listing_date'])
        self.assertEqual(len(result['offering_table_evidence']['raw_values']), 6)

    def test_invalid_dates_and_reversed_range(self):
        for value in ('2026.02.30', '26.10.01', '미정'):
            with self.assertRaises(ValueError):
                registry.parse_date(value)
        with self.assertRaises(ValueError):
            registry.parse_summary(SUMMARY.replace('2026.10.02', '2026.09.30'))

    def test_ambiguous_or_non_public_tables_rejected(self):
        for content in (SUMMARY + SUMMARY, SUMMARY.replace('<td>100주', '<td colspan="2">100주'),
                        SUMMARY.replace('일반공모', '주주배정'), SUMMARY.replace('청약기일', '다른날짜')):
            with self.subTest(content=content):
                with self.assertRaises(ValueError):
                    registry.parse_summary(content)

    def test_registry_checks_hash_calendar_and_lineage(self):
        receipt, corp = '20260909000381', '01028933'
        viewer, summary = b'viewer fixture', SUMMARY.encode()
        entry = {'corp_code': corp, 'rcept_no': receipt, 'reference_name': '멜콘',
                 'collected_at': '2026-09-22T17:00:00+09:00', 'summary_request': {},
                 'viewer_sha256': hashlib.sha256(viewer).hexdigest(),
                 'summary_sha256': hashlib.sha256(summary).hexdigest()}
        events = [{'corp_code': corp, 'rcept_no': receipt, 'corp_name': '멜콘',
                   'event_type': kind, 'subscription_date': value}
                  for kind, value in [('start', '2026-10-01'), ('end', '2026-10-02')]]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'report.json').write_text(json.dumps({'rows': [entry]}))
            (root / f'{receipt}_viewer.html').write_bytes(viewer)
            (root / f'{receipt}_summary.html').write_bytes(summary)
            with patch.object(registry, 'replay', return_value={'events': events}), \
                 patch.object(registry, 'parse_viewer', return_value={'status': 'calendar_receipt_latest_in_viewer_family'}) as lineage, \
                 patch.object(registry, 'summary_node', return_value={}):
                result = registry.build(root, root, {})
                self.assertEqual(len(result['rows']), 1)
                self.assertFalse(result['rows'][0]['model_eligible'])
                self.assertFalse(result['historical_training_data_modified'])
                events[1]['subscription_date'] = '2026-10-03'
                self.assertEqual(len(registry.build(root, root, {})['rejected']), 1)
                events[1]['subscription_date'] = '2026-10-02'
                lineage.return_value = {'status': 'newer_or_ambiguous_registration_review'}
                self.assertEqual(len(registry.build(root, root, {})['rejected']), 1)
                lineage.return_value = {'status': 'calendar_receipt_latest_in_viewer_family'}
                (root / f'{receipt}_summary.html').write_bytes(summary + b'changed')
                result = registry.build(root, root, {})
                self.assertEqual(len(result['rejected']), 1)
                self.assertEqual(result['rows'], [])
