import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.audit_dart_subscription_calendar import parse_calendar
from scripts.audit_upcoming_disclosures import calendar_candidates
from scripts.compare_subscription_snapshots import compare


class CalendarDiscoveryTests(unittest.TestCase):
    def test_empty_notice_is_not_a_date(self):
        html = '<select id="year"><option selected value="2026"></option></select><select id="month"><option selected value="12"></option></select>'
        html += '<li class="day emptyData"><div>조회 결과가 없습니다.</div></li>'
        html += ''.join(f'<li class="day"><div class="date">{day}</div></li>' for day in range(1, 32))
        self.assertEqual(parse_calendar(html, 2026, 12), [])
        with self.assertRaises(ValueError):
            parse_calendar('<li class="day emptyData">maintenance</li>', 2026, 12)

    def test_discovery_uses_receipt_not_name_and_includes_end_only(self):
        rows = [{'corp_code': '1', 'corp_name': 'same', 'rcept_no': receipt, 'event_type': event}
                for receipt, event in [('A', 'start'), ('A', 'end'), ('B', 'end')]]
        result = calendar_candidates(rows)
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]['matches']), 2)
        self.assertEqual(result[1]['matches'][0]['event_type'], 'end')

    def test_changes_do_not_infer_withdrawal_and_ignore_nonoverlap(self):
        def event(corp, receipt, value):
            return {'corp_code': corp, 'rcept_no': receipt, 'subscription_date': value, 'event_type': 'start'}
        with tempfile.TemporaryDirectory() as folder:
            old, new = Path(folder) / 'old', Path(folder) / 'new'
            for path, months in ((old, [10]), (new, [10, 11])):
                path.mkdir()
                (path / 'report.json').write_text(json.dumps({'requests': [{'year': 2026, 'month': m} for m in months]}))
            with patch('scripts.compare_subscription_snapshots.replay', side_effect=[
                {'events': [event('1', 'A', '2026-10-01'), event('2', 'B', '2026-10-02')]},
                {'events': [event('1', 'C', '2026-10-03'), event('3', 'D', '2026-11-01')]}]):
                result = compare(old, new)
            self.assertEqual([r['status'] for r in result['rows']], ['receipt_or_schedule_changed', 'no_longer_observed_requires_review'])
            self.assertFalse(result['withdrawal_inferred'])
