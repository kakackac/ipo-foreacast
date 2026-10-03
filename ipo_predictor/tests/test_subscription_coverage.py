import unittest
from datetime import date

from data.collectors.upcoming_ipo_collector import classify_schedule, schedule_range
from scripts.audit_dart_subscription_calendar import parse_calendar, compare_reference


class ScheduleCoverageTests(unittest.TestCase):
    def test_subscription_is_kept_without_listing_date(self):
        row = classify_schedule({"subscription_schedule": "2026-10-01 ~ 2026-10-02",
                                 "demand_schedule": "미정", "listing_date": None}, date(2026, 9, 22))
        self.assertEqual(row["subscription_state"], "subscription_upcoming")
        self.assertEqual(row["subscription_d_day"], 9)
        self.assertTrue(row["service_schedule_candidate"])
        self.assertIsNone(row["listing_date"])

    def test_period_boundaries_and_unconfirmed(self):
        for day, expected in ((1, "subscription_open"), (2, "subscription_open"), (3, "subscription_period_elapsed")):
            row = classify_schedule({"subscription_schedule": "2026-10-01 ~ 2026-10-02",
                                     "demand_schedule": "-", "listing_date": None}, date(2026, 10, day))
            self.assertEqual(row["subscription_state"], expected)
            self.assertTrue(row["service_schedule_candidate"])
        self.assertEqual(schedule_range("2026-10-02 ~ 2026-10-01")[2], "schedule_review_required")
        self.assertEqual(schedule_range("2026-02-30 ~ 2026-03-01")[2], "schedule_review_required")

    def test_calendar_identity_and_non_ipo_gate(self):
        html = '''<select id="year"><option selected value="2026"></option></select>
        <select id="month"><option selected value="10"></option></select>
        <li class="day"><div class="date">1</div>
        <a href="/dsaf001/main.do?rcpNo=20260909000381"><div class="gm 01028933"><span>기</span>멜콘 [시작]</div></a></li>'''
        result = parse_calendar(html, 2026, 10)
        self.assertEqual(result[0]["corp_code"], "01028933")
        self.assertEqual(result[0]["subscription_date"], "2026-10-01")
        self.assertFalse(result[0]["model_eligible"])
        with self.assertRaises(ValueError):
            parse_calendar(html, 2026, 11)
        with self.assertRaises(ValueError):
            parse_calendar("<html>점검중</html>", 2026, 10)

    def test_alias_requires_recorded_corporate_code(self):
        reference = {"companies": [{"name": "별칭", "official_name_candidate": "공식명", "official_corp_code_candidate": "00000001"}]}
        events = [{"corp_name": "공식명", "corp_code": "00000002"}]
        self.assertFalse(compare_reference(events, reference)[0]["matches"])
        events[0]["corp_code"] = "00000001"
        self.assertTrue(compare_reference(events, reference)[0]["matches"])
