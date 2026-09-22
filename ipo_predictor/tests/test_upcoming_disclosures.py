import unittest

from scripts.audit_upcoming_disclosures import parse_viewer, summary_node


HTML = '''<span onclick="openCorpInfoNew('01028933', 'x')">멜콘</span>
<select id="family">
<option value="rcpNo=20260914100001">2026.09.14 효력발생안내</option>
<option selected value="rcpNo=20260909000381">2026.09.09 [정정] 증권신고서(지분증권)</option>
<option value="rcpNo=20260821000368">2026.08.21 증권신고서(지분증권)</option></select>'''


class DisclosureAuditTests(unittest.TestCase):
    def test_effective_notice_not_selected_as_latest_registration(self):
        result = parse_viewer(HTML, "20260909000381", "01028933", "2026-09-22")
        self.assertEqual(result["latest_registration_candidates"], ["20260909000381"])
        self.assertFalse(result["model_eligible"])

    def test_identity_and_receipt_fail_closed(self):
        for receipt, corp in (("20260909000381", "00000001"), ("20260821000368", "01028933")):
            with self.assertRaises(ValueError):
                parse_viewer(HTML, receipt, corp, "2026-09-22")

    def test_newer_correction_requires_review(self):
        html = HTML.replace("</select>", '<option value="rcpNo=20260921000001">2026.09.21 [정정] 증권신고서(지분증권)</option></select>')
        result = parse_viewer(html, "20260909000381", "01028933", "2026-09-22")
        self.assertEqual(result["status"], "newer_or_ambiguous_registration_review")

    def test_summary_metadata_not_executed_and_receipt_bound(self):
        node = '''<script>var node1 = {};
node1['text'] = "요약정보";
node1['rcpNo'] = "20260909000381";
node1['dcmNo'] = "1234";
node1['eleId'] = "5";
node1['offset'] = "123";
node1['length'] = "500";
node1['dtd'] = "dart4.xsd";</script>'''
        self.assertEqual(summary_node(node, "20260909000381")["length"], "500")
        for html in (node + node, node.replace('"500"', '"invalid"')):
            with self.assertRaises(ValueError):
                summary_node(html, "20260909000381")
        with self.assertRaises(ValueError):
            summary_node(node, "20260821000368")
