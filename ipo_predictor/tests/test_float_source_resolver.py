import unittest
from data.pipelines.float_source_resolver import resolve_public_float


class FloatSourceResolverTests(unittest.TestCase):
    def document(self, receipt, date, ratio=None, conflict=False, supplementary=False, price=22500):
        return ({"rcept_no": receipt, "rcept_dt": date, "supplementary": supplementary}, {
            "offering_price": price, "public_float_ratio_disclosed": ratio,
            "public_float_parse_method": "conflicting_float_ratios_review_required" if conflict
            else "disclosed_public_float_ratio_direct_context",
        })

    def test_latest_conflict_does_not_restore_old_value(self):
        documents = [self.document("20251114001150", "2025-11-14", .3523),
                     self.document("20251205000229", "2025-12-05", conflict=True)]
        result = resolve_public_float(documents, 22500, "2025-12-18")
        self.assertIsNone(result["public_float_ratio_disclosed"])
        self.assertEqual(result["public_float_resolution_status"], "latest_conflict_blocks_older_fallback")

    def test_later_price_linked_prospectus_resolves_conflict(self):
        documents = [self.document("20251205000229", "2025-12-05", conflict=True),
                     self.document("20251205000237", "2025-12-05", .3527, supplementary=True)]
        result = resolve_public_float(documents, 22500, "2025-12-18")
        self.assertEqual(result["public_float_ratio_disclosed"], .3527)
        self.assertEqual(result["public_float_rcept_no"], "20251205000237")

    def test_wrong_price_cannot_restore_old_value(self):
        documents = [self.document("20251114001150", "2025-11-14", .3523),
                     self.document("20251205000237", "2025-12-05", .3527,
                                   supplementary=True, price=17000)]
        self.assertIsNone(resolve_public_float(documents, 22500, "2025-12-18")["public_float_ratio_disclosed"])

    def test_listing_day_source_is_not_used(self):
        result = resolve_public_float([self.document("20251218000237", "2025-12-18", .3527)],
                                      22500, "2025-12-18")
        self.assertIsNone(result["public_float_ratio_disclosed"])
