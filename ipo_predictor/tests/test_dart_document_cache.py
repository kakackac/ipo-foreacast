import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock

import requests

from data.collectors.dart_collector import DARTCollector


class DocumentCacheTests(unittest.TestCase):
    receipt = "20210111000425"

    def response(self, text="<p>공식 원문&cr;</p>"):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("document.xml", text)
        response = Mock()
        response.content = buffer.getvalue()
        return response

    def test_raw_source_reused_without_key_or_network(self):
        with tempfile.TemporaryDirectory() as directory:
            collector = DARTCollector("test-secret", directory)
            collector.session.get = Mock(return_value=self.response())
            original = collector.get_document_text(self.receipt)
            offline = DARTCollector("", directory)
            offline.session.get = Mock(side_effect=AssertionError("network forbidden"))
            self.assertEqual(offline.get_document_text(self.receipt), original)
            collector.session.get.assert_called_once()
            stored = (Path(directory) / f"{self.receipt}.json").read_text()
            self.assertNotIn("test-secret", stored)
            self.assertIn("&cr;", stored)
            self.assertIn("collected_at", json.loads(stored))

    def test_corrupt_cache_refetched_and_refresh_replaces(self):
        with tempfile.TemporaryDirectory() as directory:
            collector = DARTCollector("test-secret", directory)
            collector.session.get = Mock(return_value=self.response())
            collector.get_document_text(self.receipt)
            path = Path(directory) / f"{self.receipt}.json"
            stored = json.loads(path.read_text())
            stored["decoded_document"] = "tampered"
            path.write_text(json.dumps(stored))
            self.assertNotEqual(collector.get_document_text(self.receipt), "tampered")
            collector.get_document_text(self.receipt, refresh=True)
            self.assertEqual(collector.session.get.call_count, 3)

    def test_failed_response_is_not_cached_and_secret_not_in_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            collector = DARTCollector("test-secret", directory)
            collector.session.get = Mock(side_effect=requests.Timeout("url?crtfc_key=test-secret"))
            with self.assertRaises(RuntimeError) as error:
                collector.get_document_text(self.receipt)
            self.assertNotIn("test-secret", str(error.exception))
            response = Mock(content=b"<result><status>014</status></result>")
            collector.session.get = Mock(return_value=response)
            with self.assertRaisesRegex(RuntimeError, "014"):
                collector.get_document_text(self.receipt)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_invalid_receipt_rejected_before_request(self):
        collector = DARTCollector("test-secret", None)
        collector.session.get = Mock()
        with self.assertRaises(ValueError):
            collector.get_document_text("../bad")
        collector.session.get.assert_not_called()
