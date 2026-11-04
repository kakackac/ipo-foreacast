import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import requests

from data.pipelines.disclosure_connections import public_text, viewer_identity, official_family_document


def index(receipt='20150602000312', date='2015.06.02', corp='00659161', withdrawn=False):
    rows = [('20150430001812', '2015.04.30 증권신고서(지분증권)'),
            ('20150602000312', '2015.06.02 [발행조건확정]증권신고서(지분증권)')]
    if withdrawn: rows.append(('20150603000001', '2015.06.03 철회신고서'))
    options = ''.join(f'<option value="rcpNo={r}" {"selected" if r==receipt else ""}>{label}</option>' for r,label in rows)
    return (f'<title>변경상호/증권신고서(지분증권)/{date}</title>'
            + f"openCorpInfoNew('{corp}');<select id='family'>{options}</select>"
            + "var node1 = {};" + ''.join(f"node1['{k}'] = \"{v}\";" for k,v in
                dict(text='증권발행조건확정',rcpNo=receipt,dcmNo='1',eleId='1',offset='1',length='20',dtd='dart3.xsd').items()))


class ConnectionTests(unittest.TestCase):
    def test_cached_only_never_calls_transport(self):
        dart = Mock()
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, 'not_cached'):
                public_text(dart, 'https://dart.fss.or.kr/a', root, cached_only=True)
        dart.session.get.assert_not_called()

    def test_effective_notice_does_not_replace_the_final_terms(self):
        dart = Mock()
        initial = index(receipt='20150430001812', date='2015.04.30').replace('</select>',
            '<option value="rcpNo=20150603000001">2015.06.03 효력발생안내</option></select>')
        dart.session.get.side_effect = [Mock(text=initial), Mock(text=index()), Mock(text='<table>값</table>')]
        with tempfile.TemporaryDirectory() as root:
            result, _, _ = official_family_document(dart, root, '20150430001812', '00659161', '2015-06-17')
        self.assertEqual(result['rcept_no'], '20150602000312')

    def test_rename_is_bound_to_corp_code_not_name(self):
        result = viewer_identity(index(), '20150602000312', '00659161', '2015-06-02')
        self.assertEqual(result['publication_date'], '2015-06-02')
        with self.assertRaises(ValueError): viewer_identity(index(), '20150602000312', '99999999')
        with self.assertRaises(ValueError): viewer_identity(index(), '20150602000312', '00659161', '2015-06-01')

    def test_publication_is_not_inferred_from_receipt_prefix(self):
        html = index(date='2015.06.03').replace('2015.06.02 [', '2015.06.03 [')
        self.assertEqual(viewer_identity(html, '20150602000312', '00659161')['publication_date'], '2015-06-03')

    def test_wrong_selected_receipt_or_missing_reverse_family_is_rejected(self):
        with self.assertRaises(ValueError): viewer_identity(index(), '20150430001812', '00659161')

    def test_download_cache_tamper_and_partial_failure(self):
        dart = Mock()
        response = Mock(text='<table>valid</table>')
        dart.session.get.side_effect = [requests.Timeout('secret must not be logged')] * 3 + [response]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(requests.Timeout): public_text(dart, 'https://dart.fss.or.kr/a', directory)
            self.assertEqual(list(Path(directory).glob('*')), [])
            first = public_text(dart, 'https://dart.fss.or.kr/a', directory)
            self.assertEqual(public_text(dart, 'https://dart.fss.or.kr/a', directory), first)
            self.assertEqual(dart.session.get.call_count, 4)
            path = next(Path(directory).glob('*.json'))
            data = json.loads(path.read_text()); data['text'] = 'tampered'; path.write_text(json.dumps(data))
            with self.assertRaises(ValueError): public_text(dart, 'https://dart.fss.or.kr/a', directory)

    def test_failed_identity_validation_is_not_cached(self):
        dart = Mock(); dart.session.get.return_value = Mock(text=index(corp='99999999'))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                public_text(dart, 'https://dart.fss.or.kr/a', directory,
                            lambda text: viewer_identity(text, '20150602000312', '00659161'))
            self.assertFalse(list(Path(directory).glob('*.json')))

    def test_withdrawal_blocks_older_terms_and_future_withdrawal_does_not(self):
        dart = Mock()
        initial = index(receipt='20150430001812', date='2015.04.30', withdrawn=True)
        final = index(withdrawn=True)
        dart.session.get.side_effect = [Mock(text=initial), Mock(text=final), Mock(text='<table><tr><td>확정공모가 16,300원</td></tr></table>')]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'withdrawn'):
                official_family_document(dart, directory, '20150430001812', '00659161', '2015-06-17')
        dart.session.get.side_effect = [Mock(text=initial), Mock(text=final), Mock(text='<table>값</table>')]
        with tempfile.TemporaryDirectory() as directory:
            result, _, _ = official_family_document(dart, directory, '20150430001812', '00659161', '2015-06-04')
            self.assertEqual(result['rcept_no'], '20150602000312')


if __name__ == '__main__': unittest.main()
