import unittest
from unittest.mock import Mock

import pandas as pd
import requests

from data.collectors.krx_collector import KRXCollector


def _response(payload):
    response = Mock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


class KRXOpenAPICollectorTests(unittest.TestCase):
    def test_listing_price_uses_auth_header_and_full_issue_code(self):
        session = Mock()
        session.get.return_value = _response({"OutBlock_1": [{
            "ISU_CD": "123456",
            "ISU_NM": "테스트",
            "TDD_OPNPRC": "18,000",
            "TDD_CLSPRC": "15,000",
            "TDD_HGPRC": "19,000",
            "TDD_LWPRC": "14,000",
            "ACC_TRDVOL": "100,000",
        }]})
        collector = KRXCollector(
            api_key="test-key",
            base_url="https://example.test/svc/apis",
            session=session,
            request_delay=0,
        )

        price = collector.get_listing_day_price(
            "123456", "20240510", isu_cd="KR7123456000", market="KOSDAQ"
        )

        self.assertEqual(price["open_price"], 18000.0)
        self.assertEqual(price["close_price"], 15000.0)
        self.assertEqual(price["market"], "KOSDAQ")
        session.get.assert_called_once_with(
            "https://example.test/svc/apis/sto/ksq_bydd_trd",
            params={"basDd": "20240510"},
            headers={"AUTH_KEY": "test-key", "Accept": "application/json"},
            timeout=30,
        )

    def test_calendar_filters_listing_dates_and_counts_same_day_listings(self):
        session = Mock()
        session.get.side_effect = [
            _response({"OutBlock_1": [{
                "ISU_CD": "KR7000001000", "ISU_SRT_CD": "000001", "ISU_ABBRV": "코스피테스트",
                "LIST_DD": "20240102", "SECT_TP_NM": "제조업",
            }]}),
            _response({"OutBlock_1": [{
                "ISU_CD": "KR7000002000", "ISU_SRT_CD": "000002", "ISU_ABBRV": "코스닥테스트",
                "LIST_DD": "20240102", "SECT_TP_NM": "소프트웨어",
            }]}),
        ]
        collector = KRXCollector(api_key="test-key", session=session, request_delay=0)

        calendar = collector.get_ipo_calendar("20240101", "20241231")

        self.assertEqual(len(calendar), 2)
        self.assertEqual(set(calendar["market"]), {"KOSPI", "KOSDAQ"})
        self.assertTrue((calendar["same_day_ipo_count"] == 2).all())
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(calendar["listing_date"]))

    def test_foreign_listing_uses_company_name_after_code_match_fails(self):
        session = Mock()
        session.get.return_value = _response({"OutBlock_1": [{
            "ISU_CD": "840150", "ISU_NM": "소마젠",
            "TDD_OPNPRC": "12,000", "TDD_CLSPRC": "10,500",
            "TDD_HGPRC": "14,000", "TDD_LWPRC": "10,000", "ACC_TRDVOL": "10,000",
        }]})
        collector = KRXCollector(api_key="test-key", session=session, request_delay=0)

        price = collector.get_listing_day_price(
            "950200", "20200713", isu_cd="KR8840150005", market="KOSDAQ", corp_name="소마젠(Reg.S)"
        )

        self.assertEqual(price["open_price"], 12000.0)
        self.assertEqual(price["close_price"], 10500.0)

    def test_index_collection_selects_main_index_and_skips_non_trading_days(self):
        session = Mock()
        session.get.side_effect = [
            _response({"OutBlock_1": [{"IDX_NM": "KOSPI 200", "CLSPRC_IDX": "3500"}]}),
            _response({"OutBlock_1": [{
                "BAS_DD": "20240102", "IDX_NM": "KOSPI", "CLSPRC_IDX": "2,669.81",
            }]}),
        ]
        collector = KRXCollector(api_key="test-key", session=session, request_delay=0)

        frame = collector.get_index_ohlcv("1", "20240101", "20240102")

        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.loc[0, "index_code"], "KOSPI")
        self.assertAlmostEqual(frame.loc[0, "close"], 2669.81)

    def test_request_retries_after_transient_connection_error(self):
        session = Mock()
        session.get.side_effect = [
            requests.ConnectionError("connection reset"),
            _response({"OutBlock_1": []}),
        ]
        collector = KRXCollector(api_key="test-key", session=session, request_delay=0)

        records = collector._get_daily_records("idx/kospi_dd_trd", "20200115")

        self.assertEqual(records, [])
        self.assertEqual(session.get.call_count, 2)


if __name__ == "__main__":
    unittest.main()
