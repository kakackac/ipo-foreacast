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


def _html_response(content: str):
    response = Mock()
    response.content = content.encode("euc-kr")
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

    def test_official_kind_listing_events_include_source_and_conservative_classification(self):
        session = Mock()
        session.post.return_value = _html_response("""
            <table>
              <tr><th>회사명</th><th>종목코드</th><th>상장일</th><th>상장유형</th>
                  <th>증권구분</th><th>업종</th><th>국적</th><th>상장주선인/지정자문인</th>
                  <th>액면가 (원)</th><th>공모가 (원)</th><th>공모금액 (천원)</th><th>최초상장주식수 (주)</th></tr>
              <tr><td>테스트기업</td><td>123456</td><td>2026-08-24</td><td>신규상장</td>
                  <td>주권</td><td>소프트웨어 개발 및 공급업</td><td>대한민국</td><td>테스트증권(주)</td>
                  <td>500</td><td>12000</td><td>1200000</td><td>1000000</td></tr>
              <tr><td>메리츠제2호스팩</td><td>000001</td><td>2026-08-25</td><td>신규상장</td>
                  <td>주권</td><td>금융업</td><td>대한민국</td><td>테스트증권(주)</td>
                  <td>100</td><td>2000</td><td>1000000</td><td>5000000</td></tr>
            </table>
        """)
        collector = KRXCollector(session=session, request_delay=0)

        events = collector.get_official_listing_events("20260101", "20260827")

        self.assertEqual(len(events), 2)
        self.assertEqual(events.loc[0, "event_class"], "general_ipo")
        self.assertEqual(events.loc[1, "event_class"], "spac_ipo")
        self.assertEqual(events.loc[0, "offering_type"], "common_stock_ipo")
        self.assertEqual(events.loc[1, "offering_type"], "spac_ipo")
        self.assertEqual(
            events.loc[1, "retail_subscription_eligibility_status"],
            "candidate_requires_official_notice",
        )
        self.assertEqual(events.loc[0, "industry_name"], "소프트웨어 개발 및 공급업")
        self.assertEqual(events.loc[0, "source_name"], "KRX_KIND_new_listing_company")
        self.assertEqual(collector.official_listing_requests[-1]["status"], "success")

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
