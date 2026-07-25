from __future__ import annotations

from datetime import date

import pandas as pd

from robot_quant.data import CnIndexDataSource, TencentDataSource


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeSession:
    def __init__(self) -> None:
        self.params: list[str] = []

    def get(self, _endpoint, *, params, timeout) -> _FakeResponse:
        del timeout
        request = params["param"]
        self.params.append(request)
        year = request.split(",")[2][:4]
        payload = {
            "code": 0,
            "msg": "",
            "data": {"sz159530": {"day": [[f"{year}-01-02", "1.0", "1.1", "1.2", "0.9", "100"]]}},
        }
        return _FakeResponse(payload)


def test_tencent_payload_is_parsed_into_sorted_numeric_daily_prices() -> None:
    payload = {
        "code": 0,
        "msg": "",
        "data": {
            "sz159530": {
                "day": [
                    ["2026-07-16", "1.300", "1.320", "1.330", "1.290", "123456"],
                    ["2026-07-17", "1.325", "1.336", "1.340", "1.310", "234567"],
                ]
            }
        },
    }

    prices = TencentDataSource._parse_payload(payload, "sz159530")

    assert list(prices.columns) == ["open", "close", "high", "low", "volume"]
    assert prices.index.equals(pd.DatetimeIndex(["2026-07-16", "2026-07-17"], name="date"))
    assert prices.loc["2026-07-17", "close"] == 1.336


def test_tencent_payload_rejects_missing_history() -> None:
    payload = {"code": 0, "msg": "", "data": {"sz159530": {}}}

    try:
        TencentDataSource._parse_payload(payload, "sz159530")
    except ValueError as error:
        assert "行情返回为空" in str(error)
    else:
        raise AssertionError("空行情必须报错")


def test_fetch_daily_requests_each_calendar_year_to_preserve_full_history() -> None:
    source = TencentDataSource()
    fake_session = _FakeSession()
    source.session = fake_session

    prices = source.fetch_daily(
        "sz159530",
        start_date=date(2024, 1, 1),
        end_date=date(2025, 7, 17),
    )

    assert fake_session.params == [
        "sz159530,day,2024-01-01,2024-12-31,400,qfq",
        "sz159530,day,2025-01-01,2025-07-17,400,qfq",
    ]
    assert prices.index.tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2025-01-02")]


def test_cnindex_payload_is_parsed_into_sorted_official_index_prices() -> None:
    payload = {
        "code": 200,
        "total": 2,
        "data": {
            "data": [
                [
                    "2026-07-24",
                    2088.2823,
                    2147.4754,
                    2119.1453,
                    2088.2823,
                    2088.2823,
                    -81.379,
                    "-3.75%",
                    235.06,
                    795.72,
                    None,
                ],
                [
                    "2026-07-23",
                    2169.6613,
                    2184.0226,
                    2158.6639,
                    2142.5431,
                    2169.6613,
                    -10.8217,
                    "-0.50%",
                    258.72,
                    839.04,
                    None,
                ],
            ]
        },
    }

    prices = CnIndexDataSource._parse_payload(payload, "980022")

    assert prices.index.tolist() == [pd.Timestamp("2026-07-23"), pd.Timestamp("2026-07-24")]
    assert prices.loc["2026-07-24", "open"] == 2119.1453
    assert prices.loc["2026-07-24", "close"] == 2088.2823
    assert prices.loc["2026-07-24", "volume"] == 795.72
