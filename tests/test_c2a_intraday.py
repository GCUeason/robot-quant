import json
from datetime import date, time

import pandas as pd
import pytest

from robot_quant.c2a_intraday import (
    audit_completed_window,
    fetch_candidate_ohlc,
    parse_tencent_cumulative_minutes,
)


def _payload(rows: list[str]) -> bytes:
    return json.dumps(
        {
            "code": 0,
            "msg": "",
            "data": {"sh600378": {"data": {"data": rows}}},
        }
    ).encode()


def test_tencent_intraday_adapter_excludes_auction_and_converts_lots_to_shares() -> None:
    bars = parse_tencent_cumulative_minutes(
        _payload(
            [
                "0930 49.00 100 490000.00",
                "0931 49.20 160 785200.00",
                "0932 49.10 230 1128900.00",
            ]
        ),
        "600378",
        date(2026, 8, 14),
        cutoff=time(9, 32),
    )

    assert bars["timestamp"].dt.strftime("%H:%M").tolist() == ["09:31", "09:32"]
    assert bars["volume"].tolist() == [6_000.0, 7_000.0]
    assert bars["amount"].tolist() == [295_200.0, 343_700.0]
    assert bars["close"].tolist() == [49.2, 49.1]


def test_intraday_audit_accepts_only_the_completed_0931_to_1000_grid() -> None:
    timestamps = pd.date_range("2026-08-14 09:31", "2026-08-14 10:00", freq="1min")
    bars = pd.DataFrame(
        {
            "timestamp": list(timestamps) * 2,
            "ticker": ["600378"] * 30 + ["002131"] * 30,
            "close": [10.0] * 60,
            "volume": [100.0] * 60,
            "amount": [1_000.0] * 60,
        }
    )

    result = audit_completed_window(bars, {"600378", "002131"}, time(10, 0))

    assert result.status == "PROXY"
    assert result.complete_minutes == 30
    assert result.ticker_count == 2


def test_intraday_audit_rejects_one_missing_completed_minute() -> None:
    timestamps = pd.date_range("2026-08-14 09:31", "2026-08-14 10:00", freq="1min")
    bars = pd.DataFrame(
        {
            "timestamp": timestamps[:-1],
            "ticker": "600378",
            "close": 10.0,
            "volume": 100.0,
            "amount": 1_000.0,
        }
    )

    with pytest.raises(ValueError, match="完整分钟"):
        audit_completed_window(bars, {"600378"}, time(10, 0))


def test_candidate_ohlc_retries_only_transient_failures(monkeypatch) -> None:
    attempts: dict[str, int] = {}

    def fake_fetch(ticker: str, trade_day: date) -> pd.DataFrame:
        attempts[ticker] = attempts.get(ticker, 0) + 1
        if ticker == "000002" and attempts[ticker] == 1:
            raise RuntimeError("RemoteDisconnected")
        return pd.DataFrame(
            {
                "timestamp": [pd.Timestamp.combine(trade_day, time(9, 31))],
                "ticker": [ticker],
                "open": [10.0],
                "close": [10.1],
                "high": [10.2],
                "low": [9.9],
                "volume": [1_000.0],
                "amount": [10_000.0],
            }
        )

    monkeypatch.setattr("robot_quant.c2a_intraday._fetch_one_eastmoney", fake_fetch)

    result = fetch_candidate_ohlc(["000001", "000002"], date(2026, 8, 17))

    assert set(result["ticker"]) == {"000001", "000002"}
    assert attempts == {"000001": 1, "000002": 2}
