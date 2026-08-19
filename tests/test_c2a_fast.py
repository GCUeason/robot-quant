import json
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from robot_quant.c2a_fast import (
    FastPack,
    FastScanDataError,
    bootstrap_fast_pack,
    latest_complete_cutoff,
    load_fast_pack,
    parse_tencent_day_history,
    parse_tencent_mkline,
    render_scan_result,
    run_fast_scan,
    save_fast_pack,
)


def _day_payload(rows: list[str]) -> bytes:
    return json.dumps(
        {
            "code": 0,
            "data": {
                "sh600216": {
                    "data": [
                        {
                            "date": "20260817",
                            "prec": "13.33",
                            "data": rows,
                        }
                    ]
                }
            },
        }
    ).encode()


def test_parse_tencent_day_history_excludes_auction_from_scan_baseline() -> None:
    scan_rows = ["0930 13.30 100 133000.00"]
    cumulative_lots = 100
    cumulative_amount = 133_000.0
    for minute in range(31, 61):
        hour = 9 if minute < 60 else 10
        minute_of_hour = minute if minute < 60 else 0
        cumulative_lots += 10
        cumulative_amount += 13_300.0
        scan_rows.append(
            f"{hour:02d}{minute_of_hour:02d} 13.30 {cumulative_lots} {cumulative_amount:.2f}"
        )
    scan_rows.extend(["1130 13.40 500 670000.00", "1500 13.50 900 1215000.00"])

    result = parse_tencent_day_history(_day_payload(scan_rows), "600216")

    day = result[date(2026, 8, 17)]
    assert len(day.cumulative_amount) == 30
    assert day.cumulative_volume[0] == 1_000.0
    assert day.cumulative_amount[0] == 13_300.0
    assert day.cumulative_volume[-1] == 30_000.0
    assert np.isclose(day.high, 13.5)
    assert day.close == 13.5


def test_latest_complete_cutoff_uses_previous_minute_and_caps_at_1000() -> None:
    timezone = ZoneInfo("Asia/Shanghai")

    assert latest_complete_cutoff(datetime(2026, 8, 18, 9, 35, 20, tzinfo=timezone)) == time(9, 34)
    assert latest_complete_cutoff(datetime(2026, 8, 18, 10, 12, tzinfo=timezone)) == time(10, 0)


def test_latest_complete_cutoff_rejects_too_early_run() -> None:
    with pytest.raises(FastScanDataError, match="09:32前"):
        latest_complete_cutoff(datetime(2026, 8, 18, 9, 31, 30, tzinfo=ZoneInfo("Asia/Shanghai")))


def test_run_fast_scan_builds_non_empty_watchlist_from_cutoff_close(tmp_path, monkeypatch) -> None:
    trade_day = date(2026, 8, 19)
    universe = pd.DataFrame(
        [
            {
                "trade_date": pd.Timestamp(trade_day),
                "ticker": "600378",
                "name": "昊华科技",
                "pool": "MAIN",
                "list_date": "2001-01-11",
                "listing_trading_days": 5_000,
                "prevclose": 10.0,
                "prevhigh": 10.2,
                "avg3_amount": 200_000_000.0,
                "float_shares": 1_000_000_000.0,
                "float_mcap": 10_000_000_000.0,
                "is_st": False,
                "is_suspended": False,
                "upper_limit": 11.0,
                "lower_limit": 9.0,
                "limit_streak": 0,
            }
        ]
    )
    pack = FastPack(
        root=tmp_path,
        tickers=["600378"],
        amount_history=np.ones((1, 1, 20)),
        volume_history=np.ones((1, 1, 20)),
        pointers=np.zeros(1, dtype=np.int64),
        counts=np.full(1, 20, dtype=np.int64),
        last_processed_date=pd.Timestamp("2026-08-18"),
        universe=universe,
        manifest={},
    )
    bars = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-08-19 09:31")],
            "ticker": ["600378"],
            "close": [10.31],
            "volume": [1_000.0],
            "amount": [10_310.0],
        }
    )
    monkeypatch.setattr("robot_quant.c2a_fast.load_fast_pack", lambda *_: pack)
    monkeypatch.setattr("robot_quant.c2a_fast.previous_market_day", lambda *_: date(2026, 8, 18))
    monkeypatch.setattr("robot_quant.c2a_fast.fetch_quotes_fast", lambda *_: pd.DataFrame())
    monkeypatch.setattr("robot_quant.c2a_fast.build_fast_universe", lambda *_: universe)
    monkeypatch.setattr("robot_quant.c2a_fast.fetch_minutes_fast", lambda *_: (bars, {}))
    monkeypatch.setattr("robot_quant.c2a_fast.fetch_candidate_ohlc_fast", lambda *_: bars)
    monkeypatch.setattr(
        "robot_quant.c2a_fast.simulate_entry_day", lambda *args, **kwargs: ([], 10_000.0, [])
    )

    result = run_fast_scan(
        tmp_path,
        trade_day=trade_day,
        cutoff=time(9, 31),
        now=datetime(2026, 8, 19, 9, 32, tzinfo=ZoneInfo("Asia/Shanghai")),
    )

    assert result["candidate_count"] == 1
    assert result["watchlist"][0]["ticker"] == "600378"
    assert result["watchlist"][0]["scan_price"] == pytest.approx(10.31)


def test_parse_tencent_mkline_keeps_only_requested_day_and_cutoff() -> None:
    payload = json.dumps(
        {
            "code": 0,
            "data": {
                "sh600216": {
                    "m1": [
                        ["202608170959", "13.1", "13.2", "13.3", "13.0", "10", {}],
                        ["202608180931", "13.3", "13.4", "13.5", "13.2", "20", {}],
                        ["202608180932", "13.4", "13.6", "13.7", "13.3", "30", {}],
                    ]
                }
            },
        }
    ).encode()

    result = parse_tencent_mkline(payload, "600216", date(2026, 8, 18), time(9, 31))

    assert result["timestamp"].dt.strftime("%H:%M").tolist() == ["09:31"]
    assert result.iloc[0]["high"] == 13.5


def test_render_scan_result_prints_code_name_and_paper_boundary() -> None:
    result = {
        "as_of": "2026-08-18",
        "cutoff": "09:34",
        "status": "SIMULATED_ENTRY",
        "data_status": "PROXY",
        "elapsed_seconds": 3.2,
        "entries": [
            {
                "ticker": "600216",
                "name": "浙江医药",
                "simulated_fill_price": 13.71,
                "shares": 300,
            }
        ],
        "watchlist": [],
        "baseline_as_of": "2026-08-17",
        "baseline_coverage": 1.0,
        "promotion_gate": "FAIL",
    }

    text = render_scan_result(result)

    assert "ENTRY 600216 浙江医药" in text
    assert "仅模拟，非交易指令" in text


def test_fast_pack_manifest_rejects_tampered_state(tmp_path) -> None:
    universe = pd.DataFrame(
        [
            {
                "trade_date": "2026-08-18",
                "ticker": "600216",
                "name": "浙江医药",
                "pool": "MAIN",
                "list_date": "1999-10-21",
                "listing_trading_days": 1_000,
                "prevclose": 13.5,
                "prevhigh": 13.6,
                "avg3_amount": 200_000_000.0,
                "float_shares": 1_000_000_000.0,
                "float_mcap": 13_500_000_000.0,
                "is_st": False,
                "is_suspended": False,
                "upper_limit": 14.85,
                "lower_limit": 12.15,
                "limit_streak": 0,
            }
        ]
    )
    pack = FastPack(
        root=tmp_path,
        tickers=["600216"],
        amount_history=np.ones((1, 30, 20)),
        volume_history=np.ones((1, 30, 20)),
        pointers=np.zeros(1, dtype=np.int64),
        counts=np.full(1, 20, dtype=np.int64),
        last_processed_date=pd.Timestamp("2026-08-18"),
        universe=universe,
        manifest={},
    )
    save_fast_pack(pack)
    state_path = tmp_path / "rolling_state.npz"
    with state_path.open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(FastScanDataError, match="文件损坏"):
        load_fast_pack(tmp_path)


def test_bootstrap_fast_pack_uses_remote_adapter(tmp_path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        "robot_quant.c2a_fast.export_remote_fast_pack",
        lambda *args, **kwargs: calls.append(("export", kwargs)),
    )
    monkeypatch.setattr(
        "robot_quant.c2a_fast.fetch_remote_fast_pack",
        lambda *args, **kwargs: calls.append(("fetch", kwargs)),
    )

    bootstrap_fast_pack(tmp_path, host="quant-host", remote_root="/srv/quant")

    assert calls == [
        ("export", {"host": "quant-host", "remote_root": "/srv/quant"}),
        ("fetch", {"host": "quant-host", "remote_root": "/srv/quant"}),
    ]
