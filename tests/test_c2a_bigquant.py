from __future__ import annotations

import json
import stat

import numpy as np
import pandas as pd

from robot_quant.c2a import C2AParameters
from robot_quant.c2a_bigquant import (
    build_bigquant_universe,
    configure_bigquant_api_key_from_clipboard,
    download_bigquant_minutes,
    _prior_valid_amount_mean,
    _prior_valid_high,
    save_bigquant_api_key,
)
from robot_quant.c2a_data import C2ADataStore
from robot_quant.cli import main


def test_bigquant_cli_forwards_stream_cache_flag(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        "robot_quant.cli.run_c2a_bigquant_update",
        lambda **kwargs: calls.append(kwargs) or {},
    )

    main(
        [
            "c2a-update-bigquant",
            "--start",
            "2026-07-01",
            "--end",
            "2026-08-12",
            "--stream-cache",
        ]
    )

    assert calls == [
        {
            "data_root": "data/c2a",
            "start_date": "2026-07-01",
            "end_date": "2026-08-12",
            "universe_only": False,
            "stream_cache": True,
        }
    ]


class _FakeBigQuant:
    days = pd.bdate_range("2026-01-05", periods=5)

    def query(self, sql, *, filters):
        if "cn_stock_static_data" in sql:
            return pd.DataFrame(
                [
                    {
                        "date": day,
                        "instrument": "000001.SZ",
                        "name": "测试",
                        "pre_close": 10.0,
                        "suspended": 0,
                        "st_status": int(day == self.days[-1]),
                        "in_delist": 0,
                        "public_float_share": 1_000_000_000,
                        "upper_limit": 11.0,
                        "lower_limit": 9.0,
                    }
                    for day in self.days
                ]
            )
        if "cn_stock_factors_base" in sql:
            return pd.DataFrame(
                [
                    {
                        "date": day,
                        "instrument": "000001.SZ",
                        "list_date": pd.Timestamp("2000-01-01"),
                        "trading_days": 1_000 + index,
                    }
                    for index, day in enumerate(self.days)
                ]
            )
        if "cn_stock_bar1d" in sql:
            return pd.DataFrame(
                [
                    {
                        "date": day,
                        "instrument": "000001.SZ",
                        "high": close + 0.1,
                        "close": close,
                        "volume": 20_000_000,
                        "amount": 200_000_000.0,
                        "upper_limit": 11.0,
                    }
                    for day, close in zip(
                        self.days,
                        (10.0, 11.0, 11.0, 10.0, 10.0),
                        strict=True,
                    )
                ]
            )
        if "cn_stock_bar1m_c" in sql:
            day = pd.Timestamp(filters["date"][0]).normalize()
            rows = []
            for timestamp in (
                *pd.date_range(f"{day.date()} 09:31", f"{day.date()} 11:30", freq="1min"),
                *pd.date_range(f"{day.date()} 13:01", f"{day.date()} 15:00", freq="1min"),
            ):
                rows.append(
                    {
                        "date": timestamp,
                        "instrument": "000001.SZ",
                        "open": 10.0,
                        "high": 10.1,
                        "low": 9.9,
                        "close": 10.0,
                        "volume": 1_000_000,
                        "amount": 10_000_000.0,
                    }
                )
            return pd.DataFrame(rows)
        raise AssertionError(sql)


def test_bigquant_daily_lookbacks_skip_suspended_nan_rows() -> None:
    values = pd.Series([10.0, np.nan, 11.0, 12.0, 13.0])
    previous_high = _prior_valid_high(values)
    average_amount = _prior_valid_amount_mean(values * 10_000_000)

    assert np.isnan(previous_high.iloc[1])
    assert previous_high.iloc[2] == 10.0
    assert previous_high.iloc[4] == 12.0
    assert np.isnan(average_amount.iloc[3])
    assert average_amount.iloc[4] == 110_000_000.0


class _StaticSuspensionLagBigQuant(_FakeBigQuant):
    def query(self, sql, *, filters):
        frame = super().query(sql, filters=filters)
        if "cn_stock_bar1d" in sql:
            frame.loc[frame["date"].eq(self.days[-1]), ["high", "close", "amount"]] = np.nan
            frame.loc[frame["date"].eq(self.days[-1]), "volume"] = 0
        return frame


def test_bigquant_universe_reconciles_static_suspension_lag_with_zero_daily_volume(
    tmp_path,
) -> None:
    store = C2ADataStore(tmp_path / "c2a")
    result = build_bigquant_universe(
        store,
        "2026-01-08",
        "2026-01-09",
        client=_StaticSuspensionLagBigQuant(),
    )
    last = result.loc[result["trade_date"].eq(pd.Timestamp("2026-01-09"))].iloc[0]
    assert bool(last["is_suspended"])


def test_bigquant_api_key_is_saved_owner_only_and_preserves_other_config(tmp_path) -> None:
    path = tmp_path / ".bigquant" / "config.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"telemetry": {"enabled": False}}), encoding="utf-8")
    saved = save_bigquant_api_key("access-key.secret-key", path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert saved == path
    assert payload == {
        "telemetry": {"enabled": False},
        "auth": {"ak": "access-key", "sk": "secret-key"},
    }
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_bigquant_clipboard_configuration_does_not_put_key_in_command_args(
    tmp_path, monkeypatch
) -> None:
    calls = []

    class Completed:
        stdout = "access-key.secret-key"

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return Completed()

    monkeypatch.setattr("robot_quant.c2a_bigquant.subprocess.run", fake_run)
    path = tmp_path / "config.json"
    configured = configure_bigquant_api_key_from_clipboard(path)
    assert configured == path
    assert calls == [(["pbpaste"], {"check": True, "capture_output": True, "text": True})]


def test_bigquant_universe_uses_historical_daily_values_without_lookahead(tmp_path) -> None:
    store = C2ADataStore(tmp_path / "c2a")
    result = build_bigquant_universe(
        store,
        "2026-01-08",
        "2026-01-09",
        client=_FakeBigQuant(),
    )
    first = result.loc[result["trade_date"].eq(pd.Timestamp("2026-01-08"))].iloc[0]
    assert first["prevhigh"] == 11.1
    assert first["avg3_amount"] == 200_000_000.0
    assert first["limit_streak"] == 2
    assert first["listing_trading_days"] == 1003
    assert first["float_shares"] == 1_000_000_000
    last = result.loc[result["trade_date"].eq(pd.Timestamp("2026-01-09"))].iloc[0]
    assert bool(last["is_st"])
    assert store.manifest()["source"].startswith("bigquant:")


def test_bigquant_minutes_exclude_call_auction_and_write_240_bars(tmp_path) -> None:
    client = _FakeBigQuant()
    store = C2ADataStore(tmp_path / "c2a")
    universe = build_bigquant_universe(
        store,
        "2026-01-08",
        "2026-01-09",
        client=client,
    )
    universe["avg3_amount"] = 200_000_000.0
    universe["limit_streak"] = 0
    universe["is_st"] = False
    store.write_universe(universe)
    result = download_bigquant_minutes(
        store,
        "2026-01-08",
        "2026-01-09",
        params=C2AParameters.dynamic_snapshot(),
        client=client,
    )
    bars = store.read_minutes("2026-01-08", "2026-01-09")
    assert result["downloaded_partitions"] == 2
    assert len(bars) == 480
    assert bars["timestamp"].dt.time.min().strftime("%H:%M") == "09:31"
    assert not bars["timestamp"].dt.strftime("%H:%M").eq("09:25").any()
