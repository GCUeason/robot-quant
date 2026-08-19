from __future__ import annotations

from dataclasses import replace
from datetime import time

import numpy as np
import pandas as pd

from robot_quant.c2a import (
    C2AParameters,
    Position,
    _apply_ticker_cooldowns,
    _exit_position,
    _position_size,
    backtest_c2a,
    backtest_c2a_schedule,
    build_signal_features,
    eligible_universe,
    parameter_grid,
    prepare_c2a_data,
    prepare_c2a_market_data,
    simulate_entry_day,
)


def _universe(dates, tickers=("000001", "000002")) -> pd.DataFrame:
    rows = []
    for trade_date in dates:
        for ticker in tickers:
            rows.append(
                {
                    "trade_date": trade_date,
                    "ticker": ticker,
                    "name": ticker,
                    "pool": "MAIN",
                    "list_date": "2000-01-01",
                    "listing_trading_days": 1_000,
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
            )
    return pd.DataFrame(rows)


def _bar(timestamp, ticker, open_, high, low, close, amount=20_000_000.0):
    return {
        "timestamp": timestamp,
        "ticker": ticker,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": amount / close,
        "amount": amount,
    }


def test_signal_features_do_not_change_when_future_day_changes() -> None:
    dates = pd.bdate_range("2026-01-05", periods=4)
    rows = []
    for day_index, trade_date in enumerate(dates):
        for minute_index, minute in enumerate(("09:30", "09:31")):
            timestamp = pd.Timestamp(f"{trade_date.date()} {minute}")
            rows.append(_bar(timestamp, "000001", 10, 10.6, 10, 10.5, 10_000_000 * (day_index + 1)))
            rows.append(
                _bar(timestamp, "000002", 10, 10.2, 9.9, 10.1, 5_000_000 * (minute_index + 1))
            )
    minutes = pd.DataFrame(rows)
    params = replace(C2AParameters(), baseline_days=2, scan_end=time(9, 31))
    original = build_signal_features(minutes, _universe(dates), params)
    future_changed = minutes.copy()
    future_changed.loc[future_changed["timestamp"].dt.normalize().eq(dates[-1]), "amount"] *= 1_000
    changed = build_signal_features(future_changed, _universe(dates), params)
    cutoff = dates[-2]
    columns = ["timestamp", "ticker", "amount_burst", "turnover_metric", "c6", "signal_pass"]
    pd.testing.assert_frame_equal(
        original.loc[original["trade_date"].le(cutoff), columns].reset_index(drop=True),
        changed.loc[changed["trade_date"].le(cutoff), columns].reset_index(drop=True),
    )


def test_delisting_arrangement_name_is_excluded_from_universe() -> None:
    universe = _universe([pd.Timestamp("2026-03-02")], ("000001",))
    universe.loc[:, "name"] = "测试退"
    assert not bool(eligible_universe(universe, C2AParameters()).iloc[0])


def test_entry_trigger_does_not_use_current_minute_high() -> None:
    day = pd.Timestamp("2026-03-02")
    minutes = pd.DataFrame(
        [
            _bar(f"{day.date()} 09:30", "000001", 10.0, 10.5, 10.0, 10.4),
            _bar(f"{day.date()} 09:31", "000001", 10.4, 10.5, 10.3, 10.4),
            # 当前分钟高点20不能先进入H_prev；否则会错误触发19.4元的回撤价。
            _bar(f"{day.date()} 09:32", "000001", 12.0, 20.0, 12.0, 12.0),
        ]
    )
    feature = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp(f"{day.date()} 09:31"),
                "ticker": "000001",
                "signal_pass": True,
                "c6": 5.0,
                "amount_burst": 3.0,
                "turnover_metric": 0.02,
                "gain": 0.04,
            }
        ]
    )
    params = replace(C2AParameters(), scan_end=time(9, 31), slippage_bps=0)
    positions, _, _ = simulate_entry_day(
        minutes,
        feature,
        _universe([day], ("000001",)),
        params,
        cash=100_000,
        budget=100_000,
    )
    assert positions == []


def test_full_backtest_enters_on_pullback_and_exits_next_open_with_costs() -> None:
    dates = pd.bdate_range("2026-01-05", periods=5)
    rows = []
    for day_index, trade_date in enumerate(dates):
        for minute in ("09:31", "09:32", "09:33"):
            timestamp = pd.Timestamp(f"{trade_date.date()} {minute}")
            if day_index == 2 and minute == "09:31":
                rows.append(_bar(timestamp, "000001", 10.4, 10.6, 10.4, 10.5, 80_000_000))
            elif day_index == 2 and minute == "09:32":
                rows.append(_bar(timestamp, "000001", 10.4, 10.5, 10.2, 10.3, 80_000_000))
            elif day_index == 3 and minute == "09:31":
                rows.append(_bar(timestamp, "000001", 10.8, 10.8, 10.0, 10.0, 20_000_000))
            else:
                rows.append(_bar(timestamp, "000001", 10.0, 10.1, 9.9, 10.0, 20_000_000))
            rows.append(_bar(timestamp, "000002", 10.0, 10.1, 9.9, 10.0, 10_000_000))
    params = replace(
        C2AParameters(),
        baseline_days=2,
        scan_end=time(9, 31),
        slippage_bps=0,
    )
    trades, equity, summary = backtest_c2a(
        pd.DataFrame(rows),
        _universe(dates),
        params,
        trade_start=dates[2],
        trade_end=dates[-1],
    )
    # v1 保留60%首仓与同股40%深回撤竞争，因此本例形成两条独立成交腿。
    assert len(trades) == 2
    assert trades["entry_date"].eq(dates[2]).all()
    assert trades["exit_date"].eq(dates[3]).all()
    assert trades["exit_cost"].gt(5.0).all()
    assert trades["net_return"].lt(trades["gross_return"]).all()
    assert summary["trade_count"] == 2
    assert not equity.empty

    features = build_signal_features(pd.DataFrame(rows), _universe(dates), params)
    market = prepare_c2a_market_data(pd.DataFrame(rows), _universe(dates))
    prepared = prepare_c2a_data(market, features)
    scheduled_trades, scheduled_equity, _scheduled_summary = backtest_c2a_schedule(
        {dates[2]: params, dates[3]: params, dates[4]: params},
        {False: prepared},
    )
    pd.testing.assert_series_equal(
        trades["profit"].reset_index(drop=True),
        scheduled_trades["profit"].reset_index(drop=True),
    )
    assert scheduled_equity.iloc[-1]["portfolio_value"] == equity.iloc[-1]["portfolio_value"]


def test_one_price_limit_down_keeps_position_until_tradeable_open() -> None:
    position = Position(
        ticker="000001",
        entry_date=pd.Timestamp("2026-03-02"),
        entry_time=pd.Timestamp("2026-03-02 10:00"),
        entry_price=10.0,
        shares=100,
        entry_value=1_000.0,
        entry_cost=5.01,
        signal_id="2026-03-02-000001-1",
        signal_time=pd.Timestamp("2026-03-02 09:40"),
        signal_price=10.5,
        signal_c6=5.0,
        signal_amount_burst=3.0,
        signal_turnover_metric=0.02,
        signal_gain=0.05,
        h_prev=10.5,
        trigger_price=10.0,
        position_weight=0.6,
        leg="FIRST",
        pool="MAIN",
        data_status="PROXY",
    )
    universe = _universe([pd.Timestamp("2026-03-03")], ("000001",))
    locked_bar = pd.DataFrame([_bar("2026-03-03 09:30", "000001", 9, 9, 9, 9)])
    trade, proceeds = _exit_position(position, locked_bar, universe, C2AParameters())
    assert trade is None
    assert proceeds == 0
    assert position.locked_days == 1
    tradable_bar = pd.DataFrame([_bar("2026-03-04 09:30", "000001", 9.2, 9.3, 9.1, 9.2)])
    trade, proceeds = _exit_position(position, tradable_bar, universe, C2AParameters())
    assert trade is not None
    assert trade["locked_limit_down_days"] == 1
    assert proceeds > 0


def test_same_ticker_two_legs_use_combined_profit_for_cooldown() -> None:
    closed = [
        {"ticker": "000001", "profit": -100.0, "cooldown_triggered": True},
        {"ticker": "000001", "profit": 150.0, "cooldown_triggered": False},
    ]
    cooldown_until: dict[str, int] = {}
    _apply_ticker_cooldowns(closed, 10, 20, cooldown_until)
    assert not any(trade["cooldown_triggered"] for trade in closed)
    assert all(trade["ticker_combined_profit"] == 50.0 for trade in closed)
    assert cooldown_until == {}


def test_limit_down_position_exits_at_first_tradeable_intraday_minute() -> None:
    position = Position(
        ticker="000001",
        entry_date=pd.Timestamp("2026-03-02"),
        entry_time=pd.Timestamp("2026-03-02 10:00"),
        entry_price=10.0,
        shares=100,
        entry_value=1_000.0,
        entry_cost=5.01,
        signal_id="2026-03-02-000001-1",
        signal_time=pd.Timestamp("2026-03-02 09:40"),
        signal_price=10.5,
        signal_c6=5.0,
        signal_amount_burst=3.0,
        signal_turnover_metric=0.02,
        signal_gain=0.05,
        h_prev=10.5,
        trigger_price=10.0,
        position_weight=0.6,
        leg="FIRST",
        pool="MAIN",
        data_status="PROXY",
    )
    bars = pd.DataFrame(
        [
            _bar("2026-03-03 09:30", "000001", 9, 9, 9, 9),
            _bar("2026-03-03 09:31", "000001", 9, 9.1, 9, 9.05),
        ]
    )
    universe = _universe([pd.Timestamp("2026-03-03")], ("000001",))
    trade, _ = _exit_position(position, bars, universe, C2AParameters())
    assert trade is not None
    assert trade["next_day_open"] == 9.0
    assert trade["exit_time"] == pd.Timestamp("2026-03-03 09:31")


def test_v12_same_minute_competition_marks_loser_missed_instead_of_chasing() -> None:
    day = pd.Timestamp("2026-03-02")
    minutes = pd.DataFrame(
        [_bar(f"{day.date()} 09:30", ticker, 10, 10.5, 10, 10.4) for ticker in ("000001", "000002")]
        + [
            _bar(f"{day.date()} 09:31", ticker, 10.4, 10.5, 10.3, 10.4)
            for ticker in ("000001", "000002")
        ]
        + [
            _bar(f"{day.date()} 09:32", ticker, 10.2, 10.3, 10.18, 10.19)
            for ticker in ("000001", "000002")
        ]
    )
    features = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp(f"{day.date()} 09:31"),
                "ticker": ticker,
                "signal_pass": True,
                "c6": c6,
                "amount_burst": 3.0,
                "turnover_metric": 2.0,
                "gain": 0.04,
            }
            for ticker, c6 in (("000001", 5.0), ("000002", 10.0))
        ]
    )
    params = replace(
        C2AParameters.dynamic_snapshot(),
        confirmation_minutes=1,
        scan_end=time(9, 31),
        slippage_bps=0,
    )
    positions, _, events = simulate_entry_day(
        minutes,
        features,
        _universe([day]),
        params,
        cash=100_000,
        budget=10_000,
    )
    assert [position.ticker for position in positions] == ["000001"]
    assert positions[0].position_weight < params.first_weight
    assert any(event["event"] == "MISSED_ENTRY" and event["ticker"] == "000002" for event in events)


def test_competition_skips_unfillable_leader_before_ranking() -> None:
    day = pd.Timestamp("2026-03-02")
    minutes = pd.DataFrame(
        [_bar(f"{day.date()} 09:30", ticker, 10, 10.5, 10, 10.4) for ticker in ("000001", "000002")]
        + [
            _bar(f"{day.date()} 09:31", ticker, 10.4, 10.5, 10.3, 10.4)
            for ticker in ("000001", "000002")
        ]
        + [
            _bar(
                f"{day.date()} 09:32",
                ticker,
                10.2,
                10.3,
                10.18,
                10.19,
                1_000.0 if ticker == "000001" else 20_000_000.0,
            )
            for ticker in ("000001", "000002")
        ]
    )
    features = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp(f"{day.date()} 09:31"),
                "ticker": ticker,
                "signal_pass": True,
                "c6": c6,
                "amount_burst": 3.0,
                "turnover_metric": 2.0,
                "gain": 0.04,
            }
            for ticker, c6 in (("000001", 5.0), ("000002", 10.0))
        ]
    )
    params = replace(
        C2AParameters.dynamic_snapshot(),
        confirmation_minutes=1,
        scan_end=time(9, 31),
        slippage_bps=0,
    )
    positions, _, events = simulate_entry_day(
        minutes,
        features,
        _universe([day]),
        params,
        cash=100_000,
        budget=10_000,
    )
    assert [position.ticker for position in positions] == ["000002"]
    assert any(
        event["event"] == "UNFILLED_CAPACITY_OR_LOT" and event["ticker"] == "000001"
        for event in events
    )


def test_parameter_grid_is_preregistered_and_contains_baseline() -> None:
    grid = parameter_grid()
    assert len(grid) == 720
    assert any(
        item.scan_end == time(10, 0)
        and item.c6_threshold == 30
        and np.isclose(item.main_first_pullback, 0.03)
        and np.isclose(item.growth_first_pullback, 0.045)
        and np.isclose(item.main_second_increment, 0.01)
        and np.isclose(item.main_alt_increment, 0.01)
        for item in grid
    )


def test_optimized_challenger_matches_selected_short_window_parameters() -> None:
    params = C2AParameters.optimized_challenger()

    assert params.variant == "v1.2"
    assert params.scan_end == time(10, 0)
    assert params.c6_threshold == 40.0
    assert np.isclose(params.main_first_pullback, 0.025)
    assert np.isclose(params.growth_first_pullback, 0.035)
    assert np.isclose(params.main_second_increment, 0.005)
    assert np.isclose(params.growth_second_increment, 0.005)
    assert params.exclude_yesterday_limit_up is True


def test_star_market_buy_uses_200_share_minimum_then_one_share_steps() -> None:
    shares, _cost = _position_size(
        "688001",
        fill_price=10.0,
        desired_value=2_055.0,
        minute_amount=10_000_000,
        cash=10_000,
        params=C2AParameters(),
    )
    assert shares == 204


def test_position_size_keeps_fees_inside_single_stock_total_cost_cap() -> None:
    params = replace(C2AParameters.dynamic_snapshot(), slippage_bps=0)
    shares, entry_cost = _position_size(
        "000001",
        fill_price=10.0,
        desired_value=6_000.0,
        minute_amount=10_000_000.0,
        cash=100_000.0,
        params=params,
    )
    assert shares == 400
    assert shares * 10.0 + entry_cost <= 5_000.0
