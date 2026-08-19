from __future__ import annotations

import json

import pandas as pd

from robot_quant.c2a import C2AParameters, backtest_c2a, build_signal_features
from robot_quant.c2a_data import C2ADataStore
from robot_quant.c2a_runner import run_c2a_backtest


def test_strict_store_runs_end_to_end_and_writes_result_report(tmp_path) -> None:
    dates = pd.bdate_range("2025-12-01", periods=22)
    universe_rows = []
    minute_rows = []
    for day_index, day in enumerate(dates):
        for ticker in ("000001", "000002"):
            universe_rows.append(
                {
                    "trade_date": day,
                    "ticker": ticker,
                    "name": ticker,
                    "pool": "MAIN",
                    "list_date": "2000-01-01",
                    "listing_trading_days": 1_000 + day_index,
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
        timestamps = (
            *pd.date_range(f"{day.date()} 09:31", f"{day.date()} 11:30", freq="1min"),
            *pd.date_range(f"{day.date()} 13:01", f"{day.date()} 15:00", freq="1min"),
        )
        for timestamp in timestamps:
            for ticker in ("000001", "000002"):
                open_price = high = low = close = 10.0
                amount = 10_000_000.0 if ticker == "000001" else 5_000_000.0
                if day_index == 20 and ticker == "000001":
                    if timestamp.time() <= pd.Timestamp("09:32").time():
                        open_price, high, low, close, amount = 10.4, 10.5, 10.4, 10.5, 80_000_000
                    elif timestamp.time() == pd.Timestamp("09:33").time():
                        open_price, high, low, close, amount = 10.2, 10.3, 10.18, 10.19, 80_000_000
                if (
                    day_index == 21
                    and ticker == "000001"
                    and timestamp.time() == pd.Timestamp("09:31").time()
                ):
                    open_price, high, low, close = 10.8, 10.8, 10.7, 10.75
                minute_rows.append(
                    {
                        "timestamp": timestamp,
                        "ticker": ticker,
                        "open": open_price,
                        "high": high,
                        "low": low,
                        "close": close,
                        "volume": amount / close,
                        "amount": amount,
                    }
                )

    store = C2ADataStore(tmp_path / "data" / "c2a")
    store.initialize("licensed_fixture", metadata_verified=True, full_market=True)
    universe = pd.DataFrame(universe_rows)
    minutes = pd.DataFrame(minute_rows)
    store.write_universe(universe)
    store.write_minutes(minutes)
    params = C2AParameters.dynamic_snapshot()
    direct_features = build_signal_features(minutes, universe, params)
    _direct_trades, _direct_equity, direct_summary = backtest_c2a(
        minutes,
        universe,
        params,
        trade_start=dates[20],
        trade_end=dates[21],
        data_status="STRICT",
        signal_features=direct_features,
    )
    state = run_c2a_backtest(
        data_root=store.root,
        output_root=tmp_path,
        start_date=dates[20],
        end_date=dates[21],
        variant="v1.2",
        optimize=False,
    )
    assert state["audit"]["status"] == "STRICT"
    assert state["baseline"]["trade_count"] == 1
    assert state["baseline"]["total_return"] == direct_summary["total_return"]
    assert state["baseline"]["execution_permission"] == "PAPER_ONLY"
    assert state["daily_signal"]["status"] == "NO_ENTRY"
    assert state["daily_signal"]["real_trade_authorized"] is False
    assert (tmp_path / "data" / "c2a_results" / "latest_signal.json").exists()
    assert state["research_cache"]["processed_days"] == 22
    assert (tmp_path / "reports" / "c2a_2026_report.md").exists()
    assert (tmp_path / "data" / "c2a_results" / "baseline_trades.csv").exists()
    written_audit = json.loads(
        (tmp_path / "data" / "c2a_results" / "data_audit.json").read_text(encoding="utf-8")
    )
    assert written_audit["status"] == "STRICT"
    assert written_audit["end_date"] == dates[21].date().isoformat()
    assert written_audit["reasons"] == []
    events = pd.read_csv(tmp_path / "data" / "c2a_results" / "baseline_events.csv")
    assert {"SIGNAL", "ENTRY"}.issubset(set(events["event"]))

    repeated = run_c2a_backtest(
        data_root=store.root,
        output_root=tmp_path,
        start_date=dates[20],
        end_date=dates[21],
        variant="v1.2",
        optimize=False,
    )
    assert repeated["research_cache"]["processed_days"] == 0
    assert repeated["baseline"]["trade_count"] == 1

    prior_state = dict(repeated)
    prior_state["walk_forward"] = {
        "status": "COMPLETED",
        "optimization_as_of": dates[21].date().isoformat(),
        "promotion_gate": "FAIL",
        "candidate_count": 720,
    }
    results_dir = tmp_path / "data" / "c2a_results"
    prior_results = tmp_path / ".private-prior-results"
    prior_results.mkdir()
    (prior_results / "latest_state.json").write_text(json.dumps(prior_state), encoding="utf-8")
    for name in (
        "walk_forward_selections.csv",
        "walk_forward_oos_trades.csv",
        "latest_training_grid.csv",
    ):
        (prior_results / name).write_bytes((results_dir / name).read_bytes())
    for name in (
        "latest_state.json",
        "walk_forward_selections.csv",
        "walk_forward_oos_trades.csv",
        "latest_training_grid.csv",
    ):
        (results_dir / name).unlink()
    carried = run_c2a_backtest(
        data_root=store.root,
        output_root=tmp_path,
        prior_results_root=prior_results,
        start_date=dates[20],
        end_date=dates[21],
        variant="v1.2",
        optimize=False,
    )
    assert carried["walk_forward"]["status"] == "CARRIED_FORWARD"
    assert carried["walk_forward"]["optimization_as_of"] == dates[21].date().isoformat()
    assert (
        json.loads((results_dir / "latest_state.json").read_text(encoding="utf-8"))["as_of"]
        == dates[21].date().isoformat()
    )
    assert all(
        (results_dir / name).is_file()
        for name in (
            "walk_forward_selections.csv",
            "walk_forward_oos_trades.csv",
            "latest_training_grid.csv",
        )
    )


def test_challenger_reuses_v12_feature_cache(tmp_path) -> None:
    from robot_quant.c2a_runner import _cache_parameters, _variant_parameters

    strategy = _variant_parameters("v1.2-challenger")
    cache = _cache_parameters("v1.2-challenger")

    assert strategy.c6_threshold == 40.0
    assert strategy.exclude_yesterday_limit_up is True
    assert cache == C2AParameters.dynamic_snapshot()
