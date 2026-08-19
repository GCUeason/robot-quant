from __future__ import annotations

from dataclasses import replace

import pandas as pd

from robot_quant.c2a import C2AParameters
from robot_quant.c2a_optimize import (
    WalkForwardConfig,
    _features_for_parameters,
    _parameter_winners,
    _promotion_reasons,
    make_walk_forward_folds,
    walk_forward_optimize,
)


def test_walk_forward_folds_have_embargo_and_non_overlapping_tests() -> None:
    dates = pd.bdate_range("2026-01-05", periods=105)
    folds = make_walk_forward_folds(
        dates,
        WalkForwardConfig(min_train_days=60, test_days=20, embargo_days=1),
    )
    assert len(folds) == 3
    assert all(fold["train_end"] < fold["embargo_start"] <= fold["embargo_end"] for fold in folds)
    assert all(fold["embargo_end"] < fold["test_start"] for fold in folds)
    assert folds[0]["test_end"] < folds[1]["test_start"]


def test_historical_walk_forward_never_counts_as_forward_paper_validation() -> None:
    reasons = _promotion_reasons(pd.DataFrame(), pd.DataFrame(), "STRICT")
    assert "forward_validation_days_below_60" in reasons
    assert "forward_validation_trades_below_40" in reasons
    assert "price_cage_not_tick_verified" in reasons


def test_limit_groups_use_separate_signal_feature_caches() -> None:
    include_first_board = pd.DataFrame({"group": ["B"]})
    exclude_all_limit_up = pd.DataFrame({"group": ["A"]})
    cache = {False: include_first_board, True: exclude_all_limit_up}
    assert _features_for_parameters(cache, C2AParameters()) is include_first_board
    assert (
        _features_for_parameters(
            cache,
            replace(C2AParameters(), exclude_yesterday_limit_up=True),
        )
        is exclude_all_limit_up
    )


def test_parameter_winners_keep_return_risk_and_robustness_objectives_separate() -> None:
    frame = pd.DataFrame(
        [
            {
                "parameter_id": "return",
                "trade_count": 20,
                "total_return": 0.20,
                "max_drawdown": -0.15,
                "sharpe": 1.0,
                "robustness_score": 0.05,
                "parameters": C2AParameters(),
            },
            {
                "parameter_id": "stable",
                "trade_count": 20,
                "total_return": 0.10,
                "max_drawdown": -0.05,
                "sharpe": 1.5,
                "robustness_score": 0.12,
                "parameters": C2AParameters(),
            },
        ]
    )
    winners = _parameter_winners(frame, 10)
    assert winners["highest_total_return"]["parameter_id"] == "return"
    assert winners["smallest_drawdown"]["parameter_id"] == "stable"
    assert winners["highest_robustness"]["parameter_id"] == "stable"


def test_walk_forward_precomputes_each_training_parameter_only_once(monkeypatch, tmp_path) -> None:
    calls: list[float] = []

    def fake_backtest(_minutes, universe, params, **kwargs):
        calls.append(params.to_dict()["c6_threshold"])
        start = pd.Timestamp(kwargs["trade_start"])
        end = pd.Timestamp(kwargs["trade_end"])
        dates = universe.loc[universe["trade_date"].between(start, end), "trade_date"]
        equity = pd.DataFrame(
            {
                "trade_date": dates,
                "portfolio_value": 100_000.0,
            }
        )
        return pd.DataFrame(), equity, {}

    monkeypatch.setattr("robot_quant.c2a_optimize.backtest_c2a", fake_backtest)
    dates = pd.bdate_range("2026-01-05", periods=105)
    universe = pd.DataFrame({"trade_date": dates})
    candidates = [C2AParameters(), replace(C2AParameters(), c6_threshold=25.0)]
    selections, _trades, _summary = walk_forward_optimize(
        pd.DataFrame(),
        universe,
        candidates,
        config=WalkForwardConfig(min_train_days=60, test_days=20, embargo_days=1),
        path_cache_dir=tmp_path / "parameter_paths",
        path_cache_context="fixture-v1",
    )
    assert len(selections) == 3
    assert len(calls) == len(candidates)

    resumed, _resumed_trades, _resumed_summary = walk_forward_optimize(
        pd.DataFrame(),
        universe,
        candidates,
        config=WalkForwardConfig(min_train_days=60, test_days=20, embargo_days=1),
        path_cache_dir=tmp_path / "parameter_paths",
        path_cache_context="fixture-v1",
    )
    assert len(resumed) == 3
    assert len(calls) == len(candidates)
