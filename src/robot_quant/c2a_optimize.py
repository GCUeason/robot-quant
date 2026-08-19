"""C2-A 走样本外参数选择与稳定性审计。"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
from pathlib import Path
from collections.abc import Callable
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from robot_quant.c2a import (
    C2AParameters,
    C2APreparedData,
    backtest_c2a,
    backtest_c2a_schedule,
    build_signal_features,
    prepare_c2a_data,
    prepare_c2a_market_data,
    summarize_backtest,
)


SignalFeatureCache = pd.DataFrame | dict[bool, pd.DataFrame]
PreparedDataCache = C2APreparedData | dict[bool, C2APreparedData]


_FORK_RUN_CONTEXT: dict = {}


@dataclass(frozen=True)
class _CandidateRun:
    params: C2AParameters
    start_date: pd.Timestamp
    trades: pd.DataFrame
    equity: pd.DataFrame


@dataclass(frozen=True)
class WalkForwardConfig:
    min_train_days: int = 60
    test_days: int = 20
    embargo_days: int = 1
    min_train_trades: int = 10


def parameter_id(params: C2AParameters) -> str:
    payload = json.dumps(params.to_dict(), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def make_walk_forward_folds(
    trading_dates: Iterable[pd.Timestamp], config: WalkForwardConfig | None = None
) -> list[dict]:
    """使用扩展训练窗、1日隔离带和不重叠测试窗。"""

    policy = config or WalkForwardConfig()
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(list(trading_dates)).normalize().unique()))
    folds: list[dict] = []
    cursor = policy.min_train_days + policy.embargo_days
    fold_number = 1
    while cursor < len(dates):
        test_end_index = min(cursor + policy.test_days, len(dates))
        train_end_index = cursor - policy.embargo_days
        test_dates = dates[cursor:test_end_index]
        if len(test_dates) == 0:
            break
        folds.append(
            {
                "fold": fold_number,
                "train_start": dates[0],
                "train_end": dates[train_end_index - 1],
                "embargo_start": dates[train_end_index],
                "embargo_end": dates[cursor - 1],
                "test_start": test_dates[0],
                "test_end": test_dates[-1],
                "test_days": len(test_dates),
            }
        )
        cursor = test_end_index
        fold_number += 1
    return folds


def evaluate_parameter_grid(
    minutes: pd.DataFrame,
    universe: pd.DataFrame,
    candidates: Iterable[C2AParameters],
    *,
    start_date,
    end_date,
    initial_capital: float = 100_000.0,
    data_status: str = "PROXY",
    signal_features: SignalFeatureCache | None = None,
    prepared_data: PreparedDataCache | None = None,
    precomputed_runs: dict[str, _CandidateRun] | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    for params in candidates:
        run = precomputed_runs.get(parameter_id(params)) if precomputed_runs else None
        if run is None:
            _trades, _equity, summary = backtest_c2a(
                minutes,
                universe,
                params,
                initial_capital=initial_capital,
                trade_start=start_date,
                trade_end=end_date,
                data_status=data_status,
                signal_features=_features_for_parameters(signal_features, params),
                prepared_data=_prepared_for_parameters(prepared_data, params),
            )
        else:
            if run.start_date != pd.Timestamp(start_date).normalize():
                raise ValueError("预计算参数路径的起始日与评估窗口不一致")
            cutoff = pd.Timestamp(end_date).normalize()
            trades = (
                run.trades.loc[
                    pd.to_datetime(run.trades.get("exit_date", pd.Series(dtype="datetime64[ns]")))
                    .dt.normalize()
                    .le(cutoff)
                ].copy()
                if not run.trades.empty
                else run.trades.copy()
            )
            equity = run.equity.loc[
                pd.to_datetime(run.equity["trade_date"]).dt.normalize().le(cutoff)
            ].copy()
            summary = summarize_backtest(
                trades,
                equity,
                params,
                initial_capital,
                data_status,
            )
        rows.append(_parameter_result_row(params, summary))
    result = pd.DataFrame(rows)
    return add_robustness_scores(result)


def add_robustness_scores(results: pd.DataFrame) -> pd.DataFrame:
    """用相邻参数收益中位数惩罚孤立尖峰，默认选择稳定平台而非单点最高。"""

    if results.empty:
        return results.copy()
    frame = results.copy()
    stable_returns: list[float] = []
    positive_neighbor_rates: list[float] = []
    for _, row in frame.iterrows():
        same_family = frame["scan_end"].eq(row["scan_end"]) & frame["limit_group"].eq(
            row["limit_group"]
        )
        distance = (
            (frame["c6_threshold"] - row["c6_threshold"]).abs() / 5.0
            + (frame["main_first_pullback"] - row["main_first_pullback"]).abs() / 0.005
            + (frame["growth_first_pullback"] - row["growth_first_pullback"]).abs() / 0.005
            + (frame["second_increment"] - row["second_increment"]).abs() / 0.005
        )
        neighborhood = frame.loc[same_family & distance.le(1.01)]
        stable_returns.append(float(neighborhood["total_return"].median()))
        positive_neighbor_rates.append(float(neighborhood["total_return"].gt(0).mean()))
    frame["neighbor_median_return"] = stable_returns
    frame["positive_neighbor_rate"] = positive_neighbor_rates
    drawdown_penalty = frame["max_drawdown"].abs() * 0.5
    sparse_penalty = np.where(frame["trade_count"] < 10, 0.02, 0.0)
    frame["robustness_score"] = (
        frame["neighbor_median_return"]
        + 0.25 * frame["total_return"]
        - drawdown_penalty
        - sparse_penalty
    )
    return frame


def select_robust_parameter(results: pd.DataFrame, min_trades: int = 10) -> C2AParameters | None:
    eligible = results.loc[
        results["trade_count"].ge(min_trades) & results["positive_neighbor_rate"].ge(0.60)
    ]
    if eligible.empty:
        return None
    winner = eligible.sort_values(
        ["robustness_score", "max_drawdown", "parameter_id"],
        ascending=[False, False, True],
    ).iloc[0]
    return winner["parameters"]


def walk_forward_optimize(
    minutes: pd.DataFrame,
    universe: pd.DataFrame,
    candidates: Iterable[C2AParameters],
    *,
    initial_capital: float = 100_000.0,
    data_status: str = "PROXY",
    config: WalkForwardConfig | None = None,
    signal_features: SignalFeatureCache | None = None,
    prepared_data: PreparedDataCache | None = None,
    optimization_start=None,
    optimization_end=None,
    progress_callback: Callable[[str], None] | None = None,
    path_cache_dir: str | Path | None = None,
    path_cache_context: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """每折只用过去训练窗选参，再在隔离带之后的测试窗评估。"""

    policy = config or WalkForwardConfig()
    candidate_list = list(candidates)
    if not candidate_list:
        raise ValueError("参数网格不能为空")
    dates = universe["trade_date"].drop_duplicates().sort_values()
    if optimization_start is not None:
        dates = dates.loc[dates.ge(pd.Timestamp(optimization_start).normalize())]
    if optimization_end is not None:
        dates = dates.loc[dates.le(pd.Timestamp(optimization_end).normalize())]
    folds = make_walk_forward_folds(dates, policy)
    precomputed_runs = (
        _precompute_candidate_runs(
            minutes,
            universe,
            candidate_list,
            start_date=dates.iloc[0],
            end_date=dates.iloc[-1],
            initial_capital=initial_capital,
            data_status=data_status,
            signal_features=signal_features,
            prepared_data=prepared_data,
            progress_callback=progress_callback,
            path_cache_dir=path_cache_dir,
            path_cache_context=path_cache_context,
        )
        if folds
        else {}
    )
    selections: list[dict] = []
    selected_by_fold: dict[int, C2AParameters] = {}
    latest_training = pd.DataFrame()
    latest_training_end = None
    for fold in folds:
        training = evaluate_parameter_grid(
            minutes,
            universe,
            candidate_list,
            start_date=fold["train_start"],
            end_date=fold["train_end"],
            initial_capital=initial_capital,
            data_status=data_status,
            signal_features=signal_features,
            prepared_data=prepared_data,
            precomputed_runs=precomputed_runs,
        )
        latest_training = training
        latest_training_end = fold["train_end"]
        selected = select_robust_parameter(training, policy.min_train_trades)
        if selected is None:
            selections.append(
                {
                    **fold,
                    "status": "NO_ROBUST_PARAMETER",
                    "selected_parameter_id": None,
                    "train_trade_count": int(training["trade_count"].max()),
                    "oos_trade_count": 0,
                    "oos_total_return": None,
                }
            )
            continue
        selected_by_fold[fold["fold"]] = selected
        train_winner = training.loc[training["parameter_id"].eq(parameter_id(selected))].iloc[0]
        selections.append(
            {
                **fold,
                "status": "EVALUATED_OOS",
                "selected_parameter_id": parameter_id(selected),
                "selected_parameters": selected.to_dict(),
                "train_trade_count": int(train_winner["trade_count"]),
                "train_total_return": float(train_winner["total_return"]),
                "train_neighbor_median_return": float(train_winner["neighbor_median_return"]),
                "oos_trade_count": 0,
                "oos_total_return": None,
                "oos_max_drawdown": None,
            }
        )
    selection_frame = pd.DataFrame(selections)
    selection_frame, trades_frame = _evaluate_continuous_oos(
        selection_frame,
        folds,
        selected_by_fold,
        dates,
        minutes,
        universe,
        initial_capital,
        data_status,
        signal_features,
        prepared_data,
    )
    valid_folds = (
        selection_frame.loc[selection_frame["status"].eq("EVALUATED_OOS")]
        if "status" in selection_frame
        else selection_frame
    )
    summary = {
        "execution_permission": "PAPER_ONLY",
        "data_status": data_status,
        "execution_fidelity": "MINUTE_BAR_PROXY",
        "fold_count": len(selection_frame),
        "evaluated_fold_count": len(valid_folds),
        "oos_trade_count": len(trades_frame),
        "oos_profitable_fold_rate": (
            float(valid_folds["oos_total_return"].gt(0).mean()) if len(valid_folds) else None
        ),
        "oos_mean_trade_return": (
            float(trades_frame["net_return"].mean()) if len(trades_frame) else None
        ),
        "oos_median_trade_return": (
            float(trades_frame["net_return"].median()) if len(trades_frame) else None
        ),
        "promotion_gate": "FAIL",
        "promotion_reasons": _promotion_reasons(selection_frame, trades_frame, data_status),
        "latest_training_end": (
            pd.Timestamp(latest_training_end).date().isoformat()
            if latest_training_end is not None
            else None
        ),
        "latest_training_winners": _parameter_winners(latest_training, policy.min_train_trades),
        "_latest_training_grid_records": _export_parameter_grid(latest_training).to_dict(
            orient="records"
        ),
    }
    if not summary["promotion_reasons"]:
        summary["promotion_gate"] = "REVIEW_REQUIRED"
    return selection_frame, trades_frame, summary


def _evaluate_continuous_oos(
    selections: pd.DataFrame,
    folds: list[dict],
    selected_by_fold: dict[int, C2AParameters],
    dates: pd.Series,
    minutes: pd.DataFrame,
    universe: pd.DataFrame,
    initial_capital: float,
    data_status: str,
    signal_features: SignalFeatureCache | None,
    prepared_data: PreparedDataCache | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """用一个连续账户评估所有 OOS 折，不在折边界清空持仓或冷却。"""

    if selections.empty or not selected_by_fold:
        return selections, pd.DataFrame()
    schedule: dict[pd.Timestamp, C2AParameters | None] = {}
    fold_by_date: dict[pd.Timestamp, int] = {}
    selected_id_by_fold: dict[int, str] = {}
    normalized_dates = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    for fold in folds:
        fold_number = int(fold["fold"])
        selected = selected_by_fold.get(fold_number)
        test_dates = normalized_dates[
            (normalized_dates >= fold["test_start"]) & (normalized_dates <= fold["test_end"])
        ]
        for trade_day in test_dates:
            schedule[pd.Timestamp(trade_day)] = selected
            fold_by_date[pd.Timestamp(trade_day)] = fold_number
        if selected is not None:
            selected_id_by_fold[fold_number] = parameter_id(selected)

    prepared_groups = _prepared_groups_for_schedule(
        minutes,
        universe,
        list(selected_by_fold.values()),
        signal_features,
        prepared_data,
    )
    trades, equity, _summary = backtest_c2a_schedule(
        schedule,
        prepared_groups,
        initial_capital=initial_capital,
        data_status=data_status,
    )
    trades = trades.copy()
    if not trades.empty:
        entry_dates = pd.to_datetime(trades["entry_date"]).dt.normalize()
        trades["fold"] = entry_dates.map(fold_by_date)
        trades["selected_parameter_id"] = trades["fold"].map(selected_id_by_fold)

    equity_dates = (
        pd.to_datetime(equity["trade_date"]).dt.normalize()
        if not equity.empty
        else pd.Series(dtype="datetime64[ns]")
    )
    result = selections.copy()
    for row_index, row in result.loc[result["status"].eq("EVALUATED_OOS")].iterrows():
        fold_number = int(row["fold"])
        before = equity.loc[equity_dates.lt(row["test_start"])]
        base_value = (
            float(before.iloc[-1]["portfolio_value"]) if not before.empty else initial_capital
        )
        fold_equity = equity.loc[equity_dates.between(row["test_start"], row["test_end"])]
        if fold_equity.empty:
            continue
        end_value = float(fold_equity.iloc[-1]["portfolio_value"])
        values = pd.concat(
            [pd.Series([base_value]), fold_equity["portfolio_value"].astype(float)],
            ignore_index=True,
        )
        fold_trades = trades.loc[trades["fold"].eq(fold_number)] if not trades.empty else trades
        result.at[row_index, "oos_trade_count"] = len(fold_trades)
        result.at[row_index, "oos_total_return"] = end_value / base_value - 1.0
        result.at[row_index, "oos_max_drawdown"] = float((values / values.cummax() - 1.0).min())
    return result, trades


def _prepared_groups_for_schedule(
    minutes: pd.DataFrame,
    universe: pd.DataFrame,
    selected: list[C2AParameters],
    signal_features: SignalFeatureCache | None,
    prepared_data: PreparedDataCache | None,
) -> dict[bool, C2APreparedData]:
    if isinstance(prepared_data, dict):
        return prepared_data
    if isinstance(prepared_data, C2APreparedData):
        return {False: prepared_data, True: prepared_data}
    market = prepare_c2a_market_data(minutes, universe)
    groups: dict[bool, C2APreparedData] = {}
    for limit_group in {params.exclude_yesterday_limit_up for params in selected}:
        cached = (
            signal_features.get(limit_group)
            if isinstance(signal_features, dict)
            else signal_features
        )
        if cached is None:
            group_params = max(
                (params for params in selected if params.exclude_yesterday_limit_up == limit_group),
                key=lambda params: params.scan_end,
            )
            cached = build_signal_features(minutes, universe, group_params)
        groups[limit_group] = prepare_c2a_data(market, cached)
    return groups


def _features_for_parameters(
    signal_features: SignalFeatureCache | None,
    params: C2AParameters,
) -> pd.DataFrame | None:
    if isinstance(signal_features, dict):
        return signal_features[params.exclude_yesterday_limit_up]
    return signal_features


def _prepared_for_parameters(
    prepared_data: PreparedDataCache | None,
    params: C2AParameters,
) -> C2APreparedData | None:
    if isinstance(prepared_data, dict):
        return prepared_data[params.exclude_yesterday_limit_up]
    return prepared_data


def _precompute_candidate_runs(
    minutes: pd.DataFrame,
    universe: pd.DataFrame,
    candidates: list[C2AParameters],
    *,
    start_date,
    end_date,
    initial_capital: float,
    data_status: str,
    signal_features: SignalFeatureCache | None,
    prepared_data: PreparedDataCache | None,
    progress_callback: Callable[[str], None] | None,
    path_cache_dir: str | Path | None,
    path_cache_context: str | None,
) -> dict[str, _CandidateRun]:
    start = pd.Timestamp(start_date).normalize()
    cache_dir = Path(path_cache_dir) if path_cache_dir is not None else None
    context_id = _path_cache_context_id(
        universe,
        start,
        end_date,
        initial_capital,
        data_status,
        path_cache_context,
    )
    runs = _load_cached_candidate_runs(cache_dir, context_id, candidates, start)
    pending = [params for params in candidates if parameter_id(params) not in runs]
    if progress_callback is not None and runs:
        progress_callback(f"C2-A 参数路径：已从检查点恢复 {len(runs)}/{len(candidates)} 组")
    if not pending:
        return runs
    if len(pending) >= 4 and os.name == "posix":
        completed = _precompute_candidate_runs_forked(
            minutes,
            universe,
            pending,
            start=start,
            end_date=end_date,
            initial_capital=initial_capital,
            data_status=data_status,
            signal_features=signal_features,
            prepared_data=prepared_data,
            progress_callback=progress_callback,
            completed_offset=len(runs),
            total_candidates=len(candidates),
            cache_dir=cache_dir,
            cache_context=context_id,
        )
        runs.update(completed)
        return runs
    for params in pending:
        trades, equity, _summary = backtest_c2a(
            minutes,
            universe,
            params,
            initial_capital=initial_capital,
            trade_start=start,
            trade_end=end_date,
            data_status=data_status,
            signal_features=_features_for_parameters(signal_features, params),
            prepared_data=_prepared_for_parameters(prepared_data, params),
        )
        run = _CandidateRun(params, start, trades, equity)
        candidate_id = parameter_id(params)
        runs[candidate_id] = run
        _write_candidate_run_cache(cache_dir, context_id, candidate_id, run)
        if progress_callback is not None and (len(runs) % 10 == 0 or len(runs) == len(candidates)):
            progress_callback(f"C2-A 参数路径：已完成 {len(runs)}/{len(candidates)} 组")
    return runs


def _precompute_candidate_runs_forked(
    minutes: pd.DataFrame,
    universe: pd.DataFrame,
    candidates: list[C2AParameters],
    *,
    start: pd.Timestamp,
    end_date,
    initial_capital: float,
    data_status: str,
    signal_features: SignalFeatureCache | None,
    prepared_data: PreparedDataCache | None,
    progress_callback: Callable[[str], None] | None,
    completed_offset: int,
    total_candidates: int,
    cache_dir: Path | None,
    cache_context: str,
) -> dict[str, _CandidateRun]:
    """Linux fork 共享只读特征内存，用两个 CPU 核并行参数路径。"""

    global _FORK_RUN_CONTEXT
    _FORK_RUN_CONTEXT = {
        "minutes": minutes,
        "universe": universe,
        "start": start,
        "end_date": end_date,
        "initial_capital": initial_capital,
        "data_status": data_status,
        "signal_features": signal_features,
        "prepared_data": prepared_data,
    }
    runs: dict[str, _CandidateRun] = {}
    context = multiprocessing.get_context("fork")
    try:
        with context.Pool(processes=1, maxtasksperchild=20) as pool:
            for pair_start in range(0, len(candidates), 2):
                child_result = pool.apply_async(
                    _run_forked_candidate,
                    (candidates[pair_start],),
                )
                pair = []
                if pair_start + 1 < len(candidates):
                    # 父进程与子进程同时计算，总进程数保持为2，避免8GB配额下 OOM。
                    parent_result = _run_forked_candidate(candidates[pair_start + 1])
                    pair.append(parent_result)
                pair.append(child_result.get())
                for candidate_id, run in pair:
                    runs[candidate_id] = run
                    _write_candidate_run_cache(
                        cache_dir,
                        cache_context,
                        candidate_id,
                        run,
                    )
                completed = completed_offset + len(runs)
                if progress_callback is not None and (
                    completed % 10 == 0 or completed == total_candidates
                ):
                    progress_callback(f"C2-A 参数路径：已完成 {completed}/{total_candidates} 组")
    finally:
        _FORK_RUN_CONTEXT = {}
    return runs


def _run_forked_candidate(params: C2AParameters) -> tuple[str, _CandidateRun]:
    context = _FORK_RUN_CONTEXT
    trades, equity, _summary = backtest_c2a(
        context["minutes"],
        context["universe"],
        params,
        initial_capital=context["initial_capital"],
        trade_start=context["start"],
        trade_end=context["end_date"],
        data_status=context["data_status"],
        signal_features=_features_for_parameters(context["signal_features"], params),
        prepared_data=_prepared_for_parameters(context["prepared_data"], params),
    )
    return parameter_id(params), _CandidateRun(params, context["start"], trades, equity)


def _path_cache_context_id(
    universe: pd.DataFrame,
    start: pd.Timestamp,
    end_date,
    initial_capital: float,
    data_status: str,
    external_context: str | None,
) -> str:
    if external_context is None:
        universe_hash = hashlib.sha256(
            pd.util.hash_pandas_object(universe, index=False).to_numpy().tobytes()
        ).hexdigest()
    else:
        universe_hash = external_context
    payload = {
        "schema": 1,
        "start": start.date().isoformat(),
        "end": pd.Timestamp(end_date).date().isoformat(),
        "initial_capital": initial_capital,
        "data_status": data_status,
        "data_context": universe_hash,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _candidate_cache_path(
    cache_dir: Path | None,
    context_id: str,
    candidate_id: str,
) -> Path | None:
    return cache_dir / context_id / f"{candidate_id}.pkl.gz" if cache_dir is not None else None


def _load_cached_candidate_runs(
    cache_dir: Path | None,
    context_id: str,
    candidates: list[C2AParameters],
    start: pd.Timestamp,
) -> dict[str, _CandidateRun]:
    runs: dict[str, _CandidateRun] = {}
    for params in candidates:
        candidate_id = parameter_id(params)
        path = _candidate_cache_path(cache_dir, context_id, candidate_id)
        if path is None or not path.exists():
            continue
        try:
            payload = pd.read_pickle(path)
        except (EOFError, OSError, ValueError):
            continue
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != 1
            or payload.get("context_id") != context_id
            or payload.get("parameter_id") != candidate_id
        ):
            continue
        run = payload.get("run")
        if not isinstance(run, _CandidateRun) or run.start_date != start or run.params != params:
            continue
        runs[candidate_id] = run
    return runs


def _write_candidate_run_cache(
    cache_dir: Path | None,
    context_id: str,
    candidate_id: str,
    run: _CandidateRun,
) -> None:
    path = _candidate_cache_path(cache_dir, context_id, candidate_id)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.pkl.gz")
    pd.to_pickle(
        {
            "schema": 1,
            "context_id": context_id,
            "parameter_id": candidate_id,
            "run": run,
        },
        temporary,
        compression="gzip",
    )
    temporary.replace(path)


def _parameter_result_row(params: C2AParameters, summary: dict) -> dict:
    return {
        "parameter_id": parameter_id(params),
        "total_return": summary["total_return"],
        "max_drawdown": summary["max_drawdown"],
        "sharpe": summary["sharpe"],
        "trade_count": summary["trade_count"],
        "win_rate": summary["win_rate"],
        "mean_trade_return": summary["mean_trade_return"],
        "profit_factor": summary["profit_factor"],
        "transaction_cost_to_capital": summary["transaction_cost_to_capital"],
        "scan_end": params.scan_end.strftime("%H:%M"),
        "c6_threshold": params.c6_threshold,
        "main_first_pullback": params.main_first_pullback,
        "growth_first_pullback": params.growth_first_pullback,
        "second_increment": params.main_alt_increment,
        "limit_group": ("A_EXCLUDE_ALL" if params.exclude_yesterday_limit_up else "B_FIRST_BOARD"),
        "parameters": params,
    }


def _parameter_winners(results: pd.DataFrame, min_trades: int) -> dict:
    eligible = results.loc[results["trade_count"].ge(min_trades)].copy()
    if eligible.empty:
        return {}
    objectives = {
        "highest_total_return": ("total_return", False),
        "highest_sharpe": ("sharpe", False),
        "smallest_drawdown": ("max_drawdown", False),
        "highest_robustness": ("robustness_score", False),
        "highest_win_rate": ("win_rate", False),
        "highest_mean_trade_return": ("mean_trade_return", False),
    }
    winners: dict[str, dict] = {}
    for label, (metric, ascending) in objectives.items():
        if metric not in eligible:
            continue
        candidates = eligible.dropna(subset=[metric])
        if candidates.empty:
            continue
        row = candidates.sort_values([metric, "parameter_id"], ascending=[ascending, True]).iloc[0]
        winners[label] = {
            "parameter_id": row["parameter_id"],
            "metric": float(row[metric]),
            "trade_count": int(row["trade_count"]),
            "total_return": float(row["total_return"]),
            "max_drawdown": float(row["max_drawdown"]),
            "parameters": row["parameters"].to_dict(),
        }
    return winners


def _export_parameter_grid(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results.copy()
    exported = results.drop(columns=["parameters"]).copy()
    exported["parameters_json"] = results["parameters"].map(
        lambda params: json.dumps(params.to_dict(), ensure_ascii=False, sort_keys=True)
    )
    return exported


def _promotion_reasons(
    selections: pd.DataFrame, trades: pd.DataFrame, data_status: str
) -> list[str]:
    reasons: list[str] = []
    # 历史 walk-forward 不是从今天开始冻结信号的前向纸面验证，不能互相替代。
    reasons.extend(
        [
            "forward_validation_days_below_60",
            "forward_validation_trades_below_40",
            "extended_scan_vs_1000_forward_edge_not_proven",
            "price_cage_not_tick_verified",
        ]
    )
    if data_status != "STRICT":
        reasons.append("data_not_strict")
    if len(trades) < 40:
        reasons.append("historical_oos_trades_below_40")
    if trades.empty or trades["entry_date"].nunique() < 20:
        reasons.append("historical_oos_independent_samples_below_20")
    if trades.empty or float(trades["net_return"].mean()) <= 0:
        reasons.append("oos_expectation_not_positive")
    if not trades.empty:
        positive_profit = trades.loc[trades["profit"] > 0, "profit"].sort_values(ascending=False)
        if (
            positive_profit.sum() > 0
            and positive_profit.head(3).sum() / positive_profit.sum() > 0.50
        ):
            reasons.append("oos_profit_concentrated_in_top_3_trades")
    valid = (
        selections.loc[selections["status"].eq("EVALUATED_OOS")]
        if "status" in selections
        else selections
    )
    if valid.empty or float(valid["oos_total_return"].gt(0).mean()) < 0.60:
        reasons.append("fold_stability_insufficient")
    return reasons
