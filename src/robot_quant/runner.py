"""每日研究与模拟入口。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from robot_quant.data import fetch_live_bundle, read_offline_bundle
from robot_quant.model import PredictionConfig, WalkForwardPredictor
from robot_quant.pipeline import build_execution_frame
from robot_quant.portfolio import PortfolioConfig, PortfolioSimulator
from robot_quant.report import write_outputs


@dataclass(frozen=True)
class SimulationPlan:
    """首次建仓与后续定投计划。"""

    start_date: pd.Timestamp
    initial_contribution: float
    monthly_contribution: float
    initial_target_weight: float


SIMULATION_PLAN = SimulationPlan(
    start_date=pd.Timestamp("2026-07-20"),
    initial_contribution=15_000.0,
    monthly_contribution=1_000.0,
    initial_target_weight=1.0,
)


def run_daily(
    offline_data_dir: str | Path | None = None,
    output_root: str | Path = ".",
) -> dict:
    """重算完整历史并写出最新结果；同一行情重复运行结果一致。"""

    output_path = Path(output_root)
    bundle = (
        read_offline_bundle(Path(offline_data_dir))
        if offline_data_dir is not None
        else fetch_live_bundle()
    )
    predictor = WalkForwardPredictor(PredictionConfig())
    predictions = predictor.predict_history(bundle.robot_index, bundle.benchmark)
    execution = build_execution_frame(bundle.etf, predictions)

    plan = SIMULATION_PLAN
    portfolio_config = PortfolioConfig(
        initial_contribution=plan.initial_contribution,
        monthly_contribution=plan.monthly_contribution,
    )
    simulator = PortfolioSimulator(portfolio_config)
    simulation_execution = execution.loc[execution.index >= plan.start_date].copy()
    if simulation_execution.empty:
        history = _empty_history()
        state = _pending_state(
            bundle.etf,
            predictions,
            plan,
        )
        write_outputs(history, state, output_path)
        return state

    simulation_execution.iloc[
        0,
        simulation_execution.columns.get_loc("target_weight"),
    ] = plan.initial_target_weight
    strategy = simulator.run(simulation_execution)
    simulation_etf = bundle.etf.loc[bundle.etf.index >= plan.start_date]
    baseline_execution = simulation_etf.loc[:, ["open", "close"]].copy()
    baseline_execution["target_weight"] = 1.0
    baseline = simulator.run(baseline_execution)

    history = _combine_history(
        etf=simulation_etf,
        predictions=predictions,
        execution=simulation_execution,
        strategy=strategy,
        baseline=baseline,
    )
    state = _latest_state(
        history,
        predictions,
        plan,
    )
    write_outputs(history, state, output_path)
    return state


def _combine_history(
    etf: pd.DataFrame,
    predictions: pd.DataFrame,
    execution: pd.DataFrame,
    strategy: pd.DataFrame,
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    history = pd.DataFrame(index=etf.index)
    history["etf_open"] = etf["open"].astype(float)
    history["etf_close"] = etf["close"].astype(float)
    aligned_predictions = predictions.reindex(history.index, method="ffill")
    history["signal_probability"] = aligned_predictions["probability"]
    history["next_target_weight"] = aligned_predictions["target_weight"]
    history["executed_target_weight"] = execution["target_weight"]
    history["contribution"] = strategy["contribution"]
    history["total_contributions"] = strategy["total_contributions"]

    for prefix, account in (("strategy", strategy), ("baseline", baseline)):
        history[f"{prefix}_cash"] = account["cash"]
        history[f"{prefix}_shares"] = account["shares"]
        history[f"{prefix}_portfolio_value"] = account["portfolio_value"]
        history[f"{prefix}_nav"] = _flow_adjusted_nav(account)
    return history


def _flow_adjusted_nav(account: pd.DataFrame) -> pd.Series:
    previous_value = account["portfolio_value"].shift(1)
    daily_return = (account["portfolio_value"] - account["contribution"]) / previous_value - 1.0
    daily_return.iloc[0] = 0.0
    return (1.0 + daily_return.fillna(0.0)).cumprod()


def _latest_state(
    history: pd.DataFrame,
    predictions: pd.DataFrame,
    plan: SimulationPlan,
) -> dict:
    latest = history.iloc[-1]
    latest_prediction = predictions.iloc[-1]
    contributions = float(latest["total_contributions"])
    strategy_value = float(latest["strategy_portfolio_value"])
    baseline_value = float(latest["baseline_portfolio_value"])

    return {
        "simulation_status": "active",
        "simulation_start_date": plan.start_date.strftime("%Y-%m-%d"),
        "market_date": history.index[-1].strftime("%Y-%m-%d"),
        "etf_close": float(latest["etf_close"]),
        "prediction_probability": float(latest_prediction["probability"]),
        "next_target_weight": float(latest_prediction["target_weight"]),
        "model_kind": str(latest_prediction["model_kind"]),
        "prediction_accuracy": _prediction_accuracy(predictions),
        "initial_contribution": plan.initial_contribution,
        "initial_target_weight": plan.initial_target_weight,
        "monthly_contribution": plan.monthly_contribution,
        "total_contributions": contributions,
        "strategy_value": strategy_value,
        "strategy_profit": strategy_value - contributions,
        "strategy_roi": strategy_value / contributions - 1.0,
        "strategy_max_drawdown": _max_drawdown(history["strategy_nav"]),
        "baseline_value": baseline_value,
        "baseline_profit": baseline_value - contributions,
        "baseline_roi": baseline_value / contributions - 1.0,
        "baseline_max_drawdown": _max_drawdown(history["baseline_nav"]),
        "strategy_value_difference": strategy_value - baseline_value,
    }


def _pending_state(
    etf: pd.DataFrame,
    predictions: pd.DataFrame,
    plan: SimulationPlan,
) -> dict:
    latest_prediction = predictions.iloc[-1]
    return {
        "simulation_status": "pending",
        "simulation_start_date": plan.start_date.strftime("%Y-%m-%d"),
        "market_date": etf.index[-1].strftime("%Y-%m-%d"),
        "etf_close": float(etf.iloc[-1]["close"]),
        "prediction_probability": float(latest_prediction["probability"]),
        "next_target_weight": float(latest_prediction["target_weight"]),
        "model_kind": str(latest_prediction["model_kind"]),
        "prediction_accuracy": _prediction_accuracy(predictions),
        "initial_contribution": plan.initial_contribution,
        "initial_target_weight": plan.initial_target_weight,
        "monthly_contribution": plan.monthly_contribution,
        "total_contributions": 0.0,
        "strategy_value": 0.0,
        "strategy_profit": 0.0,
        "strategy_roi": None,
        "strategy_max_drawdown": None,
        "baseline_value": 0.0,
        "baseline_profit": 0.0,
        "baseline_roi": None,
        "baseline_max_drawdown": None,
        "strategy_value_difference": 0.0,
    }


def _prediction_accuracy(predictions: pd.DataFrame) -> float | None:
    realized = predictions.dropna(subset=["realized_label"])
    if realized.empty:
        return None
    predicted_label = (realized["probability"] >= 0.5).astype(float)
    return float((predicted_label == realized["realized_label"]).mean())


def _empty_history() -> pd.DataFrame:
    columns = [
        "etf_open",
        "etf_close",
        "signal_probability",
        "next_target_weight",
        "executed_target_weight",
        "contribution",
        "total_contributions",
        "strategy_cash",
        "strategy_shares",
        "strategy_portfolio_value",
        "strategy_nav",
        "baseline_cash",
        "baseline_shares",
        "baseline_portfolio_value",
        "baseline_nav",
    ]
    return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], name="date"))


def _max_drawdown(nav: pd.Series) -> float:
    drawdown = nav / nav.cummax() - 1.0
    return float(np.nanmin(drawdown.to_numpy()))
