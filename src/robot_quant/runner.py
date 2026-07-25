"""每日研究与模拟入口。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from robot_quant.data import fetch_live_bundle, read_offline_bundle
from robot_quant.indicators import build_forecast_indicators
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
    research_execution = build_execution_frame(bundle.etf, predictions)
    shadow_signals = predictions.copy()
    shadow_signals["target_weight"] = shadow_signals["shadow_target_weight"]
    shadow_execution = build_execution_frame(bundle.etf, shadow_signals)
    execution = research_execution.copy()
    execution["target_weight"] = 1.0

    plan = SIMULATION_PLAN
    portfolio_config = PortfolioConfig(
        initial_contribution=plan.initial_contribution,
        monthly_contribution=plan.monthly_contribution,
    )
    indicators = build_forecast_indicators(
        etf=bundle.etf,
        predictions=predictions,
        capital=plan.initial_contribution,
        reference_prices=bundle.robot_index,
        outcome_source="robot_index",
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
        state.update(indicators)
        state["data_provenance"] = _data_provenance(bundle)
        write_outputs(history, state, output_path)
        return state

    simulation_execution.iloc[
        0,
        simulation_execution.columns.get_loc("target_weight"),
    ] = plan.initial_target_weight
    strategy = simulator.run(simulation_execution)
    simulation_shadow_execution = shadow_execution.loc[
        shadow_execution.index >= plan.start_date
    ].copy()
    shadow = simulator.run(simulation_shadow_execution)
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
        shadow_execution=simulation_shadow_execution,
        shadow=shadow,
    )
    state = _latest_state(
        history,
        predictions,
        plan,
    )
    state.update(indicators)
    state["data_provenance"] = _data_provenance(bundle)
    write_outputs(history, state, output_path)
    return state


def _combine_history(
    etf: pd.DataFrame,
    predictions: pd.DataFrame,
    execution: pd.DataFrame,
    strategy: pd.DataFrame,
    baseline: pd.DataFrame,
    shadow_execution: pd.DataFrame,
    shadow: pd.DataFrame,
) -> pd.DataFrame:
    history = pd.DataFrame(index=etf.index)
    history["etf_open"] = etf["open"].astype(float)
    history["etf_close"] = etf["close"].astype(float)
    aligned_predictions = predictions.reindex(history.index, method="ffill")
    history["signal_probability"] = aligned_predictions["probability"]
    history["research_target_weight"] = aligned_predictions["research_target_weight"]
    history["next_target_weight"] = 1.0
    history["executed_target_weight"] = execution["target_weight"]
    history["shadow_target_weight"] = shadow_execution["target_weight"]
    history["quality_gate_passed"] = aligned_predictions["quality_gate_passed"].astype(bool)
    history["contribution"] = strategy["contribution"]
    history["total_contributions"] = strategy["total_contributions"]

    for prefix, account in (
        ("strategy", strategy),
        ("baseline", baseline),
        ("shadow", shadow),
    ):
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
    shadow_value = float(latest["shadow_portfolio_value"])

    return {
        "simulation_status": "active",
        "simulation_start_date": plan.start_date.strftime("%Y-%m-%d"),
        "market_date": history.index[-1].strftime("%Y-%m-%d"),
        "etf_close": float(latest["etf_close"]),
        "raw_prediction_probability": float(latest_prediction["raw_probability"]),
        "prediction_probability": float(latest_prediction["probability"]),
        "probability_shrinkage": float(latest_prediction["probability_shrinkage"]),
        "approved_probability": 0.5,
        "research_target_weight": float(latest_prediction["research_target_weight"]),
        "next_target_weight": 1.0,
        "executable_target_weight": 1.0,
        "model_control_enabled": False,
        "execution_policy": "fixed_dca_only",
        "signal_status": str(latest_prediction["signal_status"]),
        "is_out_of_distribution": bool(latest_prediction["is_out_of_distribution"]),
        "ood_features": str(latest_prediction["ood_features"]),
        "model_kind": str(latest_prediction["model_kind"]),
        "model_version": str(latest_prediction["model_version"]),
        "model_selection_status": str(latest_prediction["model_selection_status"]),
        "training_oof_sample_count": int(latest_prediction["training_oof_sample_count"]),
        "training_oof_brier": _optional_float(latest_prediction["training_oof_brier"]),
        "training_oof_auc": _optional_float(latest_prediction["training_oof_auc"]),
        "training_oof_passed": bool(latest_prediction["training_oof_passed"]),
        "quality_gate_sample_count": int(latest_prediction["quality_gate_sample_count"]),
        "quality_gate_brier": _optional_float(latest_prediction["quality_gate_brier"]),
        "quality_gate_auc": _optional_float(latest_prediction["quality_gate_auc"]),
        "quality_gate_passed": bool(latest_prediction["quality_gate_passed"]),
        "quality_gate_reason": str(latest_prediction["quality_gate_reason"]),
        "shadow_target_weight": float(latest_prediction["shadow_target_weight"]),
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
        "strategy_validation": _strategy_validation(strategy_value, baseline_value),
        "shadow_value": shadow_value,
        "shadow_profit": shadow_value - contributions,
        "shadow_roi": shadow_value / contributions - 1.0,
        "shadow_max_drawdown": _max_drawdown(history["shadow_nav"]),
        "shadow_value_difference": shadow_value - baseline_value,
        "shadow_active_days": int(history["shadow_target_weight"].gt(0.0).sum()),
        "shadow_gate_pass_days": int(history["quality_gate_passed"].sum()),
        "shadow_validation": _shadow_validation(
            shadow_value,
            baseline_value,
            len(history),
        ),
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
        "raw_prediction_probability": float(latest_prediction["raw_probability"]),
        "prediction_probability": float(latest_prediction["probability"]),
        "probability_shrinkage": float(latest_prediction["probability_shrinkage"]),
        "approved_probability": 0.5,
        "research_target_weight": float(latest_prediction["research_target_weight"]),
        "next_target_weight": 1.0,
        "executable_target_weight": 1.0,
        "model_control_enabled": False,
        "execution_policy": "fixed_dca_only",
        "signal_status": str(latest_prediction["signal_status"]),
        "is_out_of_distribution": bool(latest_prediction["is_out_of_distribution"]),
        "ood_features": str(latest_prediction["ood_features"]),
        "model_kind": str(latest_prediction["model_kind"]),
        "model_version": str(latest_prediction["model_version"]),
        "model_selection_status": str(latest_prediction["model_selection_status"]),
        "training_oof_sample_count": int(latest_prediction["training_oof_sample_count"]),
        "training_oof_brier": _optional_float(latest_prediction["training_oof_brier"]),
        "training_oof_auc": _optional_float(latest_prediction["training_oof_auc"]),
        "training_oof_passed": bool(latest_prediction["training_oof_passed"]),
        "quality_gate_sample_count": int(latest_prediction["quality_gate_sample_count"]),
        "quality_gate_brier": _optional_float(latest_prediction["quality_gate_brier"]),
        "quality_gate_auc": _optional_float(latest_prediction["quality_gate_auc"]),
        "quality_gate_passed": bool(latest_prediction["quality_gate_passed"]),
        "quality_gate_reason": str(latest_prediction["quality_gate_reason"]),
        "shadow_target_weight": float(latest_prediction["shadow_target_weight"]),
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
        "strategy_validation": _strategy_validation(0.0, 0.0),
        "shadow_value": 0.0,
        "shadow_profit": 0.0,
        "shadow_roi": None,
        "shadow_max_drawdown": None,
        "shadow_value_difference": 0.0,
        "shadow_active_days": 0,
        "shadow_gate_pass_days": 0,
        "shadow_validation": _shadow_validation(0.0, 0.0, 0),
    }


def _prediction_accuracy(predictions: pd.DataFrame) -> float | None:
    realized = predictions.dropna(subset=["realized_label"])
    if realized.empty:
        return None
    predicted_label = (realized["probability"] >= 0.5).astype(float)
    return float((predicted_label == realized["realized_label"]).mean())


def _strategy_validation(strategy_value: float, baseline_value: float) -> dict:
    return {
        "status": "baseline_only",
        "model_excess_value": strategy_value - baseline_value,
        "passes_fixed_dca": False,
        "reason": "模型未通过质量门控，当前不具备可验证的交易Alpha",
    }


def _shadow_validation(shadow_value: float, baseline_value: float, sample_days: int) -> dict:
    return {
        "status": "observation_only",
        "sample_days": sample_days,
        "model_excess_value": shadow_value - baseline_value,
        "reason": "影子策略仅用于衡量逐日成熟样本门控，不具备交易授权",
    }


def _data_provenance(bundle) -> dict:
    return {
        "etf": bundle.etf_source,
        "model_features": bundle.robot_index_source,
        "forecast_outcomes": "robot_index",
        "benchmark": bundle.benchmark_source,
        "robot_index_start": bundle.robot_index.index.min().strftime("%Y-%m-%d"),
        "robot_index_end": bundle.robot_index.index.max().strftime("%Y-%m-%d"),
        "robot_index_sample_count": len(bundle.robot_index),
        "known_methodology_breaks": [
            {
                "effective_date": "2023-03-03",
                "reason": "指数代码及选样、权重规则修订",
            },
            {
                "effective_date": "2025-04-10",
                "reason": "样本数量、选样空间及权重规则修订",
            },
        ],
    }


def _optional_float(value) -> float | None:
    return None if pd.isna(value) else float(value)


def _empty_history() -> pd.DataFrame:
    columns = [
        "etf_open",
        "etf_close",
        "signal_probability",
        "research_target_weight",
        "next_target_weight",
        "executed_target_weight",
        "shadow_target_weight",
        "quality_gate_passed",
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
        "shadow_cash",
        "shadow_shares",
        "shadow_portfolio_value",
        "shadow_nav",
    ]
    return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], name="date"))


def _max_drawdown(nav: pd.Series) -> float:
    drawdown = nav / nav.cummax() - 1.0
    return float(np.nanmin(drawdown.to_numpy()))
