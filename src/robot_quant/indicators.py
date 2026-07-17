"""基于历史相似状态的收益、下跌风险与模拟卖出指标。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HORIZONS = (5, 10, 20)
ANALOG_COUNT = 40
MIN_ANALOG_COUNT = 20
CALIBRATED_MODEL = "calibrated_logistic_regression"


@dataclass(frozen=True)
class SellRuleConfig:
    """模拟风险动作的集中阈值。"""

    reduce_probability: float = 0.45
    exit_probability: float = 0.35
    watch_loss_probability: float = 0.55
    reduce_loss_probability: float = 0.60
    exit_loss_probability: float = 0.65


SELL_RULE = SellRuleConfig()


def build_forecast_indicators(
    etf: pd.DataFrame,
    predictions: pd.DataFrame,
    capital: float,
) -> dict:
    """用当时已经实现的相似状态构造透明的条件收益分布。"""
    frame = predictions.join(etf[["close"]], how="inner").sort_index()
    if frame.empty:
        raise ValueError("预测与ETF行情没有重合日期")

    for horizon in HORIZONS:
        frame[f"future_return_{horizon}"] = frame["close"].shift(-horizon) / frame["close"] - 1.0
        frame[f"known_date_{horizon}"] = pd.Series(
            frame.index,
            index=frame.index,
        ).shift(-horizon)
        worst_return, trough_day = _future_downside(frame["close"], horizon)
        frame[f"worst_return_{horizon}"] = worst_return
        frame[f"trough_day_{horizon}"] = trough_day

    current = frame.iloc[-1]
    current_date = frame.index[-1]
    forecasts: dict[str, dict] = {}
    for horizon in HORIZONS:
        known = frame[
            frame[f"known_date_{horizon}"].le(current_date)
            & frame["model_kind"].eq(CALIBRATED_MODEL)
        ]
        if current["model_kind"] != CALIBRATED_MODEL:
            known = known.iloc[0:0]
        analogs = _select_analogs(known, current)
        forecasts[str(horizon)] = _summarize_horizon(
            analogs=analogs,
            current_close=float(current["close"]),
            capital=capital,
            horizon=horizon,
            validation=_validate_analogs(frame, horizon),
        )

    model_validation = _model_validation(frame)
    sell_indicators = _sell_indicators(frame, forecasts)
    sell_rule_validation = _validate_sell_rule(frame)
    return {
        "forecast_horizons": forecasts,
        "sell_indicators": sell_indicators,
        "sell_rule_validation": sell_rule_validation,
        "model_validation": model_validation,
    }


def _future_downside(close: pd.Series, horizon: int) -> tuple[pd.Series, pd.Series]:
    worst = pd.Series(np.nan, index=close.index, dtype=float)
    trough_day = pd.Series(np.nan, index=close.index, dtype=float)
    for position in range(len(close) - horizon):
        start = float(close.iloc[position])
        path = close.iloc[position + 1 : position + horizon + 1] / start - 1.0
        worst.iloc[position] = float(path.min())
        trough_day.iloc[position] = float(np.argmin(path.to_numpy()) + 1)
    return worst, trough_day


def _select_analogs(pool: pd.DataFrame, current: pd.Series) -> pd.DataFrame:
    required = [
        "probability",
        "target_weight",
        "market_trend_120",
        "volatility_20",
    ]
    available = pool.dropna(subset=required).copy()
    if available.empty:
        return available

    same_regime = available[
        np.sign(available["market_trend_120"]) == np.sign(float(current["market_trend_120"]))
    ]
    if len(same_regime) >= MIN_ANALOG_COUNT:
        available = same_regime

    same_weight = available[available["target_weight"].eq(float(current["target_weight"]))]
    if len(same_weight) >= MIN_ANALOG_COUNT:
        available = same_weight

    available = available.copy()
    volatility_scale = max(abs(float(current["volatility_20"])), 0.10)
    available["similarity_distance"] = (
        available["probability"] - float(current["probability"])
    ).abs() + 0.25 * (
        available["volatility_20"] - float(current["volatility_20"])
    ).abs() / volatility_scale
    return available.nsmallest(ANALOG_COUNT, "similarity_distance")


def _summarize_horizon(
    analogs: pd.DataFrame,
    current_close: float,
    capital: float,
    horizon: int,
    validation: dict,
) -> dict:
    returns = analogs[f"future_return_{horizon}"].dropna().astype(float)
    downside = analogs[f"worst_return_{horizon}"].dropna().astype(float)
    if returns.empty:
        return {
            "sample_count": 0,
            "evidence_status": "unavailable",
            "mean_return": None,
            "median_return": None,
            "return_p10": None,
            "return_p90": None,
            "loss_probability": None,
            "drawdown_event_probability": None,
            "drawdown_5pct_probability": None,
            "drawdown_event_sample_count": 0,
            "drawdown_trough_day_median": None,
            "drawdown_trough_day_p25": None,
            "drawdown_trough_day_p75": None,
            "drawdown_5pct_sample_count": 0,
            "drawdown_5pct_trough_day_median": None,
            "expected_price": None,
            "downside_price_p10": None,
            "upside_price_p90": None,
            "expected_profit_on_capital": None,
            **validation,
        }

    median_return = float(returns.median())
    return_p10 = float(returns.quantile(0.10))
    return_p90 = float(returns.quantile(0.90))
    drawdown_events = analogs.loc[
        analogs[f"worst_return_{horizon}"].lt(0.0),
        f"trough_day_{horizon}",
    ].dropna()
    drawdown_5pct_events = analogs.loc[
        analogs[f"worst_return_{horizon}"].le(-0.05),
        f"trough_day_{horizon}",
    ].dropna()
    return {
        "sample_count": int(len(returns)),
        "evidence_status": ("sufficient" if len(returns) >= MIN_ANALOG_COUNT else "insufficient"),
        "mean_return": float(returns.mean()),
        "median_return": median_return,
        "return_p10": return_p10,
        "return_p90": return_p90,
        "loss_probability": float((returns < 0.0).mean()),
        "drawdown_event_probability": float((downside < 0.0).mean()),
        "drawdown_5pct_probability": float((downside <= -0.05).mean()),
        "drawdown_event_sample_count": int(len(drawdown_events)),
        "drawdown_trough_day_median": _optional_quantile(drawdown_events, 0.50),
        "drawdown_trough_day_p25": _optional_quantile(drawdown_events, 0.25),
        "drawdown_trough_day_p75": _optional_quantile(drawdown_events, 0.75),
        "drawdown_5pct_sample_count": int(len(drawdown_5pct_events)),
        "drawdown_5pct_trough_day_median": _optional_quantile(
            drawdown_5pct_events,
            0.50,
        ),
        "expected_price": current_close * (1.0 + median_return),
        "downside_price_p10": current_close * (1.0 + return_p10),
        "upside_price_p90": current_close * (1.0 + return_p90),
        "expected_profit_on_capital": capital * median_return,
        **validation,
    }


def _validate_analogs(frame: pd.DataFrame, horizon: int) -> dict:
    actual_column = f"future_return_{horizon}"
    known_column = f"known_date_{horizon}"
    tests = frame[frame["model_kind"].eq(CALIBRATED_MODEL) & frame[actual_column].notna()]
    records: list[tuple[float, float, float, float]] = []
    for date, row in tests.iterrows():
        pool = frame[frame["model_kind"].eq(CALIBRATED_MODEL) & frame[known_column].lt(date)]
        analogs = _select_analogs(pool, row)
        values = analogs[actual_column].dropna().astype(float)
        if len(values) < MIN_ANALOG_COUNT:
            continue
        records.append(
            (
                float(row[actual_column]),
                float(values.median()),
                float(values.quantile(0.10)),
                float(values.quantile(0.90)),
            )
        )

    if not records:
        return {
            "validation_sample_count": 0,
            "validation_mae": None,
            "validation_zero_baseline_mae": None,
            "validation_direction_accuracy": None,
            "validation_interval_coverage": None,
        }

    validation = pd.DataFrame(
        records,
        columns=["actual", "forecast", "lower", "upper"],
    )
    return {
        "validation_sample_count": int(len(validation)),
        "validation_mae": float((validation["actual"] - validation["forecast"]).abs().mean()),
        "validation_zero_baseline_mae": float(validation["actual"].abs().mean()),
        "validation_direction_accuracy": float(
            ((validation["actual"] >= 0.0) == (validation["forecast"] >= 0.0)).mean()
        ),
        "validation_interval_coverage": float(
            validation["actual"]
            .between(
                validation["lower"],
                validation["upper"],
            )
            .mean()
        ),
    }


def _model_validation(frame: pd.DataFrame) -> dict:
    realized = frame[frame["model_kind"].eq(CALIBRATED_MODEL) & frame["realized_label"].notna()]
    labels = realized["realized_label"].astype(int)
    probability = realized["probability"].astype(float)
    raw_probability = realized["raw_probability"].astype(float)
    count = len(realized)
    hits = int(((probability >= 0.5).astype(int) == labels).sum())
    non_overlapping = realized.iloc[::10]
    confidence_labels = non_overlapping["realized_label"].astype(int)
    confidence_probability = non_overlapping["probability"].astype(float)
    confidence_count = len(non_overlapping)
    confidence_hits = int(((confidence_probability >= 0.5).astype(int) == confidence_labels).sum())
    interval = (
        _wilson_interval(confidence_hits, confidence_count) if confidence_count else (None, None)
    )
    calibrated_brier = float(((probability - labels) ** 2).mean()) if count else None
    raw_brier = float(((raw_probability - labels) ** 2).mean()) if count else None
    if count and labels.nunique() == 2:
        auc = float(roc_auc_score(labels, probability))
    else:
        auc = None

    if count < 100:
        status = "insufficient_samples"
    elif calibrated_brier is not None and calibrated_brier >= 0.25:
        status = "observation_only"
    elif interval[0] is None or interval[0] <= 0.50 or confidence_count < 25:
        status = "provisional"
    else:
        status = "validated"
    return {
        "calibrated_sample_count": count,
        "direction_accuracy": hits / count if count else None,
        "confidence_sample_count": confidence_count,
        "confidence_direction_accuracy": (
            confidence_hits / confidence_count if confidence_count else None
        ),
        "accuracy_wilson95_low": interval[0],
        "accuracy_wilson95_high": interval[1],
        "calibrated_brier": calibrated_brier,
        "raw_brier": raw_brier,
        "constant_50_brier": 0.25,
        "roc_auc": auc,
        "status": status,
        "calibration_method": "sigmoid_timeseries_gap10",
    }


def _sell_indicators(frame: pd.DataFrame, forecasts: dict[str, dict]) -> dict:
    current = frame.iloc[-1]
    ten_day = forecasts["10"]
    probability = float(current["probability"])
    market_trend = float(current["market_trend_120"])
    thresholds = _sell_thresholds(SELL_RULE)
    if current["model_kind"] != CALIBRATED_MODEL or ten_day["evidence_status"] != "sufficient":
        return {
            "action": "unavailable",
            "reason": "当前没有至少20条可用的校准相似样本",
            "review_horizon_trading_days": 1,
            "risk_control_price": None,
            "take_profit_price": None,
            **thresholds,
            "current_market_trend_120": market_trend,
            "current_volatility_20": float(current["volatility_20"]),
        }

    median_return = float(ten_day["median_return"])
    loss_probability = float(ten_day["loss_probability"])

    action, reason = _classify_sell_action(
        probability=probability,
        market_trend=market_trend,
        median_return=median_return,
        loss_probability=loss_probability,
        config=SELL_RULE,
    )

    risk_horizons = [
        int(horizon)
        for horizon, forecast in forecasts.items()
        if forecast["loss_probability"] is not None
        and forecast["loss_probability"] >= SELL_RULE.watch_loss_probability
    ]
    review_horizon = min(risk_horizons) if risk_horizons else 10
    current_close = float(current["close"])
    return {
        "action": action,
        "reason": reason,
        "review_horizon_trading_days": review_horizon,
        "risk_control_price": min(current_close, float(ten_day["downside_price_p10"])),
        "take_profit_price": max(current_close, float(ten_day["upside_price_p90"])),
        **thresholds,
        "current_market_trend_120": market_trend,
        "current_volatility_20": float(current["volatility_20"]),
    }


def _validate_sell_rule(frame: pd.DataFrame) -> dict:
    horizon = 10
    actual_column = f"future_return_{horizon}"
    known_column = f"known_date_{horizon}"
    tests = frame[frame["model_kind"].eq(CALIBRATED_MODEL) & frame[actual_column].notna()]
    records: list[dict[str, float | str]] = []
    for date, row in tests.iterrows():
        pool = frame[frame["model_kind"].eq(CALIBRATED_MODEL) & frame[known_column].lt(date)]
        analogs = _select_analogs(pool, row)
        returns = analogs[actual_column].dropna().astype(float)
        if len(returns) < MIN_ANALOG_COUNT:
            continue
        action, _ = _classify_sell_action(
            probability=float(row["probability"]),
            market_trend=float(row["market_trend_120"]),
            median_return=float(returns.median()),
            loss_probability=float((returns < 0.0).mean()),
            config=SELL_RULE,
        )
        records.append(
            {
                "action": action,
                "actual_return_10": float(row[actual_column]),
                "actual_worst_return_10": float(row["worst_return_10"]),
            }
        )

    if not records:
        return {"sample_count": 0, "actions": {}}

    results = pd.DataFrame.from_records(records)
    overall_returns = results["actual_return_10"].astype(float)
    overall_worst_returns = results["actual_worst_return_10"].astype(float)
    overall = {
        "sample_count": int(len(results)),
        "actual_mean_return_10": float(overall_returns.mean()),
        "actual_median_return_10": float(overall_returns.median()),
        "actual_loss_probability_10": float((overall_returns < 0.0).mean()),
        "actual_mean_worst_return_10": float(overall_worst_returns.mean()),
    }
    actions: dict[str, dict] = {}
    for action in ("hold", "watch", "reduce", "exit"):
        outcomes = results.loc[results["action"].eq(action), "actual_return_10"].astype(float)
        worst_outcomes = results.loc[
            results["action"].eq(action),
            "actual_worst_return_10",
        ].astype(float)
        if outcomes.empty:
            continue
        actions[action] = {
            "sample_count": int(len(outcomes)),
            "actual_mean_return_10": float(outcomes.mean()),
            "actual_median_return_10": float(outcomes.median()),
            "actual_loss_probability_10": float((outcomes < 0.0).mean()),
            "actual_mean_worst_return_10": float(worst_outcomes.mean()),
            "mean_return_difference_vs_all": float(outcomes.mean() - overall_returns.mean()),
        }
    return {
        "sample_count": int(len(results)),
        "actions": actions,
        "all_dates_baseline": overall,
        "method": "walk_forward_known_outcomes_only",
    }


def _classify_sell_action(
    probability: float,
    market_trend: float,
    median_return: float,
    loss_probability: float,
    config: SellRuleConfig,
) -> tuple[str, str]:
    if (
        probability < config.exit_probability
        and median_return < 0.0
        and loss_probability >= config.exit_loss_probability
    ):
        return "exit", "校准概率、条件收益和下跌概率同时触发退出阈值"
    if probability < config.reduce_probability or (
        median_return < 0.0 and loss_probability >= config.reduce_loss_probability
    ):
        return "reduce", "校准概率或10日条件收益触发减仓阈值"
    if loss_probability >= config.watch_loss_probability or market_trend < 0.0:
        return "watch", "大盘趋势或条件下跌概率要求继续观察"
    return "hold", "当前未触发减仓或退出阈值"


def _sell_thresholds(config: SellRuleConfig) -> dict:
    return {
        "probability_reduce_trigger": config.reduce_probability,
        "probability_exit_trigger": config.exit_probability,
        "watch_loss_probability_trigger": config.watch_loss_probability,
        "reduce_loss_probability_trigger": config.reduce_loss_probability,
        "exit_loss_probability_trigger": config.exit_loss_probability,
    }


def _optional_quantile(values: pd.Series, quantile: float) -> float | None:
    return None if values.empty else float(values.quantile(quantile))


def _wilson_interval(hits: int, count: int) -> tuple[float, float]:
    z = 1.959963984540054
    proportion = hits / count
    denominator = 1.0 + z**2 / count
    center = (proportion + z**2 / (2.0 * count)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / count + z**2 / (4.0 * count**2))
        / denominator
    )
    return center - half_width, center + half_width
