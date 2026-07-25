"""基于历史相似状态的收益、下跌风险与模拟卖出指标。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from robot_quant.model import WalkForwardPredictor

HORIZONS = (5, 10, 20)
ANALOG_COUNT = 40
MIN_ANALOG_COUNT = 20
RESEARCH_MODEL_KINDS = {
    "calibrated_logistic_regression",
    WalkForwardPredictor.MODEL_KIND,
}
RELATIVE_STRENGTH_TOLERANCE = 0.03
MAX_SIMILARITY_DISTANCE = 3.0
ANALOG_FEATURES = WalkForwardPredictor.FEATURE_COLUMNS
STABILITY_WINDOW_SIZE = 50
MIN_STABILITY_WINDOW_SAMPLES = 40
MIN_STABILITY_WINDOWS = 3
MIN_FORECAST_VALIDATION_SAMPLES = 20
MIN_SELL_RULE_SAMPLES = 20
MIN_SELL_ACTION_SAMPLES = 10


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
    reference_prices: pd.DataFrame | None = None,
    outcome_source: str = "etf",
) -> dict:
    """用当时已经实现的相似状态构造透明的条件收益分布。"""
    outcome_prices = etf if reference_prices is None else reference_prices
    frame = predictions.join(outcome_prices[["close"]], how="inner").sort_index()
    if frame.empty:
        raise ValueError("预测与结果参考行情没有重合日期")

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
    current_etf = etf.loc[etf.index <= current_date]
    if current_etf.empty:
        raise ValueError("ETF行情缺少当前价格")
    current_etf_close = float(current_etf.iloc[-1]["close"])
    forecasts: dict[str, dict] = {}
    for horizon in HORIZONS:
        known = frame[
            frame[f"known_date_{horizon}"].le(current_date)
            & frame["model_kind"].isin(RESEARCH_MODEL_KINDS)
        ]
        if current["model_kind"] not in RESEARCH_MODEL_KINDS:
            known = known.iloc[0:0]
        analogs = _select_analogs(known, current, horizon, frame.index)
        forecasts[str(horizon)] = _summarize_horizon(
            analogs=analogs,
            current_close=current_etf_close,
            capital=capital,
            horizon=horizon,
            validation=_validate_analogs(frame, horizon),
        )

    sell_rule_validation = _validate_sell_rule(frame)
    model_validation = _model_validation(frame)
    sell_indicators = _sell_indicators(
        frame,
        forecasts,
        model_validation,
        sell_rule_validation,
    )
    return {
        "forecast_horizons": forecasts,
        "forecast_outcome_source": outcome_source,
        "historical_risk_reference": _historical_risk_reference(
            frame,
            current,
            current_date,
        ),
        "distribution_diagnostics": _distribution_diagnostics(current),
        "sell_indicators": sell_indicators,
        "sell_rule_validation": sell_rule_validation,
        "model_validation": model_validation,
        "execution_gate": _execution_gate(current, model_validation),
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


def _historical_risk_reference(
    frame: pd.DataFrame,
    current: pd.Series,
    current_date: pd.Timestamp,
) -> dict[str, dict]:
    """提供宽口径历史基准；它不属于模型预测，也不获得交易权限。"""
    references: dict[str, dict] = {}
    timeline_positions = {pd.Timestamp(date): position for position, date in enumerate(frame.index)}
    current_strength_sign = np.sign(float(current["relative_strength_20"]))

    for horizon in HORIZONS:
        return_column = f"future_return_{horizon}"
        worst_column = f"worst_return_{horizon}"
        trough_column = f"trough_day_{horizon}"
        known_column = f"known_date_{horizon}"
        pool = frame[
            frame[known_column].le(current_date)
            & frame[return_column].notna()
            & frame["relative_strength_20"].notna()
        ]
        same_direction = pool[
            np.sign(pool["relative_strength_20"].astype(float)) == current_strength_sign
        ]
        selected_dates: list[pd.Timestamp] = []
        selected_positions: list[int] = []
        for date in reversed(same_direction.index):
            timestamp = pd.Timestamp(date)
            position = timeline_positions[timestamp]
            if all(
                abs(position - selected_position) >= horizon
                for selected_position in selected_positions
            ):
                selected_dates.append(timestamp)
                selected_positions.append(position)
            if len(selected_dates) >= ANALOG_COUNT:
                break

        selected = same_direction.loc[selected_dates]
        returns = selected[return_column].dropna().astype(float)
        downside = selected[worst_column].dropna().astype(float)
        drawdown_events = selected.loc[selected[worst_column].lt(0.0), trough_column].dropna()
        references[str(horizon)] = {
            "reference_kind": "relative_strength_direction_history",
            "status": "descriptive" if len(returns) >= MIN_ANALOG_COUNT else "limited",
            "criteria": "机器人指数20日相对强弱方向相同；不要求完整特征位于训练分布内",
            "scope_sample_count": int(len(same_direction)),
            "sample_count": int(len(returns)),
            "minimum_sample_count": MIN_ANALOG_COUNT,
            "start_date": (
                pd.Timestamp(selected.index.min()).strftime("%Y-%m-%d")
                if not selected.empty
                else None
            ),
            "end_date": (
                pd.Timestamp(selected.index.max()).strftime("%Y-%m-%d")
                if not selected.empty
                else None
            ),
            "median_return": _optional_quantile(returns, 0.50),
            "return_p10": _optional_quantile(returns, 0.10),
            "return_p90": _optional_quantile(returns, 0.90),
            "loss_probability": (float((returns < 0.0).mean()) if not returns.empty else None),
            "drawdown_event_probability": (
                float((downside < 0.0).mean()) if not downside.empty else None
            ),
            "drawdown_5pct_probability": (
                float((downside <= -0.05).mean()) if not downside.empty else None
            ),
            "drawdown_event_sample_count": int(len(drawdown_events)),
            "drawdown_trough_day_p25": _optional_quantile(drawdown_events, 0.25),
            "drawdown_trough_day_p75": _optional_quantile(drawdown_events, 0.75),
        }
    return references


def _distribution_diagnostics(current: pd.Series) -> list[dict]:
    """输出当天值与模型实际使用的OOD训练边界。"""
    diagnostics: list[dict] = []
    training_sample_count = int(current.get("ood_training_sample_count", 0) or 0)
    for feature in ANALOG_FEATURES:
        lower = current.get(f"{feature}_ood_lower")
        upper = current.get(f"{feature}_ood_upper")
        value = current.get(feature)
        if pd.isna(lower) or pd.isna(upper) or pd.isna(value):
            continue
        numeric_value = float(value)
        numeric_lower = float(lower)
        numeric_upper = float(upper)
        status = (
            "below"
            if numeric_value < numeric_lower
            else "above"
            if numeric_value > numeric_upper
            else "in_range"
        )
        diagnostics.append(
            {
                "feature": feature,
                "current_value": numeric_value,
                "lower_bound": numeric_lower,
                "upper_bound": numeric_upper,
                "status": status,
                "training_sample_count": training_sample_count,
            }
        )
    return diagnostics


def _select_analogs(
    pool: pd.DataFrame,
    current: pd.Series,
    horizon: int,
    timeline: pd.Index,
) -> pd.DataFrame:
    """严格匹配完整状态并剔除收益路径重叠的历史样本。"""
    required = [*ANALOG_FEATURES, "is_out_of_distribution"]
    missing = [column for column in required if column not in pool or column not in current.index]
    if missing:
        return _selection_result(
            pool,
            reason=f"缺少状态特征：{', '.join(missing)}",
            candidate_count=len(pool),
        )
    if bool(current["is_out_of_distribution"]):
        return _selection_result(
            pool,
            reason=f"当前状态处于训练分布外：{current.get('ood_features', '')}",
            candidate_count=len(pool),
        )

    available = pool.dropna(subset=required).copy()
    available = available.loc[~available["is_out_of_distribution"].astype(bool)]
    candidate_count = len(available)
    if available.empty:
        return _selection_result(pool, reason="没有训练分布内候选样本")

    current_market_sign = np.sign(float(current["market_trend_120"]))
    same_regime = available[
        np.sign(available["market_trend_120"].astype(float)) == current_market_sign
    ]
    regime_count = len(same_regime)
    if regime_count < MIN_ANALOG_COUNT:
        return _selection_result(
            pool,
            reason=f"同市场状态样本不足20条（当前{regime_count}条）",
            candidate_count=candidate_count,
            regime_match_count=regime_count,
        )

    current_strength = float(current["relative_strength_20"])
    same_strength = same_regime[
        (np.sign(same_regime["relative_strength_20"].astype(float)) == np.sign(current_strength))
        & (
            (same_regime["relative_strength_20"].astype(float) - current_strength).abs()
            <= RELATIVE_STRENGTH_TOLERANCE
        )
    ]
    strength_count = len(same_strength)
    if strength_count < MIN_ANALOG_COUNT:
        return _selection_result(
            pool,
            reason=f"相对强弱匹配样本不足20条（当前{strength_count}条）",
            candidate_count=candidate_count,
            regime_match_count=regime_count,
            relative_strength_match_count=strength_count,
        )

    state = same_strength.loc[:, ANALOG_FEATURES].astype(float)
    current_state = current.loc[list(ANALOG_FEATURES)].astype(float)
    scale = state.quantile(0.75) - state.quantile(0.25)
    fallback_scale = state.std(ddof=0)
    scale = scale.mask(scale <= 1e-12, fallback_scale).mask(lambda values: values <= 1e-12, 1.0)
    standardized = (state - current_state) / scale
    same_strength = same_strength.copy()
    same_strength["similarity_distance"] = np.sqrt(standardized.pow(2).mean(axis=1))
    close_enough = same_strength[
        same_strength["similarity_distance"].le(MAX_SIMILARITY_DISTANCE)
    ].sort_values("similarity_distance")
    distance_count = len(close_enough)
    if distance_count < MIN_ANALOG_COUNT:
        return _selection_result(
            pool,
            reason=f"完整状态距离合格样本不足20条（当前{distance_count}条）",
            candidate_count=candidate_count,
            regime_match_count=regime_count,
            relative_strength_match_count=strength_count,
            distance_match_count=distance_count,
        )

    timeline_positions = {pd.Timestamp(date): position for position, date in enumerate(timeline)}
    selected_dates: list[pd.Timestamp] = []
    selected_positions: list[int] = []
    for date in close_enough.index:
        timestamp = pd.Timestamp(date)
        position = timeline_positions[timestamp]
        if all(abs(position - selected) >= horizon for selected in selected_positions):
            selected_dates.append(timestamp)
            selected_positions.append(position)
        if len(selected_dates) >= ANALOG_COUNT:
            break
    selected = close_enough.loc[selected_dates].copy()
    reason = (
        ""
        if len(selected) >= MIN_ANALOG_COUNT
        else f"去除{horizon}日重叠路径后有效样本不足20条（当前{len(selected)}条）"
    )
    selected.attrs["selection"] = {
        "unavailable_reason": reason,
        "candidate_count": candidate_count,
        "regime_match_count": regime_count,
        "relative_strength_match_count": strength_count,
        "distance_match_count": distance_count,
        "effective_sample_count": len(selected),
        "maximum_similarity_distance": (
            float(selected["similarity_distance"].max()) if not selected.empty else None
        ),
    }
    return selected


def _selection_result(
    pool: pd.DataFrame,
    *,
    reason: str,
    candidate_count: int = 0,
    regime_match_count: int = 0,
    relative_strength_match_count: int = 0,
    distance_match_count: int = 0,
) -> pd.DataFrame:
    result = pool.iloc[0:0].copy()
    result.attrs["selection"] = {
        "unavailable_reason": reason,
        "candidate_count": candidate_count,
        "regime_match_count": regime_match_count,
        "relative_strength_match_count": relative_strength_match_count,
        "distance_match_count": distance_match_count,
        "effective_sample_count": 0,
        "maximum_similarity_distance": None,
    }
    return result


def _summarize_horizon(
    analogs: pd.DataFrame,
    current_close: float,
    capital: float,
    horizon: int,
    validation: dict,
) -> dict:
    selection = analogs.attrs.get("selection", {})
    returns = analogs[f"future_return_{horizon}"].dropna().astype(float)
    downside = analogs[f"worst_return_{horizon}"].dropna().astype(float)
    if len(returns) < MIN_ANALOG_COUNT:
        return _unavailable_forecast(
            returns=returns,
            selection=selection,
            evidence_status="unavailable" if returns.empty else "insufficient",
            unavailable_reason=selection.get(
                "unavailable_reason",
                f"有效非重叠样本不足20条（当前{len(returns)}条）",
            ),
            validation=validation,
        )
    if validation["validation_status"] != "validated":
        return _unavailable_forecast(
            returns=returns,
            selection=selection,
            evidence_status="unvalidated",
            unavailable_reason=validation["validation_reason"],
            validation=validation,
        )

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
        "effective_sample_count": int(len(returns)),
        "evidence_status": "sufficient",
        "unavailable_reason": "",
        "candidate_count": int(selection.get("candidate_count", 0)),
        "regime_match_count": int(selection.get("regime_match_count", 0)),
        "relative_strength_match_count": int(selection.get("relative_strength_match_count", 0)),
        "distance_match_count": int(selection.get("distance_match_count", 0)),
        "maximum_similarity_distance": selection.get("maximum_similarity_distance"),
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


def _unavailable_forecast(
    *,
    returns: pd.Series,
    selection: dict,
    evidence_status: str,
    unavailable_reason: str,
    validation: dict,
) -> dict:
    return {
        "sample_count": int(len(returns)),
        "effective_sample_count": int(len(returns)),
        "evidence_status": evidence_status,
        "unavailable_reason": unavailable_reason,
        "candidate_count": int(selection.get("candidate_count", 0)),
        "regime_match_count": int(selection.get("regime_match_count", 0)),
        "relative_strength_match_count": int(selection.get("relative_strength_match_count", 0)),
        "distance_match_count": int(selection.get("distance_match_count", 0)),
        "maximum_similarity_distance": selection.get("maximum_similarity_distance"),
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


def _validate_analogs(frame: pd.DataFrame, horizon: int) -> dict:
    actual_column = f"future_return_{horizon}"
    known_column = f"known_date_{horizon}"
    tests = frame[frame["model_kind"].isin(RESEARCH_MODEL_KINDS) & frame[actual_column].notna()]
    records: list[tuple[float, float, float, float]] = []
    timeline_positions = {pd.Timestamp(date): position for position, date in enumerate(frame.index)}
    last_record_position: int | None = None
    for date, row in tests.iterrows():
        position = timeline_positions[pd.Timestamp(date)]
        if last_record_position is not None and position - last_record_position < horizon:
            continue
        pool = frame[frame["model_kind"].isin(RESEARCH_MODEL_KINDS) & frame[known_column].lt(date)]
        analogs = _select_analogs(pool, row, horizon, frame.index)
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
        last_record_position = position

    return assess_forecast_validation(
        pd.DataFrame(
            records,
            columns=["actual", "forecast", "lower", "upper"],
        )
    )


def assess_forecast_validation(records: pd.DataFrame) -> dict:
    """判断收益预测是否在独立样本上真正优于零收益基线。"""
    if records.empty:
        return {
            "validation_sample_count": 0,
            "validation_mae": None,
            "validation_zero_baseline_mae": None,
            "validation_direction_accuracy": None,
            "validation_interval_coverage": None,
            "validation_status": "insufficient_samples",
            "validation_beats_zero_baseline": None,
            "validation_passes_direction": None,
            "validation_reason": (
                f"独立验证样本不足{MIN_FORECAST_VALIDATION_SAMPLES}条（当前0条）"
            ),
        }

    actual = records["actual"].astype(float)
    forecast = records["forecast"].astype(float)
    mae = float((actual - forecast).abs().mean())
    zero_baseline_mae = float(actual.abs().mean())
    direction_accuracy = float(((actual >= 0.0) == (forecast >= 0.0)).mean())
    beats_zero_baseline = mae < zero_baseline_mae
    passes_direction = direction_accuracy > 0.50
    sample_count = int(len(records))
    if sample_count < MIN_FORECAST_VALIDATION_SAMPLES:
        status = "insufficient_samples"
        reason = f"独立验证样本不足{MIN_FORECAST_VALIDATION_SAMPLES}条（当前{sample_count}条）"
    elif not beats_zero_baseline or not passes_direction:
        status = "baseline_failed"
        reason = "预测未同时击败零收益MAE基线并达到高于50%的方向准确率，收益预测已停用"
    else:
        status = "validated"
        reason = "独立样本MAE与方向准确率均通过基线门槛"
    return {
        "validation_sample_count": sample_count,
        "validation_mae": mae,
        "validation_zero_baseline_mae": zero_baseline_mae,
        "validation_direction_accuracy": direction_accuracy,
        "validation_interval_coverage": float(
            actual.between(
                records["lower"].astype(float),
                records["upper"].astype(float),
            ).mean()
        ),
        "validation_status": status,
        "validation_beats_zero_baseline": beats_zero_baseline,
        "validation_passes_direction": passes_direction,
        "validation_reason": reason,
    }


def _model_validation(frame: pd.DataFrame) -> dict:
    realized = frame[
        frame["model_kind"].isin(RESEARCH_MODEL_KINDS) & frame["realized_label"].notna()
    ]
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

    constant_brier = 0.25
    brier_skill_score = (
        1.0 - calibrated_brier / constant_brier if calibrated_brier is not None else None
    )
    passes_brier_baseline = calibrated_brier is not None and calibrated_brier < constant_brier
    passes_auc_baseline = auc is not None and auc > 0.50
    stability_windows = _stability_windows(realized)
    fixed_stability_windows = [
        window for window in stability_windows if window["window_kind"] == "fixed_block"
    ]
    passes_time_stability = len(fixed_stability_windows) >= MIN_STABILITY_WINDOWS and all(
        window["passes"] for window in stability_windows
    )
    regime_windows = _regime_windows(realized)
    passes_regime_stability = len(regime_windows) == 4 and all(
        regime["passes"] for regime in regime_windows
    )
    passes_stability = passes_time_stability and passes_regime_stability

    if count < 100:
        status = "insufficient_samples"
    elif not passes_brier_baseline or not passes_auc_baseline:
        status = "baseline_failed"
    elif not passes_stability:
        status = "unstable"
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
        "score_brier": calibrated_brier,
        "raw_brier": raw_brier,
        "constant_50_brier": constant_brier,
        "brier_skill_score": brier_skill_score,
        "passes_brier_baseline": passes_brier_baseline,
        "passes_auc_baseline": passes_auc_baseline,
        "stability_windows": stability_windows,
        "stability_window_count": len(fixed_stability_windows),
        "stability_pass_count": sum(window["passes"] for window in fixed_stability_windows),
        "passes_time_stability": passes_time_stability,
        "regime_windows": regime_windows,
        "passes_regime_stability": passes_regime_stability,
        "passes_stability": passes_stability,
        "roc_auc": auc,
        "status": status,
        "calibration_method": "fixed_linear_shrinkage_to_50pct",
    }


def _stability_windows(realized: pd.DataFrame) -> list[dict]:
    windows: list[dict] = []
    for start in range(0, len(realized), STABILITY_WINDOW_SIZE):
        window = realized.iloc[start : start + STABILITY_WINDOW_SIZE]
        if len(window) < MIN_STABILITY_WINDOW_SAMPLES:
            continue
        windows.append(_stability_window_summary(window, "fixed_block"))
    remainder = len(realized) % STABILITY_WINDOW_SIZE
    if 0 < remainder < MIN_STABILITY_WINDOW_SAMPLES and len(realized) >= STABILITY_WINDOW_SIZE:
        windows.append(
            _stability_window_summary(
                realized.iloc[-STABILITY_WINDOW_SIZE:],
                "rolling_tail",
            )
        )
    return windows


def _stability_window_summary(window: pd.DataFrame, window_kind: str) -> dict:
    labels = window["realized_label"].astype(int)
    probability = window["probability"].astype(float)
    brier = float(((probability - labels) ** 2).mean())
    auc = float(roc_auc_score(labels, probability)) if labels.nunique() == 2 else None
    passes_brier = brier < 0.25
    passes_auc = auc is not None and auc > 0.50
    return {
        "window_kind": window_kind,
        "start_date": pd.Timestamp(window.index[0]).strftime("%Y-%m-%d"),
        "end_date": pd.Timestamp(window.index[-1]).strftime("%Y-%m-%d"),
        "sample_count": int(len(window)),
        "brier": brier,
        "roc_auc": auc,
        "passes_brier_baseline": passes_brier,
        "passes_auc_baseline": passes_auc,
        "passes": passes_brier and passes_auc,
    }


def _regime_windows(realized: pd.DataFrame) -> list[dict]:
    """按预注册的趋势方向与35%年化波动阈值审计环境泛化。"""
    definitions = (
        ("up_low_vol", True, False),
        ("up_high_vol", True, True),
        ("down_low_vol", False, False),
        ("down_high_vol", False, True),
    )
    windows: list[dict] = []
    for name, market_up, high_volatility in definitions:
        market_mask = realized["market_trend_120"].ge(0.0) == market_up
        volatility_mask = realized["volatility_20"].ge(0.35) == high_volatility
        window = realized.loc[market_mask & volatility_mask]
        labels = window["realized_label"].astype(int)
        scores = window["probability"].astype(float)
        sample_count = len(window)
        brier = float(((scores - labels) ** 2).mean()) if sample_count else None
        auc = (
            float(roc_auc_score(labels, scores)) if sample_count and labels.nunique() == 2 else None
        )
        enough_samples = sample_count >= MIN_STABILITY_WINDOW_SAMPLES
        windows.append(
            {
                "regime": name,
                "sample_count": sample_count,
                "brier": brier,
                "roc_auc": auc,
                "passes": (
                    enough_samples
                    and brier is not None
                    and brier < 0.25
                    and auc is not None
                    and auc > 0.5
                ),
            }
        )
    return windows


def _sell_indicators(
    frame: pd.DataFrame,
    forecasts: dict[str, dict],
    model_validation: dict,
    sell_rule_validation: dict,
) -> dict:
    current = frame.iloc[-1]
    ten_day = forecasts["10"]
    probability = float(current["probability"])
    market_trend = float(current["market_trend_120"])
    thresholds = _sell_thresholds(SELL_RULE)
    if bool(current.get("is_out_of_distribution", False)):
        unavailable_reason = f"当前状态处于训练分布外：{current.get('ood_features', '')}"
    elif model_validation["status"] != "validated":
        unavailable_reason = f"模型验证状态为{model_validation['status']}，卖出指标仅观察且不可执行"
    elif current["model_kind"] not in RESEARCH_MODEL_KINDS:
        unavailable_reason = "当前没有可用的稳健研究模型"
    elif ten_day["evidence_status"] != "sufficient":
        unavailable_reason = ten_day["unavailable_reason"]
    elif sell_rule_validation["status"] != "validated":
        unavailable_reason = f"卖出规则验证状态为{sell_rule_validation['status']}，当前动作不可执行"
    else:
        unavailable_reason = ""
    if unavailable_reason:
        return {
            "action": "unavailable",
            "reason": unavailable_reason,
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
    tests = frame[frame["model_kind"].isin(RESEARCH_MODEL_KINDS) & frame[actual_column].notna()]
    records: list[dict[str, float | str]] = []
    timeline_positions = {pd.Timestamp(date): position for position, date in enumerate(frame.index)}
    last_record_position: int | None = None
    for date, row in tests.iterrows():
        position = timeline_positions[pd.Timestamp(date)]
        if last_record_position is not None and position - last_record_position < horizon:
            continue
        pool = frame[frame["model_kind"].isin(RESEARCH_MODEL_KINDS) & frame[known_column].lt(date)]
        analogs = _select_analogs(pool, row, horizon, frame.index)
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
        last_record_position = position

    return assess_sell_rule_outcomes(pd.DataFrame.from_records(records))


def assess_sell_rule_outcomes(records: pd.DataFrame) -> dict:
    """用独立结果检查卖出动作是否方向一致且具备最低样本量。"""
    if records.empty:
        return {
            "status": "insufficient_samples",
            "sample_count": 0,
            "minimum_sample_count": MIN_SELL_RULE_SAMPLES,
            "minimum_action_sample_count": MIN_SELL_ACTION_SAMPLES,
            "directionally_consistent": False,
            "actions": {},
            "all_dates_baseline": None,
            "reasons": [f"独立验证样本不足{MIN_SELL_RULE_SAMPLES}条（当前0条）"],
            "method": "walk_forward_non_overlapping_known_outcomes_only",
        }

    overall_returns = records["actual_return_10"].astype(float)
    overall_worst_returns = records["actual_worst_return_10"].astype(float)
    overall = {
        "sample_count": int(len(records)),
        "actual_mean_return_10": float(overall_returns.mean()),
        "actual_median_return_10": float(overall_returns.median()),
        "actual_loss_probability_10": float((overall_returns < 0.0).mean()),
        "actual_mean_worst_return_10": float(overall_worst_returns.mean()),
    }
    actions: dict[str, dict] = {}
    for action in ("hold", "watch", "reduce", "exit"):
        outcomes = records.loc[records["action"].eq(action), "actual_return_10"].astype(float)
        worst_outcomes = records.loc[
            records["action"].eq(action),
            "actual_worst_return_10",
        ].astype(float)
        if outcomes.empty:
            continue
        mean_difference = float(outcomes.mean() - overall_returns.mean())
        loss_probability = float((outcomes < 0.0).mean())
        if action == "hold":
            directionally_consistent = (
                mean_difference > 0.0 and loss_probability < overall["actual_loss_probability_10"]
            )
        else:
            directionally_consistent = (
                mean_difference < 0.0 and loss_probability > overall["actual_loss_probability_10"]
            )
        actions[action] = {
            "sample_count": int(len(outcomes)),
            "actual_mean_return_10": float(outcomes.mean()),
            "actual_median_return_10": float(outcomes.median()),
            "actual_loss_probability_10": loss_probability,
            "actual_mean_worst_return_10": float(worst_outcomes.mean()),
            "mean_return_difference_vs_all": mean_difference,
            "passes_minimum_samples": len(outcomes) >= MIN_SELL_ACTION_SAMPLES,
            "directionally_consistent": directionally_consistent,
        }

    reasons: list[str] = []
    sample_count = int(len(records))
    if sample_count < MIN_SELL_RULE_SAMPLES:
        reasons.append(f"独立验证样本不足{MIN_SELL_RULE_SAMPLES}条（当前{sample_count}条）")
    required_actions = ("hold", "watch", "reduce", "exit")
    action_sample_counts = {
        action: int(actions.get(action, {}).get("sample_count", 0)) for action in required_actions
    }
    for action, action_sample_count in action_sample_counts.items():
        if action_sample_count < MIN_SELL_ACTION_SAMPLES:
            action_label = {
                "hold": "保持",
                "watch": "观察",
                "reduce": "减仓",
                "exit": "退出",
            }[action]
            reasons.append(
                f"{action_label}动作独立样本仅{action_sample_count}次，"
                f"低于最低{MIN_SELL_ACTION_SAMPLES}次"
            )
    inconsistent_actions = [
        action for action, evidence in actions.items() if not evidence["directionally_consistent"]
    ]
    if inconsistent_actions:
        labels = {
            "hold": "保持",
            "watch": "观察",
            "reduce": "减仓",
            "exit": "退出",
        }
        reasons.append(
            "动作与后续行情方向倒挂："
            + "、".join(labels[action] for action in inconsistent_actions)
        )

    has_low_action_samples = any(
        count < MIN_SELL_ACTION_SAMPLES for count in action_sample_counts.values()
    )
    directionally_consistent = not inconsistent_actions
    if sample_count < MIN_SELL_RULE_SAMPLES:
        status = "insufficient_samples"
    elif has_low_action_samples:
        status = "insufficient_action_samples"
    elif not directionally_consistent:
        status = "inverse_signal"
    else:
        status = "validated"
    return {
        "status": status,
        "sample_count": sample_count,
        "minimum_sample_count": MIN_SELL_RULE_SAMPLES,
        "minimum_action_sample_count": MIN_SELL_ACTION_SAMPLES,
        "directionally_consistent": directionally_consistent,
        "actions": actions,
        "all_dates_baseline": overall,
        "reasons": reasons,
        "method": "walk_forward_non_overlapping_known_outcomes_only",
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
        return "exit", "收缩后概率型分数、条件收益和下跌概率同时触发退出阈值"
    if probability < config.reduce_probability or (
        median_return < 0.0 and loss_probability >= config.reduce_loss_probability
    ):
        return "reduce", "收缩后概率型分数或10日条件收益触发减仓阈值"
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


def _execution_gate(current: pd.Series, model_validation: dict) -> dict:
    """逐日门控只控制影子账户；实盘权限仍需长期样本外Alpha。"""
    reasons = ["回顾性挑战者尚未用新增数据证明长期优于固定定投"]
    if not bool(current.get("quality_gate_passed", False)):
        reasons.append(f"逐日成熟样本门控未通过：{current.get('quality_gate_reason', '')}")
    if model_validation["status"] != "validated":
        reasons.append(f"当前样本外验证状态：{model_validation['status']}")
    if bool(current.get("is_out_of_distribution", False)):
        reasons.append(f"当前训练分布外特征：{current.get('ood_features', '')}")
    return {
        "status": "blocked",
        "policy": "fixed_dca_only",
        "model_control_enabled": False,
        "executable_target_weight": 1.0,
        "research_target_weight": float(
            current.get("research_target_weight", current["target_weight"])
        ),
        "reasons": reasons,
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
