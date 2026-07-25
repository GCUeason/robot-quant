from __future__ import annotations

import numpy as np
import pandas as pd

from robot_quant.indicators import (
    assess_forecast_validation,
    assess_sell_rule_outcomes,
    build_forecast_indicators,
)


FEATURE_DEFAULTS = {
    "return_5": 0.01,
    "return_20": 0.02,
    "return_60": 0.03,
    "distance_ma20": 0.01,
    "distance_ma60": 0.02,
    "volume_ratio_20": 1.0,
}


def _add_state_features(predictions: pd.DataFrame) -> pd.DataFrame:
    result = predictions.copy()
    for column, value in FEATURE_DEFAULTS.items():
        result[column] = value
    result["is_out_of_distribution"] = False
    result["ood_features"] = ""
    result["research_target_weight"] = result["target_weight"]
    return result


def test_forecasts_are_unavailable_when_current_model_is_not_calibrated() -> None:
    dates = pd.bdate_range("2026-01-02", periods=50)
    predictions = pd.DataFrame(
        {
            "raw_probability": np.linspace(0.45, 0.65, len(dates)),
            "probability": np.linspace(0.46, 0.64, len(dates)),
            "target_weight": 0.5,
            "model_kind": "calibrated_logistic_regression",
            "realized_label": np.arange(len(dates)) % 2,
            "market_trend_120": -0.02,
            "volatility_20": 0.30,
            "relative_strength_20": 0.01,
        },
        index=dates,
    )
    predictions = _add_state_features(predictions)
    predictions.loc[dates[-1], "model_kind"] = "trend_fallback"
    etf = pd.DataFrame(
        {"close": 1.0 + 0.002 * np.arange(len(dates))},
        index=dates,
    )

    indicators = build_forecast_indicators(etf, predictions, capital=15_000.0)

    assert all(
        forecast["evidence_status"] == "unavailable"
        for forecast in indicators["forecast_horizons"].values()
    )
    assert indicators["sell_indicators"]["action"] == "unavailable"


def test_current_out_of_distribution_state_fails_closed() -> None:
    dates = pd.bdate_range("2024-01-02", periods=320)
    predictions = _add_state_features(
        pd.DataFrame(
            {
                "raw_probability": 0.58,
                "probability": 0.57,
                "target_weight": 0.5,
                "model_kind": "calibrated_logistic_regression",
                "realized_label": np.arange(len(dates)) % 2,
                "market_trend_120": 0.02,
                "volatility_20": 0.25,
                "relative_strength_20": 0.01,
            },
            index=dates,
        )
    )
    predictions.loc[dates[-1], "is_out_of_distribution"] = True
    predictions.loc[dates[-1], "ood_features"] = "volatility_20:above_99pct"
    etf = pd.DataFrame({"close": 1.0 + 0.001 * np.arange(len(dates))}, index=dates)

    indicators = build_forecast_indicators(etf, predictions, capital=15_000.0)

    assert all(
        forecast["evidence_status"] == "unavailable"
        for forecast in indicators["forecast_horizons"].values()
    )
    assert all(
        "分布外" in forecast["unavailable_reason"]
        for forecast in indicators["forecast_horizons"].values()
    )
    assert indicators["execution_gate"]["status"] == "blocked"
    assert indicators["execution_gate"]["model_control_enabled"] is False


def test_regime_mismatch_is_not_forced_into_forecast() -> None:
    dates = pd.bdate_range("2024-01-02", periods=180)
    predictions = _add_state_features(
        pd.DataFrame(
            {
                "raw_probability": 0.58,
                "probability": 0.57,
                "target_weight": 0.5,
                "model_kind": "calibrated_logistic_regression",
                "realized_label": np.arange(len(dates)) % 2,
                "market_trend_120": 0.02,
                "volatility_20": 0.25,
                "relative_strength_20": 0.01,
            },
            index=dates,
        )
    )
    predictions.loc[dates[-1], "market_trend_120"] = -0.02
    etf = pd.DataFrame({"close": 1.0 + 0.001 * np.arange(len(dates))}, index=dates)

    indicators = build_forecast_indicators(etf, predictions, capital=15_000.0)

    for forecast in indicators["forecast_horizons"].values():
        assert forecast["evidence_status"] == "unavailable"
        assert "市场状态" in forecast["unavailable_reason"]
        assert forecast["sample_count"] == 0


def test_overlapping_paths_use_market_observation_positions() -> None:
    dates = pd.date_range("2024-01-02", periods=80, freq="14D")
    predictions = _add_state_features(
        pd.DataFrame(
            {
                "raw_probability": 0.58,
                "probability": 0.57,
                "target_weight": 0.5,
                "model_kind": "calibrated_logistic_regression",
                "realized_label": np.arange(len(dates)) % 2,
                "market_trend_120": 0.02,
                "volatility_20": 0.25,
                "relative_strength_20": 0.01,
            },
            index=dates,
        )
    )
    etf = pd.DataFrame({"close": 1.0 + 0.001 * np.arange(len(dates))}, index=dates)

    indicators = build_forecast_indicators(etf, predictions, capital=15_000.0)

    five_day = indicators["forecast_horizons"]["5"]
    assert five_day["evidence_status"] == "insufficient"
    assert 0 < five_day["sample_count"] < 20
    assert "去除5日重叠路径" in five_day["unavailable_reason"]
    assert five_day["median_return"] is None


def test_forecast_validation_fails_when_prediction_is_worse_than_zero_baseline() -> None:
    actual = pd.Series([0.02, -0.02] * 10)
    records = pd.DataFrame(
        {
            "actual": actual,
            "forecast": -actual * 2.0,
            "lower": -0.10,
            "upper": 0.10,
        }
    )

    validation = assess_forecast_validation(records)

    assert validation["validation_sample_count"] == 20
    assert validation["validation_status"] == "baseline_failed"
    assert validation["validation_beats_zero_baseline"] is False
    assert validation["validation_passes_direction"] is False


def test_forecast_validation_uses_strict_sample_and_baseline_boundaries() -> None:
    actual = pd.Series([0.02, -0.02] * 10)
    good_records = pd.DataFrame(
        {
            "actual": actual,
            "forecast": actual * 0.5,
            "lower": -0.10,
            "upper": 0.10,
        }
    )

    insufficient = assess_forecast_validation(good_records.iloc[:19])
    equal_baseline = assess_forecast_validation(good_records.assign(forecast=0.0))

    assert insufficient["validation_status"] == "insufficient_samples"
    assert equal_baseline["validation_status"] == "baseline_failed"
    assert equal_baseline["validation_beats_zero_baseline"] is False
    assert equal_baseline["validation_passes_direction"] is False


def test_sell_rule_rejects_inverted_signals_and_four_exit_samples() -> None:
    records = pd.DataFrame(
        [
            *[
                {
                    "action": "hold",
                    "actual_return_10": -0.02,
                    "actual_worst_return_10": -0.05,
                }
                for _ in range(20)
            ],
            *[
                {
                    "action": "watch",
                    "actual_return_10": 0.04,
                    "actual_worst_return_10": -0.01,
                }
                for _ in range(16)
            ],
            *[
                {
                    "action": "exit",
                    "actual_return_10": 0.03,
                    "actual_worst_return_10": -0.005,
                }
                for _ in range(4)
            ],
        ]
    )

    validation = assess_sell_rule_outcomes(records)

    assert validation["status"] == "insufficient_action_samples"
    assert validation["directionally_consistent"] is False
    assert validation["actions"]["exit"]["passes_minimum_samples"] is False
    assert any("退出动作独立样本仅4次" in reason for reason in validation["reasons"])
    assert any("方向倒挂" in reason for reason in validation["reasons"])


def test_sell_rule_requires_enough_reduce_and_exit_samples() -> None:
    records = pd.DataFrame(
        [
            *[
                {
                    "action": "hold",
                    "actual_return_10": 0.02,
                    "actual_worst_return_10": -0.01,
                }
                for _ in range(20)
            ],
            *[
                {
                    "action": "reduce",
                    "actual_return_10": -0.02,
                    "actual_worst_return_10": -0.05,
                }
                for _ in range(20)
            ],
        ]
    )

    validation = assess_sell_rule_outcomes(records)

    assert validation["status"] == "insufficient_action_samples"
    assert any("退出动作独立样本仅0次" in reason for reason in validation["reasons"])


def test_latest_failed_predictions_are_included_in_stability_gate() -> None:
    dates = pd.bdate_range("2025-01-02", periods=289)
    labels = np.arange(len(dates)) % 2
    probability = np.where(labels == 1, 0.9, 0.1).astype(float)
    probability[-39:] = 1.0 - probability[-39:]
    predictions = _add_state_features(
        pd.DataFrame(
            {
                "raw_probability": probability,
                "probability": probability,
                "target_weight": 0.5,
                "model_kind": "calibrated_logistic_regression",
                "realized_label": labels,
                "market_trend_120": 0.02,
                "volatility_20": 0.25,
                "relative_strength_20": 0.01,
            },
            index=dates,
        )
    )
    etf = pd.DataFrame({"close": 1.0 + 0.001 * np.arange(len(dates))}, index=dates)

    indicators = build_forecast_indicators(etf, predictions, capital=15_000.0)
    validation = indicators["model_validation"]

    assert validation["passes_brier_baseline"] is True
    assert validation["passes_stability"] is False
    assert validation["status"] == "unstable"
    assert validation["stability_windows"][-1]["window_kind"] == "rolling_tail"
    assert validation["stability_windows"][-1]["passes"] is False


def test_rolling_tail_does_not_count_as_a_third_independent_stability_block() -> None:
    dates = pd.bdate_range("2025-01-02", periods=106)
    labels = np.arange(len(dates)) % 2
    probability = np.where(labels == 1, 0.9, 0.1).astype(float)
    predictions = _add_state_features(
        pd.DataFrame(
            {
                "raw_probability": probability,
                "probability": probability,
                "target_weight": 0.5,
                "model_kind": "calibrated_logistic_regression",
                "realized_label": labels,
                "market_trend_120": 0.02,
                "volatility_20": 0.25,
                "relative_strength_20": 0.01,
            },
            index=dates,
        )
    )
    etf = pd.DataFrame({"close": 1.0 + 0.001 * np.arange(len(dates))}, index=dates)

    indicators = build_forecast_indicators(etf, predictions, capital=15_000.0)

    assert indicators["model_validation"]["passes_stability"] is False


def test_sell_rule_requires_strict_separation_from_overall_baseline() -> None:
    records = pd.DataFrame(
        [
            {
                "action": action,
                "actual_return_10": 0.0,
                "actual_worst_return_10": 0.0,
            }
            for action in ("hold", "watch", "reduce", "exit")
            for _ in range(10)
        ]
    )

    validation = assess_sell_rule_outcomes(records)

    assert validation["status"] == "inverse_signal"
    assert validation["directionally_consistent"] is False
