from __future__ import annotations

import numpy as np
import pandas as pd

from robot_quant.indicators import build_forecast_indicators


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
