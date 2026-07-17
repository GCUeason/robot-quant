from __future__ import annotations

import numpy as np
import pandas as pd

from robot_quant.indicators import build_forecast_indicators


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
