from __future__ import annotations

import pandas as pd

from robot_quant.pipeline import build_execution_frame


def test_prediction_is_executed_at_next_trading_day_open() -> None:
    dates = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
    etf_prices = pd.DataFrame(
        {"open": [1.0, 1.1, 1.2], "close": [1.05, 1.15, 1.25]},
        index=dates,
    )
    predictions = pd.DataFrame(
        {"probability": [0.7, 0.4, 0.8], "target_weight": [1.0, 0.0, 1.0]},
        index=dates,
    )

    execution = build_execution_frame(etf_prices, predictions)

    assert execution["target_weight"].tolist() == [0.0, 1.0, 0.0]
    assert execution["signal_date"].tolist() == [pd.NaT, dates[0], dates[1]]
