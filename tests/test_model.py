from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt

from robot_quant.model import PredictionConfig, WalkForwardPredictor


def _price_frame(periods: int = 420) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-02", periods=periods)
    x = np.arange(periods, dtype=float)
    robot_returns = 0.0004 + 0.003 * np.sin(x / 13.0) + 0.001 * np.cos(x / 5.0)
    benchmark_returns = 0.0002 + 0.0015 * np.sin(x / 17.0)
    robot_close = 100.0 * np.exp(np.cumsum(robot_returns))
    benchmark_close = 100.0 * np.exp(np.cumsum(benchmark_returns))

    robot = pd.DataFrame(
        {
            "open": robot_close,
            "close": robot_close,
            "volume": 1_000_000.0 + 50_000.0 * np.sin(x / 9.0),
        },
        index=dates,
    )
    benchmark = pd.DataFrame(
        {
            "open": benchmark_close,
            "close": benchmark_close,
            "volume": 2_000_000.0,
        },
        index=dates,
    )
    return robot, benchmark


def test_walk_forward_predictions_do_not_change_when_future_prices_change() -> None:
    robot, benchmark = _price_frame()
    predictor = WalkForwardPredictor(PredictionConfig(horizon_days=5, minimum_training_samples=100))

    original = predictor.predict_history(robot, benchmark)
    changed_robot = robot.copy()
    future_start = changed_robot.index[380]
    changed_robot.loc[future_start:, "close"] *= 1.5
    changed_robot.loc[future_start:, "open"] *= 1.5
    changed = predictor.predict_history(changed_robot, benchmark)

    pdt.assert_series_equal(
        original.loc[original.index < future_start, "probability"],
        changed.loc[changed.index < future_start, "probability"],
    )
    assert set(original["target_weight"].unique()).issubset({0.0, 0.5, 1.0})
    assert original["probability"].between(0.0, 1.0).all()


def test_walk_forward_predictions_expose_gap_calibrated_and_raw_probabilities() -> None:
    robot, benchmark = _price_frame(periods=520)
    predictor = WalkForwardPredictor(
        PredictionConfig(
            horizon_days=5,
            minimum_training_samples=120,
            calibration_splits=3,
            calibration_gap=5,
        )
    )

    predictions = predictor.predict_history(robot, benchmark)
    calibrated = predictions[predictions["model_kind"] == "calibrated_logistic_regression"]

    assert "raw_probability" in predictions.columns
    assert not calibrated.empty
    assert calibrated["probability"].between(0.0, 1.0).all()
    assert calibrated["raw_probability"].between(0.0, 1.0).all()
    assert not np.allclose(
        calibrated["probability"].to_numpy(),
        calibrated["raw_probability"].to_numpy(),
    )


def test_walk_forward_marks_extreme_latest_state_as_out_of_distribution() -> None:
    robot, benchmark = _price_frame(periods=520)
    robot.loc[robot.index[-5] :, "close"] *= np.linspace(1.0, 0.55, 5)
    robot.loc[robot.index[-5] :, "open"] = robot.loc[robot.index[-5] :, "close"]
    predictor = WalkForwardPredictor(
        PredictionConfig(
            horizon_days=5,
            minimum_training_samples=120,
            calibration_splits=3,
            calibration_gap=5,
            ood_lower_quantile=0.05,
            ood_upper_quantile=0.95,
        )
    )

    predictions = predictor.predict_history(robot, benchmark)
    latest = predictions.iloc[-1]

    assert bool(latest["is_out_of_distribution"]) is True
    assert latest["ood_features"]
    assert "1pct" not in latest["ood_features"]
    assert "5pct" in latest["ood_features"] or "95pct" in latest["ood_features"]
    assert latest["signal_status"] == "out_of_distribution"
    assert "research_target_weight" in predictions.columns
