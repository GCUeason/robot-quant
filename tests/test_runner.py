from __future__ import annotations

import json

import numpy as np
import pandas as pd

from robot_quant.runner import run_daily


def _write_market_data(directory) -> None:
    dates = pd.bdate_range("2024-01-02", periods=420)
    x = np.arange(len(dates), dtype=float)

    def frame(base: float, drift: float, cycle: float) -> pd.DataFrame:
        returns = drift + 0.002 * np.sin(x / cycle)
        close = base * np.exp(np.cumsum(returns))
        return pd.DataFrame(
            {
                "date": dates,
                "open": close * 0.999,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1_000_000.0 + 10_000.0 * np.cos(x / 8.0),
                "amount": close * 1_000_000.0,
            }
        )

    directory.mkdir(parents=True)
    frame(1.0, 0.0004, 13.0).iloc[120:].to_csv(directory / "etf.csv", index=False)
    frame(1_000.0, 0.0005, 11.0).to_csv(directory / "robot_index.csv", index=False)
    frame(4_000.0, 0.0002, 17.0).to_csv(directory / "benchmark.csv", index=False)


def test_run_daily_is_idempotent_and_writes_observable_results(tmp_path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_market_data(input_dir)

    run_daily(offline_data_dir=input_dir, output_root=output_dir)
    first_history = (output_dir / "data" / "portfolio_history.csv").read_bytes()
    run_daily(offline_data_dir=input_dir, output_root=output_dir)
    second_history = (output_dir / "data" / "portfolio_history.csv").read_bytes()

    assert first_history == second_history
    state = json.loads((output_dir / "data" / "latest_state.json").read_text())
    history = pd.read_csv(output_dir / "data" / "portfolio_history.csv")
    expected_contributions = 10_000.0 + 1_000.0 * (history["date"].str[:7].nunique() - 1)
    assert state["total_contributions"] == expected_contributions
    assert state["initial_contribution"] == 10_000.0
    assert state["monthly_contribution"] == 1_000.0
    report = (output_dir / "reports" / "latest.md").read_text()
    assert "未来10个交易日跑赢沪深300的预测概率" in report
    assert "上涨并跑赢" not in report
    assert (output_dir / "reports" / "performance.png").exists()
