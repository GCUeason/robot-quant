from __future__ import annotations

import json

import numpy as np
import pandas as pd

from robot_quant.runner import run_daily


def _write_market_data(directory, periods: int = 720) -> None:
    dates = pd.bdate_range("2024-01-02", periods=periods)
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
    expected_contributions = 15_000.0 + 1_000.0 * (history["date"].str[:7].nunique() - 1)
    assert state["total_contributions"] == expected_contributions
    assert state["simulation_status"] == "active"
    assert state["simulation_start_date"] == "2026-07-20"
    assert state["initial_contribution"] == 15_000.0
    assert state["initial_target_weight"] == 1.0
    assert state["monthly_contribution"] == 1_000.0
    assert history.iloc[0]["date"] == "2026-07-20"
    assert history.iloc[0]["contribution"] == 15_000.0
    assert history.iloc[0]["executed_target_weight"] == 1.0
    assert history.iloc[0]["strategy_shares"] > 0
    assert history["executed_target_weight"].eq(1.0).all()
    assert {
        "shadow_target_weight",
        "shadow_portfolio_value",
        "shadow_nav",
    }.issubset(history.columns)
    assert state["execution_policy"] == "fixed_dca_only"
    assert state["model_control_enabled"] is False
    assert state["executable_target_weight"] == 1.0
    assert state["approved_probability"] == 0.5
    assert state["strategy_value"] == state["baseline_value"]
    assert state["shadow_validation"]["status"] == "observation_only"
    assert state["shadow_value_difference"] == state["shadow_value"] - state["baseline_value"]
    assert state["shadow_active_days"] >= 0
    assert state["training_oof_sample_count"] > 0
    assert "training_oof_passed" in state
    assert state["data_provenance"]["forecast_outcomes"] == "robot_index"
    assert len(state["data_provenance"]["known_methodology_breaks"]) == 2
    assert state["strategy_validation"] == {
        "status": "baseline_only",
        "model_excess_value": 0.0,
        "passes_fixed_dca": False,
        "reason": "模型未通过质量门控，当前不具备可验证的交易Alpha",
    }
    assert set(state["forecast_horizons"]) == {"5", "10", "20"}
    for forecast in state["forecast_horizons"].values():
        assert forecast["sample_count"] <= 40
        assert forecast["validation_status"] in {
            "insufficient_samples",
            "baseline_failed",
            "validated",
        }
        assert "validation_beats_zero_baseline" in forecast
        if forecast["validation_status"] != "validated":
            assert forecast["median_return"] is None
        if forecast["evidence_status"] == "sufficient":
            assert forecast["return_p10"] <= forecast["median_return"]
            assert forecast["median_return"] <= forecast["return_p90"]
            assert 0.0 <= forecast["loss_probability"] <= 1.0
        if forecast["validation_interval_coverage"] is not None:
            assert 0.0 <= forecast["validation_interval_coverage"] <= 1.0
    assert state["sell_indicators"]["action"] in {
        "hold",
        "watch",
        "reduce",
        "exit",
        "unavailable",
    }
    for evidence in state["sell_rule_validation"]["actions"].values():
        assert evidence["sample_count"] > 0
        assert 0.0 <= evidence["actual_loss_probability_10"] <= 1.0
    assert state["model_validation"]["calibrated_sample_count"] > 0
    assert "brier_skill_score" in state["model_validation"]
    assert "passes_brier_baseline" in state["model_validation"]
    assert "stability_windows" in state["model_validation"]
    assert len(state["model_validation"]["regime_windows"]) == 4
    assert "passes_regime_stability" in state["model_validation"]
    assert (
        state["model_validation"]["confidence_sample_count"]
        <= state["model_validation"]["calibrated_sample_count"]
    )
    assert state["sell_rule_validation"]["status"] in {
        "insufficient_samples",
        "insufficient_action_samples",
        "inverse_signal",
        "validated",
    }
    assert state["sell_rule_validation"]["minimum_action_sample_count"] == 10
    assert "directionally_consistent" in state["sell_rule_validation"]
    report = (output_dir / "reports" / "latest.md").read_text()
    assert "未来10个交易日跑赢沪深300的预测概率" in report
    assert "未来10日相对沪深300研究信号" in report
    assert "下一交易日执行计划" in report
    assert "下一交易日预测" not in report
    assert "固定定投是唯一可执行模拟" in report
    assert "基线优先门控" in report
    assert "回顾性挑战者" in report
    assert "当前不具备可验证的交易Alpha" in report
    assert "质量门控影子策略（不可执行）" in report
    assert "影子Alpha" in report
    assert "逐日成熟样本门控" in report
    assert "零收益基线门控" in report
    assert "模型分段稳定性" in report
    assert "独立样本" in report
    assert "方向倒挂" in report
    assert "预估收益与下跌风险" in report
    assert "宽口径历史压力参考（不可执行）" in report
    assert "不是当前行情预测" in report
    assert "已隐藏无效的“不可用”空表" in report
    assert "研究型卖出指标" in report
    assert "卖出规则历史验证" in report
    assert "样本外验证" in report
    assert "上涨并跑赢" not in report
    assert (output_dir / "reports" / "performance.png").exists()


def test_run_daily_reports_pending_before_planned_initial_purchase(tmp_path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_market_data(input_dir, periods=420)

    run_daily(offline_data_dir=input_dir, output_root=output_dir)

    state = json.loads((output_dir / "data" / "latest_state.json").read_text())
    history = pd.read_csv(output_dir / "data" / "portfolio_history.csv")
    report = (output_dir / "reports" / "latest.md").read_text()
    assert state["simulation_status"] == "pending"
    assert state["simulation_start_date"] == "2026-07-20"
    assert state["initial_contribution"] == 15_000.0
    assert state["initial_target_weight"] == 1.0
    assert state["total_contributions"] == 0.0
    assert history.empty
    assert "模拟账户尚未开始" in report
    assert "2026-07-20" in report
    assert "¥15,000" in report


def test_run_daily_degrades_when_calibrated_history_is_unavailable(tmp_path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_market_data(input_dir, periods=250)

    run_daily(offline_data_dir=input_dir, output_root=output_dir)

    state = json.loads((output_dir / "data" / "latest_state.json").read_text())
    report = (output_dir / "reports" / "latest.md").read_text()
    assert state["simulation_status"] == "pending"
    assert all(
        forecast["evidence_status"] == "unavailable"
        for forecast in state["forecast_horizons"].values()
    )
    assert state["sell_indicators"]["action"] == "unavailable"
    assert "暂无足够样本" in report
