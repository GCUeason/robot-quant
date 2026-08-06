from __future__ import annotations

import pandas as pd

from robot_quant.robot_chain import RobotChainHolding, RobotChainPlan, simulate_robot_chain
from robot_quant.robot_chain_report import write_robot_chain_outputs


HOLDING = RobotChainHolding(
    code="000001",
    symbol="sz000001",
    name="测试机器人公司",
    chain_stage="上游",
    subsector="测试部件",
    initial_shares=100,
)


def _prices(closes: list[float]) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2026-07-01", periods=len(closes))
    close = pd.Series(closes, index=dates, dtype=float)
    return {
        HOLDING.code: pd.DataFrame(
            {
                "open": close,
                "close": close,
            },
            index=dates,
        )
    }


def _plan(start_date: str, *, initial_capital: float = 10_000.0) -> RobotChainPlan:
    return RobotChainPlan(
        start_date=pd.Timestamp(start_date),
        initial_capital=initial_capital,
        commission_rate=0.0,
        minimum_commission=0.0,
        slippage_rate=0.0,
    )


def test_simulation_buys_fixed_initial_shares_and_keeps_cash_hedge() -> None:
    prices = _prices([10.0] * 30)

    history, state = simulate_robot_chain(
        prices,
        plan=_plan("2026-07-29"),
        holdings=(HOLDING,),
    )

    assert history.index[0] == pd.Timestamp("2026-07-29")
    assert history.iloc[0]["000001_shares"] == 100
    assert history.iloc[0]["cash"] == 9_000.0
    assert history.iloc[0]["portfolio_value"] == 10_000.0
    assert state["simulation_status"] == "active"
    assert state["equity_weight"] == 0.1
    assert state["automated_trading_enabled"] is False


def test_three_consecutive_closes_below_moving_average_halves_position_next_open() -> None:
    prices = _prices([10.0] * 20 + [10.0, 8.0, 8.0, 8.0, 8.0])

    history, _ = simulate_robot_chain(
        prices,
        plan=_plan("2026-07-29"),
        holdings=(HOLDING,),
    )

    assert history.iloc[-1]["000001_shares"] == 50
    assert "连续3日跌破20日均线，减半" in history.iloc[-1]["actions"]


def test_drawdown_rule_reduces_equity_exposure_at_next_open() -> None:
    holding = RobotChainHolding(
        code="000001",
        symbol="sz000001",
        name="测试机器人公司",
        chain_stage="上游",
        subsector="测试部件",
        initial_shares=500,
    )
    prices = _prices([10.0] * 20 + [10.0, 8.0, 8.0])

    history, _ = simulate_robot_chain(
        prices,
        plan=_plan("2026-07-29"),
        holdings=(holding,),
    )

    assert history.iloc[-1]["000001_shares"] == 393
    assert "股票仓位降至35%" in history.iloc[-1]["actions"]


def test_pending_simulation_writes_observable_outputs(tmp_path) -> None:
    prices = _prices([10.0] * 25)
    history, state = simulate_robot_chain(
        prices,
        plan=_plan("2026-09-01"),
        holdings=(HOLDING,),
    )

    write_robot_chain_outputs(history, state, tmp_path)

    assert history.empty
    assert state["simulation_status"] == "pending"
    assert (tmp_path / "data" / "robot_chain_history.csv").exists()
    assert (tmp_path / "data" / "robot_chain_latest_state.json").exists()
    report = (tmp_path / "reports" / "robot_chain_latest.md").read_text()
    assert "等待建仓" in report
    assert (tmp_path / "reports" / "robot_chain_performance.png").exists()
