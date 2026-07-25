from __future__ import annotations

import pandas as pd

from robot_quant.portfolio import PortfolioConfig, PortfolioSimulator


def _simulator(
    *,
    commission_rate: float = 0.0,
    minimum_commission: float = 0.0,
) -> PortfolioSimulator:
    return PortfolioSimulator(
        PortfolioConfig(
            initial_contribution=10_000.0,
            monthly_contribution=1_000.0,
            lot_size=100,
            commission_rate=commission_rate,
            minimum_commission=minimum_commission,
            slippage_rate=0.0,
        )
    )


def test_initial_contribution_is_invested_in_whole_lots() -> None:
    prices = pd.DataFrame(
        {
            "open": [10.0, 10.0],
            "close": [10.0, 10.0],
            "target_weight": [1.0, 1.0],
        },
        index=pd.to_datetime(["2026-01-02", "2026-01-05"]),
    )
    simulator = _simulator()

    history = simulator.run(prices)

    first_day = history.iloc[0]
    assert first_day["contribution"] == 10_000.0
    assert first_day["total_contributions"] == 10_000.0
    assert first_day["shares"] == 1_000
    assert first_day["cash"] == 0.0
    assert first_day["portfolio_value"] == 10_000.0


def test_monthly_contribution_occurs_once_on_first_trading_day() -> None:
    prices = pd.DataFrame(
        {
            "open": [10.0, 10.0, 10.0, 10.0],
            "close": [10.0, 10.0, 10.0, 10.0],
            "target_weight": [1.0, 1.0, 1.0, 1.0],
        },
        index=pd.to_datetime(["2026-01-30", "2026-02-02", "2026-02-03", "2026-03-02"]),
    )
    simulator = _simulator()

    history = simulator.run(prices)

    assert history["contribution"].tolist() == [10_000.0, 1_000.0, 0.0, 1_000.0]
    assert history["total_contributions"].tolist() == [10_000.0, 11_000.0, 11_000.0, 12_000.0]
    assert history["shares"].tolist() == [1_000, 1_100, 1_100, 1_200]


def test_target_weight_zero_sells_existing_position() -> None:
    prices = pd.DataFrame(
        {
            "open": [10.0, 10.0],
            "close": [10.0, 10.0],
            "target_weight": [1.0, 0.0],
        },
        index=pd.to_datetime(["2026-01-02", "2026-01-05"]),
    )
    simulator = _simulator()

    history = simulator.run(prices)

    last_day = history.iloc[-1]
    assert last_day["shares"] == 0
    assert last_day["cash"] == 10_000.0
    assert last_day["portfolio_value"] == 10_000.0


def test_buy_order_reserves_cash_for_minimum_commission() -> None:
    prices = pd.DataFrame(
        {"open": [10.0], "close": [10.0], "target_weight": [1.0]},
        index=pd.to_datetime(["2026-01-02"]),
    )
    simulator = _simulator(commission_rate=0.0003, minimum_commission=5.0)

    history = simulator.run(prices)

    first_day = history.iloc[0]
    assert first_day["shares"] == 900
    assert first_day["cash"] == 995.0
    assert first_day["portfolio_value"] == 9_995.0
