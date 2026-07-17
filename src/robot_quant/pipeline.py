"""研究信号与可交易行情的对齐。"""

from __future__ import annotations

import pandas as pd


def build_execution_frame(
    etf_prices: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """把收盘后产生的信号映射到下一交易日开盘。"""
    required_price_columns = {"open", "close"}
    if missing := required_price_columns.difference(etf_prices.columns):
        raise ValueError(f"ETF行情缺少列: {', '.join(sorted(missing))}")
    if "target_weight" not in predictions.columns:
        raise ValueError("预测结果缺少target_weight列")

    prices = etf_prices.sort_index().copy()
    price_dates = pd.DataFrame({"date": prices.index})
    signals = predictions.sort_index().loc[:, ["target_weight"]].reset_index()
    signals.columns = ["signal_date", "target_weight"]

    aligned = pd.merge_asof(
        price_dates.sort_values("date"),
        signals.sort_values("signal_date"),
        left_on="date",
        right_on="signal_date",
        direction="backward",
        allow_exact_matches=False,
    ).set_index("date")

    prices["target_weight"] = aligned["target_weight"].fillna(0.0)
    prices["signal_date"] = aligned["signal_date"]
    return prices
