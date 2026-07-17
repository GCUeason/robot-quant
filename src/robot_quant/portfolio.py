"""模拟账户与定投记账。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd


@dataclass(frozen=True)
class PortfolioConfig:
    """模拟账户参数。"""

    initial_contribution: float
    monthly_contribution: float
    lot_size: int = 100
    commission_rate: float = 0.0003
    minimum_commission: float = 5.0
    slippage_rate: float = 0.0005


class PortfolioSimulator:
    """按照每日目标仓位模拟A股ETF账户。"""

    def __init__(self, config: PortfolioConfig) -> None:
        self.config = config

    def run(self, prices: pd.DataFrame) -> pd.DataFrame:
        """执行模拟并返回逐日账户历史。"""
        required_columns = {"open", "close", "target_weight"}
        missing_columns = required_columns.difference(prices.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"缺少行情列: {missing}")
        if prices.empty:
            raise ValueError("行情不能为空")

        ordered = prices.sort_index()
        cash = 0.0
        shares = 0
        total_contributions = 0.0
        records: list[dict[str, float | int | pd.Timestamp]] = []
        previous_month: pd.Period | None = None

        for position, (date, row) in enumerate(ordered.iterrows()):
            current_month = pd.Timestamp(date).to_period("M")
            if position == 0:
                contribution = self.config.initial_contribution
            elif current_month != previous_month:
                contribution = self.config.monthly_contribution
            else:
                contribution = 0.0
            cash += contribution
            total_contributions += contribution

            target_weight = min(1.0, max(0.0, float(row["target_weight"])))
            open_price = float(row["open"])
            portfolio_value_at_open = cash + shares * open_price
            target_shares = (
                math.floor(
                    portfolio_value_at_open
                    * target_weight
                    / open_price
                    / self.config.lot_size
                )
                * self.config.lot_size
            )

            if target_shares > shares:
                buy_shares = target_shares - shares
                execution_price = open_price * (1.0 + self.config.slippage_rate)
                trade_value = buy_shares * execution_price
                commission = self._commission(trade_value)
                while buy_shares > 0 and trade_value + commission > cash:
                    buy_shares -= self.config.lot_size
                    trade_value = buy_shares * execution_price
                    commission = self._commission(trade_value)
                if trade_value + commission <= cash:
                    cash -= trade_value + commission
                    shares += buy_shares
            elif target_shares < shares:
                sell_shares = shares - target_shares
                execution_price = open_price * (1.0 - self.config.slippage_rate)
                trade_value = sell_shares * execution_price
                commission = self._commission(trade_value)
                cash += trade_value - commission
                shares -= sell_shares

            close_price = float(row["close"])
            portfolio_value = cash + shares * close_price
            records.append(
                {
                    "date": pd.Timestamp(date),
                    "contribution": contribution,
                    "total_contributions": total_contributions,
                    "cash": cash,
                    "shares": shares,
                    "close": close_price,
                    "target_weight": target_weight,
                    "portfolio_value": portfolio_value,
                }
            )
            previous_month = current_month

        return pd.DataFrame.from_records(records).set_index("date")

    def _commission(self, trade_value: float) -> float:
        if trade_value <= 0:
            return 0.0
        return max(self.config.minimum_commission, trade_value * self.config.commission_rate)
