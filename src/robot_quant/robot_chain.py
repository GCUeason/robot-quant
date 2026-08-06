"""机器人产业链个股纸面账户。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import math

import pandas as pd

from robot_quant.data import TencentDataSource


@dataclass(frozen=True)
class RobotChainHolding:
    """纸面账户中的单个机器人产业链标的。"""

    code: str
    symbol: str
    name: str
    chain_stage: str
    subsector: str
    initial_shares: int


@dataclass(frozen=True)
class RobotChainPlan:
    """纸面账户的固定资金、交易成本与风险覆盖规则。"""

    start_date: pd.Timestamp
    initial_capital: float
    commission_rate: float = 0.0003
    minimum_commission: float = 5.0
    slippage_rate: float = 0.0005
    moving_average_days: int = 20
    consecutive_below_days: int = 3
    reduce_drawdown: float = -0.08
    exit_drawdown: float = -0.12
    reduced_max_equity_weight: float = 0.35


ROBOT_CHAIN_HOLDINGS = (
    RobotChainHolding("002472", "sz002472", "双环传动", "上游", "RV减速器", 100),
    RobotChainHolding("603728", "sh603728", "鸣志电器", "上游", "电机/驱动", 100),
    RobotChainHolding("002747", "sz002747", "埃斯顿", "中游", "工业机器人本体", 100),
    RobotChainHolding("300024", "sz300024", "机器人", "中游", "本体/系统集成", 300),
    RobotChainHolding("002698", "sz002698", "博实股份", "下游", "工业自动化应用", 300),
    RobotChainHolding("603486", "sh603486", "科沃斯", "下游", "服务机器人", 100),
)

ROBOT_CHAIN_PLAN = RobotChainPlan(
    start_date=pd.Timestamp("2026-08-06"),
    initial_capital=50_000.0,
)


def fetch_robot_chain_prices(
    plan: RobotChainPlan = ROBOT_CHAIN_PLAN,
    holdings: tuple[RobotChainHolding, ...] = ROBOT_CHAIN_HOLDINGS,
    end_date: date | None = None,
) -> dict[str, pd.DataFrame]:
    """读取纸面账户所需行情，并用最近一周请求覆盖缓存滞后的日线。"""

    final_date = end_date or date.today()
    history_start = (plan.start_date - pd.Timedelta(days=45)).date()
    fresh_start = max(history_start, final_date - timedelta(days=7))
    source = TencentDataSource()
    prices: dict[str, pd.DataFrame] = {}

    for holding in holdings:
        history = source.fetch_daily(
            holding.symbol,
            start_date=history_start,
            end_date=final_date,
        )
        fresh = source.fetch_daily(
            holding.symbol,
            start_date=fresh_start,
            end_date=final_date,
        )
        merged = pd.concat([history, fresh]).sort_index()
        prices[holding.code] = merged.loc[~merged.index.duplicated(keep="last")]
    return prices


def simulate_robot_chain(
    prices: dict[str, pd.DataFrame],
    plan: RobotChainPlan = ROBOT_CHAIN_PLAN,
    holdings: tuple[RobotChainHolding, ...] = ROBOT_CHAIN_HOLDINGS,
) -> tuple[pd.DataFrame, dict]:
    """按已知收盘信号、次日开盘执行的口径重算纸面账户。"""

    _validate_prices(prices, holdings)
    common_dates = _common_dates(prices, holdings)
    trading_dates = common_dates[common_dates >= plan.start_date]
    if trading_dates.empty:
        return _empty_history(holdings), _pending_state(common_dates, plan, holdings)

    signals = _build_signals(prices, holdings, plan)
    cash = plan.initial_capital
    shares = {holding.code: 0 for holding in holdings}
    cumulative_commission = 0.0
    cumulative_slippage = 0.0
    records: list[dict[str, float | int | str | bool | pd.Timestamp]] = []

    for position, current_date in enumerate(trading_dates):
        day_prices = {holding.code: prices[holding.code].loc[current_date] for holding in holdings}
        actions: list[str] = []
        if position == 0:
            cash, commission, slippage = _buy_initial_positions(
                cash,
                shares,
                day_prices,
                plan,
                holdings,
            )
            cumulative_commission += commission
            cumulative_slippage += slippage
            actions.append("按初始计划建仓")
        else:
            previous = records[-1]
            cash, commission, slippage, actions = _apply_risk_actions(
                cash,
                shares,
                day_prices,
                previous,
                plan,
                holdings,
            )
            cumulative_commission += commission
            cumulative_slippage += slippage

        record = _mark_to_market(
            current_date,
            cash,
            shares,
            day_prices,
            signals,
            plan,
            holdings,
            plan.initial_capital,
            records,
            cumulative_commission,
            cumulative_slippage,
            actions,
        )
        records.append(record)

    history = pd.DataFrame.from_records(records).set_index("date")
    return history, _latest_state(history, plan, holdings)


def _validate_prices(
    prices: dict[str, pd.DataFrame],
    holdings: tuple[RobotChainHolding, ...],
) -> None:
    required = {"open", "close"}
    for holding in holdings:
        frame = prices.get(holding.code)
        if frame is None:
            raise ValueError(f"缺少{holding.code}行情")
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{holding.code}行情缺少列: {', '.join(sorted(missing))}")
        if frame.empty:
            raise ValueError(f"{holding.code}行情为空")


def _common_dates(
    prices: dict[str, pd.DataFrame],
    holdings: tuple[RobotChainHolding, ...],
) -> pd.DatetimeIndex:
    dates: pd.DatetimeIndex | None = None
    for holding in holdings:
        index = pd.DatetimeIndex(prices[holding.code].index)
        dates = index if dates is None else dates.intersection(index)
    if dates is None or dates.empty:
        raise ValueError("机器人产业链标的没有共同交易日")
    return dates.sort_values()


def _build_signals(
    prices: dict[str, pd.DataFrame],
    holdings: tuple[RobotChainHolding, ...],
    plan: RobotChainPlan,
) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for holding in holdings:
        close = prices[holding.code]["close"].astype(float).sort_index()
        moving_average = close.rolling(plan.moving_average_days, min_periods=plan.moving_average_days).mean()
        below = close.lt(moving_average).fillna(False)
        result[holding.code] = pd.DataFrame(
            {
                "moving_average": moving_average,
                "above_moving_average": close.gt(moving_average).fillna(False),
                "consecutive_below": (
                    below.rolling(plan.consecutive_below_days).sum().eq(plan.consecutive_below_days)
                ).fillna(False),
            }
        )
    return result


def _buy_initial_positions(
    cash: float,
    shares: dict[str, int],
    day_prices: dict[str, pd.Series],
    plan: RobotChainPlan,
    holdings: tuple[RobotChainHolding, ...],
) -> tuple[float, float, float]:
    commission_total = 0.0
    slippage_total = 0.0
    for holding in holdings:
        close_price = float(day_prices[holding.code]["close"])
        execution_price = close_price * (1.0 + plan.slippage_rate)
        trade_value = holding.initial_shares * execution_price
        commission = _commission(trade_value, plan)
        if trade_value + commission > cash:
            raise ValueError(f"初始资金不足以买入{holding.code}")
        cash -= trade_value + commission
        shares[holding.code] = holding.initial_shares
        commission_total += commission
        slippage_total += holding.initial_shares * (execution_price - close_price)
    return cash, commission_total, slippage_total


def _apply_risk_actions(
    cash: float,
    shares: dict[str, int],
    day_prices: dict[str, pd.Series],
    previous: dict[str, float | int | str | bool | pd.Timestamp],
    plan: RobotChainPlan,
    holdings: tuple[RobotChainHolding, ...],
) -> tuple[float, float, float, list[str]]:
    previous_drawdown = float(previous["drawdown"])
    target_equity_weight = _target_equity_weight(previous_drawdown, plan)
    open_prices = {holding.code: float(day_prices[holding.code]["open"]) for holding in holdings}
    open_equity_value = sum(shares[holding.code] * open_prices[holding.code] for holding in holdings)
    open_portfolio_value = cash + open_equity_value
    risk_ratio = 1.0
    actions: list[str] = []

    if target_equity_weight == 0.0:
        risk_ratio = 0.0
        actions.append("组合回撤达到12%，全数转为现金")
    elif target_equity_weight is not None and open_equity_value > 0:
        risk_ratio = min(1.0, open_portfolio_value * target_equity_weight / open_equity_value)
        if risk_ratio < 1.0:
            actions.append("组合回撤达到8%，股票仓位降至35%")

    commission_total = 0.0
    slippage_total = 0.0
    for holding in holdings:
        current_shares = shares[holding.code]
        target_shares = math.floor(current_shares * risk_ratio)
        if bool(previous[f"{holding.code}_consecutive_below"]):
            target_shares = min(target_shares, math.floor(current_shares / 2))
            actions.append(f"{holding.code}连续3日跌破20日均线，减半")
        if target_shares >= current_shares:
            continue

        sell_shares = current_shares - target_shares
        open_price = open_prices[holding.code]
        execution_price = open_price * (1.0 - plan.slippage_rate)
        trade_value = sell_shares * execution_price
        commission = _commission(trade_value, plan)
        cash += trade_value - commission
        shares[holding.code] = target_shares
        commission_total += commission
        slippage_total += sell_shares * (open_price - execution_price)

    return cash, commission_total, slippage_total, actions


def _target_equity_weight(drawdown: float, plan: RobotChainPlan) -> float | None:
    if drawdown <= plan.exit_drawdown:
        return 0.0
    if drawdown <= plan.reduce_drawdown:
        return plan.reduced_max_equity_weight
    return None


def _mark_to_market(
    current_date: pd.Timestamp,
    cash: float,
    shares: dict[str, int],
    day_prices: dict[str, pd.Series],
    signals: dict[str, pd.DataFrame],
    plan: RobotChainPlan,
    holdings: tuple[RobotChainHolding, ...],
    initial_capital: float,
    records: list[dict[str, float | int | str | bool | pd.Timestamp]],
    cumulative_commission: float,
    cumulative_slippage: float,
    actions: list[str],
) -> dict[str, float | int | str | bool | pd.Timestamp]:
    equity_value = 0.0
    record: dict[str, float | int | str | bool | pd.Timestamp] = {"date": pd.Timestamp(current_date)}
    for holding in holdings:
        close_price = float(day_prices[holding.code]["close"])
        value = shares[holding.code] * close_price
        indicator = signals[holding.code].loc[current_date]
        equity_value += value
        record.update(
            {
                f"{holding.code}_shares": shares[holding.code],
                f"{holding.code}_close": close_price,
                f"{holding.code}_value": value,
                f"{holding.code}_ma{plan.moving_average_days}": _optional_float(
                    indicator["moving_average"]
                ),
                f"{holding.code}_above_ma{plan.moving_average_days}": bool(
                    indicator["above_moving_average"]
                ),
                f"{holding.code}_consecutive_below": bool(indicator["consecutive_below"]),
            }
        )

    portfolio_value = cash + equity_value
    previous_peak = initial_capital if not records else float(records[-1]["peak_value"])
    peak_value = max(previous_peak, portfolio_value)
    record.update(
        {
            "cash": cash,
            "equity_value": equity_value,
            "equity_weight": equity_value / portfolio_value if portfolio_value else 0.0,
            "portfolio_value": portfolio_value,
            "profit": portfolio_value - initial_capital,
            "roi": portfolio_value / initial_capital - 1.0,
            "peak_value": peak_value,
            "drawdown": portfolio_value / peak_value - 1.0,
            "cumulative_commission": cumulative_commission,
            "cumulative_slippage": cumulative_slippage,
            "actions": "；".join(actions) if actions else "持有",
        }
    )
    return record


def _latest_state(
    history: pd.DataFrame,
    plan: RobotChainPlan,
    holdings: tuple[RobotChainHolding, ...],
) -> dict:
    latest = history.iloc[-1]
    holding_rows = []
    for holding in holdings:
        holding_rows.append(
            {
                "code": holding.code,
                "name": holding.name,
                "chain_stage": holding.chain_stage,
                "subsector": holding.subsector,
                "shares": int(latest[f"{holding.code}_shares"]),
                "close": float(latest[f"{holding.code}_close"]),
                "market_value": float(latest[f"{holding.code}_value"]),
                "moving_average": _optional_float(
                    latest[f"{holding.code}_ma{plan.moving_average_days}"]
                ),
                "above_moving_average": bool(latest[f"{holding.code}_above_ma{plan.moving_average_days}"]),
                "consecutive_below": bool(latest[f"{holding.code}_consecutive_below"]),
            }
        )
    return {
        "simulation_status": "active",
        "simulation_start_date": plan.start_date.strftime("%Y-%m-%d"),
        "market_date": history.index[-1].strftime("%Y-%m-%d"),
        "execution_policy": "cash_hedged_robot_chain_paper",
        "automated_trading_enabled": False,
        "initial_capital": plan.initial_capital,
        "portfolio_value": float(latest["portfolio_value"]),
        "profit": float(latest["profit"]),
        "roi": float(latest["roi"]),
        "max_drawdown": float(history["drawdown"].min()),
        "cash": float(latest["cash"]),
        "equity_value": float(latest["equity_value"]),
        "equity_weight": float(latest["equity_weight"]),
        "cumulative_commission": float(latest["cumulative_commission"]),
        "cumulative_slippage": float(latest["cumulative_slippage"]),
        "actions": str(latest["actions"]),
        "holdings": holding_rows,
        "risk_rules": {
            "moving_average_days": plan.moving_average_days,
            "consecutive_below_days": plan.consecutive_below_days,
            "reduce_drawdown": plan.reduce_drawdown,
            "exit_drawdown": plan.exit_drawdown,
            "reduced_max_equity_weight": plan.reduced_max_equity_weight,
        },
        "data_provenance": "tencent_qfq_individual_stocks",
        "limitations": "纸面账户，未验证Alpha，不连接券商或自动下单。",
    }


def _pending_state(
    common_dates: pd.DatetimeIndex,
    plan: RobotChainPlan,
    holdings: tuple[RobotChainHolding, ...],
) -> dict:
    return {
        "simulation_status": "pending",
        "simulation_start_date": plan.start_date.strftime("%Y-%m-%d"),
        "market_date": common_dates[-1].strftime("%Y-%m-%d"),
        "execution_policy": "cash_hedged_robot_chain_paper",
        "automated_trading_enabled": False,
        "initial_capital": plan.initial_capital,
        "portfolio_value": plan.initial_capital,
        "profit": 0.0,
        "roi": 0.0,
        "max_drawdown": 0.0,
        "cash": plan.initial_capital,
        "equity_value": 0.0,
        "equity_weight": 0.0,
        "cumulative_commission": 0.0,
        "cumulative_slippage": 0.0,
        "actions": "尚未到模拟建仓日",
        "holdings": [
            {
                "code": holding.code,
                "name": holding.name,
                "chain_stage": holding.chain_stage,
                "subsector": holding.subsector,
                "shares": 0,
                "close": None,
                "market_value": 0.0,
                "moving_average": None,
                "above_moving_average": False,
                "consecutive_below": False,
            }
            for holding in holdings
        ],
        "risk_rules": {
            "moving_average_days": plan.moving_average_days,
            "consecutive_below_days": plan.consecutive_below_days,
            "reduce_drawdown": plan.reduce_drawdown,
            "exit_drawdown": plan.exit_drawdown,
            "reduced_max_equity_weight": plan.reduced_max_equity_weight,
        },
        "data_provenance": "tencent_qfq_individual_stocks",
        "limitations": "纸面账户，未验证Alpha，不连接券商或自动下单。",
    }


def _empty_history(holdings: tuple[RobotChainHolding, ...]) -> pd.DataFrame:
    columns = [
        "cash",
        "equity_value",
        "equity_weight",
        "portfolio_value",
        "profit",
        "roi",
        "peak_value",
        "drawdown",
        "cumulative_commission",
        "cumulative_slippage",
        "actions",
    ]
    for holding in holdings:
        columns.extend(
            [
                f"{holding.code}_shares",
                f"{holding.code}_close",
                f"{holding.code}_value",
                f"{holding.code}_ma20",
                f"{holding.code}_above_ma20",
                f"{holding.code}_consecutive_below",
            ]
        )
    return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], name="date"))


def _commission(trade_value: float, plan: RobotChainPlan) -> float:
    return max(plan.minimum_commission, trade_value * plan.commission_rate)


def _optional_float(value: float) -> float | None:
    return None if pd.isna(value) else float(value)
