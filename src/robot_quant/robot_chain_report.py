"""机器人产业链纸面账户输出。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def write_robot_chain_outputs(history: pd.DataFrame, state: dict, output_root: Path) -> None:
    """写入独立的个股纸面账户历史、状态、日报和曲线。"""

    data_dir = output_root / "data"
    reports_dir = output_root / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    export = history.reset_index(names="date")
    if not export.empty:
        export["date"] = export["date"].dt.strftime("%Y-%m-%d")
    export.to_csv(data_dir / "robot_chain_history.csv", index=False, float_format="%.8f")
    (data_dir / "robot_chain_latest_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_chart(history, state, reports_dir / "robot_chain_performance.png")
    (reports_dir / "robot_chain_latest.md").write_text(_markdown(state), encoding="utf-8")


def _write_chart(history: pd.DataFrame, state: dict, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(11, 6))
    if history.empty:
        axis.text(
            0.5,
            0.5,
            f"Simulation starts on {state['simulation_start_date']}",
            ha="center",
            va="center",
            fontsize=18,
            transform=axis.transAxes,
        )
        axis.set_axis_off()
    else:
        axis.plot(history.index, history["portfolio_value"], label="Robot-chain paper account")
        axis.axhline(state["initial_capital"], label="Initial capital", color="gray", linestyle="--")
        axis.set_title("Robot industry-chain paper account")
        axis.set_ylabel("CNY")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _markdown(state: dict) -> str:
    rows = "\n".join(
        (
            f"| {holding['code']} | {holding['name']} | {holding['chain_stage']} | "
            f"{holding['shares']} | {_currency(holding['close'])} | "
            f"¥{holding['market_value']:,.2f} | "
            f"{'是' if holding['above_moving_average'] else '否'} |"
        )
        for holding in state["holdings"]
    )
    rules = state["risk_rules"]
    return f"""# 机器人产业链个股纸面账户日报

数据日期：**{state['market_date']}**

状态：**{'运行中' if state['simulation_status'] == 'active' else '等待建仓'}**；只做模拟，不连接券商。

## 当日盈亏

| 初始资金 | 当前市值 | 累计盈亏 | 收益率 | 最大回撤 | 现金 |
|---:|---:|---:|---:|---:|---:|
| ¥{state['initial_capital']:,.2f} | ¥{state['portfolio_value']:,.2f} | ¥{state['profit']:,.2f} | {state['roi']:.2%} | {state['max_drawdown']:.2%} | ¥{state['cash']:,.2f} |

- 股票市值：¥{state['equity_value']:,.2f}（{state['equity_weight']:.2%}）
- 累计佣金：¥{state['cumulative_commission']:,.2f}；累计滑点：¥{state['cumulative_slippage']:,.2f}
- 今日模拟动作：**{state['actions']}**

## 当前持仓

| 代码 | 公司 | 产业链 | 股数 | 收盘价 | 市值 | 高于20日均线 |
|---:|---|---|---:|---:|---:|---|
{rows}

## 风险规则

- 连续{rules['consecutive_below_days']}日低于{rules['moving_average_days']}日均线：下一交易日模拟减半。
- 组合回撤达到{rules['reduce_drawdown']:.0%}：下一交易日将股票仓位降至{rules['reduced_max_equity_weight']:.0%}以内。
- 组合回撤达到{rules['exit_drawdown']:.0%}：下一交易日全数转为现金观察。

## 口径与限制

- 这是用腾讯前复权日线、按收盘信号和次日开盘模拟执行的账户；首日按收盘价建仓。
- 该账户没有通过样本外 Alpha 验证，风控规则仅用于纸面复盘，不构成真实买卖建议。
- {state['limitations']}

![纸面账户净值](robot_chain_performance.png)
"""


def _currency(value: float | None) -> str:
    return "-" if value is None else f"¥{value:,.2f}"
