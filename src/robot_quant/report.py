"""模拟结果报告。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def write_outputs(history: pd.DataFrame, state: dict, output_root: Path) -> None:
    """写入CSV、状态JSON、Markdown和绩效曲线。"""

    data_dir = output_root / "data"
    reports_dir = output_root / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    export = history.reset_index(names="date")
    export["date"] = export["date"].dt.strftime("%Y-%m-%d")
    export.to_csv(data_dir / "portfolio_history.csv", index=False, float_format="%.8f")
    (data_dir / "latest_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_chart(history, state, reports_dir / "performance.png")
    (reports_dir / "latest.md").write_text(_markdown(state), encoding="utf-8")


def _write_chart(history: pd.DataFrame, state: dict, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(11, 6))
    if state["simulation_status"] == "pending":
        axis.text(
            0.5,
            0.5,
            (
                f"Simulation starts on {state['simulation_start_date']}\n"
                f"Initial purchase: CNY {state['initial_contribution']:,.0f}"
            ),
            ha="center",
            va="center",
            fontsize=18,
            transform=axis.transAxes,
        )
        axis.set_axis_off()
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return

    axis.plot(history.index, history["strategy_portfolio_value"], label="Prediction strategy")
    axis.plot(history.index, history["baseline_portfolio_value"], label="Fixed DCA")
    axis.plot(
        history.index,
        history["total_contributions"],
        label="Contributions",
        color="gray",
        linestyle="--",
    )
    axis.set_title("Robot ETF 159530: prediction strategy vs fixed DCA")
    axis.set_ylabel("CNY")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _markdown(state: dict) -> str:
    if state["simulation_status"] == "pending":
        return _pending_markdown(state)

    target_percent = state["next_target_weight"] * 100
    initial_target_percent = state["initial_target_weight"] * 100
    return f"""# 机器人ETF模拟账户日报

数据日期：**{state['market_date']}**

标的：机器人ETF易方达（159530）

最新收盘价：**¥{state['etf_close']:.4f}**

## 下一交易日预测

- 未来10个交易日跑赢沪深300的预测概率：**{state['prediction_probability']:.2%}**
- 建议目标仓位：**{target_percent:.0f}%**
- 当前使用模型：`{state['model_kind']}`

## 模拟结果

| 账户 | 累计投入 | 当前市值 | 净盈亏 | 资金收益率 | 最大回撤 |
|---|---:|---:|---:|---:|---:|
| 预测策略 | ¥{state['total_contributions']:,.2f} | ¥{state['strategy_value']:,.2f} | ¥{state['strategy_profit']:,.2f} | {state['strategy_roi']:.2%} | {state['strategy_max_drawdown']:.2%} |
| 固定定投 | ¥{state['total_contributions']:,.2f} | ¥{state['baseline_value']:,.2f} | ¥{state['baseline_profit']:,.2f} | {state['baseline_roi']:.2%} | {state['baseline_max_drawdown']:.2%} |

预测策略相对固定定投的市值差：**¥{state['strategy_value_difference']:,.2f}**

![累计市值对比](performance.png)

## 口径

- 从{state['simulation_start_date']}开始，首次买入¥{state['initial_contribution']:,.0f}。
- 首次建仓按{initial_target_percent:.0f}%目标仓位执行，其后按模型信号调整。
- 从次月开始，每月首个交易日追加¥{state['monthly_contribution']:,.0f}。
- 收盘后产生信号，下一交易日开盘执行。
- 只按100份整数交易，包含配置中的佣金与滑点。
- 这是模拟研究，不连接券商，也不构成收益承诺。
"""


def _pending_markdown(state: dict) -> str:
    model_target_percent = state["next_target_weight"] * 100
    initial_target_percent = state["initial_target_weight"] * 100
    return f"""# 机器人ETF模拟账户日报

数据日期：**{state['market_date']}**

标的：机器人ETF易方达（159530）

最新收盘价：**¥{state['etf_close']:.4f}**

## 模拟账户尚未开始

- 计划首次买入日期：**{state['simulation_start_date']}**
- 计划首次买入金额：**¥{state['initial_contribution']:,.0f}**
- 首次建仓：按{initial_target_percent:.0f}%目标仓位执行
- 后续定投：从次月开始，每月首个交易日投入¥{state['monthly_contribution']:,.0f}

截至当前市场日期，计划买入日尚未到达，因此累计投入、持仓和模拟收益均为0。

## 当前模型观察

- 未来10个交易日跑赢沪深300的预测概率：**{state['prediction_probability']:.2%}**
- 模型当前目标仓位：**{model_target_percent:.0f}%**
- 当前使用模型：`{state['model_kind']}`

首次买入按既定计划执行；完成初始建仓后，后续交易才按模型目标仓位调整。

## 风险说明

这是模拟研究，不连接券商，也不构成收益承诺。
"""
