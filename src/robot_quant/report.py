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

数据日期：**{state["market_date"]}**

标的：机器人ETF易方达（159530）

最新收盘价：**¥{state["etf_close"]:.4f}**

## 下一交易日预测

{_probability_markdown(state)}
- 建议目标仓位：**{target_percent:.0f}%**
- 当前使用模型：`{state["model_kind"]}`

{_indicator_markdown(state)}

## 模拟结果

| 账户 | 累计投入 | 当前市值 | 净盈亏 | 资金收益率 | 最大回撤 |
|---|---:|---:|---:|---:|---:|
| 预测策略 | ¥{state["total_contributions"]:,.2f} | ¥{state["strategy_value"]:,.2f} | ¥{state["strategy_profit"]:,.2f} | {state["strategy_roi"]:.2%} | {state["strategy_max_drawdown"]:.2%} |
| 固定定投 | ¥{state["total_contributions"]:,.2f} | ¥{state["baseline_value"]:,.2f} | ¥{state["baseline_profit"]:,.2f} | {state["baseline_roi"]:.2%} | {state["baseline_max_drawdown"]:.2%} |

预测策略相对固定定投的市值差：**¥{state["strategy_value_difference"]:,.2f}**

![累计市值对比](performance.png)

## 口径

- 从{state["simulation_start_date"]}开始，首次买入¥{state["initial_contribution"]:,.0f}。
- 首次建仓按{initial_target_percent:.0f}%目标仓位执行，其后按模型信号调整。
- 从次月开始，每月首个交易日追加¥{state["monthly_contribution"]:,.0f}。
- 收盘后产生信号，下一交易日开盘执行。
- 只按100份整数交易，包含配置中的佣金与滑点。
- 这是模拟研究，不连接券商，也不构成收益承诺。
"""


def _pending_markdown(state: dict) -> str:
    model_target_percent = state["next_target_weight"] * 100
    initial_target_percent = state["initial_target_weight"] * 100
    return f"""# 机器人ETF模拟账户日报

数据日期：**{state["market_date"]}**

标的：机器人ETF易方达（159530）

最新收盘价：**¥{state["etf_close"]:.4f}**

## 模拟账户尚未开始

- 计划首次买入日期：**{state["simulation_start_date"]}**
- 计划首次买入金额：**¥{state["initial_contribution"]:,.0f}**
- 首次建仓：按{initial_target_percent:.0f}%目标仓位执行
- 后续定投：从次月开始，每月首个交易日投入¥{state["monthly_contribution"]:,.0f}

截至当前市场日期，计划买入日尚未到达，因此累计投入、持仓和模拟收益均为0。

## 当前模型观察

{_probability_markdown(state)}
- 模型当前目标仓位：**{model_target_percent:.0f}%**
- 当前使用模型：`{state["model_kind"]}`

首次买入按既定计划执行；完成初始建仓后，后续交易才按模型目标仓位调整。

{_indicator_markdown(state)}

## 风险说明

这是模拟研究，不连接券商，也不构成收益承诺。
"""


def _indicator_markdown(state: dict) -> str:
    forecasts = state["forecast_horizons"]
    validation = state["model_validation"]
    sell = state["sell_indicators"]
    sell_validation = state["sell_rule_validation"]
    action_labels = {
        "hold": "保持",
        "watch": "观察",
        "reduce": "减仓",
        "exit": "退出",
        "unavailable": "不可用",
    }
    forecast_rows = "\n".join(
        (
            f"| {horizon}日 | {forecast['sample_count']} | "
            f"{_evidence_label(forecast['evidence_status'])} | "
            f"{_percent(forecast['median_return'])} | "
            f"{_percent_range(forecast['return_p10'], forecast['return_p90'])} | "
            f"{_percent(forecast['loss_probability'])} | "
            f"{_percent(forecast['drawdown_event_probability'])} | "
            f"{_percent(forecast['drawdown_5pct_probability'])} | "
            f"{_drawdown_window(forecast)} | "
            f"{_currency(forecast['expected_price'], 4)} | "
            f"{_currency(forecast['expected_profit_on_capital'], 0)} |"
        )
        for horizon, forecast in forecasts.items()
    )
    validation_rows = "\n".join(
        (
            f"| {horizon}日 | {forecast['validation_sample_count']} | "
            f"{_percent(forecast['validation_mae'])} | "
            f"{_percent(forecast['validation_zero_baseline_mae'])} | "
            f"{_percent(forecast['validation_direction_accuracy'])} | "
            f"{_percent(forecast['validation_interval_coverage'])} |"
        )
        for horizon, forecast in forecasts.items()
    )
    status_labels = {
        "insufficient_samples": "样本不足",
        "observation_only": "仅观察",
        "provisional": "初步可用",
        "validated": "已通过当前验证门槛",
    }
    status = status_labels.get(validation["status"], validation["status"])
    action = action_labels.get(sell["action"], sell["action"])
    sell_validation_rows = "\n".join(
        (
            f"| {action_labels.get(historical_action, historical_action)} | "
            f"{evidence['sample_count']} | "
            f"{evidence['actual_median_return_10']:.2%} | "
            f"{evidence['actual_mean_return_10']:.2%} | "
            f"{evidence['actual_loss_probability_10']:.2%} | "
            f"{evidence['actual_mean_worst_return_10']:.2%} | "
            f"{evidence['mean_return_difference_vs_all']:+.2%} |"
        )
        for historical_action, evidence in sell_validation["actions"].items()
    )
    if not sell_validation_rows:
        sell_validation_rows = "| 暂无足够样本 | 0 | 不可用 | 不可用 | 不可用 | 不可用 | 不可用 |"
    baseline = sell_validation.get("all_dates_baseline")
    baseline_summary = (
        "暂无足够样本。"
        if baseline is None
        else (
            f"全部{baseline['sample_count']}个可验证日期的10日平均收益为"
            f"{baseline['actual_mean_return_10']:.2%}、亏损率为"
            f"{baseline['actual_loss_probability_10']:.2%}、路径平均最差收益为"
            f"{baseline['actual_mean_worst_return_10']:.2%}。"
        )
    )
    model_accuracy_interval = (
        f"{_percent(validation['accuracy_wilson95_low'])} ～ "
        f"{_percent(validation['accuracy_wilson95_high'])}"
    )
    return f"""## 预估收益与下跌风险

以下区间来自与当前信号、趋势和波动率最接近的历史状态，最多取40个样本；它是条件分布，不是确定收益。

| 持有期 | 相似样本 | 证据状态 | 中位收益 | 10%～90%收益区间 | 期末亏损概率 | 期间跌破现价概率 | 期间跌超5%概率 | 跌破现价后的低点窗口 | 中位预估价 | ¥{state["initial_contribution"]:,.0f}中位盈亏 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
{forecast_rows}

“什么时候会掉”不能精确到某一天；低点窗口只对曾跌破起始价格的历史路径统计，并给出低点日的25%～75%分位及条件样本数，不混入全程上涨路径。每天收盘后重算，优先看亏损概率和收益下界是否同步恶化。

## 模拟卖出指标

- 当前动作：**{action}**；原因：{sell["reason"]}。
- 下次复核：**{sell["review_horizon_trading_days"]}个交易日内**，并在每日收盘数据更新后提前复核。
- 10日风险控制参考价：**{_currency(sell["risk_control_price"], 4)}**（相似样本收益10%分位对应价）。
- 10日止盈观察参考价：**{_currency(sell["take_profit_price"], 4)}**（相似样本收益90%分位对应价）。
- 规则阈值：校准概率低于{sell["probability_reduce_trigger"]:.0%}进入减仓判断；低于{sell["probability_exit_trigger"]:.0%}且10日中位收益为负、亏损概率不低于{sell["exit_loss_probability_trigger"]:.0%}时退出；中位收益为负且亏损概率不低于{sell["reduce_loss_probability_trigger"]:.0%}时减仓；亏损概率不低于{sell["watch_loss_probability_trigger"]:.0%}时至少保持观察。

参考价只用于模拟触发，不是挂单建议；是否卖出还需等收盘信号、趋势和概率共同确认。

### 卖出规则历史验证

规则在**{sell_validation["sample_count"]}**个滚动样本外日期上可计算；每次只使用该日以前已经实现的相似结果。

{baseline_summary}

| 当时动作 | 触发次数 | 随后10日实际中位收益 | 随后10日实际平均收益 | 随后10日实际亏损率 | 路径平均最差收益 | 平均收益较全部日期 |
|---|---:|---:|---:|---:|---:|---:|
{sell_validation_rows}

这些阈值是预先设定的工程风险线，没有用本批数据调参。“较全部日期”只是状态分组差，不是执行规则带来的因果收益。10日窗口彼此重叠，触发次数也不等于独立交易次数；若减仓或退出样本很少，该动作仍没有足够证据。

## 样本外验证

模型证据状态：**{status}**。

- 已实现的校准预测样本：**{validation["calibrated_sample_count"]}**。
- 全部重叠日标签的描述性方向准确率：**{_percent(validation["direction_accuracy"])}**。
- 每隔10个交易日抽取一次的近似非重叠样本：**{validation["confidence_sample_count"]}**；方向准确率：**{_percent(validation["confidence_direction_accuracy"])}**；Wilson 95%区间：**{model_accuracy_interval}**。
- 校准后Brier分数：**{_decimal(validation["calibrated_brier"])}**；原始模型：**{_decimal(validation["raw_brier"])}**；固定50%概率基线：**{validation["constant_50_brier"]:.4f}**。Brier越低越好。
- ROC AUC：**{_decimal(validation["roc_auc"])}**；0.5约等于随机排序。

| 预测期 | 验证样本 | 中位预测MAE | 零收益基线MAE | 方向准确率 | 10%～90%区间覆盖率 |
|---|---:|---:|---:|---:|---:|
{validation_rows}

相似样本验证严格只使用每个预测日当时已经实现的历史结果。若中位预测MAE没有低于零收益基线，或模型Brier没有低于0.25，本指标只作为风险观察，不能作为独立买卖依据。"""


def _probability_markdown(state: dict) -> str:
    model_kind = state["model_kind"]
    if model_kind == "calibrated_logistic_regression":
        return (
            f"- 逻辑回归原始分数：**{state['raw_prediction_probability']:.2%}**\n"
            "- 经时间序列隔离校准后，未来10个交易日跑赢沪深300的预测概率："
            f"**{state['prediction_probability']:.2%}**"
        )
    if model_kind == "logistic_regression_uncalibrated":
        return (
            f"- 逻辑回归原始分数：**{state['raw_prediction_probability']:.2%}**\n"
            "- 当日校准不可用，不展示校准概率。"
        )
    return (
        f"- 趋势回退分数：**{state['prediction_probability']:.2%}**\n"
        "- 当日逻辑回归及概率校准不可用，不把回退分数解释为真实概率。"
    )


def _percent(value: float | None) -> str:
    return "不可用" if value is None else f"{value:.2%}"


def _decimal(value: float | None) -> str:
    return "不可用" if value is None else f"{value:.4f}"


def _currency(value: float | None, decimals: int) -> str:
    return "不可用" if value is None else f"¥{value:,.{decimals}f}"


def _percent_range(lower: float | None, upper: float | None) -> str:
    if lower is None or upper is None:
        return "不可用"
    return f"{lower:.2%} ～ {upper:.2%}"


def _evidence_label(status: str) -> str:
    return {
        "sufficient": "可描述",
        "insufficient": "样本不足",
        "unavailable": "暂无足够样本",
    }.get(status, status)


def _drawdown_window(forecast: dict) -> str:
    count = forecast["drawdown_event_sample_count"]
    lower = forecast["drawdown_trough_day_p25"]
    upper = forecast["drawdown_trough_day_p75"]
    if not count or lower is None or upper is None:
        return "不可用"
    return f"第{lower:.0f}～{upper:.0f}日（n={count}）"
