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

    axis.plot(history.index, history["strategy_portfolio_value"], label="Quality-gated DCA")
    axis.plot(history.index, history["baseline_portfolio_value"], label="Fixed DCA")
    if "shadow_portfolio_value" in history:
        axis.plot(
            history.index,
            history["shadow_portfolio_value"],
            label="Research shadow (not executable)",
        )
    axis.plot(
        history.index,
        history["total_contributions"],
        label="Contributions",
        color="gray",
        linestyle="--",
    )
    axis.set_title("Robot ETF 159530: quality-gated plan vs fixed DCA")
    axis.set_ylabel("CNY")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _markdown(state: dict) -> str:
    if state["simulation_status"] == "pending":
        return _pending_markdown(state)

    model_selection = _model_selection_label(state["model_selection_status"])
    target_percent = state["executable_target_weight"] * 100
    research_target_percent = state["research_target_weight"] * 100
    initial_target_percent = state["initial_target_weight"] * 100
    return f"""# 机器人ETF模拟账户日报

数据日期：**{state["market_date"]}**

标的：机器人ETF易方达（159530）

最新收盘价：**¥{state["etf_close"]:.4f}**

## 未来10日相对沪深300研究信号

{_probability_markdown(state)}
- 模型原始研究仓位：**{research_target_percent:.0f}%**（不可执行）
- 当前使用模型：`{state["model_kind"]}`
- 当前状态：{_signal_status(state)}

## 下一交易日执行计划

- 可执行目标仓位：**{target_percent:.0f}%**
- 执行政策：**固定定投是唯一可执行模拟**；模型尚未获得仓位控制权。
- 门控原因：{_gate_reasons(state)}
- 逐日成熟样本门控：{_quality_gate_summary(state)}
- 不可执行影子目标仓位：**{state["shadow_target_weight"]:.0%}**。

## 基线优先门控

- 候选模型：**{model_selection}** `{state["model_version"]}`；它是在查看既有历史后提出的候选，只能用未来新增数据继续验证。
- 模型研究概率：**{state["prediction_probability"]:.2%}**；批准概率固定为**{state["approved_probability"]:.2%}**，表示当前没有可执行优势。
- 本月训练集内部隔离验证：样本{state["training_oof_sample_count"]}条，Brier={_decimal(state["training_oof_brier"])}，AUC={_decimal(state["training_oof_auc"])}，**{"通过" if state["training_oof_passed"] else "未通过"}**。
- 策略结论：**{state["strategy_validation"]["reason"]}**。

{_indicator_markdown(state)}

## 模拟结果

| 账户 | 累计投入 | 当前市值 | 净盈亏 | 资金收益率 | 最大回撤 |
|---|---:|---:|---:|---:|---:|
| 质量门控定投 | ¥{state["total_contributions"]:,.2f} | ¥{state["strategy_value"]:,.2f} | ¥{state["strategy_profit"]:,.2f} | {state["strategy_roi"]:.2%} | {state["strategy_max_drawdown"]:.2%} |
| 固定定投 | ¥{state["total_contributions"]:,.2f} | ¥{state["baseline_value"]:,.2f} | ¥{state["baseline_profit"]:,.2f} | {state["baseline_roi"]:.2%} | {state["baseline_max_drawdown"]:.2%} |
| 质量门控影子策略（不可执行） | ¥{state["total_contributions"]:,.2f} | ¥{state["shadow_value"]:,.2f} | ¥{state["shadow_profit"]:,.2f} | {state["shadow_roi"]:.2%} | {state["shadow_max_drawdown"]:.2%} |

质量门控定投相对固定定投的市值差：**¥{state["strategy_value_difference"]:,.2f}**

影子Alpha（影子策略相对固定定投的市值差）：**¥{state["shadow_value_difference"]:,.2f}**。样本仅**{state["shadow_validation"]["sample_days"]}**个交易日，门控通过{state["shadow_gate_pass_days"]}日、实际持仓{state["shadow_active_days"]}日，当前只用于检验规则是否可能创造避险或超额收益。

![累计市值对比](performance.png)

## 数据口径与来源

- 模型特征与收益验证：`{state["data_provenance"]["model_features"]}`，覆盖{state["data_provenance"]["robot_index_start"]}至{state["data_provenance"]["robot_index_end"]}，共{state["data_provenance"]["robot_index_sample_count"]}条。
- ETF交易模拟：`{state["data_provenance"]["etf"]}`。
- 市场基准：`{state["data_provenance"]["benchmark"]}`。
- 预估收益验证使用机器人指数，ETF价格只用于账户和参考价换算；两者的跟踪偏差仍是限制。
- 已知制度断点：{_methodology_breaks(state)}。长历史提高样本量，但不能视为完全同分布数据。

## 口径

- 从{state["simulation_start_date"]}开始，首次买入¥{state["initial_contribution"]:,.0f}。
- 首次建仓按{initial_target_percent:.0f}%目标仓位执行；模型保持纯观察层，其后继续固定定投。
- 从次月开始，每月首个交易日追加¥{state["monthly_contribution"]:,.0f}。
- 研究信号于收盘后更新，仅记录不执行；定投按既定现金流规则在对应交易日执行。
- 只按100份整数交易，包含配置中的佣金与滑点。
- 这是模拟研究，不连接券商，也不构成收益承诺。
"""


def _pending_markdown(state: dict) -> str:
    model_selection = _model_selection_label(state["model_selection_status"])
    model_target_percent = state["research_target_weight"] * 100
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

## 未来10日相对沪深300研究信号

{_probability_markdown(state)}
- 模型原始研究仓位：**{model_target_percent:.0f}%**（不可执行）
- 当前使用模型：`{state["model_kind"]}`
- 当前状态：{_signal_status(state)}

## 下一交易日执行计划

- 固定定投是唯一可执行模拟；模型不调整仓位。
- 门控原因：{_gate_reasons(state)}
- 逐日成熟样本门控：{_quality_gate_summary(state)}
- 不可执行影子目标仓位：**{state["shadow_target_weight"]:.0%}**。

## 基线优先门控

- 候选模型：**{model_selection}** `{state["model_version"]}`；只能用未来新增数据继续验证。
- 模型研究概率：**{state["prediction_probability"]:.2%}**；批准概率固定为**{state["approved_probability"]:.2%}**。
- 本月训练集内部隔离验证：样本{state["training_oof_sample_count"]}条，Brier={_decimal(state["training_oof_brier"])}，AUC={_decimal(state["training_oof_auc"])}，**{"通过" if state["training_oof_passed"] else "未通过"}**。
- 策略结论：**{state["strategy_validation"]["reason"]}**。

首次买入按既定计划执行；后续仍按固定定投计划运行。

{_indicator_markdown(state)}

## 风险说明

这是模拟研究，不连接券商，也不构成收益承诺。
"""


def _indicator_markdown(state: dict) -> str:
    forecasts = state["forecast_horizons"]
    references = state.get("historical_risk_reference", {})
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
    forecast_gate_rows = "\n".join(
        (
            f"| {horizon}日 | {forecast['sample_count']} | "
            f"{_evidence_label(forecast['evidence_status'])} | "
            f"{forecast['unavailable_reason'] or '通过严格匹配与验证'} |"
        )
        for horizon, forecast in forecasts.items()
    )
    forecast_return_rows = "\n".join(
        (
            f"| {horizon}日 | "
            f"{_percent(forecast['median_return'])} | "
            f"{_percent_range(forecast['return_p10'], forecast['return_p90'])} | "
            f"{_percent(forecast['loss_probability'])} | "
            f"{_currency(forecast['expected_price'], 4)} | "
            f"{_currency(forecast['expected_profit_on_capital'], 0)} |"
        )
        for horizon, forecast in forecasts.items()
    )
    forecast_risk_rows = "\n".join(
        (
            f"| {horizon}日 | "
            f"{_percent(forecast['drawdown_event_probability'])} | "
            f"{_percent(forecast['drawdown_5pct_probability'])} | "
            f"{_drawdown_window(forecast)} |"
        )
        for horizon, forecast in forecasts.items()
    )
    if any(forecast["median_return"] is not None for forecast in forecasts.values()):
        forecast_numeric_tables = f"""| 持有期 | 模型中位收益 | 模型10%～90%区间 | 模型期末亏损率 | 模型中位价 | ¥{state["initial_contribution"]:,.0f}模型中位盈亏 |
|---|---:|---:|---:|---:|---:|
{forecast_return_rows}

| 持有期 | 模型期间跌破起点概率 | 模型期间跌超5%概率 | 模型低点窗口 |
|---|---:|---:|---:|
{forecast_risk_rows}"""
    else:
        forecast_numeric_tables = (
            "当前模型数值全部停用，已隐藏无效的“不可用”空表；"
            "门控恢复后，收益、目标价与风险表会自动重新出现。"
        )
    reference_return_rows = "\n".join(
        (
            f"| {horizon}日 | {reference['sample_count']} | "
            f"{'样本达标' if reference['status'] == 'descriptive' else '小样本，仅描述'} | "
            f"{_percent(reference['median_return'])} | "
            f"{_percent_range(reference['return_p10'], reference['return_p90'])} | "
            f"{_percent(reference['loss_probability'])} |"
        )
        for horizon, reference in references.items()
    )
    reference_risk_rows = "\n".join(
        (
            f"| {horizon}日 | "
            f"{_percent(reference['drawdown_event_probability'])} | "
            f"{_percent(reference['drawdown_5pct_probability'])} | "
            f"{_reference_drawdown_window(reference)} |"
        )
        for horizon, reference in references.items()
    )
    ood_diagnostics = _ood_diagnostics_markdown(state)
    outcome_source = {
        "robot_index": "官方机器人产业指数980022",
        "official_underlying_index": "官方机器人产业指数980022",
        "etf": "机器人ETF易方达159530",
    }.get(state.get("forecast_outcome_source"), state.get("forecast_outcome_source", "未知"))
    validation_rows = "\n".join(
        (
            f"| {horizon}日 | {forecast['validation_sample_count']} | "
            f"{_percent(forecast['validation_mae'])} | "
            f"{_percent(forecast['validation_zero_baseline_mae'])} | "
            f"{_percent(forecast['validation_direction_accuracy'])} | "
            f"{_percent(forecast['validation_interval_coverage'])} | "
            f"{_forecast_validation_label(forecast['validation_status'])} |"
        )
        for horizon, forecast in forecasts.items()
    )
    status_labels = {
        "insufficient_samples": "样本不足",
        "baseline_failed": "未击败随机基线",
        "unstable": "分段表现不稳定",
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
            f"{evidence['mean_return_difference_vs_all']:+.2%} | "
            f"{'是' if evidence['passes_minimum_samples'] else '否'} | "
            f"{'是' if evidence['directionally_consistent'] else '否'} |"
        )
        for historical_action, evidence in sell_validation["actions"].items()
    )
    if not sell_validation_rows:
        sell_validation_rows = (
            "| 暂无足够样本 | 0 | 不可用 | 不可用 | 不可用 | 不可用 | 不可用 | 否 | 否 |"
        )
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
    sell_status_labels = {
        "insufficient_samples": "独立样本不足",
        "insufficient_action_samples": "极端动作样本不足",
        "inverse_signal": "方向倒挂",
        "validated": "已通过",
    }
    sell_status = sell_status_labels.get(
        sell_validation["status"],
        sell_validation["status"],
    )
    sell_reasons = "；".join(sell_validation["reasons"]) or "已通过当前门槛"
    stability_kind_labels = {
        "fixed_block": "固定分段",
        "rolling_tail": "最新尾窗",
    }
    displayed_stability_windows = validation["stability_windows"][-8:]
    stability_rows = "\n".join(
        (
            f"| {stability_kind_labels.get(window['window_kind'], window['window_kind'])} | "
            f"{window['start_date']} | {window['end_date']} | "
            f"{window['sample_count']} | {window['brier']:.4f} | "
            f"{_decimal(window['roc_auc'])} | "
            f"{'通过' if window['passes'] else '失败'} |"
        )
        for window in displayed_stability_windows
    )
    if not stability_rows:
        stability_rows = "| 暂无 | 暂无 | 暂无 | 0 | 不可用 | 不可用 | 失败 |"
    regime_labels = {
        "up_low_vol": "上涨/低波动",
        "up_high_vol": "上涨/高波动",
        "down_low_vol": "下跌/低波动",
        "down_high_vol": "下跌/高波动",
    }
    regime_rows = "\n".join(
        (
            f"| {regime_labels.get(regime['regime'], regime['regime'])} | "
            f"{regime['sample_count']} | {_decimal(regime['brier'])} | "
            f"{_decimal(regime['roc_auc'])} | "
            f"{'通过' if regime['passes'] else '失败'} |"
        )
        for regime in validation["regime_windows"]
    )
    return f"""## 研究型预估收益与下跌风险

{ood_diagnostics}

### 模型条件预测门控

模型预测只使用市场趋势、相对强弱和完整特征距离均合格的历史状态；收益结果来自**{outcome_source}**。当前状态处于训练分布外或验证未通过时，模型收益、目标价和卖出动作继续停用。

| 持有期 | 严格非重叠样本 | 状态 | 原因 |
|---|---:|---|---|
{forecast_gate_rows}

{forecast_numeric_tables}

### 宽口径历史压力参考（不可执行）

这是在**机器人指数20日相对强弱方向相同**的历史日期中，按持有期去除重叠后选取最近最多40条已实现路径。它保留了分布外历史，用于回答“过去类似弱势方向下发生过什么”，**不是当前行情预测**，不生成目标价、预估盈亏或交易动作。

| 持有期 | 非重叠样本 | 证据状态 | 历史中位收益 | 历史10%～90%区间 | 历史期末亏损率 |
|---|---:|---|---:|---:|---:|
{reference_return_rows}

| 持有期 | 历史期间跌破起点概率 | 历史期间跌超5%概率 | 跌破后的低点窗口 |
|---|---:|---:|---:|
{reference_risk_rows}

“什么时候会掉”不能精确到某一天；低点窗口只对曾跌破起始价格的历史路径统计，并给出低点日的25%～75%分位及条件样本数，不混入全程上涨路径。每天收盘后重算，优先看亏损概率和收益下界是否同步恶化。

## 研究型卖出指标（不可直接执行）

- 当前动作：**{action}**；原因：{sell["reason"]}。
- 下次复核：**{sell["review_horizon_trading_days"]}个交易日内**，并在每日收盘数据更新后提前复核。
- 10日风险控制参考价：**{_currency(sell["risk_control_price"], 4)}**（相似样本收益10%分位对应价）。
- 10日止盈观察参考价：**{_currency(sell["take_profit_price"], 4)}**（相似样本收益90%分位对应价）。
- 规则阈值：收缩后概率型分数低于{sell["probability_reduce_trigger"]:.0%}进入减仓判断；低于{sell["probability_exit_trigger"]:.0%}且10日中位收益为负、亏损概率不低于{sell["exit_loss_probability_trigger"]:.0%}时退出；中位收益为负且亏损概率不低于{sell["reduce_loss_probability_trigger"]:.0%}时减仓；亏损概率不低于{sell["watch_loss_probability_trigger"]:.0%}时至少保持观察。

参考价只用于研究观察，不是挂单建议；当前固定定投政策不会执行这些动作。

### 卖出规则历史验证

验证状态：**{sell_status}**。规则在**{sell_validation["sample_count"]}**个近似独立样本外日期上可计算；每次只使用该日以前已经实现的相似结果，相邻记录至少间隔10个交易日。

- 最低总样本：{sell_validation["minimum_sample_count"]}；每种保持/观察/减仓/退出动作最低样本：{sell_validation["minimum_action_sample_count"]}。
- 方向一致性：**{"通过" if sell_validation["directionally_consistent"] else "未通过"}**；原因：{sell_reasons}。
- 方向倒挂检查：保持应优于总体，观察/减仓/退出应弱于总体；不满足时不反向交易，直接停用规则。

{baseline_summary}

| 当时动作 | 独立触发次数 | 随后10日实际中位收益 | 随后10日实际平均收益 | 随后10日实际亏损率 | 路径平均最差收益 | 平均收益较全部日期 | 样本达标 | 方向一致 |
|---|---:|---:|---:|---:|---:|---:|---|---|
{sell_validation_rows}

这些阈值是预先设定的工程风险线，没有用本批数据调参。“较全部日期”只是状态分组差，不是执行规则带来的因果收益；卖出规则只有在独立样本量和方向一致性同时通过后才可能启用。

## 样本外验证

模型证据状态：**{status}**。

- 已实现的收缩后研究分数样本：**{validation["calibrated_sample_count"]}**。
- 全部重叠日标签的描述性方向准确率：**{_percent(validation["direction_accuracy"])}**。
- 每隔10个交易日抽取一次的近似非重叠样本：**{validation["confidence_sample_count"]}**；方向准确率：**{_percent(validation["confidence_direction_accuracy"])}**；Wilson 95%区间：**{model_accuracy_interval}**。
- 收缩后概率型分数Brier：**{_decimal(validation["calibrated_brier"])}**；原始模型：**{_decimal(validation["raw_brier"])}**；固定50%概率基线：**{validation["constant_50_brier"]:.4f}**。Brier越低越好。
- Brier技能分数：**{_percent(validation["brier_skill_score"])}**（高于0才表示击败50%基线）；分段稳定性通过：**{"是" if validation["passes_stability"] else "否"}**。
- ROC AUC：**{_decimal(validation["roc_auc"])}**；0.5约等于随机排序。

### 模型分段稳定性

每50个连续已实现预测形成一个固定分段，至少需要3个固定分段。当前共{validation["stability_window_count"]}个固定分段，通过{validation["stability_pass_count"]}个；下表仅展示最近窗口，完整结果保留在 `data/latest_state.json`。最新尾段不足40条时，额外检查最近50条滚动尾窗，但该重叠尾窗不计入独立分段数量。

| 窗口类型 | 开始日期 | 结束日期 | 样本数 | Brier | ROC AUC | 结果 |
|---|---|---|---:|---:|---:|---|
{stability_rows}

### 市场环境稳定性

按沪深300是否位于120日均线上方，以及机器人指数20日年化波动是否达到35%，形成四个预注册环境。四类必须全部满足样本不少于40、Brier低于0.25且AUC高于0.5。

| 市场环境 | 样本数 | Brier | ROC AUC | 结果 |
|---|---:|---:|---:|---|
{regime_rows}

### 零收益基线门控

只有在近似独立样本不少于20条、预测MAE低于零收益基线MAE、且方向准确率高于50%时，才展示收益、价格与盈亏预估；否则全部停用。

| 预测期 | 独立验证样本 | 中位预测MAE | 零收益基线MAE | 方向准确率 | 10%～90%区间覆盖率 | 门控状态 |
|---|---:|---:|---:|---:|---:|---|
{validation_rows}

相似样本验证严格只使用每个预测日当时已经实现的历史结果。若中位预测MAE没有低于零收益基线，或模型Brier没有低于0.25，本指标只作为风险观察，不能作为独立买卖依据。"""


def _probability_markdown(state: dict) -> str:
    model_kind = state["model_kind"]
    if model_kind == "shrunk_logistic_regression":
        return (
            f"- 逻辑回归原始分数：**{state['raw_prediction_probability']:.2%}**\n"
            f"- 向50%收缩{state['probability_shrinkage']:.0%}后，未来10个交易日"
            "跑赢沪深300的预测概率型分数："
            f"**{state['prediction_probability']:.2%}**"
        )
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


def _signal_status(state: dict) -> str:
    if state["is_out_of_distribution"]:
        return (
            "**训练分布外，模型条件预测停用**"
            f"（{state['ood_features']}）；下方仅展示不可执行的历史压力参考"
        )
    return "**仅观察，不可用于自动调仓**"


def _gate_reasons(state: dict) -> str:
    gate = state.get("execution_gate", {})
    reasons = gate.get(
        "reasons",
        ["回顾性挑战者尚未用新增数据证明长期优于固定定投"],
    )
    return "；".join(reasons)


def _quality_gate_summary(state: dict) -> str:
    status = "通过" if state["quality_gate_passed"] else "未通过"
    return (
        f"**{status}**；最近成熟样本{state['quality_gate_sample_count']}条，"
        f"Brier={_decimal(state['quality_gate_brier'])}，"
        f"AUC={_decimal(state['quality_gate_auc'])}；{state['quality_gate_reason']}"
    )


def _methodology_breaks(state: dict) -> str:
    return "；".join(
        f"{item['effective_date']}（{item['reason']}）"
        for item in state["data_provenance"]["known_methodology_breaks"]
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
        "unvalidated": "未通过历史验证",
    }.get(status, status)


def _forecast_validation_label(status: str) -> str:
    return {
        "insufficient_samples": "独立样本不足",
        "baseline_failed": "未击败零收益基线",
        "validated": "已通过",
    }.get(status, status)


def _model_selection_label(status: str) -> str:
    return {
        "retrospective_challenger": "回顾性挑战者",
    }.get(status, status)


def _drawdown_window(forecast: dict) -> str:
    count = forecast["drawdown_event_sample_count"]
    lower = forecast["drawdown_trough_day_p25"]
    upper = forecast["drawdown_trough_day_p75"]
    if not count or lower is None or upper is None:
        return "不可用"
    return f"第{lower:.0f}～{upper:.0f}日（n={count}）"


def _reference_drawdown_window(reference: dict) -> str:
    count = reference["drawdown_event_sample_count"]
    lower = reference["drawdown_trough_day_p25"]
    upper = reference["drawdown_trough_day_p75"]
    if not count or lower is None or upper is None:
        return "不可用"
    return f"第{lower:.0f}～{upper:.0f}日（n={count}）"


def _ood_diagnostics_markdown(state: dict) -> str:
    diagnostics = state.get("distribution_diagnostics", [])
    if not diagnostics:
        return ""
    feature_labels = {
        "return_5": "5日收益",
        "return_20": "20日收益",
        "return_60": "60日收益",
        "distance_ma20": "距20日均线",
        "distance_ma60": "距60日均线",
        "volatility_20": "20日年化波动",
        "volume_ratio_20": "成交量/20日均量",
        "relative_strength_20": "20日相对强弱",
        "market_trend_120": "沪深300距120日均线",
    }
    status_labels = {
        "below": "低于训练P1",
        "above": "高于训练P99",
        "in_range": "范围内",
    }
    rows = "\n".join(
        (
            f"| {feature_labels.get(item['feature'], item['feature'])} | "
            f"{_feature_value(item['feature'], item['current_value'])} | "
            f"{_feature_value(item['feature'], item['lower_bound'])} | "
            f"{_feature_value(item['feature'], item['upper_bound'])} | "
            f"{status_labels.get(item['status'], item['status'])} |"
        )
        for item in diagnostics
    )
    sample_count = max(item["training_sample_count"] for item in diagnostics)
    return f"""### 当前 OOD 证据

以下边界来自本月模型实际使用的 **{sample_count}** 条训练样本；不是另算的一套口径。

| 特征 | 当前值 | 训练P1 | 训练P99 | 判定 |
|---|---:|---:|---:|---|
{rows}"""


def _feature_value(feature: str, value: float) -> str:
    if feature == "volume_ratio_20":
        return f"{value:.2f}倍"
    return f"{value:.2%}"
