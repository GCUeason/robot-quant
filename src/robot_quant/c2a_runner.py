"""C2-A 数据更新、回测和报告编排。"""

from __future__ import annotations

import json
import hashlib
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from robot_quant.c2a import (
    C2AParameters,
    backtest_c2a,
    eligible_universe,
    parameter_grid,
)
from robot_quant.c2a_cache import C2AResearchCache
from robot_quant.c2a_bigquant import (
    BigQuantSdkClient,
    build_bigquant_universe,
    download_bigquant_minutes,
    stream_bigquant_minutes_to_cache,
)
from robot_quant.c2a_data import (
    C2ADataStore,
    TushareRestClient,
    build_tushare_universe,
    download_tushare_minutes,
)
from robot_quant.c2a_optimize import WalkForwardConfig, walk_forward_optimize


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _write_text_atomic(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def run_c2a_audit(
    data_root: str | Path = "data/c2a",
    output_root: str | Path = ".",
    start_date: date | str = "2026-01-01",
    end_date: date | str | None = None,
    variant: str = "v1.2",
) -> dict:
    final_date = end_date or date.today()
    params = _variant_parameters(variant)
    store = C2ADataStore(data_root)
    cache = C2AResearchCache(store.root / "research_cache", _cache_parameters(variant))
    audit = _audit_store(store, cache, start_date, final_date, params)
    output = Path(output_root)
    reports_dir = output / "reports"
    data_dir = output / "data" / "c2a_results"
    reports_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    _write_text_atomic(
        json.dumps(audit.to_dict(), ensure_ascii=False, indent=2),
        data_dir / "data_audit.json",
    )
    _write_text_atomic(
        _readiness_markdown(audit.to_dict(), params),
        reports_dir / "c2a_2026_report.md",
    )
    return audit.to_dict()


def run_c2a_backtest(
    data_root: str | Path = "data/c2a",
    output_root: str | Path = ".",
    start_date: date | str = "2026-01-01",
    end_date: date | str | None = None,
    variant: str = "v1.2",
    *,
    allow_proxy: bool = False,
    optimize: bool = True,
    max_candidates: int | None = None,
    walk_forward_config: WalkForwardConfig | None = None,
) -> dict:
    """运行基准与走样本外优化；默认拒绝 PROXY 数据。"""

    final_date = end_date or date.today()
    params = _variant_parameters(variant)
    store = C2ADataStore(data_root)
    cache = C2AResearchCache(store.root / "research_cache", _cache_parameters(variant))
    audit = _audit_store(store, cache, start_date, final_date, params)
    if audit.status == "DATA_NOT_READY":
        run_c2a_audit(data_root, output_root, start_date, final_date, variant)
        raise RuntimeError("C2-A 本地数据仓为空，已生成 readiness 报告；请先更新或导入数据")
    if audit.status != "STRICT" and not allow_proxy:
        run_c2a_audit(data_root, output_root, start_date, final_date, variant)
        raise RuntimeError(
            "C2-A 数据未通过 STRICT 门槛，已生成 readiness 报告；如仅做降级研究需显式 --allow-proxy"
        )
    universe = store.read_universe(end_date=final_date)
    cache_summary = (
        cache.streaming_summary(final_date)
        if cache.is_streaming()
        else cache.build(store, final_date, progress_callback=print)
    )
    _market, prepared_data = cache.load_prepared_data(
        universe,
        start_date=start_date,
        end_date=final_date,
    )
    minutes = pd.DataFrame()
    baseline_events: list[dict] = []
    trades, equity, baseline_summary = backtest_c2a(
        minutes,
        universe,
        params,
        initial_capital=100_000.0,
        trade_start=start_date,
        trade_end=final_date,
        data_status=audit.status,
        prepared_data=prepared_data[params.exclude_yesterday_limit_up],
        event_sink=baseline_events,
    )
    output = Path(output_root)
    data_dir = output / "data" / "c2a_results"
    reports_dir = output / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    selections = pd.DataFrame()
    oos_trades = pd.DataFrame()
    latest_training_grid = pd.DataFrame()
    carried_prior = False
    optimization_summary: dict = {
        "status": "SKIPPED",
        "reason": "optimization_disabled",
        "promotion_gate": "FAIL",
    }
    if optimize:
        candidates = parameter_grid(params)
        if max_candidates is not None:
            candidates = candidates[:max_candidates]
        selections, oos_trades, optimization_summary = walk_forward_optimize(
            minutes,
            universe,
            candidates,
            initial_capital=100_000.0,
            data_status=audit.status,
            config=walk_forward_config or WalkForwardConfig(),
            prepared_data=prepared_data,
            optimization_start=start_date,
            optimization_end=final_date,
            progress_callback=print,
            path_cache_dir=data_dir / "parameter_paths",
            path_cache_context=hashlib.sha256(cache.manifest_path.read_bytes()).hexdigest(),
        )
        latest_training_grid = pd.DataFrame(
            optimization_summary.pop("_latest_training_grid_records", [])
        )
        optimization_summary["status"] = "COMPLETED"
        optimization_summary["candidate_count"] = len(candidates)
        optimization_summary["optimization_as_of"] = pd.Timestamp(final_date).date().isoformat()
    else:
        prior = _load_prior_optimization(data_dir)
        if prior is not None:
            optimization_summary, selections = prior
            optimization_summary["status"] = "CARRIED_FORWARD"
            carried_prior = True
    _write_csv_atomic(trades, data_dir / "baseline_trades.csv")
    _write_csv_atomic(equity, data_dir / "baseline_equity.csv")
    _write_csv_atomic(pd.DataFrame(baseline_events), data_dir / "baseline_events.csv")
    if optimize or not carried_prior:
        _write_csv_atomic(selections, data_dir / "walk_forward_selections.csv")
        _write_csv_atomic(oos_trades, data_dir / "walk_forward_oos_trades.csv")
        _write_csv_atomic(latest_training_grid, data_dir / "latest_training_grid.csv")
    daily_signal = _daily_signal_summary(
        baseline_events,
        universe,
        final_date,
        params,
        optimization_summary.get("promotion_gate", "FAIL"),
    )
    payload = {
        "as_of": pd.Timestamp(final_date).date().isoformat(),
        "audit": audit.to_dict(),
        "research_cache": cache_summary,
        "baseline": baseline_summary,
        "walk_forward": optimization_summary,
        "daily_signal": daily_signal,
    }
    _write_text_atomic(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        data_dir / "latest_state.json",
    )
    _write_text_atomic(
        json.dumps(daily_signal, ensure_ascii=False, indent=2, default=str),
        data_dir / "latest_signal.json",
    )
    _write_text_atomic(
        _result_markdown(payload, selections),
        reports_dir / "c2a_2026_report.md",
    )
    return payload


def _load_prior_optimization(data_dir: Path) -> tuple[dict, pd.DataFrame] | None:
    state_path = data_dir / "latest_state.json"
    selections_path = data_dir / "walk_forward_selections.csv"
    if not state_path.exists():
        return None
    try:
        prior = json.loads(state_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    walk = prior.get("walk_forward", {})
    if walk.get("status") not in {"COMPLETED", "CARRIED_FORWARD"}:
        return None
    try:
        selections = pd.read_csv(selections_path) if selections_path.exists() else pd.DataFrame()
    except pd.errors.EmptyDataError:
        selections = pd.DataFrame()
    return dict(walk), selections


def run_c2a_cache(
    data_root: str | Path = "data/c2a",
    start_date: date | str = "2026-01-01",
    end_date: date | str | None = None,
    variant: str = "v1.2",
) -> dict:
    """审计后增量生成全市场研究缓存，不运行参数优化。"""

    final_date = end_date or date.today()
    params = _variant_parameters(variant)
    store = C2ADataStore(data_root)
    cache = C2AResearchCache(store.root / "research_cache", _cache_parameters(variant))
    audit = _audit_store(store, cache, start_date, final_date, params)
    if audit.status == "DATA_NOT_READY":
        raise RuntimeError("C2-A 本地数据仓为空，无法生成研究缓存")
    return {
        "audit": audit.to_dict(),
        **(
            cache.streaming_summary(final_date)
            if cache.is_streaming()
            else cache.build(store, final_date, progress_callback=print)
        ),
    }


def run_c2a_tushare_update(
    data_root: str | Path = "data/c2a",
    start_date: date | str = "2026-01-01",
    end_date: date | str | None = None,
    *,
    universe_only: bool = False,
) -> dict:
    """更新 Universe 与分钟原始数据；需用户自行配置 Tushare 授权。"""

    final_date = end_date or date.today()
    history_start = pd.Timestamp(start_date).date() - timedelta(days=45)
    store = C2ADataStore(data_root)
    client = TushareRestClient()
    universe = build_tushare_universe(store, history_start, final_date, client=client)
    result: dict = {"universe_rows": len(universe)}
    if not universe_only:
        tickers = universe.loc[
            eligible_universe(universe, C2AParameters()), "ticker"
        ].drop_duplicates()
        result.update(
            download_tushare_minutes(
                store,
                tickers,
                history_start,
                final_date,
                client=client,
                progress_callback=print,
            )
        )
    return result


def run_c2a_bigquant_update(
    data_root: str | Path = "data/c2a",
    start_date: date | str = "2026-01-01",
    end_date: date | str | None = None,
    *,
    universe_only: bool = False,
    stream_cache: bool = False,
) -> dict:
    """使用官方 BigQuant SDK 增量更新 C2-A 数据仓。"""

    final_date = end_date or date.today()
    history_start = pd.Timestamp(start_date).date() - timedelta(days=45)
    store = C2ADataStore(data_root)
    client = BigQuantSdkClient()
    universe = build_bigquant_universe(
        store,
        history_start,
        final_date,
        client=client,
    )
    result: dict = {"universe_rows": len(universe)}
    if not universe_only:
        if stream_cache:
            params = C2AParameters.dynamic_snapshot()
            cache = C2AResearchCache(store.root / "research_cache", params)
            result.update(
                stream_bigquant_minutes_to_cache(
                    store,
                    cache,
                    history_start,
                    final_date,
                    client=client,
                    progress_callback=print,
                )
            )
        else:
            result.update(
                download_bigquant_minutes(
                    store,
                    history_start,
                    final_date,
                    params=C2AParameters.dynamic_snapshot(),
                    client=client,
                    progress_callback=print,
                )
            )
    return result


def _audit_store(
    store: C2ADataStore,
    cache: C2AResearchCache,
    start_date: date | str,
    end_date: date | str,
    params: C2AParameters,
):
    if cache.is_streaming():
        return cache.audit_streaming(store, start_date, end_date)
    return store.audit(start_date, end_date, params)


def _variant_parameters(variant: str) -> C2AParameters:
    normalized = variant.lower()
    if normalized == "v1":
        return C2AParameters()
    if normalized in {"v1.2", "v1_2", "dynamic"}:
        return C2AParameters.dynamic_snapshot()
    if normalized in {"v1.2-challenger", "v1_2_challenger", "challenger"}:
        return C2AParameters.optimized_challenger()
    raise ValueError("variant 只能是 v1、v1.2 或 v1.2-challenger")


def _cache_parameters(variant: str) -> C2AParameters:
    """挑战者复用参数无关的 v1.2 特征缓存，避免重复构建全市场分钟基线。"""

    normalized = variant.lower()
    if normalized in {"v1.2-challenger", "v1_2_challenger", "challenger"}:
        return C2AParameters.dynamic_snapshot()
    return _variant_parameters(variant)


def _daily_signal_summary(
    events: list[dict],
    universe: pd.DataFrame,
    as_of: date | str,
    params: C2AParameters,
    promotion_gate: str,
) -> dict:
    """提取最后交易日的纸面入场；未通过模型门槛时不授予真实交易权限。"""

    trade_day = pd.Timestamp(as_of).normalize()
    day_events = [
        event for event in events if pd.Timestamp(event["timestamp"]).normalize() == trade_day
    ]
    day_universe = universe.loc[pd.to_datetime(universe["trade_date"]).dt.normalize().eq(trade_day)]
    names = day_universe.set_index("ticker")["name"].astype(str).to_dict()
    entries = []
    for event in day_events:
        if event.get("event") != "ENTRY":
            continue
        ticker = str(event["ticker"]).zfill(6)
        fill_price = float(event["fill_price"])
        shares = int(event["shares"])
        entries.append(
            {
                "ticker": ticker,
                "name": names.get(ticker, ticker),
                "entry_time": pd.Timestamp(event["timestamp"]).isoformat(),
                "leg": event.get("leg"),
                "simulated_fill_price": fill_price,
                "shares": shares,
                "simulated_notional": fill_price * shares,
            }
        )
    return {
        "as_of": trade_day.date().isoformat(),
        "status": "SIMULATED_ENTRY" if entries else "NO_ENTRY",
        "signal_count": sum(event.get("event") == "SIGNAL" for event in day_events),
        "entries": entries,
        "parameters": params.to_dict(),
        "promotion_gate": promotion_gate,
        "execution_permission": "PAPER_ONLY",
        "real_trade_authorized": False,
    }


def _readiness_markdown(audit: dict, params: C2AParameters) -> str:
    reason_lines = "\n".join(f"- `{reason}`" for reason in audit["reasons"]) or "- 无"
    return f"""# C2-A 2026 数据与模型就绪报告

## 结论

当前状态：**{audit["status"]} 数据 / MINUTE_BAR_PROXY 成交 / PAPER_ONLY**。

尚不能给出可信的“2026 最佳盈利参数”。C2-A 需要全A股1分钟历史、每日盘前 Universe、历史 ST/停牌、流通股本与涨跌停数据；任何缺项都会改变横截面百分位和模拟成交结果。

另外，报告截止日为 **{audit["end_date"]}**。尚未发生的2026年后续交易日不能回测，程序会在数据到来后增量补齐。

## 数据审计

- 策略版本：{params.variant}
- 数据源：{audit["source"] or "未配置"}
- 目标区间：{audit["start_date"]} 至 {audit["end_date"]}
- 目标交易日：{audit["trading_days"]}
- 基线预热交易日：{audit["baseline_days_available"]} / {params.baseline_days}
- Universe 行数：{audit["universe_rows"]:,}
- 分钟行数：{audit["minute_rows"]:,}

未通过项：

{reason_lines}

## 已实现但不得冒充收益结论的能力

- 防前视历史同分钟基线：先 `shift(1)`，再滚动20日中位数；
- 强制分钟成交量=股、成交额=元、流通股本=股，拒绝未换算的手/千元/万股数据；
- 排除09:25集合竞价，逐股校验09:31–11:30、13:01–15:00的240个连续竞价分钟结束时标；
- 主板/成长板全池横截面 c6；
- 上一完整分钟高点回撤触发、60%/40%竞争；
- T+1开盘退出、一字跌停锁定、20日亏损冷却；
- 整手、最低佣金、印花税、滑点和5%分钟成交参与率；
- 价格笼子只能用上一完整分钟收盘价作保守代理，未用逐笔委托验证；
- 全市场原始分区按日流式生成候选路径缓存，日更只重算变更日期；
- 扩展训练窗 + 1交易日隔离带 + 不重叠测试窗，OOS 折间保留持仓和冷却状态；
- 信号、未成交、竞争落选和跌停延迟事件单独留痕；
- 邻近参数稳定平台选择，不以孤立历史峰值作为默认参数。

## 补齐路径

1. 优先配置 BigQuant 本地 SDK；Tushare 仅作为已授权账户的备用数据源；
2. 在 PyCharm 运行 `C2A - Update Tushare Data`；
3. 运行 `C2A - Audit 2026 YTD`，直到状态为 `STRICT`；
4. 再运行 `C2A - Walk Forward 2026 YTD` 生成样本外报告。

Tushare 官方文档：

- 历史分钟：<https://tushare.pro/document/2?doc_id=370>
- 盘前股本：<https://tushare.pro/document/2?doc_id=329>
- 历史 ST：<https://tushare.pro/document/2?doc_id=397>
- 停复牌：<https://tushare.pro/document/2?doc_id=214>
"""


def _result_markdown(payload: dict, selections: pd.DataFrame) -> str:
    baseline = payload["baseline"]
    walk = payload["walk_forward"]
    cache = payload.get("research_cache", {})
    daily = payload.get("daily_signal", {})
    fold_rows = ""
    if not selections.empty:
        visible = selections[
            [
                "fold",
                "train_end",
                "test_start",
                "test_end",
                "status",
                "selected_parameter_id",
                "oos_trade_count",
                "oos_total_return",
            ]
        ].copy()
        fold_rows = visible.to_markdown(index=False)
    else:
        fold_rows = "无可评估折。"
    daily_entries = daily.get("entries", [])
    if daily_entries:
        daily_rows = pd.DataFrame(daily_entries)[
            [
                "ticker",
                "name",
                "entry_time",
                "leg",
                "simulated_fill_price",
                "shares",
                "simulated_notional",
            ]
        ].to_markdown(index=False)
    else:
        daily_rows = "当日没有满足全部入场条件的纸面交易。"
    return f"""# C2-A 走样本外与当日纸面信号报告

## 结论

状态：**{payload["audit"]["status"]} 数据 / {baseline["execution_fidelity"]} 成交 / {baseline["execution_permission"]}**。晋级门槛：**{walk.get("promotion_gate", "FAIL")}**。

基准参数净收益率为 **{baseline["total_return"]:.2%}**，最大回撤 **{baseline["max_drawdown"]:.2%}**，完成交易 **{baseline["trade_count"]}** 笔。该结果是历史模拟，不是收益承诺或实盘买入指令。

- 胜率：{_format_percent(baseline.get("win_rate"))}
- Sharpe：{_format_number(baseline.get("sharpe"))}
- 平均单笔净收益：{_format_percent(baseline.get("mean_trade_return"))}
- 盈亏比（总盈利/总亏损）：{_format_number(baseline.get("profit_factor"))}
- 交易成本/初始资金：{_format_percent(baseline.get("transaction_cost_to_capital"))}

## 当日纸面信号

- 数据日期：{daily.get("as_of", payload["as_of"])}
- 信号状态：{daily.get("status", "不可用")}
- 盘中初筛信号数：{daily.get("signal_count", 0)}
- 模型晋级门槛：{daily.get("promotion_gate", "FAIL")}
- 真实交易权限：**无；PAPER_ONLY**

{daily_rows}

## 基准参数

```json
{json.dumps(baseline["parameters"], ensure_ascii=False, indent=2)}
```

## 走样本外结果

- 研究缓存截止：{cache.get("last_processed_date") or "不可用"}
- 压缩后分钟行数：{cache.get("compact_bar_rows", 0):,}
- 优化状态：{walk.get("status", "不可用")}
- 参数结果截止：{walk.get("optimization_as_of") or "不可用"}
- 候选参数数：{walk.get("candidate_count", 0)}
- 已评估折数：{walk.get("evaluated_fold_count", 0)}
- OOS交易数：{walk.get("oos_trade_count", 0)}
- OOS平均单笔：{_format_percent(walk.get("oos_mean_trade_return"))}
- OOS盈利折占比：{_format_percent(walk.get("oos_profitable_fold_rate"))}
- 未通过原因：{", ".join(walk.get("promotion_reasons", [])) or "需人工复核"}

{fold_rows}

## 最近训练窗的多目标参数诊断

训练截止：{walk.get("latest_training_end") or "不可用"}。下列赢家分别按累计收益、Sharpe、最小回撤和邻近稳定性排序，属于训练窗诊断，不等同于下一测试窗收益：
全部候选组合的指标保存在 `data/c2a_results/latest_training_grid.csv`。

```json
{json.dumps(walk.get("latest_training_winners", {}), ensure_ascii=False, indent=2)}
```

## 解释边界

- 每折参数只由测试期之前的训练数据决定，中间留1个交易日隔离带；
- 各 OOS 测试窗使用同一连续账户，折边界不重置未退出持仓、跌停锁定或20日冷却；
- 默认按邻近参数收益中位数选稳定平台，不选孤立最高点；
- 佣金、最低5元佣金、卖出印花税、过户费、滑点、涨跌停与容量均已计入；价格笼子仅为上一完整分钟收盘价代理，未达到逐笔成交保真度；
- 未满60个前向交易日、40笔交易或数据非STRICT时，模型保持 `PAPER_ONLY`。
- `CARRIED_FORWARD` 表示当日只更新基准路径，参数排名沿用上述明确截止日的上次完整优化；
"""


def _format_percent(value) -> str:
    if value is None or pd.isna(value):
        return "不可用"
    return f"{float(value):.2%}"


def _format_number(value) -> str:
    if value is None or pd.isna(value):
        return "不可用"
    return f"{float(value):.3f}"
