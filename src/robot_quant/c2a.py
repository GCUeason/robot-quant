"""C2-A 早盘异常强势策略的防前视回测核心。

本模块只负责纯计算，不连接券商，也不把回测结果转换为实盘指令。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import date, time
from math import ceil, floor
from typing import Literal

import numpy as np
import pandas as pd


Pool = Literal["MAIN", "GROWTH"]
DataStatus = Literal["STRICT", "PROXY"]

MINUTE_COLUMNS = {
    "timestamp",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
}
UNIVERSE_COLUMNS = {
    "trade_date",
    "ticker",
    "name",
    "pool",
    "list_date",
    "listing_trading_days",
    "prevclose",
    "prevhigh",
    "avg3_amount",
    "float_shares",
    "float_mcap",
    "is_st",
    "is_suspended",
    "upper_limit",
    "lower_limit",
    "limit_streak",
}

MORNING_FIRST_COMPLETE_MINUTE = time(9, 31)
MORNING_LAST_COMPLETE_MINUTE = time(11, 30)
AFTERNOON_FIRST_COMPLETE_MINUTE = time(13, 1)
AFTERNOON_LAST_COMPLETE_MINUTE = time(15, 0)


@dataclass(frozen=True)
class C2AParameters:
    """预注册的策略参数；默认值对应 C2-A v1。"""

    variant: str = "v1"
    scan_end: time = time(10, 0)
    entry_cutoff: time = time(14, 57)
    c6_threshold: float = 30.0
    amount_weight: float = 0.50
    turnover_weight: float = 0.30
    gain_weight: float = 0.20
    main_gain_min: float = 0.03
    main_gain_max: float = 0.095
    growth_gain_min: float = 0.03
    growth_gain_max: float = 0.19
    main_first_pullback: float = 0.030
    growth_first_pullback: float = 0.045
    main_second_increment: float = 0.010
    growth_second_increment: float = 0.010
    main_alt_increment: float = 0.005
    growth_alt_increment: float = 0.005
    first_weight: float = 0.60
    second_weight: float = 0.40
    baseline_days: int = 20
    confirmation_minutes: int = 1
    signal_expiry_minutes: int | None = None
    reset_c6: float = 35.0
    reset_minutes: int = 2
    use_relative_turnover: bool = False
    max_overshoot_main: float | None = None
    max_overshoot_growth: float | None = None
    allow_same_stock_second: bool = True
    daily_exposure_cap: float | None = None
    single_stock_cap: float | None = None
    max_positions: int = 2
    max_participation: float = 0.05
    cooldown_days: int = 20
    commission_rate: float = 0.0003
    minimum_commission: float = 5.0
    stamp_tax_sell: float = 0.0005
    transfer_fee_rate: float = 0.00001
    slippage_bps: float = 5.0
    price_tick: float = 0.01
    exclude_yesterday_limit_up: bool = False
    max_allowed_limit_streak: int = 1

    def __post_init__(self) -> None:
        weights = self.amount_weight + self.turnover_weight + self.gain_weight
        if not np.isclose(weights, 1.0):
            raise ValueError("c6 权重之和必须为1")
        if self.confirmation_minutes < 1:
            raise ValueError("连续确认分钟数必须至少为1")
        if self.baseline_days < 1:
            raise ValueError("历史同分钟基线天数必须至少为1")
        if self.scan_end < MORNING_FIRST_COMPLETE_MINUTE:
            raise ValueError("扫描截止时间不能早于首个完整分钟09:31")
        if self.first_weight + self.second_weight > 1.0 + 1e-12:
            raise ValueError("两仓权重之和不能超过100%")

    @classmethod
    def dynamic_snapshot(cls) -> C2AParameters:
        """返回 v1.2 动态完整分钟快照的纸面验证参数。"""

        return cls(
            variant="v1.2",
            scan_end=time(11, 0),
            confirmation_minutes=2,
            signal_expiry_minutes=3,
            use_relative_turnover=True,
            max_overshoot_main=0.010,
            max_overshoot_growth=0.015,
            allow_same_stock_second=False,
            daily_exposure_cap=10_000.0,
            single_stock_cap=5_000.0,
        )

    @classmethod
    def optimized_challenger(cls) -> C2AParameters:
        """返回短窗回看候选参数；仅用于前向纸面验证。"""

        return replace(
            cls.dynamic_snapshot(),
            scan_end=time(10, 0),
            c6_threshold=40.0,
            main_first_pullback=0.025,
            growth_first_pullback=0.035,
            main_second_increment=0.005,
            growth_second_increment=0.005,
            exclude_yesterday_limit_up=True,
        )

    def to_dict(self) -> dict:
        values = asdict(self)
        values["scan_end"] = self.scan_end.strftime("%H:%M")
        values["entry_cutoff"] = self.entry_cutoff.strftime("%H:%M")
        return values


@dataclass
class Position:
    ticker: str
    entry_date: pd.Timestamp
    entry_time: pd.Timestamp
    entry_price: float
    shares: int
    entry_value: float
    entry_cost: float
    signal_id: str
    signal_time: pd.Timestamp
    signal_price: float
    signal_c6: float
    signal_amount_burst: float
    signal_turnover_metric: float
    signal_gain: float
    h_prev: float
    trigger_price: float
    position_weight: float
    leg: str
    pool: Pool
    data_status: DataStatus
    locked_days: int = 0


@dataclass
class _SignalState:
    status: str = "READY"
    pass_count: int = 0
    reset_count: int = 0
    signal_number: int = 0
    signal_time: pd.Timestamp | None = None
    signal_bar: int | None = None
    signal_price: float | None = None
    signal_c6: float | None = None
    signal_amount_burst: float | None = None
    signal_turnover_metric: float | None = None
    signal_gain: float | None = None


@dataclass(frozen=True)
class _EntryCandidate:
    ticker: str
    bar: object
    state: _SignalState
    pool: Pool
    leg: str
    h_prev: float
    trigger: float
    base_fill: float


@dataclass(frozen=True)
class _FillableCandidate:
    candidate: _EntryCandidate
    fill_price: float
    shares: int
    entry_cost: float


@dataclass(frozen=True)
class C2AMarketData:
    bars_by_date: dict[pd.Timestamp, pd.DataFrame]
    stocks_by_date: dict[pd.Timestamp, pd.DataFrame]
    last_prices_by_date: dict[pd.Timestamp, dict[str, float]]
    all_dates: tuple[pd.Timestamp, ...]


@dataclass(frozen=True)
class C2APreparedData:
    market: C2AMarketData
    features_by_date: dict[pd.Timestamp, pd.DataFrame]


def normalize_minutes(frame: pd.DataFrame) -> pd.DataFrame:
    """规范分钟行情，并拒绝重复或非法 OHLC 数据。"""

    missing = MINUTE_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"分钟行情缺少字段: {sorted(missing)}")
    result = frame.loc[:, sorted(MINUTE_COLUMNS)].copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    result["ticker"] = result["ticker"].astype(str).str.zfill(6)
    numeric = ["open", "high", "low", "close", "volume", "amount"]
    result[numeric] = result[numeric].apply(pd.to_numeric, errors="coerce")
    if result[numeric].isna().any().any():
        raise ValueError("分钟行情包含无法解析的数值")
    if result.duplicated(["timestamp", "ticker"]).any():
        raise ValueError("分钟行情存在重复的 timestamp+ticker")
    invalid_ohlc = (
        (result["high"] < result[["open", "close", "low"]].max(axis=1))
        | (result["low"] > result[["open", "close", "high"]].min(axis=1))
        | (result[numeric] < 0).any(axis=1)
    )
    if invalid_ohlc.any():
        raise ValueError("分钟行情包含非法 OHLC 或负数成交数据")
    return result.sort_values(["timestamp", "ticker"]).reset_index(drop=True)


def normalize_universe(frame: pd.DataFrame) -> pd.DataFrame:
    """规范每日 Universe 快照。"""

    missing = UNIVERSE_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Universe 缺少字段: {sorted(missing)}")
    result = frame.loc[:, sorted(UNIVERSE_COLUMNS)].copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"]).dt.normalize()
    result["list_date"] = pd.to_datetime(result["list_date"]).dt.normalize()
    result["ticker"] = result["ticker"].astype(str).str.zfill(6)
    result["pool"] = result["pool"].astype(str).str.upper()
    if not result["pool"].isin(["MAIN", "GROWTH"]).all():
        raise ValueError("pool 只能是 MAIN 或 GROWTH")
    if result.duplicated(["trade_date", "ticker"]).any():
        raise ValueError("Universe 存在重复的 trade_date+ticker")
    return result.sort_values(["trade_date", "ticker"]).reset_index(drop=True)


def eligible_universe(universe: pd.DataFrame, params: C2AParameters) -> pd.Series:
    """按前一交易日已知信息构造盘前股票池。"""

    limit_ok = universe["limit_streak"].fillna(0).astype(int) <= params.max_allowed_limit_streak
    if params.exclude_yesterday_limit_up:
        limit_ok &= universe["limit_streak"].fillna(0).astype(int) == 0
    names = universe["name"].fillna("").astype(str).str.strip()
    delisting_arrangement = names.str.contains(r"退市|退$", regex=True)
    return (
        universe["ticker"].str.match(r"^(600|601|603|605|000|001|002|300|301|688|689)")
        & ~universe["is_st"].astype(bool)
        & ~delisting_arrangement
        & ~universe["is_suspended"].astype(bool)
        & (universe["listing_trading_days"] >= 20)
        & (universe["prevclose"] >= 3.0)
        & (universe["avg3_amount"] >= 100_000_000.0)
        & universe["float_mcap"].between(2_000_000_000.0, 300_000_000_000.0)
        & limit_ok
    )


def pool_for_ticker(ticker: str) -> Pool | None:
    """把策略支持的A股代码映射到唯一板块分类。"""

    symbol = str(ticker).zfill(6)
    if symbol.startswith(("600", "601", "603", "605", "000", "001", "002")):
        return "MAIN"
    if symbol.startswith(("300", "301", "688", "689")):
        return "GROWTH"
    return None


def rank_c2a_cross_section(
    frame: pd.DataFrame,
    params: C2AParameters,
    group_columns: list[str],
) -> pd.DataFrame:
    """用唯一实现计算两个涨停组、直读和分区缓存的 c6。"""

    ranked = frame.copy()
    groups = ranked.groupby(group_columns, sort=False)
    ranked["p_amount"] = groups["amount_burst"].rank(pct=True, method="average")
    ranked["p_turnover"] = groups["turnover_metric"].rank(pct=True, method="average")
    ranked["p_gain"] = groups["gain"].rank(pct=True, method="average")
    ranked["q"] = (
        params.amount_weight * ranked["p_amount"]
        + params.turnover_weight * ranked["p_turnover"]
        + params.gain_weight * ranked["p_gain"]
    )
    ranked["c6"] = 100.0 * (1.0 - ranked["q"])
    return ranked


def build_signal_features(
    minutes: pd.DataFrame,
    universe: pd.DataFrame,
    params: C2AParameters,
) -> pd.DataFrame:
    """计算历史同分钟基线及全池横截面 c6，历史滚动窗口严格 shift(1)。"""

    bars = normalize_minutes(minutes)
    stocks = normalize_universe(universe)
    bars["trade_date"] = bars["timestamp"].dt.normalize()
    bars["minute_key"] = bars["timestamp"].dt.strftime("%H:%M")
    bars["cum_amount"] = bars.groupby(["ticker", "trade_date"], sort=False)["amount"].cumsum()
    bars["cum_volume"] = bars.groupby(["ticker", "trade_date"], sort=False)["volume"].cumsum()
    bars = bars.sort_values(["ticker", "minute_key", "trade_date"])
    baseline_groups = bars.groupby(["ticker", "minute_key"], sort=False)
    bars["amount_baseline"] = baseline_groups["cum_amount"].transform(
        lambda values: (
            values.shift(1).rolling(params.baseline_days, min_periods=params.baseline_days).median()
        )
    )
    bars["volume_baseline"] = baseline_groups["cum_volume"].transform(
        lambda values: (
            values.shift(1).rolling(params.baseline_days, min_periods=params.baseline_days).median()
        )
    )
    bars = bars.merge(stocks, on=["trade_date", "ticker"], how="inner", validate="many_to_one")
    bars["universe_pass"] = eligible_universe(bars, params)
    bars["amount_burst"] = bars["cum_amount"] / bars["amount_baseline"].replace(0, np.nan)
    if params.use_relative_turnover:
        bars["turnover_metric"] = bars["cum_volume"] / bars["volume_baseline"].replace(0, np.nan)
    else:
        bars["turnover_metric"] = bars["cum_volume"] / bars["float_shares"].replace(0, np.nan)
    bars["gain"] = bars["close"] / bars["prevclose"] - 1.0
    scan_mask = bars["timestamp"].dt.time <= params.scan_end
    valid = bars.loc[
        scan_mask
        & bars["universe_pass"]
        & bars[["amount_burst", "turnover_metric", "gain"]].notna().all(axis=1)
    ].copy()
    valid = rank_c2a_cross_section(
        valid,
        params,
        ["trade_date", "minute_key", "pool"],
    )
    gain_min = np.where(valid["pool"].eq("MAIN"), params.main_gain_min, params.growth_gain_min)
    gain_max = np.where(valid["pool"].eq("MAIN"), params.main_gain_max, params.growth_gain_max)
    valid["signal_pass"] = (
        valid["gain"].ge(gain_min)
        & valid["gain"].le(gain_max)
        & valid["c6"].lt(params.c6_threshold)
    )
    return valid.sort_values(["timestamp", "ticker"]).reset_index(drop=True)


def _pullback(pool: Pool, leg: str, params: C2AParameters) -> float:
    first = params.main_first_pullback if pool == "MAIN" else params.growth_first_pullback
    if leg == "FIRST":
        return first
    if leg == "SAME_SECOND":
        increment = (
            params.main_second_increment if pool == "MAIN" else params.growth_second_increment
        )
    else:
        increment = params.main_alt_increment if pool == "MAIN" else params.growth_alt_increment
    return first + increment


def _overshoot_limit(pool: Pool, params: C2AParameters) -> float | None:
    return params.max_overshoot_main if pool == "MAIN" else params.max_overshoot_growth


def _commission(value: float, params: C2AParameters) -> float:
    return max(params.minimum_commission, value * params.commission_rate)


def _round_price(price: float, tick: float, side: str) -> float:
    units = price / tick
    rounded = ceil(units - 1e-12) if side == "BUY" else floor(units + 1e-12)
    return round(rounded * tick, 6)


def _buy_fill(base_price: float, params: C2AParameters) -> float:
    slipped = base_price * (1.0 + params.slippage_bps / 10_000.0)
    return _round_price(slipped, params.price_tick, "BUY")


def _sell_fill(open_price: float, params: C2AParameters) -> float:
    slipped = open_price * (1.0 - params.slippage_bps / 10_000.0)
    return _round_price(slipped, params.price_tick, "SELL")


def _position_size(
    ticker: str,
    fill_price: float,
    desired_value: float,
    minute_amount: float,
    cash: float,
    params: C2AParameters,
) -> tuple[int, float]:
    participation_value_cap = minute_amount * params.max_participation
    total_cost_cap = min(desired_value, cash)
    if params.single_stock_cap is not None:
        total_cost_cap = min(total_cost_cap, params.single_stock_cap)
    share_value_cap = min(participation_value_cap, total_cost_cap)
    is_star = ticker.startswith(("688", "689"))
    minimum_shares = 200 if is_star else 100
    step = 1 if is_star else 100
    raw_shares = floor(share_value_cap / fill_price)
    shares = raw_shares if is_star else floor(raw_shares / step) * step
    while shares >= minimum_shares:
        value = shares * fill_price
        total = value + _commission(value, params) + value * params.transfer_fee_rate
        if value <= participation_value_cap + 1e-9 and total <= total_cost_cap + 1e-9:
            return shares, total - value
        shares -= step
    return 0, 0.0


def _bar_value(bar: object, column: str):
    if isinstance(bar, pd.Series):
        return bar[column]
    return getattr(bar, column)


def _one_price_limit(bar: object, limit_price: float) -> bool:
    return bool(
        np.isfinite(limit_price)
        and np.isclose(_bar_value(bar, "open"), limit_price, atol=0.005)
        and np.isclose(_bar_value(bar, "high"), limit_price, atol=0.005)
        and np.isclose(_bar_value(bar, "low"), limit_price, atol=0.005)
        and np.isclose(_bar_value(bar, "close"), limit_price, atol=0.005)
    )


def _signal_rank(state: _SignalState, ticker: str) -> tuple[float, pd.Timestamp, str]:
    assert state.signal_c6 is not None
    assert state.signal_time is not None
    return state.signal_c6, state.signal_time, ticker


def _entry_candidate(
    ticker: str,
    bar: object,
    timestamp: pd.Timestamp,
    state: _SignalState,
    h_prev: float,
    price_cage_reference: float,
    pool: Pool,
    leg: str,
    params: C2AParameters,
) -> _EntryCandidate | None:
    if state.signal_time is None or timestamp <= state.signal_time or not np.isfinite(h_prev):
        return None
    trigger = h_prev * (1.0 - _pullback(pool, leg, params))
    if _bar_value(bar, "low") > trigger:
        return None
    cage_limit = price_cage_reference * 1.02
    if not ticker.startswith(("688", "689")):
        cage_limit = max(cage_limit, price_cage_reference + 10 * params.price_tick)
    if _round_price(trigger, params.price_tick, "BUY") > cage_limit + 1e-9:
        if params.signal_expiry_minutes is not None:
            state.status = "MISSED"
        return None
    overshoot = _overshoot_limit(pool, params)
    if overshoot is not None and _bar_value(bar, "close") < trigger * (1.0 - overshoot):
        if params.signal_expiry_minutes is not None:
            state.status = "MISSED"
        return None
    if _one_price_limit(bar, _bar_value(bar, "upper_limit")) or _one_price_limit(
        bar, _bar_value(bar, "lower_limit")
    ):
        if params.signal_expiry_minutes is not None:
            state.status = "MISSED"
        return None
    opening = float(_bar_value(bar, "open"))
    base_fill = min(opening, trigger) if opening < trigger else trigger
    return _EntryCandidate(ticker, bar, state, pool, leg, h_prev, trigger, base_fill)


def simulate_entry_day(
    day_minutes: pd.DataFrame,
    day_features: pd.DataFrame,
    day_universe: pd.DataFrame,
    params: C2AParameters,
    cash: float,
    budget: float,
    blocked_tickers: set[str] | None = None,
    data_status: DataStatus = "PROXY",
    *,
    prepared_inputs: bool = False,
) -> tuple[list[Position], float, list[dict]]:
    """执行单日信号、回撤与两仓竞争，返回新持仓、剩余现金及事件。"""

    blocked = blocked_tickers or set()
    feature_frame = day_features.copy()
    if not prepared_inputs:
        stocks = normalize_universe(day_universe).set_index("ticker")
        missing_metadata = sorted(
            column
            for column in UNIVERSE_COLUMNS
            if column not in {"ticker"} and column not in feature_frame
        )
        if missing_metadata:
            feature_frame = feature_frame.merge(
                stocks[missing_metadata],
                left_on="ticker",
                right_index=True,
                how="left",
                validate="many_to_one",
            )
    main = feature_frame["pool"].eq("MAIN")
    gain_min = np.where(main, params.main_gain_min, params.growth_gain_min)
    gain_max = np.where(main, params.main_gain_max, params.growth_gain_max)
    if "universe_pass" not in feature_frame:
        feature_frame["universe_pass"] = eligible_universe(feature_frame, params)
    feature_frame["signal_pass"] = (
        feature_frame["universe_pass"]
        & feature_frame["gain"].ge(gain_min)
        & feature_frame["gain"].le(gain_max)
        & feature_frame["c6"].lt(params.c6_threshold)
    )
    possible_tickers = set(
        feature_frame.loc[
            (pd.to_datetime(feature_frame["timestamp"]).dt.time <= params.scan_end)
            & feature_frame["signal_pass"],
            "ticker",
        ].astype(str)
    ).difference(blocked)
    if not possible_tickers:
        return [], cash, []
    if prepared_inputs:
        stocks = day_universe.loc[day_universe["ticker"].isin(possible_tickers)].set_index("ticker")
        bars = day_minutes.loc[day_minutes["ticker"].isin(possible_tickers)].copy()
    else:
        ticker_values = day_minutes["ticker"].astype(str).str.zfill(6)
        bars = normalize_minutes(day_minutes.loc[ticker_values.isin(possible_tickers)]).copy()
    stocks = stocks.loc[stocks.index.isin(possible_tickers)]
    feature_frame = feature_frame.loc[feature_frame["ticker"].isin(possible_tickers)]
    bars = bars.merge(
        stocks[["pool", "upper_limit", "lower_limit"]],
        left_on="ticker",
        right_index=True,
        how="inner",
        validate="many_to_one",
    )
    bars = bars.set_index("timestamp").sort_index()
    feature_keys = ["timestamp", "ticker"]
    if feature_frame.duplicated(feature_keys).any():
        raise ValueError("信号特征存在重复的 timestamp+ticker")
    feature_lookup = {
        (row.timestamp, row.ticker): row
        for row in feature_frame[
            [
                "timestamp",
                "ticker",
                "c6",
                "gain",
                "signal_pass",
                "amount_burst",
                "turnover_metric",
            ]
        ].itertuples(index=False)
    }
    states: dict[str, _SignalState] = {ticker: _SignalState() for ticker in stocks.index}
    high_so_far: dict[str, float] = {}
    last_close: dict[str, float] = {}
    positions: list[Position] = []
    events: list[dict] = []
    first_ticker: str | None = None
    bar_number = 0

    for timestamp, minute_bars in bars.groupby(level=0, sort=True):
        bar_number += 1
        if timestamp.time() > params.entry_cutoff:
            break
        candidates_first: list[_EntryCandidate] = []
        candidates_second: list[_EntryCandidate] = []
        for bar in minute_bars.itertuples(index=False):
            ticker = str(bar.ticker)
            h_prev = high_so_far.get(ticker, np.nan)
            price_cage_reference = last_close.get(ticker, np.nan)
            state = states[ticker]
            feature_key = (timestamp, ticker)
            feature = feature_lookup.get(feature_key)

            if timestamp.time() <= params.scan_end and feature is not None:
                gain_min = params.main_gain_min if bar.pool == "MAIN" else params.growth_gain_min
                gain_max = params.main_gain_max if bar.pool == "MAIN" else params.growth_gain_max
                reset_condition = feature.c6 >= params.reset_c6 or not (
                    gain_min <= feature.gain <= gain_max
                )
                if state.status == "MISSED":
                    state.reset_count = state.reset_count + 1 if reset_condition else 0
                    if state.reset_count >= params.reset_minutes:
                        states[ticker] = state = _SignalState(signal_number=state.signal_number)
                if state.status == "READY":
                    state.pass_count = state.pass_count + 1 if bool(feature.signal_pass) else 0
                    if state.pass_count >= params.confirmation_minutes and ticker not in blocked:
                        state.status = "FRESH"
                        state.signal_number += 1
                        state.signal_time = timestamp
                        state.signal_bar = bar_number
                        state.signal_price = float(bar.close)
                        state.signal_c6 = float(feature.c6)
                        state.signal_amount_burst = float(feature.amount_burst)
                        state.signal_turnover_metric = float(feature.turnover_metric)
                        state.signal_gain = float(feature.gain)
                        events.append(
                            {
                                "event": "SIGNAL",
                                "timestamp": timestamp,
                                "ticker": ticker,
                                "c6": state.signal_c6,
                                "signal_price": state.signal_price,
                                "amount_burst": state.signal_amount_burst,
                                "turnover_metric": state.signal_turnover_metric,
                                "gain": state.signal_gain,
                                "h_prev": h_prev,
                            }
                        )

            if (
                state.status == "FRESH"
                and params.signal_expiry_minutes is not None
                and state.signal_bar is not None
                and bar_number > state.signal_bar + params.signal_expiry_minutes
            ):
                state.status = "MISSED"
                events.append({"event": "MISSED_ENTRY", "timestamp": timestamp, "ticker": ticker})

            if state.status == "FRESH" and ticker not in {p.ticker for p in positions}:
                if first_ticker is None:
                    candidate = _entry_candidate(
                        ticker,
                        bar,
                        timestamp,
                        state,
                        h_prev,
                        price_cage_reference,
                        bar.pool,
                        "FIRST",
                        params,
                    )
                    if candidate:
                        candidates_first.append(candidate)
                elif len(positions) < params.max_positions:
                    candidate = _entry_candidate(
                        ticker,
                        bar,
                        timestamp,
                        state,
                        h_prev,
                        price_cage_reference,
                        bar.pool,
                        "ALT_SECOND",
                        params,
                    )
                    if candidate:
                        candidates_second.append(candidate)
            if first_ticker == ticker and params.allow_same_stock_second and len(positions) == 1:
                candidate = _entry_candidate(
                    ticker,
                    bar,
                    timestamp,
                    state,
                    h_prev,
                    price_cage_reference,
                    bar.pool,
                    "SAME_SECOND",
                    params,
                )
                if candidate:
                    candidates_second.append(candidate)
            high_so_far[ticker] = max(high_so_far.get(ticker, -np.inf), float(bar.high))
            last_close[ticker] = float(bar.close)

        active_candidates = candidates_first if first_ticker is None else candidates_second
        if not active_candidates:
            continue
        leg_weight = params.first_weight if first_ticker is None else params.second_weight
        desired_value = budget * leg_weight
        fillable: list[_FillableCandidate] = []
        for candidate in active_candidates:
            fill_price = min(
                _buy_fill(candidate.base_fill, params),
                float(_bar_value(candidate.bar, "upper_limit")),
            )
            shares, entry_cost = _position_size(
                candidate.ticker,
                fill_price,
                desired_value,
                float(_bar_value(candidate.bar, "amount")),
                cash,
                params,
            )
            if shares == 0:
                if params.signal_expiry_minutes is not None:
                    candidate.state.status = "MISSED"
                events.append(
                    {
                        "event": "UNFILLED_CAPACITY_OR_LOT",
                        "timestamp": timestamp,
                        "ticker": candidate.ticker,
                    }
                )
                continue
            fillable.append(
                _FillableCandidate(
                    candidate=candidate,
                    fill_price=fill_price,
                    shares=shares,
                    entry_cost=entry_cost,
                )
            )
        if not fillable:
            continue
        selected = min(
            fillable,
            key=lambda item: _signal_rank(item.candidate.state, item.candidate.ticker),
        )
        winner = selected.candidate
        if params.signal_expiry_minutes is not None:
            for loser in fillable:
                if loser is selected:
                    continue
                loser.candidate.state.status = "MISSED"
                events.append(
                    {
                        "event": "MISSED_ENTRY",
                        "timestamp": timestamp,
                        "ticker": loser.candidate.ticker,
                        "reason": "lost_same_minute_competition",
                    }
                )
        fill_price = selected.fill_price
        shares = selected.shares
        entry_cost = selected.entry_cost
        value = shares * fill_price
        cash -= value + entry_cost
        state = winner.state
        assert state.signal_time is not None
        position = Position(
            ticker=winner.ticker,
            entry_date=timestamp.normalize(),
            entry_time=timestamp,
            entry_price=fill_price,
            shares=shares,
            entry_value=value,
            entry_cost=entry_cost,
            signal_id=f"{timestamp.date().isoformat()}-{winner.ticker}-{state.signal_number}",
            signal_time=state.signal_time,
            signal_price=float(state.signal_price),
            signal_c6=float(state.signal_c6),
            signal_amount_burst=float(state.signal_amount_burst),
            signal_turnover_metric=float(state.signal_turnover_metric),
            signal_gain=float(state.signal_gain),
            h_prev=float(winner.h_prev),
            trigger_price=float(winner.trigger),
            position_weight=(value + entry_cost) / budget if budget > 0 else 0.0,
            leg=winner.leg,
            pool=winner.pool,
            data_status=data_status,
        )
        positions.append(position)
        if first_ticker is None:
            first_ticker = winner.ticker
        events.append(
            {
                "event": "ENTRY",
                "timestamp": timestamp,
                "ticker": winner.ticker,
                "leg": winner.leg,
                "fill_price": fill_price,
                "shares": shares,
            }
        )
        if len(positions) >= params.max_positions:
            break
    return positions, cash, events


def _exit_position(
    position: Position,
    day_bars: pd.DataFrame,
    day_universe: pd.DataFrame,
    params: C2AParameters,
) -> tuple[dict | None, float]:
    ticker_bars = day_bars.loc[day_bars["ticker"].eq(position.ticker)].sort_values("timestamp")
    if ticker_bars.empty:
        position.locked_days += 1
        return None, 0.0
    opening = ticker_bars.iloc[0]
    stock = day_universe.loc[day_universe["ticker"].eq(position.ticker)]
    lower_limit = float(stock.iloc[0]["lower_limit"]) if not stock.empty else np.nan
    first_tradeable = next(
        (row for _, row in ticker_bars.iterrows() if not _one_price_limit(row, lower_limit)),
        None,
    )
    if first_tradeable is None:
        position.locked_days += 1
        return None, 0.0
    exit_price = max(_sell_fill(float(first_tradeable["open"]), params), lower_limit)
    exit_value = position.shares * exit_price
    exit_cost = (
        _commission(exit_value, params)
        + exit_value * params.transfer_fee_rate
        + exit_value * params.stamp_tax_sell
    )
    proceeds = exit_value - exit_cost
    profit = proceeds - position.entry_value - position.entry_cost
    net_return = profit / (position.entry_value + position.entry_cost)
    return (
        {
            **asdict(position),
            "exit_date": pd.Timestamp(first_tradeable["timestamp"]).normalize(),
            "exit_time": pd.Timestamp(first_tradeable["timestamp"]),
            "next_day_open": float(opening["open"]),
            "exit_price": exit_price,
            "exit_value": exit_value,
            "exit_cost": exit_cost,
            "gross_return": exit_price / position.entry_price - 1.0,
            "net_return": net_return,
            "profit": profit,
            "cooldown_triggered": net_return < 0,
            "locked_limit_down_days": position.locked_days,
        },
        proceeds,
    )


def backtest_c2a(
    minutes: pd.DataFrame,
    universe: pd.DataFrame,
    params: C2AParameters | None = None,
    *,
    initial_capital: float = 100_000.0,
    trade_start: date | str | None = None,
    trade_end: date | str | None = None,
    data_status: DataStatus = "PROXY",
    signal_features: pd.DataFrame | None = None,
    prepared_data: C2APreparedData | None = None,
    event_sink: list[dict] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """按交易日运行完整 T+1 纸面账户，并输出交易、净值和摘要。"""

    config = params or C2AParameters()
    if prepared_data is None:
        bars = normalize_minutes(minutes)
        stocks = normalize_universe(universe)
        features = (
            build_signal_features(bars, stocks, config)
            if signal_features is None
            else signal_features
        )
        prepared_data = prepare_c2a_data(
            prepare_c2a_market_data(bars, stocks),
            features,
        )
    market = prepared_data.market
    start = (
        pd.Timestamp(trade_start).normalize() if trade_start else pd.Timestamp(market.all_dates[0])
    )
    end = pd.Timestamp(trade_end).normalize() if trade_end else pd.Timestamp(market.all_dates[-1])
    trades, equity = _run_c2a_account(
        market,
        start,
        end,
        initial_capital,
        config,
        lambda _day: config,
        lambda day, _params: prepared_data.features_by_date.get(
            day, pd.DataFrame(columns=["timestamp", "ticker"])
        ),
        data_status,
        event_sink,
    )
    summary = summarize_backtest(trades, equity, config, initial_capital, data_status)
    return trades, equity, summary


def backtest_c2a_schedule(
    parameter_schedule: dict[pd.Timestamp, C2AParameters | None],
    prepared_data: dict[bool, C2APreparedData],
    *,
    initial_capital: float = 100_000.0,
    data_status: DataStatus = "PROXY",
    event_sink: list[dict] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """在一个连续账户中按日切换参数，保留 T+1、跌停锁定与冷却状态。"""

    schedule = {pd.Timestamp(day).normalize(): params for day, params in parameter_schedule.items()}
    selected = [params for params in schedule.values() if params is not None]
    if not schedule or not selected:
        empty = pd.DataFrame()
        config = selected[0] if selected else C2AParameters()
        return (
            empty,
            empty,
            summarize_backtest(empty, empty, config, initial_capital, data_status),
        )
    markets = {id(item.market): item.market for item in prepared_data.values()}
    if len(markets) != 1:
        raise ValueError("走样本外参数组必须共享同一市场分区")
    market = next(iter(markets.values()))
    execution_config = selected[0]
    start = min(schedule)
    end = max(schedule)

    def features_for_day(day: pd.Timestamp, params: C2AParameters) -> pd.DataFrame:
        prepared = prepared_data[params.exclude_yesterday_limit_up]
        return prepared.features_by_date.get(day, pd.DataFrame(columns=["timestamp", "ticker"]))

    trades, equity = _run_c2a_account(
        market,
        start,
        end,
        initial_capital,
        execution_config,
        schedule.get,
        features_for_day,
        data_status,
        event_sink,
    )
    summary = summarize_backtest(
        trades,
        equity,
        execution_config,
        initial_capital,
        data_status,
    )
    return trades, equity, summary


def _run_c2a_account(
    market: C2AMarketData,
    start: pd.Timestamp,
    end: pd.Timestamp,
    initial_capital: float,
    execution_config: C2AParameters,
    parameters_for_day: Callable[[pd.Timestamp], C2AParameters | None],
    features_for_day: Callable[[pd.Timestamp, C2AParameters], pd.DataFrame],
    data_status: DataStatus,
    event_sink: list[dict] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """运行单一连续账户；无当日参数时只处理持仓和冷却。"""

    all_dates = market.all_dates
    cash = float(initial_capital)
    positions: list[Position] = []
    closed: list[dict] = []
    equity_rows: list[dict] = []
    cooldown_until: dict[str, int] = {}

    for day_index, raw_day in enumerate(all_dates):
        day = pd.Timestamp(raw_day).normalize()
        if day < start or day > end:
            continue
        day_bars = market.bars_by_date.get(day, _empty_minutes())
        day_stocks = market.stocks_by_date.get(day, pd.DataFrame())
        last_prices = market.last_prices_by_date.get(day, {})
        blocked = {
            ticker for ticker, blocked_until in cooldown_until.items() if day_index <= blocked_until
        }
        remaining: list[Position] = []
        closed_today: list[dict] = []
        for position in positions:
            trade, proceeds = _exit_position(position, day_bars, day_stocks, execution_config)
            if trade is None:
                remaining.append(position)
                if event_sink is not None:
                    event_sink.append(
                        {
                            "event": "EXIT_DEFERRED_UNTRADEABLE",
                            "timestamp": day,
                            "ticker": position.ticker,
                            "locked_days": position.locked_days,
                        }
                    )
                continue
            cash += proceeds
            closed_today.append(trade)
        _apply_ticker_cooldowns(
            closed_today,
            day_index,
            execution_config.cooldown_days,
            cooldown_until,
        )
        closed.extend(closed_today)
        positions = remaining

        blocked.update(position.ticker for position in positions)
        market_value = sum(p.shares * last_prices.get(p.ticker, p.entry_price) for p in positions)
        account_value = cash + market_value
        day_parameters = parameters_for_day(day)
        exposure_cap = (
            day_parameters.daily_exposure_cap if day_parameters is not None else None
        ) or account_value
        budget = min(max(0.0, exposure_cap - market_value), cash)
        maximum_positions = day_parameters.max_positions if day_parameters is not None else 0
        available_positions = max(0, maximum_positions - len(positions))
        if (
            day_parameters is not None
            and not day_bars.empty
            and budget > 0
            and available_positions > 0
        ):
            day_features = features_for_day(day, day_parameters)
            day_config = replace(day_parameters, max_positions=available_positions)
            new_positions, cash, day_events = simulate_entry_day(
                day_bars,
                day_features,
                day_stocks,
                day_config,
                cash,
                budget,
                blocked,
                data_status,
                prepared_inputs=True,
            )
            positions.extend(new_positions)
            if event_sink is not None:
                event_sink.extend(day_events)
        closing_value = sum(p.shares * last_prices.get(p.ticker, p.entry_price) for p in positions)
        equity_rows.append(
            {
                "trade_date": day,
                "cash": cash,
                "market_value": closing_value,
                "portfolio_value": cash + closing_value,
                "open_positions": len(positions),
            }
        )

    trades = pd.DataFrame(closed)
    equity = pd.DataFrame(equity_rows)
    if not equity.empty:
        equity["portfolio_daily_return"] = equity["portfolio_value"].pct_change().fillna(0.0)
    return trades, equity


def _apply_ticker_cooldowns(
    closed_today: list[dict],
    day_index: int,
    cooldown_days: int,
    cooldown_until: dict[str, int],
) -> None:
    """同股多条成交腿按最终合并净收益判定是否冷却。"""

    grouped = pd.DataFrame(closed_today).groupby("ticker", sort=False) if closed_today else []
    for ticker, ticker_trades in grouped:
        combined_profit = float(ticker_trades["profit"].sum())
        cooldown_triggered = combined_profit < 0
        for trade_index in ticker_trades.index:
            closed_today[trade_index]["cooldown_triggered"] = cooldown_triggered
            closed_today[trade_index]["ticker_combined_profit"] = combined_profit
        if cooldown_triggered:
            cooldown_until[str(ticker)] = day_index + cooldown_days


def prepare_c2a_market_data(
    minutes: pd.DataFrame,
    universe: pd.DataFrame,
) -> C2AMarketData:
    """一次规范化并按日分区，供数百组参数共享。"""

    bars = normalize_minutes(minutes)
    stocks = normalize_universe(universe)
    bars["trade_date"] = bars["timestamp"].dt.normalize()
    return C2AMarketData(
        bars_by_date={
            pd.Timestamp(day): group.drop(columns="trade_date").reset_index(drop=True)
            for day, group in bars.groupby("trade_date", sort=True)
        },
        stocks_by_date={
            pd.Timestamp(day): group.reset_index(drop=True)
            for day, group in stocks.groupby("trade_date", sort=True)
        },
        last_prices_by_date={
            pd.Timestamp(day): {
                str(ticker): float(close)
                for ticker, close in group.groupby("ticker", sort=False)["close"].last().items()
            }
            for day, group in bars.groupby("trade_date", sort=True)
        },
        all_dates=tuple(pd.Timestamp(day) for day in sorted(stocks["trade_date"].unique())),
    )


def prepare_c2a_data(
    market: C2AMarketData,
    signal_features: pd.DataFrame,
) -> C2APreparedData:
    """把某一涨停组特征绑定到共享市场分区。"""

    features = signal_features.copy()
    if features.empty:
        return C2APreparedData(market=market, features_by_date={})
    features["timestamp"] = pd.to_datetime(features["timestamp"])
    features["ticker"] = features["ticker"].astype(str).str.zfill(6)
    if "trade_date" not in features:
        features["trade_date"] = features["timestamp"].dt.normalize()
    else:
        features["trade_date"] = pd.to_datetime(features["trade_date"]).dt.normalize()
    return C2APreparedData(
        market=market,
        features_by_date={
            pd.Timestamp(day): group.reset_index(drop=True)
            for day, group in features.groupby("trade_date", sort=True)
        },
    )


def _empty_minutes() -> pd.DataFrame:
    return pd.DataFrame(columns=sorted(MINUTE_COLUMNS))


def summarize_backtest(
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    params: C2AParameters,
    initial_capital: float,
    data_status: DataStatus,
) -> dict:
    """计算不夸大的基础表现指标。"""

    if equity.empty:
        final_value = initial_capital
        max_drawdown = 0.0
        daily_returns = pd.Series(dtype=float)
    else:
        final_value = float(equity.iloc[-1]["portfolio_value"])
        values = equity["portfolio_value"].astype(float)
        max_drawdown = float((values / values.cummax() - 1.0).min())
        daily_returns = values.pct_change().dropna()
    annualized_volatility = (
        float(daily_returns.std(ddof=1) * np.sqrt(252)) if len(daily_returns) >= 2 else None
    )
    sharpe = None
    if annualized_volatility and annualized_volatility > 0:
        sharpe = float(daily_returns.mean() * 252 / annualized_volatility)
    wins = int((trades["net_return"] > 0).sum()) if not trades.empty else 0
    losses = int((trades["net_return"] < 0).sum()) if not trades.empty else 0
    positive_profit = (
        float(trades.loc[trades["profit"] > 0, "profit"].sum()) if not trades.empty else 0.0
    )
    negative_profit = (
        float(trades.loc[trades["profit"] < 0, "profit"].sum()) if not trades.empty else 0.0
    )
    total_cost = (
        float((trades["entry_cost"] + trades["exit_cost"]).sum()) if not trades.empty else 0.0
    )
    return {
        "strategy": "C2-A",
        "variant": params.variant,
        "execution_permission": "PAPER_ONLY",
        "data_status": data_status,
        "initial_capital": initial_capital,
        "final_value": final_value,
        "net_profit": final_value - initial_capital,
        "total_return": final_value / initial_capital - 1.0,
        "max_drawdown": max_drawdown,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "trade_count": int(len(trades)),
        "win_rate": wins / len(trades) if len(trades) else None,
        "mean_trade_return": float(trades["net_return"].mean()) if len(trades) else None,
        "median_trade_return": float(trades["net_return"].median()) if len(trades) else None,
        "average_win_return": (
            float(trades.loc[trades["net_return"] > 0, "net_return"].mean()) if wins else None
        ),
        "average_loss_return": (
            float(trades.loc[trades["net_return"] < 0, "net_return"].mean()) if losses else None
        ),
        "profit_factor": (positive_profit / abs(negative_profit) if negative_profit < 0 else None),
        "total_transaction_cost": total_cost,
        "transaction_cost_to_capital": total_cost / initial_capital,
        "exposure_day_rate": (
            float(equity["open_positions"].gt(0).mean())
            if "open_positions" in equity and len(equity)
            else None
        ),
        "locked_exit_count": (
            int((trades["locked_limit_down_days"] > 0).sum()) if len(trades) else 0
        ),
        "execution_fidelity": "MINUTE_BAR_PROXY",
        "price_cage_mode": "previous_complete_minute_close_conservative_proxy",
        "parameters": params.to_dict(),
    }


def parameter_grid(base: C2AParameters | None = None) -> list[C2AParameters]:
    """返回预注册的稳定性网格，不调整评分权重和 T+1 退出核心。"""

    baseline = base or C2AParameters()
    configs: list[C2AParameters] = []
    # 主板与成长板回撤按预注册的相邻稳定性路径配对，避免2160个全笛卡尔组合造成
    # 不必要的多重检验；仍覆盖原文列出的全部边界与默认点。
    pullback_pairs = (
        (0.020, 0.030),
        (0.025, 0.035),
        (0.030, 0.045),
        (0.035, 0.050),
        (0.040, 0.055),
    )
    for scan_end in (time(10, 0), time(10, 30), time(11, 0)):
        for c6 in (10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0):
            for main_pullback, growth_pullback in pullback_pairs:
                for increment in (0.005, 0.010, 0.015):
                    for exclude_yesterday_limit_up in (False, True):
                        configs.append(
                            replace(
                                baseline,
                                scan_end=scan_end,
                                c6_threshold=c6,
                                main_first_pullback=main_pullback,
                                growth_first_pullback=growth_pullback,
                                main_second_increment=increment,
                                growth_second_increment=increment,
                                main_alt_increment=increment,
                                growth_alt_increment=increment,
                                exclude_yesterday_limit_up=exclude_yesterday_limit_up,
                            )
                        )
    return configs
