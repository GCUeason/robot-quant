"""C2-A 盘中公共分钟适配器。

BigQuant 负责截至上一交易日的 Universe 与滚动基线；腾讯提供当日累计分钟，
东方财富只为已经通过信号初筛的股票补充分钟 OHLC。混合来源固定标记为 PROXY。
"""

from __future__ import annotations

import argparse
import json
import random
import time as time_module
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime, time
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from robot_quant.c2a import (
    C2AParameters,
    eligible_universe,
    normalize_minutes,
    normalize_universe,
    rank_c2a_cross_section,
    simulate_entry_day,
)
from robot_quant.c2a_bigquant import BigQuantSdkClient
from robot_quant.c2a_cache import C2AResearchCache
from robot_quant.c2a_data import C2ADataStore


TENCENT_MINUTE_URL = "https://web.ifzq.gtimg.cn/appstock/app/minute/query"
TENCENT_QUOTE_URL = "https://qt.gtimg.cn/q="
EASTMONEY_TREND_URL = "https://push2his.eastmoney.com/api/qt/stock/trends2/get"
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://gu.qq.com/"}


@dataclass(frozen=True)
class IntradayAudit:
    status: str
    complete_minutes: int
    ticker_count: int
    cutoff: str
    source: str = "tencent+eastmoney+bigquant_baseline"

    def to_dict(self) -> dict:
        return asdict(self)


def _market_symbol(ticker: str) -> str:
    symbol = str(ticker).zfill(6)
    prefix = "sh" if symbol.startswith(("600", "601", "603", "605", "688", "689")) else "sz"
    return f"{prefix}{symbol}"


def _read_url(url: str, *, timeout: float = 12.0, retries: int = 5) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = Request(url, headers=REQUEST_HEADERS)
            return urlopen(request, timeout=timeout).read()
        except Exception as error:  # noqa: BLE001 - 网络边界统一重试后再报告
            last_error = error
            if attempt + 1 < retries:
                backoff = min(3.0, 0.2 * (2**attempt))
                time_module.sleep(backoff + random.uniform(0.0, 0.15))
    assert last_error is not None
    raise RuntimeError(f"公共行情请求失败: {url.split('?', 1)[0]}") from last_error


def parse_tencent_cumulative_minutes(
    payload: bytes | str,
    ticker: str,
    trade_date: date,
    *,
    cutoff: time,
) -> pd.DataFrame:
    """把腾讯累计量额转换为排除09:30集合竞价后的单分钟量额。"""

    body = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    parsed = json.loads(body)
    market_symbol = _market_symbol(ticker)
    rows = parsed.get("data", {}).get(market_symbol, {}).get("data", {}).get("data", [])
    records: list[dict] = []
    for row in rows:
        parts = str(row).split()
        if len(parts) < 4 or len(parts[0]) != 4:
            continue
        row_time = time(int(parts[0][:2]), int(parts[0][2:]))
        if row_time > cutoff or row_time < time(9, 30):
            continue
        records.append(
            {
                "timestamp": pd.Timestamp.combine(trade_date, row_time),
                "ticker": str(ticker).zfill(6),
                "close": float(parts[1]),
                "cum_volume_lots": float(parts[2]),
                "cum_amount": float(parts[3]),
            }
        )
    frame = pd.DataFrame(records).sort_values("timestamp") if records else pd.DataFrame()
    if frame.empty or frame.iloc[0]["timestamp"].time() != time(9, 30):
        raise ValueError(f"{ticker} 缺少09:30累计基点")
    frame["volume"] = frame["cum_volume_lots"].diff() * 100.0
    frame["amount"] = frame["cum_amount"].diff()
    frame = frame.loc[frame["timestamp"].dt.time >= time(9, 31)].copy()
    if (
        frame[["volume", "amount"]].isna().any().any()
        or (frame[["volume", "amount"]] < 0).any().any()
    ):
        raise ValueError(f"{ticker} 累计量额无法转换为合法单分钟数据")
    return frame[["timestamp", "ticker", "close", "volume", "amount"]].reset_index(drop=True)


def audit_completed_window(
    bars: pd.DataFrame,
    expected_tickers: set[str],
    cutoff: time,
) -> IntradayAudit:
    """只审计已完成的09:31至截止分钟，不要求盘后240根。"""

    expected_minutes = pd.date_range(
        "2000-01-01 09:31",
        f"2000-01-01 {cutoff.strftime('%H:%M')}",
        freq="1min",
    )
    expected_numbers = {item.hour * 60 + item.minute for item in expected_minutes}
    normalized = bars.copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"])
    normalized["ticker"] = normalized["ticker"].astype(str).str.zfill(6)
    observed_tickers = set(normalized["ticker"])
    if observed_tickers != expected_tickers:
        missing = sorted(expected_tickers - observed_tickers)
        extra = sorted(observed_tickers - expected_tickers)
        raise ValueError(f"盘中股票覆盖不完整；缺失={missing[:10]}；额外={extra[:10]}")
    if normalized.duplicated(["timestamp", "ticker"]).any():
        raise ValueError("盘中数据存在重复分钟")
    minute_number = normalized["timestamp"].dt.hour * 60 + normalized["timestamp"].dt.minute
    for ticker, group in normalized.assign(_minute=minute_number).groupby("ticker", sort=False):
        observed = set(group["_minute"].astype(int))
        if observed != expected_numbers:
            raise ValueError(f"{ticker} 不是09:31至{cutoff.strftime('%H:%M')}完整分钟")
    return IntradayAudit(
        status="PROXY",
        complete_minutes=len(expected_numbers),
        ticker_count=len(expected_tickers),
        cutoff=cutoff.strftime("%H:%M"),
    )


def _parse_quote_lines(payload: bytes) -> list[dict]:
    text = payload.decode("gbk", errors="ignore")
    records: list[dict] = []
    for line in text.splitlines():
        fields = line.split("~")
        if len(fields) < 61:
            continue
        try:
            records.append(
                {
                    "ticker": fields[2].zfill(6),
                    "name": fields[1],
                    "price": float(fields[3] or 0),
                    "prevclose": float(fields[4] or 0),
                    "quote_time": fields[30],
                    "upper_limit": float(fields[47] or 0),
                    "lower_limit": float(fields[48] or 0),
                    "float_mcap": float(fields[44] or 0) * 100_000_000.0,
                }
            )
        except (ValueError, IndexError):
            continue
    return records


def fetch_tencent_quotes(tickers: list[str], *, chunk_size: int = 250) -> pd.DataFrame:
    records: list[dict] = []
    for start in range(0, len(tickers), chunk_size):
        symbols = [_market_symbol(item) for item in tickers[start : start + chunk_size]]
        records.extend(_parse_quote_lines(_read_url(TENCENT_QUOTE_URL + ",".join(symbols))))
    frame = pd.DataFrame(records)
    if frame.empty:
        raise RuntimeError("腾讯全市场快照为空")
    return frame.drop_duplicates("ticker", keep="last").set_index("ticker")


def _fetch_one_tencent_minute(ticker: str, trade_day: date, cutoff: time) -> pd.DataFrame:
    url = f"{TENCENT_MINUTE_URL}?{urlencode({'code': _market_symbol(ticker)})}"
    return parse_tencent_cumulative_minutes(_read_url(url), ticker, trade_day, cutoff=cutoff)


def fetch_tencent_minutes(
    tickers: list[str],
    trade_day: date,
    cutoff: time,
    *,
    workers: int = 48,
) -> tuple[pd.DataFrame, dict[str, str]]:
    frames: list[pd.DataFrame] = []
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {
            executor.submit(_fetch_one_tencent_minute, ticker, trade_day, cutoff): ticker
            for ticker in tickers
        }
        for future in as_completed(pending):
            ticker = pending[future]
            try:
                frames.append(future.result())
            except Exception as error:  # noqa: BLE001 - 汇总所有缺失股票后统一拒绝
                failures[ticker] = str(error)
    if not frames:
        return pd.DataFrame(), failures
    return pd.concat(frames, ignore_index=True), failures


def _query_recent_daily(trade_day: date) -> pd.DataFrame:
    api = BigQuantSdkClient()
    end = pd.Timestamp(trade_day) - pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=10)
    raw = api.query(
        """
        SELECT date, instrument, high, close, amount, upper_limit
        FROM cn_stock_bar1d
        ORDER BY date, instrument
        """,
        filters={"date": [start.date().isoformat(), end.date().isoformat()]},
    )
    if raw.empty:
        raise RuntimeError("BigQuant 上一交易日日线为空")
    raw = raw.copy()
    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
    raw["ticker"] = raw["instrument"].astype(str).str.split(".").str[0].str.zfill(6)
    raw[["high", "close", "amount", "upper_limit"]] = raw[
        ["high", "close", "amount", "upper_limit"]
    ].apply(pd.to_numeric, errors="coerce")
    return raw.dropna(subset=["high", "close", "amount", "upper_limit"])


def build_intraday_universe(
    store: C2ADataStore,
    trade_day: date,
    quotes: pd.DataFrame,
) -> pd.DataFrame:
    """把上一交易日已审计 Universe 前滚一天，并用上一日线更新盘前字段。"""

    history = store.read_universe(end_date=trade_day)
    prior_day = pd.Timestamp(history["trade_date"].max()).normalize()
    if prior_day >= pd.Timestamp(trade_day):
        return history.loc[history["trade_date"].eq(pd.Timestamp(trade_day))].copy()
    previous = history.loc[history["trade_date"].eq(prior_day)].copy().set_index("ticker")
    daily = _query_recent_daily(trade_day).sort_values(["ticker", "date"])
    latest_date = daily["date"].max()
    if latest_date != pd.Timestamp(trade_day) - pd.offsets.BDay(1):
        raise RuntimeError(f"BigQuant 上一交易日日线截止异常: {latest_date.date().isoformat()}")
    recent = daily.groupby("ticker", sort=False).tail(3)
    avg3 = recent.groupby("ticker")["amount"].mean()
    latest = daily.loc[daily["date"].eq(latest_date)].set_index("ticker")
    common = previous.index.intersection(latest.index).intersection(quotes.index)
    if len(common) < 4_000:
        raise RuntimeError(f"盘中 Universe 覆盖不足: {len(common)}")
    output = previous.loc[common].copy()
    quote_rows = quotes.loc[common]
    daily_rows = latest.loc[common]
    output["trade_date"] = pd.Timestamp(trade_day)
    output["name"] = quote_rows["name"].to_numpy()
    output["listing_trading_days"] = output["listing_trading_days"].astype(int) + 1
    output["prevclose"] = quote_rows["prevclose"].to_numpy(dtype=float)
    output["prevhigh"] = daily_rows["high"].to_numpy(dtype=float)
    output["avg3_amount"] = output.index.map(avg3).astype(float)
    output["float_mcap"] = quote_rows["float_mcap"].to_numpy(dtype=float)
    output["float_shares"] = output["float_mcap"] / output["prevclose"].replace(0, np.nan)
    output["upper_limit"] = quote_rows["upper_limit"].to_numpy(dtype=float)
    output["lower_limit"] = quote_rows["lower_limit"].to_numpy(dtype=float)
    output["is_st"] = output["name"].str.contains("ST", case=False, regex=False)
    quote_dates = quote_rows["quote_time"].astype(str).str[:8]
    output["is_suspended"] = (
        ~quote_dates.eq(trade_day.strftime("%Y%m%d")) | quote_rows["price"].le(0).to_numpy()
    )
    was_limit_up = daily_rows["close"].ge(daily_rows["upper_limit"] - 0.005)
    output["limit_streak"] = np.where(
        was_limit_up.to_numpy(), output["limit_streak"].fillna(0).astype(int) + 1, 0
    )
    output = output.reset_index()
    return normalize_universe(output)


def compute_intraday_features(
    bars: pd.DataFrame,
    universe: pd.DataFrame,
    cache: C2AResearchCache,
    params: C2AParameters,
    cutoff: time,
) -> pd.DataFrame:
    state = cache._load_state()  # noqa: SLF001 - 同包实时只读适配器复用已审计滚动状态
    if state is None or state.last_processed_date is None:
        raise RuntimeError("C2-A 20日滚动基线不存在")
    stocks = normalize_universe(universe).set_index("ticker")
    symbols = sorted(set(bars["ticker"].astype(str)))
    state_index = {ticker: index for index, ticker in enumerate(state.tickers)}
    missing_state = sorted(set(symbols) - set(state_index))
    if missing_state:
        raise RuntimeError(f"C2-A 基线缺少股票: {missing_state[:10]}")
    minute_count = cutoff.hour * 60 + cutoff.minute - (9 * 60 + 31) + 1
    frame = bars.copy()
    frame["minute_index"] = (
        frame["timestamp"].dt.hour * 60 + frame["timestamp"].dt.minute - (9 * 60 + 31)
    )

    def pivot(column: str) -> np.ndarray:
        return (
            frame.pivot(index="ticker", columns="minute_index", values=column)
            .reindex(index=symbols, columns=range(minute_count))
            .to_numpy(dtype=float)
        )

    current_amount = np.cumsum(pivot("amount"), axis=1)
    current_volume = np.cumsum(pivot("volume"), axis=1)
    current_close = pivot("close")
    indices = np.asarray([state_index[item] for item in symbols], dtype=np.int64)
    if np.any(state.counts[indices] < params.baseline_days):
        missing = [symbols[i] for i in np.flatnonzero(state.counts[indices] < params.baseline_days)]
        raise RuntimeError(f"C2-A 股票缺少20日基线: {missing[:10]}")
    amount_baseline = np.median(state.amount_history[indices, :minute_count, :], axis=2)
    volume_baseline = np.median(state.volume_history[indices, :minute_count, :], axis=2)
    stock_rows = stocks.loc[symbols]
    amount_burst = current_amount / np.where(amount_baseline == 0, np.nan, amount_baseline)
    turnover = current_volume / np.where(volume_baseline == 0, np.nan, volume_baseline)
    gain = current_close / stock_rows["prevclose"].to_numpy(dtype=float)[:, None] - 1.0
    timestamps = pd.date_range(
        f"{universe['trade_date'].iloc[0].date().isoformat()} 09:31",
        periods=minute_count,
        freq="1min",
    )
    features = pd.DataFrame(
        {
            "timestamp": np.tile(timestamps.to_numpy(), len(symbols)),
            "ticker": np.repeat(symbols, minute_count),
            "trade_date": pd.Timestamp(universe["trade_date"].iloc[0]).normalize(),
            "pool": np.repeat(stock_rows["pool"].to_numpy(), minute_count),
            "amount_burst": amount_burst.reshape(-1),
            "turnover_metric": turnover.reshape(-1),
            "gain": gain.reshape(-1),
        }
    ).dropna()
    features = rank_c2a_cross_section(features, params, ["timestamp", "pool"])
    main = features["pool"].eq("MAIN")
    gain_min = np.where(main, params.main_gain_min, params.growth_gain_min)
    gain_max = np.where(main, params.main_gain_max, params.growth_gain_max)
    features["universe_pass"] = True
    features["signal_pass"] = (
        features["gain"].ge(gain_min)
        & features["gain"].le(gain_max)
        & features["c6"].lt(params.c6_threshold)
    )
    return features.sort_values(["timestamp", "ticker"]).reset_index(drop=True)


def baseline_ready_tickers(
    cache: C2AResearchCache,
    tickers: set[str],
    baseline_days: int,
) -> tuple[set[str], set[str]]:
    """按缓存正式口径分离可计算与基线不足股票，不把缺失值补成零。"""

    state = cache._load_state()  # noqa: SLF001 - 同包实时只读适配器复用滚动状态
    if state is None:
        raise RuntimeError("C2-A 20日滚动基线不存在")
    index = {ticker: position for position, ticker in enumerate(state.tickers)}
    ready = {
        ticker
        for ticker in tickers
        if ticker in index and int(state.counts[index[ticker]]) >= baseline_days
    }
    return ready, tickers - ready


def _parse_eastmoney_ohlc(payload: bytes, ticker: str, trade_day: date) -> pd.DataFrame:
    parsed = json.loads(payload.decode("utf-8"))
    trends = (parsed.get("data") or {}).get("trends", [])
    records: list[dict] = []
    for row in trends:
        fields = str(row).split(",")
        if len(fields) < 8:
            continue
        timestamp = pd.Timestamp(fields[0])
        if timestamp.date() != trade_day or timestamp.time() < time(9, 31):
            continue
        records.append(
            {
                "timestamp": timestamp,
                "ticker": str(ticker).zfill(6),
                "open": float(fields[1]),
                "close": float(fields[2]),
                "high": float(fields[3]),
                "low": float(fields[4]),
                "volume": float(fields[5]) * 100.0,
                "amount": float(fields[6]),
            }
        )
    return normalize_minutes(pd.DataFrame(records)) if records else pd.DataFrame()


def _fetch_one_eastmoney(ticker: str, trade_day: date) -> pd.DataFrame:
    market = "1" if _market_symbol(ticker).startswith("sh") else "0"
    query = {
        "secid": f"{market}.{str(ticker).zfill(6)}",
        "fields1": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58",
        "ndays": 1,
        "iscr": 0,
    }
    return _parse_eastmoney_ohlc(
        _read_url(f"{EASTMONEY_TREND_URL}?{urlencode(query)}"), ticker, trade_day
    )


def fetch_candidate_ohlc(
    tickers: list[str],
    trade_day: date,
    *,
    workers: int = 8,
    retry_workers: int = 2,
) -> pd.DataFrame:
    """低并发获取候选 OHLC，并在首轮结束后只重试失败项。"""

    remaining = list(dict.fromkeys(tickers))
    frames: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}
    worker_plan = (workers, retry_workers)
    for worker_count in worker_plan:
        if not remaining:
            break
        failed: list[str] = []
        with ThreadPoolExecutor(max_workers=min(worker_count, len(remaining))) as executor:
            pending = {
                executor.submit(_fetch_one_eastmoney, ticker, trade_day): ticker
                for ticker in remaining
            }
            for future in as_completed(pending):
                ticker = pending[future]
                try:
                    frame = future.result()
                    if frame.empty:
                        raise RuntimeError(f"东方财富未返回候选分钟: {ticker}")
                except Exception as error:  # noqa: BLE001 - 首轮失败项需降并发重试
                    failures[ticker] = str(error)
                    failed.append(ticker)
                else:
                    frames[ticker] = frame
                    failures.pop(ticker, None)
        remaining = failed
        if remaining and worker_count == workers:
            time_module.sleep(0.5 + random.uniform(0.0, 0.25))
    if remaining:
        sample = {ticker: failures[ticker] for ticker in sorted(remaining)[:10]}
        raise RuntimeError(f"东方财富候选分钟请求失败: {sample}")
    ordered = [frames[ticker] for ticker in dict.fromkeys(tickers)]
    return pd.concat(ordered, ignore_index=True) if ordered else pd.DataFrame()


def run_intraday_scan(
    project_root: str | Path,
    *,
    trade_day: date,
    cutoff: time = time(10, 0),
) -> dict:
    started_at = datetime.now(ZoneInfo("Asia/Shanghai"))
    root = Path(project_root).resolve()
    data_root = root / "data/c2a"
    store = C2ADataStore(data_root)
    params = C2AParameters.optimized_challenger()
    cache = C2AResearchCache(data_root / "research_cache", C2AParameters.dynamic_snapshot())
    prior = store.read_universe(end_date=trade_day)
    tickers = sorted(
        prior.loc[prior["trade_date"].eq(prior["trade_date"].max()), "ticker"].astype(str)
    )
    quotes = fetch_tencent_quotes(tickers)
    universe = build_intraday_universe(store, trade_day, quotes)
    eligible = universe.loc[eligible_universe(universe, params) & ~universe["is_suspended"]].copy()
    eligible_tickers = set(eligible["ticker"].astype(str))
    expected, baseline_exclusions = baseline_ready_tickers(
        cache, eligible_tickers, params.baseline_days
    )
    if not expected:
        raise RuntimeError("C2-A 没有具备20日基线的盘中股票")
    eligible = eligible.loc[eligible["ticker"].isin(expected)].copy()
    minute_bars, failures = fetch_tencent_minutes(sorted(expected), trade_day, cutoff)
    if failures:
        sample = dict(list(sorted(failures.items()))[:10])
        raise RuntimeError(f"腾讯盘中分钟不完整: {sample}")
    audit = audit_completed_window(minute_bars, expected, cutoff)
    features = compute_intraday_features(minute_bars, eligible, cache, params, cutoff)
    possible = sorted(set(features.loc[features["signal_pass"], "ticker"].astype(str)))
    positions = []
    events: list[dict] = []
    remaining_cash = 10_000.0
    if possible:
        ohlc = fetch_candidate_ohlc(possible, trade_day)
        end_time = datetime.combine(trade_day, cutoff) + pd.Timedelta(
            minutes=int(params.signal_expiry_minutes or 0)
        )
        ohlc = ohlc.loc[ohlc["timestamp"].le(end_time)].copy()
        positions, remaining_cash, events = simulate_entry_day(
            ohlc,
            features,
            eligible,
            params,
            cash=10_000.0,
            budget=10_000.0,
            data_status="PROXY",
        )
    name_map = universe.set_index("ticker")["name"].to_dict()
    quote_time = quotes["quote_time"].astype(str).max()
    signal_valid_until = datetime.combine(
        trade_day, cutoff, ZoneInfo("Asia/Shanghai")
    ) + pd.Timedelta(minutes=int(params.signal_expiry_minutes or 0))
    timely_run = started_at <= signal_valid_until + pd.Timedelta(minutes=1)
    result_entries = []
    for position in positions:
        latest_price = float(quotes.loc[position.ticker, "price"])
        result_entries.append(
            {
                "ticker": position.ticker,
                "name": name_map.get(position.ticker, ""),
                "entry_time": position.entry_time.isoformat(),
                "simulated_fill_price": position.entry_price,
                "shares": position.shares,
                "simulated_notional": position.entry_value,
                "entry_cost": position.entry_cost,
                "latest_price": latest_price,
                "mark_to_market": position.shares * latest_price,
            }
        )
    return {
        "as_of": trade_day.isoformat(),
        "status": (
            "SIMULATED_ENTRY"
            if positions and timely_run
            else "RECONSTRUCTED_SIMULATED_ENTRY"
            if positions
            else "NO_SIMULATED_SIGNAL"
        ),
        "run_timing": "TIMELY" if timely_run else "RECONSTRUCTED_AFTER_EXPIRY",
        "signal_valid_until": signal_valid_until.isoformat(),
        "current_new_entry_allowed": False,
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "quote_time": quote_time,
        "audit": audit.to_dict(),
        "universe_count": len(expected),
        "candidate_count": len(possible),
        "candidate_tickers": possible,
        "data_quality": {
            "baseline_ready_tickers": len(expected),
            "baseline_excluded_tickers": sorted(baseline_exclusions),
            "baseline_coverage": len(expected) / len(eligible_tickers),
        },
        "entries": result_entries,
        "remaining_cash": remaining_cash,
        "parameters": params.to_dict(),
        "promotion_gate": "FAIL",
        "execution_permission": "PAPER_ONLY",
        "real_trade_authorized": False,
        "event_count": len(events),
        "events": [
            {
                key: (value.isoformat() if isinstance(value, pd.Timestamp) else value)
                for key, value in event.items()
            }
            for event in events
        ],
    }


def _write_json_atomic(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m robot_quant.c2a_intraday")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--cutoff", default="10:00")
    parser.add_argument("--output", default="data/c2a_results/intraday_latest.json")
    args = parser.parse_args()
    hour, minute = (int(item) for item in args.cutoff.split(":"))
    payload = run_intraday_scan(
        args.project_root,
        trade_day=date.fromisoformat(args.date),
        cutoff=time(hour, minute),
    )
    destination = Path(args.project_root).resolve() / args.output
    _write_json_atomic(payload, destination)
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
