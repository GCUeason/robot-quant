"""BigQuant 数据适配器：为 C2-A 提供历史 Universe 与全池分钟分区。"""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol

import pandas as pd

from robot_quant.c2a import C2AParameters, eligible_universe, normalize_minutes, normalize_universe
from robot_quant.c2a_cache import C2AResearchCache
from robot_quant.c2a_data import C2ADataStore, prior_true_streak


BIGQUANT_API_KEY_PATTERN = re.compile(r"^([^\s.]{8,})\.([^\s.]{8,})$")


class BigQuantQueryClient(Protocol):
    def query(self, sql: str, *, filters: dict) -> pd.DataFrame: ...


class BigQuantSdkClient:
    """延迟导入官方 SDK，避免普通回测被可选数据依赖绑死。"""

    def __init__(self, dai_module=None) -> None:
        if dai_module is None:
            try:
                from bigquant import dai as dai_module  # type: ignore[no-redef]
            except ImportError as error:
                raise RuntimeError(
                    "当前解释器未安装 BigQuant SDK；请使用项目的 .venv-bigquant 运行配置"
                ) from error
        self._dai = dai_module

    def query(self, sql: str, *, filters: dict) -> pd.DataFrame:
        return self._dai.query(sql, filters=filters).df()


def default_bigquant_config_path() -> Path:
    configured = os.getenv("BIGQUANT_CONFIG_FILE")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".bigquant" / "config.json"


def save_bigquant_api_key(value: str, path: str | Path | None = None) -> Path:
    """原子保存官方 `AK.SK` 凭证，并保留配置文件中的非认证项。"""

    match = BIGQUANT_API_KEY_PATTERN.fullmatch(value.strip())
    if match is None:
        raise ValueError("BigQuant API Key 必须是 AK.SK 格式，未保存")
    destination = Path(path).expanduser() if path is not None else default_bigquant_config_path()
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination.parent.chmod(0o700)
    try:
        payload = json.loads(destination.read_text(encoding="utf-8"))
    except FileNotFoundError:
        payload = {}
    except json.JSONDecodeError as error:
        raise RuntimeError(f"BigQuant 配置文件不是有效 JSON: {destination}") from error
    payload["auth"] = {"ak": match.group(1), "sk": match.group(2)}
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            temporary_path.chmod(0o600)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(destination)
        destination.chmod(0o600)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def configure_bigquant_api_key_from_clipboard(path: str | Path | None = None) -> Path:
    """从 macOS 剪贴板配置 AK/SK，不把凭证放入命令参数。"""

    try:
        completed = subprocess.run(["pbpaste"], check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError("无法读取 macOS 剪贴板，请先复制 BigQuant AK.SK") from error
    return save_bigquant_api_key(completed.stdout, path)


def build_bigquant_universe(
    store: C2ADataStore,
    start_date: date | str,
    end_date: date | str,
    *,
    client: BigQuantQueryClient | None = None,
) -> pd.DataFrame:
    """用 BigQuant 盘前静态表和历史日线构建每日无前视 Universe。"""

    api = client or BigQuantSdkClient()
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    history_start = start - timedelta(days=45)
    date_filter = {"date": [history_start.date().isoformat(), end.date().isoformat()]}
    static = api.query(
        """
        SELECT date, instrument, name, pre_close, suspended, st_status, in_delist,
               public_float_share, upper_limit, lower_limit
        FROM cn_stock_static_data
        ORDER BY date, instrument
        """,
        filters=date_filter,
    )
    factors = api.query(
        """
        SELECT date, instrument, list_date, trading_days
        FROM cn_stock_factors_base
        ORDER BY date, instrument
        """,
        filters=date_filter,
    )
    daily = api.query(
        """
        SELECT date, instrument, high, close, volume, amount, upper_limit
        FROM cn_stock_bar1d
        ORDER BY date, instrument
        """,
        filters=date_filter,
    )
    for table_name, frame in (("static", static), ("factors", factors), ("daily", daily)):
        if frame.empty:
            raise RuntimeError(f"BigQuant {table_name} 在目标区间返回空数据")

    static = _normalize_keys(static)
    factors = _normalize_keys(factors)
    daily = _normalize_keys(daily)
    daily_numeric = ("high", "close", "volume", "amount", "upper_limit")
    daily[list(daily_numeric)] = daily[list(daily_numeric)].apply(pd.to_numeric, errors="coerce")
    daily = daily.sort_values(["instrument", "date"])
    daily_groups = daily.groupby("instrument", sort=False)
    daily["prevhigh"] = daily_groups["high"].transform(_prior_valid_high)
    daily["avg3_amount"] = daily_groups["amount"].transform(_prior_valid_amount_mean)
    daily["was_limit_up"] = daily["close"].ge(daily["upper_limit"] - 0.005)
    daily["limit_streak"] = daily_groups["was_limit_up"].transform(prior_true_streak)
    daily["has_daily_trade"] = (
        daily["volume"].gt(0)
        & daily["amount"].notna()
        & daily["high"].notna()
        & daily["close"].notna()
    )
    daily_metrics = daily[
        [
            "date",
            "instrument",
            "prevhigh",
            "avg3_amount",
            "limit_streak",
            "has_daily_trade",
        ]
    ]

    factors = factors[["date", "instrument", "list_date", "trading_days"]]
    merged = static.merge(
        factors,
        on=["date", "instrument"],
        how="left",
        validate="one_to_one",
    ).merge(
        daily_metrics,
        on=["date", "instrument"],
        how="left",
        validate="one_to_one",
    )
    merged["ticker"] = merged["instrument"].str.split(".").str[0].str.zfill(6)
    merged["pool"] = merged["ticker"].map(_pool_for_ticker)
    merged = merged.loc[merged["pool"].notna()].copy()
    merged["is_suspended"] = merged["suspended"].fillna(0).astype(int).ne(0) | ~merged[
        "has_daily_trade"
    ].fillna(False).astype(bool)
    merged["list_date"] = pd.to_datetime(merged["list_date"], errors="coerce")
    numeric = (
        "pre_close",
        "public_float_share",
        "upper_limit",
        "lower_limit",
        "trading_days",
        "prevhigh",
        "avg3_amount",
        "limit_streak",
    )
    merged[list(numeric)] = merged[list(numeric)].apply(pd.to_numeric, errors="coerce")
    supported = merged["pool"].notna()
    active = ~merged["is_suspended"]
    incomplete = (
        merged.loc[
            supported & active,
            [
                "list_date",
                "trading_days",
                "pre_close",
                "public_float_share",
                "upper_limit",
                "lower_limit",
            ],
        ]
        .isna()
        .any(axis=1)
    )
    incomplete_target = incomplete & merged.loc[supported & active, "date"].ge(start).to_numpy()
    if bool(incomplete_target.any()):
        count = int(incomplete_target.sum())
        raise RuntimeError(f"BigQuant Universe 有 {count} 条非停牌记录缺少必要历史字段")

    names = merged["name"].fillna("").astype(str)
    merged["trade_date"] = merged["date"]
    merged["listing_trading_days"] = merged["trading_days"]
    merged["prevclose"] = merged["pre_close"]
    merged["float_shares"] = merged["public_float_share"]
    merged["float_mcap"] = merged["float_shares"] * merged["prevclose"]
    merged["is_st"] = merged["st_status"].fillna(0).astype(int).ne(0) | names.str.contains(
        "ST", case=False, regex=False
    )
    output = merged.loc[
        merged["trade_date"].between(start, end),
        [
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
        ],
    ].copy()
    normalized = normalize_universe(output)
    store.initialize(
        "bigquant:cn_stock_bar1m_c+cn_stock_static_data+cn_stock_factors_base+cn_stock_bar1d",
        metadata_verified=True,
        full_market=True,
    )
    store.write_universe(normalized)
    return normalized


def _prior_valid_high(values: pd.Series) -> pd.Series:
    """取严格早于当前行的最近一个有效最高价，跳过停牌空值。"""

    valid = values.dropna()
    return valid.shift(1).reindex(values.index)


def _prior_valid_amount_mean(values: pd.Series) -> pd.Series:
    """按此前三个实际有成交的交易日计算均额，不让停牌空值占窗口。"""

    valid = values.dropna()
    return valid.shift(1).rolling(3, min_periods=3).mean().reindex(values.index)


def download_bigquant_minutes(
    store: C2ADataStore,
    start_date: date | str,
    end_date: date | str,
    *,
    params: C2AParameters | None = None,
    client: BigQuantQueryClient | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict:
    """按日下载所有曾进入策略 Universe 的股票，现有分区只补缺失代码。"""

    api = client or BigQuantSdkClient()
    config = params or C2AParameters.dynamic_snapshot()
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    universe = store.read_universe(start, end)
    relevant = set(universe.loc[eligible_universe(universe, config), "ticker"].astype(str))
    if not relevant:
        raise RuntimeError("BigQuant Universe 在目标区间没有符合 C2-A 基础门槛的股票")
    downloaded = 0
    updated = 0
    rows_written = 0
    processed = 0
    for trade_day, day_universe in universe.groupby("trade_date", sort=True):
        expected = set(
            day_universe.loc[
                day_universe["ticker"].isin(relevant) & ~day_universe["is_suspended"].astype(bool),
                "ticker",
            ].astype(str)
        )
        if not expected:
            continue
        path = store.minute_path(pd.Timestamp(trade_day).date())
        existing = normalize_minutes(pd.read_csv(path)) if path.exists() else pd.DataFrame()
        observed = set(existing["ticker"].astype(str)) if not existing.empty else set()
        missing = expected.difference(observed)
        if not missing:
            continue
        day_text = pd.Timestamp(trade_day).date().isoformat()
        raw = api.query(
            """
            SELECT date, instrument, open, high, low, close, volume, amount
            FROM cn_stock_bar1m_c
            ORDER BY date, instrument
            """,
            filters={
                "date": [f"{day_text} 09:31:00", f"{day_text} 15:00:00"],
                "instrument": [_to_bigquant_instrument(ticker) for ticker in sorted(missing)],
            },
        )
        if raw.empty:
            raise RuntimeError(f"BigQuant 分钟表未返回 {day_text} 的缺失股票数据")
        bars = raw.rename(columns={"date": "timestamp", "instrument": "ticker"}).copy()
        bars["ticker"] = bars["ticker"].astype(str).str.split(".").str[0].str.zfill(6)
        bars = normalize_minutes(bars)
        _validate_bigquant_day(bars, missing, pd.Timestamp(trade_day))
        store.append_minutes(bars)
        rows_written += len(bars)
        if existing.empty:
            downloaded += 1
        else:
            updated += 1
        processed += 1
        if progress_callback is not None and processed % 10 == 0:
            progress_callback(f"C2-A BigQuant分钟下载：已完成 {processed} 个交易日")
    return {
        "downloaded_partitions": downloaded,
        "updated_partitions": updated,
        "minute_rows_written": rows_written,
    }


def stream_bigquant_minutes_to_cache(
    store: C2ADataStore,
    cache: C2AResearchCache,
    start_date: date | str,
    end_date: date | str,
    *,
    client: BigQuantQueryClient | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict:
    """逐日校验全市场240根后只保存研究缓存，避免全年原始分钟表挤满空间。"""

    api = client or BigQuantSdkClient()
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    universe = store.read_universe(start, end)
    if universe.empty:
        raise RuntimeError("BigQuant Universe 在流式目标区间为空")
    relevant = set(universe.loc[eligible_universe(universe, cache.params), "ticker"].astype(str))
    if not relevant:
        raise RuntimeError("BigQuant Universe 在流式目标区间没有符合 C2-A 基础门槛的股票")
    last_processed = cache.last_processed_date()
    processed = 0
    raw_rows = 0
    compact_rows = 0
    possible_tickers = 0
    for trade_day, day_universe in universe.groupby("trade_date", sort=True):
        day = pd.Timestamp(trade_day).normalize()
        if last_processed is not None and day <= last_processed:
            continue
        expected = set(
            day_universe.loc[
                day_universe["ticker"].isin(relevant) & ~day_universe["is_suspended"].astype(bool),
                "ticker",
            ].astype(str)
        )
        if not expected:
            raise RuntimeError(f"BigQuant {day.date().isoformat()} 没有可校验的股票")
        day_text = day.date().isoformat()
        raw = api.query(
            """
            SELECT date, instrument, open, high, low, close, volume, amount
            FROM cn_stock_bar1m_c
            ORDER BY date, instrument
            """,
            filters={
                "date": [f"{day_text} 09:31:00", f"{day_text} 15:00:00"],
                "instrument": [_to_bigquant_instrument(ticker) for ticker in sorted(expected)],
            },
        )
        if raw.empty:
            raise RuntimeError(f"BigQuant 分钟表未返回 {day_text} 的全市场数据")
        bars = raw.rename(columns={"date": "timestamp", "instrument": "ticker"}).copy()
        bars["ticker"] = bars["ticker"].astype(str).str.split(".").str[0].str.zfill(6)
        bars = normalize_minutes(bars)
        _validate_bigquant_day(bars, expected, day)
        result = cache.ingest_validated_day(
            store,
            universe,
            day,
            bars,
            expected_tickers=expected,
            relevant_tickers=relevant,
        )
        processed += int(result["processed"])
        raw_rows += int(result.get("raw_rows_validated", 0))
        compact_rows += int(result.get("compact_rows", 0))
        possible_tickers += int(result.get("possible_tickers", 0))
        if (
            result["processed"]
            and progress_callback is not None
            and (processed % 5 == 0 or day == end)
        ):
            progress_callback(
                f"C2-A BigQuant流式缓存：已完成 {processed} 个交易日，最新 {day_text}"
            )
    return {
        "streamed_partitions": processed,
        "raw_rows_validated": raw_rows,
        "compact_rows_written": compact_rows,
        "possible_ticker_days": possible_tickers,
        "last_processed_date": (
            cache.last_processed_date().date().isoformat()
            if cache.last_processed_date() is not None
            else None
        ),
    }


def _normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    result["instrument"] = result["instrument"].astype(str)
    if result.duplicated(["date", "instrument"]).any():
        raise RuntimeError("BigQuant 数据存在重复 date+instrument")
    return result


def _pool_for_ticker(ticker: str):
    from robot_quant.c2a import pool_for_ticker

    return pool_for_ticker(ticker)


def _to_bigquant_instrument(ticker: str) -> str:
    symbol = str(ticker).zfill(6)
    suffix = "SH" if symbol.startswith(("600", "601", "603", "605", "688", "689")) else "SZ"
    return f"{symbol}.{suffix}"


def _validate_bigquant_day(
    bars: pd.DataFrame,
    expected_tickers: set[str],
    trade_day: pd.Timestamp,
) -> None:
    observed_tickers = set(bars["ticker"].astype(str))
    if observed_tickers != expected_tickers:
        raise RuntimeError(
            f"BigQuant {trade_day.date().isoformat()} 分钟股票覆盖不完整；"
            f"缺失={sorted(expected_tickers - observed_tickers)}"
        )
    morning = set(range(9 * 60 + 31, 11 * 60 + 31))
    afternoon = set(range(13 * 60 + 1, 15 * 60 + 1))
    expected_minutes = morning | afternoon
    minute_number = bars["timestamp"].dt.hour * 60 + bars["timestamp"].dt.minute
    for ticker, ticker_bars in bars.assign(_minute=minute_number).groupby("ticker", sort=False):
        observed_minutes = set(ticker_bars["_minute"].astype(int))
        if len(ticker_bars) != 240 or observed_minutes != expected_minutes:
            raise RuntimeError(
                f"BigQuant {trade_day.date().isoformat()} {ticker} 不是完整240根连续竞价分钟"
            )
