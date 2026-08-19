"""C2-A 本地数据仓、质量审计与 Tushare REST 数据适配器。"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time as time_module
from collections.abc import Callable
from tempfile import NamedTemporaryFile, TemporaryDirectory
from dataclasses import asdict, dataclass
from datetime import date, time, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from robot_quant.c2a import (
    AFTERNOON_FIRST_COMPLETE_MINUTE,
    AFTERNOON_LAST_COMPLETE_MINUTE,
    C2AParameters,
    MORNING_FIRST_COMPLETE_MINUTE,
    MORNING_LAST_COMPLETE_MINUTE,
    eligible_universe,
    normalize_minutes,
    normalize_universe,
    pool_for_ticker,
)


SCHEMA_VERSION = 3
TUSHARE_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_-]{32,128}$")


def default_tushare_token_path() -> Path:
    """返回项目外的本地凭证路径，可用 TUSHARE_TOKEN_FILE 覆盖。"""

    configured = os.getenv("TUSHARE_TOKEN_FILE")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".config" / "robot-quant" / "tushare_token"


def save_tushare_token(token: str, path: str | Path | None = None) -> Path:
    """原子写入仅当前用户可读的 Tushare Token，不返回凭证内容。"""

    value = token.strip()
    if not TUSHARE_TOKEN_PATTERN.fullmatch(value):
        raise ValueError("Tushare Token 格式无效，未保存")
    destination = Path(path).expanduser() if path is not None else default_tushare_token_path()
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination.parent.chmod(0o700)
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
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(destination)
        destination.chmod(0o600)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def configure_tushare_token_from_clipboard(path: str | Path | None = None) -> Path:
    """macOS 从剪贴板配置 Token，避免凭证出现在命令参数或项目文件。"""

    try:
        completed = subprocess.run(
            ["pbpaste"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError("无法读取 macOS 剪贴板，请先复制 Tushare Token") from error
    return save_tushare_token(completed.stdout, path)


def _read_local_tushare_token() -> str | None:
    path = default_tushare_token_path()
    try:
        value = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    if not TUSHARE_TOKEN_PATTERN.fullmatch(value):
        raise RuntimeError(f"本地 Tushare Token 文件格式无效: {path}")
    return value


def _write_gzip_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False, compression="gzip")
    temporary.replace(path)


def _write_json_atomic(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class DataAudit:
    """严格数据门槛的机器可读结果。"""

    status: str
    start_date: str | None
    end_date: str | None
    trading_days: int
    baseline_days_available: int
    universe_rows: int
    minute_rows: int
    source: str | None
    reasons: tuple[str, ...]

    @property
    def execution_permission(self) -> str:
        return "PAPER_ONLY"

    def to_dict(self) -> dict:
        return {**asdict(self), "execution_permission": self.execution_permission}


class C2ADataStore:
    """按交易日保存 gzip CSV，避免强制安装 Parquet 引擎。"""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.minute_dir = self.root / "minutes"
        self.universe_path = self.root / "universe.csv.gz"
        self.manifest_path = self.root / "manifest.json"
        self.import_transaction_path = self.root / "import_transaction.json"

    def initialize(
        self,
        source: str,
        *,
        metadata_verified: bool,
        full_market: bool,
        volume_unit: str = "shares",
        amount_unit: str = "CNY",
        float_shares_unit: str = "shares",
        bar_timestamp_convention: str = "CN_A_SHARE_1MIN_END_TIME",
    ) -> None:
        self.minute_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "source": source,
            "frequency": "1min",
            "price_adjustment": "none",
            "volume_unit": volume_unit,
            "amount_unit": amount_unit,
            "float_shares_unit": float_shares_unit,
            "bar_timestamp_convention": bar_timestamp_convention,
            "metadata_verified": metadata_verified,
            "full_market": full_market,
            "updated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        }
        _write_json_atomic(manifest, self.manifest_path)

    def manifest(self) -> dict:
        if not self.manifest_path.exists():
            return {}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def write_universe(self, universe: pd.DataFrame) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        normalized = normalize_universe(universe)
        if self.universe_path.exists():
            existing = normalize_universe(pd.read_csv(self.universe_path))
            normalized = pd.concat([existing, normalized], ignore_index=True)
            normalized = normalized.drop_duplicates(
                ["trade_date", "ticker"], keep="last"
            ).sort_values(["trade_date", "ticker"])
        _write_gzip_csv_atomic(normalized, self.universe_path)

    def read_universe(
        self, start_date: date | str | None = None, end_date: date | str | None = None
    ) -> pd.DataFrame:
        if not self.universe_path.exists():
            raise FileNotFoundError(f"缺少 Universe: {self.universe_path}")
        frame = pd.read_csv(self.universe_path)
        result = normalize_universe(frame)
        if start_date is not None:
            result = result.loc[result["trade_date"] >= pd.Timestamp(start_date)]
        if end_date is not None:
            result = result.loc[result["trade_date"] <= pd.Timestamp(end_date)]
        return result.reset_index(drop=True)

    def write_minutes(self, minutes: pd.DataFrame) -> list[Path]:
        self.minute_dir.mkdir(parents=True, exist_ok=True)
        normalized = normalize_minutes(minutes)
        normalized["trade_date"] = normalized["timestamp"].dt.normalize()
        paths: list[Path] = []
        for trade_date, day in normalized.groupby("trade_date", sort=True):
            path = self.minute_path(pd.Timestamp(trade_date).date())
            _write_gzip_csv_atomic(day.drop(columns="trade_date"), path)
            paths.append(path)
        return paths

    def append_minutes(self, minutes: pd.DataFrame) -> list[Path]:
        """合并更新日期分区；重复主键保留新数据。"""

        normalized = normalize_minutes(minutes)
        normalized["trade_date"] = normalized["timestamp"].dt.normalize()
        outputs: list[Path] = []
        for trade_date, new_day in normalized.groupby("trade_date", sort=True):
            path = self.minute_path(pd.Timestamp(trade_date).date())
            clean = new_day.drop(columns="trade_date")
            if path.exists():
                old = pd.read_csv(path)
                clean = pd.concat([old, clean], ignore_index=True)
                clean = clean.drop_duplicates(["timestamp", "ticker"], keep="last")
            outputs.extend(self.write_minutes(clean))
        return outputs

    def read_minutes(
        self, start_date: date | str | None = None, end_date: date | str | None = None
    ) -> pd.DataFrame:
        paths = self.minute_paths(start_date, end_date)
        if not paths:
            return pd.DataFrame(
                columns=["timestamp", "ticker", "open", "high", "low", "close", "volume", "amount"]
            )
        return normalize_minutes(
            pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
        )

    def minute_path(self, trade_date: date) -> Path:
        return self.minute_dir / f"{trade_date.isoformat()}.csv.gz"

    def minute_paths(
        self, start_date: date | str | None = None, end_date: date | str | None = None
    ) -> list[Path]:
        start = pd.Timestamp(start_date).date() if start_date is not None else None
        end = pd.Timestamp(end_date).date() if end_date is not None else None
        paths: list[Path] = []
        if not self.minute_dir.exists():
            return paths
        for path in sorted(self.minute_dir.glob("????-??-??.csv.gz")):
            try:
                partition_date = date.fromisoformat(path.name.removesuffix(".csv.gz"))
            except ValueError:
                continue
            if start is not None and partition_date < start:
                continue
            if end is not None and partition_date > end:
                continue
            paths.append(path)
        return paths

    def audit(
        self,
        start_date: date | str,
        end_date: date | str,
        params: C2AParameters | None = None,
    ) -> DataAudit:
        """验证严格频率、元数据、基线长度和扫描窗口全市场覆盖。"""

        config = params or C2AParameters()
        reasons: list[str] = []
        manifest = self.manifest()
        if self.import_transaction_path.exists():
            reasons.append("import_transaction_incomplete")
        if manifest.get("schema_version") != SCHEMA_VERSION:
            reasons.append("manifest_schema_invalid")
        if manifest.get("frequency") != "1min":
            reasons.append("frequency_not_1min")
        if manifest.get("price_adjustment") != "none":
            reasons.append("minute_prices_must_be_unadjusted")
        if manifest.get("volume_unit") != "shares":
            reasons.append("minute_volume_unit_must_be_shares")
        if manifest.get("amount_unit") != "CNY":
            reasons.append("minute_amount_unit_must_be_CNY")
        if manifest.get("float_shares_unit") != "shares":
            reasons.append("float_shares_unit_must_be_shares")
        if manifest.get("bar_timestamp_convention") != "CN_A_SHARE_1MIN_END_TIME":
            reasons.append("bar_timestamp_convention_invalid")
        if not manifest.get("metadata_verified"):
            reasons.append("historical_metadata_not_verified")
        if not manifest.get("full_market"):
            reasons.append("not_full_a_share_market")

        start = pd.Timestamp(start_date).normalize()
        end = pd.Timestamp(end_date).normalize()
        if start > end:
            raise ValueError("审计开始日期不能晚于结束日期")
        try:
            universe_all = self.read_universe(end_date=end)
        except (FileNotFoundError, ValueError):
            universe_all = pd.DataFrame()
            reasons.append("universe_missing_or_invalid")
        available_paths = self.minute_paths(end_date=end)
        path_by_date = {
            pd.Timestamp(date.fromisoformat(path.name.removesuffix(".csv.gz"))): path
            for path in available_paths
        }

        requested_paths = [
            path
            for path in available_paths
            if start.date() <= date.fromisoformat(path.name.removesuffix(".csv.gz")) <= end.date()
        ]
        requested_dates = {
            pd.Timestamp(date.fromisoformat(path.name.removesuffix(".csv.gz")))
            for path in requested_paths
        }
        expected_dates = (
            {
                pd.Timestamp(item)
                for item in universe_all.loc[
                    universe_all["trade_date"].between(start, end), "trade_date"
                ].unique()
            }
            if not universe_all.empty
            else set()
        )
        if expected_dates != requested_dates:
            reasons.append("missing_or_extra_trading_day_partitions")

        required_history: dict[pd.Timestamp, set[str]] = {}
        target_eligible_by_date: dict[pd.Timestamp, set[str]] = {}
        lifecycle_tickers_by_date: dict[pd.Timestamp, set[str]] = {}
        minute_rows = 0
        if not universe_all.empty:
            target_universe = universe_all.loc[
                universe_all["trade_date"].between(start, end)
            ].copy()
            eligible_target = target_universe.loc[eligible_universe(target_universe, config)].copy()
            target_eligible_by_date = {
                pd.Timestamp(day): set(group["ticker"].astype(str))
                for day, group in eligible_target.groupby("trade_date")
            }
            previously_eligible: set[str] = set()
            for target_day in sorted(expected_dates):
                previously_eligible.update(target_eligible_by_date.get(target_day, set()))
                day_rows = target_universe.loc[
                    target_universe["trade_date"].eq(target_day)
                ].set_index("ticker")
                lifecycle_tickers_by_date[target_day] = {
                    ticker
                    for ticker in previously_eligible
                    if ticker in day_rows.index and not bool(day_rows.loc[ticker, "is_suspended"])
                }
            ticker_dates = {
                str(ticker): pd.DatetimeIndex(group["trade_date"].sort_values().unique())
                for ticker, group in universe_all.loc[
                    ~universe_all["is_suspended"].astype(bool)
                ].groupby("ticker")
            }
            for row in eligible_target[["trade_date", "ticker"]].itertuples(index=False):
                trade_day = pd.Timestamp(row.trade_date)
                ticker = str(row.ticker)
                history_dates = ticker_dates[ticker][ticker_dates[ticker] < trade_day][
                    -config.baseline_days :
                ]
                if len(history_dates) < config.baseline_days:
                    reasons.append(
                        f"insufficient_ticker_baseline_days:{ticker}:{trade_day.date().isoformat()}"
                    )
                for history_day in history_dates:
                    required_history.setdefault(pd.Timestamp(history_day), set()).add(ticker)

            missing_pre_start_partition = False
            for history_day in required_history:
                if history_day not in path_by_date:
                    reasons.append(
                        f"missing_ticker_baseline_partition:{history_day.date().isoformat()}"
                    )
                    missing_pre_start_partition |= history_day < start
            if missing_pre_start_partition:
                reasons.append("insufficient_pre_start_baseline_days")

            verification_dates = sorted(expected_dates | set(required_history))
            for day in verification_dates:
                path = path_by_date.get(day)
                if path is None:
                    continue
                is_target_day = day in expected_dates
                try:
                    bars = normalize_minutes(pd.read_csv(path))
                except ValueError:
                    reasons.append(f"invalid_minute_partition:{day.date().isoformat()}")
                    continue
                if is_target_day:
                    minute_rows += len(bars)
                day_stocks = universe_all.loc[universe_all["trade_date"].eq(day)].copy()
                if day_stocks.empty:
                    reasons.append(f"missing_universe_partition:{day.date().isoformat()}")
                    continue
                expected_scan_tickers = target_eligible_by_date.get(
                    day, set()
                ) | required_history.get(day, set())
                scan = bars.loc[
                    bars["timestamp"].dt.time.between(
                        MORNING_FIRST_COMPLETE_MINUTE, config.scan_end
                    )
                ].copy()
                if not expected_scan_tickers.issubset(set(scan["ticker"])):
                    reasons.append(f"scan_universe_incomplete:{day.date().isoformat()}")
                    continue
                observed = scan.groupby("ticker")["timestamp"].nunique()
                expected_scan_labels = _expected_scan_minute_numbers(config.scan_end)
                if any(
                    observed.get(ticker, 0) != len(expected_scan_labels)
                    for ticker in expected_scan_tickers
                ) or _has_invalid_minute_labels(
                    scan.loc[scan["ticker"].isin(expected_scan_tickers)],
                    expected_scan_labels,
                ):
                    reasons.append(f"scan_minute_grid_incomplete:{day.date().isoformat()}")
                    continue
                if not is_target_day:
                    continue
                session = bars.loc[bars["timestamp"].dt.time.map(_is_full_session_minute)]
                session_observed = session.groupby("ticker")["timestamp"].nunique()
                expected_session_labels = _expected_session_minute_numbers()
                target_tickers = lifecycle_tickers_by_date.get(day, set())
                target_rows = bars.loc[bars["ticker"].isin(target_tickers)]
                if any(
                    session_observed.get(ticker, 0) != len(expected_session_labels)
                    for ticker in target_tickers
                ) or _has_invalid_minute_labels(
                    target_rows,
                    expected_session_labels,
                ):
                    reasons.append(f"session_minute_grid_incomplete:{day.date().isoformat()}")

        required_pre_start_dates = {
            day for day in required_history if day < start and day in path_by_date
        }
        baseline_count = len(required_pre_start_dates)

        unique_reasons = tuple(dict.fromkeys(reasons))
        universe_slice = (
            universe_all.loc[universe_all["trade_date"].between(start, end)]
            if not universe_all.empty
            else universe_all
        )
        status = "STRICT" if not unique_reasons else "PROXY"
        if universe_all.empty or not requested_paths:
            status = "DATA_NOT_READY"
        return DataAudit(
            status=status,
            start_date=start.date().isoformat(),
            end_date=end.date().isoformat(),
            trading_days=len(expected_dates),
            baseline_days_available=baseline_count,
            universe_rows=len(universe_slice),
            minute_rows=minute_rows,
            source=manifest.get("source"),
            reasons=unique_reasons,
        )


def _expected_scan_minute_numbers(scan_end) -> set[int]:
    start_minutes = 9 * 60 + 31
    end_minutes = scan_end.hour * 60 + scan_end.minute
    return set(range(start_minutes, end_minutes + 1))


def _is_full_session_minute(value) -> bool:
    return (
        MORNING_FIRST_COMPLETE_MINUTE <= value <= MORNING_LAST_COMPLETE_MINUTE
        or AFTERNOON_FIRST_COMPLETE_MINUTE <= value <= AFTERNOON_LAST_COMPLETE_MINUTE
    )


def _expected_session_minute_numbers() -> set[int]:
    morning = range(9 * 60 + 31, 11 * 60 + 30 + 1)
    afternoon = range(13 * 60 + 1, 15 * 60 + 1)
    return set(morning) | set(afternoon)


def _has_invalid_minute_labels(frame: pd.DataFrame, expected: set[int]) -> bool:
    if frame.empty:
        return False
    timestamps = frame["timestamp"].dt
    minute_numbers = timestamps.hour * 60 + timestamps.minute
    return bool(
        (~minute_numbers.isin(expected)).any()
        or timestamps.second.ne(0).any()
        or timestamps.microsecond.ne(0).any()
    )


class TushareRestClient:
    """无需额外 SDK 的官方 Tushare Pro REST 客户端。"""

    endpoint = "https://api.tushare.pro"

    def __init__(
        self,
        token: str | None = None,
        timeout: int = 60,
        min_interval_seconds: float = 0.14,
        max_retries: int = 3,
    ) -> None:
        self.token = (
            token
            or os.getenv("TUSHARE_TOKEN")
            or os.getenv("TS_TOKEN")
            or _read_local_tushare_token()
        )
        if not self.token:
            raise RuntimeError(
                "未配置 TUSHARE_TOKEN；可先复制网页 Token，再运行 robot-quant c2a-configure-tushare"
            )
        self.timeout = timeout
        self.min_interval_seconds = min_interval_seconds
        self.max_retries = max_retries
        self._last_request_at: float | None = None
        self.session = requests.Session()

    def query(
        self, api_name: str, params: dict | None = None, fields: Iterable[str] = ()
    ) -> pd.DataFrame:
        request_payload = {
            "api_name": api_name,
            "token": self.token,
            "params": params or {},
            "fields": ",".join(fields),
        }
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                response = self.session.post(
                    self.endpoint,
                    json=request_payload,
                    timeout=(10, self.timeout),
                )
                response.raise_for_status()
            except requests.RequestException as error:
                status = error.response.status_code if error.response is not None else None
                if attempt >= self.max_retries or (
                    status is not None and status not in {429, 500, 502, 503, 504}
                ):
                    raise
                time_module.sleep(2**attempt)
                continue
            payload = response.json()
            if payload.get("code") == 0:
                data = payload.get("data") or {}
                return pd.DataFrame(data.get("items") or [], columns=data.get("fields") or [])
            message = str(payload.get("msg") or "")
            rate_limited = any(
                marker in message.lower() for marker in ("频次", "每分钟", "too many", "rate limit")
            )
            if rate_limited and attempt < self.max_retries:
                time_module.sleep(2**attempt)
                continue
            raise RuntimeError(f"Tushare {api_name} 返回错误: {message}")
        raise RuntimeError(f"Tushare {api_name} 请求重试耗尽")

    def _throttle(self) -> None:
        if self._last_request_at is not None:
            elapsed = time_module.monotonic() - self._last_request_at
            remaining = self.min_interval_seconds - elapsed
            if remaining > 0:
                time_module.sleep(remaining)
        self._last_request_at = time_module.monotonic()


def fetch_tushare_minute_range(
    client: TushareRestClient,
    ticker: str,
    start_date: date | str,
    end_date: date | str,
) -> pd.DataFrame:
    """读取单只股票一段1分钟未复权行情；调用方应把区间限制在约一个月内。"""

    ts_code = _to_ts_code(ticker)
    raw = client.query(
        "stk_mins",
        {
            "ts_code": ts_code,
            "freq": "1min",
            "start_date": f"{pd.Timestamp(start_date).date().isoformat()} 09:00:00",
            "end_date": f"{pd.Timestamp(end_date).date().isoformat()} 15:30:00",
        },
        ("ts_code", "trade_time", "open", "close", "high", "low", "vol", "amount"),
    )
    if raw.empty:
        return pd.DataFrame(
            columns=sorted(
                {"timestamp", "ticker", "open", "high", "low", "close", "volume", "amount"}
            )
        )
    result = raw.rename(columns={"trade_time": "timestamp", "vol": "volume"})
    result["ticker"] = ticker.zfill(6)
    if len(result) >= 8_000:
        raise RuntimeError(f"{ticker} 分钟数据达到8000行上限，请缩短下载区间")
    return normalize_minutes(result)


def download_tushare_minutes(
    store: C2ADataStore,
    tickers: Iterable[str],
    start_date: date | str,
    end_date: date | str,
    *,
    client: TushareRestClient | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict:
    """按自然月断点下载原始文件，再合并为每日横截面分区。"""

    api = client or TushareRestClient()
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    raw_root = store.root / "tushare_raw"
    download_manifest_path = raw_root / "download_manifest.json"
    try:
        download_manifest = json.loads(download_manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        download_manifest = {"schema_version": 2, "queried_through": {}, "pending_slices": {}}
    download_manifest["schema_version"] = 2
    queried_through = download_manifest.setdefault("queried_through", {})
    pending_slices = download_manifest.setdefault("pending_slices", {})
    try:
        universe = store.read_universe(start, end)
    except FileNotFoundError:
        universe = pd.DataFrame()
    expected_dates = (
        {
            ticker: set(group.loc[~group["is_suspended"].astype(bool), "trade_date"])
            for ticker, group in universe.groupby("ticker", sort=False)
        }
        if not universe.empty
        else {}
    )
    downloaded = 0
    updated = 0
    skipped = 0
    changed_slices: list[tuple[Path, pd.Timestamp, pd.Timestamp]] = []
    requests_completed = 0
    for ticker in sorted({str(item).zfill(6) for item in tickers}):
        for month_start, month_end in _month_ranges(start, end):
            raw_path = raw_root / month_start.strftime("%Y-%m") / f"{ticker}.csv.gz"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            state_key = f"{month_start.strftime('%Y-%m')}/{ticker}"
            completed_through = (
                pd.Timestamp(queried_through[state_key]).normalize()
                if state_key in queried_through
                else None
            )
            if completed_through is not None and completed_through >= month_end:
                skipped += 1
                continue
            existing = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
            if not existing.empty:
                existing = normalize_minutes(existing)
            fetch_start = month_start
            if completed_through is not None:
                fetch_start = max(month_start, completed_through + timedelta(days=1))
            elif not existing.empty:
                last_timestamp = existing["timestamp"].max()
                fetch_start = last_timestamp.normalize()
                if last_timestamp.time() >= time(15, 0):
                    fetch_start += timedelta(days=1)
            if fetch_start > month_end:
                _merge_pending_slice(
                    pending_slices,
                    state_key,
                    raw_path.relative_to(raw_root),
                    month_start,
                    month_end,
                )
                continue
            new = fetch_tushare_minute_range(api, ticker, fetch_start, month_end)
            requests_completed += 1
            if ticker in expected_dates and not new.empty:
                try:
                    _validate_tushare_minute_slice(
                        new,
                        ticker,
                        {
                            trade_day
                            for trade_day in expected_dates[ticker]
                            if fetch_start <= trade_day <= month_end
                        },
                    )
                except RuntimeError:
                    _write_json_atomic(download_manifest, download_manifest_path)
                    raise
            if new.empty:
                expected = expected_dates.get(ticker)
                has_expected_session = expected is None or any(
                    fetch_start <= trade_day <= month_end for trade_day in expected
                )
                if has_expected_session:
                    _write_json_atomic(download_manifest, download_manifest_path)
                    raise RuntimeError(
                        f"{ticker} {fetch_start.date()}~{month_end.date()} "
                        "存在预期交易日但分钟接口返回空；未提交下载进度"
                    )
                queried_through[state_key] = month_end.date().isoformat()
                skipped += 1
            else:
                combined = pd.concat([existing, new], ignore_index=True)
                combined = combined.drop_duplicates(["timestamp", "ticker"], keep="last")
                combined = normalize_minutes(combined)
                _write_gzip_csv_atomic(combined, raw_path)
                _merge_pending_slice(
                    pending_slices,
                    state_key,
                    raw_path.relative_to(raw_root),
                    month_start,
                    month_end,
                )
                if existing.empty:
                    downloaded += 1
                else:
                    updated += 1
            if requests_completed % 100 == 0:
                _write_json_atomic(download_manifest, download_manifest_path)
            if progress_callback is not None and requests_completed % 250 == 0:
                progress_callback(
                    f"C2-A Tushare分钟下载：本次已完成 {requests_completed} 个股票-月份请求"
                )
    _write_json_atomic(download_manifest, download_manifest_path)
    changed_slices = [
        (
            raw_root / item["raw_path"],
            pd.Timestamp(item["slice_start"]).normalize(),
            pd.Timestamp(item["slice_end"]).normalize(),
        )
        for item in pending_slices.values()
    ]
    if pending_slices:
        pending_start = min(pd.Timestamp(item["slice_start"]) for item in pending_slices.values())
        pending_end = max(pd.Timestamp(item["slice_end"]) for item in pending_slices.values())
        consolidated = consolidate_tushare_raw(
            store,
            min(start, pending_start),
            max(end, pending_end),
            raw_slices=changed_slices,
        )
    else:
        consolidated = {"partitions": 0, "minute_rows": 0}
    for state_key, item in list(pending_slices.items()):
        queried_through[state_key] = item["completed_through"]
        pending_slices.pop(state_key)
    _write_json_atomic(download_manifest, download_manifest_path)
    return {
        "downloaded_files": downloaded,
        "updated_files": updated,
        "skipped_files": skipped,
        **consolidated,
    }


def build_tushare_universe(
    store: C2ADataStore,
    start_date: date | str,
    end_date: date | str,
    *,
    client: TushareRestClient | None = None,
) -> pd.DataFrame:
    """用历史日指标、涨跌停、ST与停牌构建无幸存者偏差的每日 Universe。"""

    api = client or TushareRestClient()
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    calendar_start = start - timedelta(days=45)
    calendar = api.query(
        "trade_cal",
        {
            "exchange": "",
            "start_date": calendar_start.strftime("%Y%m%d"),
            "end_date": end.strftime("%Y%m%d"),
            "is_open": "1",
        },
        ("cal_date", "is_open"),
    )
    if calendar.empty:
        raise RuntimeError("Tushare trade_cal 未返回交易日")
    trading_days = sorted(pd.to_datetime(calendar["cal_date"], format="%Y%m%d").dt.normalize())
    basics: list[pd.DataFrame] = []
    for list_status in ("L", "D", "P"):
        part = api.query(
            "stock_basic",
            {"exchange": "", "list_status": list_status},
            ("ts_code", "symbol", "name", "list_date", "delist_date", "list_status"),
        )
        basics.append(part)
    basic = pd.concat(basics, ignore_index=True).drop_duplicates("ts_code", keep="first")
    if basic.empty:
        raise RuntimeError("Tushare stock_basic 未返回股票列表")
    basic["list_date"] = pd.to_datetime(basic["list_date"], format="%Y%m%d", errors="coerce")
    basic["delist_date"] = pd.to_datetime(basic["delist_date"], format="%Y%m%d", errors="coerce")
    basic = basic.loc[basic["symbol"].astype(str).str.match(r"^\d{6}$")]

    snapshots: list[pd.DataFrame] = []
    universe_raw_root = store.root / "tushare_universe_raw_daily_basic"
    for trade_day in trading_days:
        day_key = trade_day.strftime("%Y%m%d")
        raw_snapshot_path = universe_raw_root / f"{trade_day.date().isoformat()}.csv.gz"
        if raw_snapshot_path.exists():
            snapshots.append(pd.read_csv(raw_snapshot_path))
            continue
        daily = api.query(
            "daily",
            {"trade_date": day_key},
            ("ts_code", "trade_date", "high", "close", "amount"),
        )
        daily_basic = api.query(
            "daily_basic",
            {"trade_date": day_key},
            ("ts_code", "trade_date", "float_share"),
        )
        limits = api.query(
            "stk_limit",
            {"trade_date": day_key},
            ("ts_code", "trade_date", "pre_close", "up_limit", "down_limit"),
        )
        st = api.query(
            "stock_st",
            {"trade_date": day_key},
            ("ts_code", "name", "trade_date"),
        )
        suspended = api.query(
            "suspend_d",
            {"trade_date": day_key, "suspend_type": "S"},
            ("ts_code", "trade_date", "suspend_type", "suspend_timing"),
        )
        if daily.empty:
            raise RuntimeError(f"Tushare daily 未返回 {day_key} 数据，未写入断点")
        if daily_basic.empty:
            raise RuntimeError(f"Tushare daily_basic 未返回 {day_key} 数据，未写入断点")
        if limits.empty:
            raise RuntimeError(f"Tushare stk_limit 未返回 {day_key} 数据，未写入断点")
        suspended_codes = set(suspended["ts_code"]) if not suspended.empty else set()
        expected_codes = set(daily["ts_code"]).difference(suspended_codes)
        for api_name, frame in (("daily_basic", daily_basic), ("stk_limit", limits)):
            coverage = (
                len(expected_codes.intersection(set(frame["ts_code"]))) / len(expected_codes)
                if expected_codes
                else 1.0
            )
            if coverage < 1.0:
                raise RuntimeError(
                    f"Tushare {api_name} {day_key} 覆盖率仅 {coverage:.2%}，未写入断点"
                )
        day = daily.loc[daily["ts_code"].isin(expected_codes)].merge(
            daily_basic,
            on=["ts_code", "trade_date"],
            how="inner",
            validate="one_to_one",
        )
        day = day.merge(
            limits,
            on=["ts_code", "trade_date"],
            how="inner",
            validate="one_to_one",
        )
        day = day.merge(
            basic[["ts_code", "symbol", "name", "list_date", "delist_date"]],
            on="ts_code",
            how="left",
            validate="many_to_one",
        )
        active = day["list_date"].le(trade_day) & (
            day["delist_date"].isna() | day["delist_date"].ge(trade_day)
        )
        day = day.loc[active].copy()
        st_codes = set(st["ts_code"]) if not st.empty else set()
        if not st.empty and "name" in st:
            historical_names = st.drop_duplicates("ts_code").set_index("ts_code")["name"]
            day["name"] = day["ts_code"].map(historical_names).fillna(day["name"])
        day["is_st"] = day["ts_code"].isin(st_codes)
        day["is_suspended"] = False
        _write_gzip_csv_atomic(day, raw_snapshot_path)
        snapshots.append(day)

    history = pd.concat(snapshots, ignore_index=True)
    history["trade_date"] = pd.to_datetime(history["trade_date"], format="%Y%m%d")
    numeric = ["float_share", "pre_close", "up_limit", "down_limit", "high", "close", "amount"]
    history[numeric] = history[numeric].apply(pd.to_numeric, errors="coerce")
    history = history.sort_values(["ts_code", "trade_date"])
    groups = history.groupby("ts_code", sort=False)
    history["prevhigh"] = groups["high"].shift(1)
    history["avg3_amount"] = groups["amount"].transform(
        lambda values: values.shift(1).rolling(3, min_periods=3).mean() * 1_000.0
    )
    history["was_limit_up"] = history["close"].ge(history["up_limit"] - 0.005)
    history["limit_streak"] = groups["was_limit_up"].transform(prior_true_streak)
    history["ticker"] = history["symbol"].astype(str).str.zfill(6)
    history["listing_trading_days"] = groups.cumcount() + 1
    history["pool"] = history["ticker"].map(pool_for_ticker)
    history["float_shares"] = history["float_share"] * 10_000.0
    history["prevclose"] = history["pre_close"]
    history["float_mcap"] = history["float_shares"] * history["prevclose"]
    history["upper_limit"] = history["up_limit"]
    history["lower_limit"] = history["down_limit"]
    output = history.loc[
        history["trade_date"].between(start, end) & history["pool"].notna(),
        sorted(
            {
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
        ),
    ].copy()
    normalized = normalize_universe(output)
    store.initialize(
        "tushare_pro_stk_mins+daily_basic+stk_limit+stock_st+suspend_d",
        metadata_verified=True,
        full_market=True,
    )
    store.write_universe(normalized)
    return normalized


def consolidate_tushare_raw(
    store: C2ADataStore,
    start_date: date | str,
    end_date: date | str,
    *,
    raw_slices: Iterable[tuple[Path, pd.Timestamp, pd.Timestamp]] | None = None,
) -> dict:
    """把变更过的 ticker-month 切片增量合并到 daily gzip 分区。"""

    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    selected_slices = list(raw_slices) if raw_slices is not None else None
    if selected_slices is None:
        selected_slices = []
        for month_start, month_end in _month_ranges(start, end):
            month_dir = store.root / "tushare_raw" / month_start.strftime("%Y-%m")
            if not month_dir.exists():
                continue
            selected_slices.extend(
                (path, month_start, month_end) for path in sorted(month_dir.glob("*.csv.gz"))
            )
    if not selected_slices:
        return {"partitions": 0, "minute_rows": 0}

    rows = 0
    partitions = 0
    store.minute_dir.mkdir(parents=True, exist_ok=True)
    store.root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="c2a-staging-", dir=store.root) as staging_dir:
        staging = Path(staging_dir)
        for raw_path, slice_start, slice_end in selected_slices:
            raw = pd.read_csv(raw_path)
            if raw.empty:
                continue
            frame = normalize_minutes(raw)
            frame = frame.loc[
                frame["timestamp"].dt.normalize().between(slice_start, slice_end)
            ].copy()
            frame["trade_date"] = frame["timestamp"].dt.normalize()
            for trade_date, day in frame.groupby("trade_date", sort=True):
                if not start <= trade_date <= end:
                    continue
                stage_path = staging / f"{trade_date.date().isoformat()}.csv"
                day.drop(columns="trade_date").to_csv(
                    stage_path,
                    mode="a",
                    header=not stage_path.exists(),
                    index=False,
                )
        for stage_path in sorted(staging.glob("????-??-??.csv")):
            output = store.minute_dir / f"{stage_path.stem}.csv.gz"
            pieces = [pd.read_csv(stage_path)]
            if output.exists():
                pieces.insert(0, pd.read_csv(output))
            raw_stage = pd.concat(pieces, ignore_index=True).drop_duplicates(
                ["timestamp", "ticker"], keep="last"
            )
            frame = normalize_minutes(raw_stage)
            _write_gzip_csv_atomic(frame, output)
            rows += len(frame)
            partitions += 1
    return {"partitions": partitions, "minute_rows": rows}


def import_c2a_csv(
    store: C2ADataStore,
    minute_csv: str | Path,
    universe_csv: str | Path,
    *,
    source: str = "user_import",
    metadata_verified: bool = False,
    full_market: bool = False,
) -> dict:
    """分块导入供应商 CSV，避免年度全市场分钟数据一次性进内存。"""

    universe = pd.read_csv(universe_csv)
    normalized_universe = normalize_universe(universe)
    minute_rows = 0
    partitions = 0
    store.root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="c2a-import-", dir=store.root) as staging_dir:
        staging = Path(staging_dir)
        for chunk in pd.read_csv(minute_csv, chunksize=500_000):
            normalized = normalize_minutes(chunk)
            minute_rows += len(normalized)
            normalized["trade_date"] = normalized["timestamp"].dt.normalize()
            for trade_date, day in normalized.groupby("trade_date", sort=False):
                stage_path = staging / f"{pd.Timestamp(trade_date).date().isoformat()}.csv"
                day.drop(columns="trade_date").to_csv(
                    stage_path,
                    mode="a",
                    header=not stage_path.exists(),
                    index=False,
                )
        staged_paths = sorted(staging.glob("????-??-??.csv"))
        if not staged_paths:
            raise ValueError("分钟 CSV 为空")
        _write_json_atomic(
            {
                "status": "COMMITTING",
                "source": source,
                "started_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
                "staged_partitions": len(staged_paths),
            },
            store.import_transaction_path,
        )
        store.write_universe(normalized_universe)
        for stage_path in staged_paths:
            frame = pd.read_csv(stage_path)
            output = store.minute_dir / f"{stage_path.stem}.csv.gz"
            if output.exists():
                frame = pd.concat([pd.read_csv(output), frame], ignore_index=True)
            frame = frame.drop_duplicates(["timestamp", "ticker"], keep="last")
            _write_gzip_csv_atomic(normalize_minutes(frame), output)
            partitions += 1
        store.initialize(source, metadata_verified=metadata_verified, full_market=full_market)
        store.import_transaction_path.unlink()
    return {
        "minute_partitions": partitions,
        "minute_rows": minute_rows,
        "universe_rows": len(normalized_universe),
    }


def _month_ranges(start: pd.Timestamp, end: pd.Timestamp):
    current = start.replace(day=1)
    while current <= end:
        next_month = current + pd.offsets.MonthBegin(1)
        yield max(current, start), min(next_month - timedelta(days=1), end)
        current = next_month


def _merge_pending_slice(
    pending_slices: dict,
    state_key: str,
    relative_raw_path: Path,
    slice_start: pd.Timestamp,
    slice_end: pd.Timestamp,
) -> None:
    """合并同一股票-月的待合并范围，窄区间重启不得覆盖旧断点。"""

    existing = pending_slices.get(state_key)
    if existing:
        slice_start = min(slice_start, pd.Timestamp(existing["slice_start"]).normalize())
        slice_end = max(slice_end, pd.Timestamp(existing["slice_end"]).normalize())
    pending_slices[state_key] = {
        "raw_path": str(relative_raw_path),
        "slice_start": slice_start.date().isoformat(),
        "slice_end": slice_end.date().isoformat(),
        "completed_through": slice_end.date().isoformat(),
    }


def _validate_tushare_minute_slice(
    frame: pd.DataFrame,
    ticker: str,
    expected_dates: set[pd.Timestamp],
) -> None:
    """非空响应也必须覆盖全部预期交易日和240个连续竞价分钟。"""

    bars = normalize_minutes(frame)
    bars["trade_date"] = bars["timestamp"].dt.normalize()
    observed_dates = {pd.Timestamp(item).normalize() for item in bars["trade_date"].unique()}
    if observed_dates != expected_dates:
        missing = sorted(expected_dates.difference(observed_dates))
        extra = sorted(observed_dates.difference(expected_dates))
        raise RuntimeError(
            f"{ticker} 分钟响应交易日不完整；"
            f"缺失={[item.date().isoformat() for item in missing]}，"
            f"额外={[pd.Timestamp(item).date().isoformat() for item in extra]}；未提交下载进度"
        )
    expected_minutes = _expected_session_minute_numbers()
    for trade_day, day_bars in bars.groupby("trade_date", sort=False):
        observed_minutes = set(day_bars["timestamp"].dt.hour * 60 + day_bars["timestamp"].dt.minute)
        if (
            len(day_bars) != len(expected_minutes)
            or observed_minutes != expected_minutes
            or _has_invalid_minute_labels(day_bars, expected_minutes)
        ):
            raise RuntimeError(
                f"{ticker} {pd.Timestamp(trade_day).date().isoformat()} "
                "分钟网格不是完整240根；未提交下载进度"
            )


def _to_ts_code(ticker: str) -> str:
    symbol = str(ticker).zfill(6)
    suffix = "SH" if symbol.startswith(("600", "601", "603", "605", "688", "689")) else "SZ"
    return f"{symbol}.{suffix}"


def prior_true_streak(values: pd.Series) -> pd.Series:
    streaks: list[int] = []
    current = 0
    for value in values.fillna(False).astype(bool):
        streaks.append(current)
        current = current + 1 if value else 0
    return pd.Series(streaks, index=values.index, dtype=int)
