"""C2-A 本地快速盘中扫描。

慢速数据准备与盘中扫描严格分离：

* ``prepare`` 在盘前或前一交易日收盘后准备本地滚动基线；
* ``scan`` 只读取本地基线，并从公共行情源获取当日数据；
* 所有输出固定为 ``PROXY / PAPER_ONLY``，不会连接券商或生成真实交易权限。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time as time_module
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from robot_quant.c2a import C2AParameters, eligible_universe, normalize_universe, simulate_entry_day
from robot_quant.c2a_cache import C2AResearchCache
from robot_quant.c2a_data import C2ADataStore
from robot_quant.c2a_intraday import (
    TENCENT_MINUTE_URL,
    TENCENT_QUOTE_URL,
    _market_symbol,
    _parse_quote_lines,
    audit_completed_window,
    baseline_ready_tickers,
    compute_intraday_features,
    parse_tencent_cumulative_minutes,
)
from robot_quant.c2a_remote import (
    DEFAULT_HOST,
    DEFAULT_REMOTE_ROOT,
    export_remote_fast_pack,
    fetch_remote_fast_pack,
)


SHANGHAI = ZoneInfo("Asia/Shanghai")
FAST_MINUTES = 30
MIN_FAST_COVERAGE = 0.99
DEFAULT_FAST_ROOT = "data/c2a_fast"
DEFAULT_OUTPUT = "data/c2a_results/fast_latest.json"
TENCENT_DAY_URL = "https://web.ifzq.gtimg.cn/appstock/app/day/query"
TENCENT_MKLINE_URL = "https://ifzq.gtimg.cn/appstock/app/kline/mkline"
_HTTP_LOCAL = threading.local()


class FastScanDataError(RuntimeError):
    """快速扫描核心输入不完整。"""


def _fast_get(url: str, *, timeout: float) -> bytes:
    """每个工作线程复用一个HTTP连接池，避免逐股重复握手。"""

    session = getattr(_HTTP_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://gu.qq.com/"})
        _HTTP_LOCAL.session = session
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


@dataclass(frozen=True)
class TencentDay:
    trade_date: date
    previous_close: float
    close: float
    high: float
    amount: float
    cumulative_amount: np.ndarray
    cumulative_volume: np.ndarray


@dataclass
class FastPack:
    root: Path
    tickers: list[str]
    amount_history: np.ndarray
    volume_history: np.ndarray
    pointers: np.ndarray
    counts: np.ndarray
    last_processed_date: pd.Timestamp
    universe: pd.DataFrame
    manifest: dict

    def state_view(self) -> SimpleNamespace:
        return SimpleNamespace(
            tickers=self.tickers,
            amount_history=self.amount_history,
            volume_history=self.volume_history,
            pointers=self.pointers,
            counts=self.counts,
            last_processed_date=self.last_processed_date,
        )


class _PackCacheView:
    """让既有特征函数读取紧凑盘中基线，不复制特征算法。"""

    def __init__(self, pack: FastPack) -> None:
        self._state = pack.state_view()

    def _load_state(self) -> SimpleNamespace:
        return self._state


def _write_json_atomic(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_frame_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False, compression="gzip")
    temporary.replace(path)


def _write_npz_atomic(pack: FastPack, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            tickers=np.asarray(pack.tickers, dtype="U6"),
            amount_history=pack.amount_history,
            volume_history=pack.volume_history,
            pointers=pack.pointers,
            counts=pack.counts,
            last_processed_date=np.asarray(pack.last_processed_date.date().isoformat()),
        )
    temporary.replace(path)


def save_fast_pack(pack: FastPack) -> None:
    pack.root.mkdir(parents=True, exist_ok=True)
    _write_npz_atomic(pack, pack.root / "rolling_state.npz")
    _write_frame_atomic(pack.universe, pack.root / "universe.csv.gz")
    manifest = dict(pack.manifest)
    manifest["last_processed_date"] = pack.last_processed_date.date().isoformat()
    manifest["ticker_count"] = len(pack.tickers)
    manifest["execution_permission"] = "PAPER_ONLY"
    manifest["real_trade_authorized"] = False
    manifest["file_sha256"] = {
        "rolling_state.npz": _sha256_file(pack.root / "rolling_state.npz"),
        "universe.csv.gz": _sha256_file(pack.root / "universe.csv.gz"),
    }
    _write_json_atomic(manifest, pack.root / "manifest.json")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_fast_pack(root: str | Path) -> FastPack:
    pack_root = Path(root).resolve()
    state_path = pack_root / "rolling_state.npz"
    universe_path = pack_root / "universe.csv.gz"
    manifest_path = pack_root / "manifest.json"
    missing = [
        path.name for path in (state_path, universe_path, manifest_path) if not path.exists()
    ]
    if missing:
        raise FastScanDataError(f"本地快速基线缺失: {', '.join(missing)}；先运行 --prepare")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in manifest.get("file_sha256", {}).items():
        candidate = pack_root / name
        if name not in {"rolling_state.npz", "universe.csv.gz"} or not candidate.is_file():
            raise FastScanDataError("本地快速基线哈希清单包含非法文件")
        if _sha256_file(candidate) != expected:
            raise FastScanDataError(f"本地快速基线文件损坏: {name}")
    with np.load(state_path, allow_pickle=False) as payload:
        pack = FastPack(
            root=pack_root,
            tickers=[str(item) for item in payload["tickers"].tolist()],
            amount_history=payload["amount_history"],
            volume_history=payload["volume_history"],
            pointers=payload["pointers"],
            counts=payload["counts"],
            last_processed_date=pd.Timestamp(
                str(payload["last_processed_date"].item())
            ).normalize(),
            universe=normalize_universe(pd.read_csv(universe_path, dtype={"ticker": str})),
            manifest=manifest,
        )
    if pack.amount_history.shape != pack.volume_history.shape:
        raise FastScanDataError("本地快速基线量额数组形状不一致")
    if pack.amount_history.shape[0] != len(pack.tickers):
        raise FastScanDataError("本地快速基线股票索引不一致")
    if pack.amount_history.shape[1] < FAST_MINUTES:
        raise FastScanDataError("本地快速基线不足09:31—10:00")
    return pack


def export_fast_pack(project_root: str | Path, output_root: str | Path) -> dict:
    """从已审计研究缓存导出仅包含早盘窗口的紧凑基线。"""

    project = Path(project_root).resolve()
    destination = (project / output_root).resolve()
    data_root = project / "data/c2a"
    params = C2AParameters.dynamic_snapshot()
    cache = C2AResearchCache(data_root / "research_cache", params)
    state = cache._load_state()  # noqa: SLF001 - 同包导出只读已审计滚动状态
    if state is None or state.last_processed_date is None:
        raise FastScanDataError("远端C2-A滚动基线不存在")
    universe_history = C2ADataStore(data_root).read_universe(end_date=state.last_processed_date)
    latest = universe_history.loc[
        universe_history["trade_date"].eq(state.last_processed_date)
    ].copy()
    if latest.empty:
        raise FastScanDataError("远端C2-A滚动基线缺少同日Universe")
    state_index = {ticker: index for index, ticker in enumerate(state.tickers)}
    latest = latest.loc[latest["ticker"].isin(state_index)].sort_values("ticker")
    positions = np.asarray([state_index[ticker] for ticker in latest["ticker"]], dtype=np.int64)
    tickers = latest["ticker"].astype(str).tolist()
    manifest = {
        "schema_version": 1,
        "source": "bigquant_validated_seed",
        "baseline_status": "PROXY",
        "seed_date": state.last_processed_date.date().isoformat(),
        "scan_window": "09:31-10:00",
        "parameters": C2AParameters.optimized_challenger().to_dict(),
        "promotion_gate": "FAIL",
    }
    pack = FastPack(
        root=destination,
        tickers=tickers,
        amount_history=state.amount_history[positions, :FAST_MINUTES, :].copy(),
        volume_history=state.volume_history[positions, :FAST_MINUTES, :].copy(),
        pointers=state.pointers[positions].copy(),
        counts=state.counts[positions].copy(),
        last_processed_date=state.last_processed_date,
        universe=latest.reset_index(drop=True),
        manifest=manifest,
    )
    save_fast_pack(pack)
    return {
        "status": "READY",
        "last_processed_date": state.last_processed_date.date().isoformat(),
        "ticker_count": len(tickers),
        "output_root": str(destination),
    }


def parse_tencent_day_history(payload: bytes | str, ticker: str) -> dict[date, TencentDay]:
    body = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    parsed = json.loads(body)
    symbol = _market_symbol(ticker)
    days = (parsed.get("data", {}).get(symbol, {}) or {}).get("data", [])
    result: dict[date, TencentDay] = {}
    expected_scan = [time(9, 30), *pd.date_range("2000-01-01 09:31", periods=30, freq="1min").time]
    expected_keys = {item.strftime("%H%M") for item in expected_scan}
    for day in days:
        try:
            trade_day = datetime.strptime(str(day["date"]), "%Y%m%d").date()
            previous_close = float(day["prec"])
        except (KeyError, TypeError, ValueError):
            continue
        rows: dict[str, tuple[float, float, float]] = {}
        for raw in day.get("data", []):
            fields = str(raw).split()
            if len(fields) < 4 or len(fields[0]) != 4:
                continue
            try:
                rows[fields[0]] = (float(fields[1]), float(fields[2]), float(fields[3]))
            except ValueError:
                continue
        if not expected_keys.issubset(rows):
            continue
        auction = rows["0930"]
        scan_rows = [rows[item.strftime("%H%M")] for item in expected_scan[1:]]
        cumulative_volume = np.asarray(
            [(item[1] - auction[1]) * 100.0 for item in scan_rows], dtype=np.float64
        )
        cumulative_amount = np.asarray(
            [item[2] - auction[2] for item in scan_rows], dtype=np.float64
        )
        if (
            not np.isfinite(cumulative_volume).all()
            or not np.isfinite(cumulative_amount).all()
            or (cumulative_volume < 0).any()
            or (cumulative_amount < 0).any()
        ):
            continue
        session_rows = {
            key: value
            for key, value in rows.items()
            if "0930" <= key <= "1130" or "1300" <= key <= "1500"
        }
        if "1500" not in session_rows:
            continue
        result[trade_day] = TencentDay(
            trade_date=trade_day,
            previous_close=previous_close,
            close=session_rows["1500"][0],
            high=max(item[0] for item in session_rows.values()),
            amount=session_rows["1500"][2],
            cumulative_amount=cumulative_amount,
            cumulative_volume=cumulative_volume,
        )
    return result


def _fetch_one_day_history(ticker: str, *, timeout: float, retries: int) -> dict[date, TencentDay]:
    url = f"{TENCENT_DAY_URL}?{urlencode({'code': _market_symbol(ticker)})}"
    last_error: Exception | None = None
    for _ in range(retries):
        try:
            return parse_tencent_day_history(_fast_get(url, timeout=timeout), ticker)
        except Exception as error:  # noqa: BLE001 - prepare阶段按调用方次数重试
            last_error = error
    assert last_error is not None
    raise last_error


def fetch_day_histories(
    tickers: list[str],
    *,
    workers: int = 128,
) -> tuple[dict[str, dict[date, TencentDay]], dict[str, str]]:
    """并发抓取五日分钟；首轮失败项用低并发重试。"""

    remaining = list(dict.fromkeys(tickers))
    histories: dict[str, dict[date, TencentDay]] = {}
    errors: dict[str, str] = {}
    for worker_count, timeout, retries in ((workers, 4.0, 1), (32, 8.0, 2)):
        if not remaining:
            break
        failed: list[str] = []
        with ThreadPoolExecutor(max_workers=min(worker_count, len(remaining))) as executor:
            pending = {
                executor.submit(
                    _fetch_one_day_history,
                    ticker,
                    timeout=timeout,
                    retries=retries,
                ): ticker
                for ticker in remaining
            }
            for future in as_completed(pending):
                ticker = pending[future]
                try:
                    history = future.result()
                    if not history:
                        raise FastScanDataError("腾讯五日分钟为空")
                except Exception as error:  # noqa: BLE001 - 汇总失败项统一审计
                    errors[ticker] = str(error)
                    failed.append(ticker)
                else:
                    histories[ticker] = history
                    errors.pop(ticker, None)
        remaining = failed
    return histories, errors


def _is_limit_up(record: TencentDay, pool: str) -> bool:
    ratio = Decimal("0.10") if pool == "MAIN" else Decimal("0.20")
    upper = (Decimal(str(record.previous_close)) * (Decimal("1") + ratio)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    return record.close >= float(upper) - 0.005


def update_fast_pack_from_tencent(pack: FastPack, target: date) -> dict:
    """用腾讯已完成交易日补齐紧凑滚动基线和下一日Universe字段。"""

    if pd.Timestamp(target) <= pack.last_processed_date:
        return {
            "status": "CURRENT",
            "last_processed_date": pack.last_processed_date.date().isoformat(),
            "updated_days": 0,
            "coverage": 1.0,
        }
    histories, failures = fetch_day_histories(pack.tickers)
    market_days = sorted(
        {
            day
            for history in histories.values()
            for day in history
            if pack.last_processed_date.date() < day <= target
        }
    )
    if not market_days or market_days[-1] != target:
        raise FastScanDataError(
            f"腾讯五日分钟无法从{pack.last_processed_date.date()}连续补到{target}"
        )
    if len(market_days) > 5:
        raise FastScanDataError("快速基线落后超过腾讯五日窗口，需要重新 --bootstrap")
    coverage_by_day = {
        trade_day: sum(trade_day in history for history in histories.values())
        for trade_day in market_days
    }
    coverage = min(coverage_by_day.values()) / len(pack.tickers) if pack.tickers else 0.0
    if coverage < MIN_FAST_COVERAGE:
        raise FastScanDataError(f"腾讯日线增量覆盖不足: {coverage:.2%}")
    index = {ticker: position for position, ticker in enumerate(pack.tickers)}
    updated_by_day: dict[str, int] = {}
    for trade_day in market_days:
        updated = 0
        for ticker, history in histories.items():
            record = history.get(trade_day)
            if record is None:
                continue
            position = index[ticker]
            slot = int(pack.pointers[position])
            pack.amount_history[position, :, slot] = record.cumulative_amount
            pack.volume_history[position, :, slot] = record.cumulative_volume
            pack.pointers[position] = (slot + 1) % pack.amount_history.shape[2]
            pack.counts[position] = min(int(pack.counts[position]) + 1, 20)
            updated += 1
        updated_by_day[trade_day.isoformat()] = updated
    latest = pack.universe.set_index("ticker", drop=False).copy()
    for ticker in pack.tickers:
        history = histories.get(ticker, {})
        record = history.get(target)
        if record is None:
            latest.at[ticker, "is_suspended"] = True
            continue
        recent = [history[day] for day in sorted(history) if day <= target][-3:]
        latest.at[ticker, "trade_date"] = pd.Timestamp(target)
        latest.at[ticker, "listing_trading_days"] = int(
            latest.at[ticker, "listing_trading_days"]
        ) + len(market_days)
        latest.at[ticker, "prevclose"] = record.close
        latest.at[ticker, "prevhigh"] = record.high
        if len(recent) == 3:
            latest.at[ticker, "avg3_amount"] = float(np.mean([item.amount for item in recent]))
        prior_streak = int(latest.at[ticker, "limit_streak"] or 0)
        latest.at[ticker, "limit_streak"] = (
            prior_streak + 1 if _is_limit_up(record, str(latest.at[ticker, "pool"])) else 0
        )
        latest.at[ticker, "is_suspended"] = False
    pack.universe = normalize_universe(latest.reset_index(drop=True))
    pack.last_processed_date = pd.Timestamp(target).normalize()
    eligible_count = int((pack.counts >= 20).sum())
    pack.manifest.update(
        {
            "source": "bigquant_validated_seed+tencent_day_query",
            "baseline_status": "PROXY",
            "public_updates": updated_by_day,
            "public_fetch_failures": len(failures),
            "baseline_ready_tickers": eligible_count,
            "updated_at": datetime.now(SHANGHAI).isoformat(),
        }
    )
    save_fast_pack(pack)
    return {
        "status": "READY",
        "last_processed_date": target.isoformat(),
        "updated_days": len(market_days),
        "updated_by_day": updated_by_day,
        "fetch_failures": len(failures),
        "coverage": coverage,
    }


def latest_completed_market_day(now: datetime | None = None) -> date:
    current = now or datetime.now(SHANGHAI)
    history = _fetch_one_day_history("600000", timeout=8.0, retries=2)
    available = sorted(history)
    if not available:
        raise FastScanDataError("无法确定最近已完成A股交易日")
    if current.time() >= time(15, 5) and current.date() in history:
        return current.date()
    prior = [item for item in available if item < current.date()]
    if not prior:
        raise FastScanDataError("腾讯五日窗口内没有上一交易日")
    return prior[-1]


def bootstrap_fast_pack(
    project_root: str | Path,
    *,
    host: str = DEFAULT_HOST,
    remote_root: str = DEFAULT_REMOTE_ROOT,
) -> None:
    """一次性从远端已审计缓存导出紧凑种子；不在盘中扫描路径调用。"""

    project = Path(project_root).resolve()
    export_remote_fast_pack(
        project,
        host=host,
        remote_root=remote_root,
    )
    fetch_remote_fast_pack(project, host=host, remote_root=remote_root)


def prepare_fast_pack(
    project_root: str | Path,
    *,
    through: date | None = None,
    force_bootstrap: bool = False,
    host: str = DEFAULT_HOST,
    remote_root: str = DEFAULT_REMOTE_ROOT,
) -> dict:
    project = Path(project_root).resolve()
    pack_root = project / DEFAULT_FAST_ROOT
    if force_bootstrap or not (pack_root / "rolling_state.npz").exists():
        bootstrap_fast_pack(project, host=host, remote_root=remote_root)
    pack = load_fast_pack(pack_root)
    target = through or latest_completed_market_day()
    result = update_fast_pack_from_tencent(pack, target)
    if result["status"] == "CURRENT" and "file_sha256" not in pack.manifest:
        save_fast_pack(pack)
    return result


def latest_complete_cutoff(now: datetime) -> time:
    if now.tzinfo is None:
        now = now.replace(tzinfo=SHANGHAI)
    if now.time() < time(9, 32):
        raise FastScanDataError("09:32前尚不足一根可用于确认的完整连续竞价分钟")
    completed = now - timedelta(minutes=1)
    cutoff = completed.time().replace(second=0, microsecond=0)
    if cutoff > time(10, 0):
        cutoff = time(10, 0)
    if cutoff < time(9, 31):
        raise FastScanDataError("当前没有已完成的连续竞价分钟")
    return cutoff


def _fetch_quote_chunk(chunk: list[str], *, timeout: float, retries: int) -> list[dict]:
    symbols = [_market_symbol(ticker) for ticker in chunk]
    url = TENCENT_QUOTE_URL + ",".join(symbols)
    last_error: Exception | None = None
    for _ in range(retries):
        try:
            return _parse_quote_lines(_fast_get(url, timeout=timeout))
        except Exception as error:  # noqa: BLE001 - 失败分片由调用方低并发重试
            last_error = error
    assert last_error is not None
    raise last_error


def fetch_quotes_fast(tickers: list[str], *, chunk_size: int = 250) -> pd.DataFrame:
    chunks = [tickers[start : start + chunk_size] for start in range(0, len(tickers), chunk_size)]
    records: list[dict] = []
    failures: list[list[str]] = []
    with ThreadPoolExecutor(max_workers=min(16, len(chunks))) as executor:
        pending = {
            executor.submit(_fetch_quote_chunk, chunk, timeout=4.0, retries=1): chunk
            for chunk in chunks
        }
        for future in as_completed(pending):
            try:
                records.extend(future.result())
            except Exception:  # noqa: BLE001 - 失败分片低并发重试
                failures.append(pending[future])
    for chunk in failures:
        records.extend(_fetch_quote_chunk(chunk, timeout=8.0, retries=2))
    frame = pd.DataFrame(records)
    if frame.empty:
        raise FastScanDataError("腾讯全市场快照为空")
    frame = frame.drop_duplicates("ticker", keep="last").set_index("ticker")
    coverage = len(frame.index.intersection(tickers)) / len(tickers)
    if coverage < MIN_FAST_COVERAGE:
        raise FastScanDataError(f"腾讯全市场快照覆盖不足: {coverage:.2%}")
    return frame


def _fetch_minute_fast(ticker: str, trade_day: date, cutoff: time, timeout: float) -> pd.DataFrame:
    url = f"{TENCENT_MINUTE_URL}?{urlencode({'code': _market_symbol(ticker)})}"
    return parse_tencent_cumulative_minutes(
        _fast_get(url, timeout=timeout),
        ticker,
        trade_day,
        cutoff=cutoff,
    )


def fetch_minutes_fast(
    tickers: list[str],
    trade_day: date,
    cutoff: time,
    *,
    workers: int = 160,
) -> tuple[pd.DataFrame, dict[str, str]]:
    frames: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}
    remaining = list(tickers)
    for worker_count, timeout in ((workers, 4.0), (32, 8.0)):
        if not remaining:
            break
        failed: list[str] = []
        with ThreadPoolExecutor(max_workers=min(worker_count, len(remaining))) as executor:
            pending = {
                executor.submit(_fetch_minute_fast, ticker, trade_day, cutoff, timeout): ticker
                for ticker in remaining
            }
            for future in as_completed(pending):
                ticker = pending[future]
                try:
                    frames[ticker] = future.result()
                    failures.pop(ticker, None)
                except Exception as error:  # noqa: BLE001 - 汇总后统一拒绝
                    failures[ticker] = str(error)
                    failed.append(ticker)
        remaining = failed
    ordered = [frames[ticker] for ticker in tickers if ticker in frames]
    return (pd.concat(ordered, ignore_index=True) if ordered else pd.DataFrame()), failures


def parse_tencent_mkline(
    payload: bytes | str,
    ticker: str,
    trade_day: date,
    cutoff: time,
) -> pd.DataFrame:
    """解析腾讯1分钟OHLC；成交量额随后用累计分钟精确值覆盖。"""

    body = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    parsed = json.loads(body)
    rows = parsed.get("data", {}).get(_market_symbol(ticker), {}).get("m1", [])
    records: list[dict] = []
    for row in rows:
        if len(row) < 6:
            continue
        try:
            timestamp = datetime.strptime(str(row[0]), "%Y%m%d%H%M")
            if timestamp.date() != trade_day or not time(9, 31) <= timestamp.time() <= cutoff:
                continue
            records.append(
                {
                    "timestamp": pd.Timestamp(timestamp),
                    "ticker": str(ticker).zfill(6),
                    "open": float(row[1]),
                    "close": float(row[2]),
                    "high": float(row[3]),
                    "low": float(row[4]),
                }
            )
        except (TypeError, ValueError):
            continue
    frame = pd.DataFrame(records)
    if frame.empty:
        raise FastScanDataError(f"腾讯分钟K线为空: {ticker}")
    return frame.sort_values("timestamp").reset_index(drop=True)


def _fetch_one_mkline(
    ticker: str,
    trade_day: date,
    cutoff: time,
    *,
    timeout: float,
) -> pd.DataFrame:
    query = urlencode({"param": f"{_market_symbol(ticker)},m1,,320"})
    return parse_tencent_mkline(
        _fast_get(f"{TENCENT_MKLINE_URL}?{query}", timeout=timeout),
        ticker,
        trade_day,
        cutoff,
    )


def fetch_candidate_ohlc_fast(
    tickers: list[str],
    trade_day: date,
    cutoff: time,
    exact_minutes: pd.DataFrame,
    *,
    workers: int = 64,
) -> pd.DataFrame:
    """腾讯补候选OHLC，并复用已获取的精确分钟量额。"""

    frames: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}
    remaining = list(dict.fromkeys(tickers))
    for worker_count, timeout in ((workers, 4.0), (16, 8.0)):
        if not remaining:
            break
        failed: list[str] = []
        with ThreadPoolExecutor(max_workers=min(worker_count, len(remaining))) as executor:
            pending = {
                executor.submit(
                    _fetch_one_mkline,
                    ticker,
                    trade_day,
                    cutoff,
                    timeout=timeout,
                ): ticker
                for ticker in remaining
            }
            for future in as_completed(pending):
                ticker = pending[future]
                try:
                    frames[ticker] = future.result()
                    failures.pop(ticker, None)
                except Exception as error:  # noqa: BLE001 - 汇总后统一拒绝
                    failures[ticker] = str(error)
                    failed.append(ticker)
        remaining = failed
    if failures:
        sample = dict(list(sorted(failures.items()))[:5])
        raise FastScanDataError(f"腾讯候选分钟K线缺失{len(failures)}只: {sample}")
    ohlc = pd.concat([frames[ticker] for ticker in tickers], ignore_index=True)
    amounts = exact_minutes.loc[
        exact_minutes["ticker"].isin(tickers),
        ["timestamp", "ticker", "volume", "amount"],
    ]
    merged = ohlc.merge(
        amounts,
        on=["timestamp", "ticker"],
        how="left",
        validate="one_to_one",
    )
    if merged[["volume", "amount"]].isna().any().any():
        raise FastScanDataError("腾讯候选OHLC与精确分钟量额无法一一对应")
    return merged.sort_values(["timestamp", "ticker"]).reset_index(drop=True)


def build_fast_universe(pack: FastPack, trade_day: date, quotes: pd.DataFrame) -> pd.DataFrame:
    prior = pack.universe.set_index("ticker", drop=False)
    common = prior.index.intersection(quotes.index)
    if len(common) / len(prior) < 0.99:
        raise FastScanDataError("本地Universe与实时快照交集不足99%")
    output = prior.loc[common].copy()
    quote_rows = quotes.loc[common]
    output["trade_date"] = pd.Timestamp(trade_day)
    output["name"] = quote_rows["name"].to_numpy()
    output["prevclose"] = quote_rows["prevclose"].to_numpy(dtype=float)
    output["float_mcap"] = quote_rows["float_mcap"].to_numpy(dtype=float)
    output["float_shares"] = output["float_mcap"] / output["prevclose"].replace(0, np.nan)
    output["upper_limit"] = quote_rows["upper_limit"].to_numpy(dtype=float)
    output["lower_limit"] = quote_rows["lower_limit"].to_numpy(dtype=float)
    output["is_st"] = output["name"].str.contains("ST", case=False, regex=False)
    quote_dates = quote_rows["quote_time"].astype(str).str[:8]
    output["is_suspended"] = (
        ~quote_dates.eq(trade_day.strftime("%Y%m%d")) | quote_rows["price"].le(0).to_numpy()
    )
    return normalize_universe(output.reset_index(drop=True))


def previous_market_day(trade_day: date) -> date:
    history = _fetch_one_day_history("600000", timeout=8.0, retries=2)
    prior = [item for item in history if item < trade_day]
    if not prior:
        raise FastScanDataError("腾讯五日窗口内无法确认上一交易日")
    return max(prior)


def run_fast_scan(
    project_root: str | Path,
    *,
    trade_day: date | None = None,
    cutoff: time | None = None,
    now: datetime | None = None,
) -> dict:
    started = time_module.perf_counter()
    stage_started = started
    stage_seconds: dict[str, float] = {}

    def finish_stage(name: str) -> None:
        nonlocal stage_started
        current_tick = time_module.perf_counter()
        stage_seconds[name] = current_tick - stage_started
        stage_started = current_tick

    current = now or datetime.now(SHANGHAI)
    day = trade_day or current.date()
    scan_cutoff = cutoff or latest_complete_cutoff(current)
    if scan_cutoff > time(10, 0):
        raise FastScanDataError("C2-A扫描截止不能晚于10:00")
    project = Path(project_root).resolve()
    pack = load_fast_pack(project / DEFAULT_FAST_ROOT)
    required_baseline_day = previous_market_day(day)
    if pack.last_processed_date.date() != required_baseline_day:
        raise FastScanDataError(
            f"本地基线截止{pack.last_processed_date.date()}，需要{required_baseline_day}；"
            "先运行 python main.py --prepare"
        )
    finish_stage("preflight")
    params = C2AParameters.optimized_challenger()
    quotes = fetch_quotes_fast(pack.tickers)
    universe = build_fast_universe(pack, day, quotes)
    eligible = universe.loc[eligible_universe(universe, params) & ~universe["is_suspended"]].copy()
    eligible_tickers = set(eligible["ticker"].astype(str))
    cache_view = _PackCacheView(pack)
    expected, baseline_exclusions = baseline_ready_tickers(
        cache_view, eligible_tickers, params.baseline_days
    )
    coverage = len(expected) / len(eligible_tickers) if eligible_tickers else 0.0
    if coverage < MIN_FAST_COVERAGE:
        raise FastScanDataError(f"20日同分钟基线覆盖不足: {coverage:.2%}")
    eligible = eligible.loc[eligible["ticker"].isin(expected)].copy()
    finish_stage("quotes_and_universe")
    bars, failures = fetch_minutes_fast(sorted(expected), day, scan_cutoff)
    if failures:
        sample = dict(list(sorted(failures.items()))[:5])
        raise FastScanDataError(f"腾讯完整分钟缺失{len(failures)}只: {sample}")
    audit = audit_completed_window(bars, expected, scan_cutoff)
    finish_stage("full_market_minutes")
    features = compute_intraday_features(bars, eligible, cache_view, params, scan_cutoff)
    possible = sorted(set(features.loc[features["signal_pass"], "ticker"].astype(str)))
    finish_stage("feature_ranking")
    positions = []
    events: list[dict] = []
    remaining_cash = 10_000.0
    if possible:
        ohlc = fetch_candidate_ohlc_fast(possible, day, scan_cutoff, bars)
        positions, remaining_cash, events = simulate_entry_day(
            ohlc,
            features,
            eligible,
            params,
            cash=10_000.0,
            budget=10_000.0,
            data_status="PROXY",
        )
    finish_stage("candidate_ohlc_and_simulation")
    name_map = universe.set_index("ticker")["name"].to_dict()
    latest_features = features.loc[
        features["timestamp"].dt.time.eq(scan_cutoff) & features["signal_pass"]
    ].sort_values(["c6", "ticker"])
    watchlist = [
        {
            "ticker": str(row.ticker),
            "name": name_map.get(str(row.ticker), ""),
            "scan_price": float(row.close),
            "c6": float(row.c6),
            "gain": float(row.gain),
            "amount_burst": float(row.amount_burst),
        }
        for row in latest_features.itertuples(index=False)
    ]
    entries = [
        {
            "ticker": position.ticker,
            "name": name_map.get(position.ticker, ""),
            "entry_time": position.entry_time.isoformat(),
            "simulated_fill_price": position.entry_price,
            "shares": position.shares,
            "simulated_notional": position.entry_value,
        }
        for position in positions
    ]
    valid_until = datetime.combine(day, scan_cutoff, SHANGHAI) + timedelta(
        minutes=int(params.signal_expiry_minutes or 0)
    )
    completed_at = datetime.now(SHANGHAI)
    timely = completed_at <= valid_until
    elapsed = time_module.perf_counter() - started
    return {
        "as_of": day.isoformat(),
        "status": (
            "SIMULATED_ENTRY"
            if entries and timely
            else "RECONSTRUCTED_SIMULATED_ENTRY"
            if entries
            else "NO_SIMULATED_SIGNAL"
        ),
        "signal_status": "SIMULATED_ENTRY" if entries else "NO_SIMULATED_SIGNAL",
        "delivery_timing": "ON_TIME" if timely else "LATE",
        "requested_at": current.isoformat(),
        "data_status": "PROXY",
        "cutoff": scan_cutoff.strftime("%H:%M"),
        "complete_minutes": audit.complete_minutes,
        "generated_at": completed_at.isoformat(),
        "signal_valid_until": valid_until.isoformat(),
        "elapsed_seconds": elapsed,
        "stage_seconds": stage_seconds,
        "universe_count": len(expected),
        "baseline_coverage": coverage,
        "baseline_as_of": pack.last_processed_date.date().isoformat(),
        "baseline_excluded_tickers": sorted(baseline_exclusions),
        "candidate_count": len(possible),
        "entry_count": len(entries),
        "entries": entries,
        "watchlist": watchlist,
        "remaining_cash": remaining_cash,
        "parameters": params.to_dict(),
        "promotion_gate": "FAIL",
        "execution_permission": "PAPER_ONLY",
        "real_trade_authorized": False,
        "current_new_entry_allowed": False,
        "event_count": len(events),
    }


def render_scan_result(result: dict, *, watchlist_limit: int = 10) -> str:
    lines = [
        (
            f"C2-A FAST | {result['as_of']} {result['cutoff']} | "
            f"{result['status']} | {result['data_status']} / PAPER_ONLY | "
            f"{result['elapsed_seconds']:.1f}s"
        ),
        "仅模拟，非交易指令",
    ]
    if result["status"] == "RECONSTRUCTED_SIMULATED_ENTRY":
        lines.append("信号已过期，仅供复盘；不得按当前价追入")
    entries = result.get("entries", [])
    if entries:
        lines.append("模拟入场：")
        for item in entries:
            lines.append(
                f"ENTRY {item['ticker']} {item['name']} | "
                f"{item['simulated_fill_price']:.2f}元 × {item['shares']}股"
            )
    else:
        lines.append("无模拟入场信号")
    watchlist = result.get("watchlist", [])[:watchlist_limit]
    if watchlist:
        lines.append("当前强度观察：")
        for item in watchlist:
            lines.append(
                f"WATCHLIST {item['ticker']} {item['name']} | "
                f"C6={item['c6']:.2f} | 涨幅={item['gain']:.2%}"
            )
    lines.append(
        f"基线={result['baseline_as_of']} | 覆盖={result['baseline_coverage']:.2%} | "
        f"promotion gate={result['promotion_gate']}"
    )
    return "\n".join(lines)


def _parse_clock(value: str) -> time:
    hour, minute = (int(item) for item in value.split(":"))
    return time(hour, minute)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python main.py")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--prepare", action="store_true", help="准备本地快速基线")
    parser.add_argument("--bootstrap", action="store_true", help="强制重新导出远端紧凑种子")
    parser.add_argument("--through", help="准备基线截止日 YYYY-MM-DD")
    parser.add_argument("--date", help="扫描交易日 YYYY-MM-DD")
    parser.add_argument("--cutoff", help="扫描截止 HH:MM；默认上一完整分钟")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--watchlist-limit", type=int, default=10)
    subcommands = parser.add_subparsers(dest="internal_command")
    export_parser = subcommands.add_parser("export-pack")
    export_parser.add_argument("--project-root", default=".")
    export_parser.add_argument("--output-root", default=DEFAULT_FAST_ROOT)
    args = parser.parse_args(argv)
    try:
        if args.internal_command == "export-pack":
            result = export_fast_pack(args.project_root, args.output_root)
            print(json.dumps(result, ensure_ascii=False))
            return
        if args.prepare or args.bootstrap:
            result = prepare_fast_pack(
                args.project_root,
                through=date.fromisoformat(args.through) if args.through else None,
                force_bootstrap=args.bootstrap,
            )
            print(
                f"C2-A FAST PREPARED | 基线={result['last_processed_date']} | "
                f"更新交易日={result['updated_days']} | 覆盖={result['coverage']:.2%} | PAPER_ONLY"
            )
            return
        result = run_fast_scan(
            args.project_root,
            trade_day=date.fromisoformat(args.date) if args.date else None,
            cutoff=_parse_clock(args.cutoff) if args.cutoff else None,
        )
        output = Path(args.project_root).resolve() / args.output
        _write_json_atomic(result, output)
        print(render_scan_result(result, watchlist_limit=args.watchlist_limit))
    except (FastScanDataError, RuntimeError, ValueError, OSError) as error:
        parser.exit(2, f"DATA_NOT_READY | {error} | 仅模拟，未生成候选\n")


if __name__ == "__main__":
    main()
