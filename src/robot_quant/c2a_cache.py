"""C2-A 全市场分区特征缓存。

原始分钟数据按日流式读取，只保留可能触发参数网格的候选股日内路径，
以及全市场次日退出和盯市估值所需的少量生命周期行。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from robot_quant.c2a import (
    C2AMarketData,
    C2AParameters,
    C2APreparedData,
    MORNING_FIRST_COMPLETE_MINUTE,
    eligible_universe,
    normalize_minutes,
    normalize_universe,
    parameter_grid,
    rank_c2a_cross_section,
)
from robot_quant.c2a_data import SCHEMA_VERSION, C2ADataStore, DataAudit


CACHE_SCHEMA_VERSION = 2
STREAM_SOURCE_MODE = "validated_bigquant_stream"
CSV_GZIP_STORAGE = "csv_gzip"
PARQUET_ZSTD_STORAGE = "parquet_zstd"
FEATURE_COLUMNS = (
    "timestamp",
    "ticker",
    "trade_date",
    "amount_burst",
    "turnover_metric",
    "gain",
    "c6",
)


@dataclass
class _RollingState:
    tickers: list[str]
    amount_history: np.ndarray
    volume_history: np.ndarray
    pointers: np.ndarray
    counts: np.ndarray
    last_processed_date: pd.Timestamp | None = None


class C2AResearchCache:
    """将全市场原始分钟数据压缩为可重复优化的研究分区。"""

    def __init__(
        self,
        root: str | Path,
        params: C2AParameters,
        *,
        max_scan_end: time = time(11, 0),
        max_c6_threshold: float | None = None,
    ) -> None:
        self.params = params
        self.max_scan_end = max_scan_end
        self.max_c6_threshold = max_c6_threshold or max(
            item.c6_threshold for item in parameter_grid(params)
        )
        fingerprint = _cache_fingerprint(
            params,
            max_scan_end=max_scan_end,
            max_c6_threshold=self.max_c6_threshold,
        )
        variant = params.variant.replace(".", "_")
        self.root = Path(root) / f"{variant}-{fingerprint}"
        self.bars_dir = self.root / "bars"
        self.feature_dirs = {
            False: self.root / "features_b_first_board",
            True: self.root / "features_a_exclude_limit_up",
        }
        self.state_path = self.root / "rolling_state.npz"
        self.manifest_path = self.root / "manifest.json"

    def build(
        self,
        store: C2ADataStore,
        end_date: date | str,
        *,
        progress_callback: Callable[[str], None] | None = None,
    ) -> dict:
        """增量生成缓存；历史原始分区或 Universe 变更时自动重建。"""

        end = pd.Timestamp(end_date).normalize()
        universe = store.read_universe(end_date=end)
        source_paths = store.minute_paths(end_date=end)
        if not source_paths:
            raise RuntimeError("C2-A 缓存无可用原始分钟分区")

        self._ensure_directories()
        manifest = self._manifest()
        state = self._load_state()
        rebuilt = False
        if state is not None and self._source_changed(store, universe, state, manifest):
            self._clear_partitions()
            state = None
            rebuilt = True
        if state is None:
            state = _new_state(
                universe["ticker"].astype(str).unique(),
                self.params.baseline_days,
                _scan_minute_count(self.max_scan_end),
            )
        else:
            _expand_state(
                state,
                universe["ticker"].astype(str).unique(),
                self.params.baseline_days,
            )

        pending = [
            path
            for path in source_paths
            if state.last_processed_date is None
            or _partition_date(path) > state.last_processed_date
        ]
        processed = 0
        candidate_days = 0
        for path in pending:
            trade_day = _partition_date(path)
            day_universe = universe.loc[universe["trade_date"].eq(trade_day)].copy()
            if day_universe.empty:
                raise RuntimeError(f"C2-A 缓存缺少 Universe: {trade_day.date().isoformat()}")
            day_bars = normalize_minutes(_read_partition(path))
            features, possible_tickers = self._build_day_features(
                trade_day,
                day_bars,
                day_universe,
                state,
            )
            compact_bars = _compact_day_bars(
                day_bars,
                day_universe,
                possible_tickers,
                self.params,
                self.max_scan_end,
            )
            _write_partition(self._bars_path(trade_day), compact_bars)
            for limit_group, frame in features.items():
                _write_optional_partition(self._features_path(limit_group, trade_day), frame)
            candidate_days += int(bool(possible_tickers))
            state.last_processed_date = trade_day
            processed += 1
            if processed % 20 == 0:
                self._save_checkpoint(state, store, universe)
                if progress_callback is not None:
                    progress_callback(f"C2-A 研究缓存：已处理 {processed}/{len(pending)} 个交易日")
        if processed or not self.state_path.exists():
            self._save_checkpoint(state, store, universe)

        compact_rows = sum(
            _partition_row_count(path) for path in self._partition_paths(self.bars_dir, None, end)
        )
        result = {
            "status": "READY",
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "cache_root": str(self.root),
            "last_processed_date": (
                state.last_processed_date.date().isoformat()
                if state.last_processed_date is not None
                else None
            ),
            "processed_days": processed,
            "candidate_days_processed": candidate_days,
            "compact_bar_rows": compact_rows,
            "rebuilt_after_source_change": rebuilt,
        }
        return result

    def last_processed_date(self) -> pd.Timestamp | None:
        """返回当前缓存已提交的最后交易日。"""

        state = self._load_state()
        return state.last_processed_date if state is not None else None

    def is_streaming(self) -> bool:
        return self._manifest().get("source_mode") == STREAM_SOURCE_MODE

    def ingest_validated_day(
        self,
        store: C2ADataStore,
        universe: pd.DataFrame,
        trade_day: date | str | pd.Timestamp,
        day_bars: pd.DataFrame,
        *,
        expected_tickers: set[str],
        relevant_tickers: set[str],
    ) -> dict:
        """提交一个已经按全会话校验的 BigQuant 日分区，只持久化研究所需压缩行。"""

        day = pd.Timestamp(trade_day).normalize()
        stocks = normalize_universe(universe)
        day_universe = stocks.loc[stocks["trade_date"].eq(day)].copy()
        if day_universe.empty:
            raise RuntimeError(f"C2-A 流式缓存缺少 Universe: {day.date().isoformat()}")
        bars = normalize_minutes(day_bars)
        _validate_streamed_day(bars, expected_tickers, day)
        self._ensure_directories()
        manifest = self._manifest()
        state = self._load_state()
        if manifest.get("source_mode") == STREAM_SOURCE_MODE:
            manifest = self._reconcile_stream_checkpoint(stocks, state, manifest)
        if state is None:
            if manifest and manifest.get("source_mode") not in {None, STREAM_SOURCE_MODE}:
                raise RuntimeError("C2-A 缓存已由另一种数据模式创建，不能混合写入")
            state = _new_state(
                stocks["ticker"].astype(str).unique(),
                self.params.baseline_days,
                _scan_minute_count(self.max_scan_end),
            )
        else:
            if manifest.get("source_mode") != STREAM_SOURCE_MODE:
                raise RuntimeError("C2-A 原始分区缓存不能直接切换为 BigQuant 流式缓存")
            _assert_processed_universe_unchanged(stocks, state, manifest)
            _expand_state(
                state,
                stocks["ticker"].astype(str).unique(),
                self.params.baseline_days,
            )
        if state.last_processed_date is not None and day <= state.last_processed_date:
            return {
                "processed": False,
                "trade_date": day.date().isoformat(),
                "compact_rows": _partition_row_count(self._bars_path(day)),
            }
        _assert_next_trade_day(stocks, state.last_processed_date, day)

        features, possible_tickers = self._build_day_features(
            day,
            bars,
            day_universe,
            state,
        )
        compact_bars = _compact_day_bars(
            bars,
            day_universe,
            possible_tickers,
            self.params,
            self.max_scan_end,
        )
        _write_partition(self._bars_path(day), compact_bars)
        for limit_group, frame in features.items():
            _write_optional_partition(self._features_path(limit_group, day), frame)
        state.last_processed_date = day
        self._save_stream_checkpoint(
            state,
            stocks,
            manifest,
            expected_tickers=expected_tickers,
            relevant_tickers=relevant_tickers,
            raw_rows=len(bars),
            possible_ticker_count=len(possible_tickers),
        )
        return {
            "processed": True,
            "trade_date": day.date().isoformat(),
            "raw_rows_validated": len(bars),
            "compact_rows": len(compact_bars),
            "possible_tickers": len(possible_tickers),
        }

    def streaming_summary(self, end_date: date | str | None = None) -> dict:
        """返回流式缓存的可复核摘要。"""

        manifest = self._manifest()
        if manifest.get("source_mode") != STREAM_SOURCE_MODE:
            raise RuntimeError("C2-A 缓存不是 BigQuant 流式模式")
        end = pd.Timestamp(end_date).normalize() if end_date is not None else None
        validated = manifest.get("validated_days", {})
        selected_days = {
            key: value
            for key, value in validated.items()
            if end is None or pd.Timestamp(key).normalize() <= end
        }
        return {
            "status": "READY",
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "cache_root": str(self.root),
            "source_mode": STREAM_SOURCE_MODE,
            "storage_format": manifest.get("storage_format", self._storage_format()),
            "last_processed_date": manifest.get("last_processed_date"),
            "processed_days": 0,
            "candidate_days_processed": sum(
                int(value.get("possible_tickers", 0) > 0) for value in selected_days.values()
            ),
            "validated_raw_rows": sum(
                int(value.get("raw_rows", 0)) for value in selected_days.values()
            ),
            "compact_bar_rows": sum(
                _partition_row_count(path)
                for path in self._partition_paths(self.bars_dir, None, end)
            ),
            "rebuilt_after_source_change": False,
        }

    def migrate_stream_partitions_to_parquet(self) -> dict:
        """逐文件校验迁移为 Parquet/Zstandard；源CSV仅在目标行数一致后移除。"""

        if not self.is_streaming():
            raise RuntimeError("只有 BigQuant 流式缓存可以执行列式迁移")
        if not _parquet_available():
            raise RuntimeError("当前解释器未安装 pyarrow，不能迁移 Parquet")
        directories = (self.bars_dir, *self.feature_dirs.values())
        sources = [
            path
            for directory in directories
            for path in sorted(directory.glob("????-??-??.csv.gz"))
        ]
        bytes_before = sum(path.stat().st_size for path in sources)
        migrated = 0
        for source in sources:
            target = source.with_name(f"{source.name.removesuffix('.csv.gz')}.parquet")
            source_rows = _partition_row_count(source)
            if not target.exists():
                _write_partition(target, _read_partition(source))
            if _partition_row_count(target) != source_rows:
                raise RuntimeError(f"Parquet迁移行数不一致，保留源文件: {source}")
            source.unlink()
            migrated += 1
        manifest = self._manifest()
        manifest["storage_format"] = PARQUET_ZSTD_STORAGE
        self._write_manifest_atomic(manifest)
        targets = [
            path
            for directory in directories
            for path in sorted(directory.glob("????-??-??.parquet"))
        ]
        return {
            "migrated_files": migrated,
            "source_bytes_removed": bytes_before,
            "parquet_bytes": sum(path.stat().st_size for path in targets),
            "storage_format": PARQUET_ZSTD_STORAGE,
        }

    def audit_streaming(
        self,
        store: C2ADataStore,
        start_date: date | str,
        end_date: date | str,
    ) -> DataAudit:
        """以逐日 240 根校验账本审计不落原始大表的 BigQuant 缓存。"""

        start = pd.Timestamp(start_date).normalize()
        end = pd.Timestamp(end_date).normalize()
        if start > end:
            raise ValueError("审计开始日期不能晚于结束日期")
        reasons = _stream_manifest_reasons(store.manifest(), self._manifest())
        try:
            universe = store.read_universe()
        except (FileNotFoundError, ValueError):
            universe = pd.DataFrame()
            reasons.append("universe_missing_or_invalid")
        manifest = self._manifest()
        validated = manifest.get("validated_days", {})
        state = self._load_state()
        state_day = state.last_processed_date if state is not None else None
        manifest_text = manifest.get("last_processed_date")
        manifest_day = pd.Timestamp(manifest_text).normalize() if manifest_text else None
        if state_day != manifest_day:
            reasons.append("stream_checkpoint_mismatch")
        relevant = {str(item).zfill(6) for item in manifest.get("relevant_tickers", [])}
        target = (
            universe.loc[universe["trade_date"].between(start, end)].copy()
            if not universe.empty
            else universe
        )
        target_dates = sorted(pd.Timestamp(item) for item in target["trade_date"].unique())
        minute_rows = 0
        for day in target_dates:
            record = validated.get(day.date().isoformat())
            if record is None:
                reasons.append(f"missing_validated_stream_day:{day.date().isoformat()}")
                continue
            ticker_count = int(record.get("ticker_count", -1))
            ticker_hash = str(record.get("ticker_hash", ""))
            raw_rows = int(record.get("raw_rows", -1))
            if ticker_count < 0 or len(ticker_hash) != 64 or raw_rows != 240 * ticker_count:
                reasons.append(f"validated_stream_coverage_mismatch:{day.date().isoformat()}")
            minute_rows += max(raw_rows, 0)
            if not self._bars_path(day).exists():
                reasons.append(f"compact_bar_partition_missing:{day.date().isoformat()}")

        required_history = _required_baseline_history(
            universe,
            target,
            relevant,
            self.params,
        )
        validated_baseline_days = {
            pd.Timestamp(day).normalize()
            for day in validated
            if pd.Timestamp(day).normalize() < start
        }
        if len(validated_baseline_days) < self.params.baseline_days:
            reasons.append("insufficient_stream_global_baseline_days")
        for day, tickers in required_history.items():
            record = validated.get(day.date().isoformat())
            if record is None:
                reasons.append(f"missing_stream_baseline_day:{day.date().isoformat()}")
                continue
        if state is None or state.last_processed_date is None:
            reasons.append("stream_rolling_state_missing")
        elif target_dates and state.last_processed_date < target_dates[-1]:
            reasons.append("stream_rolling_state_stale")
        elif not universe.empty:
            try:
                _assert_processed_universe_unchanged(universe, state, manifest)
            except RuntimeError:
                reasons.append("processed_universe_changed")

        unique_reasons = tuple(dict.fromkeys(reasons))
        status = "STRICT" if target_dates and not unique_reasons else "PROXY"
        if universe.empty or not target_dates or not validated:
            status = "DATA_NOT_READY"
        return DataAudit(
            status=status,
            start_date=start.date().isoformat(),
            end_date=end.date().isoformat(),
            trading_days=len(target_dates),
            baseline_days_available=len(validated_baseline_days),
            universe_rows=len(target),
            minute_rows=minute_rows,
            source=f"{store.manifest().get('source', 'bigquant')}:stream_cache",
            reasons=unique_reasons,
        )

    def read_bars(
        self,
        start_date: date | str | None = None,
        end_date: date | str | None = None,
    ) -> pd.DataFrame:
        paths = self._partition_paths(self.bars_dir, start_date, end_date)
        if not paths:
            return pd.DataFrame(
                columns=(
                    "timestamp",
                    "ticker",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "amount",
                )
            )
        return normalize_minutes(
            pd.concat([_read_partition(path) for path in paths], ignore_index=True)
        )

    def read_features(
        self,
        exclude_yesterday_limit_up: bool,
        start_date: date | str | None = None,
        end_date: date | str | None = None,
    ) -> pd.DataFrame:
        paths = self._partition_paths(
            self.feature_dirs[exclude_yesterday_limit_up], start_date, end_date
        )
        if not paths:
            return pd.DataFrame(columns=FEATURE_COLUMNS)
        frame = pd.concat([_read_partition(path) for path in paths], ignore_index=True)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
        frame["ticker"] = frame["ticker"].astype(str).str.zfill(6)
        return frame.sort_values(["timestamp", "ticker"]).reset_index(drop=True)

    def load_prepared_data(
        self,
        universe: pd.DataFrame,
        start_date: date | str | None = None,
        end_date: date | str | None = None,
    ) -> tuple[C2AMarketData, dict[bool, C2APreparedData]]:
        """按日分区直接构造共享研究对象，避免先拼接全年大表再拆分的内存峰值。"""

        stocks = normalize_universe(universe)
        bar_paths = self._partition_paths(self.bars_dir, start_date, end_date)
        dates = tuple(_partition_date(path) for path in bar_paths)
        date_set = set(dates)
        bars_by_date = {
            trade_day: normalize_minutes(_read_partition(path))
            for trade_day, path in zip(dates, bar_paths, strict=True)
        }
        stock_groups = {
            pd.Timestamp(trade_day): group.reset_index(drop=True)
            for trade_day, group in stocks.groupby("trade_date", sort=True)
            if pd.Timestamp(trade_day) in date_set
        }
        market = C2AMarketData(
            bars_by_date=bars_by_date,
            stocks_by_date=stock_groups,
            last_prices_by_date={
                trade_day: {
                    str(ticker): float(close)
                    for ticker, close in frame.groupby("ticker", sort=False)["close"].last().items()
                }
                for trade_day, frame in bars_by_date.items()
            },
            all_dates=dates,
        )
        prepared: dict[bool, C2APreparedData] = {}
        for limit_group, directory in self.feature_dirs.items():
            features_by_date: dict[pd.Timestamp, pd.DataFrame] = {}
            for path in self._partition_paths(directory, start_date, end_date):
                trade_day = _partition_date(path)
                frame = _read_partition(path)
                frame["timestamp"] = pd.to_datetime(frame["timestamp"])
                frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
                frame["ticker"] = frame["ticker"].astype(str).str.zfill(6)
                day_stocks = stock_groups[trade_day].set_index("ticker")
                frame["pool"] = frame["ticker"].map(day_stocks["pool"])
                frame["universe_pass"] = True
                features_by_date[trade_day] = frame.sort_values(
                    ["timestamp", "ticker"]
                ).reset_index(drop=True)
            prepared[limit_group] = C2APreparedData(
                market=market,
                features_by_date=features_by_date,
            )
        return market, prepared

    def _build_day_features(
        self,
        trade_day: pd.Timestamp,
        day_bars: pd.DataFrame,
        day_universe: pd.DataFrame,
        state: _RollingState,
    ) -> tuple[dict[bool, pd.DataFrame], set[str]]:
        tickers, current_amount, current_volume, current_close = _scan_matrices(
            day_bars, self.max_scan_end
        )
        index = {ticker: position for position, ticker in enumerate(state.tickers)}
        state_indices = np.asarray([index[ticker] for ticker in tickers], dtype=np.int64)
        complete = (
            np.isfinite(current_amount).all(axis=1)
            & np.isfinite(current_volume).all(axis=1)
            & np.isfinite(current_close).all(axis=1)
        )
        current_amount = np.cumsum(current_amount, axis=1)
        current_volume = np.cumsum(current_volume, axis=1)
        amount_baseline = np.full_like(current_amount, np.nan, dtype=float)
        volume_baseline = np.full_like(current_volume, np.nan, dtype=float)
        ready = complete & (state.counts[state_indices] >= self.params.baseline_days)
        ready_positions = np.flatnonzero(ready)
        for chunk_start in range(0, len(ready_positions), 500):
            local = ready_positions[chunk_start : chunk_start + 500]
            global_indices = state_indices[local]
            amount_baseline[local] = np.median(state.amount_history[global_indices], axis=2)
            volume_baseline[local] = np.median(state.volume_history[global_indices], axis=2)

        feature_groups: dict[bool, pd.DataFrame] = {}
        possible_tickers: set[str] = set()
        for exclude_limit_up in (False, True):
            group = self._rank_day(
                trade_day,
                tickers,
                current_amount,
                current_volume,
                current_close,
                amount_baseline,
                volume_baseline,
                day_universe,
                exclude_limit_up,
            )
            feature_groups[exclude_limit_up] = group
            possible_tickers.update(group["ticker"].astype(str).unique())

        update_positions = np.flatnonzero(complete)
        update_indices = state_indices[update_positions]
        update_slots = state.pointers[update_indices]
        for slot in np.unique(update_slots):
            selected = update_positions[update_slots == slot]
            selected_indices = state_indices[selected]
            state.amount_history[selected_indices, :, slot] = current_amount[selected]
            state.volume_history[selected_indices, :, slot] = current_volume[selected]
        state.pointers[update_indices] = (
            state.pointers[update_indices] + 1
        ) % self.params.baseline_days
        state.counts[update_indices] = np.minimum(
            state.counts[update_indices] + 1,
            self.params.baseline_days,
        )
        return feature_groups, possible_tickers

    def _rank_day(
        self,
        trade_day: pd.Timestamp,
        tickers: list[str],
        current_amount: np.ndarray,
        current_volume: np.ndarray,
        current_close: np.ndarray,
        amount_baseline: np.ndarray,
        volume_baseline: np.ndarray,
        day_universe: pd.DataFrame,
        exclude_limit_up: bool,
    ) -> pd.DataFrame:
        config = replace(
            self.params,
            scan_end=self.max_scan_end,
            exclude_yesterday_limit_up=exclude_limit_up,
            max_allowed_limit_streak=1,
        )
        stocks = normalize_universe(day_universe).set_index("ticker", drop=False)
        eligible = set(stocks.loc[eligible_universe(stocks, config)].index.astype(str))
        selected_positions = [
            position for position, ticker in enumerate(tickers) if ticker in eligible
        ]
        if not selected_positions:
            return pd.DataFrame(columns=FEATURE_COLUMNS)

        local_tickers = [tickers[position] for position in selected_positions]
        amount = current_amount[selected_positions]
        volume = current_volume[selected_positions]
        close = current_close[selected_positions]
        amount_base = amount_baseline[selected_positions]
        volume_base = volume_baseline[selected_positions]
        stock_rows = stocks.loc[local_tickers]
        if self.params.use_relative_turnover:
            turnover = volume / np.where(volume_base == 0, np.nan, volume_base)
        else:
            shares = stock_rows["float_shares"].to_numpy(dtype=float)[:, None]
            turnover = volume / np.where(shares == 0, np.nan, shares)
        amount_burst = amount / np.where(amount_base == 0, np.nan, amount_base)
        previous_close = stock_rows["prevclose"].to_numpy(dtype=float)[:, None]
        gain = close / previous_close - 1.0
        minute_count = _scan_minute_count(self.max_scan_end)
        timestamps = pd.date_range(
            f"{trade_day.date().isoformat()} 09:31",
            periods=minute_count,
            freq="1min",
        )
        frame = pd.DataFrame(
            {
                "timestamp": np.tile(timestamps.to_numpy(), len(local_tickers)),
                "ticker": np.repeat(local_tickers, minute_count),
                "trade_date": trade_day,
                "pool": np.repeat(stock_rows["pool"].astype(str).to_numpy(), minute_count),
                "amount_burst": amount_burst.reshape(-1),
                "turnover_metric": turnover.reshape(-1),
                "gain": gain.reshape(-1),
            }
        )
        frame = frame.dropna(subset=["amount_burst", "turnover_metric", "gain"])
        if frame.empty:
            return pd.DataFrame(columns=FEATURE_COLUMNS)
        frame = rank_c2a_cross_section(
            frame,
            self.params,
            ["timestamp", "pool"],
        )
        main = frame["pool"].eq("MAIN")
        gain_min = np.where(main, self.params.main_gain_min, self.params.growth_gain_min)
        gain_max = np.where(main, self.params.main_gain_max, self.params.growth_gain_max)
        possible = set(
            frame.loc[
                frame["gain"].ge(gain_min)
                & frame["gain"].le(gain_max)
                & frame["c6"].lt(self.max_c6_threshold),
                "ticker",
            ].astype(str)
        )
        if not possible:
            return pd.DataFrame(columns=FEATURE_COLUMNS)
        return (
            frame.loc[frame["ticker"].isin(possible), list(FEATURE_COLUMNS)]
            .sort_values(["timestamp", "ticker"])
            .reset_index(drop=True)
        )

    def _ensure_directories(self) -> None:
        self.bars_dir.mkdir(parents=True, exist_ok=True)
        for directory in self.feature_dirs.values():
            directory.mkdir(parents=True, exist_ok=True)

    def _bars_path(self, trade_day: pd.Timestamp) -> Path:
        return self._partition_path(self.bars_dir, trade_day)

    def _features_path(self, exclude_limit_up: bool, trade_day: pd.Timestamp) -> Path:
        return self._partition_path(self.feature_dirs[exclude_limit_up], trade_day)

    def _partition_path(self, directory: Path, trade_day: pd.Timestamp) -> Path:
        date_text = pd.Timestamp(trade_day).date().isoformat()
        parquet = directory / f"{date_text}.parquet"
        csv_gzip = directory / f"{date_text}.csv.gz"
        if parquet.exists():
            return parquet
        if csv_gzip.exists():
            return csv_gzip
        suffix = ".parquet" if self._storage_format() == PARQUET_ZSTD_STORAGE else ".csv.gz"
        return directory / f"{date_text}{suffix}"

    def _storage_format(self) -> str:
        configured = self._manifest().get("storage_format")
        if configured in {CSV_GZIP_STORAGE, PARQUET_ZSTD_STORAGE}:
            return configured
        directories = (self.bars_dir, *self.feature_dirs.values())
        if any(next(directory.glob("????-??-??.parquet"), None) for directory in directories):
            return PARQUET_ZSTD_STORAGE
        if any(next(directory.glob("????-??-??.csv.gz"), None) for directory in directories):
            return CSV_GZIP_STORAGE
        return PARQUET_ZSTD_STORAGE if _parquet_available() else CSV_GZIP_STORAGE

    def _clear_partitions(self) -> None:
        for directory in (self.bars_dir, *self.feature_dirs.values()):
            for pattern in ("????-??-??.csv.gz", "????-??-??.parquet"):
                for path in directory.glob(pattern):
                    path.unlink()

    def _partition_paths(
        self,
        directory: Path,
        start_date: date | str | None,
        end_date: date | str | None,
    ) -> list[Path]:
        start = pd.Timestamp(start_date).normalize() if start_date is not None else None
        end = pd.Timestamp(end_date).normalize() if end_date is not None else None
        by_date: dict[pd.Timestamp, Path] = {}
        for pattern in ("????-??-??.csv.gz", "????-??-??.parquet"):
            for path in sorted(directory.glob(pattern)):
                trade_day = _partition_date(path)
                if (start is not None and trade_day < start) or (
                    end is not None and trade_day > end
                ):
                    continue
                if trade_day not in by_date or path.suffix == ".parquet":
                    by_date[trade_day] = path
        return [by_date[trade_day] for trade_day in sorted(by_date)]

    def _manifest(self) -> dict:
        if not self.manifest_path.exists():
            return {}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def _load_state(self) -> _RollingState | None:
        if not self.state_path.exists():
            return None
        with np.load(self.state_path, allow_pickle=False) as payload:
            last_text = str(payload["last_processed_date"].item())
            return _RollingState(
                tickers=[str(item) for item in payload["tickers"].tolist()],
                amount_history=payload["amount_history"],
                volume_history=payload["volume_history"],
                pointers=payload["pointers"],
                counts=payload["counts"],
                last_processed_date=(pd.Timestamp(last_text).normalize() if last_text else None),
            )

    def _source_changed(
        self,
        store: C2ADataStore,
        universe: pd.DataFrame,
        state: _RollingState,
        manifest: dict,
    ) -> bool:
        if state.last_processed_date is None:
            return False
        expected_signatures = manifest.get("source_partition_signatures", {})
        current_paths = store.minute_paths(end_date=state.last_processed_date)
        current_signatures = _partition_signatures(current_paths)
        if expected_signatures != current_signatures:
            return True
        expected_hash = manifest.get("processed_universe_hash")
        current_hash = _universe_hash(
            universe.loc[universe["trade_date"].le(state.last_processed_date)]
        )
        return expected_hash != current_hash

    def _save_checkpoint(
        self,
        state: _RollingState,
        store: C2ADataStore,
        universe: pd.DataFrame,
    ) -> None:
        self._save_state(state)
        processed_paths = (
            store.minute_paths(end_date=state.last_processed_date)
            if state.last_processed_date is not None
            else []
        )
        processed_universe = (
            universe.loc[universe["trade_date"].le(state.last_processed_date)]
            if state.last_processed_date is not None
            else universe.iloc[0:0]
        )
        manifest = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "storage_format": self._storage_format(),
            "variant": self.params.variant,
            "parameters": self.params.to_dict(),
            "max_scan_end": self.max_scan_end.strftime("%H:%M"),
            "max_c6_threshold": self.max_c6_threshold,
            "last_processed_date": (
                state.last_processed_date.date().isoformat()
                if state.last_processed_date is not None
                else None
            ),
            "source_partition_signatures": _partition_signatures(processed_paths),
            "processed_universe_hash": _universe_hash(processed_universe),
            "execution_permission": "PAPER_ONLY",
        }
        temporary_manifest = self.manifest_path.with_name(f".{self.manifest_path.name}.tmp")
        temporary_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary_manifest.replace(self.manifest_path)

    def _save_stream_checkpoint(
        self,
        state: _RollingState,
        universe: pd.DataFrame,
        prior_manifest: dict,
        *,
        expected_tickers: set[str],
        relevant_tickers: set[str],
        raw_rows: int,
        possible_ticker_count: int,
    ) -> None:
        if state.last_processed_date is None:
            raise RuntimeError("C2-A 流式缓存不能提交空检查点")
        day_text = state.last_processed_date.date().isoformat()
        validated_days = dict(prior_manifest.get("validated_days", {}))
        validated_days[day_text] = {
            "ticker_count": len(expected_tickers),
            "ticker_hash": _ticker_set_hash(expected_tickers),
            "raw_rows": raw_rows,
            "possible_tickers": possible_ticker_count,
        }
        processed_universe = universe.loc[universe["trade_date"].le(state.last_processed_date)]
        manifest = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "source_mode": STREAM_SOURCE_MODE,
            "storage_format": self._storage_format(),
            "variant": self.params.variant,
            "parameters": self.params.to_dict(),
            "max_scan_end": self.max_scan_end.strftime("%H:%M"),
            "max_c6_threshold": self.max_c6_threshold,
            "last_processed_date": day_text,
            "processed_universe_hash": _universe_hash(processed_universe),
            "relevant_tickers": sorted({str(item).zfill(6) for item in relevant_tickers}),
            "validated_days": validated_days,
            "execution_permission": "PAPER_ONLY",
        }
        temporary_state: Path | None = None
        temporary_manifest: Path | None = None
        try:
            temporary_state = self._write_state_temporary(state)
            temporary_manifest = self._write_manifest_temporary(manifest)
            # 清单先行；若随后状态替换失败，下次会把清单回退到旧状态后重算当天。
            temporary_manifest.replace(self.manifest_path)
            temporary_state.replace(self.state_path)
        finally:
            if temporary_state is not None and temporary_state.exists():
                temporary_state.unlink()
            if temporary_manifest is not None and temporary_manifest.exists():
                temporary_manifest.unlink()

    def _reconcile_stream_checkpoint(
        self,
        universe: pd.DataFrame,
        state: _RollingState | None,
        manifest: dict,
    ) -> dict:
        """恢复清单与滚动状态跨文件提交中断，不猜测未通过校验的原始数据。"""

        state_day = state.last_processed_date if state is not None else None
        manifest_text = manifest.get("last_processed_date")
        manifest_day = pd.Timestamp(manifest_text).normalize() if manifest_text else None
        if state_day == manifest_day:
            return manifest
        repaired = dict(manifest)
        validated = dict(repaired.get("validated_days", {}))
        relevant = {str(item).zfill(6) for item in repaired.get("relevant_tickers", [])}
        if state_day is None:
            validated = {}
        elif manifest_day is None or state_day > manifest_day:
            dates = pd.DatetimeIndex(universe["trade_date"].unique()).sort_values()
            missing_days = dates[dates <= state_day]
            if manifest_day is not None:
                missing_days = missing_days[missing_days > manifest_day]
            for raw_day in missing_days:
                day = pd.Timestamp(raw_day).normalize()
                bar_path = self._bars_path(day)
                if not bar_path.exists():
                    raise RuntimeError(
                        f"C2-A 流式状态领先清单，但缺少压缩分区: {day.date().isoformat()}"
                    )
                day_universe = universe.loc[universe["trade_date"].eq(day)]
                expected = set(
                    day_universe.loc[
                        day_universe["ticker"].isin(relevant)
                        & ~day_universe["is_suspended"].astype(bool),
                        "ticker",
                    ].astype(str)
                )
                validated[day.date().isoformat()] = {
                    "ticker_count": len(expected),
                    "ticker_hash": _ticker_set_hash(expected),
                    "raw_rows": 240 * len(expected),
                    "possible_tickers": self._possible_ticker_count(day),
                    "recovered_after_interrupted_commit": True,
                }
        else:
            validated = {
                key: value
                for key, value in validated.items()
                if pd.Timestamp(key).normalize() <= state_day
            }
        repaired["validated_days"] = validated
        repaired["last_processed_date"] = (
            state_day.date().isoformat() if state_day is not None else None
        )
        processed = (
            universe.loc[universe["trade_date"].le(state_day)]
            if state_day is not None
            else universe.iloc[0:0]
        )
        repaired["processed_universe_hash"] = _universe_hash(processed)
        self._write_manifest_atomic(repaired)
        return repaired

    def _possible_ticker_count(self, trade_day: pd.Timestamp) -> int:
        tickers: set[str] = set()
        for limit_group in self.feature_dirs:
            path = self._features_path(limit_group, trade_day)
            if path.exists():
                tickers.update(_read_partition(path, columns=["ticker"])["ticker"].astype(str))
        return len(tickers)

    def _write_manifest_atomic(self, manifest: dict) -> None:
        temporary = self._write_manifest_temporary(manifest)
        try:
            temporary.replace(self.manifest_path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _write_manifest_temporary(self, manifest: dict) -> Path:
        temporary = self.manifest_path.with_name(f".{self.manifest_path.name}.tmp")
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return temporary

    def _save_state(self, state: _RollingState) -> None:
        temporary_state = self._write_state_temporary(state)
        temporary_state.replace(self.state_path)

    def _write_state_temporary(self, state: _RollingState) -> Path:
        temporary_state = self.state_path.with_name(f".{self.state_path.name}.tmp")
        with temporary_state.open("wb") as handle:
            np.savez(
                handle,
                tickers=np.asarray(state.tickers),
                amount_history=state.amount_history,
                volume_history=state.volume_history,
                pointers=state.pointers,
                counts=state.counts,
                last_processed_date=(
                    state.last_processed_date.date().isoformat()
                    if state.last_processed_date is not None
                    else ""
                ),
            )
        return temporary_state


def _validate_streamed_day(
    bars: pd.DataFrame,
    expected_tickers: set[str],
    trade_day: pd.Timestamp,
) -> None:
    observed_tickers = set(bars["ticker"].astype(str))
    if observed_tickers != expected_tickers:
        raise RuntimeError(
            f"C2-A {trade_day.date().isoformat()} 流式股票覆盖不完整；"
            f"缺失={sorted(expected_tickers - observed_tickers)}"
        )
    if not bars["timestamp"].dt.normalize().eq(trade_day).all():
        raise RuntimeError(f"C2-A {trade_day.date().isoformat()} 流式分区混入其他日期")
    expected_minutes = _expected_full_session_minute_numbers()
    minute_number = bars["timestamp"].dt.hour * 60 + bars["timestamp"].dt.minute
    for ticker, ticker_bars in bars.assign(_minute=minute_number).groupby("ticker", sort=False):
        if len(ticker_bars) != 240 or set(ticker_bars["_minute"].astype(int)) != expected_minutes:
            raise RuntimeError(
                f"C2-A {trade_day.date().isoformat()} {ticker} 不是完整240根连续竞价分钟"
            )


def _expected_full_session_minute_numbers() -> set[int]:
    return set(range(9 * 60 + 31, 11 * 60 + 31)) | set(range(13 * 60 + 1, 15 * 60 + 1))


def _assert_next_trade_day(
    universe: pd.DataFrame,
    last_processed_date: pd.Timestamp | None,
    trade_day: pd.Timestamp,
) -> None:
    dates = pd.DatetimeIndex(universe["trade_date"].unique()).sort_values()
    pending = dates[dates > last_processed_date] if last_processed_date is not None else dates
    if len(pending) == 0 or pd.Timestamp(pending[0]).normalize() != trade_day:
        expected = pd.Timestamp(pending[0]).date().isoformat() if len(pending) else "无"
        raise RuntimeError(
            f"C2-A 流式分区必须按交易日顺序提交；期望={expected}，收到={trade_day.date()}"
        )


def _assert_processed_universe_unchanged(
    universe: pd.DataFrame,
    state: _RollingState,
    manifest: dict,
) -> None:
    if state.last_processed_date is None:
        return
    expected_hash = manifest.get("processed_universe_hash")
    processed = universe.loc[universe["trade_date"].le(state.last_processed_date)]
    if expected_hash != _universe_hash(processed):
        raise RuntimeError("C2-A 已处理日期的 Universe 发生变化，需要显式重建流式缓存")


def _ticker_set_hash(tickers: set[str]) -> str:
    encoded = "\n".join(sorted({str(item).zfill(6) for item in tickers})).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stream_manifest_reasons(store_manifest: dict, cache_manifest: dict) -> list[str]:
    reasons: list[str] = []
    if store_manifest.get("schema_version") != SCHEMA_VERSION:
        reasons.append("manifest_schema_invalid")
    if store_manifest.get("frequency") != "1min":
        reasons.append("frequency_not_1min")
    if store_manifest.get("price_adjustment") != "none":
        reasons.append("minute_prices_must_be_unadjusted")
    if store_manifest.get("volume_unit") != "shares":
        reasons.append("minute_volume_unit_must_be_shares")
    if store_manifest.get("amount_unit") != "CNY":
        reasons.append("minute_amount_unit_must_be_CNY")
    if store_manifest.get("float_shares_unit") != "shares":
        reasons.append("float_shares_unit_must_be_shares")
    if store_manifest.get("bar_timestamp_convention") != "CN_A_SHARE_1MIN_END_TIME":
        reasons.append("bar_timestamp_convention_invalid")
    if not store_manifest.get("metadata_verified"):
        reasons.append("historical_metadata_not_verified")
    if not store_manifest.get("full_market"):
        reasons.append("not_full_a_share_market")
    if cache_manifest.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
        reasons.append("stream_cache_schema_invalid")
    if cache_manifest.get("source_mode") != STREAM_SOURCE_MODE:
        reasons.append("stream_source_mode_invalid")
    return reasons


def _required_baseline_history(
    universe: pd.DataFrame,
    target: pd.DataFrame,
    relevant_tickers: set[str],
    params: C2AParameters,
) -> dict[pd.Timestamp, set[str]]:
    required: dict[pd.Timestamp, set[str]] = {}
    eligible = target.loc[eligible_universe(target, params)]
    active_dates = {
        str(ticker): pd.DatetimeIndex(group["trade_date"].sort_values().unique())
        for ticker, group in universe.loc[~universe["is_suspended"].astype(bool)].groupby("ticker")
    }
    for row in eligible[["trade_date", "ticker"]].itertuples(index=False):
        trade_day = pd.Timestamp(row.trade_date)
        ticker = str(row.ticker)
        if ticker not in relevant_tickers or ticker not in active_dates:
            continue
        history = active_dates[ticker][active_dates[ticker] < trade_day][-params.baseline_days :]
        for history_day in history:
            required.setdefault(pd.Timestamp(history_day), set()).add(ticker)
    return required


def _new_state(tickers, baseline_days: int, minute_count: int) -> _RollingState:
    symbols = sorted({str(ticker).zfill(6) for ticker in tickers})
    shape = (len(symbols), minute_count, baseline_days)
    return _RollingState(
        tickers=symbols,
        amount_history=np.full(shape, np.nan, dtype=np.float64),
        volume_history=np.full(shape, np.nan, dtype=np.float64),
        pointers=np.zeros(len(symbols), dtype=np.int16),
        counts=np.zeros(len(symbols), dtype=np.int16),
    )


def _expand_state(state: _RollingState, tickers, baseline_days: int) -> None:
    additions = sorted({str(ticker).zfill(6) for ticker in tickers}.difference(state.tickers))
    if not additions:
        return
    minute_count = state.amount_history.shape[1]
    history_shape = (len(additions), minute_count, baseline_days)
    state.tickers.extend(additions)
    state.amount_history = np.concatenate(
        [state.amount_history, np.full(history_shape, np.nan)], axis=0
    )
    state.volume_history = np.concatenate(
        [state.volume_history, np.full(history_shape, np.nan)], axis=0
    )
    state.pointers = np.concatenate([state.pointers, np.zeros(len(additions), dtype=np.int16)])
    state.counts = np.concatenate([state.counts, np.zeros(len(additions), dtype=np.int16)])


def _scan_matrices(
    day_bars: pd.DataFrame, scan_end: time
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    bars = day_bars.copy()
    minute_number = bars["timestamp"].dt.hour * 60 + bars["timestamp"].dt.minute
    start_number = 9 * 60 + 31
    end_number = scan_end.hour * 60 + scan_end.minute
    bars = bars.loc[minute_number.between(start_number, end_number)].copy()
    bars["minute_index"] = minute_number.loc[bars.index] - start_number
    columns = range(_scan_minute_count(scan_end))
    tickers = sorted(bars["ticker"].astype(str).unique())

    def pivot(value: str) -> np.ndarray:
        return (
            bars.pivot(index="ticker", columns="minute_index", values=value)
            .reindex(index=tickers, columns=columns)
            .to_numpy(dtype=float)
        )

    return tickers, pivot("amount"), pivot("volume"), pivot("close")


def _compact_day_bars(
    day_bars: pd.DataFrame,
    day_universe: pd.DataFrame,
    possible_tickers: set[str],
    params: C2AParameters,
    max_scan_end: time,
) -> pd.DataFrame:
    bars = day_bars.sort_values(["ticker", "timestamp"]).copy()
    candidate_cutoff = params.entry_cutoff
    if params.signal_expiry_minutes is not None:
        cutoff = datetime.combine(date(2000, 1, 1), max_scan_end) + timedelta(
            minutes=int(params.signal_expiry_minutes)
        )
        candidate_cutoff = cutoff.time()
    in_entry_session = bars["timestamp"].dt.time.map(
        lambda value: (
            MORNING_FIRST_COMPLETE_MINUTE <= value <= time(11, 30)
            or time(13, 1) <= value <= candidate_cutoff
        )
    )
    in_entry_session &= bars["timestamp"].dt.time.le(candidate_cutoff)
    candidates = bars.loc[in_entry_session & bars["ticker"].isin(possible_tickers)]
    first = bars.groupby("ticker", sort=False).head(1)
    last = bars.groupby("ticker", sort=False).tail(1)
    limits = normalize_universe(day_universe).set_index("ticker")["lower_limit"]
    lower = bars["ticker"].map(limits).to_numpy(dtype=float)
    locked = (
        np.isclose(bars["open"], lower, atol=0.005)
        & np.isclose(bars["high"], lower, atol=0.005)
        & np.isclose(bars["low"], lower, atol=0.005)
        & np.isclose(bars["close"], lower, atol=0.005)
    )
    first_tradeable = bars.loc[~locked].groupby("ticker", sort=False).head(1)
    pieces = [frame for frame in (candidates, first, first_tradeable, last) if not frame.empty]
    compact = pd.concat(pieces, ignore_index=True)
    compact = compact.drop_duplicates(["timestamp", "ticker"], keep="last")
    return normalize_minutes(compact)


def _cache_fingerprint(
    params: C2AParameters,
    *,
    max_scan_end: time,
    max_c6_threshold: float,
) -> str:
    payload = {
        "schema": CACHE_SCHEMA_VERSION,
        "parameters": params.to_dict(),
        "max_scan_end": max_scan_end.strftime("%H:%M"),
        "max_c6_threshold": max_c6_threshold,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def _scan_minute_count(scan_end: time) -> int:
    return scan_end.hour * 60 + scan_end.minute - (9 * 60 + 31) + 1


def _partition_date(path: Path) -> pd.Timestamp:
    if path.name.endswith(".csv.gz"):
        date_text = path.name.removesuffix(".csv.gz")
    elif path.name.endswith(".parquet"):
        date_text = path.name.removesuffix(".parquet")
    else:
        raise ValueError(f"未知的 C2-A 分区格式: {path}")
    return pd.Timestamp(date.fromisoformat(date_text))


def _parquet_available() -> bool:
    return importlib.util.find_spec("pyarrow") is not None


def _read_partition(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if path.name.endswith(".parquet"):
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=columns)


def _write_partition(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if path.name.endswith(".parquet"):
        frame.to_parquet(
            temporary,
            engine="pyarrow",
            compression="zstd",
            index=False,
        )
    else:
        frame.to_csv(temporary, index=False, compression="gzip")
    temporary.replace(path)


def _write_optional_partition(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        if path.exists():
            path.unlink()
        return
    _write_partition(path, frame)


def _partition_signatures(paths: list[Path]) -> dict[str, dict[str, int]]:
    return {
        path.name: {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for path in paths
    }


def _universe_hash(universe: pd.DataFrame) -> str:
    if universe.empty:
        return hashlib.sha256(b"").hexdigest()
    ordered = normalize_universe(universe).sort_values(["trade_date", "ticker"])
    values = pd.util.hash_pandas_object(ordered, index=False).to_numpy().tobytes()
    return hashlib.sha256(values).hexdigest()


def _partition_row_count(path: Path) -> int:
    if path.name.endswith(".parquet"):
        import pyarrow.parquet as parquet

        return parquet.ParquetFile(path).metadata.num_rows
    with pd.read_csv(path, usecols=["ticker"], chunksize=200_000) as chunks:
        return sum(len(chunk) for chunk in chunks)
