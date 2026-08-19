from __future__ import annotations

import json
from dataclasses import replace
from datetime import time

import pandas as pd
import pytest

from robot_quant.c2a import C2AParameters
from robot_quant.c2a_cache import C2AResearchCache
from robot_quant.c2a_data import C2ADataStore


def _stream_universe(dates) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trade_date": day,
                "ticker": "000001",
                "name": "测试",
                "pool": "MAIN",
                "list_date": "2000-01-01",
                "listing_trading_days": 1_000,
                "prevclose": 10.0,
                "prevhigh": 10.2,
                "avg3_amount": 200_000_000.0,
                "float_shares": 1_000_000_000.0,
                "float_mcap": 10_000_000_000.0,
                "is_st": False,
                "is_suspended": False,
                "upper_limit": 11.0,
                "lower_limit": 9.0,
                "limit_streak": 0,
            }
            for day in dates
        ]
    )


def _stream_minutes(day) -> pd.DataFrame:
    timestamps = (
        *pd.date_range(f"{day.date()} 09:31", f"{day.date()} 11:30", freq="1min"),
        *pd.date_range(f"{day.date()} 13:01", f"{day.date()} 15:00", freq="1min"),
    )
    return pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "ticker": "000001",
                "open": 10.0,
                "high": 10.1,
                "low": 9.9,
                "close": 10.0,
                "volume": 1_000_000.0,
                "amount": 10_000_000.0,
            }
            for timestamp in timestamps
        ]
    )


def test_bigquant_stream_cache_is_strict_without_persisting_raw_minutes(tmp_path) -> None:
    dates = pd.bdate_range("2026-01-05", periods=3)
    params = replace(C2AParameters.dynamic_snapshot(), baseline_days=2, scan_end=time(9, 31))
    store = C2ADataStore(tmp_path / "c2a")
    store.initialize("bigquant:test", metadata_verified=True, full_market=True)
    universe = _stream_universe(dates)
    store.write_universe(universe)
    cache = C2AResearchCache(store.root / "research_cache", params)

    for day in dates:
        cache.ingest_validated_day(
            store,
            universe,
            day,
            _stream_minutes(day),
            expected_tickers={"000001"},
            relevant_tickers={"000001"},
        )

    audit = cache.audit_streaming(store, dates[-1], dates[-1])
    summary = cache.streaming_summary(dates[-1])
    assert audit.status == "STRICT"
    assert audit.reasons == ()
    assert audit.minute_rows == 240
    assert summary["last_processed_date"] == dates[-1].date().isoformat()
    assert not store.minute_paths()
    assert cache.read_bars(dates[-1], dates[-1]).shape[0] == 2


def test_stream_audit_keeps_new_stock_without_full_ticker_baseline_strict(
    tmp_path,
) -> None:
    dates = pd.bdate_range("2026-01-05", periods=3)
    params = replace(C2AParameters.dynamic_snapshot(), baseline_days=2, scan_end=time(9, 31))
    store = C2ADataStore(tmp_path / "c2a")
    store.initialize("bigquant:test", metadata_verified=True, full_market=True)
    universe = _stream_universe(dates)
    newcomer = _stream_universe(dates[1:]).assign(
        ticker="001001",
        list_date=dates[1],
        listing_trading_days=[1, 20],
    )
    universe = pd.concat([universe, newcomer], ignore_index=True)
    store.write_universe(universe)
    cache = C2AResearchCache(store.root / "research_cache", params)

    for day in dates:
        day_tickers = set(universe.loc[universe["trade_date"].eq(day), "ticker"])
        minutes = pd.concat(
            [_stream_minutes(day).assign(ticker=ticker) for ticker in sorted(day_tickers)],
            ignore_index=True,
        )
        cache.ingest_validated_day(
            store,
            universe,
            day,
            minutes,
            expected_tickers=day_tickers,
            relevant_tickers={"000001", "001001"},
        )

    audit = cache.audit_streaming(store, dates[-1], dates[-1])
    assert audit.status == "STRICT"
    assert audit.reasons == ()
    assert audit.baseline_days_available == 2


def test_stream_audit_does_not_retroactively_expand_historical_ticker_sets(
    tmp_path,
) -> None:
    dates = pd.bdate_range("2026-01-05", periods=3)
    params = replace(C2AParameters.dynamic_snapshot(), baseline_days=2, scan_end=time(9, 31))
    store = C2ADataStore(tmp_path / "c2a")
    store.initialize("bigquant:test", metadata_verified=True, full_market=True)
    universe = _stream_universe(dates)
    store.write_universe(universe)
    cache = C2AResearchCache(store.root / "research_cache", params)
    for day in dates:
        cache.ingest_validated_day(
            store,
            universe,
            day,
            _stream_minutes(day),
            expected_tickers={"000001"},
            relevant_tickers={"000001"},
        )

    manifest = json.loads(cache.manifest_path.read_text(encoding="utf-8"))
    manifest["relevant_tickers"].append("001001")
    cache.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    audit = cache.audit_streaming(store, dates[-1], dates[-1])
    assert audit.status == "STRICT"
    assert audit.reasons == ()


def test_stream_audit_rejects_insufficient_global_baseline(tmp_path) -> None:
    dates = pd.bdate_range("2026-01-05", periods=2)
    params = replace(C2AParameters.dynamic_snapshot(), baseline_days=2, scan_end=time(9, 31))
    store = C2ADataStore(tmp_path / "c2a")
    store.initialize("bigquant:test", metadata_verified=True, full_market=True)
    universe = _stream_universe(dates)
    store.write_universe(universe)
    cache = C2AResearchCache(store.root / "research_cache", params)

    for day in dates:
        cache.ingest_validated_day(
            store,
            universe,
            day,
            _stream_minutes(day),
            expected_tickers={"000001"},
            relevant_tickers={"000001"},
        )

    audit = cache.audit_streaming(store, dates[-1], dates[-1])
    assert audit.status == "PROXY"
    assert audit.reasons == ("insufficient_stream_global_baseline_days",)
    assert audit.baseline_days_available == 1


def test_bigquant_stream_cache_rejects_incomplete_240_bar_grid(tmp_path) -> None:
    day = pd.Timestamp("2026-01-05")
    store = C2ADataStore(tmp_path / "c2a")
    store.initialize("bigquant:test", metadata_verified=True, full_market=True)
    universe = _stream_universe([day])
    store.write_universe(universe)
    cache = C2AResearchCache(store.root / "research_cache", C2AParameters.dynamic_snapshot())

    with pytest.raises(RuntimeError, match="240根"):
        cache.ingest_validated_day(
            store,
            universe,
            day,
            _stream_minutes(day).iloc[:-1],
            expected_tickers={"000001"},
            relevant_tickers={"000001"},
        )


def test_stream_cache_recovers_when_state_commit_preceded_manifest(tmp_path) -> None:
    dates = pd.bdate_range("2026-01-05", periods=4)
    params = replace(C2AParameters.dynamic_snapshot(), baseline_days=2, scan_end=time(9, 31))
    store = C2ADataStore(tmp_path / "c2a")
    store.initialize("bigquant:test", metadata_verified=True, full_market=True)
    universe = _stream_universe(dates)
    store.write_universe(universe)
    cache = C2AResearchCache(store.root / "research_cache", params)
    for day in dates[:3]:
        cache.ingest_validated_day(
            store,
            universe,
            day,
            _stream_minutes(day),
            expected_tickers={"000001"},
            relevant_tickers={"000001"},
        )

    manifest = json.loads(cache.manifest_path.read_text(encoding="utf-8"))
    manifest["validated_days"].pop(dates[2].date().isoformat())
    manifest["last_processed_date"] = dates[1].date().isoformat()
    cache.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    cache.ingest_validated_day(
        store,
        universe,
        dates[3],
        _stream_minutes(dates[3]),
        expected_tickers={"000001"},
        relevant_tickers={"000001"},
    )

    recovered = json.loads(cache.manifest_path.read_text(encoding="utf-8"))
    assert recovered["last_processed_date"] == dates[3].date().isoformat()
    assert recovered["validated_days"][dates[2].date().isoformat()][
        "recovered_after_interrupted_commit"
    ]
    assert cache.audit_streaming(store, dates[2], dates[3]).status == "STRICT"


def test_stream_cache_rolls_manifest_ahead_back_to_state_before_reprocessing(
    tmp_path,
) -> None:
    dates = pd.bdate_range("2026-01-05", periods=3)
    params = replace(C2AParameters.dynamic_snapshot(), baseline_days=2, scan_end=time(9, 31))
    store = C2ADataStore(tmp_path / "c2a")
    store.initialize("bigquant:test", metadata_verified=True, full_market=True)
    universe = _stream_universe(dates)
    store.write_universe(universe)
    cache = C2AResearchCache(store.root / "research_cache", params)
    for day in dates[:2]:
        cache.ingest_validated_day(
            store,
            universe,
            day,
            _stream_minutes(day),
            expected_tickers={"000001"},
            relevant_tickers={"000001"},
        )

    manifest = json.loads(cache.manifest_path.read_text(encoding="utf-8"))
    manifest["validated_days"][dates[2].date().isoformat()] = {
        "ticker_count": 1,
        "ticker_hash": "premature",
        "raw_rows": 240,
        "possible_tickers": 0,
    }
    manifest["last_processed_date"] = dates[2].date().isoformat()
    cache.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    cache.ingest_validated_day(
        store,
        universe,
        dates[2],
        _stream_minutes(dates[2]),
        expected_tickers={"000001"},
        relevant_tickers={"000001"},
    )

    repaired = json.loads(cache.manifest_path.read_text(encoding="utf-8"))
    assert repaired["validated_days"][dates[2].date().isoformat()]["ticker_hash"] != "premature"
    assert cache.audit_streaming(store, dates[2], dates[2]).status == "STRICT"
