from __future__ import annotations

import json
import stat
from dataclasses import replace
from datetime import time

import pandas as pd
import pytest

from robot_quant.c2a import C2AParameters
from robot_quant.c2a_data import (
    C2ADataStore,
    TushareRestClient,
    build_tushare_universe,
    configure_tushare_token_from_clipboard,
    download_tushare_minutes,
    import_c2a_csv,
    save_tushare_token,
)


def _universe(dates) -> pd.DataFrame:
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


def _minutes(dates) -> pd.DataFrame:
    rows = []
    for day in dates:
        morning = pd.date_range(f"{day.date()} 09:31", f"{day.date()} 11:30", freq="1min")
        afternoon = pd.date_range(f"{day.date()} 13:01", f"{day.date()} 15:00", freq="1min")
        for timestamp in (*morning, *afternoon):
            rows.append(
                {
                    "timestamp": timestamp,
                    "ticker": "000001",
                    "open": 10,
                    "high": 10.1,
                    "low": 9.9,
                    "close": 10,
                    "volume": 1_000_000,
                    "amount": 10_000_000,
                }
            )
    return pd.DataFrame(rows)


def test_data_store_audit_only_marks_complete_full_market_data_strict(tmp_path) -> None:
    dates = pd.bdate_range("2026-01-05", periods=3)
    store = C2ADataStore(tmp_path / "c2a")
    store.initialize("licensed_test", metadata_verified=True, full_market=True)
    store.write_universe(_universe(dates))
    store.write_minutes(_minutes(dates))
    params = replace(C2AParameters(), baseline_days=2, scan_end=time(9, 31))
    audit = store.audit(dates[-1], dates[-1], params)
    assert audit.status == "STRICT"
    assert audit.reasons == ()
    assert audit.baseline_days_available == 2


def test_data_store_defaults_to_proxy_when_provenance_is_unverified(tmp_path) -> None:
    dates = pd.bdate_range("2026-01-05", periods=3)
    store = C2ADataStore(tmp_path / "c2a")
    store.initialize("unknown", metadata_verified=False, full_market=False)
    store.write_universe(_universe(dates))
    store.write_minutes(_minutes(dates))
    params = replace(C2AParameters(), baseline_days=2, scan_end=time(9, 31))
    audit = store.audit(dates[-1], dates[-1], params)
    assert audit.status == "PROXY"
    assert "historical_metadata_not_verified" in audit.reasons
    assert "not_full_a_share_market" in audit.reasons


def test_data_store_rejects_unscaled_amount_units(tmp_path) -> None:
    dates = pd.bdate_range("2026-01-05", periods=3)
    store = C2ADataStore(tmp_path / "c2a")
    store.initialize(
        "vendor_test",
        metadata_verified=True,
        full_market=True,
        amount_unit="thousand_CNY",
    )
    store.write_universe(_universe(dates))
    store.write_minutes(_minutes(dates))
    params = replace(C2AParameters(), baseline_days=2, scan_end=time(9, 31))
    audit = store.audit(dates[-1], dates[-1], params)
    assert audit.status == "PROXY"
    assert "minute_amount_unit_must_be_CNY" in audit.reasons


def test_data_store_rejects_gap_inside_required_baseline_window(tmp_path) -> None:
    dates = pd.bdate_range("2026-01-05", periods=3)
    store = C2ADataStore(tmp_path / "c2a")
    store.initialize("licensed_test", metadata_verified=True, full_market=True)
    store.write_universe(_universe(dates))
    store.write_minutes(_minutes([dates[0], dates[-1]]))
    params = replace(C2AParameters(), baseline_days=2, scan_end=time(9, 31))
    audit = store.audit(dates[-1], dates[-1], params)
    assert audit.status == "PROXY"
    assert "insufficient_pre_start_baseline_days" in audit.reasons


def test_data_store_validates_exact_session_labels_not_only_row_count(tmp_path) -> None:
    dates = pd.bdate_range("2026-01-05", periods=3)
    minutes = _minutes(dates)
    target = dates[-1]
    missing_close = minutes["timestamp"].eq(pd.Timestamp(f"{target.date()} 15:00"))
    minutes.loc[missing_close, "timestamp"] = pd.Timestamp(f"{target.date()} 13:00")
    store = C2ADataStore(tmp_path / "c2a")
    store.initialize("licensed_test", metadata_verified=True, full_market=True)
    store.write_universe(_universe(dates))
    store.write_minutes(minutes)
    params = replace(C2AParameters(), baseline_days=2, scan_end=time(9, 31))
    audit = store.audit(target, target, params)
    assert audit.status == "PROXY"
    assert f"session_minute_grid_incomplete:{target.date().isoformat()}" in audit.reasons


def test_universe_updates_merge_dates_instead_of_overwriting_history(tmp_path) -> None:
    dates = pd.bdate_range("2026-01-05", periods=2)
    store = C2ADataStore(tmp_path / "c2a")
    store.write_universe(_universe([dates[0]]))
    store.write_universe(_universe([dates[1]]))
    result = store.read_universe()
    assert result["trade_date"].tolist() == list(dates)


def test_late_eligible_stock_uses_its_own_rolling_baseline_dates(tmp_path) -> None:
    dates = pd.bdate_range("2026-01-05", periods=42)
    late_dates = dates[21:]
    universe = _universe(dates)
    late_universe = _universe(late_dates).assign(
        ticker="300001",
        name="后上市测试",
        pool="GROWTH",
        list_date=late_dates[0],
        listing_trading_days=range(len(late_dates)),
        upper_limit=12.0,
        lower_limit=8.0,
    )
    minutes = pd.concat(
        [_minutes(dates), _minutes(late_dates).assign(ticker="300001")],
        ignore_index=True,
    )
    store = C2ADataStore(tmp_path / "c2a")
    store.initialize("licensed_test", metadata_verified=True, full_market=True)
    store.write_universe(pd.concat([universe, late_universe], ignore_index=True))
    store.write_minutes(minutes)
    params = replace(C2AParameters(), baseline_days=20, scan_end=time(9, 31))
    audit = store.audit(dates[20], dates[-1], params)
    assert audit.status == "STRICT"
    assert audit.reasons == ()


def test_suspended_calendar_day_does_not_consume_stock_baseline_slot(tmp_path) -> None:
    dates = pd.bdate_range("2026-01-05", periods=4)
    universe = _universe(dates)
    universe.loc[universe["trade_date"].eq(dates[1]), "is_suspended"] = True
    store = C2ADataStore(tmp_path / "c2a")
    store.initialize("licensed_test", metadata_verified=True, full_market=True)
    store.write_universe(universe)
    store.write_minutes(_minutes([dates[0], dates[2], dates[3]]))
    params = replace(C2AParameters(), baseline_days=2, scan_end=time(9, 31))
    audit = store.audit(dates[3], dates[3], params)
    assert audit.status == "STRICT"
    assert audit.reasons == ()


def test_tushare_client_fails_closed_without_token(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    monkeypatch.delenv("TS_TOKEN", raising=False)
    monkeypatch.setenv("TUSHARE_TOKEN_FILE", str(tmp_path / "missing-token"))
    with pytest.raises(RuntimeError, match="TUSHARE_TOKEN"):
        TushareRestClient()


def test_tushare_token_is_saved_outside_repo_with_owner_only_permissions(tmp_path) -> None:
    token_path = tmp_path / "config" / "tushare_token"
    saved = save_tushare_token("a" * 64, token_path)
    assert saved == token_path
    assert token_path.read_text(encoding="utf-8") == "a" * 64
    assert stat.S_IMODE(token_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(token_path.parent.stat().st_mode) == 0o700


def test_tushare_client_reads_secure_token_file(tmp_path, monkeypatch) -> None:
    token_path = tmp_path / "tushare_token"
    save_tushare_token("b" * 64, token_path)
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    monkeypatch.delenv("TS_TOKEN", raising=False)
    monkeypatch.setenv("TUSHARE_TOKEN_FILE", str(token_path))
    client = TushareRestClient()
    assert client.token == "b" * 64


def test_clipboard_configuration_never_requires_token_in_command_arguments(
    tmp_path, monkeypatch
) -> None:
    calls = []

    class Completed:
        stdout = "c" * 64

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return Completed()

    monkeypatch.setattr("robot_quant.c2a_data.subprocess.run", fake_run)
    token_path = tmp_path / "tushare_token"
    configured = configure_tushare_token_from_clipboard(token_path)
    assert configured == token_path
    assert calls == [(["pbpaste"], {"check": True, "capture_output": True, "text": True})]
    assert token_path.read_text(encoding="utf-8") == "c" * 64


def test_empty_store_is_data_not_ready_not_a_zero_return_proxy(tmp_path) -> None:
    audit = C2ADataStore(tmp_path / "c2a").audit("2026-01-01", "2026-08-12")
    assert audit.status == "DATA_NOT_READY"
    assert audit.minute_rows == 0


class _FakeTushare:
    days = ("20260105", "20260106", "20260107", "20260108", "20260109")

    def query(self, api_name, params=None, fields=()):
        params = params or {}
        if api_name == "trade_cal":
            return pd.DataFrame({"cal_date": self.days, "is_open": [1] * len(self.days)})
        if api_name == "stock_basic":
            if params["list_status"] != "L":
                return pd.DataFrame(columns=fields)
            return pd.DataFrame(
                [["000001.SZ", "000001", "测试", "20000101", "", "L"]],
                columns=fields,
            )
        day = params["trade_date"]
        if api_name == "daily":
            close = 11.0 if day in {"20260106", "20260107"} else 10.0
            return pd.DataFrame([["000001.SZ", day, close + 0.1, close, 200_000.0]], columns=fields)
        if api_name == "daily_basic":
            return pd.DataFrame([["000001.SZ", day, 100_000.0]], columns=fields)
        if api_name == "stk_limit":
            return pd.DataFrame([["000001.SZ", day, 10.0, 11.0, 9.0]], columns=fields)
        if api_name == "stock_st":
            if day == "20260109":
                return pd.DataFrame([["000001.SZ", "ST测试", day]], columns=fields)
            return pd.DataFrame(columns=fields)
        if api_name == "suspend_d":
            return pd.DataFrame(columns=fields)
        raise AssertionError(api_name)


def test_tushare_universe_uses_prior_days_for_avg_amount_high_and_limit_streak(tmp_path) -> None:
    store = C2ADataStore(tmp_path / "c2a")
    result = build_tushare_universe(
        store,
        "2026-01-08",
        "2026-01-09",
        client=_FakeTushare(),
    )
    first = result.loc[result["trade_date"].eq(pd.Timestamp("2026-01-08"))].iloc[0]
    assert first["avg3_amount"] == 200_000_000.0
    assert first["prevhigh"] == 11.1
    assert first["limit_streak"] == 2
    last = result.loc[result["trade_date"].eq(pd.Timestamp("2026-01-09"))].iloc[0]
    assert bool(last["is_st"])
    assert store.manifest()["metadata_verified"] is True


def test_tushare_universe_does_not_checkpoint_empty_daily_response(tmp_path) -> None:
    class EmptyDailyTushare(_FakeTushare):
        def query(self, api_name, params=None, fields=()):
            if api_name == "daily" and (params or {}).get("trade_date") == "20260105":
                return pd.DataFrame(columns=fields)
            return super().query(api_name, params, fields)

    store = C2ADataStore(tmp_path / "c2a")
    with pytest.raises(RuntimeError, match="daily 未返回 20260105"):
        build_tushare_universe(
            store,
            "2026-01-08",
            "2026-01-09",
            client=EmptyDailyTushare(),
        )
    assert not (store.root / "tushare_universe_raw_daily_basic" / "2026-01-05.csv.gz").exists()


def test_tushare_universe_requires_all_non_suspended_daily_codes(tmp_path) -> None:
    class PartialDailyTushare(_FakeTushare):
        def query(self, api_name, params=None, fields=()):
            if api_name == "daily":
                day = (params or {})["trade_date"]
                return pd.DataFrame(
                    [
                        ["000001.SZ", day, 10.1, 10.0, 200_000.0],
                        ["000002.SZ", day, 10.1, 10.0, 200_000.0],
                    ],
                    columns=fields,
                )
            return super().query(api_name, params, fields)

    store = C2ADataStore(tmp_path / "c2a")
    with pytest.raises(RuntimeError, match="覆盖率仅 50.00%"):
        build_tushare_universe(
            store,
            "2026-01-08",
            "2026-01-09",
            client=PartialDailyTushare(),
        )


def test_tushare_current_month_file_is_incrementally_extended(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_fetch(_client, ticker, start_date, end_date):
        calls.append((pd.Timestamp(start_date).date(), pd.Timestamp(end_date).date()))
        rows = []
        for day in pd.date_range(start_date, end_date, freq="D"):
            rows.append(
                {
                    "timestamp": f"{day.date()} 15:00",
                    "ticker": ticker,
                    "open": 10,
                    "high": 10,
                    "low": 10,
                    "close": 10,
                    "volume": 100,
                    "amount": 1_000,
                }
            )
        return pd.DataFrame(rows)

    monkeypatch.setattr("robot_quant.c2a_data.fetch_tushare_minute_range", fake_fetch)
    store = C2ADataStore(tmp_path / "c2a")
    download_tushare_minutes(store, ["000001"], "2026-01-05", "2026-01-05", client=object())
    result = download_tushare_minutes(
        store, ["000001"], "2026-01-05", "2026-01-06", client=object()
    )
    raw = pd.read_csv(store.root / "tushare_raw" / "2026-01" / "000001.csv.gz")
    assert len(raw) == 2
    assert result["updated_files"] == 1
    assert calls[-1][0].isoformat() == "2026-01-06"

    download_tushare_minutes(store, ["000002"], "2026-01-05", "2026-01-05", client=object())
    daily = pd.read_csv(store.minute_path(pd.Timestamp("2026-01-05").date()))
    assert set(daily["ticker"].astype(str).str.zfill(6)) == {"000001", "000002"}

    call_count = len(calls)
    repeated = download_tushare_minutes(
        store, ["000001", "000002"], "2026-01-05", "2026-01-05", client=object()
    )
    assert len(calls) == call_count
    assert repeated["partitions"] == 0


def test_tushare_empty_response_on_expected_session_does_not_commit_progress(
    tmp_path, monkeypatch
) -> None:
    store = C2ADataStore(tmp_path / "c2a")
    store.initialize("test", metadata_verified=True, full_market=True)
    store.write_universe(_universe([pd.Timestamp("2026-01-05")]))
    monkeypatch.setattr(
        "robot_quant.c2a_data.fetch_tushare_minute_range",
        lambda *_args, **_kwargs: pd.DataFrame(),
    )
    with pytest.raises(RuntimeError, match="存在预期交易日"):
        download_tushare_minutes(
            store,
            ["000001"],
            "2026-01-05",
            "2026-01-05",
            client=object(),
        )
    manifest = json.loads(
        (store.root / "tushare_raw" / "download_manifest.json").read_text(encoding="utf-8")
    )
    assert "2026-01/000001" not in manifest["queried_through"]


def test_tushare_partial_nonempty_session_does_not_commit_progress(tmp_path, monkeypatch) -> None:
    store = C2ADataStore(tmp_path / "c2a")
    store.initialize("test", metadata_verified=True, full_market=True)
    store.write_universe(_universe([pd.Timestamp("2026-01-05")]))
    monkeypatch.setattr(
        "robot_quant.c2a_data.fetch_tushare_minute_range",
        lambda *_args, **_kwargs: pd.DataFrame(
            [
                {
                    "timestamp": "2026-01-05 15:00",
                    "ticker": "000001",
                    "open": 10,
                    "high": 10,
                    "low": 10,
                    "close": 10,
                    "volume": 100,
                    "amount": 1_000,
                }
            ]
        ),
    )
    with pytest.raises(RuntimeError, match="不是完整240根"):
        download_tushare_minutes(
            store,
            ["000001"],
            "2026-01-05",
            "2026-01-05",
            client=object(),
        )
    manifest = json.loads(
        (store.root / "tushare_raw" / "download_manifest.json").read_text(encoding="utf-8")
    )
    assert "2026-01/000001" not in manifest["queried_through"]


def test_tushare_resume_consolidates_pending_raw_before_committing_progress(
    tmp_path, monkeypatch
) -> None:
    from robot_quant import c2a_data

    def fake_fetch(_client, ticker, start_date, _end_date):
        return pd.DataFrame(
            [
                {
                    "timestamp": f"{pd.Timestamp(start_date).date()} 15:00",
                    "ticker": ticker,
                    "open": 10,
                    "high": 10,
                    "low": 10,
                    "close": 10,
                    "volume": 100,
                    "amount": 1_000,
                }
            ]
        )

    store = C2ADataStore(tmp_path / "c2a")
    monkeypatch.setattr("robot_quant.c2a_data.fetch_tushare_minute_range", fake_fetch)
    original_consolidate = c2a_data.consolidate_tushare_raw

    def interrupted(*_args, **_kwargs):
        raise RuntimeError("模拟合并中断")

    monkeypatch.setattr("robot_quant.c2a_data.consolidate_tushare_raw", interrupted)
    with pytest.raises(RuntimeError, match="合并中断"):
        download_tushare_minutes(
            store,
            ["000001"],
            "2026-01-05",
            "2026-01-05",
            client=object(),
        )
    manifest_path = store.root / "tushare_raw" / "download_manifest.json"
    interrupted_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "2026-01/000001" in interrupted_manifest["pending_slices"]
    assert "2026-01/000001" not in interrupted_manifest["queried_through"]

    monkeypatch.setattr("robot_quant.c2a_data.consolidate_tushare_raw", original_consolidate)
    download_tushare_minutes(
        store,
        ["000001"],
        "2026-01-06",
        "2026-01-06",
        client=object(),
    )
    resumed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert resumed_manifest["pending_slices"] == {}
    assert resumed_manifest["queried_through"]["2026-01/000001"] == "2026-01-06"
    assert store.minute_path(pd.Timestamp("2026-01-05").date()).exists()
    assert store.minute_path(pd.Timestamp("2026-01-06").date()).exists()


def test_incomplete_supplier_import_transaction_blocks_strict_audit(tmp_path) -> None:
    dates = pd.bdate_range("2026-01-05", periods=3)
    store = C2ADataStore(tmp_path / "c2a")
    store.initialize("old_source", metadata_verified=True, full_market=True)
    store.write_universe(_universe(dates))
    store.write_minutes(_minutes(dates))
    store.import_transaction_path.write_text("{}", encoding="utf-8")
    params = replace(C2AParameters(), baseline_days=2, scan_end=time(9, 31))
    audit = store.audit(dates[-1], dates[-1], params)
    assert audit.status == "PROXY"
    assert "import_transaction_incomplete" in audit.reasons


def test_supplier_import_commits_manifest_last_and_clears_transaction(tmp_path) -> None:
    dates = pd.bdate_range("2026-01-05", periods=3)
    minute_csv = tmp_path / "minutes.csv"
    universe_csv = tmp_path / "universe.csv"
    _minutes(dates).to_csv(minute_csv, index=False)
    _universe(dates).to_csv(universe_csv, index=False)
    store = C2ADataStore(tmp_path / "c2a")
    result = import_c2a_csv(
        store,
        minute_csv,
        universe_csv,
        source="licensed_vendor",
        metadata_verified=True,
        full_market=True,
    )
    assert result["minute_partitions"] == 3
    assert not store.import_transaction_path.exists()
    assert store.manifest()["source"] == "licensed_vendor"
