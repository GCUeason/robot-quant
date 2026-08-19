from __future__ import annotations

import hashlib
import json
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from robot_quant.c2a_cloud import (
    main as cloud_main,
    record_phase_failure,
    run_prepare_phase,
    run_research_phase,
    run_review_phase,
    run_scan_phase,
)


SHANGHAI = ZoneInfo("Asia/Shanghai")


def _scan_result() -> dict:
    return {
        "as_of": "2026-08-19",
        "status": "RECONSTRUCTED_SIMULATED_ENTRY",
        "signal_status": "SIMULATED_ENTRY",
        "delivery_timing": "LATE",
        "data_status": "PROXY",
        "cutoff": "10:00",
        "generated_at": "2026-08-19T10:03:40+08:00",
        "signal_valid_until": "2026-08-19T10:03:00+08:00",
        "elapsed_seconds": 15.9,
        "entries": [
            {
                "ticker": "000523",
                "name": "红棉股份",
                "simulated_fill_price": 3.26,
                "shares": 1500,
                "simulated_notional": 4890.0,
            }
        ],
        "watchlist": [],
        "entry_count": 1,
        "candidate_count": 1,
        "baseline_as_of": "2026-08-18",
        "baseline_coverage": 1.0,
        "promotion_gate": "FAIL",
    }


def test_prepare_writes_latest_and_dated_status_without_publishing_pack(
    tmp_path, monkeypatch
) -> None:
    calls = []
    manifest = tmp_path / "data/c2a_fast/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "robot_quant.c2a_cloud._ensure_local_fast_pack",
        lambda *args, **kwargs: calls.append("local-pack"),
    )
    monkeypatch.setattr(
        "robot_quant.c2a_cloud.prepare_fast_pack",
        lambda *args, **kwargs: {
            "last_processed_date": "2026-08-18",
            "updated_days": 1,
            "coverage": 0.997,
        },
    )
    monkeypatch.setattr(
        "robot_quant.c2a_cloud.push_remote_fast_pack",
        lambda *args, **kwargs: calls.append("push"),
    )

    result = run_prepare_phase(
        tmp_path,
        now=datetime(2026, 8, 19, 8, 45, tzinfo=SHANGHAI),
    )

    assert result["status"] == "READY"
    assert calls == ["local-pack", "push"]
    assert result["scheduled_at"] == "2026-08-19T08:45:00+08:00"
    assert result["started_at"] == "2026-08-19T08:45:00+08:00"
    assert result["finished_at"]
    canonical = json.dumps(
        {key: value for key, value in result.items() if key != "payload_sha256"},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert result["payload_sha256"] == hashlib.sha256(canonical).hexdigest()
    assert result["source_manifest_sha256"] == hashlib.sha256(b"{}").hexdigest()
    assert (tmp_path / "data/c2a_results/cloud_prepare_latest.json").exists()
    assert (tmp_path / "data/c2a_results/prepare/2026-08-19.json").exists()
    assert not (tmp_path / "data/c2a_fast/rolling_state.npz").exists()


def test_scan_failure_overwrites_old_latest_with_empty_dated_result(tmp_path, monkeypatch) -> None:
    latest = tmp_path / "data" / "c2a_results" / "fast_latest.json"
    latest.parent.mkdir(parents=True)
    latest.write_text(
        json.dumps({"as_of": "2026-08-18", "entries": [{"ticker": "old"}]}),
        encoding="utf-8",
    )

    def unavailable(*args, **kwargs):
        raise RuntimeError("remote pack unavailable")

    monkeypatch.setattr("robot_quant.c2a_cloud._ensure_local_fast_pack", unavailable)
    result = run_scan_phase(
        tmp_path,
        now=datetime(2026, 8, 19, 10, 3, tzinfo=SHANGHAI),
    )

    assert result["status"] == "DATA_NOT_READY"
    assert result["entries"] == []
    assert result["current_new_entry_allowed"] is False
    written = json.loads(latest.read_text(encoding="utf-8"))
    assert written["as_of"] == "2026-08-19"
    assert written["entries"] == []
    assert (tmp_path / "data/c2a_results/intraday/2026-08-19.json").exists()


def test_prepare_rejects_low_coverage_before_remote_push(tmp_path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr("robot_quant.c2a_cloud._ensure_local_fast_pack", lambda *args: None)
    monkeypatch.setattr(
        "robot_quant.c2a_cloud.prepare_fast_pack",
        lambda *args: {
            "last_processed_date": "2026-08-18",
            "updated_days": 1,
            "coverage": 0.98,
        },
    )
    monkeypatch.setattr(
        "robot_quant.c2a_cloud.push_remote_fast_pack",
        lambda *args, **kwargs: calls.append("push"),
    )

    result = run_prepare_phase(
        tmp_path,
        now=datetime(2026, 8, 19, 8, 45, tzinfo=SHANGHAI),
    )

    assert result["status"] == "DATA_NOT_READY"
    assert "覆盖不足" in result["reason"]
    assert calls == []


def test_late_scheduled_scan_fails_closed_without_contacting_remote(tmp_path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        "robot_quant.c2a_cloud._ensure_local_fast_pack",
        lambda *args, **kwargs: calls.append("fetch"),
    )

    result = run_scan_phase(
        tmp_path,
        now=datetime(2026, 8, 19, 10, 30, tzinfo=SHANGHAI),
    )

    assert result["status"] == "DATA_NOT_READY"
    assert "错误时点触发" in result["reason"]
    assert result["entries"] == []
    assert calls == []


def test_scan_keeps_signal_status_separate_from_late_delivery(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "robot_quant.c2a_cloud._ensure_local_fast_pack", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "robot_quant.c2a_cloud.run_fast_scan", lambda *args, **kwargs: _scan_result()
    )

    result = run_scan_phase(
        tmp_path,
        now=datetime(2026, 8, 19, 10, 3, tzinfo=SHANGHAI),
    )

    assert result["signal_status"] == "SIMULATED_ENTRY"
    assert result["delivery_timing"] == "LATE"
    assert result["current_new_entry_allowed"] is False
    report = (tmp_path / "reports/c2a_scan_latest.md").read_text(encoding="utf-8")
    assert "SIMULATED_ENTRY" in report
    assert "当前追入权限：**无**" in report


def test_review_reconciles_only_same_day_scan_and_marks_next_open_pending(
    tmp_path, monkeypatch
) -> None:
    scan_path = tmp_path / "data" / "c2a_results" / "intraday" / "2026-08-19.json"
    scan_path.parent.mkdir(parents=True)
    scan_path.write_text(json.dumps(_scan_result(), ensure_ascii=False), encoding="utf-8")
    quotes = pd.DataFrame(
        [
            {
                "ticker": "600000",
                "price": 12.0,
                "quote_time": "20260819153001",
            },
            {
                "ticker": "000523",
                "price": 3.42,
                "quote_time": "20260819153001",
            },
        ]
    ).set_index("ticker")
    monkeypatch.setattr(
        "robot_quant.c2a_cloud.fetch_quotes_fast", lambda tickers: quotes.loc[tickers]
    )

    result = run_review_phase(
        tmp_path,
        now=datetime(2026, 8, 19, 16, 30, tzinfo=SHANGHAI),
    )

    assert result["status"] == "READY"
    entry = result["morning_signal"]["entries"][0]
    assert entry["close_price"] == 3.42
    assert entry["gross_mark_to_close_pnl"] == pytest.approx(240.0)
    assert entry["exit_status"] == "NEXT_TRADING_DAY_OPEN_PENDING"
    assert result["execution_permission"] == "PAPER_ONLY"


def test_review_does_not_reuse_yesterdays_latest_signal(tmp_path, monkeypatch) -> None:
    latest = tmp_path / "data" / "c2a_results" / "fast_latest.json"
    latest.parent.mkdir(parents=True)
    latest.write_text(
        json.dumps({**_scan_result(), "as_of": "2026-08-18"}),
        encoding="utf-8",
    )
    result = run_review_phase(
        tmp_path,
        now=datetime(2026, 8, 19, 16, 30, tzinfo=SHANGHAI),
    )

    assert result["status"] == "DATA_NOT_READY"
    assert result["morning_signal"]["entries"] == []
    assert "未沿用旧信号" in result["morning_signal"]["reason"]


def test_review_rejects_same_day_quote_before_market_close(tmp_path, monkeypatch) -> None:
    scan_path = tmp_path / "data" / "c2a_results" / "intraday" / "2026-08-19.json"
    scan_path.parent.mkdir(parents=True)
    scan_path.write_text(json.dumps(_scan_result(), ensure_ascii=False), encoding="utf-8")
    quotes = pd.DataFrame(
        [
            {"ticker": "600000", "price": 12.0, "quote_time": "20260819153001"},
            {"ticker": "000523", "price": 3.42, "quote_time": "20260819100000"},
        ]
    ).set_index("ticker")
    monkeypatch.setattr(
        "robot_quant.c2a_cloud.fetch_quotes_fast", lambda tickers: quotes.loc[tickers]
    )

    result = run_review_phase(
        tmp_path,
        now=datetime(2026, 8, 19, 16, 30, tzinfo=SHANGHAI),
    )

    assert result["status"] == "DATA_NOT_READY"
    assert result["morning_signal"]["entries"] == []
    assert "早于15:00" in result["morning_signal"]["reason"]


def test_research_runs_after_close_without_delaying_review(tmp_path, monkeypatch) -> None:
    quotes = pd.DataFrame(
        [{"ticker": "600000", "price": 12.0, "quote_time": "20260819153001"}]
    ).set_index("ticker")
    calls = []
    monkeypatch.setattr("robot_quant.c2a_cloud.fetch_quotes_fast", lambda tickers: quotes)
    monkeypatch.setattr(
        "robot_quant.c2a_cloud.run_remote_pipeline",
        lambda *args, **kwargs: calls.append("pipeline"),
    )
    monkeypatch.setattr(
        "robot_quant.c2a_cloud.export_remote_fast_pack",
        lambda *args, **kwargs: calls.append("export"),
    )
    monkeypatch.setattr(
        "robot_quant.c2a_cloud.fetch_remote_fast_pack",
        lambda *args, **kwargs: calls.append("fetch"),
    )
    monkeypatch.setattr(
        "robot_quant.c2a_cloud.load_fast_pack",
        lambda *args, **kwargs: SimpleNamespace(last_processed_date=pd.Timestamp("2026-08-19")),
    )

    result = run_research_phase(
        tmp_path,
        now=datetime(2026, 8, 19, 16, 35, tzinfo=SHANGHAI),
        optimize=False,
    )

    assert result["status"] == "READY"
    assert result["research_pipeline"]["status"] == "COMPLETED"
    assert result["next_session_baseline"]["baseline_as_of"] == "2026-08-19"
    assert calls == ["pipeline", "export", "fetch"]
    assert (tmp_path / "reports/c2a_research_latest.md").exists()


def test_research_fails_closed_on_non_closing_market_snapshot(tmp_path, monkeypatch) -> None:
    quotes = pd.DataFrame(
        [{"ticker": "600000", "price": 12.0, "quote_time": "20260818153001"}]
    ).set_index("ticker")
    calls = []
    monkeypatch.setattr("robot_quant.c2a_cloud.fetch_quotes_fast", lambda tickers: quotes)
    monkeypatch.setattr(
        "robot_quant.c2a_cloud.run_remote_pipeline",
        lambda *args, **kwargs: calls.append("pipeline"),
    )

    result = run_research_phase(
        tmp_path,
        now=datetime(2026, 8, 19, 16, 35, tzinfo=SHANGHAI),
        optimize=False,
    )

    assert result["status"] == "DATA_NOT_READY"
    assert "日期不是2026-08-19" in result["market_close_check"]["reason"]
    assert calls == []


def test_preflight_failure_overwrites_scan_latest_with_empty_candidate(tmp_path) -> None:
    latest = tmp_path / "data" / "c2a_results" / "fast_latest.json"
    latest.parent.mkdir(parents=True)
    latest.write_text(json.dumps(_scan_result()), encoding="utf-8")

    result = record_phase_failure(
        tmp_path,
        "scan",
        "云端仓库无法同步 origin/main",
        now=datetime(2026, 8, 19, 9, 59, tzinfo=SHANGHAI),
    )

    written = json.loads(latest.read_text(encoding="utf-8"))
    assert result["status"] == "DATA_NOT_READY"
    assert written["as_of"] == "2026-08-19"
    assert written["entries"] == []


def test_service_mode_returns_retry_exit_code_after_writing_status(monkeypatch) -> None:
    monkeypatch.setattr(
        "robot_quant.c2a_cloud.run_prepare_phase",
        lambda *args, **kwargs: {"status": "DATA_NOT_READY", "retryable": True},
    )

    with pytest.raises(SystemExit) as error:
        cloud_main(["prepare", "--service-mode"])

    assert error.value.code == 75
