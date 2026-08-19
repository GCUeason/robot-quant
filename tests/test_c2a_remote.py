from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from robot_quant.c2a_remote import (
    _run,
    _validate_date,
    _validate_host,
    _validate_remote_root,
    export_remote_fast_pack,
    fetch_remote_fast_pack,
    fetch_remote_results,
    push_remote_fast_pack,
    run_remote_backtest,
    run_remote_pipeline,
    sync_c2a_code,
    update_remote_data,
)
from robot_quant.c2a_scheduler import _run_once


def _project(tmp_path: Path) -> Path:
    package = tmp_path / "src" / "robot_quant"
    package.mkdir(parents=True)
    for name in (
        "__init__.py",
        "__main__.py",
        "cli.py",
        "c2a.py",
        "c2a_bigquant.py",
        "c2a_remote.py",
    ):
        (package / name).write_text("", encoding="utf-8")
    return tmp_path


def test_sync_only_transmits_c2a_source_files(tmp_path, monkeypatch) -> None:
    root = _project(tmp_path)
    calls = []
    monkeypatch.setattr(
        "robot_quant.c2a_remote._run", lambda command, *, cwd: calls.append((command, cwd))
    )

    sync_c2a_code(root)

    assert calls[0][0] == [
        "ssh",
        "bigquant-aistudio",
        "mkdir -p /home/aiuser/work/robot-quant/src/robot_quant",
    ]
    rsync = calls[1][0]
    assert rsync[:2] == ["rsync", "-az"]
    assert all("data/" not in item and ".ssh" not in item for item in rsync[2:-1])
    assert rsync[-1].endswith(":/home/aiuser/work/robot-quant/src/robot_quant/")


def test_remote_update_uses_stream_cache_and_validated_dates(tmp_path, monkeypatch) -> None:
    root = _project(tmp_path)
    calls = []
    monkeypatch.setattr(
        "robot_quant.c2a_remote._run", lambda command, *, cwd: calls.append((command, cwd))
    )

    update_remote_data(root, "2026-01-01", "2026-08-11", sync_code=False)

    command = calls[-1][0]
    assert command[:2] == ["ssh", "bigquant-aistudio"]
    assert "c2a-update-bigquant" in command[-1]
    assert "--stream-cache" in command[-1]
    assert "2026-08-11" in command[-1]


def test_fetch_uses_explicit_result_allowlist(tmp_path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        "robot_quant.c2a_remote._run", lambda command, *, cwd: calls.append((command, cwd))
    )

    fetch_remote_results(tmp_path)

    result_sync = calls[-1][0]
    assert result_sync[:2] == ["rsync", "-az"]
    assert "--include" in result_sync
    assert "baseline_equity.csv" in result_sync
    assert "walk_forward_selections.csv" in result_sync
    assert result_sync[-4:-2] == ["--exclude", "*"]
    assert "parameter_paths" not in " ".join(result_sync)


def test_fast_pack_round_trip_only_uses_ignored_compact_directory(tmp_path, monkeypatch) -> None:
    root = _project(tmp_path)
    pack = root / "data" / "c2a_fast"
    pack.mkdir(parents=True)
    for name in ("rolling_state.npz", "universe.csv.gz", "manifest.json"):
        (pack / name).write_bytes(b"fixture")
    calls = []
    monkeypatch.setattr(
        "robot_quant.c2a_remote._run", lambda command, *, cwd: calls.append((command, cwd))
    )

    export_remote_fast_pack(root, sync_code=False)
    fetch_remote_fast_pack(root)
    push_remote_fast_pack(root)

    assert "robot_quant.c2a_fast export-pack" in calls[0][0][-1]
    assert calls[1][0] == [
        "rsync",
        "-az",
        "bigquant-aistudio:/home/aiuser/work/robot-quant/data/c2a_fast/",
        f"{pack.resolve()}/",
    ]
    assert calls[-1][0] == [
        "rsync",
        "-az",
        f"{pack.resolve()}/",
        "bigquant-aistudio:/home/aiuser/work/robot-quant/data/c2a_fast/",
    ]


def test_push_fast_pack_fails_closed_when_pack_is_incomplete(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="紧凑基线缺少文件"):
        push_remote_fast_pack(tmp_path)


def test_remote_backtest_can_carry_forward_without_optimization(tmp_path, monkeypatch) -> None:
    root = _project(tmp_path)
    calls = []
    monkeypatch.setattr(
        "robot_quant.c2a_remote._run", lambda command, *, cwd: calls.append((command, cwd))
    )

    run_remote_backtest(root, end_date="2026-08-11", optimize=False)

    assert "c2a-backtest" in calls[-1][0][-1]
    assert "--no-optimize" in calls[-1][0][-1]
    assert "--variant v1.2-challenger" in calls[-1][0][-1]


def test_pipeline_keeps_data_foundation_while_backtesting_short_window(
    tmp_path, monkeypatch
) -> None:
    calls = []
    monkeypatch.setattr(
        "robot_quant.c2a_remote.sync_c2a_code",
        lambda *args, **kwargs: calls.append(("sync", args, kwargs)),
    )
    monkeypatch.setattr(
        "robot_quant.c2a_remote.update_remote_data",
        lambda *args, **kwargs: calls.append(("update", args, kwargs)),
    )
    monkeypatch.setattr(
        "robot_quant.c2a_remote.run_remote_backtest",
        lambda *args, **kwargs: calls.append(("backtest", args, kwargs)),
    )
    monkeypatch.setattr(
        "robot_quant.c2a_remote.fetch_remote_results",
        lambda *args, **kwargs: calls.append(("fetch", args, kwargs)),
    )

    run_remote_pipeline(
        tmp_path,
        "2026-07-01",
        "2026-08-12",
        optimize=False,
    )

    assert calls[1][0] == "update"
    assert calls[1][1][1:3] == ("2026-01-01", "2026-08-12")
    assert calls[2][0] == "backtest"
    assert calls[2][1][1:3] == ("2026-07-01", "2026-08-12")


def test_remote_arguments_reject_shell_injection() -> None:
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        _validate_date("2026-08-11; touch bad")
    with pytest.raises(ValueError, match="SSH Host"):
        _validate_host("host; touch bad")
    with pytest.raises(ValueError, match="SSH Host"):
        _validate_host("-V")
    with pytest.raises(ValueError, match="SSH Host"):
        _validate_host("-Ffoo")
    with pytest.raises(ValueError, match="远端目录"):
        _validate_remote_root("/home/aiuser/work/robot-quant; touch bad")
    with pytest.raises(ValueError, match="远端目录"):
        _validate_remote_root("/")
    with pytest.raises(ValueError, match="远端目录"):
        _validate_remote_root("/home/aiuser/../root")
    with pytest.raises(ValueError, match="远端目录"):
        _validate_remote_root("/home/./robot-quant")
    with pytest.raises(ValueError, match="远端目录"):
        _validate_remote_root("//home/aiuser/work/robot-quant")
    with pytest.raises(ValueError, match="远端目录"):
        _validate_remote_root("/./")
    with pytest.raises(ValueError, match="远端目录"):
        _validate_remote_root("///")
    with pytest.raises(ValueError, match="远端目录"):
        _validate_remote_root("//")


def test_remote_runner_enforces_batch_mode_connection_and_process_timeouts(
    tmp_path, monkeypatch
) -> None:
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))

    monkeypatch.setattr("robot_quant.c2a_remote.subprocess.run", fake_run)

    _run(["ssh", "quant-host", "true"], cwd=tmp_path)
    _run(["rsync", "-az", "source/", "quant-host:/srv/quant/"], cwd=tmp_path)

    command, kwargs = calls[0]
    assert command[:4] == ["ssh", "-o", "BatchMode=yes", "-o"]
    assert "ConnectTimeout=15" in command
    assert kwargs["timeout"] == 3 * 60 * 60
    rsync_command, rsync_kwargs = calls[1]
    assert rsync_command[:3] == ["rsync", "--timeout=120", "-e"]
    assert "BatchMode=yes" in rsync_command[3]
    assert "ConnectTimeout=15" in rsync_command[3]
    assert rsync_kwargs["timeout"] == 3 * 60 * 60


def test_scheduler_uses_remote_pipeline_and_only_optimizes_on_friday(tmp_path, monkeypatch) -> None:
    calls = []

    class Wednesday:
        @classmethod
        def now(cls, timezone):
            return datetime(2026, 8, 12, 16, 30, tzinfo=timezone)

    monkeypatch.setattr("robot_quant.c2a_scheduler.datetime", Wednesday)
    monkeypatch.setattr(
        "robot_quant.c2a_scheduler.run_remote_pipeline",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    _run_once(tmp_path)
    assert calls[-1] == (
        (tmp_path, "2026-06-28", "2026-08-12"),
        {"optimize": False, "variant": "v1.2-challenger"},
    )

    class Friday:
        @classmethod
        def now(cls, timezone):
            return datetime(2026, 8, 14, 16, 30, tzinfo=ZoneInfo("Asia/Shanghai"))

    monkeypatch.setattr("robot_quant.c2a_scheduler.datetime", Friday)
    _run_once(tmp_path)
    assert calls[-1][1]["optimize"] is True
