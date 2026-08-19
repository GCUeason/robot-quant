from __future__ import annotations

import hashlib
import json
import shlex
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

import robot_quant.c2a_remote as c2a_remote
from robot_quant.c2a_fast import FastPack, save_fast_pack
from robot_quant.c2a_remote import (
    _run,
    _validate_date,
    _validate_host,
    _validate_remote_root,
    export_remote_fast_pack,
    fetch_remote_audit_results,
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


def _write_fast_pack(root: Path, last_processed_date: str) -> None:
    universe = pd.DataFrame(
        [
            {
                "trade_date": last_processed_date,
                "ticker": "600216",
                "name": "浙江医药",
                "pool": "MAIN",
                "list_date": "1999-10-21",
                "listing_trading_days": 1_000,
                "prevclose": 13.5,
                "prevhigh": 13.6,
                "avg3_amount": 200_000_000.0,
                "float_shares": 1_000_000_000.0,
                "float_mcap": 13_500_000_000.0,
                "is_st": False,
                "is_suspended": False,
                "upper_limit": 14.85,
                "lower_limit": 12.15,
                "limit_streak": 0,
            }
        ]
    )
    save_fast_pack(
        FastPack(
            root=root,
            tickers=["600216"],
            amount_history=np.ones((1, 30, 20)),
            volume_history=np.ones((1, 30, 20)),
            pointers=np.zeros(1, dtype=np.int64),
            counts=np.full(1, 20, dtype=np.int64),
            last_processed_date=pd.Timestamp(last_processed_date),
            universe=universe,
            manifest={"schema_version": 1},
        )
    )


def _pack_file_hashes(root: Path) -> dict[str, str]:
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("rolling_state.npz", "universe.csv.gz", "manifest.json")
    }


def _write_carried_snapshot(root: Path, marker: str, status: str = "COMPLETED") -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "latest_state.json").write_text(
        json.dumps(
            {
                "walk_forward": {
                    "status": status,
                    "optimization_as_of": "2026-08-14",
                },
                "snapshot_marker": marker,
            }
        ),
        encoding="utf-8",
    )
    for name in c2a_remote.CARRIED_FORWARD_RESULT_FILES[1:]:
        (root / name).write_text(f"{marker} {name}\n", encoding="utf-8")


def _install_fake_fast_pack_remote(
    monkeypatch,
    remote_project: Path,
    *,
    transfer_error: Exception | None = None,
    cleanup_error: Exception | None = None,
) -> list[list[str]]:
    """在 push 公共边界外模拟 SSH/rsync，远端提升逻辑仍运行真实代码。"""

    calls: list[list[str]] = []

    def fake_run(command, *, cwd):
        calls.append(command)
        if command[0] == "rsync":
            destination = command[-1]
            if ".c2a_fast.push-" not in destination:
                return
            staging_name = Path(destination.rstrip("/").split(":", 1)[1]).name
            staging = remote_project / "data" / staging_name
            if transfer_error is not None:
                (staging / "manifest.json").write_text("partial", encoding="utf-8")
                raise transfer_error
            source = Path(command[-2].rstrip("/"))
            for path in source.iterdir():
                shutil.copy2(path, staging / path.name)
            return
        if command[0] != "ssh":
            raise AssertionError(f"unexpected command: {command}")
        remote_command = command[-1]
        if "PYTHONPATH=src" not in remote_command:
            return
        tokens = shlex.split(remote_command.split("&&", 1)[1])
        action, suffix = tokens[-2:]
        if action == "cleanup" and cleanup_error is not None:
            raise cleanup_error
        c2a_remote._remote_fast_pack_push(remote_project, action, suffix)

    monkeypatch.setattr("robot_quant.c2a_remote._run", fake_run)
    return calls


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


def test_fetch_stages_complete_allowlist_and_returns_sha_manifest(tmp_path, monkeypatch) -> None:
    root = tmp_path / "project"
    remote = tmp_path / "remote"
    remote_results = remote / "data" / "c2a_results"
    remote_report = remote / "reports" / "c2a_2026_report.md"
    expected_data = {
        "baseline_equity.csv",
        "baseline_events.csv",
        "baseline_trades.csv",
        "data_audit.json",
        "latest_signal.json",
        "latest_state.json",
        "latest_training_grid.csv",
        "walk_forward_oos_trades.csv",
        "walk_forward_selections.csv",
    }
    remote_results.mkdir(parents=True)
    remote_report.parent.mkdir(parents=True)
    remote_report.write_text("current report\n", encoding="utf-8")
    for name in expected_data:
        (remote_results / name).write_text(f"current {name}\n", encoding="utf-8")
    calls = []

    def fake_run(command, *, cwd):
        calls.append((command, cwd))
        destination = Path(command[-1])
        destination.mkdir(parents=True, exist_ok=True)
        if command[-2].endswith("/reports/c2a_2026_report.md"):
            shutil.copy2(remote_report, destination / remote_report.name)
            return
        for name in expected_data:
            shutil.copy2(remote_results / name, destination / name)

    monkeypatch.setattr("robot_quant.c2a_remote._run", fake_run)

    run_name = f".c2a-research-run-{'4' * 32}"
    manifest = fetch_remote_results(root, remote_output_root=run_name)

    expected_paths = {
        "reports/c2a_2026_report.md",
        *(f"data/c2a_results/{name}" for name in expected_data),
    }
    assert set(manifest) == expected_paths
    for relative_path, expected_hash in manifest.items():
        target = root / relative_path
        assert target.is_file() and not target.is_symlink()
        assert hashlib.sha256(target.read_bytes()).hexdigest() == expected_hash
    result_sync = calls[-1][0]
    assert result_sync[:2] == ["rsync", "-az"]
    assert "--include" in result_sync
    assert "short_window_analysis.json" not in result_sync
    assert result_sync[-4:-2] == ["--exclude", "*"]
    assert "parameter_paths" not in " ".join(result_sync)
    assert all(f"/home/aiuser/work/robot-quant/{run_name}/" in command[-2] for command, _ in calls)


def test_fetch_missing_current_result_preserves_all_local_results(tmp_path, monkeypatch) -> None:
    root = tmp_path / "project"
    local_paths = [root / relative_path for relative_path in c2a_remote.REMOTE_RESULT_ALLOWLIST]
    for path in local_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"old {path.name}\n", encoding="utf-8")
    before = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in local_paths
    }

    def incomplete_run(command, *, cwd):
        destination = Path(command[-1])
        destination.mkdir(parents=True, exist_ok=True)
        if command[-2].endswith("/reports/c2a_2026_report.md"):
            (destination / "c2a_2026_report.md").write_text("new report\n", encoding="utf-8")
            return
        for name in c2a_remote.REMOTE_DATA_RESULT_ALLOWLIST[:-1]:
            (destination / name).write_text(f"new {name}\n", encoding="utf-8")

    monkeypatch.setattr("robot_quant.c2a_remote._run", incomplete_run)

    with pytest.raises(RuntimeError, match="缺少本轮固定产物"):
        fetch_remote_results(root)

    after = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in local_paths
    }
    assert after == before


def test_fetch_rejects_symlinked_current_result_before_landing(tmp_path, monkeypatch) -> None:
    root = tmp_path / "project"
    outside = tmp_path / "outside.csv"
    outside.write_text("not a regular staged artifact\n", encoding="utf-8")

    def symlinked_run(command, *, cwd):
        destination = Path(command[-1])
        destination.mkdir(parents=True, exist_ok=True)
        if command[-2].endswith("/reports/c2a_2026_report.md"):
            (destination / "c2a_2026_report.md").write_text("report\n", encoding="utf-8")
            return
        for name in c2a_remote.REMOTE_DATA_RESULT_ALLOWLIST:
            path = destination / name
            if name == "baseline_equity.csv":
                path.symlink_to(outside)
            else:
                path.write_text(f"current {name}\n", encoding="utf-8")

    monkeypatch.setattr("robot_quant.c2a_remote._run", symlinked_run)

    with pytest.raises(RuntimeError, match="baseline_equity.csv"):
        fetch_remote_results(root)

    assert not (root / "reports" / "c2a_2026_report.md").exists()
    assert not (root / "data" / "c2a_results" / "latest_state.json").exists()


def test_fetch_audit_results_does_not_require_backtest_artifacts(tmp_path, monkeypatch) -> None:
    root = tmp_path / "project"

    def audit_run(command, *, cwd):
        destination = Path(command[-1])
        destination.mkdir(parents=True, exist_ok=True)
        if command[-2].endswith("/reports/c2a_2026_report.md"):
            (destination / "c2a_2026_report.md").write_text("audit report\n", encoding="utf-8")
        else:
            assert "data_audit.json" in command
            assert "baseline_equity.csv" not in command
            (destination / "data_audit.json").write_text('{"status":"STRICT"}\n')

    monkeypatch.setattr("robot_quant.c2a_remote._run", audit_run)

    manifest = fetch_remote_audit_results(root)

    assert set(manifest) == {
        "reports/c2a_2026_report.md",
        "data/c2a_results/data_audit.json",
    }


def test_fast_pack_round_trip_only_uses_ignored_compact_directory(tmp_path, monkeypatch) -> None:
    root = _project(tmp_path / "project")
    pack = root / "data" / "c2a_fast"
    remote_pack = tmp_path / "remote-pack"
    _write_fast_pack(pack, "2026-08-18")
    _write_fast_pack(remote_pack, "2026-08-19")
    calls = []

    def fake_run(command, *, cwd):
        calls.append((command, cwd))
        if command[:3] == [
            "rsync",
            "-az",
            "bigquant-aistudio:/home/aiuser/work/robot-quant/data/c2a_fast/",
        ]:
            destination = Path(command[-1])
            for source in remote_pack.iterdir():
                shutil.copy2(source, destination / source.name)

    monkeypatch.setattr("robot_quant.c2a_remote._run", fake_run)
    monkeypatch.setattr("robot_quant.c2a_remote.secrets.token_hex", lambda _: "f" * 32)

    export_remote_fast_pack(root, sync_code=False)
    fetch_remote_fast_pack(root)
    push_remote_fast_pack(root)

    assert "robot_quant.c2a_fast export-pack" in calls[0][0][-1]
    assert calls[1][0][:3] == [
        "rsync",
        "-az",
        "bigquant-aistudio:/home/aiuser/work/robot-quant/data/c2a_fast/",
    ]
    assert Path(calls[1][0][-1]).parent == pack.parent.resolve()
    assert Path(calls[1][0][-1]).name.startswith(".c2a_fast.fetch-")
    assert (
        json.loads((pack / "manifest.json").read_text(encoding="utf-8"))["last_processed_date"]
        == "2026-08-19"
    )
    upload = next(
        command
        for command, _ in calls
        if command[:3]
        == [
            "rsync",
            "-az",
            f"{pack.resolve()}/",
        ]
    )
    assert upload == [
        "rsync",
        "-az",
        f"{pack.resolve()}/",
        f"bigquant-aistudio:/home/aiuser/work/robot-quant/data/.c2a_fast.push-{'f' * 32}/",
    ]
    helper_calls = [
        command[-1]
        for command, _ in calls
        if command[0] == "ssh" and "_remote_fast_pack_push" in command[-1]
    ]
    assert [shlex.split(item)[-2] for item in helper_calls] == [
        "prepare",
        "promote",
        "cleanup",
    ]


def test_fetch_rejects_older_remote_pack_without_changing_local_pack(tmp_path, monkeypatch) -> None:
    root = _project(tmp_path / "project")
    local_pack = root / "data" / "c2a_fast"
    remote_pack = tmp_path / "remote-pack"
    _write_fast_pack(local_pack, "2026-08-19")
    _write_fast_pack(remote_pack, "2026-08-18")
    before = _pack_file_hashes(local_pack)

    def fake_run(command, *, cwd):
        assert command[:2] == ["rsync", "-az"]
        destination = Path(command[-1])
        destination.mkdir(parents=True, exist_ok=True)
        for source in remote_pack.iterdir():
            shutil.copy2(source, destination / source.name)

    monkeypatch.setattr("robot_quant.c2a_remote._run", fake_run)

    with pytest.raises(RuntimeError, match="早于本地"):
        fetch_remote_fast_pack(root)

    assert _pack_file_hashes(local_pack) == before


def test_fetch_rejects_incomplete_remote_hash_manifest_without_changing_local_pack(
    tmp_path, monkeypatch
) -> None:
    root = _project(tmp_path / "project")
    local_pack = root / "data" / "c2a_fast"
    remote_pack = tmp_path / "remote-pack"
    _write_fast_pack(local_pack, "2026-08-18")
    _write_fast_pack(remote_pack, "2026-08-19")
    manifest_path = remote_pack / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["file_sha256"]["universe.csv.gz"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    before = _pack_file_hashes(local_pack)

    def fake_run(command, *, cwd):
        assert command[:2] == ["rsync", "-az"]
        destination = Path(command[-1])
        destination.mkdir(parents=True, exist_ok=True)
        for source in remote_pack.iterdir():
            shutil.copy2(source, destination / source.name)

    monkeypatch.setattr("robot_quant.c2a_remote._run", fake_run)

    with pytest.raises(RuntimeError, match="哈希清单"):
        fetch_remote_fast_pack(root)

    assert _pack_file_hashes(local_pack) == before


def test_fetch_rejects_remote_manifest_date_mismatch_without_changing_local_pack(
    tmp_path, monkeypatch
) -> None:
    root = _project(tmp_path / "project")
    local_pack = root / "data" / "c2a_fast"
    remote_pack = tmp_path / "remote-pack"
    _write_fast_pack(local_pack, "2026-08-18")
    _write_fast_pack(remote_pack, "2026-08-19")
    manifest_path = remote_pack / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["last_processed_date"] = "2026-08-17"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    before = _pack_file_hashes(local_pack)

    def fake_run(command, *, cwd):
        destination = Path(command[-1])
        destination.mkdir(parents=True, exist_ok=True)
        for source in remote_pack.iterdir():
            shutil.copy2(source, destination / source.name)

    monkeypatch.setattr("robot_quant.c2a_remote._run", fake_run)

    with pytest.raises(RuntimeError, match="日期不一致"):
        fetch_remote_fast_pack(root)

    assert _pack_file_hashes(local_pack) == before


def test_fetch_transfer_failure_keeps_local_pack_unchanged(tmp_path, monkeypatch) -> None:
    root = _project(tmp_path / "project")
    local_pack = root / "data" / "c2a_fast"
    _write_fast_pack(local_pack, "2026-08-19")
    before = _pack_file_hashes(local_pack)

    def interrupted_run(command, *, cwd):
        staging = Path(command[-1])
        (staging / "manifest.json").write_text("partial", encoding="utf-8")
        raise RuntimeError("模拟 rsync 传输中断")

    monkeypatch.setattr("robot_quant.c2a_remote._run", interrupted_run)

    with pytest.raises(RuntimeError, match="传输中断"):
        fetch_remote_fast_pack(root)

    assert _pack_file_hashes(local_pack) == before


def test_fetch_tampered_remote_file_keeps_local_pack_unchanged(tmp_path, monkeypatch) -> None:
    root = _project(tmp_path / "project")
    local_pack = root / "data" / "c2a_fast"
    remote_pack = tmp_path / "remote-pack"
    _write_fast_pack(local_pack, "2026-08-19")
    _write_fast_pack(remote_pack, "2026-08-20")
    with (remote_pack / "rolling_state.npz").open("ab") as handle:
        handle.write(b"tampered")
    before = _pack_file_hashes(local_pack)

    def fake_run(command, *, cwd):
        destination = Path(command[-1])
        for source in remote_pack.iterdir():
            shutil.copy2(source, destination / source.name)

    monkeypatch.setattr("robot_quant.c2a_remote._run", fake_run)

    with pytest.raises(RuntimeError, match="文件损坏"):
        fetch_remote_fast_pack(root)

    assert _pack_file_hashes(local_pack) == before


def test_fetch_malformed_remote_manifest_fails_closed_as_runtime_error(
    tmp_path, monkeypatch
) -> None:
    root = _project(tmp_path / "project")
    local_pack = root / "data" / "c2a_fast"
    remote_pack = tmp_path / "remote-pack"
    _write_fast_pack(local_pack, "2026-08-19")
    _write_fast_pack(remote_pack, "2026-08-20")
    (remote_pack / "manifest.json").write_text("not-json", encoding="utf-8")
    before = _pack_file_hashes(local_pack)

    def fake_run(command, *, cwd):
        destination = Path(command[-1])
        for source in remote_pack.iterdir():
            shutil.copy2(source, destination / source.name)

    monkeypatch.setattr("robot_quant.c2a_remote._run", fake_run)

    with pytest.raises(RuntimeError, match="远端紧凑基线校验失败"):
        fetch_remote_fast_pack(root)

    assert _pack_file_hashes(local_pack) == before


def test_push_fast_pack_fails_closed_when_pack_is_incomplete(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="紧凑基线缺少文件"):
        push_remote_fast_pack(tmp_path)


def test_push_fast_pack_validates_integrity_before_remote_write(tmp_path, monkeypatch) -> None:
    pack = tmp_path / "data" / "c2a_fast"
    _write_fast_pack(pack, "2026-08-19")
    with (pack / "rolling_state.npz").open("ab") as handle:
        handle.write(b"tampered")
    calls = []
    monkeypatch.setattr(
        "robot_quant.c2a_remote._run", lambda command, *, cwd: calls.append((command, cwd))
    )

    with pytest.raises(RuntimeError, match="文件损坏"):
        push_remote_fast_pack(tmp_path)

    assert calls == []


def test_push_rejects_older_local_pack_without_changing_remote(tmp_path, monkeypatch) -> None:
    root = _project(tmp_path / "project")
    local_pack = root / "data" / "c2a_fast"
    remote_project = tmp_path / "remote-project"
    remote_pack = remote_project / "data" / "c2a_fast"
    _write_fast_pack(local_pack, "2026-08-18")
    _write_fast_pack(remote_pack, "2026-08-19")
    before = _pack_file_hashes(remote_pack)
    monkeypatch.setattr("robot_quant.c2a_remote.secrets.token_hex", lambda _: "a" * 32)
    _install_fake_fast_pack_remote(monkeypatch, remote_project)

    with pytest.raises(RuntimeError, match="早于远端"):
        push_remote_fast_pack(root)

    assert _pack_file_hashes(remote_pack) == before
    assert list((remote_project / "data").glob(".c2a_fast.push-*")) == []


def test_push_transfer_failure_does_not_change_remote_authority(tmp_path, monkeypatch) -> None:
    root = _project(tmp_path / "project")
    local_pack = root / "data" / "c2a_fast"
    remote_project = tmp_path / "remote-project"
    remote_pack = remote_project / "data" / "c2a_fast"
    _write_fast_pack(local_pack, "2026-08-19")
    _write_fast_pack(remote_pack, "2026-08-18")
    before = _pack_file_hashes(remote_pack)
    monkeypatch.setattr("robot_quant.c2a_remote.secrets.token_hex", lambda _: "b" * 32)
    _install_fake_fast_pack_remote(
        monkeypatch,
        remote_project,
        transfer_error=RuntimeError("模拟上传中断"),
    )

    with pytest.raises(RuntimeError, match="模拟上传中断"):
        push_remote_fast_pack(root)

    assert _pack_file_hashes(remote_pack) == before
    assert list((remote_project / "data").glob(".c2a_fast.push-*")) == []


def test_push_promotion_failure_rolls_back_remote_authority(tmp_path, monkeypatch) -> None:
    root = _project(tmp_path / "project")
    local_pack = root / "data" / "c2a_fast"
    remote_project = tmp_path / "remote-project"
    remote_pack = remote_project / "data" / "c2a_fast"
    _write_fast_pack(local_pack, "2026-08-19")
    _write_fast_pack(remote_pack, "2026-08-18")
    before = _pack_file_hashes(remote_pack)
    suffix = "c" * 32
    monkeypatch.setattr("robot_quant.c2a_remote.secrets.token_hex", lambda _: suffix)
    _install_fake_fast_pack_remote(monkeypatch, remote_project)
    original_replace = Path.replace

    def fail_staging_promotion(path: Path, target: Path):
        if path.name == f".c2a_fast.push-{suffix}" and target.name == "c2a_fast":
            raise OSError("模拟目录提升失败")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_staging_promotion)

    with pytest.raises(OSError, match="模拟目录提升失败"):
        push_remote_fast_pack(root)

    assert _pack_file_hashes(remote_pack) == before
    assert list((remote_project / "data").glob(".c2a_fast.*-*")) == []


def test_push_cleanup_failure_does_not_mask_transfer_failure(tmp_path, monkeypatch) -> None:
    root = _project(tmp_path / "project")
    pack = root / "data" / "c2a_fast"
    remote_project = tmp_path / "remote-project"
    _write_fast_pack(pack, "2026-08-19")
    monkeypatch.setattr("robot_quant.c2a_remote.secrets.token_hex", lambda _: "d" * 32)
    _install_fake_fast_pack_remote(
        monkeypatch,
        remote_project,
        transfer_error=RuntimeError("主上传失败"),
        cleanup_error=RuntimeError("次要清理失败"),
    )

    with pytest.raises(RuntimeError, match="主上传失败"):
        push_remote_fast_pack(root)


def test_push_promotes_complete_newer_pack_and_removes_transport_paths(
    tmp_path, monkeypatch
) -> None:
    root = _project(tmp_path / "project")
    local_pack = root / "data" / "c2a_fast"
    remote_project = tmp_path / "remote-project"
    remote_pack = remote_project / "data" / "c2a_fast"
    _write_fast_pack(local_pack, "2026-08-19")
    _write_fast_pack(remote_pack, "2026-08-18")
    monkeypatch.setattr("robot_quant.c2a_remote.secrets.token_hex", lambda _: "e" * 32)
    calls = _install_fake_fast_pack_remote(monkeypatch, remote_project)

    push_remote_fast_pack(root)

    manifest = json.loads((remote_pack / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["last_processed_date"] == "2026-08-19"
    assert list((remote_project / "data").glob(".c2a_fast.*-*")) == []
    upload = next(command for command in calls if ".c2a_fast.push-" in command[-1])
    assert upload[0:2] == ["rsync", "-az"]
    assert upload[-1].endswith(f"/data/.c2a_fast.push-{'e' * 32}/")


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
    suffix = "9" * 32
    monkeypatch.setattr("robot_quant.c2a_remote.secrets.token_hex", lambda _: suffix)
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
        lambda *args, **kwargs: (
            calls.append(("fetch", args, kwargs)) or {"reports/c2a_2026_report.md": "a" * 64}
        ),
    )
    monkeypatch.setattr(
        "robot_quant.c2a_remote._run_remote_research_run",
        lambda *args, **kwargs: calls.append(("research-run", args, kwargs)),
    )

    manifest = run_remote_pipeline(
        tmp_path,
        "2026-07-01",
        "2026-08-12",
        optimize=False,
    )

    assert calls[1][0] == "update"
    assert calls[1][1][1:3] == ("2026-01-01", "2026-08-12")
    assert calls[2][0] == "research-run"
    assert calls[2][1][-3:] == ("prepare", suffix, True)
    assert calls[3][0] == "backtest"
    assert calls[3][1][1:3] == ("2026-07-01", "2026-08-12")
    assert calls[3][2]["output_root"] == f".c2a-research-run-{suffix}"
    assert calls[4][0] == "fetch"
    assert calls[4][2]["remote_output_root"] == f".c2a-research-run-{suffix}"
    assert calls[5][0] == "research-run"
    assert calls[5][1][-3:] == ("promote-carry", suffix, False)
    assert calls[6][0] == "research-run"
    assert calls[6][1][-3:] == ("cleanup", suffix, False)
    assert manifest == {"reports/c2a_2026_report.md": "a" * 64}


def test_pipeline_failure_cleans_isolated_run_without_masking_primary_error(
    tmp_path, monkeypatch
) -> None:
    calls = []
    suffix = "8" * 32
    monkeypatch.setattr("robot_quant.c2a_remote.secrets.token_hex", lambda _: suffix)
    monkeypatch.setattr("robot_quant.c2a_remote.sync_c2a_code", lambda *args, **kwargs: None)
    monkeypatch.setattr("robot_quant.c2a_remote.update_remote_data", lambda *args, **kwargs: None)

    def fail_backtest(*args, **kwargs):
        raise RuntimeError("本轮隔离回测失败")

    def research_run(*args, **kwargs):
        action = args[-3]
        calls.append(action)
        if action == "cleanup":
            raise RuntimeError("次要清理失败")

    monkeypatch.setattr("robot_quant.c2a_remote.run_remote_backtest", fail_backtest)
    monkeypatch.setattr("robot_quant.c2a_remote._run_remote_research_run", research_run)

    with pytest.raises(RuntimeError, match="本轮隔离回测失败"):
        run_remote_pipeline(tmp_path, "2026-07-01", "2026-08-12", optimize=True)

    assert calls == ["prepare", "cleanup"]


def test_pipeline_missing_current_latest_state_preserves_old_carry(tmp_path, monkeypatch) -> None:
    local_project = tmp_path / "local-project"
    remote_project = tmp_path / "remote-project"
    carry = remote_project / c2a_remote.RESEARCH_CARRY_ROOT
    _write_carried_snapshot(carry, "old-authority")
    before = {
        name: hashlib.sha256((carry / name).read_bytes()).hexdigest()
        for name in c2a_remote.CARRIED_FORWARD_RESULT_FILES
    }
    suffix = "a" * 32
    actions = []
    backtest_arguments = {}
    monkeypatch.setattr("robot_quant.c2a_remote.secrets.token_hex", lambda _: suffix)
    monkeypatch.setattr("robot_quant.c2a_remote.sync_c2a_code", lambda *args, **kwargs: None)
    monkeypatch.setattr("robot_quant.c2a_remote.update_remote_data", lambda *args, **kwargs: None)

    def research_run(*args, **kwargs):
        action, run_suffix, carry_forward = args[-3:]
        actions.append(action)
        c2a_remote._remote_research_run(
            remote_project,
            action,
            run_suffix,
            carry_forward,
        )

    def backtest(*args, **kwargs):
        backtest_arguments.update(kwargs)
        run_root = remote_project / kwargs["output_root"]
        for relative_path in c2a_remote.REMOTE_RESULT_ALLOWLIST:
            if relative_path == "data/c2a_results/latest_state.json":
                continue
            path = run_root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"current {relative_path}\n", encoding="utf-8")

    def fetch(*args, **kwargs):
        run_root = remote_project / kwargs["remote_output_root"]
        if (run_root / "data" / "c2a_results" / "latest_state.json").is_file():
            return {relative_path: "f" * 64 for relative_path in c2a_remote.REMOTE_RESULT_ALLOWLIST}
        raise RuntimeError("C2-A 远程缺少本轮固定产物: data/c2a_results/latest_state.json")

    monkeypatch.setattr("robot_quant.c2a_remote._run_remote_research_run", research_run)
    monkeypatch.setattr("robot_quant.c2a_remote.run_remote_backtest", backtest)
    monkeypatch.setattr("robot_quant.c2a_remote.fetch_remote_results", fetch)

    with pytest.raises(RuntimeError, match="latest_state.json"):
        run_remote_pipeline(
            local_project,
            "2026-07-01",
            "2026-08-19",
            optimize=False,
        )

    after = {
        name: hashlib.sha256((carry / name).read_bytes()).hexdigest()
        for name in c2a_remote.CARRIED_FORWARD_RESULT_FILES
    }
    assert after == before
    assert actions == ["prepare", "cleanup"]
    assert backtest_arguments["prior_results_root"] == (
        f".c2a-research-run-{suffix}/{c2a_remote.RESEARCH_CARRY_INPUT_DIR}"
    )
    assert not (remote_project / f".c2a-research-run-{suffix}").exists()


def test_research_run_carries_only_complete_prior_optimization(tmp_path) -> None:
    canonical = tmp_path / "data" / "c2a_results"
    canonical.mkdir(parents=True)
    prior_state = {"walk_forward": {"status": "COMPLETED", "optimization_as_of": "2026-08-14"}}
    (canonical / "latest_state.json").write_text(json.dumps(prior_state), encoding="utf-8")
    for name in c2a_remote.CARRIED_FORWARD_RESULT_FILES[1:]:
        (canonical / name).write_text(f"prior {name}\n", encoding="utf-8")
    (canonical / "baseline_equity.csv").write_text("stale canonical baseline\n", encoding="utf-8")
    suffix = "7" * 32

    c2a_remote._remote_research_run(tmp_path, "prepare", suffix, True)

    run_root = tmp_path / f".c2a-research-run-{suffix}"
    carry_input = run_root / c2a_remote.RESEARCH_CARRY_INPUT_DIR
    copied = {path.name for path in carry_input.iterdir()}
    assert copied == set(c2a_remote.CARRIED_FORWARD_RESULT_FILES)
    assert not (run_root / "data" / "c2a_results").exists()
    c2a_remote._remote_research_run(tmp_path, "cleanup", suffix, False)
    assert not run_root.exists()


def test_research_run_rejects_incomplete_carried_forward_set_and_cleans_stage(
    tmp_path,
) -> None:
    canonical = tmp_path / "data" / "c2a_results"
    canonical.mkdir(parents=True)
    (canonical / "latest_state.json").write_text(
        json.dumps({"walk_forward": {"status": "CARRIED_FORWARD"}}),
        encoding="utf-8",
    )
    suffix = "6" * 32

    with pytest.raises(RuntimeError, match="carried-forward 结果不完整"):
        c2a_remote._remote_research_run(tmp_path, "prepare", suffix, True)

    assert not (tmp_path / f".c2a-research-run-{suffix}").exists()


def test_optimized_research_run_does_not_prefill_stale_canonical_results(tmp_path) -> None:
    canonical = tmp_path / "data" / "c2a_results"
    canonical.mkdir(parents=True)
    for relative_path in c2a_remote.REMOTE_RESULT_ALLOWLIST:
        if not relative_path.startswith("data/c2a_results/"):
            continue
        path = tmp_path / relative_path
        path.write_text(f"stale {path.name}\n", encoding="utf-8")
    suffix = "5" * 32

    c2a_remote._remote_research_run(tmp_path, "prepare", suffix, False)

    run_root = tmp_path / f".c2a-research-run-{suffix}"
    assert list(run_root.iterdir()) == []
    c2a_remote._remote_research_run(tmp_path, "cleanup", suffix, False)


def test_optimized_run_promotes_carry_used_by_next_nonoptimized_run(tmp_path) -> None:
    optimized_suffix = "4" * 32
    c2a_remote._remote_research_run(tmp_path, "prepare", optimized_suffix, False)
    optimized_root = tmp_path / f".c2a-research-run-{optimized_suffix}"
    optimized_results = optimized_root / "data" / "c2a_results"
    _write_carried_snapshot(optimized_results, "new-friday")

    c2a_remote._remote_research_run(
        tmp_path,
        "promote-carry",
        optimized_suffix,
        False,
    )
    c2a_remote._remote_research_run(tmp_path, "cleanup", optimized_suffix, False)

    canonical = tmp_path / "data" / "c2a_results"
    _write_carried_snapshot(canonical, "old-canonical")
    next_suffix = "3" * 32
    c2a_remote._remote_research_run(tmp_path, "prepare", next_suffix, True)

    carry = tmp_path / c2a_remote.RESEARCH_CARRY_ROOT
    next_run_root = tmp_path / f".c2a-research-run-{next_suffix}"
    next_results = next_run_root / c2a_remote.RESEARCH_CARRY_INPUT_DIR
    assert {path.name for path in carry.iterdir()} == set(c2a_remote.CARRIED_FORWARD_RESULT_FILES)
    assert (
        json.loads((carry / "latest_state.json").read_text(encoding="utf-8"))["snapshot_marker"]
        == "new-friday"
    )
    assert (
        json.loads((next_results / "latest_state.json").read_text(encoding="utf-8"))[
            "snapshot_marker"
        ]
        == "new-friday"
    )
    assert "new-friday" in (next_results / "walk_forward_selections.csv").read_text(
        encoding="utf-8"
    )
    assert "old-canonical" not in (next_results / "walk_forward_selections.csv").read_text(
        encoding="utf-8"
    )
    assert not (next_run_root / "data" / "c2a_results").exists()
    c2a_remote._remote_research_run(tmp_path, "cleanup", next_suffix, False)


def test_incomplete_run_does_not_replace_existing_carry_snapshot(tmp_path) -> None:
    carry = tmp_path / c2a_remote.RESEARCH_CARRY_ROOT
    _write_carried_snapshot(carry, "old-authority")
    before = {
        name: hashlib.sha256((carry / name).read_bytes()).hexdigest()
        for name in c2a_remote.CARRIED_FORWARD_RESULT_FILES
    }
    suffix = "2" * 32
    c2a_remote._remote_research_run(tmp_path, "prepare", suffix, False)
    run_results = tmp_path / f".c2a-research-run-{suffix}" / "data" / "c2a_results"
    run_results.mkdir(parents=True)
    (run_results / "latest_state.json").write_text(
        json.dumps({"walk_forward": {"status": "COMPLETED"}}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="carried-forward 结果不完整"):
        c2a_remote._remote_research_run(tmp_path, "promote-carry", suffix, False)

    after = {
        name: hashlib.sha256((carry / name).read_bytes()).hexdigest()
        for name in c2a_remote.CARRIED_FORWARD_RESULT_FILES
    }
    assert after == before
    assert list((tmp_path / "data").glob(".c2a_research_carry.*-*")) == []
    c2a_remote._remote_research_run(tmp_path, "cleanup", suffix, False)


def test_carry_promotion_failure_rolls_back_existing_snapshot(tmp_path, monkeypatch) -> None:
    carry = tmp_path / c2a_remote.RESEARCH_CARRY_ROOT
    _write_carried_snapshot(carry, "old-authority")
    before = {
        name: hashlib.sha256((carry / name).read_bytes()).hexdigest()
        for name in c2a_remote.CARRIED_FORWARD_RESULT_FILES
    }
    suffix = "1" * 32
    c2a_remote._remote_research_run(tmp_path, "prepare", suffix, False)
    run_results = tmp_path / f".c2a-research-run-{suffix}" / "data" / "c2a_results"
    _write_carried_snapshot(run_results, "new-but-failed")
    original_replace = Path.replace

    def fail_staging_promotion(path: Path, target: Path) -> Path:
        if (
            path.name == f"{c2a_remote.RESEARCH_CARRY_STAGING_PREFIX}{suffix}"
            and Path(target) == carry
        ):
            raise OSError("模拟 carry 目录提升失败")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_staging_promotion)

    with pytest.raises(OSError, match="模拟 carry 目录提升失败"):
        c2a_remote._remote_research_run(tmp_path, "promote-carry", suffix, False)

    after = {
        name: hashlib.sha256((carry / name).read_bytes()).hexdigest()
        for name in c2a_remote.CARRIED_FORWARD_RESULT_FILES
    }
    assert after == before
    assert list((tmp_path / "data").glob(".c2a_research_carry.*-*")) == []
    c2a_remote._remote_research_run(tmp_path, "cleanup", suffix, False)


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
    with pytest.raises(ValueError, match="研究隔离目录"):
        run_remote_backtest(".", output_root="../canonical")


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
