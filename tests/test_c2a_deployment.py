import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _cloud_repository(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, str]]:
    remote = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    cloud = tmp_path / "cloud"
    fake_bin = tmp_path / "bin"
    subprocess.run(
        ["git", "init", "--bare", "--initial-branch=main", str(remote)],
        check=True,
        capture_output=True,
        text=True,
    )
    seed.mkdir()
    _git("init", "--initial-branch=main", cwd=seed)
    _git("config", "user.name", "test", cwd=seed)
    _git("config", "user.email", "test@example.com", cwd=seed)
    (seed / "README.md").write_text("clean\n", encoding="utf-8")
    _git("add", "README.md", cwd=seed)
    _git("commit", "-m", "seed", cwd=seed)
    _git("remote", "add", "origin", str(remote), cwd=seed)
    _git("push", "-u", "origin", "main", cwd=seed)
    subprocess.run(
        ["git", "clone", str(remote), str(cloud)],
        check=True,
        capture_output=True,
        text=True,
    )
    (cloud / "scripts").mkdir()
    shutil.copy2(ROOT / "scripts/record_c2a_failure.py", cloud / "scripts")
    fake_bin.mkdir()
    fake_flock = fake_bin / "flock"
    fake_flock.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_flock.chmod(0o755)
    fake_timeout = fake_bin / "timeout"
    fake_timeout.write_text('#!/usr/bin/env bash\nshift 3\nexec "$@"\n', encoding="utf-8")
    fake_timeout.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "ROBOT_QUANT_PROJECT_ROOT": str(cloud),
        "ROBOT_QUANT_PYTHON": sys.executable,
    }
    return remote, seed, cloud, environment


def test_cloud_runner_uses_result_allowlist_instead_of_broad_git_add() -> None:
    script = (ROOT / "scripts/run_c2a_cloud_phase.sh").read_text(encoding="utf-8")

    assert 'git add -- "$path"' in script
    assert "git add data/" not in script
    assert "parameter_paths" not in script
    assert "PAPER_ONLY" not in script  # 模型边界由 Python 产物统一提供
    assert "record_c2a_failure.py" in script
    assert "flock -w 90 9" in script
    assert "publish_failure_outbox" in script
    assert "trap handle_termination TERM INT" in script
    assert script.index("trap handle_termination TERM INT") < script.index("flock -w 90 9")
    assert "--service-mode" in script
    assert "for attempt in 1 2 3" in script
    assert "timeout --signal=TERM --kill-after=2s 8s git" in script
    assert "拒绝无界 Git 网络请求" in script


def test_existing_daily_workflow_cannot_stage_c2a_cache_or_arbitrary_data() -> None:
    workflow = (ROOT / ".github/workflows/daily.yml").read_text(encoding="utf-8")

    assert "git add data/ reports/" not in workflow
    assert "data/latest_state.json" in workflow
    assert "data/robot_chain_latest_state.json" in workflow
    assert "data/c2a" not in workflow
    assert "actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803" in workflow
    assert "actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1" in workflow
    assert "git rebase origin/main" in workflow
    assert "persist-credentials: false" in workflow
    assert "password=$GITHUB_TOKEN" in workflow


def test_sensitive_or_regenerable_local_state_is_ignored() -> None:
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert ".idea/" in ignore
    assert ".env" in ignore
    assert "data/c2a/" in ignore
    assert "data/c2a_fast/" in ignore
    assert "data/c2a_results/parameter_paths/" in ignore
    assert ".c2a-cloud.lock" in ignore
    assert "*.pem" in ignore


def test_systemd_timers_use_shanghai_wall_clock_without_random_delay() -> None:
    expected = {
        "prepare": "08:45:00",
        "scan": "10:02:30",
        "review": "16:30:00",
        "research": "16:35:00",
    }
    for phase, clock in expected.items():
        timer = (ROOT / f"deploy/systemd/robot-quant-c2a-{phase}.timer").read_text(encoding="utf-8")
        assert f"OnCalendar=Mon..Fri *-*-* {clock} Asia/Shanghai" in timer
        assert "AccuracySec=1s" in timer
        assert "RandomizedDelaySec=0" in timer
        assert f"Unit=robot-quant-c2a@{phase}.service" in timer

    service = (ROOT / "deploy/systemd/robot-quant-c2a@.service").read_text(encoding="utf-8")
    assert "Restart=on-failure" in service
    assert "StartLimitBurst=3" in service
    assert "InaccessiblePaths=-/usr/local/etc/xray" in service


def test_dependency_free_failure_recorder_clears_stale_scan(tmp_path) -> None:
    script = ROOT / "scripts/record_c2a_failure.py"
    latest = tmp_path / "data/c2a_results/fast_latest.json"
    latest.parent.mkdir(parents=True)
    latest.write_text('{"as_of":"2026-08-18","entries":[{"ticker":"old"}]}')

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--project-root",
            str(tmp_path),
            "--phase",
            "scan",
            "--date",
            "2026-08-19",
            "--reason",
            "云端虚拟环境解释器不可用",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(latest.read_text(encoding="utf-8"))
    assert payload["status"] == "DATA_NOT_READY"
    assert payload["entries"] == []
    assert payload["as_of"] == "2026-08-19"
    assert payload["retryable"] is True
    assert payload["scheduled_at"] == "2026-08-19T10:02:30+08:00"
    assert len(payload["payload_sha256"]) == 64


def test_dirty_cloud_tree_publishes_failure_from_clean_outbox(tmp_path) -> None:
    remote, _, cloud, environment = _cloud_repository(tmp_path)
    inspect = tmp_path / "inspect"
    (cloud / "README.md").write_text("dirty user change\n", encoding="utf-8")

    completed = subprocess.run(
        [str(ROOT / "scripts/run_c2a_cloud_phase.sh"), "scan"],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    subprocess.run(
        ["git", "clone", str(remote), str(inspect)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(
        (inspect / "data/c2a_results/fast_latest.json").read_text(encoding="utf-8")
    )
    assert payload["status"] == "DATA_NOT_READY"
    assert payload["entries"] == []
    assert len(payload["source_commit"]) == 40
    assert (inspect / "README.md").read_text(encoding="utf-8") == "clean\n"


def test_unpushed_cloud_commit_is_rejected_and_not_pushed(tmp_path) -> None:
    remote, _, cloud, environment = _cloud_repository(tmp_path)
    inspect = tmp_path / "inspect"
    _git("config", "user.name", "cloud", cwd=cloud)
    _git("config", "user.email", "cloud@example.com", cwd=cloud)
    (cloud / "LOCAL_ONLY.txt").write_text("must not be published\n", encoding="utf-8")
    _git("add", "LOCAL_ONLY.txt", cwd=cloud)
    _git("commit", "-m", "local-only commit", cwd=cloud)

    completed = subprocess.run(
        [str(ROOT / "scripts/run_c2a_cloud_phase.sh"), "scan"],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    subprocess.run(
        ["git", "clone", str(remote), str(inspect)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(
        (inspect / "data/c2a_results/fast_latest.json").read_text(encoding="utf-8")
    )
    assert payload["status"] == "DATA_NOT_READY"
    assert "未推送的本地提交" in payload["reason"]
    assert not (inspect / "LOCAL_ONLY.txt").exists()


def test_commit_created_during_phase_is_not_pushed(tmp_path) -> None:
    remote, _, cloud, environment = _cloud_repository(tmp_path)
    inspect = tmp_path / "inspect"
    phase_python = tmp_path / "phase-python"
    phase_python.write_text(
        """#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

if sys.argv[1:4] != ["-m", "robot_quant.c2a_cloud", "scan"]:
    raise SystemExit(1)
Path("LOCAL_DURING.txt").write_text("must not be published\\n", encoding="utf-8")
subprocess.run(["git", "config", "user.name", "phase"], check=True)
subprocess.run(["git", "config", "user.email", "phase@example.com"], check=True)
subprocess.run(["git", "add", "LOCAL_DURING.txt"], check=True)
subprocess.run(["git", "commit", "-m", "commit during phase"], check=True)
result = Path("data/c2a_results/cloud_scan_latest.json")
result.parent.mkdir(parents=True, exist_ok=True)
result.write_text("{}\\n", encoding="utf-8")
""",
        encoding="utf-8",
    )
    phase_python.chmod(0o755)
    environment["ROBOT_QUANT_PYTHON"] = str(phase_python)

    completed = subprocess.run(
        [str(ROOT / "scripts/run_c2a_cloud_phase.sh"), "scan"],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    subprocess.run(
        ["git", "clone", str(remote), str(inspect)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert not (inspect / "LOCAL_DURING.txt").exists()
    assert not (inspect / "data/c2a_results/cloud_scan_latest.json").exists()


def test_diverged_cloud_history_publishes_failure_from_clean_outbox(tmp_path) -> None:
    remote, seed, cloud, environment = _cloud_repository(tmp_path)
    inspect = tmp_path / "inspect"
    _git("config", "user.name", "cloud", cwd=cloud)
    _git("config", "user.email", "cloud@example.com", cwd=cloud)
    (cloud / "README.md").write_text("local divergent change\n", encoding="utf-8")
    _git("add", "README.md", cwd=cloud)
    _git("commit", "-m", "local divergence", cwd=cloud)
    (seed / "README.md").write_text("remote authoritative change\n", encoding="utf-8")
    _git("add", "README.md", cwd=seed)
    _git("commit", "-m", "remote divergence", cwd=seed)
    _git("push", "origin", "main", cwd=seed)

    completed = subprocess.run(
        [str(ROOT / "scripts/run_c2a_cloud_phase.sh"), "scan"],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    subprocess.run(
        ["git", "clone", str(remote), str(inspect)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(
        (inspect / "data/c2a_results/fast_latest.json").read_text(encoding="utf-8")
    )
    assert payload["status"] == "DATA_NOT_READY"
    assert "未推送的本地提交" in payload["reason"]
    assert (inspect / "README.md").read_text(encoding="utf-8") == "remote authoritative change\n"
