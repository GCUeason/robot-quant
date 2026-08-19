import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest


ROOT = Path(__file__).resolve().parents[1]

RESEARCH_RESULT_PATHS = {
    "data/c2a_results/baseline_equity.csv",
    "data/c2a_results/baseline_events.csv",
    "data/c2a_results/baseline_trades.csv",
    "data/c2a_results/data_audit.json",
    "data/c2a_results/latest_signal.json",
    "data/c2a_results/latest_state.json",
    "data/c2a_results/latest_training_grid.csv",
    "data/c2a_results/walk_forward_oos_trades.csv",
    "data/c2a_results/walk_forward_selections.csv",
    "reports/c2a_2026_report.md",
}


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
    (seed / "scripts").mkdir()
    shutil.copy2(ROOT / "scripts/record_c2a_failure.py", seed / "scripts")
    _git("add", "README.md", "scripts/record_c2a_failure.py", cwd=seed)
    _git("commit", "-m", "seed", cwd=seed)
    _git("remote", "add", "origin", str(remote), cwd=seed)
    _git("push", "-u", "origin", "main", cwd=seed)
    subprocess.run(
        ["git", "clone", str(remote), str(cloud)],
        check=True,
        capture_output=True,
        text=True,
    )
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


def _research_phase_python(
    tmp_path: Path,
    *,
    status: str,
    pipeline_status: str,
    valid_hash: bool = True,
    report_as_of: str | None = None,
    boundary_overrides: dict[str, object] | None = None,
    missing_artifact: str | None = None,
    tampered_artifact: str | None = None,
    symlink_artifact: str | None = None,
    symlink_status_report: bool = False,
    omit_mapping_path: str | None = None,
    extra_mapping_path: str | None = None,
) -> Path:
    boundary_overrides = boundary_overrides or {}
    phase_python = tmp_path / "research-phase-python"
    phase_python.write_text(
        f"""#!/usr/bin/env python3
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

if sys.argv[1:2] == ["-"]:
    source = sys.stdin.read()
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    exec(compile(source, "<stdin>", "exec"), {{"__name__": "__main__"}})
    raise SystemExit(0)
if sys.argv[1:4] != ["-m", "robot_quant.c2a_cloud", "research"]:
    raise SystemExit(1)

day = datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()
report_as_of = {report_as_of!r} or day
artifact_content = b"research artifact\\n"
artifact_sha256 = {{}}
for raw_path in {sorted(RESEARCH_RESULT_PATHS)!r}:
    path = Path(raw_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact_sha256[raw_path] = hashlib.sha256(artifact_content).hexdigest()
    if raw_path == {missing_artifact!r}:
        continue
    if raw_path == {symlink_artifact!r}:
        target = Path(".research-artifact-target")
        target.write_bytes(artifact_content)
        path.symlink_to(target.resolve())
    else:
        path.write_bytes(artifact_content)
if {tampered_artifact!r}:
    Path({tampered_artifact!r}).write_text("tampered\\n", encoding="utf-8")
if {omit_mapping_path!r}:
    artifact_sha256.pop({omit_mapping_path!r}, None)
if {extra_mapping_path!r}:
    artifact_sha256[{extra_mapping_path!r}] = "a" * 64

payload = {{
    "phase": "RESEARCH",
    "as_of": report_as_of,
    "status": {status!r},
    "execution_permission": "PAPER_ONLY",
    "real_trade_authorized": False,
    "current_new_entry_allowed": False,
    "research_pipeline": {{
        "status": {pipeline_status!r},
        "artifact_sha256": artifact_sha256,
    }},
    "promotion_gate": "FAIL",
    "retryable": False,
}}
payload.update({boundary_overrides!r})
encoded = json.dumps(
    payload,
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8")
payload["payload_sha256"] = hashlib.sha256(encoded).hexdigest()
if not {valid_hash!r}:
    payload["payload_sha256"] = "0" * 64

json_text = json.dumps(payload, ensure_ascii=False, indent=2) + "\\n"
for path in (
    Path("data/c2a_results/cloud_research_latest.json"),
    Path(f"data/c2a_results/research/{{day}}.json"),
):
    path.parent.mkdir(parents=True, exist_ok=True)
    if {symlink_status_report!r} and path.name == "cloud_research_latest.json":
        target = Path(".research-status-target.json")
        target.write_text(json_text, encoding="utf-8")
        path.symlink_to(target.resolve())
    else:
        path.write_text(json_text, encoding="utf-8")
for path in (
    Path("reports/c2a_research_latest.md"),
    Path(f"reports/c2a/{{day}}-research.md"),
):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# research status\\n", encoding="utf-8")

""",
        encoding="utf-8",
    )
    phase_python.chmod(0o755)
    return phase_python


def _published_paths(remote: Path, tmp_path: Path) -> set[str]:
    inspect = tmp_path / "inspect-results"
    subprocess.run(
        ["git", "clone", str(remote), str(inspect)],
        check=True,
        capture_output=True,
        text=True,
    )
    completed = subprocess.run(
        ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"],
        cwd=inspect,
        check=True,
        capture_output=True,
        text=True,
    )
    return set(completed.stdout.splitlines())


def _published_research_payload(tmp_path: Path) -> dict:
    return json.loads(
        (tmp_path / "inspect-results" / "data/c2a_results/cloud_research_latest.json").read_text(
            encoding="utf-8"
        )
    )


def _research_status_paths(day: str) -> set[str]:
    return {
        "data/c2a_results/cloud_research_latest.json",
        f"data/c2a_results/research/{day}.json",
        "reports/c2a_research_latest.md",
        f"reports/c2a/{day}-research.md",
    }


@pytest.mark.parametrize(
    ("status", "pipeline_status"),
    [
        ("PARTIAL", "DATA_NOT_READY"),
        ("DATA_NOT_READY", "DATA_NOT_READY"),
        ("READY", "DATA_NOT_READY"),
    ],
)
def test_incomplete_research_publishes_exactly_four_status_files(
    tmp_path: Path,
    status: str,
    pipeline_status: str,
) -> None:
    remote, _, _, environment = _cloud_repository(tmp_path)
    environment["ROBOT_QUANT_PYTHON"] = str(
        _research_phase_python(
            tmp_path,
            status=status,
            pipeline_status=pipeline_status,
        )
    )
    day = datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()

    completed = subprocess.run(
        [str(ROOT / "scripts/run_c2a_cloud_phase.sh"), "research"],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert _published_paths(remote, tmp_path) == _research_status_paths(day)


def test_completed_research_publishes_allowlisted_research_artifacts(tmp_path: Path) -> None:
    remote, _, _, environment = _cloud_repository(tmp_path)
    environment["ROBOT_QUANT_PYTHON"] = str(
        _research_phase_python(
            tmp_path,
            status="READY",
            pipeline_status="COMPLETED",
        )
    )
    day = datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()

    completed = subprocess.run(
        [str(ROOT / "scripts/run_c2a_cloud_phase.sh"), "research"],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert _published_paths(remote, tmp_path) == (
        _research_status_paths(day) | RESEARCH_RESULT_PATHS
    )


@pytest.mark.parametrize(
    ("failure_kind", "helper_argument"),
    [
        ("missing", "missing_artifact"),
        ("tampered", "tampered_artifact"),
        ("symlink", "symlink_artifact"),
    ],
)
def test_untrusted_research_artifact_publishes_only_status_files(
    tmp_path: Path,
    failure_kind: str,
    helper_argument: str,
) -> None:
    remote, _, _, environment = _cloud_repository(tmp_path)
    artifact = "data/c2a_results/baseline_trades.csv"
    environment["ROBOT_QUANT_PYTHON"] = str(
        _research_phase_python(
            tmp_path,
            status="READY",
            pipeline_status="COMPLETED",
            **{helper_argument: artifact},
        )
    )
    day = datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()

    completed = subprocess.run(
        [str(ROOT / "scripts/run_c2a_cloud_phase.sh"), "research"],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 75, f"{failure_kind}: {completed.stderr}"
    assert _published_paths(remote, tmp_path) == _research_status_paths(day)
    payload = _published_research_payload(tmp_path)
    assert payload["status"] == "DATA_NOT_READY"
    assert payload["research_pipeline"]["status"] == "DATA_NOT_READY"
    assert payload["reason"] == "研究产物发布校验失败"


def test_symlink_research_status_report_is_never_published(tmp_path: Path) -> None:
    remote, _, _, environment = _cloud_repository(tmp_path)
    environment["ROBOT_QUANT_PYTHON"] = str(
        _research_phase_python(
            tmp_path,
            status="READY",
            pipeline_status="COMPLETED",
            symlink_status_report=True,
        )
    )

    completed = subprocess.run(
        [str(ROOT / "scripts/run_c2a_cloud_phase.sh"), "research"],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 75
    day = datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()
    assert _published_paths(remote, tmp_path) == _research_status_paths(day)
    payload = _published_research_payload(tmp_path)
    assert payload["status"] == "DATA_NOT_READY"
    assert payload["reason"] == "研究产物发布校验失败"


@pytest.mark.parametrize("mapping_case", ["missing", "extra"])
def test_non_exact_research_artifact_mapping_publishes_only_status_files(
    tmp_path: Path,
    mapping_case: str,
) -> None:
    remote, _, _, environment = _cloud_repository(tmp_path)
    artifact = "data/c2a_results/baseline_trades.csv"
    mapping_arguments = (
        {"omit_mapping_path": artifact}
        if mapping_case == "missing"
        else {"extra_mapping_path": "data/c2a_results/not-allowlisted.json"}
    )
    environment["ROBOT_QUANT_PYTHON"] = str(
        _research_phase_python(
            tmp_path,
            status="READY",
            pipeline_status="COMPLETED",
            **mapping_arguments,
        )
    )
    day = datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()

    completed = subprocess.run(
        [str(ROOT / "scripts/run_c2a_cloud_phase.sh"), "research"],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 75, completed.stderr
    assert _published_paths(remote, tmp_path) == _research_status_paths(day)
    assert _published_research_payload(tmp_path)["status"] == "DATA_NOT_READY"


@pytest.mark.parametrize(
    "boundary_overrides",
    [
        {"execution_permission": "LIVE"},
        {"real_trade_authorized": True},
        {"current_new_entry_allowed": True},
        {"promotion_gate": "PASS"},
    ],
)
def test_invalid_model_boundary_publishes_only_research_status_files(
    tmp_path: Path,
    boundary_overrides: dict[str, object],
) -> None:
    remote, _, _, environment = _cloud_repository(tmp_path)
    environment["ROBOT_QUANT_PYTHON"] = str(
        _research_phase_python(
            tmp_path,
            status="READY",
            pipeline_status="COMPLETED",
            boundary_overrides=boundary_overrides,
        )
    )
    day = datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()

    completed = subprocess.run(
        [str(ROOT / "scripts/run_c2a_cloud_phase.sh"), "research"],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 75, completed.stderr
    assert _published_paths(remote, tmp_path) == _research_status_paths(day)
    assert _published_research_payload(tmp_path)["status"] == "DATA_NOT_READY"


def test_research_with_invalid_report_hash_publishes_exactly_four_status_files(
    tmp_path: Path,
) -> None:
    remote, _, _, environment = _cloud_repository(tmp_path)
    environment["ROBOT_QUANT_PYTHON"] = str(
        _research_phase_python(
            tmp_path,
            status="READY",
            pipeline_status="COMPLETED",
            valid_hash=False,
        )
    )
    day = datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()

    completed = subprocess.run(
        [str(ROOT / "scripts/run_c2a_cloud_phase.sh"), "research"],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 75, completed.stderr
    assert _published_paths(remote, tmp_path) == _research_status_paths(day)
    assert _published_research_payload(tmp_path)["status"] == "DATA_NOT_READY"


def test_research_with_stale_report_date_publishes_exactly_four_status_files(
    tmp_path: Path,
) -> None:
    remote, _, _, environment = _cloud_repository(tmp_path)
    today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    environment["ROBOT_QUANT_PYTHON"] = str(
        _research_phase_python(
            tmp_path,
            status="READY",
            pipeline_status="COMPLETED",
            report_as_of=(today - timedelta(days=1)).isoformat(),
        )
    )

    completed = subprocess.run(
        [str(ROOT / "scripts/run_c2a_cloud_phase.sh"), "research"],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 75, completed.stderr
    assert _published_paths(remote, tmp_path) == _research_status_paths(today.isoformat())
    assert _published_research_payload(tmp_path)["status"] == "DATA_NOT_READY"


def test_cloud_runner_uses_result_allowlist_instead_of_broad_git_add() -> None:
    script = (ROOT / "scripts/run_c2a_cloud_phase.sh").read_text(encoding="utf-8")

    assert 'git add -- "$path"' in script
    assert "git add data/" not in script
    assert "parameter_paths" not in script
    assert 'payload.get("execution_permission") != "PAPER_ONLY"' in script
    assert 'payload.get("real_trade_authorized") is not False' in script
    assert 'payload.get("current_new_entry_allowed") is not False' in script
    assert 'payload.get("promotion_gate") != "FAIL"' in script
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


def test_dirty_outbox_never_executes_tampered_worktree_failure_recorder(tmp_path) -> None:
    remote, _, cloud, environment = _cloud_repository(tmp_path)
    inspect = tmp_path / "inspect"
    marker = tmp_path / "tampered-recorder-executed"
    (cloud / "scripts/record_c2a_failure.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed', encoding='utf-8')\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [str(ROOT / "scripts/run_c2a_cloud_phase.sh"), "scan"],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert not marker.exists()
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
