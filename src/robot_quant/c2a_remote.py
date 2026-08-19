"""通过已配置的 SSH 别名编排 BigQuant AIStudio 上的 C2-A 任务。"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
from datetime import date
from pathlib import Path, PurePosixPath


DEFAULT_HOST = "bigquant-aistudio"
DEFAULT_REMOTE_ROOT = "/home/aiuser/work/robot-quant"
DEFAULT_FAST_ROOT = "data/c2a_fast"
SSH_HOST_PATTERN = re.compile(r"^[A-Za-z0-9._@-]+$")
REMOTE_ROOT_PATTERN = re.compile(r"^/[A-Za-z0-9._/-]+$")
REMOTE_COMMAND_TIMEOUT_SECONDS = 3 * 60 * 60
SSH_OPTIONS = (
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=15",
    "-o",
    "ServerAliveInterval=30",
    "-o",
    "ServerAliveCountMax=3",
)
REMOTE_RESULT_ALLOWLIST = (
    "baseline_equity.csv",
    "baseline_events.csv",
    "baseline_trades.csv",
    "data_audit.json",
    "latest_signal.json",
    "latest_state.json",
    "latest_training_grid.csv",
    "short_window_analysis.json",
    "walk_forward_oos_trades.csv",
    "walk_forward_selections.csv",
)


def sync_c2a_code(
    project_root: str | Path,
    *,
    host: str = DEFAULT_HOST,
    remote_root: str = DEFAULT_REMOTE_ROOT,
) -> None:
    """只同步 C2-A 代码，不上传凭证、本地数据、报告或 Git 元数据。"""

    root = Path(project_root).resolve()
    host = _validate_host(host)
    remote_root = _validate_remote_root(remote_root)
    package = root / "src" / "robot_quant"
    sources = [
        package / "__init__.py",
        package / "__main__.py",
        package / "cli.py",
        *sorted(package.glob("c2a*.py")),
    ]
    missing = [path for path in sources if not path.is_file()]
    if missing:
        raise RuntimeError(f"C2-A 远端同步缺少代码文件: {missing}")
    remote_package = f"{remote_root}/src/robot_quant"
    _run(["ssh", host, f"mkdir -p {shlex.quote(remote_package)}"], cwd=root)
    _run(
        ["rsync", "-az", *(str(path) for path in sources), f"{host}:{remote_package}/"],
        cwd=root,
    )


def update_remote_data(
    project_root: str | Path,
    start_date: str = "2026-01-01",
    end_date: str | None = None,
    *,
    host: str = DEFAULT_HOST,
    remote_root: str = DEFAULT_REMOTE_ROOT,
    sync_code: bool = True,
) -> None:
    root = Path(project_root).resolve()
    host = _validate_host(host)
    remote_root = _validate_remote_root(remote_root)
    start = _validate_date(start_date)
    end = _validate_date(end_date or date.today().isoformat())
    if sync_code:
        sync_c2a_code(root, host=host, remote_root=remote_root)
    _run_remote(
        host,
        remote_root,
        [
            "python",
            "-m",
            "robot_quant",
            "c2a-update-bigquant",
            "--data-root",
            "data/c2a",
            "--start",
            start,
            "--end",
            end,
            "--stream-cache",
        ],
        cwd=root,
    )


def run_remote_backtest(
    project_root: str | Path,
    start_date: str = "2026-01-01",
    end_date: str | None = None,
    *,
    optimize: bool = True,
    variant: str = "v1.2-challenger",
    host: str = DEFAULT_HOST,
    remote_root: str = DEFAULT_REMOTE_ROOT,
) -> None:
    root = Path(project_root).resolve()
    host = _validate_host(host)
    remote_root = _validate_remote_root(remote_root)
    command = [
        "python",
        "-m",
        "robot_quant",
        "c2a-backtest",
        "--data-root",
        "data/c2a",
        "--output-root",
        ".",
        "--start",
        _validate_date(start_date),
        "--end",
        _validate_date(end_date or date.today().isoformat()),
        "--variant",
        variant,
    ]
    if not optimize:
        command.append("--no-optimize")
    _run_remote(host, remote_root, command, cwd=root)


def run_remote_audit(
    project_root: str | Path,
    start_date: str = "2026-01-01",
    end_date: str | None = None,
    *,
    host: str = DEFAULT_HOST,
    remote_root: str = DEFAULT_REMOTE_ROOT,
) -> None:
    root = Path(project_root).resolve()
    host = _validate_host(host)
    remote_root = _validate_remote_root(remote_root)
    _run_remote(
        host,
        remote_root,
        [
            "python",
            "-m",
            "robot_quant",
            "c2a-audit",
            "--data-root",
            "data/c2a",
            "--output-root",
            ".",
            "--start",
            _validate_date(start_date),
            "--end",
            _validate_date(end_date or date.today().isoformat()),
            "--variant",
            "v1.2",
        ],
        cwd=root,
    )


def fetch_remote_results(
    project_root: str | Path,
    *,
    host: str = DEFAULT_HOST,
    remote_root: str = DEFAULT_REMOTE_ROOT,
) -> None:
    root = Path(project_root).resolve()
    host = _validate_host(host)
    remote_root = _validate_remote_root(remote_root)
    reports = root / "reports"
    results = root / "data" / "c2a_results"
    reports.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "rsync",
            "-az",
            f"{host}:{remote_root}/reports/c2a_2026_report.md",
            f"{reports}/",
        ],
        cwd=root,
    )
    include_arguments = [
        argument for name in REMOTE_RESULT_ALLOWLIST for argument in ("--include", name)
    ]
    _run(
        [
            "rsync",
            "-az",
            *include_arguments,
            "--exclude",
            "*",
            f"{host}:{remote_root}/data/c2a_results/",
            f"{results}/",
        ],
        cwd=root,
    )


def export_remote_fast_pack(
    project_root: str | Path,
    *,
    host: str = DEFAULT_HOST,
    remote_root: str = DEFAULT_REMOTE_ROOT,
    sync_code: bool = True,
) -> None:
    """在 AIStudio 内从已审计缓存导出不含凭证的早盘紧凑基线。"""

    root = Path(project_root).resolve()
    host = _validate_host(host)
    remote_root = _validate_remote_root(remote_root)
    if sync_code:
        sync_c2a_code(root, host=host, remote_root=remote_root)
    _run_remote(
        host,
        remote_root,
        [
            "python",
            "-m",
            "robot_quant.c2a_fast",
            "export-pack",
            "--project-root",
            ".",
            "--output-root",
            DEFAULT_FAST_ROOT,
        ],
        cwd=root,
    )


def fetch_remote_fast_pack(
    project_root: str | Path,
    *,
    host: str = DEFAULT_HOST,
    remote_root: str = DEFAULT_REMOTE_ROOT,
) -> None:
    """把远端紧凑基线拉到临时运行器；基线目录始终由 git 忽略。"""

    root = Path(project_root).resolve()
    host = _validate_host(host)
    remote_root = _validate_remote_root(remote_root)
    destination = root / DEFAULT_FAST_ROOT
    destination.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "rsync",
            "-az",
            f"{host}:{remote_root}/{DEFAULT_FAST_ROOT}/",
            f"{destination}/",
        ],
        cwd=root,
    )


def push_remote_fast_pack(
    project_root: str | Path,
    *,
    host: str = DEFAULT_HOST,
    remote_root: str = DEFAULT_REMOTE_ROOT,
) -> None:
    """把盘前增量后的紧凑基线写回 AIStudio 的持久工作区。"""

    root = Path(project_root).resolve()
    host = _validate_host(host)
    remote_root = _validate_remote_root(remote_root)
    source = root / DEFAULT_FAST_ROOT
    required = ("rolling_state.npz", "universe.csv.gz", "manifest.json")
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise RuntimeError(f"C2-A 紧凑基线缺少文件: {missing}")
    remote_destination = f"{remote_root}/{DEFAULT_FAST_ROOT}"
    _run(
        ["ssh", host, f"mkdir -p {shlex.quote(remote_destination)}"],
        cwd=root,
    )
    _run(
        ["rsync", "-az", f"{source}/", f"{host}:{remote_destination}/"],
        cwd=root,
    )


def run_remote_pipeline(
    project_root: str | Path,
    start_date: str = "2026-01-01",
    end_date: str | None = None,
    *,
    optimize: bool = True,
    variant: str = "v1.2-challenger",
    data_start_date: str = "2026-01-01",
    host: str = DEFAULT_HOST,
    remote_root: str = DEFAULT_REMOTE_ROOT,
) -> None:
    """同步代码、增量数据、运行回测并把机器结果拉回 PyCharm 项目。"""

    sync_c2a_code(project_root, host=host, remote_root=remote_root)
    update_remote_data(
        project_root,
        data_start_date,
        end_date,
        host=host,
        remote_root=remote_root,
        sync_code=False,
    )
    run_remote_backtest(
        project_root,
        start_date,
        end_date,
        optimize=optimize,
        variant=variant,
        host=host,
        remote_root=remote_root,
    )
    fetch_remote_results(project_root, host=host, remote_root=remote_root)


def _run_remote(
    host: str,
    remote_root: str,
    command: list[str],
    *,
    cwd: Path,
) -> None:
    host = _validate_host(host)
    remote_root = _validate_remote_root(remote_root)
    remote_command = (
        f"cd {shlex.quote(remote_root)} && PYTHONUNBUFFERED=1 PYTHONPATH=src {shlex.join(command)}"
    )
    _run(["ssh", host, remote_command], cwd=cwd)


def _run(command: list[str], *, cwd: Path) -> None:
    effective = list(command)
    if effective and effective[0] == "ssh":
        effective[1:1] = SSH_OPTIONS
    elif effective and effective[0] == "rsync":
        effective[1:1] = [
            "--timeout=120",
            "-e",
            shlex.join(("ssh", *SSH_OPTIONS)),
        ]
    try:
        subprocess.run(
            effective,
            cwd=cwd,
            check=True,
            timeout=REMOTE_COMMAND_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as error:
        raise RuntimeError(f"缺少命令: {command[0]}") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("C2-A 远端命令超时") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"C2-A 远端命令失败，退出码 {error.returncode}") from error


def _validate_date(value: str) -> str:
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as error:
        raise ValueError("日期必须是 YYYY-MM-DD") from error


def _validate_host(value: str) -> str:
    if value.startswith("-") or not SSH_HOST_PATTERN.fullmatch(value):
        raise ValueError("SSH Host 只能使用已配置的主机名或 user@host")
    return value


def _validate_remote_root(value: str) -> str:
    if not REMOTE_ROOT_PATTERN.fullmatch(value):
        raise ValueError("远端目录必须是安全的绝对 POSIX 路径")
    normalized = PurePosixPath(value)
    if (
        value == "/"
        or value.startswith("//")
        or ".." in normalized.parts
        or str(normalized) != value
    ):
        raise ValueError("远端目录不能是根目录、上级跳转或非规范路径")
    return str(normalized)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m robot_quant.c2a_remote")
    parser.add_argument("action", choices=("sync", "update", "audit", "backtest", "run", "fetch"))
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--no-optimize", action="store_true")
    parser.add_argument(
        "--variant",
        choices=("v1.2", "v1.2-challenger"),
        default="v1.2-challenger",
    )
    args = parser.parse_args()
    common = {
        "host": args.host,
        "remote_root": args.remote_root,
    }
    if args.action == "sync":
        sync_c2a_code(args.project_root, **common)
    elif args.action == "update":
        update_remote_data(args.project_root, args.start, args.end, **common)
    elif args.action == "audit":
        run_remote_audit(args.project_root, args.start, args.end, **common)
        fetch_remote_results(args.project_root, **common)
    elif args.action == "backtest":
        run_remote_backtest(
            args.project_root,
            args.start,
            args.end,
            optimize=not args.no_optimize,
            variant=args.variant,
            **common,
        )
        fetch_remote_results(args.project_root, **common)
    elif args.action == "run":
        run_remote_pipeline(
            args.project_root,
            args.start,
            args.end,
            optimize=not args.no_optimize,
            variant=args.variant,
            **common,
        )
    else:
        fetch_remote_results(args.project_root, **common)


if __name__ == "__main__":
    main()
