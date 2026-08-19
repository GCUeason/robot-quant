"""通过已配置的 SSH 别名编排 BigQuant AIStudio 上的 C2-A 任务。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import secrets
import shlex
import shutil
import subprocess
import tempfile
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
REMOTE_DATA_RESULT_ALLOWLIST = (
    "baseline_equity.csv",
    "baseline_events.csv",
    "baseline_trades.csv",
    "data_audit.json",
    "latest_signal.json",
    "latest_state.json",
    "latest_training_grid.csv",
    "walk_forward_oos_trades.csv",
    "walk_forward_selections.csv",
)
REMOTE_RESULT_ALLOWLIST = (
    "reports/c2a_2026_report.md",
    *(f"data/c2a_results/{name}" for name in REMOTE_DATA_RESULT_ALLOWLIST),
)
REMOTE_AUDIT_RESULT_ALLOWLIST = (
    "reports/c2a_2026_report.md",
    "data/c2a_results/data_audit.json",
)
FAST_PACK_HASHED_FILES = frozenset(("rolling_state.npz", "universe.csv.gz"))
FAST_PACK_REQUIRED_FILES = (*sorted(FAST_PACK_HASHED_FILES), "manifest.json")
FAST_PACK_PUSH_SUFFIX_PATTERN = re.compile(r"^[0-9a-f]{32}$")
FAST_PACK_PUSH_STAGING_PREFIX = ".c2a_fast.push-"
FAST_PACK_PUSH_BACKUP_PREFIX = ".c2a_fast.previous-"
RESEARCH_RUN_PREFIX = ".c2a-research-run-"
RESEARCH_RUN_SUFFIX_PATTERN = re.compile(r"^[0-9a-f]{32}$")
CARRIED_FORWARD_RESULT_FILES = (
    "latest_state.json",
    "walk_forward_selections.csv",
    "walk_forward_oos_trades.csv",
    "latest_training_grid.csv",
)
RESEARCH_CARRY_ROOT = "data/c2a_research_carry"
RESEARCH_CARRY_INPUT_DIR = ".c2a-carry-input"
RESEARCH_CARRY_STAGING_PREFIX = ".c2a_research_carry.push-"
RESEARCH_CARRY_BACKUP_PREFIX = ".c2a_research_carry.previous-"
REMOTE_FAST_PACK_PUSH_CODE = (
    "import sys; "
    "from robot_quant.c2a_remote import _remote_fast_pack_push; "
    "_remote_fast_pack_push('.', sys.argv[1], sys.argv[2])"
)
REMOTE_RESEARCH_RUN_CODE = (
    "import sys; "
    "from robot_quant.c2a_remote import _remote_research_run; "
    "_remote_research_run('.', sys.argv[1], sys.argv[2], sys.argv[3] == '1')"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_verified_fast_pack(path: Path, *, source: str):
    """在运输边界完整校验 fast pack，不信任可选哈希或 manifest 单方声明。"""

    for name in FAST_PACK_REQUIRED_FILES:
        candidate = path / name
        if candidate.is_symlink() or not candidate.is_file():
            raise RuntimeError(f"C2-A {source}紧凑基线校验失败: 缺少安全文件 {name}")

    # 局部导入避免 c2a_fast 在模块加载时与本模块形成循环依赖。
    from robot_quant.c2a_fast import load_fast_pack

    try:
        pack = load_fast_pack(path)
    except Exception as error:  # noqa: BLE001 - 传输边界必须把任何解析损坏统一收敛为失败
        raise RuntimeError(f"C2-A {source}紧凑基线校验失败: {error}") from error

    hashes = pack.manifest.get("file_sha256")
    if not isinstance(hashes, dict) or set(hashes) != FAST_PACK_HASHED_FILES:
        raise RuntimeError(f"C2-A {source}紧凑基线哈希清单不完整")
    state_date = pack.last_processed_date.date().isoformat()
    if pack.manifest.get("last_processed_date") != state_date:
        raise RuntimeError(f"C2-A {source}紧凑基线 manifest 与状态日期不一致")
    if pack.manifest.get("ticker_count") != len(pack.tickers):
        raise RuntimeError(f"C2-A {source}紧凑基线 manifest 与股票数量不一致")
    if (
        pack.manifest.get("execution_permission") != "PAPER_ONLY"
        or pack.manifest.get("real_trade_authorized") is not False
    ):
        raise RuntimeError(f"C2-A {source}紧凑基线缺少 PAPER_ONLY 权限边界")
    return pack


def _fast_pack_push_paths(project_root: str | Path, suffix: str) -> tuple[Path, Path, Path]:
    if not FAST_PACK_PUSH_SUFFIX_PATTERN.fullmatch(suffix):
        raise ValueError("C2-A FastPack 推送后缀无效")
    root = Path(project_root).resolve()
    data_root = root / "data"
    if data_root.is_symlink():
        raise RuntimeError("C2-A 远端 data 目录不得是符号链接")
    data_root.mkdir(parents=True, exist_ok=True)
    if data_root.resolve() != data_root:
        raise RuntimeError("C2-A 远端 data 目录不在项目根目录内")
    return (
        data_root / "c2a_fast",
        data_root / f"{FAST_PACK_PUSH_STAGING_PREFIX}{suffix}",
        data_root / f"{FAST_PACK_PUSH_BACKUP_PREFIX}{suffix}",
    )


def _remove_fast_pack_transport_path(path: Path) -> None:
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _promote_remote_fast_pack(project_root: str | Path, suffix: str) -> None:
    destination, staging, backup = _fast_pack_push_paths(project_root, suffix)
    staged_pack = _load_verified_fast_pack(staging, source="远端暂存")
    if destination.is_symlink():
        raise RuntimeError("C2-A 远端权威基线不得是符号链接")
    if destination.exists():
        current_pack = _load_verified_fast_pack(destination, source="远端权威")
        if staged_pack.last_processed_date < current_pack.last_processed_date:
            raise RuntimeError("C2-A 本地紧凑基线早于远端权威基线，拒绝回退")
    if backup.exists() or backup.is_symlink():
        raise RuntimeError("C2-A FastPack 推送备份路径已存在")

    had_destination = destination.exists()
    if had_destination:
        destination.replace(backup)
    try:
        staging.replace(destination)
        _load_verified_fast_pack(destination, source="远端提升后")
    except BaseException as primary_error:
        try:
            _remove_fast_pack_transport_path(destination)
            if had_destination and backup.exists():
                backup.replace(destination)
        except BaseException as rollback_error:
            raise RuntimeError(
                f"C2-A 远端紧凑基线提升失败且回滚失败: {rollback_error}"
            ) from primary_error
        raise
    else:
        try:
            _remove_fast_pack_transport_path(backup)
        except OSError:
            # 权威目录已经完整提升；残留唯一命名备份不应把成功误报为失败。
            pass


def _remote_fast_pack_push(project_root: str | Path, action: str, suffix: str) -> None:
    destination, staging, backup = _fast_pack_push_paths(project_root, suffix)
    if action == "prepare":
        if staging.exists() or staging.is_symlink() or backup.exists() or backup.is_symlink():
            raise RuntimeError("C2-A FastPack 推送暂存路径已存在")
        staging.mkdir(mode=0o700)
    elif action == "promote":
        _promote_remote_fast_pack(project_root, suffix)
    elif action == "cleanup":
        _remove_fast_pack_transport_path(staging)
    else:
        raise ValueError("C2-A FastPack 远端动作无效")


def _run_remote_fast_pack_push(
    root: Path,
    host: str,
    remote_root: str,
    action: str,
    suffix: str,
) -> None:
    _run_remote(
        host,
        remote_root,
        ["python", "-c", REMOTE_FAST_PACK_PUSH_CODE, action, suffix],
        cwd=root,
    )


def _research_run_path(project_root: str | Path, suffix: str) -> Path:
    if not RESEARCH_RUN_SUFFIX_PATTERN.fullmatch(suffix):
        raise ValueError("C2-A 研究隔离目录后缀无效")
    root = Path(project_root).resolve()
    return root / f"{RESEARCH_RUN_PREFIX}{suffix}"


def _validate_carried_forward_results(
    source_root: Path,
    *,
    source: str,
    exact: bool,
) -> dict[str, str]:
    if source_root.is_symlink() or not source_root.is_dir():
        raise RuntimeError(f"C2-A {source} carried-forward 目录无效")
    if exact:
        names = {path.name for path in source_root.iterdir()}
        if names != set(CARRIED_FORWARD_RESULT_FILES):
            raise RuntimeError(f"C2-A {source} carried-forward 文件集合不完整")
    paths = [source_root / name for name in CARRIED_FORWARD_RESULT_FILES]
    invalid = [path.name for path in paths if path.is_symlink() or not path.is_file()]
    if invalid:
        raise RuntimeError(f"C2-A {source} carried-forward 结果不完整: {invalid}")
    state_path = source_root / "latest_state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"C2-A {source} latest_state.json 无法解析") from error
    if not isinstance(state, dict) or not isinstance(state.get("walk_forward"), dict):
        raise RuntimeError(f"C2-A {source} latest_state.json 缺少 walk_forward")
    return {path.name: _sha256_file(path) for path in paths}


def _copy_carried_forward_results(source_root: Path, destination: Path) -> None:
    source_hashes = _validate_carried_forward_results(
        source_root,
        source="来源",
        exact=False,
    )
    destination.mkdir(parents=True)
    for name in CARRIED_FORWARD_RESULT_FILES:
        shutil.copy2(source_root / name, destination / name)
    copied_hashes = _validate_carried_forward_results(
        destination,
        source="复制后",
        exact=True,
    )
    if copied_hashes != source_hashes:
        raise RuntimeError("C2-A carried-forward 复制后哈希不一致")


def _seed_carried_forward_results(project_root: Path, run_root: Path) -> None:
    carry_root = project_root / RESEARCH_CARRY_ROOT
    destination = run_root / RESEARCH_CARRY_INPUT_DIR
    if carry_root.exists() or carry_root.is_symlink():
        _copy_carried_forward_results(carry_root, destination)
        return

    canonical = project_root / "data" / "c2a_results"
    if canonical.is_symlink():
        raise RuntimeError("C2-A canonical 研究结果目录不得是符号链接")
    state_path = canonical / "latest_state.json"
    if state_path.is_symlink():
        raise RuntimeError("C2-A canonical latest_state.json 不得是符号链接")
    if not state_path.exists():
        return
    if not state_path.is_file():
        raise RuntimeError("C2-A canonical latest_state.json 不是普通文件")
    try:
        prior_state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("C2-A canonical latest_state.json 无法解析") from error
    walk_forward = prior_state.get("walk_forward") if isinstance(prior_state, dict) else None
    if not isinstance(walk_forward, dict):
        return
    if walk_forward.get("status") not in {"COMPLETED", "CARRIED_FORWARD"}:
        return

    _copy_carried_forward_results(canonical, destination)


def _promote_research_carry(project_root: Path, run_root: Path, suffix: str) -> None:
    source_root = run_root / "data" / "c2a_results"
    source_hashes = _validate_carried_forward_results(
        source_root,
        source="本轮",
        exact=False,
    )
    data_root = project_root / "data"
    if data_root.is_symlink():
        raise RuntimeError("C2-A 远端 data 目录不得是符号链接")
    data_root.mkdir(parents=True, exist_ok=True)
    destination = project_root / RESEARCH_CARRY_ROOT
    staging = data_root / f"{RESEARCH_CARRY_STAGING_PREFIX}{suffix}"
    backup = data_root / f"{RESEARCH_CARRY_BACKUP_PREFIX}{suffix}"
    if staging.exists() or staging.is_symlink() or backup.exists() or backup.is_symlink():
        raise RuntimeError("C2-A carry 快照运输路径已存在")

    staging.mkdir(mode=0o700)
    transport_error: BaseException | None = None
    try:
        for name in CARRIED_FORWARD_RESULT_FILES:
            shutil.copy2(source_root / name, staging / name)
        staged_hashes = _validate_carried_forward_results(
            staging,
            source="暂存",
            exact=True,
        )
        if staged_hashes != source_hashes:
            raise RuntimeError("C2-A carry 快照暂存哈希不一致")
        if destination.is_symlink():
            raise RuntimeError("C2-A carry 权威目录不得是符号链接")
        if destination.exists():
            _validate_carried_forward_results(
                destination,
                source="旧权威",
                exact=True,
            )
        had_destination = destination.exists()
        if had_destination:
            destination.replace(backup)
        try:
            staging.replace(destination)
            promoted_hashes = _validate_carried_forward_results(
                destination,
                source="提升后",
                exact=True,
            )
            if promoted_hashes != source_hashes:
                raise RuntimeError("C2-A carry 快照提升后哈希不一致")
        except BaseException as primary_error:
            try:
                _remove_fast_pack_transport_path(destination)
                if had_destination and backup.exists():
                    backup.replace(destination)
            except BaseException as rollback_error:
                raise RuntimeError(
                    f"C2-A carry 快照提升失败且回滚失败: {rollback_error}"
                ) from primary_error
            raise
        else:
            try:
                _remove_fast_pack_transport_path(backup)
            except OSError:
                pass
    except BaseException as error:
        transport_error = error
        raise
    finally:
        try:
            _remove_fast_pack_transport_path(staging)
        except BaseException:
            if transport_error is None:
                raise


def _remote_research_run(
    project_root: str | Path,
    action: str,
    suffix: str,
    carry_forward: bool,
) -> None:
    root = Path(project_root).resolve()
    run_root = _research_run_path(root, suffix)
    if action == "prepare":
        if run_root.exists() or run_root.is_symlink():
            raise RuntimeError("C2-A 研究隔离目录已存在")
        run_root.mkdir(mode=0o700)
        try:
            if carry_forward:
                _seed_carried_forward_results(root, run_root)
        except BaseException:
            _remove_fast_pack_transport_path(run_root)
            raise
    elif action == "cleanup":
        _remove_fast_pack_transport_path(run_root)
    elif action == "promote-carry":
        _promote_research_carry(root, run_root, suffix)
    else:
        raise ValueError("C2-A 研究隔离目录动作无效")


def _run_remote_research_run(
    root: Path,
    host: str,
    remote_root: str,
    action: str,
    suffix: str,
    carry_forward: bool,
) -> None:
    _run_remote(
        host,
        remote_root,
        [
            "python",
            "-c",
            REMOTE_RESEARCH_RUN_CODE,
            action,
            suffix,
            "1" if carry_forward else "0",
        ],
        cwd=root,
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
    output_root: str = ".",
    prior_results_root: str | None = None,
    host: str = DEFAULT_HOST,
    remote_root: str = DEFAULT_REMOTE_ROOT,
) -> None:
    root = Path(project_root).resolve()
    host = _validate_host(host)
    remote_root = _validate_remote_root(remote_root)
    validated_output_root = _validate_remote_output_root(output_root)
    command = [
        "python",
        "-m",
        "robot_quant",
        "c2a-backtest",
        "--data-root",
        "data/c2a",
        "--output-root",
        validated_output_root,
        "--start",
        _validate_date(start_date),
        "--end",
        _validate_date(end_date or date.today().isoformat()),
        "--variant",
        variant,
    ]
    if prior_results_root is not None:
        expected_prior_root = f"{validated_output_root}/{RESEARCH_CARRY_INPUT_DIR}"
        if prior_results_root != expected_prior_root:
            raise ValueError("C2-A prior results 必须位于当前研究隔离目录")
        command.extend(("--prior-results-root", prior_results_root))
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
    remote_output_root: str = ".",
) -> dict[str, str]:
    return _fetch_remote_result_set(
        project_root,
        host=host,
        remote_root=remote_root,
        remote_output_root=remote_output_root,
        data_allowlist=REMOTE_DATA_RESULT_ALLOWLIST,
        result_allowlist=REMOTE_RESULT_ALLOWLIST,
    )


def fetch_remote_audit_results(
    project_root: str | Path,
    *,
    host: str = DEFAULT_HOST,
    remote_root: str = DEFAULT_REMOTE_ROOT,
) -> dict[str, str]:
    return _fetch_remote_result_set(
        project_root,
        host=host,
        remote_root=remote_root,
        remote_output_root=".",
        data_allowlist=("data_audit.json",),
        result_allowlist=REMOTE_AUDIT_RESULT_ALLOWLIST,
    )


def _fetch_remote_result_set(
    project_root: str | Path,
    *,
    host: str,
    remote_root: str,
    remote_output_root: str,
    data_allowlist: tuple[str, ...],
    result_allowlist: tuple[str, ...],
) -> dict[str, str]:
    root = Path(project_root).resolve()
    host = _validate_host(host)
    remote_root = _validate_remote_root(remote_root)
    output_root = _validate_remote_output_root(remote_output_root)
    remote_source_root = remote_root if output_root == "." else f"{remote_root}/{output_root}"
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root, prefix=".c2a-results.fetch-") as temporary:
        staging = Path(temporary)
        staged_reports = staging / "reports"
        staged_results = staging / "data" / "c2a_results"
        staged_reports.mkdir(parents=True)
        staged_results.mkdir(parents=True)
        _run(
            [
                "rsync",
                "-az",
                f"{host}:{remote_source_root}/reports/c2a_2026_report.md",
                f"{staged_reports}/",
            ],
            cwd=root,
        )
        include_arguments = [
            argument for name in data_allowlist for argument in ("--include", name)
        ]
        _run(
            [
                "rsync",
                "-az",
                *include_arguments,
                "--exclude",
                "*",
                f"{host}:{remote_source_root}/data/c2a_results/",
                f"{staged_results}/",
            ],
            cwd=root,
        )

        manifest: dict[str, str] = {}
        for relative_path in result_allowlist:
            staged_path = staging / relative_path
            if staged_path.is_symlink() or not staged_path.is_file():
                raise RuntimeError(f"C2-A 远端缺少本轮固定产物: {relative_path}")
            manifest[relative_path] = _sha256_file(staged_path)
        for relative_path in result_allowlist:
            staged_path = staging / relative_path
            destination = root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            staged_path.replace(destination)
        return manifest


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
    """先在同文件系统暂存并校验远端基线，再替换本地权威包。"""

    root = Path(project_root).resolve()
    host = _validate_host(host)
    remote_root = _validate_remote_root(remote_root)
    destination = root / DEFAULT_FAST_ROOT
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=destination.parent,
        prefix=f".{destination.name}.fetch-",
    ) as temporary:
        staging = Path(temporary)
        _run(
            [
                "rsync",
                "-az",
                f"{host}:{remote_root}/{DEFAULT_FAST_ROOT}/",
                f"{staging}/",
            ],
            cwd=root,
        )

        remote_pack = _load_verified_fast_pack(staging, source="远端")
        local_pack = None
        if destination.is_dir():
            try:
                local_pack = _load_verified_fast_pack(destination, source="本地")
            except RuntimeError:
                pass
        if (
            local_pack is not None
            and remote_pack.last_processed_date < local_pack.last_processed_date
        ):
            raise RuntimeError("C2-A 远端紧凑基线早于本地权威基线，拒绝回退")

        backup = staging.with_name(f"{staging.name}.previous")
        if destination.exists():
            destination.replace(backup)
        try:
            staging.replace(destination)
        except BaseException:
            if backup.exists() and not destination.exists():
                backup.replace(destination)
            raise
        else:
            if backup.exists():
                shutil.rmtree(backup)


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
    missing = [name for name in FAST_PACK_REQUIRED_FILES if not (source / name).is_file()]
    if missing:
        raise RuntimeError(f"C2-A 紧凑基线缺少文件: {missing}")
    _load_verified_fast_pack(source, source="本地")
    sync_c2a_code(root, host=host, remote_root=remote_root)
    suffix = secrets.token_hex(16)
    if not FAST_PACK_PUSH_SUFFIX_PATTERN.fullmatch(suffix):
        raise RuntimeError("C2-A FastPack 推送随机后缀无效")
    remote_staging = f"{remote_root}/data/{FAST_PACK_PUSH_STAGING_PREFIX}{suffix}"
    cleanup_needed = False
    primary_error: BaseException | None = None
    try:
        _run_remote_fast_pack_push(root, host, remote_root, "prepare", suffix)
        cleanup_needed = True
        _run(
            ["rsync", "-az", f"{source}/", f"{host}:{remote_staging}/"],
            cwd=root,
        )
        _run_remote_fast_pack_push(root, host, remote_root, "promote", suffix)
    except BaseException as error:
        primary_error = error
        raise
    finally:
        if cleanup_needed:
            try:
                _run_remote_fast_pack_push(root, host, remote_root, "cleanup", suffix)
            except BaseException:
                if primary_error is None:
                    raise


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
) -> dict[str, str]:
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
    root = Path(project_root).resolve()
    host = _validate_host(host)
    remote_root = _validate_remote_root(remote_root)
    suffix = secrets.token_hex(16)
    if not RESEARCH_RUN_SUFFIX_PATTERN.fullmatch(suffix):
        raise RuntimeError("C2-A 研究隔离目录随机后缀无效")
    output_root = f"{RESEARCH_RUN_PREFIX}{suffix}"
    cleanup_needed = False
    primary_error: BaseException | None = None
    try:
        _run_remote_research_run(
            root,
            host,
            remote_root,
            "prepare",
            suffix,
            not optimize,
        )
        cleanup_needed = True
        run_remote_backtest(
            root,
            start_date,
            end_date,
            optimize=optimize,
            variant=variant,
            output_root=output_root,
            prior_results_root=(
                f"{output_root}/{RESEARCH_CARRY_INPUT_DIR}" if not optimize else None
            ),
            host=host,
            remote_root=remote_root,
        )
        manifest = fetch_remote_results(
            root,
            host=host,
            remote_root=remote_root,
            remote_output_root=output_root,
        )
        _run_remote_research_run(
            root,
            host,
            remote_root,
            "promote-carry",
            suffix,
            False,
        )
        return manifest
    except BaseException as error:
        primary_error = error
        raise
    finally:
        if cleanup_needed:
            try:
                _run_remote_research_run(
                    root,
                    host,
                    remote_root,
                    "cleanup",
                    suffix,
                    False,
                )
            except BaseException:
                if primary_error is None:
                    raise


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


def _validate_remote_output_root(value: str) -> str:
    if value == ".":
        return value
    prefix = RESEARCH_RUN_PREFIX
    if not value.startswith(prefix) or not RESEARCH_RUN_SUFFIX_PATTERN.fullmatch(
        value[len(prefix) :]
    ):
        raise ValueError("远端输出目录必须是当前 C2-A 研究隔离目录")
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
        fetch_remote_audit_results(args.project_root, **common)
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
