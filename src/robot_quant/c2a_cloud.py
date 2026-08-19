"""C2-A 云端四阶段入口与可审计报告。

持续在线的调度器依次调用 ``prepare``、``scan``、``review``、``research``。
本模块始终写入当日状态；输入不完整时输出空候选 ``DATA_NOT_READY``，
绝不沿用上一交易日的信号。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from robot_quant.c2a_fast import (
    FastScanDataError,
    MIN_FAST_COVERAGE,
    fetch_quotes_fast,
    load_fast_pack,
    prepare_fast_pack,
    render_scan_result,
    run_fast_scan,
)
from robot_quant.c2a_remote import (
    DEFAULT_HOST,
    DEFAULT_REMOTE_ROOT,
    export_remote_fast_pack,
    fetch_remote_fast_pack,
    push_remote_fast_pack,
    run_remote_pipeline,
)


SHANGHAI = ZoneInfo("Asia/Shanghai")
PHASES = ("prepare", "scan", "review", "research")
DEFAULT_SCAN_CUTOFF = time(10, 0)
PHASE_WINDOWS = {
    "prepare": (time(8, 0), time(9, 25)),
    "scan": (time(10, 0), time(10, 8)),
    "review": (time(16, 30), time(16, 40)),
    "research": (time(16, 35), time(18, 30)),
}
PHASE_SCHEDULES = {
    "prepare": time(8, 45),
    "scan": time(10, 2, 30),
    "review": time(16, 30),
    "research": time(16, 35),
}


def _write_json_atomic(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_text_atomic(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def _safe_error(error: Exception) -> str:
    """错误可公开写入仓库，但不得泄露本机/远端用户目录。"""

    message = str(error).replace("\n", " ").strip()
    message = re.sub(r"/(?:Users|home)/[^\s:]+", "<private-path>", message)
    return message[:500] or type(error).__name__


def _remote_settings() -> tuple[str, str]:
    return (
        os.environ.get("C2A_SSH_HOST", DEFAULT_HOST),
        os.environ.get("C2A_REMOTE_ROOT", DEFAULT_REMOTE_ROOT),
    )


def _ensure_local_fast_pack(root: Path, host: str, remote_root: str) -> None:
    try:
        load_fast_pack(root / "data" / "c2a_fast")
        return
    except (FastScanDataError, ValueError, OSError):
        fetch_remote_fast_pack(root, host=host, remote_root=remote_root)
    load_fast_pack(root / "data" / "c2a_fast")


def _base_payload(phase: str, now: datetime, trade_day: date | None = None) -> dict:
    day = trade_day or now.date()
    scheduled_at = datetime.combine(day, PHASE_SCHEDULES[phase], SHANGHAI)
    return {
        "phase": phase.upper(),
        "as_of": day.isoformat(),
        "generated_at": now.isoformat(),
        "scheduled_at": scheduled_at.isoformat(),
        "started_at": now.isoformat(),
        "execution_permission": "PAPER_ONLY",
        "real_trade_authorized": False,
        "current_new_entry_allowed": False,
        "retryable": False,
    }


def _require_phase_window(phase: str, current: datetime) -> None:
    start, end = PHASE_WINDOWS[phase]
    wall_clock = current.astimezone(SHANGHAI).time().replace(tzinfo=None)
    if not start <= wall_clock <= end:
        raise FastScanDataError(
            f"错误时点触发：{phase} 只允许 {start.strftime('%H:%M')}—{end.strftime('%H:%M')}"
        )


def _phase_paths(root: Path, phase: str, trade_day: date) -> tuple[Path, Path, Path, Path]:
    data_root = root / "data" / "c2a_results"
    report_root = root / "reports" / "c2a"
    return (
        data_root / f"cloud_{phase}_latest.json",
        data_root / phase / f"{trade_day.isoformat()}.json",
        root / "reports" / f"c2a_{phase}_latest.md",
        report_root / f"{trade_day.isoformat()}-{phase}.md",
    )


def _source_commit(root: Path) -> str | None:
    configured = os.environ.get("ROBOT_QUANT_SOURCE_COMMIT") or os.environ.get("GITHUB_SHA")
    if configured and re.fullmatch(r"[0-9a-fA-F]{7,40}", configured):
        return configured.lower()
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value if re.fullmatch(r"[0-9a-f]{40}", value) else None


def _finalize_payload(root: Path, payload: dict) -> None:
    payload.setdefault("source_commit", _source_commit(root))
    source_manifest = root / "data" / "c2a_fast" / "manifest.json"
    payload.setdefault(
        "source_manifest_sha256",
        _sha256_file(source_manifest) if source_manifest.is_file() else None,
    )
    payload.setdefault("finished_at", datetime.now(SHANGHAI).isoformat())
    hash_input = {key: value for key, value in payload.items() if key != "payload_sha256"}
    encoded = json.dumps(
        hash_input,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload["payload_sha256"] = hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_phase_artifacts(root: Path, phase: str, payload: dict, markdown: str) -> None:
    _finalize_payload(root, payload)
    paths = _phase_paths(root, phase, date.fromisoformat(payload["as_of"]))
    _write_json_atomic(payload, paths[0])
    _write_json_atomic(payload, paths[1])
    _write_text_atomic(markdown, paths[2])
    _write_text_atomic(markdown, paths[3])


def run_prepare_phase(
    project_root: str | Path,
    *,
    now: datetime | None = None,
    enforce_time: bool = True,
) -> dict:
    current = now or datetime.now(SHANGHAI)
    root = Path(project_root).resolve()
    host, remote_root = _remote_settings()
    payload = _base_payload("prepare", current)
    try:
        if enforce_time:
            _require_phase_window("prepare", current)
        try:
            _ensure_local_fast_pack(root, host, remote_root)
        except RuntimeError:
            export_remote_fast_pack(root, host=host, remote_root=remote_root)
            fetch_remote_fast_pack(root, host=host, remote_root=remote_root)
            load_fast_pack(root / "data" / "c2a_fast")
        result = prepare_fast_pack(root)
        if float(result["coverage"]) < MIN_FAST_COVERAGE:
            raise FastScanDataError(f"盘前基线覆盖不足: {float(result['coverage']):.2%}")
        push_remote_fast_pack(root, host=host, remote_root=remote_root)
        payload.update(
            {
                "status": "READY",
                "baseline_as_of": result["last_processed_date"],
                "updated_trading_days": result["updated_days"],
                "baseline_coverage": result["coverage"],
                "reason": None,
                "retryable": False,
            }
        )
    except (FastScanDataError, RuntimeError, ValueError, OSError) as error:
        payload.update(
            {
                "status": "DATA_NOT_READY",
                "baseline_as_of": None,
                "updated_trading_days": 0,
                "baseline_coverage": 0.0,
                "reason": _safe_error(error),
                "retryable": "错误时点触发" not in str(error),
            }
        )
    markdown = _prepare_markdown(payload)
    _write_phase_artifacts(root, "prepare", payload, markdown)
    return payload


def run_scan_phase(
    project_root: str | Path,
    *,
    now: datetime | None = None,
    trade_day: date | None = None,
    cutoff: time = DEFAULT_SCAN_CUTOFF,
    enforce_time: bool = True,
) -> dict:
    current = now or datetime.now(SHANGHAI)
    day = trade_day or current.date()
    root = Path(project_root).resolve()
    host, remote_root = _remote_settings()
    try:
        if enforce_time:
            _require_phase_window("scan", current)
        _ensure_local_fast_pack(root, host, remote_root)
        payload = run_fast_scan(
            root,
            trade_day=day,
            cutoff=cutoff,
            now=current,
        )
        payload["phase"] = "SCAN"
        payload.setdefault(
            "signal_status",
            "SIMULATED_ENTRY" if payload.get("entries") else "NO_SIMULATED_SIGNAL",
        )
        payload["retryable"] = False
    except (FastScanDataError, RuntimeError, ValueError, OSError) as error:
        payload = {
            **_base_payload("scan", current, day),
            "as_of": day.isoformat(),
            "status": "DATA_NOT_READY",
            "signal_status": "NO_SIMULATED_SIGNAL",
            "data_status": "DATA_NOT_READY",
            "cutoff": cutoff.strftime("%H:%M"),
            "entries": [],
            "watchlist": [],
            "entry_count": 0,
            "candidate_count": 0,
            "promotion_gate": "FAIL",
            "reason": _safe_error(error),
            "retryable": "错误时点触发" not in str(error),
        }
    payload["execution_permission"] = "PAPER_ONLY"
    payload["real_trade_authorized"] = False
    payload["current_new_entry_allowed"] = False
    scan_metadata = _base_payload("scan", current, day)
    payload.setdefault("scheduled_at", scan_metadata["scheduled_at"])
    payload.setdefault("started_at", scan_metadata["started_at"])
    _finalize_payload(root, payload)
    latest = root / "data" / "c2a_results" / "fast_latest.json"
    archive = root / "data" / "c2a_results" / "intraday" / f"{day.isoformat()}.json"
    _write_json_atomic(payload, latest)
    _write_json_atomic(payload, archive)
    markdown = _scan_markdown(payload)
    _write_phase_artifacts(root, "scan", payload, markdown)
    return payload


def run_review_phase(
    project_root: str | Path,
    *,
    now: datetime | None = None,
    trade_day: date | None = None,
    enforce_time: bool = True,
) -> dict:
    current = now or datetime.now(SHANGHAI)
    day = trade_day or current.date()
    root = Path(project_root).resolve()
    payload = _base_payload("review", current, day)
    payload["as_of"] = day.isoformat()
    try:
        if enforce_time:
            _require_phase_window("review", current)
        scan = _load_scan_for_day(root, day)
        if scan is None:
            raise FastScanDataError("没有当日早盘扫描记录；未沿用旧信号")
        if scan.get("status") == "DATA_NOT_READY":
            raise FastScanDataError(scan.get("reason") or "当日早盘扫描数据未就绪")
        market_close = _market_close_snapshot(day)
        morning_signal = _review_morning_signal(scan, day)
        if morning_signal["status"] == "DATA_NOT_READY":
            raise FastScanDataError(morning_signal.get("reason") or "当日收盘对账失败")
        payload.update(
            {
                "status": "READY",
                "market_close_check": market_close,
                "morning_signal": morning_signal,
                "promotion_gate": "FAIL",
                "retryable": False,
            }
        )
    except (FastScanDataError, RuntimeError, ValueError, OSError) as error:
        payload.update(
            {
                "status": "DATA_NOT_READY",
                "market_close_check": {
                    "status": "DATA_NOT_READY",
                    "quote_time": None,
                    "reason": _safe_error(error),
                },
                "morning_signal": {
                    "status": "DATA_NOT_READY",
                    "reason": _safe_error(error),
                    "entries": [],
                },
                "promotion_gate": "FAIL",
                "retryable": "错误时点触发" not in str(error),
            }
        )
    _write_phase_artifacts(root, "review", payload, _review_markdown(payload))
    return payload


def run_research_phase(
    project_root: str | Path,
    *,
    now: datetime | None = None,
    trade_day: date | None = None,
    optimize: bool | None = None,
    enforce_time: bool = True,
) -> dict:
    current = now or datetime.now(SHANGHAI)
    day = trade_day or current.date()
    root = Path(project_root).resolve()
    host, remote_root = _remote_settings()
    payload = _base_payload("research", current, day)
    payload["as_of"] = day.isoformat()
    should_optimize = day.weekday() == 4 if optimize is None else optimize
    try:
        if enforce_time:
            _require_phase_window("research", current)
        market_close = _market_close_snapshot(day)
    except (FastScanDataError, RuntimeError, ValueError, OSError) as error:
        payload.update(
            {
                "status": "DATA_NOT_READY",
                "market_close_check": {
                    "status": "DATA_NOT_READY",
                    "quote_time": None,
                    "reason": _safe_error(error),
                },
                "research_pipeline": {
                    "status": "DATA_NOT_READY",
                    "optimized": should_optimize,
                    "reason": _safe_error(error),
                },
                "next_session_baseline": {
                    "status": "DATA_NOT_READY",
                    "baseline_as_of": None,
                    "reason": "未在有效A股收盘后启动研究流水线",
                },
                "promotion_gate": "FAIL",
                "retryable": "错误时点触发" not in str(error),
            }
        )
        _write_phase_artifacts(root, "research", payload, _research_markdown(payload))
        return payload

    research_status = "COMPLETED"
    research_reason = None
    try:
        run_remote_pipeline(
            root,
            (day - timedelta(days=45)).isoformat(),
            day.isoformat(),
            optimize=should_optimize,
            variant="v1.2-challenger",
            host=host,
            remote_root=remote_root,
        )
    except (RuntimeError, ValueError, OSError) as error:
        research_status = "DATA_NOT_READY"
        research_reason = _safe_error(error)

    next_baseline_status = "READY"
    next_baseline_as_of = None
    next_baseline_reason = None
    try:
        export_remote_fast_pack(
            root,
            host=host,
            remote_root=remote_root,
            sync_code=research_status != "COMPLETED",
        )
        fetch_remote_fast_pack(root, host=host, remote_root=remote_root)
        next_baseline_as_of = load_fast_pack(root / "data" / "c2a_fast").last_processed_date.date()
        next_baseline_as_of = next_baseline_as_of.isoformat()
        if next_baseline_as_of != day.isoformat():
            next_baseline_status = "DATA_NOT_READY"
            next_baseline_reason = f"盘后基线截止{next_baseline_as_of}，尚未到{day.isoformat()}"
    except (FastScanDataError, RuntimeError, ValueError, OSError) as error:
        next_baseline_status = "DATA_NOT_READY"
        next_baseline_reason = _safe_error(error)

    components = (research_status, next_baseline_status)
    status = "READY" if components == ("COMPLETED", "READY") else "PARTIAL"
    if all(item == "DATA_NOT_READY" for item in components):
        status = "DATA_NOT_READY"
    payload.update(
        {
            "status": status,
            "market_close_check": market_close,
            "research_pipeline": {
                "status": research_status,
                "optimized": should_optimize,
                "reason": research_reason,
            },
            "next_session_baseline": {
                "status": next_baseline_status,
                "baseline_as_of": next_baseline_as_of,
                "reason": next_baseline_reason,
            },
            "promotion_gate": "FAIL",
            "retryable": status != "READY",
        }
    )
    markdown = _research_markdown(payload)
    _write_phase_artifacts(root, "research", payload, markdown)
    return payload


def _load_scan_for_day(root: Path, day: date) -> dict | None:
    candidates = (
        root / "data" / "c2a_results" / "intraday" / f"{day.isoformat()}.json",
        root / "data" / "c2a_results" / "fast_latest.json",
    )
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if payload.get("as_of") == day.isoformat():
            return payload
    return None


def _validated_close_quote(quote: object, ticker: str, day: date) -> tuple[float, str]:
    try:
        quote_time = str(quote["quote_time"])  # type: ignore[index]
        timestamp = datetime.strptime(quote_time[:14], "%Y%m%d%H%M%S")
        price = float(quote["price"])  # type: ignore[index]
    except (KeyError, TypeError, ValueError) as error:
        raise FastScanDataError(f"{ticker}收盘快照字段无效") from error
    if len(quote_time) < 14 or not quote_time.isdigit():
        raise FastScanDataError(f"{ticker}收盘快照时间无效")
    if timestamp.date() != day:
        raise FastScanDataError(f"{ticker}收盘快照日期不是{day.isoformat()}")
    if timestamp.time() < time(15, 0):
        raise FastScanDataError(f"{ticker}行情时间早于15:00，不作为收盘快照")
    if not math.isfinite(price) or price <= 0:
        raise FastScanDataError(f"{ticker}收盘价无效")
    return price, quote_time


def _market_close_snapshot(day: date) -> dict:
    ticker = "600000"
    quotes = fetch_quotes_fast([ticker])
    if ticker not in quotes.index:
        raise FastScanDataError("市场收盘校验缺少基准股票600000")
    _, quote_time = _validated_close_quote(quotes.loc[ticker], ticker, day)
    return {
        "status": "READY",
        "benchmark": ticker,
        "quote_time": quote_time,
        "reason": None,
    }


def _review_morning_signal(scan: dict | None, day: date) -> dict:
    if scan is None:
        return {
            "status": "DATA_NOT_READY",
            "reason": "没有当日早盘扫描记录；未沿用旧信号",
            "entries": [],
        }
    entries = scan.get("entries", [])
    if not entries:
        return {
            "status": "NO_ENTRY" if scan.get("status") != "DATA_NOT_READY" else "DATA_NOT_READY",
            "reason": scan.get("reason"),
            "scan_status": scan.get("status"),
            "entries": [],
        }
    tickers = [str(item["ticker"]).zfill(6) for item in entries]
    try:
        quotes = fetch_quotes_fast(tickers)
    except (FastScanDataError, RuntimeError, ValueError, OSError) as error:
        return {
            "status": "DATA_NOT_READY",
            "reason": _safe_error(error),
            "scan_status": scan.get("status"),
            "entries": [],
        }
    reviewed = []
    for item in entries:
        ticker = str(item["ticker"]).zfill(6)
        if ticker not in quotes.index:
            return {
                "status": "DATA_NOT_READY",
                "reason": f"收盘快照缺少{ticker}",
                "scan_status": scan.get("status"),
                "entries": [],
            }
        try:
            close_price, quote_time = _validated_close_quote(quotes.loc[ticker], ticker, day)
        except FastScanDataError as error:
            return {
                "status": "DATA_NOT_READY",
                "reason": _safe_error(error),
                "scan_status": scan.get("status"),
                "entries": [],
            }
        fill_price = float(item["simulated_fill_price"])
        shares = int(item["shares"])
        reviewed.append(
            {
                **item,
                "close_price": close_price,
                "quote_time": quote_time,
                "gross_mark_to_close_return": close_price / fill_price - 1.0,
                "gross_mark_to_close_pnl": (close_price - fill_price) * shares,
                "exit_status": "NEXT_TRADING_DAY_OPEN_PENDING",
            }
        )
    return {
        "status": "COMPLETED",
        "reason": None,
        "scan_status": scan.get("status"),
        "signal_status": scan.get("signal_status"),
        "entries": reviewed,
    }


def _prepare_markdown(payload: dict) -> str:
    return f"""# C2-A 盘前准备

- 日期：{payload["as_of"]}
- 状态：**{payload["status"]}**
- 基线截止：{payload.get("baseline_as_of") or "缺失"}
- 基线覆盖：{payload.get("baseline_coverage", 0):.2%}
- 原因：{payload.get("reason") or "无"}
- 权限：**PAPER_ONLY；不连接券商**
"""


def _scan_markdown(payload: dict) -> str:
    if payload.get("status") == "DATA_NOT_READY":
        detail = f"- 原因：{payload.get('reason') or '核心数据不完整'}"
    else:
        detail = "```text\n" + render_scan_result(payload) + "\n```"
    return f"""# C2-A 10:03 早盘模拟扫描

- 日期：{payload["as_of"]}
- 扫描截止：{payload.get("cutoff", "10:00")}
- 信号状态：**{payload.get("signal_status", payload.get("status"))}**
- 数据状态：**{payload.get("data_status", "DATA_NOT_READY")}**
- 生成时间：{payload.get("generated_at")}
- 当前追入权限：**无**
- 权限：**PAPER_ONLY；不连接券商**

{detail}
"""


def _review_markdown(payload: dict) -> str:
    morning = payload["morning_signal"]
    rows = []
    for item in morning.get("entries", []):
        rows.append(
            "| {ticker} | {name} | {fill:.2f} | {close:.2f} | {ret:.2%} | {pnl:.2f} |".format(
                ticker=item["ticker"],
                name=item.get("name", ""),
                fill=item["simulated_fill_price"],
                close=item["close_price"],
                ret=item["gross_mark_to_close_return"],
                pnl=item["gross_mark_to_close_pnl"],
            )
        )
    table = (
        "\n".join(
            [
                "| 代码 | 名称 | 模拟成交价 | 收盘价 | 浮动毛收益率 | 浮动毛盈亏 |",
                "|---|---|---:|---:|---:|---:|",
                *rows,
            ]
        )
        if rows
        else "无可复盘的当日模拟入场。"
    )
    market = payload["market_close_check"]
    return f"""# C2-A 16:30 盘后复盘

- 日期：{payload["as_of"]}
- 总体状态：**{payload["status"]}**
- 收盘校验：**{market["status"]}**；{market.get("quote_time") or market.get("reason") or "缺失"}
- 早盘对账：**{morning["status"]}**；{morning.get("reason") or "已按当日收盘快照核对"}
- 模型晋级门槛：**FAIL**
- 权限：**PAPER_ONLY；不连接券商**

盘中浮动结果不等于策略最终收益；C2-A 的模拟退出仍需等待下一交易日开盘。

{table}
"""


def _research_markdown(payload: dict) -> str:
    market = payload["market_close_check"]
    research = payload["research_pipeline"]
    baseline = payload["next_session_baseline"]
    return f"""# C2-A 16:35 盘后研究

- 日期：{payload["as_of"]}
- 总体状态：**{payload["status"]}**
- 收盘校验：**{market["status"]}**；{market.get("quote_time") or market.get("reason") or "缺失"}
- 研究流水线：**{research["status"]}**；{research.get("reason") or "完成"}
- 下一交易日基线：**{baseline["status"]}**；截止 {baseline.get("baseline_as_of") or "缺失"}
- 模型晋级门槛：**FAIL**
- 权限：**PAPER_ONLY；不连接券商**

该阶段用于更新研究结果与下一交易日基线，不替代 16:30 的当日模拟信号复盘。
"""


def record_phase_failure(
    project_root: str | Path,
    phase: str,
    reason: str,
    *,
    now: datetime | None = None,
    trade_day: date | None = None,
) -> dict:
    """把调度、Git 或解释器前置失败落成当日公开安全状态。"""

    if phase not in PHASES:
        raise ValueError(f"未知阶段: {phase}")
    current = now or datetime.now(SHANGHAI)
    day = trade_day or current.date()
    root = Path(project_root).resolve()
    safe_reason = _safe_error(RuntimeError(reason))
    payload = _base_payload(phase, current, day)
    payload["as_of"] = day.isoformat()
    payload["status"] = "DATA_NOT_READY"
    payload["reason"] = safe_reason
    payload["promotion_gate"] = "FAIL"
    payload["retryable"] = True

    if phase == "prepare":
        payload.update(
            {
                "baseline_as_of": None,
                "updated_trading_days": 0,
                "baseline_coverage": 0.0,
            }
        )
        markdown = _prepare_markdown(payload)
    elif phase == "scan":
        payload.update(
            {
                "signal_status": "NO_SIMULATED_SIGNAL",
                "data_status": "DATA_NOT_READY",
                "cutoff": DEFAULT_SCAN_CUTOFF.strftime("%H:%M"),
                "entries": [],
                "watchlist": [],
                "entry_count": 0,
                "candidate_count": 0,
            }
        )
        _finalize_payload(root, payload)
        latest = root / "data" / "c2a_results" / "fast_latest.json"
        archive = root / "data" / "c2a_results" / "intraday" / f"{day.isoformat()}.json"
        _write_json_atomic(payload, latest)
        _write_json_atomic(payload, archive)
        markdown = _scan_markdown(payload)
    elif phase == "review":
        payload.update(
            {
                "market_close_check": {
                    "status": "DATA_NOT_READY",
                    "quote_time": None,
                    "reason": safe_reason,
                },
                "morning_signal": {
                    "status": "DATA_NOT_READY",
                    "reason": safe_reason,
                    "entries": [],
                },
            }
        )
        markdown = _review_markdown(payload)
    else:
        payload.update(
            {
                "market_close_check": {
                    "status": "DATA_NOT_READY",
                    "quote_time": None,
                    "reason": safe_reason,
                },
                "research_pipeline": {
                    "status": "DATA_NOT_READY",
                    "optimized": False,
                    "reason": safe_reason,
                },
                "next_session_baseline": {
                    "status": "DATA_NOT_READY",
                    "baseline_as_of": None,
                    "reason": safe_reason,
                },
            }
        )
        markdown = _research_markdown(payload)

    _write_phase_artifacts(root, phase, payload, markdown)
    return payload


def _parse_clock(value: str) -> time:
    try:
        return time.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("时间必须是 HH:MM") from error


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m robot_quant.c2a_cloud")
    parser.add_argument("phase", choices=(*PHASES, "failure"))
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--date", type=date.fromisoformat)
    parser.add_argument("--cutoff", type=_parse_clock, default=DEFAULT_SCAN_CUTOFF)
    parser.add_argument("--no-optimize", action="store_true")
    parser.add_argument("--allow-off-schedule", action="store_true")
    parser.add_argument("--service-mode", action="store_true")
    parser.add_argument("--failed-phase", choices=PHASES)
    parser.add_argument("--reason")
    args = parser.parse_args(argv)
    now = datetime.now(SHANGHAI)
    if args.phase == "failure":
        if not args.failed_phase or not args.reason:
            parser.error("failure 必须提供 --failed-phase 和 --reason")
        payload = record_phase_failure(
            args.project_root,
            args.failed_phase,
            args.reason,
            now=now,
            trade_day=args.date,
        )
    elif args.phase == "prepare":
        payload = run_prepare_phase(
            args.project_root,
            now=now,
            enforce_time=not args.allow_off_schedule,
        )
    elif args.phase == "scan":
        payload = run_scan_phase(
            args.project_root,
            now=now,
            trade_day=args.date,
            cutoff=args.cutoff,
            enforce_time=not args.allow_off_schedule,
        )
    elif args.phase == "review":
        payload = run_review_phase(
            args.project_root,
            now=now,
            trade_day=args.date,
            enforce_time=not args.allow_off_schedule,
        )
    else:
        payload = run_research_phase(
            args.project_root,
            now=now,
            trade_day=args.date,
            optimize=False if args.no_optimize else None,
            enforce_time=not args.allow_off_schedule,
        )
    print(json.dumps(payload, ensure_ascii=False))
    if args.service_mode and payload.get("retryable"):
        raise SystemExit(75)


if __name__ == "__main__":
    main()
