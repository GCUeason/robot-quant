"""无第三方依赖地记录 C2-A 云端前置失败。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo


SHANGHAI = ZoneInfo("Asia/Shanghai")
PHASES = ("prepare", "scan", "review", "research")
PHASE_SCHEDULES = {
    "prepare": "08:45:00",
    "scan": "10:02:30",
    "review": "16:30:00",
    "research": "16:35:00",
}


def _safe_reason(value: str) -> str:
    message = value.replace("\n", " ").strip()
    message = re.sub(r"/(?:Users|home)/[^\s:]+", "<private-path>", message)
    return message[:500] or "云端阶段前置失败"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _source_commit(root: Path) -> str | None:
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


def record_failure(root: Path, phase: str, trade_day: date, reason: str) -> dict:
    safe_reason = _safe_reason(reason)
    now = datetime.now(SHANGHAI)
    payload = {
        "phase": phase.upper(),
        "as_of": trade_day.isoformat(),
        "generated_at": now.isoformat(),
        "scheduled_at": f"{trade_day.isoformat()}T{PHASE_SCHEDULES[phase]}+08:00",
        "started_at": now.isoformat(),
        "status": "DATA_NOT_READY",
        "reason": safe_reason,
        "execution_permission": "PAPER_ONLY",
        "real_trade_authorized": False,
        "current_new_entry_allowed": False,
        "promotion_gate": "FAIL",
        "retryable": True,
        "source_commit": _source_commit(root),
        "source_manifest_sha256": None,
    }
    if phase == "prepare":
        payload.update(
            {
                "baseline_as_of": None,
                "updated_trading_days": 0,
                "baseline_coverage": 0.0,
            }
        )
    elif phase == "scan":
        payload.update(
            {
                "signal_status": "NO_SIMULATED_SIGNAL",
                "data_status": "DATA_NOT_READY",
                "cutoff": "10:00",
                "entries": [],
                "watchlist": [],
                "entry_count": 0,
                "candidate_count": 0,
            }
        )
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
    payload["finished_at"] = datetime.now(SHANGHAI).isoformat()
    hash_input = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload["payload_sha256"] = hashlib.sha256(hash_input).hexdigest()
    markdown = (
        f"# C2-A {phase} 状态\n\n"
        f"- 日期：{trade_day.isoformat()}\n"
        "- 状态：**DATA_NOT_READY**\n"
        f"- 原因：{safe_reason}\n"
        "- 权限：**PAPER_ONLY；不连接券商**\n"
    )
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    data_root = root / "data" / "c2a_results"
    _write(data_root / f"cloud_{phase}_latest.json", encoded)
    _write(data_root / phase / f"{trade_day.isoformat()}.json", encoded)
    _write(root / "reports" / f"c2a_{phase}_latest.md", markdown)
    _write(root / "reports" / "c2a" / f"{trade_day.isoformat()}-{phase}.md", markdown)
    if phase == "scan":
        _write(data_root / "fast_latest.json", encoded)
        _write(data_root / "intraday" / f"{trade_day.isoformat()}.json", encoded)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument("--date", required=True, type=date.fromisoformat)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    payload = record_failure(
        Path(args.project_root).resolve(),
        args.phase,
        args.date,
        args.reason,
    )
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
