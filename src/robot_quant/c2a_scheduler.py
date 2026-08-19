"""供 PyCharm 长驻运行的 C2-A 日更调度器。"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from robot_quant.c2a_remote import run_remote_pipeline


def run_scheduler(
    project_root: str | Path,
    run_at: str = "16:30",
    poll_seconds: int = 60,
    *,
    run_on_start: bool = False,
) -> None:
    """交易日收盘后更新；进程退出即停止，不修改系统 LaunchAgent。"""

    root = Path(project_root).resolve()
    hour, minute = _parse_time(run_at)
    timezone = ZoneInfo("Asia/Shanghai")
    last_attempted = None
    if run_on_start:
        _run_once(root)
        last_attempted = datetime.now(timezone).date()
    print(f"C2-A 调度器已启动：工作日 {run_at} Asia/Shanghai；Ctrl+C 停止")
    while True:
        now = datetime.now(timezone)
        should_run = (
            now.weekday() < 5
            and (now.hour, now.minute) >= (hour, minute)
            and last_attempted != now.date()
        )
        if should_run:
            try:
                _run_once(root)
            except Exception as error:  # noqa: BLE001 - 长驻调度器必须保留下一交易日重试能力
                print(f"C2-A 日更失败：{error}")
            finally:
                last_attempted = now.date()
        time.sleep(poll_seconds)


def _run_once(root: Path) -> None:
    today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    print(f"开始 C2-A 日更：{today.isoformat()}")
    full_optimization = today.weekday() == 4
    start = today - timedelta(days=45)
    run_remote_pipeline(
        root,
        start.isoformat(),
        today.isoformat(),
        optimize=full_optimization,
        variant="v1.2-challenger",
    )
    mode = "完整走样本外优化" if full_optimization else "基准路径+沿用上次参数排名"
    print(f"C2-A 日更完成：{mode}")


def _parse_time(value: str) -> tuple[int, int]:
    try:
        hour_text, minute_text = value.split(":", maxsplit=1)
        hour = int(hour_text)
        minute = int(minute_text)
    except (ValueError, TypeError) as error:
        raise ValueError("--at 必须使用 HH:MM") from error
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ValueError("--at 必须是有效的24小时时间")
    return hour, minute


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m robot_quant.c2a_scheduler")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--at", default="16:30")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--run-on-start", action="store_true")
    args = parser.parse_args()
    run_scheduler(
        args.project_root,
        args.at,
        args.poll_seconds,
        run_on_start=args.run_on_start,
    )


if __name__ == "__main__":
    main()
