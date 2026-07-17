"""命令行入口。"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from robot_quant.runner import run_daily


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="robot-quant")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run-daily", help="更新预测与模拟账户")
    run_parser.add_argument("--offline-data-dir")
    run_parser.add_argument("--output-root", default=".")
    args = parser.parse_args(argv)

    if args.command == "run-daily":
        state = run_daily(
            offline_data_dir=args.offline_data_dir,
            output_root=args.output_root,
        )
        print(
            f"{state['market_date']} | 预测概率 {state['prediction_probability']:.2%} | "
            f"目标仓位 {state['next_target_weight']:.0%}"
        )
