"""命令行入口。"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from robot_quant.runner import run_daily, run_robot_chain_daily


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="robot-quant")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run-daily", help="更新预测与模拟账户")
    run_parser.add_argument("--offline-data-dir")
    run_parser.add_argument("--output-root", default=".")
    chain_parser = subparsers.add_parser("run-robot-chain-paper", help="更新机器人产业链纸面账户")
    chain_parser.add_argument("--output-root", default=".")
    args = parser.parse_args(argv)

    if args.command == "run-daily":
        state = run_daily(
            offline_data_dir=args.offline_data_dir,
            output_root=args.output_root,
        )
        print(
            f"{state['market_date']} | 10日相对研究概率 "
            f"{state['prediction_probability']:.2%} | "
            f"研究仓位 {state['research_target_weight']:.0%}（不可执行） | "
            f"固定定投仓位 {state['executable_target_weight']:.0%}"
        )
    elif args.command == "run-robot-chain-paper":
        state = run_robot_chain_daily(output_root=args.output_root)
        print(
            f"{state['market_date']} | 产业链纸面账户市值 "
            f"¥{state['portfolio_value']:,.2f} | "
            f"累计盈亏 ¥{state['profit']:,.2f}"
        )
