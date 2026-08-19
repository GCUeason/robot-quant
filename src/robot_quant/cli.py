"""命令行入口。"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from robot_quant.c2a_bigquant import configure_bigquant_api_key_from_clipboard
from robot_quant.c2a import C2AParameters
from robot_quant.c2a_cache import C2AResearchCache
from robot_quant.c2a_data import (
    C2ADataStore,
    configure_tushare_token_from_clipboard,
    import_c2a_csv,
)
from robot_quant.c2a_optimize import WalkForwardConfig
from robot_quant.c2a_runner import (
    run_c2a_audit,
    run_c2a_backtest,
    run_c2a_cache,
    run_c2a_bigquant_update,
    run_c2a_tushare_update,
)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="robot-quant")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run-daily", help="更新预测与模拟账户")
    run_parser.add_argument("--offline-data-dir")
    run_parser.add_argument("--output-root", default=".")
    chain_parser = subparsers.add_parser("run-robot-chain-paper", help="更新机器人产业链纸面账户")
    chain_parser.add_argument("--output-root", default=".")
    c2a_audit = subparsers.add_parser("c2a-audit", help="审计C2-A本地分钟数据是否可严格回测")
    _add_c2a_common_arguments(c2a_audit)
    c2a_backtest = subparsers.add_parser("c2a-backtest", help="运行C2-A基准和走样本外回测")
    _add_c2a_common_arguments(c2a_backtest)
    c2a_backtest.add_argument("--allow-proxy", action="store_true")
    c2a_backtest.add_argument("--no-optimize", action="store_true")
    c2a_backtest.add_argument("--max-candidates", type=int)
    c2a_backtest.add_argument("--min-train-days", type=int, default=60)
    c2a_backtest.add_argument("--test-days", type=int, default=20)
    c2a_backtest.add_argument("--embargo-days", type=int, default=1)
    c2a_backtest.add_argument("--min-train-trades", type=int, default=10)
    c2a_cache = subparsers.add_parser("c2a-build-cache", help="增量生成C2-A全市场研究缓存")
    _add_c2a_common_arguments(c2a_cache)
    c2a_update = subparsers.add_parser("c2a-update-tushare", help="通过已授权Tushare更新C2-A数据")
    c2a_update.add_argument("--data-root", default="data/c2a")
    c2a_update.add_argument("--start", default="2026-01-01")
    c2a_update.add_argument("--end")
    c2a_update.add_argument("--universe-only", action="store_true")
    c2a_bigquant = subparsers.add_parser(
        "c2a-update-bigquant", help="通过 BigQuant SDK 更新C2-A数据"
    )
    c2a_bigquant.add_argument("--data-root", default="data/c2a")
    c2a_bigquant.add_argument("--start", default="2026-01-01")
    c2a_bigquant.add_argument("--end")
    c2a_bigquant.add_argument("--universe-only", action="store_true")
    c2a_bigquant.add_argument(
        "--stream-cache",
        action="store_true",
        help="逐日校验全市场后只保存压缩研究缓存，适合 AIStudio 有限空间",
    )
    c2a_migrate = subparsers.add_parser(
        "c2a-migrate-cache-parquet",
        help="将 BigQuant 流式研究缓存校验迁移为 Parquet/Zstandard",
    )
    c2a_migrate.add_argument("--data-root", default="data/c2a")
    subparsers.add_parser(
        "c2a-configure-bigquant",
        help="从 macOS 剪贴板安全保存 BigQuant AK.SK",
    )
    subparsers.add_parser(
        "c2a-configure-tushare",
        help="从 macOS 剪贴板安全保存 Tushare Token",
    )
    c2a_import = subparsers.add_parser("c2a-import", help="导入供应商C2-A分钟和Universe CSV")
    c2a_import.add_argument("--data-root", default="data/c2a")
    c2a_import.add_argument("--minutes", required=True)
    c2a_import.add_argument("--universe", required=True)
    c2a_import.add_argument("--source", default="user_import")
    c2a_import.add_argument("--metadata-verified", action="store_true")
    c2a_import.add_argument("--full-market", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "run-daily":
        from robot_quant.runner import run_daily

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
        from robot_quant.runner import run_robot_chain_daily

        state = run_robot_chain_daily(output_root=args.output_root)
        print(
            f"{state['market_date']} | 产业链纸面账户市值 "
            f"¥{state['portfolio_value']:,.2f} | "
            f"累计盈亏 ¥{state['profit']:,.2f}"
        )
    elif args.command == "c2a-audit":
        audit = run_c2a_audit(
            data_root=args.data_root,
            output_root=args.output_root,
            start_date=args.start,
            end_date=args.end,
            variant=args.variant,
        )
        print(
            f"C2-A {audit['start_date']}~{audit['end_date']} | "
            f"数据状态 {audit['status']} | 原因 {len(audit['reasons'])} 项 | PAPER_ONLY"
        )
    elif args.command == "c2a-backtest":
        try:
            state = run_c2a_backtest(
                data_root=args.data_root,
                output_root=args.output_root,
                start_date=args.start,
                end_date=args.end,
                variant=args.variant,
                allow_proxy=args.allow_proxy,
                optimize=not args.no_optimize,
                max_candidates=args.max_candidates,
                walk_forward_config=WalkForwardConfig(
                    min_train_days=args.min_train_days,
                    test_days=args.test_days,
                    embargo_days=args.embargo_days,
                    min_train_trades=args.min_train_trades,
                ),
            )
        except RuntimeError as error:
            parser.exit(2, f"C2-A 回测未运行：{error}\n")
        print(
            f"C2-A {state['as_of']} | 基准净收益 "
            f"{state['baseline']['total_return']:.2%} | "
            f"OOS交易 {state['walk_forward'].get('oos_trade_count', 0)} | PAPER_ONLY"
        )
    elif args.command == "c2a-update-tushare":
        try:
            result = run_c2a_tushare_update(
                data_root=args.data_root,
                start_date=args.start,
                end_date=args.end,
                universe_only=args.universe_only,
            )
        except RuntimeError as error:
            parser.exit(2, f"C2-A 数据未更新：{error}\n")
        print(f"C2-A Tushare 更新完成 | {result}")
    elif args.command == "c2a-configure-tushare":
        try:
            path = configure_tushare_token_from_clipboard()
        except (RuntimeError, ValueError) as error:
            parser.exit(2, f"C2-A Tushare Token 未配置：{error}\n")
        print(f"C2-A Tushare Token 已安全保存至 {path}")
    elif args.command == "c2a-update-bigquant":
        try:
            result = run_c2a_bigquant_update(
                data_root=args.data_root,
                start_date=args.start,
                end_date=args.end,
                universe_only=args.universe_only,
                stream_cache=args.stream_cache,
            )
        except RuntimeError as error:
            parser.exit(2, f"C2-A BigQuant 数据未更新：{error}\n")
        print(f"C2-A BigQuant 更新完成 | {result}")
    elif args.command == "c2a-configure-bigquant":
        try:
            path = configure_bigquant_api_key_from_clipboard()
        except (RuntimeError, ValueError) as error:
            parser.exit(2, f"C2-A BigQuant API Key 未配置：{error}\n")
        print(f"C2-A BigQuant API Key 已安全保存至 {path}")
    elif args.command == "c2a-migrate-cache-parquet":
        store = C2ADataStore(args.data_root)
        cache = C2AResearchCache(
            store.root / "research_cache",
            C2AParameters.dynamic_snapshot(),
        )
        try:
            result = cache.migrate_stream_partitions_to_parquet()
        except RuntimeError as error:
            parser.exit(2, f"C2-A 缓存未迁移：{error}\n")
        print(f"C2-A 缓存迁移完成 | {result}")
    elif args.command == "c2a-build-cache":
        try:
            result = run_c2a_cache(
                data_root=args.data_root,
                start_date=args.start,
                end_date=args.end,
                variant=args.variant,
            )
        except RuntimeError as error:
            parser.exit(2, f"C2-A 缓存未生成：{error}\n")
        print(
            f"C2-A 缓存已更新至 {result['last_processed_date']} | "
            f"新增 {result['processed_days']} 个分区 | PAPER_ONLY"
        )
    elif args.command == "c2a-import":
        result = import_c2a_csv(
            C2ADataStore(args.data_root),
            args.minutes,
            args.universe,
            source=args.source,
            metadata_verified=args.metadata_verified,
            full_market=args.full_market,
        )
        print(f"C2-A 导入完成 | {result}")


def _add_c2a_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", default="data/c2a")
    parser.add_argument("--output-root", default=".")
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end")
    parser.add_argument(
        "--variant",
        choices=("v1", "v1.2", "v1.2-challenger"),
        default="v1.2",
    )
