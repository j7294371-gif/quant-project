#!/usr/bin/env python3
"""Quant Project — CLI entry point for backtest/paper/live trading modes."""

import argparse
import signal
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv

from src.utils.config import AppConfig
from src.utils.logging import setup_logging
from src.utils.state import StateStore
from src.strategy.registry import get_strategy, STRATEGY_REGISTRY

# Import all strategy modules to trigger @register_strategy decorators
import src.strategy.sma_cross  # noqa: F401
import src.strategy.macd  # noqa: F401
import src.strategy.rsi  # noqa: F401
import src.strategy.bollinger  # noqa: F401
import src.strategy.fusion  # noqa: F401


def date_to_utc_ms(date_str: str) -> int:
    """Convert YYYY-MM-DD to UTC milliseconds."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def parse_args() -> argparse.Namespace:
    """Build argparse CLI with 5 subcommands."""
    parser = argparse.ArgumentParser(
        prog="quant",
        description="Quantitative Trading System — multi-strategy backtest/paper/live",
    )
    sub = parser.add_subparsers(dest="command", help="Trading mode")

    # --- backtest ---
    bt = sub.add_parser("backtest", help="Historical backtest")
    bt.add_argument("--config-dir", default="./config", help="Config directory")
    bt.add_argument("--strategy", required=True, help="Strategy name (e.g. sma_cross)")
    bt.add_argument("--symbol", required=True, help="Trading pair (e.g. BTC/USDT)")
    bt.add_argument("--timeframe", default="1h", help="Kline interval")
    bt.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    bt.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    bt.add_argument("--initial-equity", type=float, default=10000, help="Initial equity USDT")
    bt.add_argument("--output", default=None, help="HTML report output path")
    bt.add_argument("--csv-data", default=None, help="Local CSV data path (skip CCXT download)")
    # OOS parameters (mutually exclusive with --start/--end)
    bt.add_argument("--train-start", default=None, help="OOS train start YYYY-MM-DD")
    bt.add_argument("--train-end", default=None, help="OOS train end YYYY-MM-DD")
    bt.add_argument("--test-start", default=None, help="OOS test start YYYY-MM-DD")
    bt.add_argument("--test-end", default=None, help="OOS test end YYYY-MM-DD")

    # --- paper ---
    pp = sub.add_parser("paper", help="Paper trading (simulated)")
    pp.add_argument("--config-dir", default="./config", help="Config directory")
    pp.add_argument("--strategy", required=True, help="Strategy name")
    pp.add_argument("--symbols", nargs="+", default=None, help="Trading pairs")
    pp.add_argument("--timeframe", default=None, help="Kline interval")

    # --- live ---
    lv = sub.add_parser("live", help="Live trading (real money)")
    lv.add_argument("--config-dir", default="./config", help="Config directory")
    lv.add_argument("--strategy", required=True, help="Strategy name")
    lv.add_argument("--symbols", nargs="+", default=None, help="Trading pairs")
    lv.add_argument("--timeframe", default=None, help="Kline interval")
    lv.add_argument("--dry-run", action="store_true", default=False, help="Print signals only, don't trade")

    # --- circuit-reset ---
    cr = sub.add_parser("circuit-reset", help="Reset circuit breaker")
    cr.add_argument("--config-dir", default="./config", help="Config directory")

    # --- list-strategies ---
    sub.add_parser("list-strategies", help="List registered strategies")

    return parser.parse_args()


def _build_execution(args, config, state_store):
    """Factory: build execution engine based on command."""
    from src.execution.backtest import BacktestEngine
    from src.execution.paper import PaperEngine
    from src.execution.live import LiveEngine

    if args.command == "backtest":
        return BacktestEngine(initial_equity=Decimal(str(args.initial_equity)))
    elif args.command == "paper":
        return PaperEngine(state_store, initial_equity=Decimal("10000"))
    elif args.command == "live":
        return LiveEngine(
            exchange_id=config.exchange.exchange,
            api_key=config.api_key,
            api_secret=config.api_secret,
            testnet=config.exchange.testnet,
        )
    else:
        raise ValueError(f"Unknown command: {args.command}")


def _build_orchestrator(args, strategy, execution, config, state_store, logger):
    """Factory: build orchestrator based on command."""
    from src.risk.manager import RiskManager
    from src.risk.circuit import CircuitBreaker

    risk_manager = RiskManager(config.risk, state_store)
    circuit_breaker = risk_manager.circuit_breaker

    symbols = [args.symbol] if args.command == "backtest" else (
        args.symbols if args.symbols else config.exchange.symbols
    )

    if args.command == "backtest":
        from src.orchestrator.backtest import BacktestOrchestrator
        from src.execution.backtest import BacktestEngine, BacktestResult
        from src.data.loader import load_or_fetch
        import pandas as pd

        # Load data
        if args.csv_data:
            logger.info(f"Loading CSV data from {args.csv_data}")
            df = pd.read_csv(args.csv_data)
        else:
            since = date_to_utc_ms(args.start) if args.start else None
            df = load_or_fetch(
                exchange_id=config.exchange.exchange,
                symbol=args.symbol,
                timeframe=args.timeframe,
                since=since or int((datetime.now(timezone.utc).timestamp() - 365 * 86400) * 1000),
                cache_dir=config.app.data_cache_dir,
            )

        # Filter by date range
        if args.start:
            start_ms = date_to_utc_ms(args.start)
            df = df[df["timestamp"] >= start_ms]
        if args.end:
            end_ms = date_to_utc_ms(args.end)
            df = df[df["timestamp"] <= end_ms]

        df = df.reset_index(drop=True)
        logger.info(f"Loaded {len(df)} bars for {args.symbol}")

        if len(df) < strategy.min_bars:
            logger.error(f"Insufficient data: {len(df)} bars < min_bars ({strategy.min_bars})")
            sys.exit(1)

        # Check for OOS split
        has_oos = all([args.train_start, args.train_end, args.test_start, args.test_end])

        if has_oos:
            train_start_ms = date_to_utc_ms(args.train_start)
            train_end_ms = date_to_utc_ms(args.train_end)
            test_start_ms = date_to_utc_ms(args.test_start)
            test_end_ms = date_to_utc_ms(args.test_end)

            train_df = df[(df["timestamp"] >= train_start_ms) & (df["timestamp"] <= train_end_ms)].reset_index(drop=True)
            test_df = df[(df["timestamp"] >= test_start_ms) & (df["timestamp"] <= test_end_ms)].reset_index(drop=True)

            logger.info(f"OOS: train={len(train_df)} bars, test={len(test_df)} bars")

            initial_equity = Decimal(str(args.initial_equity))

            # Run training
            train_engine = BacktestEngine(initial_equity=initial_equity)
            train_orch = BacktestOrchestrator(
                strategy=strategy, execution=train_engine, symbols=symbols,
                risk_config=config.risk, state_store=state_store,
                circuit_breaker=circuit_breaker, logger=logger,
                data_df=train_df, initial_equity=initial_equity,
                risk_manager=risk_manager,
            )
            train_result = train_orch.run()

            # Run testing (fresh engine, same strategy)
            test_risk_manager = RiskManager(config.risk, state_store)
            test_engine = BacktestEngine(initial_equity=initial_equity)
            test_orch = BacktestOrchestrator(
                strategy=type(strategy)(strategy.params),  # fresh state
                execution=test_engine, symbols=symbols,
                risk_config=config.risk, state_store=state_store,
                circuit_breaker=CircuitBreaker(config.risk, state_store),
                logger=logger, data_df=test_df,
                initial_equity=initial_equity,
                risk_manager=test_risk_manager,
            )
            test_result = test_orch.run()

            # Populate OOS metadata
            test_result = BacktestResult(
                initial_equity=test_result.initial_equity,
                final_equity=test_result.final_equity,
                equity_curve=test_result.equity_curve,
                trades=test_result.trades,
                is_oos=True,
                train_result=train_result,
                oos_warning="",
            )

            # Compute OOS warnings
            from src.analytics.metrics import calculate_metrics
            train_metrics = calculate_metrics(train_result, timeframe=args.timeframe, risk_free_rate=config.app.analytics.get("risk_free_rate", 0.02))
            test_metrics = calculate_metrics(test_result, timeframe=args.timeframe, risk_free_rate=config.app.analytics.get("risk_free_rate", 0.02))

            oos_warnings = []
            if test_metrics.sharpe_ratio < train_metrics.sharpe_ratio * 0.3:
                oos_warnings.append(f"样本外衰减严重: test_sharpe={test_metrics.sharpe_ratio:.2f} < train_sharpe={train_metrics.sharpe_ratio:.2f}×0.3")
            if float(test_metrics.total_return) < 0:
                oos_warnings.append("样本外测试亏损，建议不实盘")
            if len(test_result.trades) < 10:
                oos_warnings.append(f"样本外交易次数不足 ({len(test_result.trades)} < 10)")

            test_result = BacktestResult(
                initial_equity=test_result.initial_equity,
                final_equity=test_result.final_equity,
                equity_curve=test_result.equity_curve,
                trades=test_result.trades,
                is_oos=True,
                train_result=train_result,
                oos_warning="; ".join(oos_warnings) if oos_warnings else "",
            )

            # Print side-by-side comparison
            from src.analytics.report import print_backtest_report
            logger.info("=" * 60)
            logger.info("  TRAINING SET RESULTS")
            logger.info("=" * 60)
            print_backtest_report(train_metrics, train_result.trades, train_orch.warnings if hasattr(train_orch, 'warnings') else [])
            logger.info("=" * 60)
            logger.info("  TEST SET RESULTS (OOS)")
            logger.info("=" * 60)
            print_backtest_report(test_metrics, test_result.trades, test_orch.warnings if hasattr(test_orch, 'warnings') else [])
            if oos_warnings:
                logger.warning("OOS WARNINGS: " + "; ".join(oos_warnings))

            return test_result

        else:
            return BacktestOrchestrator(
                strategy=strategy,
                execution=execution,
                symbols=symbols,
                risk_config=config.risk,
                state_store=state_store,
                circuit_breaker=circuit_breaker,
                logger=logger,
                data_df=df,
                initial_equity=Decimal(str(args.initial_equity)),
                risk_manager=risk_manager,
            )

    elif args.command == "paper":
        from src.orchestrator.paper import PaperOrchestrator
        from src.data.stream import OHLCVStream

        timeframe = args.timeframe or config.exchange.timeframe
        stream = OHLCVStream(
            exchange_id=config.exchange.exchange,
            symbols=symbols,
            timeframe=timeframe,
        )

        return PaperOrchestrator(
            strategy=strategy,
            execution=execution,
            symbols=symbols,
            risk_config=config.risk,
            state_store=state_store,
            circuit_breaker=circuit_breaker,
            logger=logger,
            stream=stream,
            risk_manager=risk_manager,
        )

    elif args.command == "live":
        from src.orchestrator.live import LiveOrchestrator
        from src.data.stream import OHLCVStream

        timeframe = args.timeframe or config.exchange.timeframe
        stream = OHLCVStream(
            exchange_id=config.exchange.exchange,
            symbols=symbols,
            timeframe=timeframe,
        )

        return LiveOrchestrator(
            strategy=strategy,
            execution=execution,
            symbols=symbols,
            risk_config=config.risk,
            state_store=state_store,
            circuit_breaker=circuit_breaker,
            logger=logger,
            stream=stream,
            dry_run=args.dry_run,
            risk_manager=risk_manager,
        )

    else:
        raise ValueError(f"Unknown command: {args.command}")


def _run_recovery(state_store, config, args, logger):
    """Execute §9.2 recovery flow for paper/live modes."""
    from datetime import datetime, timezone

    # Step 1: Check shutdown marker (already done in main)
    # Step 2: Freshness check
    for key in ["positions", "circuit", "daily_stats", "pending_orders"]:
        data = state_store.load(key)
        if data:
            updated_at = data.get("updated_at", 0)
            now_ms = int(time.time() * 1000)
            age_hours = (now_ms - updated_at) / 3600000
            if age_hours > 24:
                logger.error(f"State {key} is stale ({age_hours:.1f}h old)")
            elif age_hours > 1:
                logger.warning(f"State {key} is somewhat stale ({age_hours:.1f}h old)")

    # Step 3: Cross-day reset
    stats = state_store.load("daily_stats")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if stats is None or stats.get("date") != today:
        logger.info(f"New day: resetting daily_stats ({stats.get('date') if stats else 'None'} → {today})")
        state_store.save("daily_stats", {"date": today, "trade_count": 0, "daily_pnl": "0.00"})

    logger.info("Recovery flow complete")


def _register_signal_handlers(orchestrator_ref: dict, state_store: StateStore, logger):
    """Register SIGINT/SIGTERM handlers for graceful shutdown."""

    def graceful_exit(signum, frame):
        sig_name = signal.Signals(signum).name
        logger.warning(f"Received {sig_name}, shutting down gracefully...")

        orchestrator = orchestrator_ref.get("instance")
        if orchestrator:
            try:
                orchestrator.execution.cancel_all_orders()
            except Exception as e:
                logger.error(f"Failed to cancel orders: {e}")
            try:
                orchestrator.execution.close_all_positions(f"{sig_name}_shutdown")
            except Exception as e:
                logger.error(f"Failed to close positions: {e}")

        state_store.mark_shutdown("clean")
        state_store.close()
        logger.info("Graceful shutdown complete")
        sys.exit(0)

    signal.signal(signal.SIGINT, graceful_exit)
    signal.signal(signal.SIGTERM, graceful_exit)


def _validate_args(args):
    """Validate CLI argument constraints."""
    if args.command == "backtest":
        has_oos = all([args.train_start, args.train_end, args.test_start, args.test_end])
        has_simple = args.start is not None or args.end is not None
        has_partial_oos = any([args.train_start, args.train_end, args.test_start, args.test_end])

        if has_partial_oos and not has_oos:
            print("错误: OOS 参数必须同时提供 --train-start/--train-end/--test-start/--test-end", file=sys.stderr)
            sys.exit(1)
        if has_oos and has_simple:
            print("错误: OOS 参数与 --start/--end 互斥，请二选一", file=sys.stderr)
            sys.exit(1)
        if has_oos:
            train_end_ms = date_to_utc_ms(args.train_end)
            test_start_ms = date_to_utc_ms(args.test_start)
            if test_start_ms < train_end_ms:
                print("错误: --test-start 必须 >= --train-end（时间不可重叠）", file=sys.stderr)
                sys.exit(1)


def main():
    # === Phase 0: Parse ===
    args = parse_args()

    # Handle list-strategies without config load
    if args.command == "list-strategies":
        print("Registered strategies:")
        for name in sorted(STRATEGY_REGISTRY.keys()):
            cls = STRATEGY_REGISTRY[name]
            print(f"  {name:<15} — min_bars={cls({}).min_bars}")
        return

    # Handle circuit-reset
    if args.command == "circuit-reset":
        config = AppConfig.load(args.config_dir)
        state_store = StateStore(config.app.state_dir)
        from src.risk.circuit import CircuitBreaker
        cb = CircuitBreaker(config.risk, state_store)
        cb.reset()
        print("Circuit breaker has been reset.")
        state_store.close()
        return

    _validate_args(args)

    # === Phase 1: Load environment ===
    load_dotenv()
    config = AppConfig.load(args.config_dir)
    logger = setup_logging(
        level=config.app.log_level,
        log_dir=config.app.log_dir,
        retention_days=config.app.log_retention_days,
    )

    # === Phase 2: State recovery ===
    state_store = StateStore(config.app.state_dir)
    if not state_store.is_clean_shutdown():
        logger.warning("检测到非正常关闭，启用增强恢复")

    if args.command in ("paper", "live"):
        _run_recovery(state_store, config, args, logger)

    # Register signal handlers
    orchestrator_ref = {"instance": None}
    _register_signal_handlers(orchestrator_ref, state_store, logger)

    # === Phase 3: Build components ===
    try:
        strategy_name = getattr(args, "strategy", None) or config.strategy.active
        strategy = get_strategy(strategy_name, config.strategy.params)
        execution = _build_execution(args, config, state_store)
        orchestrator = _build_orchestrator(args, strategy, execution, config, state_store, logger)
        orchestrator_ref["instance"] = orchestrator
        state_store.mark_shutdown("running")
    except Exception as e:
        logger.critical(f"启动失败: {e}")
        sys.exit(1)

    # === Phase 4: Run ===
    exit_code = 0
    try:
        result = orchestrator.run()

        if args.command == "backtest" and result is not None:
            from src.analytics.metrics import calculate_metrics
            from src.analytics.report import print_backtest_report, generate_html_report

            metrics = calculate_metrics(
                result,
                timeframe=args.timeframe,
                benchmark_close_series=None,
                risk_free_rate=config.app.analytics.get("risk_free_rate", 0.02),
            )
            print_backtest_report(metrics, result.trades, orchestrator.warnings)

            if args.output:
                generate_html_report(metrics, result.trades, result.equity_curve, args.output)

    except KeyboardInterrupt:
        logger.info("收到 Ctrl+C，正在安全退出...")
    except Exception as e:
        logger.critical(f"运行时错误: {e}", exc_info=True)
        exit_code = 1
    finally:
        if orchestrator_ref["instance"]:
            try:
                orchestrator_ref["instance"].shutdown()
            except Exception:
                pass
        state_store.mark_shutdown("clean")
        state_store.close()
        logger.info("系统已安全退出")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
