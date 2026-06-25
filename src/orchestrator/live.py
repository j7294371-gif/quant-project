import time
import queue
from decimal import Decimal
from src.orchestrator.paper import PaperOrchestrator


class LiveOrchestrator(PaperOrchestrator):
    def __init__(
        self,
        strategy,
        execution,
        symbols: list[str],
        risk_config,
        state_store,
        circuit_breaker,
        logger,
        stream,
        dry_run: bool = False,
        risk_manager=None,
    ):
        super().__init__(
            strategy, execution, symbols, risk_config,
            state_store, circuit_breaker, logger, stream, risk_manager,
        )
        self.mode = "live"
        self.dry_run = dry_run

    def run(self):
        # Live mode: mandatory human confirmation
        print()
        print("=" * 60)
        print("  ⚠️  LIVE TRADING MODE — 实盘交易模式")
        print("=" * 60)
        print(f"  Strategy:  {self.strategy.__class__.__name__}")
        print(f"  Symbols:   {', '.join(self.symbols)}")
        print(f"  Dry Run:   {self.dry_run}")
        print("=" * 60)
        print()

        if not self.dry_run:
            confirm = input("输入 'yes' 确认开始实盘交易: ")
            if confirm.strip().lower() != "yes":
                self.logger.info("Live trading cancelled by user")
                return None

        self.logger.info(f"Live trading started: {self.symbols} (dry_run={self.dry_run})")
        return super().run()
