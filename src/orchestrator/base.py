from abc import ABC, abstractmethod
from decimal import Decimal
import pandas as pd
from src.strategy.base import Signal, SignalAction


class BaseOrchestrator(ABC):
    """Abstract base for all trading mode orchestrators."""

    def __init__(
        self,
        strategy,
        execution,
        symbols: list[str],
        risk_config,
        state_store,
        circuit_breaker,
        logger,
    ):
        self.strategy = strategy
        self.execution = execution
        self.symbols = symbols
        self.risk_config = risk_config
        self.state_store = state_store
        self.circuit = circuit_breaker
        self.logger = logger

    @abstractmethod
    def run(self):
        """Start main loop, return results."""
        ...

    def _process_bar(self, symbol: str, df: pd.DataFrame) -> Signal:
        """Core logic shared by all modes: strategy -> signal."""
        signal = self.strategy.on_bar(symbol, df)
        if signal.action == SignalAction.HOLD:
            return signal

        self.logger.info(
            f"[{symbol}] {signal.action.value.upper()} signal | "
            f"strength={signal.strength:.2f} | price={signal.price} | "
            f"reason={signal.reason}"
        )
        return signal
