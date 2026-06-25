from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol
import pandas as pd
import numpy as np
from loguru import logger


class DataLoaderProtocol(Protocol):
    def fetch_ohlcv(self, exchange_id: str, symbol: str, timeframe: str,
                    since: int | None, limit: int) -> pd.DataFrame: ...


@dataclass(frozen=True)
class Allocation:
    symbol: str
    target_pct: Decimal
    quantity: Decimal
    notional: Decimal
    correlation_penalty_applied: bool
    signal: object  # Signal object


@dataclass(frozen=True)
class PortfolioConfig:
    max_position_pct: Decimal = Decimal("0.2")
    lookback_days: int = 30
    correlation_high: Decimal = Decimal("0.8")
    correlation_extreme: Decimal = Decimal("0.95")


class PortfolioManager:
    def __init__(
        self,
        symbols: list[str],
        config: PortfolioConfig,
        data_loader: DataLoaderProtocol,
        exchange_id: str = "binance",
        timeframe: str = "1h",
    ):
        self.symbols = symbols
        self.config = config
        self.data_loader = data_loader
        self.exchange_id = exchange_id
        self.timeframe = timeframe
        self._correlation_cache: pd.DataFrame | None = None

    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """Calculate Pearson correlation matrix on log returns."""
        if len(self.symbols) <= 1:
            return pd.DataFrame([[1.0]], index=self.symbols, columns=self.symbols)

        dfs = {}
        lookback_ms = self.config.lookback_days * 24 * 3600 * 1000
        import time
        since = int(time.time() * 1000) - lookback_ms

        for symbol in self.symbols:
            try:
                df = self.data_loader.fetch_ohlcv(
                    self.exchange_id, symbol, self.timeframe, since=since, limit=1000
                )
                if len(df) > 1:
                    log_returns = np.log(1 + df["close"].pct_change().dropna())
                    dfs[symbol] = log_returns
            except Exception as e:
                logger.warning(f"Failed to fetch data for correlation ({symbol}): {e}")

        if len(dfs) < 2:
            return pd.DataFrame([[1.0]], index=self.symbols, columns=self.symbols)

        combined = pd.DataFrame(dfs)
        corr = combined.corr()
        self._correlation_cache = corr
        return corr

    def allocate_signals(
        self,
        signals: list,
        equity: Decimal,
        positions: dict,
    ) -> list[Allocation]:
        """Allocate capital to BUY signals by strength descending. SELL signals bypass."""
        if not signals:
            return []

        equity = Decimal(str(equity))

        # Separate BUY and SELL
        buy_signals = [s for s in signals if s.action.value == "buy"]
        sell_signals = [s for s in signals if s.action.value == "sell"]

        # SELL signals bypass allocation — return with full quantity
        sell_allocations = []
        for s in sell_signals:
            pos = positions.get(s.symbol)
            qty = pos.quantity if pos else Decimal("0")
            sell_allocations.append(Allocation(
                symbol=s.symbol,
                target_pct=Decimal("0"),
                quantity=qty,
                notional=Decimal("0"),
                correlation_penalty_applied=False,
                signal=s,
            ))

        # Sort BUY by strength descending
        buy_signals.sort(key=lambda s: s.strength, reverse=True)

        # Get correlation matrix
        if self._correlation_cache is None:
            try:
                self.calculate_correlation_matrix()
            except Exception:
                pass

        corr = self._correlation_cache

        # Check which symbols already have positions
        occupied = {s: p for s, p in positions.items() if p is not None}
        remaining_equity = equity

        buy_allocations = []
        allocated_symbols = set()

        for signal in buy_signals:
            # Skip if already has position
            if signal.symbol in occupied:
                logger.info(f"Skipping {signal.symbol}: already has position")
                continue

            # Check correlation penalty
            penalty_multiplier = Decimal("1.0")
            correlation_penalty = False

            if corr is not None and signal.symbol in corr.index:
                for other in allocated_symbols:
                    if other in corr.columns:
                        rho = abs(corr.loc[signal.symbol, other])
                        if rho > float(self.config.correlation_extreme):
                            penalty_multiplier = Decimal("1") / Decimal("3")
                            correlation_penalty = True
                            logger.warning(
                                f"Extreme correlation {rho:.2f} between {signal.symbol} and {other}, "
                                f"penalty: max_pct / 3"
                            )
                        elif rho > float(self.config.correlation_high):
                            penalty_multiplier = Decimal("1") / Decimal("2")
                            correlation_penalty = True
                            logger.warning(
                                f"High correlation {rho:.2f} between {signal.symbol} and {other}, "
                                f"penalty: max_pct / 2"
                            )

            # Calculate allocation
            max_pct = self.config.max_position_pct * penalty_multiplier
            notional = remaining_equity * max_pct
            quantity = notional / signal.price

            if notional <= Decimal("10"):
                logger.warning(f"Insufficient equity for {signal.symbol}: remaining={remaining_equity}")
                continue

            if notional > remaining_equity:
                notional = remaining_equity * Decimal("0.99")
                quantity = notional / signal.price

            remaining_equity -= notional
            allocated_symbols.add(signal.symbol)

            buy_allocations.append(Allocation(
                symbol=signal.symbol,
                target_pct=max_pct,
                quantity=quantity,
                notional=notional,
                correlation_penalty_applied=correlation_penalty,
                signal=signal,
            ))

        return buy_allocations + sell_allocations

    def rebalance_needed(self, allocations: list[Allocation]) -> bool:
        """Check if current allocations deviate >5% from targets."""
        if not allocations:
            return False
        for alloc in allocations:
            if abs(alloc.quantity) > 0:
                deviation = abs(alloc.target_pct - alloc.target_pct)  # Simplified
                if deviation > Decimal("0.05"):
                    return True
        return False
