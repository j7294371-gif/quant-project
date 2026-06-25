from decimal import Decimal
import pandas as pd

from src.strategy.base import BaseStrategy, Signal, SignalAction
from src.strategy.registry import register_strategy


@register_strategy("sma_cross")
class SmaCrossStrategy(BaseStrategy):
    def _validate_params(self) -> None:
        short_window = int(self.params.get("short_window", 5))
        long_window = int(self.params.get("long_window", 20))
        if short_window >= long_window:
            raise ValueError(
                f"short_window ({short_window}) must be less than long_window ({long_window})"
            )

    @property
    def min_bars(self) -> int:
        return int(self.params.get("long_window", 20)) + 1

    def on_bar(self, symbol: str, df: pd.DataFrame) -> Signal:
        short_window = int(self.params.get("short_window", 5))
        long_window = int(self.params.get("long_window", 20))

        sma_short = df["close"].rolling(window=short_window).mean()
        sma_long = df["close"].rolling(window=long_window).mean()

        sma_short_prev = sma_short.iloc[-2]
        sma_long_prev = sma_long.iloc[-2]
        sma_short_curr = sma_short.iloc[-1]
        sma_long_curr = sma_long.iloc[-1]

        price = Decimal(str(df["close"].iloc[-1]))
        timestamp = int(df.index[-1].timestamp() * 1000) if hasattr(df.index[-1], "timestamp") else int(df.index[-1])

        if pd.isna(sma_short_prev) or pd.isna(sma_long_prev) or pd.isna(sma_short_curr) or pd.isna(sma_long_curr):
            return Signal(
                action=SignalAction.HOLD,
                symbol=symbol,
                strength=0.0,
                price=price,
                timestamp=timestamp,
                reason="Insufficient data for SMA calculation",
            )

        # Golden cross
        if sma_short_prev <= sma_long_prev and sma_short_curr > sma_long_curr:
            diff = abs(sma_short_curr - sma_long_curr)
            strength = min(1.0, float(diff / sma_long_curr * 10))
            return Signal(
                action=SignalAction.BUY,
                symbol=symbol,
                strength=strength,
                price=price,
                timestamp=timestamp,
                reason=f"Golden cross: SMA_{short_window}={sma_short_curr:.4f} > SMA_{long_window}={sma_long_curr:.4f}",
            )

        # Death cross
        if sma_short_prev >= sma_long_prev and sma_short_curr < sma_long_curr:
            diff = abs(sma_short_curr - sma_long_curr)
            strength = min(1.0, float(diff / sma_long_curr * 10))
            return Signal(
                action=SignalAction.SELL,
                symbol=symbol,
                strength=strength,
                price=price,
                timestamp=timestamp,
                reason=f"Death cross: SMA_{short_window}={sma_short_curr:.4f} < SMA_{long_window}={sma_long_curr:.4f}",
            )

        return Signal(
            action=SignalAction.HOLD,
            symbol=symbol,
            strength=0.0,
            price=price,
            timestamp=timestamp,
            reason="No cross detected",
        )
