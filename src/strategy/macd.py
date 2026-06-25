from decimal import Decimal
import pandas as pd

from src.strategy.base import BaseStrategy, Signal, SignalAction
from src.strategy.registry import register_strategy


@register_strategy("macd")
class MacdStrategy(BaseStrategy):
    def _validate_params(self) -> None:
        fast = int(self.params.get("fast", 12))
        slow = int(self.params.get("slow", 26))
        signal_period = int(self.params.get("signal", 9))
        if fast >= slow:
            raise ValueError(
                f"fast ({fast}) must be less than slow ({slow})"
            )

    @property
    def min_bars(self) -> int:
        slow = int(self.params.get("slow", 26))
        signal_period = int(self.params.get("signal", 9))
        return slow + signal_period + 1

    @property
    def warmup_bars(self) -> int:
        slow = int(self.params.get("slow", 26))
        return slow * 2

    def on_bar(self, symbol: str, df: pd.DataFrame) -> Signal:
        fast = int(self.params.get("fast", 12))
        slow = int(self.params.get("slow", 26))
        signal_period = int(self.params.get("signal", 9))

        close = df["close"]

        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        hist_prev = histogram.iloc[-2]
        hist_curr = histogram.iloc[-1]
        close_curr = close.iloc[-1]

        price = Decimal(str(close_curr))
        timestamp = int(df.index[-1].timestamp() * 1000) if hasattr(df.index[-1], "timestamp") else int(df.index[-1])

        if pd.isna(hist_prev) or pd.isna(hist_curr):
            return Signal(
                action=SignalAction.HOLD,
                symbol=symbol,
                strength=0.0,
                price=price,
                timestamp=timestamp,
                reason="Insufficient data for MACD calculation",
            )

        # Histogram crosses above zero → BUY
        if hist_prev <= 0 and hist_curr > 0:
            strength = min(1.0, float(abs(hist_curr) / close_curr * 500))
            return Signal(
                action=SignalAction.BUY,
                symbol=symbol,
                strength=strength,
                price=price,
                timestamp=timestamp,
                reason=f"MACD histogram crosses above zero: hist={hist_curr:.6f}",
            )

        # Histogram crosses below zero → SELL
        if hist_prev >= 0 and hist_curr < 0:
            strength = min(1.0, float(abs(hist_curr) / close_curr * 500))
            return Signal(
                action=SignalAction.SELL,
                symbol=symbol,
                strength=strength,
                price=price,
                timestamp=timestamp,
                reason=f"MACD histogram crosses below zero: hist={hist_curr:.6f}",
            )

        return Signal(
            action=SignalAction.HOLD,
            symbol=symbol,
            strength=0.0,
            price=price,
            timestamp=timestamp,
            reason="No MACD histogram cross detected",
        )
