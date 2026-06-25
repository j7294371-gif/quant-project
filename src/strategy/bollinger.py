from decimal import Decimal
import pandas as pd

from src.strategy.base import BaseStrategy, Signal, SignalAction
from src.strategy.registry import register_strategy


@register_strategy("bollinger")
class BollingerStrategy(BaseStrategy):
    def _validate_params(self) -> None:
        period = int(self.params.get("period", 20))
        std_dev = float(self.params.get("std_dev", 2.0))
        if period < 2:
            raise ValueError(f"period ({period}) must be at least 2")
        if std_dev <= 0:
            raise ValueError(f"std_dev ({std_dev}) must be positive")

    @property
    def min_bars(self) -> int:
        period = int(self.params.get("period", 20))
        return period + 1

    def on_bar(self, symbol: str, df: pd.DataFrame) -> Signal:
        period = int(self.params.get("period", 20))
        std_dev = float(self.params.get("std_dev", 2.0))

        close = df["close"]

        sma = close.rolling(window=period).mean()
        sigma = close.rolling(window=period).std(ddof=0)

        upper_band = sma + std_dev * sigma
        lower_band = sma - std_dev * sigma

        close_prev = close.iloc[-2]
        close_curr = close.iloc[-1]
        upper_prev = upper_band.iloc[-2]
        upper_curr = upper_band.iloc[-1]
        lower_prev = lower_band.iloc[-2]
        lower_curr = lower_band.iloc[-1]

        price = Decimal(str(close_curr))
        timestamp = int(df.index[-1].timestamp() * 1000) if hasattr(df.index[-1], "timestamp") else int(df.index[-1])

        if pd.isna(upper_prev) or pd.isna(lower_prev) or pd.isna(upper_curr) or pd.isna(lower_curr):
            return Signal(
                action=SignalAction.HOLD,
                symbol=symbol,
                strength=0.0,
                price=price,
                timestamp=timestamp,
                reason="Insufficient data for Bollinger Band calculation",
            )

        # BUY: price crosses ABOVE lower band (was below, now at or above)
        if close_prev < lower_prev and close_curr >= lower_curr:
            diff = float(lower_curr - close_curr)
            strength = min(1.0, abs(diff) / float(lower_curr) * 10)
            return Signal(
                action=SignalAction.BUY,
                symbol=symbol,
                strength=strength,
                price=price,
                timestamp=timestamp,
                reason=f"Price crosses above lower band: close={close_curr:.4f} >= lower={lower_curr:.4f}",
            )

        # SELL: price crosses BELOW upper band (was above, now at or below)
        if close_prev > upper_prev and close_curr <= upper_curr:
            diff = float(close_curr - upper_curr)
            strength = min(1.0, abs(diff) / float(upper_curr) * 10)
            return Signal(
                action=SignalAction.SELL,
                symbol=symbol,
                strength=strength,
                price=price,
                timestamp=timestamp,
                reason=f"Price crosses below upper band: close={close_curr:.4f} <= upper={upper_curr:.4f}",
            )

        return Signal(
            action=SignalAction.HOLD,
            symbol=symbol,
            strength=0.0,
            price=price,
            timestamp=timestamp,
            reason="No Bollinger Band cross detected",
        )
