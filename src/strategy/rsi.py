from decimal import Decimal
import pandas as pd

from src.strategy.base import BaseStrategy, Signal, SignalAction
from src.strategy.registry import register_strategy


@register_strategy("rsi")
class RsiStrategy(BaseStrategy):
    def _validate_params(self) -> None:
        period = int(self.params.get("period", 14))
        oversold = int(self.params.get("oversold", 30))
        overbought = int(self.params.get("overbought", 70))
        if oversold >= overbought:
            raise ValueError(
                f"oversold ({oversold}) must be less than overbought ({overbought})"
            )
        if period < 2:
            raise ValueError(f"period ({period}) must be at least 2")

    @property
    def min_bars(self) -> int:
        period = int(self.params.get("period", 14))
        return period * 3

    @property
    def warmup_bars(self) -> int:
        period = int(self.params.get("period", 14))
        return period * 4

    def on_bar(self, symbol: str, df: pd.DataFrame) -> Signal:
        period = int(self.params.get("period", 14))
        oversold = float(self.params.get("oversold", 30))
        overbought = float(self.params.get("overbought", 70))

        close = df["close"]
        delta = close.diff()

        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        # Wilder smoothing: ewm with alpha=1/period, adjust=False
        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

        rsi_prev = rsi.iloc[-2]
        rsi_curr = rsi.iloc[-1]

        price = Decimal(str(close.iloc[-1]))
        timestamp = int(df.index[-1].timestamp() * 1000) if hasattr(df.index[-1], "timestamp") else int(df.index[-1])

        if pd.isna(rsi_prev) or pd.isna(rsi_curr):
            return Signal(
                action=SignalAction.HOLD,
                symbol=symbol,
                strength=0.0,
                price=price,
                timestamp=timestamp,
                reason="Insufficient data for RSI calculation",
            )

        # Track RSI extremes via self.state
        # When RSI enters oversold zone, track the minimum
        if rsi_curr < oversold:
            current_min = self.state.get("_rsi_min")
            if current_min is None or rsi_curr < current_min:
                self.state["_rsi_min"] = float(rsi_curr)

        # When RSI enters overbought zone, track the maximum
        if rsi_curr > overbought:
            current_max = self.state.get("_rsi_max")
            if current_max is None or rsi_curr > current_max:
                self.state["_rsi_max"] = float(rsi_curr)

        # When RSI exits zone (back to normal), reset extremes
        if oversold <= rsi_curr <= overbought:
            self.state["_rsi_min"] = None
            self.state["_rsi_max"] = None

        # BUY: RSI crosses ABOVE oversold (exits oversold zone upward)
        rsi_min_tracked = self.state.get("_rsi_min")
        if rsi_prev < oversold and rsi_curr > oversold:
            overshoot = oversold - float(rsi_min_tracked) if rsi_min_tracked is not None else oversold - float(rsi_prev)
            strength = min(1.0, overshoot / oversold)
            self.state["_rsi_min"] = None
            return Signal(
                action=SignalAction.BUY,
                symbol=symbol,
                strength=strength,
                price=price,
                timestamp=timestamp,
                reason=f"RSI crosses above oversold ({oversold}): RSI_prev={rsi_prev:.2f}, RSI_curr={rsi_curr:.2f}, min={rsi_min_tracked}",
            )

        # SELL: RSI crosses BELOW overbought (exits overbought zone downward)
        rsi_max_tracked = self.state.get("_rsi_max")
        if rsi_prev > overbought and rsi_curr < overbought:
            peak_excess = float(rsi_max_tracked) - overbought if rsi_max_tracked is not None else float(rsi_prev) - overbought
            strength = min(1.0, peak_excess / (100.0 - overbought))
            self.state["_rsi_max"] = None
            return Signal(
                action=SignalAction.SELL,
                symbol=symbol,
                strength=strength,
                price=price,
                timestamp=timestamp,
                reason=f"RSI crosses below overbought ({overbought}): RSI_prev={rsi_prev:.2f}, RSI_curr={rsi_curr:.2f}, max={rsi_max_tracked}",
            )

        return Signal(
            action=SignalAction.HOLD,
            symbol=symbol,
            strength=0.0,
            price=price,
            timestamp=timestamp,
            reason="No RSI signal",
        )
