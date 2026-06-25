from decimal import Decimal
import pandas as pd

from src.strategy.base import BaseStrategy, Signal, SignalAction
from src.strategy.registry import register_strategy
from src.strategy.indicators import compute_macd, detect_macd_cross, extract_bar_info


@register_strategy("macd")
class MacdStrategy(BaseStrategy):
    def _validate_params(self) -> None:
        fast = int(self.params.get("fast", 12))
        slow = int(self.params.get("slow", 26))
        if fast >= slow:
            raise ValueError(f"fast ({fast}) must be less than slow ({slow})")

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

        _, _, histogram = compute_macd(df["close"], fast, slow, signal_period)
        close_curr = float(df["close"].iloc[-1])
        result = detect_macd_cross(histogram, close_curr)

        price, timestamp = extract_bar_info(df)

        if result.score >= 25:
            action = SignalAction.BUY
        elif result.score <= -25:
            action = SignalAction.SELL
        else:
            action = SignalAction.HOLD

        return Signal(
            action=action,
            symbol=symbol,
            strength=result.strength,
            price=price,
            timestamp=timestamp,
            reason=result.reason,
        )
