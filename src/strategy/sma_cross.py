from decimal import Decimal
import pandas as pd

from src.strategy.base import BaseStrategy, Signal, SignalAction
from src.strategy.registry import register_strategy
from src.strategy.indicators import compute_sma, detect_sma_cross, extract_bar_info


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

        sma_short, sma_long = compute_sma(df["close"], short_window, long_window)
        result = detect_sma_cross(sma_short, sma_long,
                                  short_name=str(short_window), long_name=str(long_window))

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
