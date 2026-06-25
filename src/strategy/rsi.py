from decimal import Decimal
import pandas as pd

from src.strategy.base import BaseStrategy, Signal, SignalAction
from src.strategy.registry import register_strategy
from src.strategy.indicators import compute_rsi, detect_rsi_signal, extract_bar_info


@register_strategy("rsi")
class RsiStrategy(BaseStrategy):
    def _validate_params(self) -> None:
        period = int(self.params.get("period", 14))
        oversold = int(self.params.get("oversold", 30))
        overbought = int(self.params.get("overbought", 70))
        if oversold >= overbought:
            raise ValueError(f"oversold ({oversold}) must be less than overbought ({overbought})")
        if period < 2:
            raise ValueError(f"period ({period}) must be at least 2")

    @property
    def min_bars(self) -> int:
        return int(self.params.get("period", 14)) * 3

    @property
    def warmup_bars(self) -> int:
        return int(self.params.get("period", 14)) * 4

    def on_bar(self, symbol: str, df: pd.DataFrame) -> Signal:
        period = int(self.params.get("period", 14))
        oversold = float(self.params.get("oversold", 30))
        overbought = float(self.params.get("overbought", 70))

        rsi = compute_rsi(df["close"], period)
        result = detect_rsi_signal(rsi, oversold, overbought, self.state)

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
