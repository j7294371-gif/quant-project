from decimal import Decimal
import pandas as pd

from src.strategy.base import BaseStrategy, Signal, SignalAction
from src.strategy.registry import register_strategy
from src.strategy.indicators import compute_bollinger, detect_bollinger_cross, extract_bar_info


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
        return int(self.params.get("period", 20)) + 1

    def on_bar(self, symbol: str, df: pd.DataFrame) -> Signal:
        period = int(self.params.get("period", 20))
        std_dev = float(self.params.get("std_dev", 2.0))

        sma, upper, lower = compute_bollinger(df["close"], period, std_dev)
        result = detect_bollinger_cross(df["close"], upper, lower)

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
