from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any
import pandas as pd


class SignalAction(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass(frozen=True)
class Signal:
    action: SignalAction
    symbol: str
    strength: float  # 0.0 ~ 1.0, HOLD=0.0
    price: Decimal
    timestamp: int  # UTC ms
    reason: str


class BaseStrategy(ABC):
    def __init__(self, params: dict):
        self.params = params
        self.state: dict[str, Any] = {}
        self._validate_params()

    def _validate_params(self) -> None:
        pass

    @abstractmethod
    def on_bar(self, symbol: str, df: pd.DataFrame) -> Signal:
        ...

    @property
    @abstractmethod
    def min_bars(self) -> int:
        ...

    @property
    def warmup_bars(self) -> int:
        return self.min_bars
