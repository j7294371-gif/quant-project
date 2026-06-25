"""Abstract execution interface, Order dataclass, and OrderStatus state machine."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
import uuid


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass(frozen=True)
class Order:
    id: str
    symbol: str
    side: str  # "buy" | "sell"
    type: OrderType
    quantity: Decimal
    price: Decimal | None  # None for market orders
    status: OrderStatus
    filled_quantity: Decimal
    filled_price: Decimal | None
    fee: Decimal
    timestamp: int  # UTC ms
    reason: str

    def with_status(
        self,
        new_status: OrderStatus,
        *,
        filled_quantity: Decimal | None = None,
        filled_price: Decimal | None = None,
        fee: Decimal | None = None,
    ) -> "Order":
        return Order(
            id=self.id,
            symbol=self.symbol,
            side=self.side,
            type=self.type,
            quantity=self.quantity,
            price=self.price,
            status=new_status,
            filled_quantity=filled_quantity if filled_quantity is not None else self.filled_quantity,
            filled_price=filled_price if filled_price is not None else self.filled_price,
            fee=fee if fee is not None else self.fee,
            timestamp=self.timestamp,
            reason=self.reason,
        )

    @staticmethod
    def create_pending(
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal | None,
        timestamp: int,
        reason: str,
    ) -> "Order":
        return Order(
            id=str(uuid.uuid4()),
            symbol=symbol,
            side=side,
            type=OrderType.MARKET if price is None else OrderType.LIMIT,
            quantity=quantity,
            price=price,
            status=OrderStatus.PENDING,
            filled_quantity=Decimal("0"),
            filled_price=None,
            fee=Decimal("0"),
            timestamp=timestamp,
            reason=reason,
        )


@dataclass(frozen=True)
class Position:
    symbol: str
    quantity: Decimal
    entry_price: Decimal
    timestamp: int  # UTC ms


class ExecutionEngine(ABC):
    @abstractmethod
    def execute_signal(self, signal, equity: Decimal, position: Position | None,
                       quantity: Decimal | None = None) -> Order:
        """Execute a trade signal. quantity=None means engine decides sizing."""
        ...

    @abstractmethod
    def close_all_positions(self, reason: str) -> list[Order]:
        ...

    @abstractmethod
    def cancel_all_orders(self) -> list[Order]:
        ...

    @abstractmethod
    def get_current_position(self, symbol: str) -> Position | None:
        ...

    @abstractmethod
    def get_equity(self) -> Decimal:
        ...
