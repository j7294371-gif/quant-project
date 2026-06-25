"""Paper trading engine — simulate fills at current price with friction costs.

Friction costs: 0.1% fee + 0.05% slippage.
"""

from decimal import Decimal
from loguru import logger
from src.execution.base import ExecutionEngine, Order, OrderStatus, OrderType, Position


class PaperEngine(ExecutionEngine):
    def __init__(self, state_store):
        self.state_store = state_store
        self._last_price: dict[str, Decimal] = {}
        self._trades: list[Order] = []
        self._realized_pnl: Decimal = Decimal("0")

    @property
    def trades(self) -> list[Order]:
        return list(self._trades)

    @property
    def realized_pnl(self) -> Decimal:
        return self._realized_pnl

    def execute_signal(self, signal, equity: Decimal, position: Position | None) -> Order:
        """Simulate fill at signal.price with slippage and fee."""
        base_price = Decimal(str(signal.price))

        if signal.action.value == "buy":
            execution_price = base_price * Decimal("1.0005")  # slippage
            quantity = self._calc_quantity(equity, execution_price, position)
            filled_value = quantity * execution_price
            fee = filled_value * Decimal("0.001")

            order = Order(
                id=Order.create_pending(signal.symbol, "buy", quantity, execution_price, signal.timestamp, signal.reason).id,
                symbol=signal.symbol,
                side="buy",
                type=OrderType.MARKET,
                quantity=quantity,
                price=execution_price,
                status=OrderStatus.FILLED,
                filled_quantity=quantity,
                filled_price=execution_price,
                fee=fee,
                timestamp=signal.timestamp,
                reason=signal.reason,
            )

        elif signal.action.value == "sell":
            execution_price = base_price * Decimal("0.9995")
            qty = position.quantity if position else Decimal("0")
            filled_value = qty * execution_price
            fee = filled_value * Decimal("0.001")

            if position:
                pnl = filled_value - qty * position.entry_price - fee
                self._realized_pnl += pnl

            order = Order(
                id=Order.create_pending(signal.symbol, "sell", qty, execution_price, signal.timestamp, signal.reason).id,
                symbol=signal.symbol,
                side="sell",
                type=OrderType.MARKET,
                quantity=qty,
                price=execution_price,
                status=OrderStatus.FILLED,
                filled_quantity=qty,
                filled_price=execution_price,
                fee=fee,
                timestamp=signal.timestamp,
                reason=signal.reason,
            )
        else:
            return Order.create_pending(signal.symbol, "hold", Decimal("0"), base_price, signal.timestamp, "hold")

        self._trades.append(order)
        self._last_price[signal.symbol] = base_price
        self._persist_position(order)
        return order

    def _calc_quantity(self, equity: Decimal, price: Decimal, position: Position | None) -> Decimal:
        if position is not None:
            return Decimal("0")  # Already have position
        return equity * Decimal("0.2") / price  # 20% of equity

    def _persist_position(self, order: Order):
        positions_data = self.state_store.load("positions") or {"positions": {}}
        if order.side == "buy" and order.status == OrderStatus.FILLED:
            positions_data["positions"][order.symbol] = {
                "symbol": order.symbol,
                "quantity": str(order.filled_quantity),
                "entry_price": str(order.filled_price),
                "timestamp": order.timestamp,
            }
        elif order.side == "sell" and order.status == OrderStatus.FILLED:
            positions_data["positions"].pop(order.symbol, None)
        self.state_store.save("positions", positions_data)

    def close_all_positions(self, reason: str) -> list[Order]:
        positions_data = self.state_store.load("positions") or {"positions": {}}
        orders = []
        for symbol, pos in list(positions_data.get("positions", {}).items()):
            price = self._last_price.get(symbol, Decimal(str(pos["entry_price"])))

            FakeSignal = type('FakeSignal', (), {})
            FakeAction = type('FakeAction', (), {})
            signal = FakeSignal()
            signal.symbol = symbol
            signal.action = FakeAction()
            signal.action.value = 'sell'
            signal.price = price
            signal.timestamp = 0
            signal.reason = reason

            pos_obj = Position(
                symbol=symbol,
                quantity=Decimal(pos["quantity"]),
                entry_price=Decimal(pos["entry_price"]),
                timestamp=int(pos["timestamp"]),
            )
            order = self.execute_signal(signal, Decimal("0"), pos_obj)
            orders.append(order)
        return orders

    def cancel_all_orders(self) -> list[Order]:
        return []  # Paper mode has no pending orders

    def get_current_position(self, symbol: str) -> Position | None:
        data = self.state_store.load("positions")
        if not data:
            return None
        pos = data.get("positions", {}).get(symbol)
        if not pos:
            return None
        return Position(
            symbol=pos["symbol"],
            quantity=Decimal(pos["quantity"]),
            entry_price=Decimal(pos["entry_price"]),
            timestamp=int(pos["timestamp"]),
        )

    def get_equity(self) -> Decimal:
        """Calculate equity from state store."""
        # Simplified: return cash estimate. In real paper mode this would track cash properly.
        positions_data = self.state_store.load("positions") or {"positions": {}}
        total = Decimal("10000")  # Default initial
        for symbol, pos in positions_data.get("positions", {}).items():
            last = self._last_price.get(symbol, Decimal(pos["entry_price"]))
            qty = Decimal(pos["quantity"])
            entry = Decimal(pos["entry_price"])
            total += qty * (last - entry)  # Unrealized PnL
        return total + self._realized_pnl
