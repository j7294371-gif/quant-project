"""Execution engine tests: Order state machine, backtest, paper."""
import pytest
from decimal import Decimal
from src.execution.base import Order, OrderStatus, OrderType, Position


class TestOrderStateMachine:
    def test_create_pending(self):
        order = Order.create_pending("BTC/USDT", "buy", Decimal("1"), Decimal("50000"), 1704067200000, "test")
        assert order.status == OrderStatus.PENDING
        assert order.side == "buy"
        assert order.filled_quantity == Decimal("0")

    def test_with_status_immutable(self):
        order = Order.create_pending("BTC/USDT", "buy", Decimal("1"), Decimal("50000"), 1704067200000, "test")
        filled = order.with_status(OrderStatus.FILLED, filled_quantity=Decimal("1"), filled_price=Decimal("50000"), fee=Decimal("50"))
        assert filled.status == OrderStatus.FILLED
        assert order.status == OrderStatus.PENDING  # Original unchanged
        assert filled is not order

    def test_pending_to_cancelled(self):
        order = Order.create_pending("BTC/USDT", "buy", Decimal("1"), Decimal("50000"), 1704067200000, "test")
        cancelled = order.with_status(OrderStatus.CANCELLED)
        assert cancelled.status == OrderStatus.CANCELLED

    def test_frozen_prevents_mutation(self):
        order = Order.create_pending("BTC/USDT", "buy", Decimal("1"), Decimal("50000"), 1704067200000, "test")
        with pytest.raises(Exception):
            order.status = OrderStatus.FILLED  # type: ignore

    def test_order_id_unique(self):
        o1 = Order.create_pending("BTC/USDT", "buy", Decimal("1"), Decimal("50000"), 1704067200000, "test")
        o2 = Order.create_pending("BTC/USDT", "buy", Decimal("1"), Decimal("50000"), 1704067200000, "test")
        assert o1.id != o2.id


class TestPosition:
    def test_position_frozen(self):
        pos = Position(symbol="BTC/USDT", quantity=Decimal("1"), entry_price=Decimal("50000"), timestamp=1704067200000)
        assert pos.symbol == "BTC/USDT"
        with pytest.raises(Exception):
            pos.quantity = Decimal("2")  # type: ignore


class TestBacktestEngine:
    def test_buy_with_friction(self):
        from src.execution.backtest import BacktestEngine
        engine = BacktestEngine(initial_equity=Decimal("10000"))
        engine.set_last_close(Decimal("100"))
        signal = type('Signal', (), {
            'action': type('Action', (), {'value': 'buy'})(),
            'symbol': 'BTC/USDT', 'strength': 0.8, 'price': Decimal("100"),
            'timestamp': 1704067200000, 'reason': 'test',
        })()
        order = engine.execute_signal(signal, Decimal("10000"), None)
        assert order.status == OrderStatus.FILLED
        assert float(order.filled_price) > 100  # Slippage applied
        assert float(order.fee) > 0  # Fee applied

    def test_sell_closes_position(self):
        from src.execution.backtest import BacktestEngine
        engine = BacktestEngine(initial_equity=Decimal("10000"))
        engine.set_last_close(Decimal("100"))
        # First buy
        buy_signal = type('Signal', (), {
            'action': type('Action', (), {'value': 'buy'})(),
            'symbol': 'BTC/USDT', 'strength': 0.8, 'price': Decimal("100"),
            'timestamp': 1704067200000, 'reason': 'test buy',
        })()
        engine.execute_signal(buy_signal, Decimal("10000"), None)
        assert engine.get_current_position("BTC/USDT") is not None
        # Then sell
        engine.set_last_close(Decimal("110"))
        sell_signal = type('Signal', (), {
            'action': type('Action', (), {'value': 'sell'})(),
            'symbol': 'BTC/USDT', 'strength': 0.8, 'price': Decimal("110"),
            'timestamp': 1704070800000, 'reason': 'test sell',
        })()
        order = engine.execute_signal(sell_signal, Decimal("0"), engine.get_current_position("BTC/USDT"))
        assert order.status == OrderStatus.FILLED
        assert order.side == "sell"

    def test_close_all_positions(self):
        from src.execution.backtest import BacktestEngine
        engine = BacktestEngine(initial_equity=Decimal("10000"))
        engine.set_last_close(Decimal("100"))
        buy_signal = type('Signal', (), {
            'action': type('Action', (), {'value': 'buy'})(),
            'symbol': 'BTC/USDT', 'strength': 0.8, 'price': Decimal("100"),
            'timestamp': 1704067200000, 'reason': 'test',
        })()
        engine.execute_signal(buy_signal, Decimal("10000"), None)
        orders = engine.close_all_positions("circuit_trip")
        assert len(orders) >= 1
        assert orders[0].side == "sell"

    def test_equity_curve(self):
        from src.execution.backtest import BacktestEngine
        engine = BacktestEngine(initial_equity=Decimal("10000"))
        engine.update_equity_curve(Decimal("100"))
        engine.update_equity_curve(Decimal("101"))
        assert len(engine.equity_curve) == 3  # initial + 2 updates

    def test_get_result(self):
        from src.execution.backtest import BacktestEngine
        engine = BacktestEngine(initial_equity=Decimal("10000"))
        engine.update_equity_curve(Decimal("100"))
        result = engine.get_result(Decimal("10000"))
        assert result.initial_equity == Decimal("10000")
        assert len(result.equity_curve) > 0
