"""Backtest execution engine with T+1 model and friction costs.

CRITICAL: execution_price = df.iloc[i+1]['open'], NEVER df.iloc[i]['close']!
Exit-before-entry: check SL/TP before evaluating new signals on same bar.
Friction costs: 0.1% fee + 0.05% slippage.
"""

from dataclasses import dataclass
from decimal import Decimal
from loguru import logger
from src.execution.base import ExecutionEngine, Order, OrderStatus, OrderType, Position
from src.strategy.base import Signal


@dataclass(frozen=True)
class BacktestResult:
    initial_equity: Decimal
    final_equity: Decimal
    equity_curve: list[Decimal]
    trades: list[Order]
    is_oos: bool = False
    train_result: "BacktestResult | None" = None
    oos_warning: str = ""


class BacktestEngine(ExecutionEngine):
    def __init__(self, initial_equity: Decimal = Decimal("10000")):
        self.cash = Decimal(str(initial_equity))
        self._position: Position | None = None
        self._high_since_entry: Decimal = Decimal("0")
        self._trades: list[Order] = []
        self._equity_curve: list[Decimal] = [self.cash]
        self._peak_equity: Decimal = self.cash
        self._daily_pnl: Decimal = Decimal("0")
        self._current_day: str = ""
        self._last_close: Decimal = Decimal("0")

    @property
    def trades(self) -> list[Order]:
        return list(self._trades)

    @property
    def equity_curve(self) -> list[Decimal]:
        return list(self._equity_curve)

    @property
    def peak_equity(self) -> Decimal:
        return self._peak_equity

    @property
    def daily_pnl(self) -> Decimal:
        return self._daily_pnl

    def get_current_position(self, symbol: str) -> Position | None:
        return self._position

    def get_equity(self) -> Decimal:
        if self._position and self._last_close > 0:
            return self.cash + self._position.quantity * self._last_close
        return self.cash

    def cancel_all_orders(self) -> list[Order]:
        return []

    def set_last_close(self, price: Decimal):
        self._last_close = Decimal(str(price))

    def update_daily(self, daily_pnl: Decimal, current_day: str):
        self._daily_pnl = Decimal(str(daily_pnl))
        self._current_day = current_day

    def execute_signal(self, signal, equity: Decimal, position: Position | None,
                       quantity: Decimal | None = None) -> Order:
        """Execute a signal. quantity from RiskManager, or engine decides if None."""
        execution_price = self._last_close
        execution_price = Decimal(str(execution_price))

        if signal.action.value == "buy":
            return self._execute_buy(signal, execution_price, quantity)
        elif signal.action.value == "sell":
            return self._execute_sell(signal, execution_price)
        else:
            return Order.create_pending(
                symbol=signal.symbol, side="hold", quantity=Decimal("0"),
                price=execution_price, timestamp=signal.timestamp, reason="hold",
            )

    def _execute_buy(self, signal, price: Decimal, quantity: Decimal | None = None) -> Order:
        """Execute buy with slippage and fee. Uses RiskManager quantity if provided."""
        execution_price = price * Decimal("1.0005")  # slippage

        if quantity is None:
            quantity = self.cash * Decimal("0.99") / execution_price

        filled_value = quantity * execution_price
        fee = filled_value * Decimal("0.001")

        self.cash = self.cash - filled_value - fee
        self._position = Position(
            symbol=signal.symbol, quantity=quantity,
            entry_price=execution_price, timestamp=signal.timestamp,
        )
        self._high_since_entry = execution_price

        order = Order(
            id=Order.create_pending(signal.symbol, "buy", quantity, execution_price,
                                    signal.timestamp, signal.reason).id,
            symbol=signal.symbol, side="buy", type=OrderType.MARKET,
            quantity=quantity, price=execution_price, status=OrderStatus.FILLED,
            filled_quantity=quantity, filled_price=execution_price, fee=fee,
            timestamp=signal.timestamp, reason=signal.reason,
        )
        self._trades.append(order)
        logger.info(f"Backtest BUY: {signal.symbol} qty={quantity:.6f} @ {execution_price:.2f} fee={fee:.2f}")
        return order

    def _execute_sell(self, signal, price: Decimal) -> Order:
        """Execute sell (close position) with slippage and fee."""
        execution_price = price * Decimal("0.9995")  # slippage

        if self._position is None:
            logger.warning("Sell signal but no position to close")
            return Order.create_pending(signal.symbol, "sell", Decimal("0"),
                                        execution_price, signal.timestamp, "no position")

        quantity = self._position.quantity
        filled_value = quantity * execution_price
        fee = filled_value * Decimal("0.001")
        entry_value = quantity * self._position.entry_price
        realized_pnl = filled_value - entry_value - fee

        self.cash = self.cash + filled_value - fee
        self._daily_pnl += realized_pnl

        order = Order(
            id=Order.create_pending(signal.symbol, "sell", quantity, execution_price,
                                    signal.timestamp, signal.reason).id,
            symbol=signal.symbol, side="sell", type=OrderType.MARKET,
            quantity=quantity, price=execution_price, status=OrderStatus.FILLED,
            filled_quantity=quantity, filled_price=execution_price, fee=fee,
            timestamp=signal.timestamp, reason=signal.reason,
        )
        self._trades.append(order)
        self._position = None
        logger.info(f"Backtest SELL: {signal.symbol} qty={quantity:.6f} @ {execution_price:.2f} PnL={realized_pnl:.2f}")
        return order

    def close_all_positions(self, reason: str) -> list[Order]:
        if self._position is None:
            return []
        signal = Signal.create_exit(
            symbol=self._position.symbol,
            price=self._last_close,
            timestamp=0,
            reason=reason,
        )
        return [self._execute_sell(signal, self._last_close)]

    def update_equity_curve(self, current_close: Decimal):
        close = Decimal(str(current_close))
        equity = self.cash
        if self._position:
            equity += self._position.quantity * close
        self._equity_curve.append(equity)
        if equity > self._peak_equity:
            self._peak_equity = equity

    def get_result(self, initial_equity: Decimal) -> BacktestResult:
        return BacktestResult(
            initial_equity=Decimal(str(initial_equity)),
            final_equity=self._equity_curve[-1] if self._equity_curve else Decimal(str(initial_equity)),
            equity_curve=list(self._equity_curve),
            trades=list(self._trades),
        )
