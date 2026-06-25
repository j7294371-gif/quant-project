"""Live trading engine via CCXT. Error classification: recoverable/semi-fatal/fatal.

Friction costs handled by exchange; slippage is real market slippage.
"""

import time
import sys
from decimal import Decimal
import ccxt
from loguru import logger
from src.execution.base import ExecutionEngine, Order, OrderStatus, OrderType, Position
from src.utils.errors import PositionMismatchError


class LiveEngine(ExecutionEngine):
    def __init__(self, exchange_id: str, api_key: str, api_secret: str, testnet: bool):
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "timeout": 30000,
        })
        if testnet:
            self.exchange.set_sandbox_mode(True)
        self._trades: list[Order] = []

    @property
    def trades(self) -> list[Order]:
        return list(self._trades)

    def execute_signal(self, signal, equity: Decimal, position: Position | None,
                       quantity: Decimal | None = None) -> Order:
        """Real order via CCXT. quantity from RiskManager (required for live)."""
        # Position sync check
        exchange_pos = self.get_current_position(signal.symbol)
        if (position is None) != (exchange_pos is None):
            raise PositionMismatchError(
                f"Position mismatch for {signal.symbol}: "
                f"local={position is not None}, exchange={exchange_pos is not None}"
            )

        if quantity is None:
            logger.error("LiveEngine requires quantity from RiskManager, got None")
            return Order.create_pending(signal.symbol, signal.action.value, Decimal("0"),
                                        signal.price, signal.timestamp, "missing quantity"
                                        ).with_status(OrderStatus.REJECTED)

        try:
            if signal.action.value == "buy":
                return self._place_order(signal, "buy", quantity)
            elif signal.action.value == "sell":
                return self._place_order(signal, "sell", quantity)
            else:
                return Order.create_pending(signal.symbol, "hold", Decimal("0"),
                                            signal.price, signal.timestamp, "hold")
        except ccxt.AuthenticationError as e:
            logger.critical(f"Authentication failed: {e}")
            self.cancel_all_orders()
            sys.exit(1)
        except ccxt.InsufficientFunds as e:
            logger.error(f"Insufficient funds for {signal.symbol}: {e}")
            return Order.create_pending(
                signal.symbol, signal.action.value, quantity, signal.price,
                signal.timestamp, f"rejected: {e}"
            ).with_status(OrderStatus.REJECTED)
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            logger.warning(f"Network error, retrying: {e}")
            time.sleep(2)
            return self._place_order(signal, signal.action.value, quantity)

    def _place_order(self, signal, side: str, quantity: Decimal) -> Order:
        """Place market order and poll for fill (2s interval, 30s timeout)."""
        try:
            if side == "buy":
                exchange_order = self.exchange.create_market_buy_order(
                    signal.symbol, float(quantity)
                )
            else:
                exchange_order = self.exchange.create_market_sell_order(
                    signal.symbol, float(quantity)
                )
        except Exception as e:
            logger.error(f"Order creation failed: {e}")
            return Order.create_pending(
                signal.symbol, side, quantity, signal.price,
                signal.timestamp, signal.reason
            ).with_status(OrderStatus.REJECTED)

        order_id = exchange_order.get("id", "")
        for _ in range(15):  # 30s timeout at 2s intervals
            time.sleep(2)
            try:
                status = self.exchange.fetch_order(order_id, signal.symbol)
                if status.get("status") == "closed":
                    filled = Decimal(str(status.get("filled", status.get("amount", 0))))
                    price = Decimal(str(status.get("average", status.get("price", 0))))
                    fee_info = status.get("fee", {})
                    fee = Decimal(str(fee_info.get("cost", 0))) if fee_info else Decimal("0")

                    order = Order(
                        id=order_id, symbol=signal.symbol, side=side,
                        type=OrderType.MARKET, quantity=quantity, price=signal.price,
                        status=OrderStatus.FILLED, filled_quantity=filled,
                        filled_price=price, fee=fee,
                        timestamp=signal.timestamp, reason=signal.reason,
                    )
                    self._trades.append(order)
                    logger.info(f"Live {side.upper()}: {signal.symbol} qty={filled} @ {price}")
                    return order
            except Exception:
                continue

        logger.error(f"Order {order_id} timed out, cancelling")
        try:
            self.exchange.cancel_order(order_id, signal.symbol)
        except Exception:
            pass
        return Order.create_pending(
            signal.symbol, side, quantity, signal.price,
            signal.timestamp, signal.reason
        ).with_status(OrderStatus.CANCELLED)

    def close_all_positions(self, reason: str) -> list[Order]:
        orders = []
        try:
            positions = self.exchange.fetch_positions()
            for pos in positions:
                if float(pos.get("contracts", 0)) > 0:
                    qty = Decimal(str(abs(float(pos["contracts"]))))
                    symbol = pos["symbol"]
                    try:
                        self.exchange.create_market_sell_order(symbol, float(qty))
                        orders.append(Order.create_pending(
                            symbol, "sell", qty, None,
                            int(time.time() * 1000), reason,
                        ))
                    except Exception as e:
                        logger.error(f"Failed to close {symbol}: {e}")
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
        return orders

    def cancel_all_orders(self) -> list[Order]:
        try:
            self.exchange.cancel_all_orders()
            logger.info("All orders cancelled")
        except Exception as e:
            logger.error(f"Failed to cancel orders: {e}")
        return []

    def get_current_position(self, symbol: str) -> Position | None:
        try:
            balance = self.exchange.fetch_balance()
            free = balance.get(symbol.split("/")[0], {}).get("free", 0)
            if float(free) <= 0:
                return None
            return Position(
                symbol=symbol, quantity=Decimal(str(free)),
                entry_price=Decimal("0"),
                timestamp=int(time.time() * 1000),
            )
        except Exception as e:
            logger.error(f"Failed to fetch position for {symbol}: {e}")
            return None

    def get_equity(self) -> Decimal:
        try:
            balance = self.exchange.fetch_balance()
            return Decimal(str(balance.get("total", {}).get("USDT",
                           balance.get("free", {}).get("USDT", 0))))
        except Exception as e:
            logger.error(f"Failed to fetch equity: {e}")
            return Decimal("0")
