from decimal import Decimal
from loguru import logger
from src.risk.position import calculate_position, PositionSize
from src.risk.stoploss import (
    check_stop_loss,
    check_take_profit,
    StopLossResult,
    TakeProfitResult,
)
from src.risk.circuit import CircuitBreaker


class RiskManager:
    def __init__(self, config, state_store):
        self.config = config
        self.position_config = {
            "max_position_pct": str(config.max_position_pct),
            "risk_per_trade_pct": str(config.risk_per_trade_pct),
            "position_method": config.position_method,
        }
        self.stop_loss_config = {
            "type": config.stop_loss.type,
            "fixed_pct": str(config.stop_loss.fixed_pct),
            "atr_multiplier": str(config.stop_loss.atr_multiplier),
            "trailing_pct": str(config.stop_loss.trailing_pct),
        }
        self.take_profit_config = {
            "type": config.take_profit.type,
            "fixed_pct": str(config.take_profit.fixed_pct),
            "atr_multiplier": str(config.take_profit.atr_multiplier),
        }
        self.circuit_breaker = CircuitBreaker(config, state_store)
        self.daily_trade_count = 0
        self.max_daily_trades = 50

    def check_signal(
        self, signal, equity: Decimal, position: dict | None
    ) -> bool:
        """Pre-trade check: circuit breaker + position cap + daily trade limit."""
        # Check circuit breaker
        if self.circuit_breaker.is_tripped:
            logger.warning(
                f"Signal rejected: circuit breaker tripped ({self.circuit_breaker.reason})"
            )
            return False

        # Check daily trade count
        if self.daily_trade_count >= self.max_daily_trades:
            logger.warning(
                f"Signal rejected: daily trade limit ({self.max_daily_trades}) reached"
            )
            return False

        # Check position cap (don't open new if already at max)
        if position is not None and signal.action.value == "buy":
            logger.warning(
                f"Signal rejected: already have position for {signal.symbol}"
            )
            return False

        return True

    def calculate_position(
        self, signal, equity: Decimal, stop_loss_price: Decimal | None
    ) -> PositionSize:
        """Calculate position size based on configured method."""
        return calculate_position(
            method=self.config.position_method,
            equity=equity,
            price=signal.price,
            stop_loss_price=stop_loss_price,
            params=self.position_config,
        )

    def check_exits(
        self, position: dict, current_price: Decimal, current_atr: Decimal | None
    ) -> tuple[StopLossResult, TakeProfitResult]:
        """Check stop-loss and take-profit on each bar."""
        entry_price = Decimal(str(position["entry_price"]))
        high_since_entry = Decimal(
            str(position.get("high_since_entry", current_price))
        )

        sl_result = check_stop_loss(
            entry_price=entry_price,
            current_price=current_price,
            high_since_entry=high_since_entry,
            current_atr=current_atr,
            config=self.stop_loss_config,
        )

        tp_result = check_take_profit(
            entry_price=entry_price,
            current_price=current_price,
            current_atr=current_atr,
            config=self.take_profit_config,
        )

        return sl_result, tp_result

    def update_equity(
        self, equity: Decimal, daily_pnl: Decimal, peak_equity: Decimal
    ) -> bool:
        """Update equity and check circuit breaker. Returns True if OK."""
        return self.circuit_breaker.check(equity, peak_equity, daily_pnl)

    def record_trade(self):
        """Increment daily trade counter."""
        self.daily_trade_count += 1
