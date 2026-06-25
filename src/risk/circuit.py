import time
from decimal import Decimal
from loguru import logger


class CircuitBreaker:
    def __init__(self, config, state_store):
        self.max_drawdown_limit = Decimal(str(config.max_drawdown_limit))
        self.daily_loss_limit = Decimal(str(config.daily_loss_limit))
        self.cooldown_hours = config.circuit_cooldown_hours
        self.state_store = state_store
        self._tripped = False
        self._reason = ""
        self._tripped_at = 0
        self.event_count = 0

        # Restore from state
        self._restore()

    def _restore(self):
        data = self.state_store.load("circuit")
        if data and data.get("is_tripped"):
            self._tripped = True
            self._reason = data.get("reason", "")
            self._tripped_at = data.get("tripped_at", 0)
            # Check cooldown
            if self.cooldown_hours > 0:
                cooldown_ms = self.cooldown_hours * 3600 * 1000
                now_ms = int(time.time() * 1000)
                if now_ms - self._tripped_at >= cooldown_ms:
                    logger.info("Circuit breaker cooldown expired, auto-resetting")
                    self.reset()

    def check(
        self, equity: Decimal, peak_equity: Decimal, daily_pnl: Decimal
    ) -> bool:
        """Returns True = normal, False = tripped."""
        if self._tripped:
            return False

        equity = Decimal(str(equity))
        peak_equity = Decimal(str(peak_equity))
        daily_pnl = Decimal(str(daily_pnl))

        # Max drawdown check
        if peak_equity > 0:
            drawdown = (peak_equity - equity) / peak_equity
            if drawdown >= self.max_drawdown_limit:
                self.trip(
                    f"Max drawdown {float(drawdown)*100:.1f}% >= {float(self.max_drawdown_limit)*100:.1f}%"
                )
                return False

        # Daily loss check
        if daily_pnl < 0:
            loss_ratio = abs(daily_pnl) / equity
            if loss_ratio >= self.daily_loss_limit:
                self.trip(
                    f"Daily loss {float(loss_ratio)*100:.1f}% >= {float(self.daily_loss_limit)*100:.1f}%"
                )
                return False

        return True

    def trip(self, reason: str) -> None:
        if self._tripped:
            return
        self._tripped = True
        self._reason = reason
        self._tripped_at = int(time.time() * 1000)
        self.event_count += 1
        logger.critical(f"CIRCUIT BREAKER TRIPPED: {reason}")
        self._persist()

    def reset(self) -> None:
        self._tripped = False
        self._reason = ""
        self._tripped_at = 0
        logger.info("Circuit breaker manually reset")
        self._persist()

    def _persist(self):
        self.state_store.save("circuit", {
            "is_tripped": self._tripped,
            "reason": self._reason,
            "tripped_at": self._tripped_at,
        })

    @property
    def is_tripped(self) -> bool:
        return self._tripped

    @property
    def reason(self) -> str:
        return self._reason
