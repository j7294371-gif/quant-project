"""Risk management tests: position sizing, stop-loss, take-profit, circuit breaker."""
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from src.risk.position import calculate_position, PositionSize
from src.risk.stoploss import check_stop_loss, check_take_profit


class TestPositionSizing:
    def test_fixed_pct(self):
        result = calculate_position(
            "fixed_pct", Decimal("10000"), Decimal("100"), None,
            {"max_position_pct": "0.2"},
        )
        assert float(result.notional) == pytest.approx(2000, rel=0.01)
        assert float(result.quantity) == pytest.approx(20, rel=0.01)

    def test_kelly_default(self):
        """Half-Kelly with W=0.45, R=1.5 → f* ≈ 4.17%."""
        result = calculate_position(
            "kelly", Decimal("10000"), Decimal("100"), None, {},
        )
        pct = float(result.pct_of_equity)
        assert 0.03 < pct < 0.06, f"Expected ~4.17%, got {pct*100:.2f}%"

    def test_kelly_with_history(self):
        result = calculate_position(
            "kelly", Decimal("10000"), Decimal("100"), None, {},
            history_stats={"win_rate": 0.55, "avg_win_ratio": 2.0},
        )
        assert float(result.quantity) > 0

    def test_risk_per_trade(self):
        result = calculate_position(
            "risk_per_trade", Decimal("10000"), Decimal("100"),
            Decimal("95"), {"risk_per_trade_pct": "0.01"},
        )
        # risk_amount = 100, price_diff = 5 → qty = 20
        assert float(result.quantity) == pytest.approx(20, rel=0.01)

    def test_risk_per_trade_no_sl_raises(self):
        with pytest.raises(ValueError):
            calculate_position(
                "risk_per_trade", Decimal("10000"), Decimal("100"),
                None, {"risk_per_trade_pct": "0.01"},
            )

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            calculate_position("invalid", Decimal("10000"), Decimal("100"), None, {})


class TestStopLoss:
    def test_fixed_pct_triggered(self):
        result = check_stop_loss(
            Decimal("100"), Decimal("94.99"), Decimal("100"), None,
            {"type": "fixed_pct", "fixed_pct": "0.05"},
        )
        assert result.triggered

    def test_fixed_pct_not_triggered(self):
        result = check_stop_loss(
            Decimal("100"), Decimal("95.01"), Decimal("100"), None,
            {"type": "fixed_pct", "fixed_pct": "0.05"},
        )
        assert not result.triggered

    def test_atr_triggered(self):
        result = check_stop_loss(
            Decimal("100"), Decimal("93.99"), Decimal("100"),
            Decimal("3"), {"type": "atr", "atr_multiplier": "2.0"},
        )
        assert result.triggered

    def test_trailing_only_goes_up(self):
        """Trailing SL: SL = max(entry*(1-pct), high*(1-pct))."""
        result = check_stop_loss(
            Decimal("100"), Decimal("108"), Decimal("110"),
            None, {"type": "trailing", "trailing_pct": "0.03"},
        )
        # entry SL = 97, trailing SL = 110 * 0.97 = 106.7
        assert float(result.stop_price) == pytest.approx(106.7, rel=0.01)
        assert not result.triggered

    def test_trailing_triggered(self):
        result = check_stop_loss(
            Decimal("100"), Decimal("106"), Decimal("110"),
            None, {"type": "trailing", "trailing_pct": "0.03"},
        )
        # SL = 106.7, current = 106 → triggered
        assert result.triggered


class TestTakeProfit:
    def test_fixed_pct_triggered(self):
        result = check_take_profit(
            Decimal("100"), Decimal("110.01"), None,
            {"type": "fixed_pct", "fixed_pct": "0.10"},
        )
        assert result.triggered

    def test_fixed_pct_not_triggered(self):
        result = check_take_profit(
            Decimal("100"), Decimal("109.99"), None,
            {"type": "fixed_pct", "fixed_pct": "0.10"},
        )
        assert not result.triggered

    def test_atr_tp_triggered(self):
        result = check_take_profit(
            Decimal("100"), Decimal("113"), Decimal("3"),
            {"type": "atr", "atr_multiplier": "4.0"},
        )
        assert result.triggered


class TestCircuitBreaker:
    def test_drawdown_triggers(self, temp_state_dir):
        from src.risk.circuit import CircuitBreaker
        from unittest.mock import MagicMock
        config = MagicMock()
        config.max_drawdown_limit = Decimal("0.3")
        config.daily_loss_limit = Decimal("0.05")
        config.circuit_cooldown_hours = 24
        store = MagicMock()
        store.load.return_value = None
        cb = CircuitBreaker(config, store)
        # 30% drawdown: peak=10000, equity=6999
        result = cb.check(Decimal("6999"), Decimal("10000"), Decimal("0"))
        assert not result
        assert cb.is_tripped

    def test_daily_loss_triggers(self, temp_state_dir):
        from src.risk.circuit import CircuitBreaker
        from unittest.mock import MagicMock
        config = MagicMock()
        config.max_drawdown_limit = Decimal("0.3")
        config.daily_loss_limit = Decimal("0.05")
        config.circuit_cooldown_hours = 24
        store = MagicMock()
        store.load.return_value = None
        cb = CircuitBreaker(config, store)
        # 6% daily loss
        result = cb.check(Decimal("10000"), Decimal("10000"), Decimal("-600"))
        assert not result
        assert cb.is_tripped

    def test_normal_passes(self, temp_state_dir):
        from src.risk.circuit import CircuitBreaker
        from unittest.mock import MagicMock
        config = MagicMock()
        config.max_drawdown_limit = Decimal("0.3")
        config.daily_loss_limit = Decimal("0.05")
        config.circuit_cooldown_hours = 24
        store = MagicMock()
        store.load.return_value = None
        cb = CircuitBreaker(config, store)
        result = cb.check(Decimal("10000"), Decimal("10000"), Decimal("100"))
        assert result
        assert not cb.is_tripped

    def test_reset(self, temp_state_dir):
        from src.risk.circuit import CircuitBreaker
        from unittest.mock import MagicMock
        config = MagicMock()
        config.max_drawdown_limit = Decimal("0.3")
        config.daily_loss_limit = Decimal("0.05")
        config.circuit_cooldown_hours = 24
        store = MagicMock()
        store.load.return_value = None
        cb = CircuitBreaker(config, store)
        cb.trip("test")
        assert cb.is_tripped
        cb.reset()
        assert not cb.is_tripped

    def test_persistence_restore(self, temp_state_dir):
        from src.risk.circuit import CircuitBreaker
        from src.utils.state import StateStore
        store = StateStore(temp_state_dir)
        config = MagicMock()
        config.max_drawdown_limit = Decimal("0.3")
        config.daily_loss_limit = Decimal("0.05")
        config.circuit_cooldown_hours = 24
        cb1 = CircuitBreaker(config, store)
        cb1.trip("persistence test")
        # New breaker from same store
        cb2 = CircuitBreaker(config, store)
        assert cb2.is_tripped
        store.close()
