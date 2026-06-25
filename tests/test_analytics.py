"""Metrics calculation tests."""
import pytest
from decimal import Decimal
from src.analytics.metrics import calculate_metrics, _timeframe_to_periods_per_year


class TestTimeframeMapping:
    def test_1h_periods_per_year(self):
        assert _timeframe_to_periods_per_year("1h") == 8760

    def test_4h_periods_per_year(self):
        assert _timeframe_to_periods_per_year("4h") == 2190

    def test_1d_periods_per_year(self):
        assert _timeframe_to_periods_per_year("1d") == 365

    def test_1m_periods_per_year(self):
        """1m bars: 8760 * 60 = 525600 periods/year."""
        assert _timeframe_to_periods_per_year("1m") == 525600


class TestMetricsCalculation:
    def test_total_return_positive(self):
        from unittest.mock import MagicMock
        result = MagicMock()
        result.equity_curve = [
            Decimal("10000"), Decimal("10100"), Decimal("10200"),
            Decimal("10150"), Decimal("10300"),
        ]
        result.trades = []
        metrics = calculate_metrics(result, "1h")
        assert float(metrics.total_return) > 0

    def test_max_drawdown(self):
        from unittest.mock import MagicMock
        result = MagicMock()
        result.equity_curve = [
            Decimal("10000"), Decimal("10500"), Decimal("9500"), Decimal("10200"),
        ]
        result.trades = []
        metrics = calculate_metrics(result, "1h")
        # MDD from peak 10500 to trough 9500
        mdd = float(metrics.max_drawdown)
        assert mdd > 0.05  # ~9.5% drawdown

    def test_calmar_zero_mdd(self):
        from unittest.mock import MagicMock
        result = MagicMock()
        result.equity_curve = [Decimal("10000"), Decimal("10100"), Decimal("10200")]
        result.trades = []
        metrics = calculate_metrics(result, "1h")
        assert metrics.calmar_ratio == float('inf')

    def test_profit_factor_no_losses(self):
        from unittest.mock import MagicMock
        result = MagicMock()
        result.equity_curve = [Decimal("10000"), Decimal("10100")]
        result.trades = []
        metrics = calculate_metrics(result, "1h")
        assert metrics.profit_factor == float('inf')

    def test_empty_result_returns_zeros(self):
        from unittest.mock import MagicMock
        result = MagicMock()
        result.equity_curve = [Decimal("10000")]
        result.trades = []
        metrics = calculate_metrics(result, "1h")
        assert metrics.total_return == Decimal("0")
        assert metrics.total_trades == 0
        assert metrics.win_rate == 0.0
