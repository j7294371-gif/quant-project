"""Portfolio management tests."""
import pytest
from decimal import Decimal
import pandas as pd
import numpy as np
from src.portfolio.manager import PortfolioManager, PortfolioConfig, Allocation


class TestPortfolioConfig:
    def test_default_config(self):
        config = PortfolioConfig()
        assert config.max_position_pct == Decimal("0.2")
        assert config.lookback_days == 30
        assert config.correlation_high == Decimal("0.8")
        assert config.correlation_extreme == Decimal("0.95")


class TestPortfolioManager:
    def test_init(self):
        from unittest.mock import MagicMock
        loader = MagicMock()
        config = PortfolioConfig()
        pm = PortfolioManager(["BTC/USDT", "ETH/USDT"], config, loader)
        assert len(pm.symbols) == 2

    def test_allocate_sells_bypass(self):
        """SELL signals bypass allocation and are returned directly."""
        from unittest.mock import MagicMock
        loader = MagicMock()
        config = PortfolioConfig()
        pm = PortfolioManager(["BTC/USDT"], config, loader)
        sell_signal = type('Signal', (), {
            'action': type('Action', (), {'value': 'sell'})(),
            'symbol': 'BTC/USDT', 'strength': 0.5, 'price': Decimal("50000"),
            'timestamp': 1704067200000, 'reason': 'test sell',
        })()
        result = pm.allocate_signals([sell_signal], Decimal("10000"), {})
        assert len(result) == 1
        assert result[0].symbol == "BTC/USDT"

    def test_allocate_buys_by_strength(self):
        """BUY signals sorted by strength, limited by equity."""
        from unittest.mock import MagicMock
        loader = MagicMock()
        config = PortfolioConfig(max_position_pct=Decimal("0.2"))
        pm = PortfolioManager(["BTC/USDT", "ETH/USDT"], config, loader)
        weak = type('Signal', (), {
            'action': type('Action', (), {'value': 'buy'})(),
            'symbol': 'ETH/USDT', 'strength': 0.3, 'price': Decimal("3000"),
            'timestamp': 1704067200000, 'reason': 'weak',
        })()
        strong = type('Signal', (), {
            'action': type('Action', (), {'value': 'buy'})(),
            'symbol': 'BTC/USDT', 'strength': 0.9, 'price': Decimal("50000"),
            'timestamp': 1704067200000, 'reason': 'strong',
        })()
        result = pm.allocate_signals([weak, strong], Decimal("10000"), {})
        assert len(result) > 0
        # Stronger signal (BTC) should be allocated more
        btc_allocs = [a for a in result if a.symbol == "BTC/USDT"]
        assert len(btc_allocs) > 0
        assert float(btc_allocs[0].notional) > 0

    def test_correlation_matrix_single_symbol(self):
        from unittest.mock import MagicMock
        loader = MagicMock()
        config = PortfolioConfig()
        pm = PortfolioManager(["BTC/USDT"], config, loader)
        corr = pm.calculate_correlation_matrix()
        assert corr.shape == (1, 1)
        assert corr.iloc[0, 0] == 1.0

    def test_rebalance_needed_empty(self):
        from unittest.mock import MagicMock
        loader = MagicMock()
        config = PortfolioConfig()
        pm = PortfolioManager(["BTC/USDT"], config, loader)
        assert not pm.rebalance_needed([])
