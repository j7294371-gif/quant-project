"""Configuration loading tests."""
import pytest
import os
import yaml
from pathlib import Path


class TestConfigLoading:
    def test_valid_config_loads(self, tmp_path):
        """AppConfig.load() with minimal valid config."""
        from src.utils.config import AppConfig
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        # Write minimal configs
        (config_dir / "app.yaml").write_text("log_level: INFO\nlog_dir: ./logs\nlog_retention_days: 30\ndata_cache_dir: ./data\nstate_dir: ./state\n")
        (config_dir / "strategy.yaml").write_text("active: sma_cross\nsma_cross:\n  short_window: 5\n  long_window: 20\nmacd:\n  fast: 12\n  slow: 26\n  signal: 9\nrsi:\n  period: 14\n  oversold: 30\n  overbought: 70\nbollinger:\n  period: 20\n  std_dev: 2.0\n")
        (config_dir / "risk.yaml").write_text("max_position_pct: 0.2\nmax_drawdown_limit: 0.3\ndaily_loss_limit: 0.05\ncircuit_cooldown_hours: 24\nposition_method: fixed_pct\nrisk_per_trade_pct: 0.01\n")
        (config_dir / "exchange.yaml").write_text("exchange: binance\ntestnet: true\nsymbols:\n  - BTC/USDT\ntimeframe: 1h\n")
        config = AppConfig.load(str(config_dir))
        assert config.exchange.exchange == "binance"
        assert config.strategy.active == "sma_cross"
        assert float(config.risk.max_position_pct) == 0.2

    def test_invalid_percentage_raises(self, tmp_path):
        """Risk percentage > 1.0 should cause exit."""
        from src.utils.config import AppConfig
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "app.yaml").write_text("log_level: INFO\nlog_dir: ./logs\nlog_retention_days: 30\ndata_cache_dir: ./data\nstate_dir: ./state\n")
        (config_dir / "strategy.yaml").write_text("active: sma_cross\nsma_cross:\n  short_window: 5\n  long_window: 20\nmacd:\n  fast: 12\n  slow: 26\n  signal: 9\nrsi:\n  period: 14\n  oversold: 30\n  overbought: 70\nbollinger:\n  period: 20\n  std_dev: 2.0\n")
        (config_dir / "risk.yaml").write_text("max_position_pct: 2.5\nmax_drawdown_limit: 0.3\ndaily_loss_limit: 0.05\ncircuit_cooldown_hours: 24\nposition_method: fixed_pct\nrisk_per_trade_pct: 0.01\n")
        (config_dir / "exchange.yaml").write_text("exchange: binance\ntestnet: true\nsymbols:\n  - BTC/USDT\ntimeframe: 1h\n")
        with pytest.raises(SystemExit):
            AppConfig.load(str(config_dir))

    def test_missing_config_dir(self):
        from src.utils.config import AppConfig
        with pytest.raises(SystemExit):
            AppConfig.load("/nonexistent/path")

    def test_invalid_symbol_format(self, tmp_path):
        """Symbols without '/' should fail validation."""
        from src.utils.config import ExchangeConfig
        with pytest.raises(Exception):
            ExchangeConfig(exchange="binance", testnet=True, symbols=["BTCUSDT"], timeframe="1h")

    def test_valid_symbols(self):
        from src.utils.config import ExchangeConfig
        config = ExchangeConfig(exchange="binance", testnet=True, symbols=["BTC/USDT", "ETH/USDT"], timeframe="1h")
        assert len(config.symbols) == 2
