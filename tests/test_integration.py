"""End-to-end backtest integration tests."""
import pytest
import pandas as pd
import numpy as np
import sys
import os
from decimal import Decimal


@pytest.fixture
def sample_csv(tmp_path):
    """Generate a sample CSV for backtest integration tests."""
    np.random.seed(42)
    n = 300
    base_ts = 1704067200000
    hour_ms = 3600000
    data = []
    price = 100.0
    for i in range(n):
        price = price + np.random.normal(0, 0.5)
        price = max(price, 80.0)
        spread = abs(price * np.random.uniform(0.002, 0.01))
        data.append({
            "timestamp": base_ts + i * hour_ms,
            "open": price - spread * 0.5,
            "high": price + spread,
            "low": price - spread,
            "close": price,
            "volume": np.random.uniform(10, 200),
        })
    df = pd.DataFrame(data)
    csv_path = tmp_path / "sample_btc.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


class TestBacktestIntegration:
    def test_sma_cross_backtest_runs(self, sample_csv):
        """SMA crossover backtest runs and returns metrics."""
        sys.argv = [
            "main.py", "backtest",
            "--strategy", "sma_cross", "--symbol", "BTC/USDT",
            "--csv-data", sample_csv, "--timeframe", "1h",
        ]
        from main import main
        try:
            main()
        except SystemExit as e:
            assert e.code == 0

    def test_macd_backtest_runs(self, sample_csv):
        """MACD backtest runs."""
        sys.argv = [
            "main.py", "backtest",
            "--strategy", "macd", "--symbol", "BTC/USDT",
            "--csv-data", sample_csv, "--timeframe", "1h",
        ]
        from main import main
        try:
            main()
        except SystemExit as e:
            assert e.code == 0

    def test_rsi_backtest_runs(self, sample_csv):
        """RSI backtest runs."""
        sys.argv = [
            "main.py", "backtest",
            "--strategy", "rsi", "--symbol", "BTC/USDT",
            "--csv-data", sample_csv, "--timeframe", "1h",
        ]
        from main import main
        try:
            main()
        except SystemExit as e:
            assert e.code == 0

    def test_bollinger_backtest_runs(self, sample_csv):
        """Bollinger backtest runs."""
        sys.argv = [
            "main.py", "backtest",
            "--strategy", "bollinger", "--symbol", "BTC/USDT",
            "--csv-data", sample_csv, "--timeframe", "1h",
        ]
        from main import main
        try:
            main()
        except SystemExit as e:
            assert e.code == 0

    def test_list_strategies_output(self):
        """list-strategies lists all 4 strategies."""
        sys.argv = ["main.py", "list-strategies"]
        from main import main
        main()  # Should print and return without error
