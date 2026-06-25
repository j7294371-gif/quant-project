"""Data pipeline tests: validation, caching, forward-fill."""
import pytest
import pandas as pd
import numpy as np
from decimal import Decimal
from src.data.validator import validate_ohlcv


class TestValidateOHLCV:
    def test_timestamp_monotonicity(self):
        """Duplicate/out-of-order timestamps dropped."""
        df = pd.DataFrame([
            {"timestamp": 1, "open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 10},
            {"timestamp": 2, "open": 101, "high": 102, "low": 100, "close": 101.5, "volume": 10},
            {"timestamp": 3, "open": 102, "high": 103, "low": 101, "close": 102.0, "volume": 10},
            {"timestamp": 2, "open": 99, "high": 100, "low": 98, "close": 99.5, "volume": 10},  # out-of-order
            {"timestamp": 4, "open": 103, "high": 104, "low": 102, "close": 103.0, "volume": 10},
        ])
        result = validate_ohlcv(df)
        assert len(result) == 4  # One dropped

    def test_negative_price_dropped(self):
        df = pd.DataFrame([
            {"timestamp": 1, "open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 10},
            {"timestamp": 2, "open": -5, "high": 102, "low": 100, "close": 101.5, "volume": 10},  # bad
        ])
        result = validate_ohlcv(df)
        assert len(result) == 1

    def test_ohlc_logic_violation(self):
        """low > high should be dropped."""
        df = pd.DataFrame([
            {"timestamp": 1, "open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 10},
            {"timestamp": 2, "open": 100, "high": 95, "low": 101, "close": 100, "volume": 10},  # low > high
        ])
        result = validate_ohlcv(df)
        assert len(result) == 1

    def test_valid_data_passes(self):
        df = pd.DataFrame([
            {"timestamp": 1, "open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 10},
            {"timestamp": 2, "open": 101, "high": 102, "low": 100, "close": 101.5, "volume": 20},
        ])
        result = validate_ohlcv(df)
        assert len(result) == 2

    def test_open_below_low_dropped(self):
        df = pd.DataFrame([
            {"timestamp": 1, "open": 100, "high": 105, "low": 102, "close": 103, "volume": 10},
        ])
        result = validate_ohlcv(df)
        assert len(result) == 0  # open < low violates OHLC logic


class TestCSVCache:
    def test_write_read_roundtrip(self, tmp_path):
        """Write DataFrame to CSV, read back, verify consistency."""
        cache_file = tmp_path / "test.csv"
        df = pd.DataFrame([
            {"timestamp": 1704067200000, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10.0},
        ])
        df.to_csv(cache_file, index=False)
        df2 = pd.read_csv(cache_file)
        assert len(df2) == 1
        assert df2.iloc[0]["close"] == 100.5
