"""Hardcoded OHLCV test data for strategy/execution testing."""

import pandas as pd
import numpy as np


def make_trending_up_200() -> pd.DataFrame:
    """200 bars trending up from 100 to ~120."""
    np.random.seed(42)
    n = 200
    base_ts = 1704067200000
    hour_ms = 3600000
    data = []
    price = 100.0
    for i in range(n):
        price = price + np.random.uniform(-0.3, 0.6)
        price = max(price, 80.0)
        spread = price * np.random.uniform(0.002, 0.01)
        high = price + spread * 1.5
        low = price - spread
        open_p = np.random.uniform(low, high)
        data.append({
            "timestamp": base_ts + i * hour_ms,
            "open": open_p,
            "high": max(high, open_p),
            "low": min(low, open_p),
            "close": price,
            "volume": np.random.uniform(10, 200),
        })
    return pd.DataFrame(data)


def make_ranging_200() -> pd.DataFrame:
    """200 bars ranging around 100."""
    np.random.seed(99)
    n = 200
    base_ts = 1704067200000
    hour_ms = 3600000
    data = []
    for i in range(n):
        close = 100.0 + np.random.uniform(-3, 3)
        high = close + np.random.uniform(0.1, 1)
        low = close - np.random.uniform(0.1, 1)
        data.append({
            "timestamp": base_ts + i * hour_ms,
            "open": np.random.uniform(low, high),
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.uniform(10, 100),
        })
    return pd.DataFrame(data)


def make_bull_bear_cycle_500() -> pd.DataFrame:
    """500 bars with bull-bear cycle for integration testing."""
    np.random.seed(123)
    n = 500
    base_ts = 1704067200000
    hour_ms = 3600000
    data = []
    price = 100.0
    for i in range(n):
        # Cycle: bull 100 bars, bear 50 bars, side 100 bars, repeat
        if i % 250 < 100:
            price = price * (1 + np.random.normal(0.002, 0.01))
        elif i % 250 < 150:
            price = price * (1 + np.random.normal(-0.003, 0.01))
        else:
            price = price * (1 + np.random.normal(0, 0.005))
        price = max(price, 50.0)
        spread = abs(price * np.random.uniform(0.002, 0.015))
        data.append({
            "timestamp": base_ts + i * hour_ms,
            "open": price - spread * np.random.uniform(-0.5, 0.5),
            "high": price + spread,
            "low": price - spread,
            "close": price,
            "volume": np.random.uniform(10, 500),
        })
    return pd.DataFrame(data)
