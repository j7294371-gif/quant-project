"""Shared pytest fixtures and helper functions for quant project tests."""

import pytest
import pandas as pd
import numpy as np
from decimal import Decimal


@pytest.fixture
def sample_trending_up_df() -> pd.DataFrame:
    """
    Generate 200 bars of trending-up OHLCV data.
    close rises from 100 to ~120 linearly with ±1% noise per bar.
    1-hour interval, starting from 2024-01-01 UTC.
    """
    np.random.seed(42)
    n = 200
    base_ts = 1704067200000  # 2024-01-01 00:00:00 UTC in ms
    hour_ms = 3600000

    closes = []
    price = 100.0
    for i in range(n):
        drift = 100.0 * 0.0005 * (i + 1)  # Upward drift
        noise = np.random.normal(0, 0.5)
        price = 100.0 + drift + noise
        closes.append(max(price, 50.0))

    data = []
    for i, close in enumerate(closes):
        spread = close * np.random.uniform(0.001, 0.01)
        high = close + spread
        low = close - spread
        open_p = close - np.random.uniform(-spread, spread)
        volume = np.random.uniform(1, 100)

        data.append({
            "timestamp": base_ts + i * hour_ms,
            "open": open_p,
            "high": max(high, open_p, close),
            "low": min(low, open_p, close),
            "close": close,
            "volume": volume,
        })

    return pd.DataFrame(data)


@pytest.fixture
def sample_ranging_df() -> pd.DataFrame:
    """
    Generate 200 bars of ranging/sideways OHLCV data.
    close oscillates around 100 ± 2, no clear trend.
    """
    np.random.seed(99)
    n = 200
    base_ts = 1704067200000
    hour_ms = 3600000

    data = []
    for i in range(n):
        close = 100.0 + np.random.uniform(-2, 2)
        spread = close * np.random.uniform(0.001, 0.005)
        high = close + spread
        low = close - spread
        open_p = close - np.random.uniform(-spread, spread)

        data.append({
            "timestamp": base_ts + i * hour_ms,
            "open": open_p,
            "high": max(high, open_p, close),
            "low": min(low, open_p, close),
            "close": close,
            "volume": np.random.uniform(1, 50),
        })

    return pd.DataFrame(data)


@pytest.fixture
def sample_extreme_df() -> pd.DataFrame:
    """
    Generate 200 bars of extreme market data:
    - 30% crash (test circuit breaker)
    - V-shaped recovery (test stop-loss + re-entry)
    """
    np.random.seed(7)
    n = 200
    base_ts = 1704067200000
    hour_ms = 3600000

    data = []
    price = 100.0
    for i in range(n):
        if 50 <= i < 70:
            # Crash phase: -30%
            price = price * 0.97 + np.random.normal(0, 0.3)
        elif 70 <= i < 120:
            # Recovery phase: V-shape
            price = price * 1.02 + np.random.normal(0, 0.5)
        else:
            price = price + np.random.normal(0, 0.3)

        price = max(price, 30.0)
        spread = price * np.random.uniform(0.005, 0.02)
        high = price + spread
        low = max(price - spread, 1.0)
        open_p = np.random.uniform(low, high)

        data.append({
            "timestamp": base_ts + i * hour_ms,
            "open": open_p,
            "high": high,
            "low": low,
            "close": price,
            "volume": np.random.uniform(1, 200),
        })

    return pd.DataFrame(data)


@pytest.fixture
def sample_equity_curve() -> list[Decimal]:
    """Known equity curve for metrics testing."""
    return [
        Decimal("10000"), Decimal("10100"), Decimal("10200"),
        Decimal("10150"), Decimal("10300"), Decimal("10250"),
        Decimal("10500"),
    ]


@pytest.fixture
def sample_trades() -> list[dict]:
    """4 known trades: 2 wins, 1 loss, 1 breakeven."""
    return [
        {"side": "buy", "price": 100.0, "qty": 1.0, "fee": 0.1, "ts": 1704067200000},
        {"side": "sell", "price": 110.0, "qty": 1.0, "fee": 0.11, "ts": 1704070800000},  # +9.79 win
        {"side": "buy", "price": 105.0, "qty": 1.0, "fee": 0.105, "ts": 1704074400000},
        {"side": "sell", "price": 95.0, "qty": 1.0, "fee": 0.095, "ts": 1704078000000},  # -10.20 loss
    ]


@pytest.fixture
def temp_state_dir(tmp_path) -> str:
    """Temporary state directory, auto-cleaned after test."""
    return str(tmp_path / "state")
