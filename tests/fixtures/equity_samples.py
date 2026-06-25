"""Hardcoded equity curve and trade list samples for metrics testing."""

from decimal import Decimal


def make_sample_equity_curve() -> list[Decimal]:
    """Known equity curve: [10000, 10100, 10200, 10150, 10300, 10250, 10500]."""
    return [
        Decimal("10000"),
        Decimal("10100"),
        Decimal("10200"),
        Decimal("10150"),
        Decimal("10300"),
        Decimal("10250"),
        Decimal("10500"),
    ]


def make_sample_trades() -> list[dict]:
    """4 trades: 2 wins, 1 loss, 1 breakeven."""
    return [
        {"side": "buy", "price": 100.0, "qty": 1.0, "fee": 0.1, "ts": 1704067200000},
        {"side": "sell", "price": 110.0, "qty": 1.0, "fee": 0.11, "ts": 1704070800000},
        {"side": "buy", "price": 105.0, "qty": 1.0, "fee": 0.105, "ts": 1704074400000},
        {"side": "sell", "price": 95.0, "qty": 1.0, "fee": 0.095, "ts": 1704078000000},
    ]


def make_zero_volatility_curve() -> list[Decimal]:
    """Flat equity curve (no change) for edge case testing."""
    return [Decimal("10000")] * 100
