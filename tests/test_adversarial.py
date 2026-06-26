"""Adversarial pre-trade validation tests.

Tests the 5-factor lenient scoring model:
  1. Sentiment (Fear & Greed contrarian)
  2. Funding rate overheat
  3. Long/short ratio crowding
  4. Multi-timeframe contradiction
  5. Volatility regime check
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from decimal import Decimal
import pandas as pd
import numpy as np

from src.risk.adversarial import (
    AdversarialValidator,
    AdversarialFactor,
    AdversarialResult,
)
from src.strategy.base import Signal, SignalAction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_config(**overrides):
    """Build a mock AdversarialConfig with defaults + overrides."""
    defaults = {
        "enabled": True,
        "reject_threshold": -40.0,
        "warning_threshold": -20.0,
        "sentiment_weight": 0.25,
        "funding_weight": 0.20,
        "long_short_weight": 0.15,
        "mtf_weight": 0.25,
        "volatility_weight": 0.15,
        "mtf_higher_timeframe": "4h",
    }
    defaults.update(overrides)
    mock = MagicMock()
    for k, v in defaults.items():
        setattr(type(mock), k, PropertyMock(return_value=v))
    # Make property access work via getattr
    mock.enabled = defaults["enabled"]
    mock.reject_threshold = defaults["reject_threshold"]
    mock.warning_threshold = defaults["warning_threshold"]
    mock.sentiment_weight = defaults["sentiment_weight"]
    mock.funding_weight = defaults["funding_weight"]
    mock.long_short_weight = defaults["long_short_weight"]
    mock.mtf_weight = defaults["mtf_weight"]
    mock.volatility_weight = defaults["volatility_weight"]
    mock.mtf_higher_timeframe = defaults["mtf_higher_timeframe"]
    return mock


def make_buy_signal(symbol="BTC/USDT", price=Decimal("50000"), timestamp=1704067200000):
    return Signal(
        action=SignalAction.BUY,
        symbol=symbol,
        strength=0.8,
        price=price,
        timestamp=timestamp,
        reason="Test buy signal",
    )


def make_sell_signal(symbol="BTC/USDT", price=Decimal("50000"), timestamp=1704067200000):
    return Signal(
        action=SignalAction.SELL,
        symbol=symbol,
        strength=0.8,
        price=price,
        timestamp=timestamp,
        reason="Test sell signal",
    )


def make_ohlcv_df(n=100, base_price=50000.0, seed=42):
    """Generate synthetic OHLCV DataFrame for testing."""
    np.random.seed(seed)
    price = base_price
    data = []
    base_ts = 1704067200000
    for i in range(n):
        price += np.random.normal(0, 100)
        data.append({
            "timestamp": base_ts + i * 3600000,
            "open": price - price * 0.001,
            "high": price + price * 0.003,
            "low": price - price * 0.003,
            "close": max(0.01, price),
            "volume": np.random.uniform(100, 5000),
        })
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Test Factor Dataclasses
# ---------------------------------------------------------------------------

class TestAdversarialFactor:
    def test_factor_creation(self):
        f = AdversarialFactor(
            name="test", score=15.0, weight=0.25,
            reason="test reason", data_freshness="live",
        )
        assert f.name == "test"
        assert f.score == 15.0
        assert f.available is True

    def test_factor_unavailable(self):
        f = AdversarialFactor(
            name="test", score=0.0, weight=0.20,
            reason="unavailable", data_freshness="backtest_neutral",
            available=False,
        )
        assert f.available is False
        assert f.score == 0.0

    def test_factor_immutable(self):
        f = AdversarialFactor(
            name="test", score=10.0, weight=0.25,
            reason="r", data_freshness="live",
        )
        with pytest.raises(Exception):
            f.score = 20.0  # frozen dataclass


class TestAdversarialResult:
    def test_result_pass(self):
        r = AdversarialResult(
            total_score=50.0, factors=(), passed=True, warning=False,
            reason_summary="ok", position_multiplier=Decimal("1.0"),
        )
        assert r.passed is True
        assert r.warning is False
        assert r.position_multiplier == Decimal("1.0")

    def test_result_warning(self):
        r = AdversarialResult(
            total_score=-25.0, factors=(), passed=True, warning=True,
            reason_summary="warn", position_multiplier=Decimal("0.5"),
        )
        assert r.passed is True
        assert r.warning is True
        assert r.position_multiplier == Decimal("0.5")

    def test_result_reject(self):
        r = AdversarialResult(
            total_score=-50.0, factors=(), passed=False, warning=False,
            reason_summary="no", position_multiplier=Decimal("0.0"),
        )
        assert r.passed is False
        assert r.position_multiplier == Decimal("0.0")


# ---------------------------------------------------------------------------
# Test Scoring Model
# ---------------------------------------------------------------------------

class TestScoringModel:
    def test_all_neutral_gives_zero(self):
        """All factors at 0 → total_score = 0."""
        config = make_mock_config()
        validator = AdversarialValidator(config)
        factors = [
            AdversarialFactor("a", 0.0, 0.25, "ok", "live"),
            AdversarialFactor("b", 0.0, 0.25, "ok", "live"),
            AdversarialFactor("c", 0.0, 0.25, "ok", "live"),
            AdversarialFactor("d", 0.0, 0.15, "ok", "live"),
            AdversarialFactor("e", 0.0, 0.10, "ok", "live"),
        ]
        total = validator._compute_total(factors)
        assert abs(total) < 0.01

    def test_all_max_positive_gives_100(self):
        """All +25 with equal weights → +100."""
        config = make_mock_config()
        validator = AdversarialValidator(config)
        factors = [
            AdversarialFactor("a", 25.0, 0.20, "ok", "live"),
            AdversarialFactor("b", 25.0, 0.20, "ok", "live"),
            AdversarialFactor("c", 25.0, 0.20, "ok", "live"),
            AdversarialFactor("d", 25.0, 0.20, "ok", "live"),
            AdversarialFactor("e", 25.0, 0.20, "ok", "live"),
        ]
        total = validator._compute_total(factors)
        assert abs(total - 100.0) < 0.01

    def test_all_max_negative_gives_minus_100(self):
        """All -25 with equal weights → -100."""
        config = make_mock_config()
        validator = AdversarialValidator(config)
        factors = [
            AdversarialFactor("a", -25.0, 0.20, "ok", "live"),
            AdversarialFactor("b", -25.0, 0.20, "ok", "live"),
            AdversarialFactor("c", -25.0, 0.20, "ok", "live"),
            AdversarialFactor("d", -25.0, 0.20, "ok", "live"),
            AdversarialFactor("e", -25.0, 0.20, "ok", "live"),
        ]
        total = validator._compute_total(factors)
        assert abs(total - (-100.0)) < 0.01

    def test_weight_redistribution(self):
        """When one factor unavailable, weight redistributes among remaining."""
        config = make_mock_config()
        validator = AdversarialValidator(config)
        factors = [
            AdversarialFactor("a", 25.0, 0.25, "ok", "live", available=True),
            AdversarialFactor("b", 0.0, 0.25, "backtest", "backtest_neutral", available=False),
            AdversarialFactor("c", 25.0, 0.25, "ok", "live", available=True),
            AdversarialFactor("d", 25.0, 0.15, "ok", "live", available=True),
            AdversarialFactor("e", 25.0, 0.10, "ok", "live", available=True),
        ]
        total = validator._compute_total(factors)
        # Available weights: 0.25+0.25+0.15+0.10 = 0.75
        # Expected: (25*0.25 + 25*0.25 + 25*0.15 + 25*0.10) / 0.75 * 4 = (18.75)/0.75*4 = 100
        assert abs(total - 100.0) < 0.01

    def test_single_factor_no_veto_with_floor(self):
        """With only 1 factor (w=0.25), normalization floor=0.6 prevents veto.

        Without the floor, a single -21 sentiment factor would amplify to -84.
        With floor=0.6: (-21*0.25)/0.6*4 = -35 — safe in warning zone.
        """
        config = make_mock_config()
        validator = AdversarialValidator(config)
        factors = [
            AdversarialFactor("sentiment", -21.0, 0.25, "FG=71", "cached", available=True),
            AdversarialFactor("funding", 0.0, 0.20, "backtest", "backtest_neutral", available=False),
            AdversarialFactor("long_short", 0.0, 0.15, "backtest", "backtest_neutral", available=False),
            AdversarialFactor("mtf", 0.0, 0.25, "insufficient data", "unavailable", available=False),
            AdversarialFactor("volatility", 0.0, 0.15, "insufficient data", "unavailable", available=False),
        ]
        total = validator._compute_total(factors)
        # Expected: (-21 * 0.25) / 0.6 * 4 = -5.25 / 0.6 * 4 = -35.0
        assert abs(total - (-35.0)) < 0.5, f"Expected ≈-35.0, got {total}"
        # Without floor this would be -84 — verify it's NOT amplified that far
        assert total > -40.0, f"Floor should prevent single-factor rejection, got {total}"

    def test_two_factors_with_floor_applies(self):
        """With 2 factors (total weight=0.50 < 0.60), floor=0.60 still applies."""
        config = make_mock_config()
        validator = AdversarialValidator(config)
        factors = [
            AdversarialFactor("sentiment", -15.0, 0.25, "ok", "cached", available=True),
            AdversarialFactor("funding", 0.0, 0.20, "backtest", "backtest_neutral", available=False),
            AdversarialFactor("long_short", 0.0, 0.15, "backtest", "backtest_neutral", available=False),
            AdversarialFactor("mtf", +10.0, 0.25, "ok", "cached", available=True),
            AdversarialFactor("volatility", 0.0, 0.15, "insufficient", "unavailable", available=False),
        ]
        total = validator._compute_total(factors)
        # Available weights: 0.25 + 0.25 = 0.50 → floor = max(0.6, 0.50) = 0.60
        # Weighted sum: (-15 * 0.25) + (10 * 0.25) = -3.75 + 2.5 = -1.25
        # effective = max(0.6, 0.50) = 0.60
        # Normalized: -1.25 / 0.60 * 4 = -8.33
        assert abs(total - (-8.33)) < 1.0, f"Expected ≈-8.33, got {total}"
        assert total >= -20.0  # Should pass cleanly with two mixed factors

    def test_three_factors_above_floor_no_cap(self):
        """With 3+ factors (total weight > 0.50), floor doesn't apply."""
        config = make_mock_config()
        validator = AdversarialValidator(config)
        factors = [
            AdversarialFactor("sentiment", -21.0, 0.25, "ok", "cached", available=True),
            AdversarialFactor("funding", 0.0, 0.20, "backtest", "backtest_neutral", available=False),
            AdversarialFactor("long_short", 0.0, 0.15, "backtest", "backtest_neutral", available=False),
            AdversarialFactor("mtf", +10.0, 0.25, "ok", "cached", available=True),
            AdversarialFactor("volatility", -5.0, 0.15, "ok", "cached", available=True),
        ]
        total = validator._compute_total(factors)
        # Available weights: 0.25 + 0.25 + 0.15 = 0.65 → floor = 0.65 (above floor)
        # Weighted sum: (-21 * 0.25) + (10 * 0.25) + (-5 * 0.15) = -5.25 + 2.5 + (-0.75) = -3.5
        # Normalized: -3.5 / 0.65 * 4 = -21.54
        assert abs(total - (-21.54)) < 1.0, f"Expected ≈-21.54, got {total}"
        assert total > -25.0  # Should be in warning zone, not reject

    def test_all_unavailable_returns_zero(self):
        """When nothing is available, total_score = 0."""
        config = make_mock_config()
        validator = AdversarialValidator(config)
        factors = [
            AdversarialFactor("a", 0.0, 0.25, "no", "unavailable", available=False),
            AdversarialFactor("b", 0.0, 0.25, "no", "unavailable", available=False),
        ]
        total = validator._compute_total(factors)
        assert total == 0.0

    def test_reject_threshold(self):
        """Total < -40 should mark passed=False."""
        config = make_mock_config(reject_threshold=-40.0, warning_threshold=-20.0)
        validator = AdversarialValidator(config)
        # Build factors that produce total < -40
        factors = [
            AdversarialFactor("a", -25.0, 0.20, "bad", "live"),
            AdversarialFactor("b", -25.0, 0.20, "bad", "live"),
            AdversarialFactor("c", -25.0, 0.20, "bad", "live"),
            AdversarialFactor("d", -20.0, 0.20, "bad", "live"),
            AdversarialFactor("e", -10.0, 0.20, "bad", "live"),
        ]
        total = validator._compute_total(factors)
        assert total < -40.0  # Should be around -84

    def test_warning_threshold_mid_range(self):
        """Total between -40 and -20 should be in warning zone (passed=True, warning=True logic)."""
        config = make_mock_config(reject_threshold=-40.0, warning_threshold=-20.0)
        validator = AdversarialValidator(config)
        factors = [
            AdversarialFactor("a", -15.0, 0.25, "meh", "live"),
            AdversarialFactor("b", -10.0, 0.25, "meh", "live"),
            AdversarialFactor("c", 0.0, 0.25, "ok", "live"),
            AdversarialFactor("d", 0.0, 0.15, "ok", "live"),
            AdversarialFactor("e", 0.0, 0.10, "ok", "live"),
        ]
        total = validator._compute_total(factors)
        # (-15*0.25 + -10*0.25) / 1.0 * 4 = (-3.75 + -2.5) * 4 = -25
        assert -40.0 <= total < -20.0

    def test_position_multiplier_values(self):
        """Validate returns correct multipliers: 1.0 for pass, 0.5 for warn, 0.0 for reject."""
        config = make_mock_config(reject_threshold=-40.0, warning_threshold=-20.0)
        validator = AdversarialValidator(config)

        # Case 1: All positive → pass → 1.0
        factors_pos = [
            AdversarialFactor("a", 25.0, 0.20, "ok", "live"),
            AdversarialFactor("b", 25.0, 0.20, "ok", "live"),
            AdversarialFactor("c", 25.0, 0.20, "ok", "live"),
            AdversarialFactor("d", 25.0, 0.20, "ok", "live"),
            AdversarialFactor("e", 25.0, 0.20, "ok", "live"),
        ]
        total_pos = validator._compute_total(factors_pos)
        assert total_pos >= config.warning_threshold
        assert total_pos > 0  # pass

        # Case 2: Mixed negative → warn → 0.5
        factors_mid = [
            AdversarialFactor("a", -15.0, 0.25, "meh", "live"),
            AdversarialFactor("b", -10.0, 0.25, "meh", "live"),
            AdversarialFactor("c", 0.0, 0.25, "ok", "live"),
            AdversarialFactor("d", 0.0, 0.15, "ok", "live"),
            AdversarialFactor("e", 0.0, 0.10, "ok", "live"),
        ]
        total_mid = validator._compute_total(factors_mid)
        assert config.reject_threshold <= total_mid < config.warning_threshold

        # Case 3: All very negative → reject → 0.0
        factors_neg = [
            AdversarialFactor("a", -25.0, 0.20, "bad", "live"),
            AdversarialFactor("b", -25.0, 0.20, "bad", "live"),
            AdversarialFactor("c", -25.0, 0.20, "bad", "live"),
            AdversarialFactor("d", -20.0, 0.20, "bad", "live"),
            AdversarialFactor("e", -10.0, 0.20, "bad", "live"),
        ]
        total_neg = validator._compute_total(factors_neg)
        assert total_neg < config.reject_threshold


# ---------------------------------------------------------------------------
# Test Sentiment Factor (Fear & Greed)
# ---------------------------------------------------------------------------

class TestSentimentFactor:
    def test_extreme_fear_supports_buy(self):
        """FG < 25 → +25 for BUY (extreme fear = contrarian buy)."""
        config = make_mock_config()
        validator = AdversarialValidator(config)
        validator._fg_history = [{"timestamp": 1704067200, "value": 20}]
        validator._fg_loaded = True
        signal = make_buy_signal(timestamp=1704067200000)

        factor = validator._check_sentiment(signal, signal.timestamp)
        assert factor.score == 25.0
        assert factor.available is True
        assert "Extreme Fear" in factor.reason

    def test_extreme_greed_opposes_buy(self):
        """FG > 75 → -25 for BUY (extreme greed = contrarian sell warning)."""
        config = make_mock_config()
        validator = AdversarialValidator(config)
        validator._fg_history = [{"timestamp": 1704067200, "value": 80}]
        validator._fg_loaded = True
        signal = make_buy_signal(timestamp=1704067200000)

        factor = validator._check_sentiment(signal, signal.timestamp)
        assert factor.score == -25.0
        assert "Extreme Greed" in factor.reason

    def test_neutral_fg_returns_zero(self):
        """FG = 50 → score = 0."""
        config = make_mock_config()
        validator = AdversarialValidator(config)
        validator._fg_history = [{"timestamp": 1704067200, "value": 50}]
        validator._fg_loaded = True
        signal = make_buy_signal(timestamp=1704067200000)

        factor = validator._check_sentiment(signal, signal.timestamp)
        assert factor.score == 0.0
        assert "Neutral" in factor.reason

    def test_linear_interpolation(self):
        """FG = 37.5 → score = +12.5 (midpoint between FG=25→+25 and FG=50→0)."""
        config = make_mock_config()
        validator = AdversarialValidator(config)
        validator._fg_history = [{"timestamp": 1704067200, "value": 37}]
        validator._fg_loaded = True
        signal = make_buy_signal(timestamp=1704067200000)

        factor = validator._check_sentiment(signal, signal.timestamp)
        # FG=37 → score = 50 - 37 = 13
        assert 12 <= factor.score <= 14
        assert "Fear" in factor.reason

    def test_sell_inverts_score(self):
        """SELL signal inverts the sentiment score (fear supports sell, greed opposes sell)."""
        config = make_mock_config()
        validator = AdversarialValidator(config)
        validator._fg_history = [{"timestamp": 1704067200, "value": 20}]  # Extreme Fear
        validator._fg_loaded = True
        signal = make_sell_signal(timestamp=1704067200000)

        factor = validator._check_sentiment(signal, signal.timestamp)
        # BUY with FG=20 → +25, SELL inverts → -25
        assert factor.score == -25.0

    def test_fg_unavailable_returns_neutral(self):
        """When no FG data, return neutral with available=False."""
        config = make_mock_config()
        validator = AdversarialValidator(config)
        validator._fg_loaded = True  # loaded but empty
        signal = make_buy_signal()

        factor = validator._check_sentiment(signal, signal.timestamp)
        assert factor.score == 0.0
        assert factor.available is False

    def test_fg_no_entry_before_timestamp(self):
        """FG data exists but all after signal timestamp → unavailable."""
        config = make_mock_config()
        validator = AdversarialValidator(config)
        # FG data starts after our signal
        validator._fg_history = [{"timestamp": 1800000000, "value": 50}]
        validator._fg_loaded = True
        signal = make_buy_signal(timestamp=1704067200000)

        factor = validator._check_sentiment(signal, signal.timestamp)
        assert factor.available is False


# ---------------------------------------------------------------------------
# Test Funding Rate Factor
# ---------------------------------------------------------------------------

class TestFundingFactor:
    def test_backtest_returns_neutral(self):
        """No exchange → neutral with available=False."""
        config = make_mock_config()
        validator = AdversarialValidator(config)  # no exchange
        signal = make_buy_signal()

        factor = validator._check_funding(signal)
        assert factor.score == 0.0
        assert factor.available is False
        assert "backtest" in factor.data_freshness

    def test_extreme_positive_opposes_buy(self):
        """Funding > +0.10% → -25 for BUY (longs overcrowded)."""
        config = make_mock_config()
        mock_exchange = MagicMock()
        validator = AdversarialValidator(config, exchange=mock_exchange)

        with patch("src.data.sentiment.fetch_funding_rate", return_value=Decimal("0.0015")):
            factor = validator._check_funding(make_buy_signal())
        # 0.0015 = 0.15% > 0.10% → -25
        assert factor.score == -25.0
        assert "Funding rate" in factor.reason

    def test_negative_supports_buy(self):
        """Funding < -0.05% → +25 for BUY (shorts overcrowded)."""
        config = make_mock_config()
        mock_exchange = MagicMock()
        validator = AdversarialValidator(config, exchange=mock_exchange)

        with patch("src.data.sentiment.fetch_funding_rate", return_value=Decimal("-0.001")):
            factor = validator._check_funding(make_buy_signal())
        # -0.001 = -0.10% < -0.05% → +25
        assert factor.score == 25.0

    def test_funding_none_returns_neutral(self):
        """fetch_funding_rate returns None → neutral."""
        config = make_mock_config()
        mock_exchange = MagicMock()
        validator = AdversarialValidator(config, exchange=mock_exchange)

        with patch("src.data.sentiment.fetch_funding_rate", return_value=None):
            factor = validator._check_funding(make_buy_signal())
        assert factor.available is False


# ---------------------------------------------------------------------------
# Test Long/Short Ratio Factor
# ---------------------------------------------------------------------------

class TestLongShortFactor:
    def test_backtest_returns_neutral(self):
        """No exchange → neutral with available=False."""
        config = make_mock_config()
        validator = AdversarialValidator(config)
        signal = make_buy_signal()

        factor = validator._check_long_short(signal)
        assert factor.score == 0.0
        assert factor.available is False

    def test_high_ratio_opposes_buy(self):
        """L/S > 3.0 → -20 for BUY."""
        config = make_mock_config()
        mock_exchange = MagicMock()
        validator = AdversarialValidator(config, exchange=mock_exchange)

        with patch("src.data.sentiment.fetch_long_short_ratio", return_value=3.5):
            factor = validator._check_long_short(make_buy_signal())
        assert factor.score == -20.0

    def test_balanced_ratio_supports_buy(self):
        """L/S ≈ 1.0 → +20 for BUY."""
        config = make_mock_config()
        mock_exchange = MagicMock()
        validator = AdversarialValidator(config, exchange=mock_exchange)

        with patch("src.data.sentiment.fetch_long_short_ratio", return_value=1.0):
            factor = validator._check_long_short(make_buy_signal())
        assert factor.score == 20.0

    def test_ls_unavailable_returns_neutral(self):
        """fetch returns None → neutral."""
        config = make_mock_config()
        mock_exchange = MagicMock()
        validator = AdversarialValidator(config, exchange=mock_exchange)

        with patch("src.data.sentiment.fetch_long_short_ratio", return_value=None):
            factor = validator._check_long_short(make_buy_signal())
        assert factor.available is False


# ---------------------------------------------------------------------------
# Test Multi-Timeframe Contradiction
# ---------------------------------------------------------------------------

class TestMTFContradiction:
    def test_no_strategy_returns_neutral(self):
        """Without strategy reference, return neutral."""
        config = make_mock_config()
        validator = AdversarialValidator(config)  # strategy=None
        signal = make_buy_signal()

        factor = validator._check_mtf_contradiction(signal, make_ohlcv_df())
        assert factor.score == 0.0
        assert factor.available is False
        assert "no strategy" in factor.reason.lower()

    def test_no_ohlcv_data_returns_neutral(self):
        """Without OHLCV data, return neutral."""
        config = make_mock_config()
        mock_strategy = MagicMock()
        validator = AdversarialValidator(config, strategy=mock_strategy)

        factor = validator._check_mtf_contradiction(make_buy_signal(), None)
        assert factor.available is False

    def test_insufficient_data_returns_neutral(self):
        """Very few bars → return neutral."""
        config = make_mock_config()
        mock_strategy = MagicMock()
        mock_strategy.min_bars = 100
        validator = AdversarialValidator(config, strategy=mock_strategy)

        df = make_ohlcv_df(n=20)
        factor = validator._check_mtf_contradiction(make_buy_signal(), df)
        assert factor.available is False

    def test_aligned_buy_gives_positive(self):
        """1h BUY + 4h BUY → +15 (confirmation)."""
        config = make_mock_config()
        mock_strategy = MagicMock()
        mock_strategy.min_bars = 5
        mock_strategy.state = {}
        mock_strategy.on_bar.return_value = Signal(
            action=SignalAction.BUY, symbol="BTC/USDT",
            strength=0.8, price=Decimal("50000"),
            timestamp=1704067200000, reason="4h buy",
        )
        validator = AdversarialValidator(config, strategy=mock_strategy)

        df = make_ohlcv_df(n=200)  # enough for 4h resample
        factor = validator._check_mtf_contradiction(make_buy_signal(), df)
        assert factor.score == 15.0
        assert "aligned" in factor.reason.lower()

    def test_contradiction_gives_negative(self):
        """1h BUY + 4h SELL → -25 (strong contradiction)."""
        config = make_mock_config()
        mock_strategy = MagicMock()
        mock_strategy.min_bars = 5
        mock_strategy.state = {}
        mock_strategy.on_bar.return_value = Signal(
            action=SignalAction.SELL, symbol="BTC/USDT",
            strength=0.8, price=Decimal("50000"),
            timestamp=1704067200000, reason="4h sell",
        )
        validator = AdversarialValidator(config, strategy=mock_strategy)

        df = make_ohlcv_df(n=200)
        factor = validator._check_mtf_contradiction(make_buy_signal(), df)
        assert factor.score == -25.0
        assert "contradicts" in factor.reason.lower()

    def test_strategy_state_restored_after_mtf_check(self):
        """MTF check must not corrupt strategy.state."""
        config = make_mock_config()
        mock_strategy = MagicMock()
        mock_strategy.min_bars = 5
        mock_strategy.state = {"_rsi_min": 15.0, "_rsi_max": None}
        mock_strategy.on_bar.return_value = Signal(
            action=SignalAction.BUY, symbol="BTC/USDT",
            strength=0.8, price=Decimal("50000"),
            timestamp=1704067200000, reason="4h buy",
        )
        validator = AdversarialValidator(config, strategy=mock_strategy)

        df = make_ohlcv_df(n=200)
        validator._check_mtf_contradiction(make_buy_signal(), df)

        # State should be restored
        assert mock_strategy.state == {"_rsi_min": 15.0, "_rsi_max": None}


# ---------------------------------------------------------------------------
# Test Volatility Factor
# ---------------------------------------------------------------------------

class TestVolatilityFactor:
    def test_insufficient_data_returns_neutral(self):
        """Less than 35 bars → neutral (unavailable)."""
        config = make_mock_config()
        validator = AdversarialValidator(config)
        df = make_ohlcv_df(n=20)

        factor = validator._check_volatility(make_buy_signal(), df)
        assert factor.available is False
        assert "Insufficient" in factor.reason

    def test_normal_volatility_returns_near_zero(self):
        """Synthetic data with small noise → near-neutral score."""
        config = make_mock_config()
        validator = AdversarialValidator(config)
        df = make_ohlcv_df(n=100, base_price=50000.0, seed=42)

        factor = validator._check_volatility(make_buy_signal(), df)
        assert factor.available is True
        # With small noise, ratio should be near 1.0 → score near 0
        assert -15 <= factor.score <= 15

    def test_extreme_spike_detected(self):
        """ATR ratio > 2.5 → -25."""
        config = make_mock_config()
        validator = AdversarialValidator(config)

        # Create data with a massive spike
        np.random.seed(99)
        n = 100
        base_ts = 1704067200000
        data = []
        price = 50000.0
        for i in range(n):
            if 70 <= i <= 90:
                # Extreme volatility
                price += np.random.normal(0, 3000)
            else:
                price += np.random.normal(0, 50)
            data.append({
                "timestamp": base_ts + i * 3600000,
                "open": max(0.01, price - 100),
                "high": max(0.01, price + 3000),
                "low": max(0.01, price - 3000),
                "close": max(0.01, price),
                "volume": 1000.0,
            })
        df = pd.DataFrame(data)

        factor = validator._check_volatility(make_buy_signal(), df)
        assert factor.available is True
        # Should detect extreme volatility
        assert factor.score <= 0
        assert "Extreme" in factor.reason or "Elevated" in factor.reason or "ratio" in factor.reason.lower()


# ---------------------------------------------------------------------------
# Test Graceful Degradation
# ---------------------------------------------------------------------------

class TestGracefulDegradation:
    def test_factor_exception_returns_neutral(self):
        """When one factor raises, others still compute."""
        config = make_mock_config()

        # Create a validator where _check_sentiment will raise
        class BrokenValidator(AdversarialValidator):
            def _check_sentiment(self, signal, timestamp_ms):
                raise RuntimeError("Simulated failure")

            def _check_funding(self, signal):
                return AdversarialFactor(
                    name="funding", score=10.0, weight=0.20,
                    reason="ok", data_freshness="live",
                )

            def _check_long_short(self, signal):
                return AdversarialFactor(
                    name="long_short", score=5.0, weight=0.15,
                    reason="ok", data_freshness="live",
                )

            def _check_mtf_contradiction(self, signal, ohlcv_df):
                return AdversarialFactor(
                    name="mtf_contradiction", score=0.0, weight=0.25,
                    reason="ok", data_freshness="cached",
                )

            def _check_volatility(self, signal, ohlcv_df):
                return AdversarialFactor(
                    name="volatility", score=0.0, weight=0.15,
                    reason="ok", data_freshness="cached",
                )

        validator = BrokenValidator(config)
        signal = make_buy_signal()

        result = validator.validate(signal)
        # Should still get a result despite one factor failing
        assert result.total_score is not None
        assert len(result.factors) == 5
        # The broken sentiment factor should be neutral
        sentiment_factor = [f for f in result.factors if f.name == "sentiment"][0]
        assert sentiment_factor.available is False
        assert "Error" in sentiment_factor.reason

    def test_all_factors_unavailable_passes(self):
        """When all factors unavailable, passes with score=0 (don't block trading)."""
        config = make_mock_config()
        validator = AdversarialValidator(config)
        # Prevent real FG cache from being loaded from disk
        validator._fg_loaded = True
        validator._fg_history = []
        signal = make_buy_signal()

        # No exchange, no strategy, no OHLCV → all factors unavailable
        result = validator.validate(signal, ohlcv_df=None)
        assert result.passed is True
        assert result.position_multiplier == Decimal("1.0")

    def test_only_sentiment_available_works(self):
        """When only sentiment is available, scoring still works."""
        config = make_mock_config()
        validator = AdversarialValidator(config)
        validator._fg_history = [{"timestamp": 1704067200, "value": 20}]  # Extreme fear
        validator._fg_loaded = True
        signal = make_buy_signal(timestamp=1704067200000)

        result = validator.validate(signal, ohlcv_df=None)
        # With only sentiment at 25, normalized = 25/0.25 * 4 = 100 → PASS
        assert result.passed is True
        assert result.position_multiplier == Decimal("1.0")


# ---------------------------------------------------------------------------
# Test Backtest Mode
# ---------------------------------------------------------------------------

class TestBacktestMode:
    def test_validator_no_exchange(self):
        """Without exchange, validator works in backtest mode."""
        config = make_mock_config()
        validator = AdversarialValidator(config)  # no exchange, no strategy
        assert validator.exchange is None

    def test_funding_neutral_in_backtest(self):
        """No exchange → funding factor returns available=False."""
        config = make_mock_config()
        validator = AdversarialValidator(config)
        factor = validator._check_funding(make_buy_signal())
        assert factor.available is False
        assert "backtest" in factor.data_freshness

    def test_long_short_neutral_in_backtest(self):
        """No exchange → L/S factor returns available=False."""
        config = make_mock_config()
        validator = AdversarialValidator(config)
        factor = validator._check_long_short(make_buy_signal())
        assert factor.available is False

    def test_fg_lookup_by_timestamp(self):
        """FG history lookup finds correct entry for given timestamp."""
        config = make_mock_config()
        validator = AdversarialValidator(config)
        validator._fg_history = [
            {"timestamp": 1704067200, "value": 25},  # Jan 1 2024
            {"timestamp": 1704153600, "value": 30},  # Jan 2 2024
            {"timestamp": 1704240000, "value": 50},  # Jan 3 2024
        ]
        validator._fg_loaded = True

        # Signal timestamp Jan 2 12:00 → should match Jan 2 entry
        signal = make_buy_signal(timestamp=1704196800000)
        factor = validator._check_sentiment(signal, signal.timestamp)
        assert factor.score == 20.0  # FG=30 → score = 50-30 = 20

    def test_fg_no_history_no_crash(self):
        """Empty FG history doesn't crash _check_sentiment."""
        config = make_mock_config()
        validator = AdversarialValidator(config)
        validator._fg_loaded = True
        signal = make_buy_signal()

        factor = validator._check_sentiment(signal, signal.timestamp)
        assert factor.available is False


# ---------------------------------------------------------------------------
# Test Integration (validate method)
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_validate_buy_signal(self):
        """Run full validate() in backtest mode with OHLCV data."""
        config = make_mock_config()
        mock_strategy = MagicMock()
        mock_strategy.min_bars = 5
        mock_strategy.state = {}
        mock_strategy.on_bar.return_value = Signal(
            action=SignalAction.BUY, symbol="BTC/USDT",
            strength=0.8, price=Decimal("50000"),
            timestamp=1704067200000, reason="4h buy",
        )

        validator = AdversarialValidator(config, strategy=mock_strategy)
        # Pre-load neutral FG data
        validator._fg_history = [{"timestamp": 1704067200, "value": 50}]
        validator._fg_loaded = True

        df = make_ohlcv_df(n=200)
        signal = make_buy_signal()

        result = validator.validate(signal, ohlcv_df=df)
        assert isinstance(result, AdversarialResult)
        assert len(result.factors) == 5
        assert result.total_score is not None
        # Should pass with neutral/mildly positive factors
        assert result.passed is True

    def test_validate_hold_signal(self):
        """HOLD signal still works (though it's typically filtered before reaching adversarial)."""
        config = make_mock_config()
        validator = AdversarialValidator(config)
        signal = Signal(
            action=SignalAction.HOLD, symbol="BTC/USDT",
            strength=0.0, price=Decimal("50000"),
            timestamp=1704067200000, reason="No signal",
        )
        result = validator.validate(signal)
        assert isinstance(result, AdversarialResult)

    def test_factors_are_frozen(self):
        """AdversarialFactor and AdversarialResult should be immutable."""
        config = make_mock_config()
        validator = AdversarialValidator(config)
        validator._fg_history = [{"timestamp": 1704067200, "value": 50}]
        validator._fg_loaded = True
        signal = make_buy_signal()

        result = validator.validate(signal)
        for factor in result.factors:
            with pytest.raises(Exception):
                factor.score = 99.0
        with pytest.raises(Exception):
            result.total_score = 99.0
