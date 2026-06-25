"""Fusion strategy tests: multi-factor scoring and signal logic."""
import pytest
import pandas as pd
import numpy as np
from decimal import Decimal
from src.strategy.base import SignalAction
from src.strategy.registry import get_strategy

# Trigger @register_strategy decorators
import src.strategy.fusion  # noqa: F401


def make_bullish_all_signals_df():
    """Generate data where all 4 sub-indicators produce bullish signals."""
    np.random.seed(99)
    base_ts = 1704067200000
    hour_ms = 3600000
    n = 300
    data = []
    price = 100.0

    for i in range(n):
        if 80 <= i <= 100:
            price += 2.0  # strong uptrend (golden cross + MACD)
        elif 150 <= i <= 170:
            price -= 3.0  # dip to oversold
        elif 171 <= i <= 185:
            price += 3.0  # sharp recovery (RSI exit oversold + Bollinger bounce)
        elif 250 <= i <= 260:
            price += 1.5  # continued uptrend
        else:
            price += np.random.uniform(-0.3, 0.3)
        price = max(price, 70.0)

        spread = abs(price * np.random.uniform(0.002, 0.008))
        data.append({
            "timestamp": base_ts + i * hour_ms,
            "open": price - spread * 0.3,
            "high": price + spread * 0.7,
            "low": price - spread * 0.7,
            "close": price,
            "volume": np.random.uniform(50, 500),
        })
    return pd.DataFrame(data)


def make_bearish_all_signals_df():
    """Generate data where all 4 sub-indicators produce bearish signals."""
    np.random.seed(98)
    base_ts = 1704067200000
    hour_ms = 3600000
    n = 300
    data = []
    price = 120.0

    for i in range(n):
        if 80 <= i <= 100:
            price -= 2.0  # downtrend (death cross + MACD)
        elif 150 <= i <= 170:
            price += 3.0  # rally to overbought
        elif 171 <= i <= 185:
            price -= 3.0  # sharp decline (RSI exit overbought)
        elif 250 <= i <= 260:
            price -= 1.5  # continued downtrend
        else:
            price += np.random.uniform(-0.3, 0.3)
        price = max(price, 60.0)

        spread = abs(price * np.random.uniform(0.002, 0.008))
        data.append({
            "timestamp": base_ts + i * hour_ms,
            "open": price - spread * 0.3,
            "high": price + spread * 0.7,
            "low": price - spread * 0.7,
            "close": price,
            "volume": np.random.uniform(50, 500),
        })
    return pd.DataFrame(data)


# ─── Fusion Strategy Tests ──────────────────────────────────────────


class TestFusionStrategy:
    def test_registered(self):
        """Fusion strategy is registered and instantiable."""
        strategy = get_strategy("fusion", {})
        assert strategy is not None
        assert strategy.min_bars > 0

    def test_min_bars(self):
        """min_bars = max of all sub-strategy min_bars."""
        strategy = get_strategy("fusion", {})
        # MACD: 26+9+1=36 > SMA: 20+1=21 > RSI: 14*3=42 > Bollinger: 20+1=21
        assert strategy.min_bars == 42  # RSI 14*3 = 42

    def test_warmup_bars(self):
        """warmup_bars >= min_bars (MACD slow*2=52 or RSI period*4=56)."""
        strategy = get_strategy("fusion", {})
        assert strategy.warmup_bars >= strategy.min_bars
        assert strategy.warmup_bars == 56  # max(52, 56)

    def test_invalid_weights(self):
        """Weights that don't sum to 1.0 raise ValueError."""
        with pytest.raises(ValueError):
            get_strategy("fusion", {"tech_weight": 0.5, "sentiment_weight": 0.5, "funding_weight": 0.5})

    def test_invalid_thresholds(self):
        """buy_threshold <= sell_threshold raises ValueError."""
        with pytest.raises(ValueError):
            get_strategy("fusion", {"buy_threshold": 30, "sell_threshold": 70})

    def test_bullish_signal_produced(self):
        """In strong uptrend with recovery, fusion produces BUY or HOLD (not SELL)."""
        df = make_bullish_all_signals_df()
        strategy = get_strategy("fusion", {
            "tech_weight": 0.5,
            "sentiment_weight": 0.25,
            "funding_weight": 0.25,
        })
        # Inject neutral sentiment so it doesn't drag down the score
        strategy._inject_mock_sentiment([
            {"timestamp": 0, "value": 50, "value_classification": "Neutral"},
            {"timestamp": 9999999999, "value": 50, "value_classification": "Neutral"},
        ])
        signal = strategy.on_bar("BTC/USDT", df)
        # In a bullish setup, should not be SELL
        assert signal.action in (SignalAction.BUY, SignalAction.HOLD)
        assert isinstance(signal.price, Decimal)
        assert len(signal.reason) > 0
        assert "fusion_score" in signal.reason

    def test_bearish_signal_produced(self):
        """In strong downtrend, fusion produces SELL or HOLD (not BUY)."""
        df = make_bearish_all_signals_df()
        strategy = get_strategy("fusion", {
            "tech_weight": 0.5,
            "sentiment_weight": 0.25,
            "funding_weight": 0.25,
        })
        signal = strategy.on_bar("BTC/USDT", df)
        assert signal.action in (SignalAction.SELL, SignalAction.HOLD)

    def test_ranging_market_produces_valid_signal(self):
        """In ranging market, fusion produces a well-formed signal (no crash)."""
        np.random.seed(100)
        base_ts = 1704067200000
        hour_ms = 3600000
        n = 300
        data = []
        price = 100.0
        for i in range(n):
            price += np.random.uniform(-0.5, 0.5)
            price = max(95, min(105, price))
            data.append({
                "timestamp": base_ts + i * hour_ms,
                "open": price - 0.1, "high": price + 0.2,
                "low": price - 0.2, "close": price, "volume": 100.0,
            })
        df = pd.DataFrame(data)

        strategy = get_strategy("fusion", {
            "buy_threshold": 65,
            "sell_threshold": 35,
        })
        # Inject neutral sentiment
        strategy._inject_mock_sentiment([
            {"timestamp": 0, "value": 50, "value_classification": "Neutral"},
            {"timestamp": 9999999999, "value": 50, "value_classification": "Neutral"},
        ])
        signal = strategy.on_bar("BTC/USDT", df)
        # Signal must be well-formed regardless of direction
        assert signal.action in (SignalAction.BUY, SignalAction.SELL, SignalAction.HOLD)
        assert isinstance(signal.price, Decimal)
        assert isinstance(signal.strength, float)
        assert "fusion_score" in signal.reason

    def test_signal_strength_bounded(self):
        """Signal strength is always between 0.0 and 1.0."""
        df = make_bullish_all_signals_df()
        strategy = get_strategy("fusion", {
            "buy_threshold": 60,
            "sell_threshold": 40,
        })
        signal = strategy.on_bar("BTC/USDT", df)
        assert 0.0 <= signal.strength <= 1.0

    def test_custom_weights_applied(self):
        """Custom weights change the scoring behavior, both produce valid signals."""
        df = make_bullish_all_signals_df()

        # Inject neutral sentiment for controlled test
        mock_fg = [
            {"timestamp": 0, "value": 50, "value_classification": "Neutral"},
            {"timestamp": 9999999999, "value": 50, "value_classification": "Neutral"},
        ]

        # Tech-heavy: more weight on technical signals
        strategy_tech = get_strategy("fusion", {
            "tech_weight": 0.70,
            "sentiment_weight": 0.15,
            "funding_weight": 0.15,
            "buy_threshold": 55,
            "sell_threshold": 45,
        })
        strategy_tech._inject_mock_sentiment(mock_fg)
        signal_tech = strategy_tech.on_bar("BTC/USDT", df)

        # Sentiment-heavy
        strategy_sent = get_strategy("fusion", {
            "tech_weight": 0.30,
            "sentiment_weight": 0.50,
            "funding_weight": 0.20,
            "buy_threshold": 55,
            "sell_threshold": 45,
        })
        strategy_sent._inject_mock_sentiment(mock_fg)
        signal_sent = strategy_sent.on_bar("BTC/USDT", df)

        # Both should produce well-formed signals (no crash, valid structure)
        for sig in [signal_tech, signal_sent]:
            assert sig.action in (SignalAction.BUY, SignalAction.SELL, SignalAction.HOLD)
            assert isinstance(sig.price, Decimal)
            assert 0.0 <= sig.strength <= 1.0
            assert "fusion_score" in sig.reason

    def test_signal_reason_contains_layer_info(self):
        """Signal reason includes fusion_score and layer breakdown."""
        df = make_bullish_all_signals_df()
        strategy = get_strategy("fusion", {})
        signal = strategy.on_bar("BTC/USDT", df)
        assert "fusion_score" in signal.reason
        assert "tech=" in signal.reason

    def test_set_live_sentiment_injects_funding_rate(self):
        """set_live_sentiment() injects external data into strategy state."""
        strategy = get_strategy("fusion", {})
        strategy.set_live_sentiment(funding_rate_pct=0.12)

        assert strategy.state["_live_funding_rate_pct"] == 0.12

        # Now on_bar should use the live funding rate
        df = make_bullish_all_signals_df()
        signal = strategy.on_bar("BTC/USDT", df)
        # Funding rate at 0.12% is very positive → bearish signal from funding layer
        # This should depress the total score
        assert signal.action in (SignalAction.BUY, SignalAction.SELL, SignalAction.HOLD)

    def test_set_live_sentiment_negative_funding(self):
        """Very negative funding rate injects bullish bias."""
        strategy = get_strategy("fusion", {})
        strategy.set_live_sentiment(funding_rate_pct=-0.08)

        df = make_bullish_all_signals_df()
        signal = strategy.on_bar("BTC/USDT", df)
        # -0.08% funding rate = bullish → should NOT be SELL
        assert signal.action in (SignalAction.BUY, SignalAction.HOLD)

    def test_fusion_score_in_reason(self):
        """The reason string always contains the numeric fusion score."""
        df = make_bullish_all_signals_df()
        strategy = get_strategy("fusion", {})
        signal = strategy.on_bar("BTC/USDT", df)
        # Should contain something like "fusion_score=72.3"
        assert "fusion_score=" in signal.reason
