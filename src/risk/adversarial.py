"""Adversarial pre-trade validation — second-opinion check before execution.

Checks 5 information dimensions after risk gates pass but before position sizing.
Each factor scores -25 (opposes trade) to +25 (supports trade). Total weighted
score normalized to -100..+100. Rejects only when negative factors accumulate
beyond configured threshold (lenient mode — single factor cannot veto).

Dimensions:
  1. Sentiment (Fear & Greed contrarian)  — weight 25%
  2. Funding rate overheat                — weight 20%
  3. Long/short ratio crowding            — weight 15%
  4. Multi-timeframe contradiction        — weight 25% (most important)
  5. Volatility regime check (ATR)        — weight 15%
"""

import time
from dataclasses import dataclass
from decimal import Decimal

import numpy as np
import pandas as pd
from loguru import logger

from src.strategy.base import Signal, SignalAction
from src.data.loader import resample_ohlcv


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdversarialFactor:
    """Result of one adversarial factor check."""

    name: str  # e.g. "sentiment", "funding", "mtf_contradiction"
    score: float  # -25 to +25
    weight: float  # configured weight (0.0 to 1.0)
    reason: str  # human-readable
    data_freshness: str  # "live", "cached", "backtest_neutral", "unavailable"
    available: bool = True


@dataclass(frozen=True)
class AdversarialResult:
    """Aggregated result of full adversarial validation."""

    total_score: float  # -100 to +100, weighted sum normalized
    factors: tuple[AdversarialFactor, ...]
    passed: bool  # total_score >= reject_threshold
    warning: bool  # reject_threshold <= total_score < warning_threshold
    reason_summary: str
    position_multiplier: Decimal  # 1.0 (normal), 0.5 (warning), 0.0 (rejected)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class AdversarialValidator:
    """Lenient-mode adversarial pre-trade validator.

    Each factor scores -25 to +25. Weighted sum normalized to -100..+100.
    Reject only when negative factors accumulate beyond threshold.
    Single factors cannot veto — only combined score can reject.

    Graceful degradation: any factor that cannot fetch data returns neutral
    (score=0, available=False). Its weight is redistributed to the remaining
    available factors so the total still reflects genuine signal quality.
    """

    def __init__(self, config, strategy=None, exchange=None, cache_dir: str = "./data"):
        self.config = config  # AdversarialConfig (Pydantic frozen model)
        self.strategy = strategy  # For MTF contradiction check (BaseStrategy)
        self.exchange = exchange  # CCXT exchange instance (None in backtest)
        self._cache_dir = cache_dir
        self._fg_history: list[dict] = []
        self._fg_loaded: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, signal, ohlcv_df=None) -> AdversarialResult:
        """Run all 5 adversarial factor checks.

        Args:
            signal: Signal dataclass from strategy.
            ohlcv_df: pd.DataFrame with OHLCV data (for MTF and volatility checks).

        Returns:
            AdversarialResult with aggregated score and pass/warning/reject decision.
        """
        timestamp_ms = signal.timestamp
        factors: list[AdversarialFactor] = []

        # Each factor wrapped in try/except for graceful degradation.
        # fmt: off
        factor_checks = [
            ("sentiment",         lambda: self._check_sentiment(signal, timestamp_ms)),
            ("funding",           lambda: self._check_funding(signal)),
            ("long_short",        lambda: self._check_long_short(signal)),
            ("mtf_contradiction", lambda: self._check_mtf_contradiction(signal, ohlcv_df)),
            ("volatility",        lambda: self._check_volatility(signal, ohlcv_df)),
        ]
        # fmt: on

        weight_map = {
            "sentiment": self.config.sentiment_weight,
            "funding": self.config.funding_weight,
            "long_short": self.config.long_short_weight,
            "mtf_contradiction": self.config.mtf_weight,
            "volatility": self.config.volatility_weight,
        }

        for name, check_fn in factor_checks:
            try:
                factor = check_fn()
            except Exception as e:
                logger.warning(f"Adversarial factor '{name}' failed: {e}")
                factor = AdversarialFactor(
                    name=name,
                    score=0.0,
                    weight=weight_map.get(name, 0.0),
                    reason=f"Error: {e}",
                    data_freshness="error",
                    available=False,
                )
            factors.append(factor)

        total = self._compute_total(factors)

        # Decision logic
        if total < self.config.reject_threshold:
            passed = False
            warning = False
            mult = Decimal("0.0")
            verdict = "REJECT"
        elif total < self.config.warning_threshold:
            passed = True
            warning = True
            mult = Decimal("0.5")
            verdict = "WARNING"
        else:
            passed = True
            warning = False
            mult = Decimal("1.0")
            verdict = "PASS"

        parts = [f"{f.name}={f.score:+.0f}" for f in factors]
        reason_summary = f"ADV {verdict} [{total:+.1f}]: " + " | ".join(parts)

        # Individual factor debug logging
        for f in factors:
            logger.debug(
                f"ADV {f.name}: {f.score:+.0f} ({f.data_freshness}) — {f.reason}"
            )

        if not passed:
            logger.warning(f"ADVERSARIAL REJECT: {reason_summary}")
        elif warning:
            logger.info(f"ADVERSARIAL WARNING: {reason_summary}")
        else:
            logger.debug(f"ADVERSARIAL PASS: {reason_summary}")

        return AdversarialResult(
            total_score=total,
            factors=tuple(factors),
            passed=passed,
            warning=warning,
            reason_summary=reason_summary,
            position_multiplier=mult,
        )

    # ------------------------------------------------------------------
    # Scoring math
    # ------------------------------------------------------------------

    def _compute_total(self, factors: list[AdversarialFactor]) -> float:
        """Compute normalized weighted score.

        When factors are unavailable, their weight is redistributed among
        available factors so the total still reflects genuine signal quality.
        Score normalized from weighted sum in [-25, +25] to [-100, +100].

        CRITICAL: uses an effective weight floor of 0.6 to prevent
        single-factor amplification. With only one 25%-weight factor
        available, the worst score is -41.7 (barely rejectable), and
        a typical -21 sentiment score gives -35 (warning zone).
        This upholds "no single-factor veto" in lenient mode.
        """
        available = [f for f in factors if f.available]
        if not available:
            return 0.0

        total_avail_weight = sum(f.weight for f in available)
        if total_avail_weight <= 0:
            return 0.0

        weighted_sum = sum(f.score * f.weight for f in available)
        # Floor at 0.6 prevents single-factor amplification:
        #   e.g. only sentiment (w=0.25) at -25 → (-25*0.25)/0.6*4 = -41.7
        #        only sentiment (w=0.25) at -21 → (-21*0.25)/0.6*4 = -35.0 (warning)
        effective_weight = max(0.6, total_avail_weight)
        normalized = (weighted_sum / effective_weight) * 4.0
        return float(max(-100.0, min(100.0, normalized)))

    # ------------------------------------------------------------------
    # Factor 1: Sentiment (Fear & Greed Index, contrarian)
    # ------------------------------------------------------------------

    def _load_fg_history(self) -> None:
        """Lazy-load Fear & Greed historical data (once)."""
        if self._fg_loaded:
            return
        try:
            from src.data.sentiment import fetch_fear_greed_history

            self._fg_history = fetch_fear_greed_history(self._cache_dir)
            self._fg_loaded = True
            if self._fg_history:
                logger.debug(
                    f"Adversarial: loaded {len(self._fg_history)} FG history records"
                )
        except Exception as e:
            logger.warning(f"Adversarial: FG history unavailable ({e})")
            self._fg_loaded = True  # Mark loaded to avoid retry

    def _find_fg_at_timestamp(self, timestamp_ms: int) -> dict | None:
        """Find the Fear & Greed data point closest to the given timestamp."""
        if not self._fg_history:
            return None
        target_s = timestamp_ms // 1000
        best = None
        for entry in self._fg_history:
            entry_ts = int(entry.get("timestamp", 0))
            if entry_ts <= target_s:
                best = entry
            else:
                break
        return best

    def _check_sentiment(self, signal, timestamp_ms: int) -> AdversarialFactor:
        """Contrarian sentiment: extreme greed → oppose buy, extreme fear → support buy.

        Backtest: looks up historical Fear & Greed by timestamp.
        Live/Paper: fetches latest FG from alternative.me API.
        """
        weight = self.config.sentiment_weight

        # Load FG history lazily (needed for backtest timestamp lookup)
        self._load_fg_history()

        fg_value: int | None = None
        freshness = "unavailable"

        if self.exchange is not None:
            # Live/paper mode: fetch current FG
            from src.data.sentiment import fetch_fear_greed_index

            fg_data = fetch_fear_greed_index(self._cache_dir)
            if fg_data:
                fg_value = fg_data.get("value")
                freshness = "live"
        else:
            # Backtest mode: look up by timestamp
            fg_entry = self._find_fg_at_timestamp(timestamp_ms)
            if fg_entry:
                fg_value = fg_entry.get("value")
                freshness = "cached"

        if fg_value is None:
            return AdversarialFactor(
                name="sentiment",
                score=0.0,
                weight=weight,
                reason="Fear & Greed data unavailable",
                data_freshness=freshness,
                available=False,
            )

        # Map FG [0, 100] to score [-25, +25]:
        #   FG = 25  → +25  (extreme fear = contrarian buy opportunity)
        #   FG = 50  →   0  (neutral)
        #   FG = 75  → -25  (extreme greed = contrarian sell warning)
        if fg_value <= 25:
            # Extreme fear zone: full +25
            score = 25.0
        elif fg_value >= 75:
            # Extreme greed zone: full -25
            score = -25.0
        else:
            # Linear interpolation: 25→+25, 50→0, 75→-25
            score = 50.0 - float(fg_value)

        # Invert for SELL signals (if already selling, greed supports it)
        if signal.action == SignalAction.SELL:
            score = -score

        # Classification label
        if fg_value <= 25:
            classification = "Extreme Fear"
        elif fg_value <= 45:
            classification = "Fear"
        elif fg_value <= 55:
            classification = "Neutral"
        elif fg_value <= 75:
            classification = "Greed"
        else:
            classification = "Extreme Greed"

        return AdversarialFactor(
            name="sentiment",
            score=score,
            weight=weight,
            reason=f"FG={fg_value} ({classification})",
            data_freshness=freshness,
        )

    # ------------------------------------------------------------------
    # Factor 2: Funding rate overheat
    # ------------------------------------------------------------------

    def _check_funding(self, signal) -> AdversarialFactor:
        """Funding rate: extremely positive = longs overcrowded → oppose buy.

        Live/paper only. Returns neutral in backtest mode.
        """
        weight = self.config.funding_weight

        if self.exchange is None:
            return AdversarialFactor(
                name="funding",
                score=0.0,
                weight=weight,
                reason="Funding rate unavailable in backtest",
                data_freshness="backtest_neutral",
                available=False,
            )

        from src.data.sentiment import fetch_funding_rate

        fr = fetch_funding_rate(self.exchange, signal.symbol)
        if fr is None:
            return AdversarialFactor(
                name="funding",
                score=0.0,
                weight=weight,
                reason="Funding rate not supported by exchange",
                data_freshness="unavailable",
                available=False,
            )

        fr_pct = float(fr) * 100  # Convert Decimal to percentage

        # Map funding rate to score:
        #   fr_pct < -0.05% → +25  (shorts paying longs = supports buy)
        #   fr_pct > +0.10% → -25  (longs paying shorts = oppose buy)
        #   Linear in between
        if fr_pct <= -0.05:
            score = 25.0
        elif fr_pct >= 0.10:
            score = -25.0
        else:
            # Linear: -0.05→+25, +0.10→-25, range = 0.15
            score = 25.0 - (fr_pct + 0.05) / 0.15 * 50.0

        if signal.action == SignalAction.SELL:
            score = -score

        return AdversarialFactor(
            name="funding",
            score=max(-25.0, min(25.0, score)),
            weight=weight,
            reason=f"Funding rate={fr_pct:+.4f}%",
            data_freshness="live",
        )

    # ------------------------------------------------------------------
    # Factor 3: Long/short ratio crowding
    # ------------------------------------------------------------------

    def _check_long_short(self, signal) -> AdversarialFactor:
        """Long/short ratio: extreme ratio = directional overcrowding → oppose.

        Live/paper only. Returns neutral in backtest mode.
        """
        weight = self.config.long_short_weight

        if self.exchange is None:
            return AdversarialFactor(
                name="long_short",
                score=0.0,
                weight=weight,
                reason="L/S ratio unavailable in backtest",
                data_freshness="backtest_neutral",
                available=False,
            )

        from src.data.sentiment import fetch_long_short_ratio

        ls = fetch_long_short_ratio(self.exchange, signal.symbol)
        if ls is None:
            return AdversarialFactor(
                name="long_short",
                score=0.0,
                weight=weight,
                reason="L/S ratio not supported by exchange",
                data_freshness="unavailable",
                available=False,
            )

        # Map L/S ratio to score:
        #   L/S = 1.0 → +20  (balanced)
        #   L/S = 3.0 → -20  (longs overcrowded)
        #   Linear in between; cap at ±25
        if ls >= 3.0:
            score = -20.0
        else:
            score = 20.0 - (ls - 1.0) * 20.0  # 1→+20, 2→0, 3→-20

        if signal.action == SignalAction.SELL:
            score = -score

        return AdversarialFactor(
            name="long_short",
            score=max(-25.0, min(25.0, score)),
            weight=weight,
            reason=f"Long/short ratio={ls:.2f}",
            data_freshness="live",
        )

    # ------------------------------------------------------------------
    # Factor 4: Multi-timeframe contradiction (MOST IMPORTANT)
    # ------------------------------------------------------------------

    def _check_mtf_contradiction(self, signal, ohlcv_df) -> AdversarialFactor:
        """Run same strategy on higher timeframe to detect divergence.

        If 1h says BUY but 4h says SELL → strong contradiction (-25).
        If both timeframes aligned → confirmation (+15).
        """
        weight = self.config.mtf_weight

        if self.strategy is None:
            return AdversarialFactor(
                name="mtf_contradiction",
                score=0.0,
                weight=weight,
                reason="MTF check unavailable (no strategy reference)",
                data_freshness="unavailable",
                available=False,
            )

        if ohlcv_df is None or len(ohlcv_df) < 10:
            return AdversarialFactor(
                name="mtf_contradiction",
                score=0.0,
                weight=weight,
                reason="MTF check unavailable (insufficient OHLCV data)",
                data_freshness="unavailable",
                available=False,
            )

        higher_tf = self.config.mtf_higher_timeframe  # e.g. "4h"

        try:
            resampled = resample_ohlcv(ohlcv_df, higher_tf)
            if len(resampled) < self.strategy.min_bars:
                return AdversarialFactor(
                    name="mtf_contradiction",
                    score=0.0,
                    weight=weight,
                    reason=(
                        f"Resampled {higher_tf} has only {len(resampled)} bars "
                        f"(need {self.strategy.min_bars})"
                    ),
                    data_freshness="unavailable",
                    available=False,
                )

            # Save/restore strategy state — MTF check must not
            # corrupt the live RSI extremum tracking or other state.
            saved_state = dict(self.strategy.state)
            try:
                mtf_signal = self.strategy.on_bar(signal.symbol, resampled)
            finally:
                self.strategy.state.clear()
                self.strategy.state.update(saved_state)

            if signal.action == SignalAction.BUY:
                if mtf_signal.action == SignalAction.SELL:
                    score = -25.0
                    desc = f"{higher_tf} SELL contradicts {signal.symbol} BUY"
                elif mtf_signal.action == SignalAction.BUY:
                    score = 15.0
                    desc = f"Both timeframes aligned BUY on {higher_tf}"
                else:
                    # HOLD on higher TF → fall back to trend direction
                    score, desc = self._mtf_trend_fallback(
                        resampled, signal, higher_tf,
                    )
            elif signal.action == SignalAction.SELL:
                if mtf_signal.action == SignalAction.BUY:
                    score = -25.0
                    desc = f"{higher_tf} BUY contradicts {signal.symbol} SELL"
                elif mtf_signal.action == SignalAction.SELL:
                    score = 15.0
                    desc = f"Both timeframes aligned SELL on {higher_tf}"
                else:
                    # HOLD on higher TF → fall back to trend direction
                    score, desc = self._mtf_trend_fallback(
                        resampled, signal, higher_tf,
                    )
            else:
                score = 0.0
                desc = "HOLD signal, no MTF check needed"

            return AdversarialFactor(
                name="mtf_contradiction",
                score=score,
                weight=weight,
                reason=f"MTF {desc}",
                data_freshness="live" if self.exchange else "cached",
            )
        except Exception as e:
            logger.warning(f"MTF contradiction check failed: {e}")
            return AdversarialFactor(
                name="mtf_contradiction",
                score=0.0,
                weight=weight,
                reason=f"MTF check error: {e}",
                data_freshness="error",
                available=False,
            )

    def _mtf_trend_fallback(self, resampled_df, signal, higher_tf: str) -> tuple[float, str]:
        """Fallback: when higher-TF strategy returns HOLD, check simple trend direction.

        Uses price vs SMA-20 on the higher timeframe as a directional bias.
        - BUY signal + higher TF uptrend → +10 (mild confirmation)
        - BUY signal + higher TF downtrend → -20 (mild contradiction)
        (Inverted for SELL signals.)
        """
        try:
            close = resampled_df["close"].astype(float)
            if len(close) < 21:
                return 0.0, f"{higher_tf} HOLD (insufficient data for trend fallback)"

            sma20 = close.rolling(20).mean()
            last_close = close.iloc[-1]
            last_sma20 = sma20.iloc[-1]

            if pd.isna(last_sma20):
                return 0.0, f"{higher_tf} HOLD (SMA20 NaN)"

            is_uptrend = last_close > last_sma20

            if signal.action == SignalAction.BUY:
                if is_uptrend:
                    return 10.0, f"{higher_tf} uptrend supports BUY (close={last_close:.1f} > SMA20={last_sma20:.1f})"
                else:
                    return -20.0, f"{higher_tf} downtrend vs BUY (close={last_close:.1f} < SMA20={last_sma20:.1f})"
            else:
                if not is_uptrend:
                    return 10.0, f"{higher_tf} downtrend supports SELL (close={last_close:.1f} < SMA20={last_sma20:.1f})"
                else:
                    return -20.0, f"{higher_tf} uptrend vs SELL (close={last_close:.1f} > SMA20={last_sma20:.1f})"
        except Exception as e:
            logger.warning(f"MTF trend fallback failed: {e}")
            return 0.0, f"{higher_tf} HOLD (trend fallback error: {e})"

    # ------------------------------------------------------------------
    # Factor 5: Volatility regime check (ATR-based)
    # ------------------------------------------------------------------

    def _check_volatility(self, signal, ohlcv_df) -> AdversarialFactor:
        """Volatility spike check using ATR ratio.

        ATR(14) / avg(ATR(14) over last 20 bars) > 2.5 → chaotic, avoid entry.
        """
        weight = self.config.volatility_weight

        if ohlcv_df is None or len(ohlcv_df) < 35:  # 14 for ATR + 20 for avg
            return AdversarialFactor(
                name="volatility",
                score=0.0,
                weight=weight,
                reason="Insufficient data for volatility check (need ≥35 bars)",
                data_freshness="unavailable",
                available=False,
            )

        try:
            high = ohlcv_df["high"].astype(float)
            low = ohlcv_df["low"].astype(float)
            close = ohlcv_df["close"].astype(float)

            # True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # ATR(14) at current bar
            atr14 = tr.iloc[-14:].mean()
            # Average ATR(14) over the 20 bars before the current ATR window
            atr14_rolling = tr.rolling(14).mean()
            atr14_avg = atr14_rolling.iloc[-35:-14].mean()

            if pd.isna(atr14) or pd.isna(atr14_avg) or atr14_avg == 0:
                return AdversarialFactor(
                    name="volatility",
                    score=0.0,
                    weight=weight,
                    reason="ATR calculation returned NaN",
                    data_freshness="unavailable",
                    available=False,
                )

            ratio = float(atr14 / atr14_avg)

            if ratio > 2.5:
                score = -25.0
                desc = f"Extreme volatility spike (ATR ratio={ratio:.1f}x)"
            elif ratio < 0.5:
                score = -10.0
                desc = f"Dead market (ATR ratio={ratio:.1f}x)"
            elif ratio >= 1.5:
                # 1.5→0, 2.5→-25, linear
                score = -(ratio - 1.5) / 1.0 * 25.0
                desc = f"Elevated volatility (ATR ratio={ratio:.2f}x)"
            else:
                # 0.5→-10, 1.5→0, linear
                score = -(1.5 - ratio) / 1.0 * 10.0
                desc = f"Normal/low volatility (ATR ratio={ratio:.2f}x)"

            # Volatility is direction-agnostic — don't invert for SELL
            # (chaotic market is bad for both directions)

            return AdversarialFactor(
                name="volatility",
                score=max(-25.0, min(25.0, score)),
                weight=weight,
                reason=desc,
                data_freshness="cached",
            )
        except Exception as e:
            logger.warning(f"Volatility check failed: {e}")
            return AdversarialFactor(
                name="volatility",
                score=0.0,
                weight=weight,
                reason=f"Volatility check error: {e}",
                data_freshness="error",
                available=False,
            )
