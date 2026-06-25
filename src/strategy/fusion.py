"""Multi-factor fusion strategy.

Combines 3 layers into a weighted 0-100 score:
  Layer 1 — Technical (40%): SMA + MACD + RSI + Bollinger via shared indicators module
  Layer 2 — Sentiment (30%): Fear & Greed Index (contrarian)
  Layer 3 — Market Structure (30%): Funding rate / price-deviation proxy

Fusion signals fire only when multiple dimensions confirm.
"""

import time
from decimal import Decimal

import pandas as pd
from loguru import logger

from src.strategy.base import BaseStrategy, Signal, SignalAction
from src.strategy.registry import register_strategy
from src.strategy.indicators import (
    compute_sma, detect_sma_cross,
    compute_macd, detect_macd_cross,
    compute_rsi, detect_rsi_signal,
    compute_bollinger, detect_bollinger_cross,
    extract_bar_info,
)
from src.data.sentiment import fetch_fear_greed_history


@register_strategy("fusion")
class FusionStrategy(BaseStrategy):
    """Multi-factor strategy: tech + sentiment + market structure."""

    def _validate_params(self) -> None:
        w_tech = float(self.params.get("tech_weight", 0.40))
        w_sent = float(self.params.get("sentiment_weight", 0.30))
        w_fund = float(self.params.get("funding_weight", 0.30))
        total = w_tech + w_sent + w_fund
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"Weights must sum to 1.0, got tech={w_tech}+sent={w_sent}+fund={w_fund}={total}"
            )
        buy_threshold = float(self.params.get("buy_threshold", 65))
        sell_threshold = float(self.params.get("sell_threshold", 35))
        if buy_threshold <= sell_threshold:
            raise ValueError(
                f"buy_threshold ({buy_threshold}) must be > sell_threshold ({sell_threshold})"
            )

    @property
    def min_bars(self) -> int:
        slow = int(self.params.get("macd_slow", 26))
        signal_period = int(self.params.get("macd_signal", 9))
        return max(
            int(self.params.get("sma_long", 20)) + 1,
            slow + signal_period + 1,
            int(self.params.get("rsi_period", 14)) * 3,
            int(self.params.get("bollinger_period", 20)) + 1,
        )

    @property
    def warmup_bars(self) -> int:
        slow = int(self.params.get("macd_slow", 26))
        rsi_period = int(self.params.get("rsi_period", 14))
        return max(slow * 2, rsi_period * 4)

    def __init__(self, params: dict):
        super().__init__(params)
        self._fg_history: list[dict] = []
        self._fg_loaded = False
        self._cache_dir = params.get("cache_dir", "./data")

    def _ensure_fg_loaded(self) -> None:
        if self._fg_loaded:
            return
        try:
            self._fg_history = fetch_fear_greed_history(self._cache_dir)
            self._fg_loaded = True
            if self._fg_history:
                logger.info(f"Fusion: loaded {len(self._fg_history)} Fear & Greed records")
        except Exception as e:
            logger.warning(f"Fusion: Fear & Greed history unavailable ({e}), sentiment layer disabled")
            self._fg_loaded = True

    def _inject_mock_sentiment(self, fg_data: list[dict] | None = None) -> None:
        """Inject mock Fear & Greed data for testing."""
        self._fg_history = fg_data if fg_data is not None else []
        self._fg_loaded = True

    def on_bar(self, symbol: str, df: pd.DataFrame) -> Signal:
        price, timestamp = extract_bar_info(df)
        close = df["close"]

        # ─── Layer 1: Technical (shared indicators, no duplication) ──
        tech_score = self._compute_tech_score(close)

        # ─── Layer 2: Sentiment ──────────────────────────────────────
        sentiment_score = self._compute_sentiment_score(timestamp)

        # ─── Layer 3: Market Structure ───────────────────────────────
        funding_score = self._compute_funding_score(close)

        # ─── Weighted fusion ─────────────────────────────────────────
        w_tech = float(self.params.get("tech_weight", 0.40))
        w_sent = float(self.params.get("sentiment_weight", 0.30))
        w_fund = float(self.params.get("funding_weight", 0.30))

        if sentiment_score is None:
            w_tech += w_sent
            w_sent = 0.0
            sentiment_score = 50
        if funding_score is None:
            w_tech += w_fund
            w_fund = 0.0
            funding_score = 50

        total_score = (w_tech * tech_score + w_sent * sentiment_score + w_fund * funding_score)

        buy_threshold = float(self.params.get("buy_threshold", 65))
        sell_threshold = float(self.params.get("sell_threshold", 35))

        parts = [f"fusion_score={total_score:.1f}",
                 f"tech={tech_score:.0f}(w={w_tech:.2f})"]
        if w_sent > 0:
            parts.append(f"sentiment={sentiment_score:.0f}(w={w_sent:.2f})")
        if w_fund > 0:
            parts.append(f"funding={funding_score:.0f}(w={w_fund:.2f})")
        reason = " | ".join(parts)

        if total_score >= buy_threshold:
            strength = min(1.0, (total_score - buy_threshold) / (100 - buy_threshold))
            return Signal(SignalAction.BUY, symbol, round(strength, 4), price, timestamp,
                          f"FUSION BUY: {reason}")
        elif total_score <= sell_threshold:
            strength = min(1.0, (sell_threshold - total_score) / sell_threshold)
            return Signal(SignalAction.SELL, symbol, round(strength, 4), price, timestamp,
                          f"FUSION SELL: {reason}")
        else:
            return Signal(SignalAction.HOLD, symbol, 0.0, price, timestamp,
                          f"FUSION HOLD: {reason}")

    # ─── Layer 1 ────────────────────────────────────────────────────

    def _compute_tech_score(self, close: pd.Series) -> float:
        """Aggregate 4 indicators via shared functions. Baseline 50, ±25 per indicator."""
        # SMA
        sma_s, sma_l = compute_sma(close,
                                   int(self.params.get("sma_short", 5)),
                                   int(self.params.get("sma_long", 20)))
        sma_r = detect_sma_cross(sma_s, sma_l)

        # MACD
        _, _, hist = compute_macd(close,
                                  int(self.params.get("macd_fast", 12)),
                                  int(self.params.get("macd_slow", 26)),
                                  int(self.params.get("macd_signal", 9)))
        macd_r = detect_macd_cross(hist, float(close.iloc[-1]))

        # RSI
        rsi = compute_rsi(close, int(self.params.get("rsi_period", 14)))
        rsi_r = detect_rsi_signal(rsi,
                                  float(self.params.get("rsi_oversold", 30)),
                                  float(self.params.get("rsi_overbought", 70)),
                                  self.state)

        # Bollinger
        _, upper, lower = compute_bollinger(close,
                                            int(self.params.get("bollinger_period", 20)),
                                            float(self.params.get("bollinger_std_dev", 2.0)))
        bb_r = detect_bollinger_cross(close, upper, lower)

        raw = 50 + sma_r.score + macd_r.score + rsi_r.score + bb_r.score
        return max(0.0, min(100.0, float(raw)))

    # ─── Layer 2 ────────────────────────────────────────────────────

    def _compute_sentiment_score(self, timestamp_ms: int) -> int | None:
        """Contrarian Fear & Greed mapping: fear → high score (buy bias)."""
        self._ensure_fg_loaded()
        if not self._fg_history:
            return None

        target_s = timestamp_ms // 1000
        best = None
        for entry in self._fg_history:
            if int(entry.get("timestamp", 0)) <= target_s:
                best = entry
            else:
                break

        if best is None:
            best = self._fg_history[0] if self._fg_history else None
        if best is None:
            return None

        fg_value = int(best.get("value", 50))
        return max(0, min(100, 100 - fg_value))

    # ─── Layer 3 ────────────────────────────────────────────────────

    def _compute_funding_score(self, close: pd.Series) -> int | None:
        """Market structure score from funding rate (live) or price deviation (backtest)."""
        live_fr = self.state.get("_live_funding_rate_pct")
        if live_fr is not None:
            if live_fr < -0.05:
                return 80
            elif live_fr < -0.01:
                return 65
            elif live_fr > 0.1:
                return 20
            elif live_fr > 0.05:
                return 35
            else:
                return 50

        # Backtest proxy: price deviation from SMA20 + consecutive direction
        if len(close) < 21:
            return None

        sma20 = close.rolling(20).mean()
        price = close.iloc[-1]
        sma = sma20.iloc[-1]
        if pd.isna(sma) or sma == 0:
            return None

        deviation_pct = float((price - sma) / sma * 100)

        # Count consecutive bars in the same direction
        consecutive_up, consecutive_down = 0, 0
        for i in range(len(close) - 1, max(0, len(close) - 10), -1):
            if close.iloc[i] > close.iloc[i - 1]:
                consecutive_up += 1
                consecutive_down = 0
            elif close.iloc[i] < close.iloc[i - 1]:
                consecutive_down += 1
                consecutive_up = 0
            else:
                break

        if deviation_pct > 8 and consecutive_up >= 5:
            return 20
        elif deviation_pct > 5 and consecutive_up >= 4:
            return 35
        elif deviation_pct < -8 and consecutive_down >= 5:
            return 80
        elif deviation_pct < -5 and consecutive_down >= 4:
            return 65
        elif deviation_pct > 3:
            return 40
        elif deviation_pct < -3:
            return 60
        return 50

    # ─── Live mode ──────────────────────────────────────────────────

    def set_live_sentiment(self, fear_greed_value: int | None = None,
                           funding_rate_pct: float | None = None,
                           long_short_ratio: float | None = None) -> None:
        """Inject real-time external data before on_bar() in paper/live mode."""
        if funding_rate_pct is not None:
            self.state["_live_funding_rate_pct"] = funding_rate_pct
        if long_short_ratio is not None:
            self.state["_live_long_short_ratio"] = long_short_ratio
        if fear_greed_value is not None:
            self.state["_live_fg_value"] = fear_greed_value
