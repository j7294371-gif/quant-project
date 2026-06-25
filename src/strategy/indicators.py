"""Pure-function indicator calculations shared by all strategies.

Each indicator has two layers:
  compute_*   → returns raw indicator Series (vectorized, no side effects)
  detect_*    → returns (score: int, strength: float) for signal detection

Scores are in [-25, 25] range for individual indicators, designed to be
summed into a 0-100 aggregate.
"""

import pandas as pd
import numpy as np
from decimal import Decimal
from dataclasses import dataclass
from typing import Any


@dataclass
class IndicatorResult:
    """Unified result from any indicator detector."""

    score: int  # -25 to 25 (0 = neutral, ±25 = strong signal, ±15 = trend bias)
    strength: float  # 0.0 to 1.0
    reason: str  # Human-readable description


# ─── SMA ────────────────────────────────────────────────────────────


def compute_sma(close: pd.Series, short_window: int, long_window: int) -> tuple[pd.Series, pd.Series]:
    """Compute SMA short and long series. Pure pandas vectorized."""
    sma_short = close.rolling(window=short_window).mean()
    sma_long = close.rolling(window=long_window).mean()
    return sma_short, sma_long


def detect_sma_cross(sma_short: pd.Series, sma_long: pd.Series,
                     short_name: str = "5", long_name: str = "20") -> IndicatorResult:
    """Detect golden cross / death cross from pre-computed SMA series.

    Golden cross: short crosses ABOVE long (prev short <= long, curr short > long)
    Death cross: short crosses BELOW long (prev short >= long, curr short < long)
    """
    if len(sma_short) < 2 or len(sma_long) < 2:
        return IndicatorResult(0, 0.0, "Insufficient SMA data")

    ss_prev, ss_curr = sma_short.iloc[-2], sma_short.iloc[-1]
    sl_prev, sl_curr = sma_long.iloc[-2], sma_long.iloc[-1]

    if any(pd.isna([ss_prev, ss_curr, sl_prev, sl_curr])):
        return IndicatorResult(0, 0.0, "SMA values are NaN")

    if ss_prev <= sl_prev and ss_curr > sl_curr:
        diff = abs(ss_curr - sl_curr)
        strength = min(1.0, float(diff / sl_curr * 10))
        return IndicatorResult(25, strength,
                               f"Golden cross: SMA_{short_name}={ss_curr:.4f} > SMA_{long_name}={sl_curr:.4f}")

    if ss_prev >= sl_prev and ss_curr < sl_curr:
        diff = abs(ss_curr - sl_curr)
        strength = min(1.0, float(diff / sl_curr * 10))
        return IndicatorResult(-25, strength,
                               f"Death cross: SMA_{short_name}={ss_curr:.4f} < SMA_{long_name}={sl_curr:.4f}")

    # No cross, but indicate trend bias
    if ss_curr > sl_curr:
        return IndicatorResult(15, 0.0, f"Uptrend: SMA_{short_name} > SMA_{long_name}")
    else:
        return IndicatorResult(-15, 0.0, f"Downtrend: SMA_{short_name} < SMA_{long_name}")


# ─── MACD ───────────────────────────────────────────────────────────


def compute_macd(close: pd.Series, fast: int, slow: int, signal_period: int
                 ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute MACD line, signal line, and histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def detect_macd_cross(histogram: pd.Series, close_price: float) -> IndicatorResult:
    """Detect MACD histogram zero-cross.

    BUY: histogram crosses ABOVE zero (prev <= 0, curr > 0)
    SELL: histogram crosses BELOW zero (prev >= 0, curr < 0)
    """
    if len(histogram) < 2:
        return IndicatorResult(0, 0.0, "Insufficient MACD data")

    hist_prev, hist_curr = histogram.iloc[-2], histogram.iloc[-1]

    if pd.isna(hist_prev) or pd.isna(hist_curr):
        return IndicatorResult(0, 0.0, "MACD values are NaN")

    if hist_prev <= 0 and hist_curr > 0:
        strength = min(1.0, float(abs(hist_curr) / close_price * 500))
        return IndicatorResult(25, strength,
                               f"MACD histogram crosses above zero: hist={hist_curr:.6f}")

    if hist_prev >= 0 and hist_curr < 0:
        strength = min(1.0, float(abs(hist_curr) / close_price * 500))
        return IndicatorResult(-25, strength,
                               f"MACD histogram crosses below zero: hist={hist_curr:.6f}")

    if hist_curr > 0:
        return IndicatorResult(15, 0.0, "MACD histogram positive (bullish)")
    else:
        return IndicatorResult(-15, 0.0, "MACD histogram negative (bearish)")


# ─── RSI ────────────────────────────────────────────────────────────


def compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """Compute RSI using Wilder smoothing (alpha=1/period, adjust=False)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def detect_rsi_signal(rsi: pd.Series, oversold: float, overbought: float,
                      state: dict[str, Any]) -> IndicatorResult:
    """Detect RSI oversold exit (BUY) / overbought exit (SELL).

    State is mutated in-place to track RSI extremes within zones.
    Keys used: '_rsi_min', '_rsi_max'
    """
    if len(rsi) < 2:
        return IndicatorResult(0, 0.0, "Insufficient RSI data")

    rsi_prev = rsi.iloc[-2]
    rsi_curr = rsi.iloc[-1]

    if pd.isna(rsi_prev) or pd.isna(rsi_curr):
        return IndicatorResult(0, 0.0, "RSI values are NaN")

    # Track extremes
    if rsi_curr < oversold:
        current_min = state.get("_rsi_min")
        if current_min is None or rsi_curr < current_min:
            state["_rsi_min"] = float(rsi_curr)

    if rsi_curr > overbought:
        current_max = state.get("_rsi_max")
        if current_max is None or rsi_curr > current_max:
            state["_rsi_max"] = float(rsi_curr)

    if oversold <= rsi_curr <= overbought:
        state["_rsi_min"] = None
        state["_rsi_max"] = None

    # BUY: crosses ABOVE oversold
    if rsi_prev < oversold and rsi_curr > oversold:
        rsi_min_tracked = state.get("_rsi_min")
        overshoot = (oversold - float(rsi_min_tracked)
                     if rsi_min_tracked is not None
                     else oversold - float(rsi_prev))
        strength = min(1.0, overshoot / oversold)
        state["_rsi_min"] = None
        return IndicatorResult(25, strength,
                               f"RSI exits oversold ({oversold}): prev={rsi_prev:.2f}, curr={rsi_curr:.2f}")

    # SELL: crosses BELOW overbought
    if rsi_prev > overbought and rsi_curr < overbought:
        rsi_max_tracked = state.get("_rsi_max")
        peak_excess = (float(rsi_max_tracked) - overbought
                       if rsi_max_tracked is not None
                       else float(rsi_prev) - overbought)
        strength = min(1.0, peak_excess / (100.0 - overbought))
        state["_rsi_max"] = None
        return IndicatorResult(-25, strength,
                               f"RSI exits overbought ({overbought}): prev={rsi_prev:.2f}, curr={rsi_curr:.2f}")

    # Zone bias
    if rsi_curr < oversold:
        return IndicatorResult(10, 0.0, f"RSI in oversold zone: {rsi_curr:.2f}")
    elif rsi_curr > overbought:
        return IndicatorResult(-10, 0.0, f"RSI in overbought zone: {rsi_curr:.2f}")
    else:
        return IndicatorResult(0, 0.0, f"RSI neutral: {rsi_curr:.2f}")


# ─── Bollinger Bands ────────────────────────────────────────────────


def compute_bollinger(close: pd.Series, period: int, std_dev: float
                      ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute Bollinger Bands: (sma, upper, lower). Uses ddof=0 for population std."""
    sma = close.rolling(window=period).mean()
    sigma = close.rolling(window=period).std(ddof=0)
    upper = sma + std_dev * sigma
    lower = sma - std_dev * sigma
    return sma, upper, lower


def detect_bollinger_cross(close: pd.Series, upper: pd.Series, lower: pd.Series) -> IndicatorResult:
    """Detect Bollinger Band crosses.

    BUY: price crosses ABOVE lower band (was below, now >= lower)
    SELL: price crosses BELOW upper band (was above, now <= upper)
    """
    if len(close) < 2 or len(upper) < 2 or len(lower) < 2:
        return IndicatorResult(0, 0.0, "Insufficient Bollinger data")

    c_prev, c_curr = close.iloc[-2], close.iloc[-1]
    u_prev, u_curr = upper.iloc[-2], upper.iloc[-1]
    l_prev, l_curr = lower.iloc[-2], lower.iloc[-1]

    if any(pd.isna([c_prev, c_curr, u_prev, u_curr, l_prev, l_curr])):
        return IndicatorResult(0, 0.0, "Bollinger values are NaN")

    # BUY: crosses above lower band
    if c_prev < l_prev and c_curr >= l_curr:
        diff = float(l_curr - c_curr)
        strength = min(1.0, abs(diff) / float(l_curr) * 10)
        return IndicatorResult(25, strength,
                               f"Price bounces from lower band: close={c_curr:.4f} >= lower={l_curr:.4f}")

    # SELL: crosses below upper band
    if c_prev > u_prev and c_curr <= u_curr:
        diff = float(c_curr - u_curr)
        strength = min(1.0, abs(diff) / float(u_curr) * 10)
        return IndicatorResult(-25, strength,
                               f"Price breaks from upper band: close={c_curr:.4f} <= upper={u_curr:.4f}")

    # Band position bias
    if c_curr < l_curr:
        return IndicatorResult(15, 0.0, f"Price below lower band: {c_curr:.4f} < {l_curr:.4f}")
    elif c_curr > u_curr:
        return IndicatorResult(-15, 0.0, f"Price above upper band: {c_curr:.4f} > {u_curr:.4f}")
    else:
        return IndicatorResult(0, 0.0, f"Price within bands: {c_curr:.4f}")


# ─── Utility ────────────────────────────────────────────────────────


def extract_bar_info(df: pd.DataFrame) -> tuple[Decimal, int]:
    """Extract (price, timestamp) from the last bar of a DataFrame."""
    price = Decimal(str(df["close"].iloc[-1]))
    if "timestamp" in df.columns:
        timestamp = int(df["timestamp"].iloc[-1])
    else:
        timestamp = int(df.index[-1]) if hasattr(df.index[-1], '__int__') else 0
    return price, timestamp
