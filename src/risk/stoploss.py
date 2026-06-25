from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class StopLossResult:
    triggered: bool
    stop_price: Decimal
    type: str = "stop_loss"


@dataclass(frozen=True)
class TakeProfitResult:
    triggered: bool
    tp_price: Decimal
    type: str = "take_profit"


def check_stop_loss(
    entry_price: Decimal,
    current_price: Decimal,
    high_since_entry: Decimal,
    current_atr: Decimal | None,
    config: dict,
) -> StopLossResult:
    """
    Check if stop loss is triggered.
    Types: fixed_pct, atr, trailing
    For trailing: SL only moves up (max), never down.
    """
    entry_price = Decimal(str(entry_price))
    current_price = Decimal(str(current_price))
    high_since_entry = Decimal(str(high_since_entry))

    sl_type = config.get("type", "fixed_pct")

    if sl_type == "fixed_pct":
        pct = Decimal(str(config.get("fixed_pct", "0.05")))
        stop_price = entry_price * (Decimal("1") - pct)

    elif sl_type == "atr":
        if current_atr is None:
            return StopLossResult(triggered=False, stop_price=Decimal("0"))
        atr = Decimal(str(current_atr))
        multiplier = Decimal(str(config.get("atr_multiplier", "2.0")))
        stop_price = entry_price - atr * multiplier

    elif sl_type == "trailing":
        pct = Decimal(str(config.get("trailing_pct", "0.03")))
        # Initial SL from entry
        base_sl = entry_price * (Decimal("1") - pct)
        # Trailing: SL = max(prev_SL, current_price * (1 - pct))
        # Since we compute fresh each call, use the higher of base and trailing
        trailing_sl = high_since_entry * (Decimal("1") - pct)
        stop_price = max(base_sl, trailing_sl)
    else:
        stop_price = Decimal("0")

    triggered = current_price <= stop_price if stop_price > 0 else False
    return StopLossResult(triggered=triggered, stop_price=stop_price)


def check_take_profit(
    entry_price: Decimal,
    current_price: Decimal,
    current_atr: Decimal | None,
    config: dict,
) -> TakeProfitResult:
    """
    Check if take profit is triggered.
    Types: fixed_pct, atr
    """
    entry_price = Decimal(str(entry_price))
    current_price = Decimal(str(current_price))

    tp_type = config.get("type", "fixed_pct")

    if tp_type == "fixed_pct":
        pct = Decimal(str(config.get("fixed_pct", "0.10")))
        tp_price = entry_price * (Decimal("1") + pct)

    elif tp_type == "atr":
        if current_atr is None:
            return TakeProfitResult(triggered=False, tp_price=Decimal("0"))
        atr = Decimal(str(current_atr))
        multiplier = Decimal(str(config.get("atr_multiplier", "4.0")))
        tp_price = entry_price + atr * multiplier
    else:
        tp_price = Decimal("0")

    triggered = current_price >= tp_price if tp_price > 0 else False
    return TakeProfitResult(triggered=triggered, tp_price=tp_price)
