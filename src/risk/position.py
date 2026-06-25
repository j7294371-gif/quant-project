from dataclasses import dataclass
from decimal import Decimal
from loguru import logger


@dataclass(frozen=True)
class PositionSize:
    quantity: Decimal
    notional: Decimal
    pct_of_equity: Decimal


def calculate_position(
    method: str,
    equity: Decimal,
    price: Decimal,
    stop_loss_price: Decimal | None,
    params: dict,
    history_stats: dict | None = None,
    min_quantity: Decimal = Decimal("0.00001"),
    min_notional: Decimal = Decimal("10.0"),
) -> PositionSize:
    """
    Calculate position size based on configured method.

    Raises:
        ValueError: if method="risk_per_trade" but stop_loss_price is None or equals entry_price
    """
    equity = Decimal(str(equity))
    price = Decimal(str(price))

    if method == "fixed_pct":
        max_pct = Decimal(str(params.get("max_position_pct", "0.2")))
        notional = equity * max_pct
        quantity = notional / price

    elif method == "kelly":
        if history_stats:
            W = history_stats.get("win_rate", 0.45)
            R = history_stats.get("avg_win_ratio", 1.5)
        else:
            W = 0.45
            R = 1.5
        # Kelly: f* = W - (1-W)/R, use half-Kelly
        f_star = Decimal(str(W)) - (Decimal("1") - Decimal(str(W))) / Decimal(str(R))
        half_kelly = f_star / Decimal("2")
        if half_kelly <= 0:
            half_kelly = Decimal("0.01")  # minimum 1% if kelly is non-positive
        notional = equity * half_kelly
        quantity = notional / price

    elif method == "risk_per_trade":
        if stop_loss_price is None or stop_loss_price == price:
            raise ValueError(
                "risk_per_trade requires valid stop_loss_price (not None and not equal to entry_price)"
            )
        risk_pct = Decimal(str(params.get("risk_per_trade_pct", "0.01")))
        risk_amount = equity * risk_pct
        price_diff = abs(price - Decimal(str(stop_loss_price)))
        if price_diff == 0:
            raise ValueError(
                "stop_loss_price equals entry_price, cannot calculate position"
            )
        quantity = risk_amount / price_diff
        notional = quantity * price
    else:
        raise ValueError(f"Unknown position method: {method}")

    # Clamp notional
    if notional > equity * Decimal("0.99"):
        notional = equity * Decimal("0.99")
        quantity = notional / price

    # Check minimum
    if notional < min_notional:
        logger.warning(
            f"Position notional ({notional}) below minimum ({min_notional}), setting to 0"
        )
        return PositionSize(
            quantity=Decimal("0"),
            notional=Decimal("0"),
            pct_of_equity=Decimal("0"),
        )

    pct = notional / equity

    return PositionSize(
        quantity=quantity,
        notional=notional,
        pct_of_equity=pct,
    )
