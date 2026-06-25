import sys
import json
import os
from decimal import Decimal
from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_dir: str = "./logs",
    retention_days: int = 30,
):
    """Configure loguru global logger. Returns the logger instance."""
    # Remove default handler
    logger.remove()

    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Console sink - colored, INFO level
    logger.add(
        sys.stderr,
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {module}:{function}:{line} | {message}",
        colorize=True,
    )

    # System log - JSON serialized, DEBUG, daily rotation
    logger.add(
        os.path.join(log_dir, "system.log"),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {module}:{function}:{line} | {message}",
        serialize=True,
        rotation="00:00",
        retention=f"{retention_days} days",
    )

    # Error log - plain text, ERROR+, daily rotation
    logger.add(
        os.path.join(log_dir, "error.log"),
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {module}:{function}:{line} | {message}",
        rotation="00:00",
        retention=f"{retention_days} days",
    )

    return logger


def log_trade(order, realized_pnl: Decimal, mode: str, log_path: str = "./logs/trades.log") -> None:
    """Append a JSON line to trades.log with file locking."""
    import msvcrt

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    entry = {
        "order_id": order.id,
        "timestamp": order.timestamp,
        "symbol": order.symbol,
        "side": order.side,
        "quantity": str(order.quantity),
        "price": str(order.filled_price) if order.filled_price else str(order.price) if order.price else "0",
        "fee": str(order.fee),
        "realized_pnl": str(realized_pnl),
        "reason": order.reason,
        "mode": mode,
    }

    with open(log_path, "a", encoding="utf-8") as f:
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
        finally:
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            except Exception:
                pass
