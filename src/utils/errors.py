class QuantError(Exception):
    """Base exception for all quant system errors."""
    pass


class ConfigError(QuantError):
    """Configuration loading/validation error."""
    pass


class DataFetchError(QuantError):
    """Data fetching failed after all retries."""
    pass


class StreamDisconnectedError(QuantError):
    """WebSocket stream disconnected beyond recovery."""
    pass


class PositionMismatchError(QuantError):
    """Local position doesn't match exchange position."""
    pass


class OrderExecutionError(QuantError):
    """Order execution failed."""
    pass


class CircuitBreakerError(QuantError):
    """Circuit breaker is tripped."""
    pass
