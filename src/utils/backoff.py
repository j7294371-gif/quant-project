import time
import functools
from loguru import logger


def retry_with_backoff(
    func,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retry_on: tuple = (Exception,),
):
    """
    Call func() with exponential backoff on failure.
    Wait = min(base_delay * 2^n, max_delay) seconds before retry n.
    After max_retries failures, raises the last exception.
    """
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return func()
        except retry_on as e:
            last_exception = e
            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.warning(
                    f"Retry {attempt + 1}/{max_retries} after error: {e}. "
                    f"Waiting {delay:.1f}s..."
                )
                time.sleep(delay)
            else:
                logger.error(f"All {max_retries} retries exhausted: {e}")
                raise


def with_backoff(
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retry_on: tuple = (Exception,),
):
    """Decorator version of retry_with_backoff."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return retry_with_backoff(
                lambda: func(*args, **kwargs),
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                retry_on=retry_on,
            )
        return wrapper
    return decorator
