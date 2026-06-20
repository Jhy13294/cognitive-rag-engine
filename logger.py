import logging
import os
from contextvars import ContextVar, Token
from logging.handlers import RotatingFileHandler
from typing import Optional

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(funcName)s | %(message)s"
_REQUEST_ID: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


class _RequestContextFilter(logging.Filter):
    """Add the active request ID to existing log messages without changing formatters."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Prefix a log message when request context is active."""
        request_id = _REQUEST_ID.get()
        if request_id and not str(record.msg).startswith("request_id="):
            record.msg = f"request_id={request_id} | {record.msg}"
        return True


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create and configure a logger.

    Args:
        name: Logger name, usually __name__.
        level: Logging level.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not any(isinstance(item, _RequestContextFilter) for item in logger.filters):
        logger.addFilter(_RequestContextFilter())

    if logger.handlers:
        return logger

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        filename=os.path.join(LOG_DIR, f"{name}.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(file_handler)

    return logger


def set_request_id(request_id: str) -> Token:
    """Set request context and return a token used to restore it."""
    return _REQUEST_ID.set(request_id)


def reset_request_id(token: Token) -> None:
    """Restore request context using a token returned by set_request_id."""
    _REQUEST_ID.reset(token)


def get_request_id() -> Optional[str]:
    """Return the active request ID when one is set."""
    return _REQUEST_ID.get()


def mask_sensitive_info(text: str, mask_char: str = "*", visible_length: int = 4) -> str:
    """Mask sensitive text such as API keys.

    Args:
        text: Raw sensitive text.
        mask_char: Mask character.
        visible_length: Number of characters to keep at the start and end.

    Returns:
        Masked text.
    """
    if not text or len(text) <= visible_length * 2:
        return mask_char * len(text) if text else text

    visible_start = text[:visible_length]
    visible_end = text[-visible_length:]
    masked_middle = mask_char * (len(text) - visible_length * 2)

    return f"{visible_start}{masked_middle}{visible_end}"
