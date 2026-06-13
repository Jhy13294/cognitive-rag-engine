import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(funcName)s | %(message)s"


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
