"""Structured logging for training and evaluation."""

import logging
import sys
from typing import Any

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger with the given name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


def setup_logging(
    level: int = logging.INFO,
    format_string: str = LOG_FORMAT,
    date_format: str = DATE_FORMAT,
) -> None:
    """Configure root logging for the process."""
    logging.basicConfig(
        level=level,
        format=format_string,
        datefmt=date_format,
        stream=sys.stdout,
        force=True,
    )
