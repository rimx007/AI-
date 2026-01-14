"""Structured logging utilities.

Example:
    from src.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("hello", extra={"stage":"train"})
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional

_LOGGER_CONFIGURED = False


def get_logger(name: str = "sentiment_project", log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """Create/retrieve a configured logger with console + rotating file handlers."""
    global _LOGGER_CONFIGURED

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if _LOGGER_CONFIGURED and logger.handlers:
        return logger

    os.makedirs(log_dir, exist_ok=True)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = RotatingFileHandler(os.path.join(log_dir, "app.log"), maxBytes=2_000_000, backupCount=5, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    _LOGGER_CONFIGURED = True
    return logger
