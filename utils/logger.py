"""Lightweight logging utilities used across agents."""

from __future__ import annotations

import logging
from typing import Any, Dict


_LOGGER_CACHE: Dict[str, logging.Logger] = {}


def get_logger(name: str) -> logging.Logger:
    if name in _LOGGER_CACHE:
        return _LOGGER_CACHE[name]

    logger = logging.getLogger(f"meta_agent.{name}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.propagate = False
    _LOGGER_CACHE[name] = logger
    return logger


def log_kv(logger: logging.Logger, **fields: Any) -> None:
    message = " ".join(f"{k}={v}" for k, v in fields.items())
    logger.info(message)


__all__ = ["get_logger", "log_kv"]
