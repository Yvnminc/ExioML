"""Central logging helpers used across the package."""
from __future__ import annotations

import logging

LOGGER_NAME = "exioml"


def get_logger() -> logging.Logger:
    """Return the package logger, creating it if necessary."""
    return logging.getLogger(LOGGER_NAME)
