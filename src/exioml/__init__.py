"""ExioML public API surface.

This module exposes convenience helpers for loading emission factor
snapshots derived from Exiobase exports. Typical usage::

    from exioml import load_factor
    df = load_factor(schema="PxP", years=[2010, 2020], regions=["US", "CN"])

Environment variables
---------------------
``EXIOML_DATA_DIR``
    Optional override pointing to a directory that already contains the raw
    CSV exports (useful for offline development).
``EXIOML_CACHE_DIR``
    Directory where downloaded or copied CSV files are cached. Defaults to
    ``~/.cache/exioml``.
``EXIOML_REMOTE_BASE``
    HTTP(S) location that hosts the CSV exports. When the cache is empty and
    ``EXIOML_DATA_DIR`` does not exist the files are fetched from this base
    URL.
``EXIOML_LOGLEVEL``
    Adjusts the verbosity for the internal ``exioml`` logger (defaults to
    ``WARNING``).
"""
from __future__ import annotations

import logging
import os
from typing import Iterable, List

from .factors import load_factor, list_regions, list_years
from .datasets import (
    DatasetSplit,
    LeaveOneOutEncoder,
    build_preprocessor,
    frame_to_xy,
    prepare_dataset,
    preprocess_xy,
    split_xy,
)
from .preprocessing import RegressionSplits, prepare_regression_splits
from .training import TrainingResult, train

__all__: List[str] = [
    "load_factor",
    "list_regions",
    "list_years",
    "frame_to_xy",
    "build_preprocessor",
    "LeaveOneOutEncoder",
    "preprocess_xy",
    "split_xy",
    "prepare_dataset",
    "DatasetSplit",
    "train",
    "TrainingResult",
    "prepare_regression_splits",
    "RegressionSplits",
]
__version__ = "0.2.1"

_logger = logging.getLogger("exioml")

_env_level = os.getenv("EXIOML_LOGLEVEL")
if _env_level:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    _logger.addHandler(_handler)
    try:
        _logger.setLevel(getattr(logging, _env_level.upper()))
    except AttributeError:
        _logger.setLevel(logging.WARNING)
else:
    _logger.addHandler(logging.NullHandler())
    _logger.setLevel(logging.WARNING)

# Make help(exioml.load_factor) informative when accessed via package root
load_factor.__doc__ = load_factor.__doc__
list_regions.__doc__ = list_regions.__doc__
list_years.__doc__ = list_years.__doc__
