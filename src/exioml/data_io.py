"""Utility helpers for retrieving and caching ExioML factor tables."""
from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, Optional

import fsspec
import pandas as pd

from .logging_utils import get_logger

LOGGER = get_logger()

SCHEMA_TO_FILE = {
    "PXP": "ExioML_factor_accounting_PxP.csv",
    "IXI": "ExioML_factor_accounting_IxI.csv",
}
CANONICAL_SCHEMA_NAMES = {"PXP": "PxP", "IXI": "IxI"}

DEFAULT_FACTOR_COLUMN = "GHG emissions [kg CO2 eq.]"
ESSENTIAL_COLUMNS = {"region", "sector", "Year", DEFAULT_FACTOR_COLUMN}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

RENAMED_COLUMNS = {
    "region": "region",
    "sector": "sector",
    "Year": "year",
    DEFAULT_FACTOR_COLUMN: "ghg_emissions",
    "Value Added [M.EUR]": "value_added_meur",
    "Employment [1000 p.]": "employment_k",
    "Energy Carrier Net Total [TJ]": "energy_carrier_tj",
}

ALIAS_TO_SOURCE = {value: key for key, value in RENAMED_COLUMNS.items()}
CANONICAL_FACTOR_COLUMN = RENAMED_COLUMNS[DEFAULT_FACTOR_COLUMN]


def normalize_schema(schema: str) -> str:
    if not schema:
        raise ValueError("schema must be provided")
    key = schema.upper()
    if key not in SCHEMA_TO_FILE:
        raise ValueError(f"schema must be one of {sorted(SCHEMA_TO_FILE)}")
    return key


def _cache_dir() -> Path:
    return Path(os.environ.get("EXIOML_CACHE_DIR", Path.home() / ".cache" / "exioml"))


def _remote_base() -> Optional[str]:
    return os.environ.get("EXIOML_REMOTE_BASE")


def _candidate_directories() -> Iterable[Path]:
    override = os.environ.get("EXIOML_DATA_DIR")
    if override:
        yield Path(override)
    yield DEFAULT_DATA_DIR


def _expected_hash_key(schema: str) -> Optional[str]:
    env_key = f"EXIOML_{schema.upper()}_SHA256"
    return os.environ.get(env_key)


def _ensure_cache_dir(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)


def ensure_dataset(schema: str) -> Path:
    normalized = normalize_schema(schema)
    filename = SCHEMA_TO_FILE[normalized]
    cache_dir = _cache_dir()
    destination = cache_dir / filename

    if destination.exists():
        return destination

    _ensure_cache_dir(cache_dir)

    for directory in _candidate_directories():
        candidate = directory / filename
        if candidate.exists():
            LOGGER.info("Caching %s from %s", filename, candidate)
            shutil.copy(candidate, destination)
            return destination

    remote_base = _remote_base()
    if not remote_base:
        raise FileNotFoundError(
            f"Could not locate {filename}. Provide EXIOML_DATA_DIR or EXIOML_REMOTE_BASE."
        )

    url = "/".join([remote_base.rstrip("/"), filename])
    LOGGER.info("Downloading %s", url)
    _download_remote(url, destination)
    _validate_hash(destination, normalized)
    return destination


def _download_remote(url: str, destination: Path) -> None:
    tmp_path = destination.with_suffix(".tmp")
    with fsspec.open(url, "rb") as source, tmp_path.open("wb") as sink:
        shutil.copyfileobj(source, sink)
    tmp_path.replace(destination)


def _validate_hash(path: Path, schema: str) -> None:
    expected = _expected_hash_key(schema)
    if not expected:
        return
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest.lower() != expected.lower():
        raise ValueError(
            f"Checksum mismatch for {path.name}: expected {expected}, got {digest}"
        )


def _resolve_source_columns(requested: Optional[Iterable[str]]) -> Iterable[str]:
    resolved = set(ESSENTIAL_COLUMNS)
    if requested:
        for column in requested:
            if column in ALIAS_TO_SOURCE:
                resolved.add(ALIAS_TO_SOURCE[column])
            else:
                resolved.add(column)
    return sorted(resolved)


def load_raw_frame(schema: str, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    path = ensure_dataset(schema)
    usecols = _resolve_source_columns(columns)
    df = pd.read_csv(path, usecols=usecols)
    df = df.rename(columns=RENAMED_COLUMNS)
    if "year" in df.columns:
        df["year"] = df["year"].astype(int)
    return df
