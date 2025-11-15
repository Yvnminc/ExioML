"""Public-facing helpers for loading ExioML emission factors."""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Set, Union

import pandas as pd

from .data_io import (
    ALIAS_TO_SOURCE,
    CANONICAL_FACTOR_COLUMN,
    CANONICAL_SCHEMA_NAMES,
    RENAMED_COLUMNS,
    load_raw_frame,
    normalize_schema,
)
from .logging_utils import get_logger

LOGGER = get_logger()
BASE_OUTPUT_COLUMNS = ["schema", "region", "sector", "year", CANONICAL_FACTOR_COLUMN, "factor_value"]


def load_factor(
    *,
    schema: str,
    years: Optional[Union[int, Sequence[int]]] = None,
    regions: Optional[Sequence[str]] = None,
    columns: Optional[Iterable[str]] = None,
    backend: str = "pandas",
) -> pd.DataFrame:
    """Load a filtered emission-factor table.

    Parameters
    ----------
    schema:
        Either ``"PxP"`` or ``"IxI"`` (case-insensitive).
    years:
        Optional year or collection of years to keep. When omitted, all years
        remain in the output.
    regions:
        Optional ISO / Exio region codes (case-insensitive). When omitted all
        regions are returned.
    columns:
        Additional column names to include in the output. Names defined in the
        CSV header or their normalized snake_case aliases are accepted.
    backend:
        Currently the only supported backend is ``"pandas"``. The argument is
        reserved for future Polars/Arrow integrations.
    """

    backend_key = backend.lower()
    if backend_key != "pandas":
        raise ValueError("Only the pandas backend is currently supported")

    normalized_schema = normalize_schema(schema)
    display_schema = CANONICAL_SCHEMA_NAMES[normalized_schema]
    normalized_years = _normalize_years(years)
    normalized_regions = _normalize_regions(regions)
    normalized_columns = _normalize_columns(columns)

    frame = load_raw_frame(normalized_schema, columns=normalized_columns)
    frame["schema"] = display_schema

    if normalized_years is not None:
        frame = frame[frame["year"].isin(normalized_years)]
    if normalized_regions is not None:
        frame = frame[frame["region"].str.upper().isin(normalized_regions)]

    if CANONICAL_FACTOR_COLUMN not in frame.columns:
        raise ValueError(
            f"Column {CANONICAL_FACTOR_COLUMN} missing from dataset {normalized_schema}"
        )

    frame["factor_value"] = frame[CANONICAL_FACTOR_COLUMN]

    desired_columns = _desired_columns(frame, normalized_columns)
    frame = frame.loc[:, desired_columns].reset_index(drop=True)
    return frame


def list_regions(schema: str) -> List[str]:
    """Return the sorted region codes available for a schema."""

    normalized_schema = normalize_schema(schema)
    frame = load_raw_frame(normalized_schema, columns={"region"})
    return sorted(frame["region"].astype(str).str.upper().unique().tolist())


def list_years(schema: str) -> List[int]:
    """Return sorted years available for a schema."""

    normalized_schema = normalize_schema(schema)
    frame = load_raw_frame(normalized_schema, columns={"year"})
    return sorted(frame["year"].astype(int).unique().tolist())


def _normalize_years(years: Optional[Union[int, Sequence[int]]]) -> Optional[Set[int]]:
    if years is None:
        return None
    if isinstance(years, int):
        return {int(years)}
    normalized: Set[int] = set()
    for value in years:
        normalized.add(int(value))
    return normalized


def _normalize_regions(regions: Optional[Sequence[str]]) -> Optional[Set[str]]:
    if regions is None:
        return None
    normalized = {region.upper() for region in regions}
    return normalized


def _normalize_columns(columns: Optional[Iterable[str]]) -> Set[str]:
    normalized: Set[str] = set()
    if not columns:
        return normalized
    for column in columns:
        if column in RENAMED_COLUMNS:
            normalized.add(RENAMED_COLUMNS[column])
        elif column in ALIAS_TO_SOURCE:
            normalized.add(column)
        else:
            normalized.add(column)
    return normalized


def _desired_columns(frame: pd.DataFrame, requested: Set[str]) -> List[str]:
    desired = []
    for column in BASE_OUTPUT_COLUMNS:
        if column == CANONICAL_FACTOR_COLUMN and column not in frame.columns:
            continue
        if column in frame.columns or column == "schema":
            desired.append(column)
    for column in requested:
        if column not in desired and column in frame.columns:
            desired.append(column)
    return desired
