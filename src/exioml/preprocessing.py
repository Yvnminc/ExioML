"""Feature preprocessing helpers replicating the ExioML paper pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split

from .factors import load_factor

NumericList = Sequence[str]
CategoricalList = Sequence[str]

DEFAULT_NUMERIC_FEATURES = [
    "value_added_meur",
    "employment_k",
    "energy_carrier_tj",
    "year",
]
DEFAULT_CATEGORICAL_FEATURES = ["region", "sector"]
DEFAULT_TARGET = "factor_value"


@dataclass
class RegressionSplits:
    """Container with train/val/test frames and preprocessing metadata."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    feature_columns: List[str]
    target_column: str
    normalization_stats: Dict[str, Tuple[float, float]]
    leave_one_out_columns: List[str]


def prepare_regression_splits(
    *,
    schema: str,
    years: Optional[Union[int, Sequence[int]]] = None,
    regions: Optional[Sequence[str]] = None,
    numeric_features: Optional[NumericList] = None,
    categorical_features: Optional[CategoricalList] = None,
    target: str = DEFAULT_TARGET,
    apply_normalization: bool = True,
    add_leave_one_out: bool = True,
    dropna: bool = True,
    random_state: int = 42,
    test_size: float = 0.2,
    validation_size: float = 0.16,
) -> RegressionSplits:
    """Load ExioML factors and reproduce the paper's preprocessing pipeline.

    The returned frames follow the 64/16/20 (train/validation/test) split used in
    the ExioML paper, apply min-max normalization on numeric features, and add
    leave-one-out encoded columns for categorical variables.
    """

    numeric = list(numeric_features or DEFAULT_NUMERIC_FEATURES)
    categorical = list(categorical_features or DEFAULT_CATEGORICAL_FEATURES)
    _ensure_non_empty("numeric_features", numeric)
    _ensure_non_empty("categorical_features", categorical)

    frame = load_factor(
        schema=schema,
        years=years,
        regions=regions,
        columns=set(numeric),
    ).copy()

    required_columns = set(numeric) | set(categorical) | {target}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    if dropna:
        frame = frame.dropna(subset=list(required_columns))
    if frame.empty:
        raise ValueError("Dataset is empty after applying filters/NA dropping")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    train_share = 1.0 - test_size
    if not 0 < validation_size < train_share:
        raise ValueError("validation_size must be between 0 and (1 - test_size)")
    val_relative = validation_size / train_share

    train_val, test = train_test_split(
        frame, test_size=test_size, random_state=random_state
    )
    train, validation = train_test_split(
        train_val, test_size=val_relative, random_state=random_state
    )

    normalization_stats: Dict[str, Tuple[float, float]] = {}
    if apply_normalization:
        normalization_stats = _fit_min_max(train, numeric)
        for subset in (train, validation, test):
            _apply_min_max(subset, numeric, normalization_stats)

    leave_one_out_columns: List[str] = []
    if add_leave_one_out:
        leave_one_out_columns = _add_leave_one_out_columns(
            train, validation, test, categorical, target
        )

    feature_columns = leave_one_out_columns + list(numeric)

    return RegressionSplits(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
        feature_columns=feature_columns,
        target_column=target,
        normalization_stats=normalization_stats,
        leave_one_out_columns=leave_one_out_columns,
    )


def _ensure_non_empty(name: str, values: Sequence[str]) -> None:
    if not values:
        raise ValueError(f"{name} must contain at least one column")


def _fit_min_max(frame: pd.DataFrame, columns: Iterable[str]) -> Dict[str, Tuple[float, float]]:
    stats: Dict[str, Tuple[float, float]] = {}
    for column in columns:
        col_min = float(frame[column].min())
        col_max = float(frame[column].max())
        stats[column] = (col_min, col_max)
    return stats


def _apply_min_max(
    frame: pd.DataFrame,
    columns: Iterable[str],
    stats: Mapping[str, Tuple[float, float]],
) -> None:
    for column in columns:
        col_min, col_max = stats[column]
        denom = col_max - col_min
        if denom == 0:
            frame[column] = 0.0
        else:
            frame[column] = (frame[column] - col_min) / denom


def _add_leave_one_out_columns(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    categorical: Sequence[str],
    target: str,
) -> List[str]:
    new_columns: List[str] = []
    for column in categorical:
        grouped = train.groupby(column)[target].agg(["sum", "count"])
        global_mean = float(train[target].mean())
        column_name = f"{column}_looe"

        sums = train[column].map(grouped["sum"])
        counts = train[column].map(grouped["count"])
        numerators = sums - train[target]
        denominators = counts - 1
        encoded_train = numerators / denominators
        encoded_train = encoded_train.where(denominators > 0, global_mean)
        encoded_train = encoded_train.fillna(global_mean)

        means = grouped["sum"] / grouped["count"]
        encoded_validation = validation[column].map(means).fillna(global_mean)
        encoded_test = test[column].map(means).fillna(global_mean)

        train[column_name] = encoded_train
        validation[column_name] = encoded_validation
        test[column_name] = encoded_test
        new_columns.append(column_name)
    return new_columns

