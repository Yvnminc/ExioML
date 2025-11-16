"""Dataset preparation helpers transforming factor frames into ML-ready splits."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

DEFAULT_RATIOS = (0.6, 0.2, 0.2)


@dataclass
class DatasetSplit:
    """Container for train/validation/test numpy arrays."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray

    def as_tuple(self) -> Tuple[np.ndarray, ...]:
        """Return splits as a tuple (X_train, y_train, X_val, y_val, X_test, y_test)."""

        return (
            self.x_train,
            self.y_train,
            self.x_val,
            self.y_val,
            self.x_test,
            self.y_test,
        )

    def as_dict(self) -> dict:
        """Return splits as a dict keyed by split/role names."""

        return {
            "x_train": self.x_train,
            "y_train": self.y_train,
            "x_val": self.x_val,
            "y_val": self.y_val,
            "x_test": self.x_test,
            "y_test": self.y_test,
        }


class LeaveOneOutEncoder(BaseEstimator, TransformerMixin):
    """Lightweight leave-one-out target encoder for categorical columns."""

    def __init__(
        self,
        columns: Sequence[str],
        *,
        suffix: str = "_looe",
        drop_original: bool = True,
        fill_value: str = "__missing__",
    ) -> None:
        self.columns = list(columns)
        self.suffix = suffix
        self.drop_original = drop_original
        self.fill_value = fill_value
        self._stats: dict = {}
        self.global_mean_: float = 0.0
        self.feature_names_in_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):  # type: ignore[override]
        df, y_series = self._prepare_inputs(X, y, require_y=True)
        self.feature_names_in_ = list(df.columns)
        self.global_mean_ = float(y_series.mean())
        self._stats = {}
        for column in self.columns:
            grouped = (
                pd.DataFrame({column: df[column], "target": y_series})
                .groupby(column)["target"]
                .agg(["sum", "count"])
            )
            self._stats[column] = grouped
        return self

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params):  # type: ignore[override]
        self.fit(X, y)
        df, y_series = self._prepare_inputs(X, y, require_y=True)
        return self._encode(df, y_series, use_leave_one_out=True)

    def transform(self, X: pd.DataFrame, y=None):  # type: ignore[override]
        df, y_series = self._prepare_inputs(X, y, require_y=False)
        return self._encode(df, y_series, use_leave_one_out=y_series is not None)

    def _prepare_inputs(
        self, X: pd.DataFrame, y, *, require_y: bool
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("LeaveOneOutEncoder expects a pandas DataFrame input")
        df = X.copy()
        missing_cols = set(self.columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing categorical columns: {sorted(missing_cols)}")
        df[self.columns] = df[self.columns].fillna(self.fill_value)

        y_series: Optional[pd.Series] = None
        if y is not None:
            y_arr = np.asarray(y).reshape(-1)
            if len(y_arr) != len(df):
                raise ValueError("X and y must align for leave-one-out encoding")
            y_series = pd.Series(y_arr, index=df.index)
        elif require_y:
            raise ValueError("y is required to fit leave-one-out encoder")
        return df, y_series

    def _encoded_name(self, column: str) -> str:
        return f"{column}{self.suffix}"

    def _encode(
        self, df: pd.DataFrame, y_series: Optional[pd.Series], *, use_leave_one_out: bool
    ) -> pd.DataFrame:
        result = df.copy()
        for column in self.columns:
            stats = self._stats.get(column)
            if stats is None:
                raise ValueError("Encoder has not been fitted")
            mean_map = stats["sum"] / stats["count"]
            if use_leave_one_out and y_series is not None:
                sums = result[column].map(stats["sum"])
                counts = result[column].map(stats["count"])
                numerators = sums - y_series
                denominators = counts - 1
                encoded = numerators / denominators
                encoded = encoded.where(denominators > 0, self.global_mean_)
            else:
                encoded = result[column].map(mean_map)
            encoded = encoded.fillna(self.global_mean_).astype(np.float32)
            result[self._encoded_name(column)] = encoded
        if self.drop_original:
            result = result.drop(columns=self.columns)
        return result


def frame_to_xy(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    dtype: np.dtype = np.float32,
    dropna: Optional[str] = "any",
    *,
    categorical_cols: Optional[Sequence[str]] = None,
    as_frame: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract feature/target arrays from a DataFrame.

    Parameters
    ----------
    df:
        Source table containing the requested columns.
    feature_cols:
        Column names to use as features; preserved order defines column order.
    target_col:
        Column name to use as the target.
    dtype:
        dtype for the returned feature array (default float32).
    dropna:
        ``\"any\"``/``\"all\"`` mirrors pandas NA dropping behaviour; ``None`` keeps
        missing values. Samples removed here are removed from both X and y.
    """

    if df is None:
        raise ValueError("df must not be None")
    if not feature_cols:
        raise ValueError("feature_cols must contain at least one column")
    missing = set(feature_cols) | {target_col}
    missing -= set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if dropna not in ("any", "all", None):
        raise ValueError('dropna must be one of "any", "all", or None')

    subset = df.loc[:, list(feature_cols) + [target_col]].copy()
    if dropna in ("any", "all"):
        subset = subset.dropna(how=dropna)
    if subset.empty:
        raise ValueError("Dataset is empty after applying NA handling")

    categorical_cols = list(categorical_cols or [])
    missing_cats = set(categorical_cols) - set(feature_cols)
    if missing_cats:
        raise ValueError(f"categorical_cols not in feature_cols: {sorted(missing_cats)}")

    X_frame = subset.loc[:, feature_cols].copy()
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]
    if dtype is not None and numeric_cols:
        X_frame.loc[:, numeric_cols] = X_frame.loc[:, numeric_cols].astype(dtype)

    y = subset.loc[:, target_col].to_numpy(copy=True).reshape(-1)
    if as_frame or categorical_cols:
        X = X_frame
    else:
        X = X_frame.to_numpy(dtype=dtype, copy=True)
        if X.ndim != 2:
            X = np.reshape(X, (X.shape[0], -1))
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples")
    return X, y


def build_preprocessor(
    strategy: str = "standard",
    *,
    scaler_kwargs: Optional[dict] = None,
    imputer: Optional[str] = "drop",
    encoder=None,
    categorical_cols: Optional[Sequence[str]] = None,
    leave_one_out: bool = True,
) -> Pipeline:
    """Build a preprocessing pipeline with optional imputation and scaling.

    ``strategy`` accepts ``\"standard\"`` (default) or ``\"minmax\"``; when set to
    ``\"minmax\"`` the pipeline swaps to :class:`MinMaxScaler`. ``imputer=\"median\"``
    inserts a :class:`SimpleImputer` and otherwise rows are passed through
    untouched. A custom ``encoder`` (e.g. OneHotEncoder/ColumnTransformer) can
    be inserted ahead of the scaler. When ``categorical_cols`` is provided the
    default encoder becomes a leave-one-out target encoder so categorical
    columns can be handled alongside numeric columns.
    """

    scaler_kwargs = scaler_kwargs or {}
    if encoder is not None and "with_mean" not in scaler_kwargs and strategy.lower() == "standard":
        scaler_kwargs["with_mean"] = False
    strategy_key = strategy.lower()
    if strategy_key == "standard":
        scaler = StandardScaler(**scaler_kwargs)
    elif strategy_key == "minmax":
        scaler = MinMaxScaler(**scaler_kwargs)
    else:
        raise ValueError('strategy must be either "standard" or "minmax"')

    steps = []
    if imputer not in ("drop", "median", None):
        raise ValueError('imputer must be "drop", "median", or None')

    cat_columns = list(categorical_cols or [])
    if encoder is None and cat_columns and leave_one_out:
        encoder = LeaveOneOutEncoder(columns=cat_columns)
    if encoder is not None:
        steps.append(("encoder", encoder))
    if imputer == "median":
        steps.append(("imputer", SimpleImputer(strategy="median")))
    steps.append(("scaler", scaler))

    preprocessor = Pipeline(steps)
    preprocessor._exioml_dropna = imputer == "drop"  # type: ignore[attr-defined]
    preprocessor._exioml_expect_dataframe = encoder is not None or bool(cat_columns)  # type: ignore[attr-defined]
    return preprocessor


def preprocess_xy(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    *,
    preprocessor: Optional[Pipeline] = None,
    fit: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], Pipeline]:
    """Apply the preprocessing pipeline and keep X/y aligned."""

    preprocessor = preprocessor or build_preprocessor()

    if isinstance(X, pd.DataFrame):
        X_frame = X.copy()
    else:
        X_arr = np.asarray(X)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array")
        X_frame = pd.DataFrame(X_arr)

    y_arr = None
    if y is not None:
        y_arr = np.asarray(y).reshape(-1)
        if y_arr.shape[0] != len(X_frame):
            raise ValueError("X and y must have compatible first dimensions")

    if getattr(preprocessor, "_exioml_dropna", False):  # type: ignore[attr-defined]
        mask = ~X_frame.isna().any(axis=1)
        X_frame = X_frame.loc[mask]
        if y_arr is not None:
            y_arr = y_arr[mask]

    if X_frame.shape[0] == 0:
        raise ValueError("No samples left after NA handling")

    expects_frame = getattr(preprocessor, "_exioml_expect_dataframe", False)
    preprocessor_input = X_frame if expects_frame else X_frame.to_numpy()

    if fit:
        X_proc = preprocessor.fit_transform(preprocessor_input, y_arr)
    else:
        X_proc = preprocessor.transform(preprocessor_input)

    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()
    X_proc = np.asarray(X_proc, dtype=np.float32)
    if X_proc.shape[0] == 0:
        raise ValueError("No samples produced during preprocessing")
    return X_proc, y_arr, preprocessor


def split_xy(
    X: np.ndarray,
    y: np.ndarray,
    *,
    train_ratio: float = DEFAULT_RATIOS[0],
    val_ratio: float = DEFAULT_RATIOS[1],
    test_ratio: float = DEFAULT_RATIOS[2],
    stratify: bool = True,
    random_state: int = 42,
    shuffle: bool = True,
) -> DatasetSplit:
    """Split arrays into train/validation/test sets with stratified fallback."""

    X_arr = np.asarray(X)
    y_arr = np.asarray(y).reshape(-1)
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D array")
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("X and y must share the sample dimension")

    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1")
    if min(train_ratio, val_ratio, test_ratio) < 0:
        raise ValueError("ratios must be non-negative")

    test_size = test_ratio
    stratify_labels = y_arr if stratify else None
    try:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_arr,
            y_arr,
            test_size=test_size,
            stratify=stratify_labels,
            random_state=random_state,
            shuffle=shuffle,
        )
    except ValueError:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_arr,
            y_arr,
            test_size=test_size,
            stratify=None,
            random_state=random_state,
            shuffle=shuffle,
        )
        stratify_labels = None

    val_share = val_ratio / (train_ratio + val_ratio)
    stratify_val = y_train_val if stratify and stratify_labels is not None else None
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=val_share,
            stratify=stratify_val,
            random_state=random_state,
            shuffle=shuffle,
        )
    except ValueError:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=val_share,
            stratify=None,
            random_state=random_state,
            shuffle=shuffle,
        )

    return DatasetSplit(
        x_train=X_train,
        y_train=y_train,
        x_val=X_val,
        y_val=y_val,
        x_test=X_test,
        y_test=y_test,
    )


def prepare_dataset(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    *,
    preprocessor: Optional[Pipeline] = None,
    categorical_cols: Optional[Sequence[str]] = None,
    leave_one_out: bool = True,
    ratios: Tuple[float, float, float] = DEFAULT_RATIOS,
    stratify: bool = True,
    random_state: int = 42,
    dropna: Optional[str] = "any",
    dtype: np.dtype = np.float32,
) -> Tuple[DatasetSplit, Pipeline]:
    """One-stop entry to extract X/y, preprocess, and split."""

    categorical_cols = list(categorical_cols or [])
    X, y = frame_to_xy(
        df,
        feature_cols,
        target_col,
        dtype=dtype,
        dropna=dropna,
        categorical_cols=categorical_cols,
        as_frame=bool(categorical_cols),
    )
    if preprocessor is None:
        preprocessor = build_preprocessor(
            categorical_cols=categorical_cols,
            leave_one_out=leave_one_out,
        )
    X_proc, y_proc, fitted = preprocess_xy(
        X,
        y,
        preprocessor=preprocessor,
        fit=True,
    )
    train_ratio, val_ratio, test_ratio = ratios
    splits = split_xy(
        X_proc,
        y_proc,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        stratify=stratify,
        random_state=random_state,
    )
    return splits, fitted
