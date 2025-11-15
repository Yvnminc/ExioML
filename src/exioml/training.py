"""Lightweight sklearn-based training helpers exposed via ``exioml.train``."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split


ModelInput = Union[str, BaseEstimator]

MODEL_REGISTRY = {
    "gdbt": HistGradientBoostingRegressor,
    "gbdt": HistGradientBoostingRegressor,
    "hist_gbdt": HistGradientBoostingRegressor,
    "hist_gradient_boosting": HistGradientBoostingRegressor,
    "random_forest": RandomForestRegressor,
    "rf": RandomForestRegressor,
    "ridge": Ridge,
}

METRIC_FUNCTIONS = {
    "mse": mean_squared_error,
    "mae": mean_absolute_error,
    "r2": r2_score,
}

SCORING_MAP = {
    "mse": "neg_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
}


@dataclass
class TrainingResult:
    """Container returned by :func:`train` holding the fitted estimator."""

    estimator: BaseEstimator
    feature_names: List[str]
    target_name: str
    metric_name: str
    train_score: float
    test_score: float
    best_params: Optional[Dict[str, Any]] = None
    cv_results: Optional[Mapping[str, Any]] = None

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Run predictions using the fitted estimator."""

        missing = set(self.feature_names) - set(data.columns)
        if missing:
            raise ValueError(f"Missing feature columns for prediction: {sorted(missing)}")
        return self.estimator.predict(data[self.feature_names])


def train(
    data: pd.DataFrame,
    *,
    target: str,
    model: ModelInput = "gdbt",
    features: Optional[Sequence[str]] = None,
    param_grid: Optional[Mapping[str, Sequence[Any]]] = None,
    test_size: float = 0.2,
    random_state: Optional[int] = 42,
    metric: str = "mse",
    scoring: Optional[str] = None,
    cv: int = 3,
    n_jobs: Optional[int] = None,
) -> TrainingResult:
    """Train a scikit-learn regressor on the provided ExioML factor frame.

    Parameters
    ----------
    data:
        Source dataframe containing the ``target`` column alongside features.
    target:
        Name of the column to predict.
    model:
        Either a short alias (``"gdbt"``, ``"random_forest"``, ``"ridge"``) or a
        fully configured sklearn estimator instance.
    features:
        Optional list of feature names. When omitted, every column except
        ``target`` is used.
    param_grid:
        Mapping compatible with :class:`~sklearn.model_selection.GridSearchCV`.
        When provided a grid-search is executed before returning the best
        estimator.
    test_size:
        Fraction of data reserved for evaluation.
    random_state:
        RNG seed forwarded to the data split and estimators that expose the
        parameter.
    metric:
        Either ``"mse"``, ``"mae"`` or ``"r2"``. Determines both the
        reporting metric and default grid-search scoring.
    scoring:
        Optional override for the sklearn ``scoring`` argument when running a
        grid-search.
    cv:
        Number of folds used by the grid-search. Ignored when ``param_grid``
        is ``None``.
    n_jobs:
        Parallelism passed to :class:`~sklearn.model_selection.GridSearchCV`.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas.DataFrame")
    if target not in data.columns:
        raise ValueError(f"Column '{target}' missing from dataframe")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if cv < 2:
        raise ValueError("cv must be >= 2")

    metric_key = _normalize_metric(metric)
    metric_fn = METRIC_FUNCTIONS[metric_key]
    scoring_key = scoring or SCORING_MAP[metric_key]

    feature_names = _resolve_features(data.columns, target, features)

    estimator = _resolve_estimator(model, random_state)

    X = data[feature_names]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    trained_estimator, best_params, cv_results = _fit_estimator(
        estimator,
        X_train,
        y_train,
        param_grid=param_grid,
        scoring=scoring_key,
        cv=cv,
        n_jobs=n_jobs,
    )

    train_score = float(metric_fn(y_train, trained_estimator.predict(X_train)))
    test_score = float(metric_fn(y_test, trained_estimator.predict(X_test)))

    return TrainingResult(
        estimator=trained_estimator,
        feature_names=feature_names,
        target_name=target,
        metric_name=metric_key,
        train_score=train_score,
        test_score=test_score,
        best_params=best_params,
        cv_results=cv_results,
    )


def _resolve_features(
    available_columns: Iterable[str], target: str, requested: Optional[Sequence[str]]
) -> List[str]:
    columns = list(available_columns)
    if requested is None:
        features = [col for col in columns if col != target]
        if not features:
            raise ValueError("No features remain after dropping the target column")
        return features

    missing = set(requested) - set(columns)
    if missing:
        raise ValueError(f"Unknown feature columns requested: {sorted(missing)}")
    if target in requested:
        raise ValueError("target column cannot be part of 'features'")
    return list(requested)


def _resolve_estimator(model: ModelInput, random_state: Optional[int]) -> BaseEstimator:
    if isinstance(model, str):
        key = model.lower()
        if key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model alias '{model}'. Available: {sorted(MODEL_REGISTRY)}")
        estimator_cls = MODEL_REGISTRY[key]
        estimator = estimator_cls()
    elif isinstance(model, BaseEstimator):
        estimator = clone(model)
    else:
        raise TypeError("model must be a string alias or an sklearn estimator")

    if hasattr(estimator, "random_state") and random_state is not None:
        setattr(estimator, "random_state", random_state)
    return estimator


def _fit_estimator(
    estimator: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    param_grid: Optional[Mapping[str, Sequence[Any]]],
    scoring: str,
    cv: int,
    n_jobs: Optional[int],
) -> Tuple[BaseEstimator, Optional[Dict[str, Any]], Optional[Mapping[str, Any]]]:
    estimator = clone(estimator)
    if param_grid:
        grid = GridSearchCV(
            estimator,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
        )
        grid.fit(X_train, y_train)
        return grid.best_estimator_, dict(grid.best_params_), grid.cv_results_

    estimator.fit(X_train, y_train)
    return estimator, None, None


def _normalize_metric(metric: str) -> str:
    key = metric.lower()
    if key not in METRIC_FUNCTIONS:
        raise ValueError(f"metric must be one of {sorted(METRIC_FUNCTIONS)}")
    return key

