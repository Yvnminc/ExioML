from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from exioml import TrainingResult, train


@pytest.fixture()
def toy_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = 80
    frame = pd.DataFrame(
        {
            "value_added_meur": rng.normal(loc=10.0, scale=2.0, size=rows),
            "employment_k": rng.normal(loc=5.0, scale=1.0, size=rows),
            "energy_carrier_tj": rng.normal(loc=7.0, scale=1.5, size=rows),
        }
    )
    noise = rng.normal(scale=0.1, size=rows)
    frame["ghg_emissions"] = 2 * frame["value_added_meur"] - frame["employment_k"] + noise
    return frame


def test_train_returns_training_result_for_hist_gbdt(toy_frame: pd.DataFrame) -> None:
    result = train(toy_frame, target="ghg_emissions", model="gdbt", random_state=0, test_size=0.25)

    assert isinstance(result, TrainingResult)
    assert result.metric_name == "mse"
    assert result.test_score >= 0

    preds = result.predict(toy_frame[result.feature_names])
    assert preds.shape[0] == len(toy_frame)


def test_train_supports_custom_estimator_and_param_grid(toy_frame: pd.DataFrame) -> None:
    estimator = RandomForestRegressor(random_state=0)
    grid = {"n_estimators": [5], "max_depth": [2, 3]}

    result = train(
        toy_frame,
        target="ghg_emissions",
        model=estimator,
        param_grid=grid,
        cv=2,
        random_state=0,
        test_size=0.2,
    )

    assert result.best_params is not None
    assert set(result.best_params).issuperset({"n_estimators", "max_depth"})
    assert result.cv_results is not None
