from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from exioml import preprocessing
from exioml.preprocessing import prepare_regression_splits


def _example_frame() -> pd.DataFrame:
    rows = []
    for i in range(25):
        region = "UNIQUE" if i == 0 else f"R{i % 3}"
        rows.append(
            {
                "schema": "PxP",
                "region": region,
                "sector": f"S{i % 5}",
                "year": 1995 + (i % 5),
                "value_added_meur": float(i),
                "employment_k": float(i + 1),
                "energy_carrier_tj": float(i + 2),
                "ghg_emissions": float(i + 3),
            }
        )
    frame = pd.DataFrame(rows)
    frame["factor_value"] = frame["ghg_emissions"]
    return frame


def _patch_loader(monkeypatch: pytest.MonkeyPatch, frame: pd.DataFrame) -> None:
    monkeypatch.setattr(preprocessing, "load_factor", lambda **_: frame.copy())


def test_prepare_regression_splits_shapes_and_features(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _example_frame()
    _patch_loader(monkeypatch, frame)

    splits = prepare_regression_splits(schema="PxP", random_state=42)

    assert len(splits.train) == 16
    assert len(splits.validation) == 4
    assert len(splits.test) == 5

    expected_features = [
        "region_looe",
        "sector_looe",
        "value_added_meur",
        "employment_k",
        "energy_carrier_tj",
        "year",
    ]
    assert splits.feature_columns == expected_features

    for column in ["value_added_meur", "employment_k", "energy_carrier_tj", "year"]:
        assert splits.train[column].min() == pytest.approx(0.0)
        assert splits.train[column].max() == pytest.approx(1.0)

    for column in ["region_looe", "sector_looe"]:
        assert column in splits.train.columns
        assert column in splits.validation.columns
        assert column in splits.test.columns


def test_leave_one_out_values_follow_training_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _example_frame()
    _patch_loader(monkeypatch, frame)

    splits = prepare_regression_splits(schema="PxP", random_state=2)

    train = splits.train
    validation = splits.validation
    test = splits.test
    global_mean = float(train["factor_value"].mean())

    for column in ["region", "sector"]:
        sum_series = train.groupby(column)["factor_value"].transform("sum")
        count_series = train.groupby(column)["factor_value"].transform("count")
        expected_train = (sum_series - train["factor_value"]) / (count_series - 1)
        expected_train = expected_train.where(count_series > 1, global_mean)
        expected_train = expected_train.fillna(global_mean)

        assert_series_equal(train[f"{column}_looe"], expected_train, check_names=False)

        means = train.groupby(column)["factor_value"].mean()
        expected_validation = validation[column].map(means).fillna(global_mean)
        expected_test = test[column].map(means).fillna(global_mean)

        assert_series_equal(
            validation[f"{column}_looe"], expected_validation, check_names=False
        )
        assert_series_equal(test[f"{column}_looe"], expected_test, check_names=False)

    assert "UNIQUE" in set(test["region"])
    unseen_mask = test["region"] == "UNIQUE"
    assert unseen_mask.any()
    unseen_values = test.loc[unseen_mask, "region_looe"].unique()
    assert len(unseen_values) == 1
    assert unseen_values[0] == pytest.approx(global_mean)
