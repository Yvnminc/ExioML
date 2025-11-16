from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from exioml import load_factor
from exioml.datasets import (
    DatasetSplit,
    LeaveOneOutEncoder,
    build_preprocessor,
    frame_to_xy,
    prepare_dataset,
    preprocess_xy,
    split_xy,
)


@pytest.fixture()
def pxp_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data_dir = tmp_path / "data"
    cache_dir = tmp_path / "cache"
    data_dir.mkdir()
    cache_dir.mkdir()
    monkeypatch.setenv("EXIOML_DATA_DIR", str(data_dir))
    monkeypatch.setenv("EXIOML_CACHE_DIR", str(cache_dir))
    monkeypatch.delenv("EXIOML_REMOTE_BASE", raising=False)

    rows: List[Dict[str, float]] = []
    for idx in range(6):
        rows.append(
            {
                "region": "US" if idx % 2 == 0 else "CN",
                "sector": f"S{idx}",
                "Value Added [M.EUR]": 10.0 + idx,
                "Employment [1000 p.]": 2.0 + idx,
                "GHG emissions [kg CO2 eq.]": 100.0 + idx,
                "Energy Carrier Net Total [TJ]": 50.0 + idx,
                "Year": 2010 + idx,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(data_dir / "ExioML_factor_accounting_PxP.csv", index=False)
    return data_dir


def test_frame_to_xy_validates_and_casts() -> None:
    df = pd.DataFrame(
        [
            {"a": 1, "b": 2, "target": 3},
            {"a": 4, "b": np.nan, "target": 5},
        ]
    )

    X, y = frame_to_xy(df, ["a", "b"], "target", dropna="any")

    assert X.dtype == np.float32
    assert X.shape == (1, 2)
    assert y.tolist() == [3]

    with pytest.raises(ValueError):
        frame_to_xy(df, ["missing"], "target")


def test_preprocess_xy_drop_strategy_aligns_y() -> None:
    X = np.array([[1.0, np.nan], [2.0, 2.0], [4.0, 4.0]])
    y = np.array([0, 1, 2])
    preprocessor = build_preprocessor(imputer="drop", strategy="standard")

    X_proc, y_proc, fitted = preprocess_xy(X, y, preprocessor=preprocessor, fit=True)

    assert X_proc.shape[0] == 2
    assert y_proc.tolist() == [1, 2]
    assert hasattr(fitted, "transform")


def test_leave_one_out_encoder_expected_values() -> None:
    df = pd.DataFrame(
        {
            "region": ["US", "US", "CN", "CN", "CN"],
            "value": [1, 2, 3, 4, 5],
        }
    )
    y = np.array([10, 20, 30, 40, 50])

    encoder = LeaveOneOutEncoder(columns=["region"])
    encoded = encoder.fit_transform(df, y)

    assert list(encoded.columns) == ["value", "region_looe"]
    assert np.isclose(encoded["region_looe"].tolist(), [20, 10, 45, 40, 35]).all()


def test_split_xy_handles_stratify_fallback() -> None:
    X = np.arange(8, dtype=float).reshape(4, 2)
    y = np.array([0, 0, 0, 1])

    splits = split_xy(
        X,
        y,
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        stratify=True,
        random_state=0,
    )

    assert isinstance(splits, DatasetSplit)
    total = len(splits.y_train) + len(splits.y_val) + len(splits.y_test)
    assert total == 4
    assert splits.x_train.shape[1] == 2


def test_prepare_dataset_with_load_factor(pxp_env: Path) -> None:
    frame = load_factor(
        schema="PxP",
        columns=[
            "value_added_meur",
            "employment_k",
            "energy_carrier_tj",
        ],
    )
    feature_cols = ["value_added_meur", "employment_k", "energy_carrier_tj", "year"]

    splits, preprocessor = prepare_dataset(
        frame,
        feature_cols=feature_cols,
        target_col="factor_value",
        stratify=False,
        ratios=(0.5, 0.25, 0.25),
        random_state=1,
        dropna="any",
    )

    kept_rows = frame.dropna(subset=feature_cols + ["factor_value"])
    total = len(splits.y_train) + len(splits.y_val) + len(splits.y_test)
    assert total == len(kept_rows)
    assert splits.x_train.dtype == np.float32
    assert splits.x_train.shape[1] == len(feature_cols)
    assert hasattr(preprocessor, "transform")


def test_prepare_dataset_with_categoricals() -> None:
    frame = pd.DataFrame(
        {
            "region": ["US", "US", "CN", "CN"],
            "sector": ["A", "B", "A", "B"],
            "value": [1.0, 2.0, 3.0, 4.0],
            "target": [10.0, 20.0, 30.0, 40.0],
        }
    )

    splits, preprocessor = prepare_dataset(
        frame,
        feature_cols=["region", "sector", "value"],
        target_col="target",
        categorical_cols=["region", "sector"],
        ratios=(0.5, 0.25, 0.25),
        stratify=False,
        random_state=0,
    )

    assert splits.x_train.shape[1] == 3  # two encoded categoricals + numeric
    assert splits.x_train.dtype == np.float32
    assert hasattr(preprocessor, "transform")
