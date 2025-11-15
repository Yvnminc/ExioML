from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import pytest

from exioml import list_regions, list_years, load_factor

COLUMNS = [
    "region",
    "sector",
    "Value Added [M.EUR]",
    "Employment [1000 p.]",
    "GHG emissions [kg CO2 eq.]",
    "Energy Carrier Net Total [TJ]",
    "Year",
]


@pytest.fixture()
def dataset_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data_dir = tmp_path / "data"
    cache_dir = tmp_path / "cache"
    data_dir.mkdir()
    cache_dir.mkdir()
    monkeypatch.setenv("EXIOML_DATA_DIR", str(data_dir))
    monkeypatch.setenv("EXIOML_CACHE_DIR", str(cache_dir))
    monkeypatch.delenv("EXIOML_REMOTE_BASE", raising=False)

    _write_dataset(
        data_dir / "ExioML_factor_accounting_PxP.csv",
        [
            {
                "region": "US",
                "sector": "Agriculture",
                "Value Added [M.EUR]": 10.5,
                "Employment [1000 p.]": 2.1,
                "GHG emissions [kg CO2 eq.]": 100.0,
                "Energy Carrier Net Total [TJ]": 50.0,
                "Year": 2010,
            },
            {
                "region": "CN",
                "sector": "Manufacturing",
                "Value Added [M.EUR]": 12.0,
                "Employment [1000 p.]": 3.0,
                "GHG emissions [kg CO2 eq.]": 200.0,
                "Energy Carrier Net Total [TJ]": 60.0,
                "Year": 2020,
            },
        ],
    )
    _write_dataset(
        data_dir / "ExioML_factor_accounting_IxI.csv",
        [
            {
                "region": "US",
                "sector": "Services",
                "Value Added [M.EUR]": 30.0,
                "Employment [1000 p.]": 5.0,
                "GHG emissions [kg CO2 eq.]": 150.0,
                "Energy Carrier Net Total [TJ]": 70.0,
                "Year": 2015,
            },
            {
                "region": "EU",
                "sector": "Energy",
                "Value Added [M.EUR]": 40.0,
                "Employment [1000 p.]": 4.0,
                "GHG emissions [kg CO2 eq.]": 300.0,
                "Energy Carrier Net Total [TJ]": 90.0,
                "Year": 2018,
            },
        ],
    )
    return data_dir


def _write_dataset(path: Path, rows: List[Dict[str, float]]) -> None:
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame.to_csv(path, index=False)


def test_load_factor_filters_years_and_regions(dataset_env: Path) -> None:
    frame = load_factor(schema="PxP", years=[2010], regions=["us"])
    assert list(frame["schema"].unique()) == ["PxP"]
    assert frame["year"].tolist() == [2010]
    assert frame["region"].tolist() == ["US"]
    assert frame["factor_value"].iloc[0] == frame["ghg_emissions"].iloc[0]


def test_columns_argument_accepts_aliases(dataset_env: Path) -> None:
    frame = load_factor(
        schema="IxI",
        columns=["value_added_meur", "Employment [1000 p.]"],
    )
    assert "value_added_meur" in frame.columns
    assert "employment_k" in frame.columns


def test_listing_helpers(dataset_env: Path) -> None:
    assert list_regions("PxP") == ["CN", "US"]
    assert list_years("IxI") == [2015, 2018]


def test_invalid_schema_raises() -> None:
    with pytest.raises(ValueError):
        load_factor(schema="invalid")
