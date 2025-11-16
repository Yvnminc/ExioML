# ExioML
ExioML is a production-grade, ML-friendly layer on top of Exiobase 3.8.2. It delivers PxP/IxI emission-factor tables across 49 regions and 28 years, bundles ML-ready preprocessing and splits, and ships as the official PyPI package `exioml`.

## Introduction
Repository for paper ExioML: Eco-economic dataset for Machine Learning in Global Sectoral Sustainability, accepted at ICLR 2024 Climate Change AI workshop.

ExioML is the first ML-ready benchmark dataset in Eco-economic research for global sectoral sustainability analysis. The overall architecture is illustrated below:

![ExioML Architecture](https://github.com/Yvnminc/ExioML/blob/main/visualisations/ExioML.png)

Built on the high-quality open-source EE-MRIO dataset ExioBase 3.8.2 with high spatiotemporal resolution, ExioML covers 163 sectors across 49 regions from 1995 to 2022. The EE-MRIO structure is shown here:

![EE-MRIO Structure](https://github.com/Yvnminc/ExioML/blob/main/visualisations/EE_MRIO.png)

Both factor accounting tables and footprint networks are included. We demonstrate a GHG emission regression task comparing shallow and deep models; results achieve low MSE, quantifying sectoral GHG emission in terms of value-added, employment, and energy consumption. The footprint network enables tracking resource flow between international sectors and supports research such as embodied emission prediction via trade, regional transition estimation, and global trading network topology analysis. ExioML lowers preprocessing barriers for ML researchers and supports policy-relevant sustainability insights.

## Highlights
- One-line install: `pip install exioml`, plus CLI entry (`exioml` or `python -m exioml --list-regions`) to inspect regions/years.
- `load_factor` lazily fetches and caches PxP/IxI factor tables with column aliasing and hash checks.
- Unified X/y preparation: `frame_to_xy`, `build_preprocessor`, `preprocess_xy`, `split_xy`, `prepare_dataset` provide leave-one-out target encoding for categoricals and scaling for numerics by default.
- pymrio compatible: direct MRIO → tidy factors → ML splits (`notebooks/pymrio_demo.ipynb`).
- Baseline shallow/deep models (GBDT, GANDALF, etc.) with reproducible splits.

## PyPI Package Usage

### Installation

```bash
pip install exioml
# Developer install (shares local data/ assets)
pip install -e .
```

### Loading emission-factor tables

```python
from exioml import load_factor

frame = load_factor(
    schema="PxP",
    years=[1995],
    regions=["AT"],
    columns=["value_added_meur", "employment_k", "energy_carrier_tj"],
)
print(frame.head().to_markdown(index=False))
```

Sample output:

| schema   | region   | sector                  |   year |   ghg_emissions |   factor_value |   value_added_meur |   energy_carrier_tj |   employment_k |
|:---------|:---------|:------------------------|-------:|----------------:|---------------:|-------------------:|--------------------:|---------------:|
| PxP      | AT       | Wheat                   |   1995 |     4.03721e+08 |    4.03721e+08 |           173.076  |            1956.55  |       12.4647  |
| PxP      | AT       | Cereal grains nec       |   1995 |     8.25645e+08 |    8.25645e+08 |           389.064  |            3520.85  |       24.8328  |
| PxP      | AT       | Vegetables, fruit, nuts |   1995 |     2.7892e+08  |    2.7892e+08  |           830.191  |            2974.92  |       48.6157  |
| PxP      | AT       | Oil seeds               |   1995 |     1.60796e+08 |    1.60796e+08 |           102.858  |             265.091 |        2.9306  |
| PxP      | AT       | Sugar cane, sugar beet  |   1995 |     1.00478e+08 |    1.00478e+08 |            31.7525 |             219.926 |        3.14141 |

`factor_value` mirrors the canonical greenhouse-gas column so downstream pipelines can rely on a stable field name regardless of CSV header formatting.

### Preparing regression-ready splits

Benchmark experiments rely on deterministic splits, normalization, and categorical encodings that you can recreate via `prepare_regression_splits`:

```python
from exioml import prepare_regression_splits

splits = prepare_regression_splits(schema="PxP", years=[2010, 2011], regions=["US", "CN"])
print(splits.train.shape, splits.validation.shape, splits.test.shape)
```

`prepare_regression_splits` returns a `RegressionSplits` dataclass with `.train`, `.validation`, and `.test` frames (64/16/20 split), plus metadata describing the `feature_columns` and `target_column` expected by training code. Continuous fields (`value_added_meur`, `employment_k`, `energy_carrier_tj`, `year`) are min-max scaled and `region`/`sector` receive leave-one-out encodings to prevent leakage.

### Training baseline models

```python
from exioml import load_factor, train

df = load_factor(
    schema="PxP",
    years=[1995],
    regions=["AT"],
    columns=["value_added_meur", "employment_k", "energy_carrier_tj"],
)
result = train(
    df,
    target="factor_value",
    model="gdbt",
    features=["value_added_meur", "employment_k", "energy_carrier_tj"],
    test_size=0.25,
    random_state=7,
)
print(f"Hold-out {result.metric_name.upper()}: {result.test_score:.2e}")
```

Typical metrics for the Austrian 1995 PxP slice are:

```
{
  "train_mse": 2.16e17,
  "test_mse": 3.58e17,
  "best_params": null
}
```

The magnitude reflects kilograms of CO₂-equivalent; apply logarithmic transforms if you need to stabilize error scales. The `train` helper accepts `model="gdbt" | "random_forest" | "ridge"` or any scikit-learn estimator instance; `param_grid` enables GridSearchCV before exporting a `TrainingResult` with prediction helpers and cross-validation diagnostics.

### Command-line inspection

```bash
python -m exioml --list-regions --schema PxP | head -n 5
AT
AU
BE
BG
BR

python -m exioml --schema PxP --years 1995 --regions AT --columns value_added_meur energy_carrier_tj --limit 3
schema region                  sector  year  ghg_emissions  factor_value  energy_carrier_tj  value_added_meur
   PxP     AT                   Wheat  1995   4.037211e+08  4.037211e+08        1956.549408        173.076067
   PxP     AT       Cereal grains nec  1995   8.256448e+08  8.256448e+08        3520.851032        389.064273
   PxP     AT Vegetables, fruit, nuts  1995   2.789203e+08  2.789203e+08        2974.922772        830.191187
```

### Repository training entry points

```bash
python - <<'PY'
from src.train import ShallowModel

model = ShallowModel(type="pxp", data="clean")
print(model.train(mode="val", iter=3))
PY
```

`ShallowModel` and `DeepModel` respect the PxP/IxI splits implemented in `src/data.py`, shuffle features with seeded reproducibility, and report wall-clock time plus MSE so you can compare against the published GBDT and GANDALF baselines.

## Quickstart Examples

### 1) Factor table + decision tree grid search (`notebooks/exioml_demo.ipynb`)

```python
from exioml import load_factor, list_regions, list_years
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
import numpy as np

years = list_years("PxP")
regions = list_regions("PxP")
frame = load_factor(
    schema="PxP",
    years=[2010, 2011, 2012],
    regions=["US", "CN", "DE", "JP"],
    columns=["value_added_meur", "employment_k", "energy_carrier_tj"],
).dropna()

# Log-scale numeric columns
num_cols = frame.select_dtypes(include=[np.number]).columns.tolist()
frame[num_cols] = frame[num_cols].apply(np.log1p)

X = frame[["region", "sector", "year", "employment_k", "energy_carrier_tj", "value_added_meur"]]
y = frame["factor_value"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocess = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"), ["region", "sector"]),
    ("numeric", "passthrough", ["year", "employment_k", "energy_carrier_tj", "value_added_meur"]),
])

grid = GridSearchCV(
    Pipeline([("preprocess", preprocess), ("model", DecisionTreeRegressor(random_state=42))]),
    param_grid={
        "model__max_depth": [4, 6, None],
        "model__min_samples_split": [2, 10],
        "model__min_samples_leaf": [1, 5],
    },
    scoring="neg_mean_squared_error",
    cv=3,
    n_jobs=-1,
)
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)
test_mse = mean_squared_error(y_test, grid.best_estimator_.predict(X_test))
print("Test MSE:", test_mse)
```

### 2) pymrio × ExioML end-to-end (`notebooks/pymrio_demo.ipynb`)

```python
import pandas as pd
import pymrio
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from exioml.datasets import prepare_dataset

mrio = pymrio.load_test()
mrio.calc_all()
ext = mrio.emissions.F

factors = ext.stack(["region", "sector"]).reset_index().rename(columns={0: "emission"})
factors["region_code"] = pd.factorize(factors["region"])[0].astype("float32")
factors["sector_code"] = pd.factorize(factors["sector"])[0].astype("float32")

sample = factors.sample(min(120, len(factors)), random_state=0)
feature_cols = ["region", "sector", "region_code", "sector_code"]

splits, preproc = prepare_dataset(
    sample,
    feature_cols=feature_cols,
    target_col="emission",
    categorical_cols=["region", "sector"],
    ratios=(0.7, 0.15, 0.15),
    stratify=False,
)

model = HistGradientBoostingRegressor(random_state=0).fit(splits.x_train, splits.y_train)

def report(split, X, y):
    preds = model.predict(X)
    return {"split": split, "mae": mean_absolute_error(y, preds), "r2": r2_score(y, preds)}

print(pd.DataFrame([
    report("train", splits.x_train, splits.y_train),
    report("val", splits.x_val, splits.y_val),
    report("test", splits.x_test, splits.y_test),
]))
```

## X/y Preparation API
- `frame_to_xy(df, feature_cols, target_col, ...)`: validate required columns, configure NA handling, return numpy or DataFrame.
- `build_preprocessor(strategy="standard"|"minmax", imputer="drop"|"median", categorical_cols=..., leave_one_out=True)`: build preprocessing pipeline; default leave-one-out encoding + scaling.
- `preprocess_xy(X, y, preprocessor)`: fit/transform while keeping X/y aligned.
- `split_xy(X, y, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, stratify=True)`: two-stage split with stratified fallback.
- `prepare_dataset(...)`: one-stop call returning `DatasetSplit` and the fitted preprocessor.

## Dataset

ExioML supports both graph and tabular learning via footprint networks and factor accounting tables. Factors in PxP/IxI include:

- `Region (Categorical feature)`: 49 regions with region code (e.g., AU, US, CN)
- `Sector (Categorical feature)`: Product (200) or industry (163) (e.g., biogasoline, construction)
- `Value Added [M.EUR] (Numerical feature)`
- `Employment [1000 p.] (Numerical feature)`
- `GHG emissions [kg CO2 eq.] (Numerical feature)`
- `Energy Carrier Net Total [TJ] (Numerical feature)`
- `Year (Numerical feature)`: 28 timesteps (1995–2022)

Due to size limits, the footprint network is hosted externally on Zenodo: https://zenodo.org/records/10604610.

### Footprint Network

The footprint network captures directed sectoral input-output relationships across regions, with attributes such as value-added, emissions, and energy inputs. It enables tracing supply-chain dependencies and critical pathways for sustainability management.

![Footprint Network](https://github.com/Yvnminc/ExioML/blob/main/visualisations/footprint.png)

### Factor Accounting

Factor accounting tables summarize sector characteristics shared with the footprint network.

![Boxplot](https://github.com/Yvnminc/ExioML/blob/main/visualisations/boxplot.png)

![Pairplot](https://github.com/Yvnminc/ExioML/blob/main/visualisations/pairplot.png)

## Repository Layout (key paths)
```
├── data/ExioML_factor_accounting_{PxP,IxI}.csv       # Factor accounting tables
├── notebooks/
│   ├── exioml_demo.ipynb                             # PyPI + decision tree grid search
│   ├── pymrio_demo.ipynb                             # pymrio → ExioML end-to-end
│   ├── EDA.ipynb / ExioML_toolkit.ipynb              # Exploration and toolkit
│   ├── ExioML_shallow.ipynb / ExioML_deep.ipynb      # Model walkthroughs
├── src/
│   ├── data.py / model.py / train.py / tune.py        # Workshop experiment pipeline
│   └── exioml/                                        # PyPI package source
│       ├── __init__.py / cli.py / __main__.py         # Public API and CLI entry points
│       ├── data_io.py / factors.py                    # Factor loading and caching
│       ├── datasets.py / preprocessing.py             # X/y conversion, preprocessing, splits
│       └── training.py / logging_utils.py             # Training helpers and logging
├── tests/                                            # pytest coverage for public API and preprocessing
├── visualisations/                                   # Figures used in the paper
└── supply_material/                                  # Slides and poster assets
```

## Citation

```
@article{guo2024exioml,
  title={ExioML: Eco-economic dataset for Machine Learning in Global Sectoral Sustainability},
  author={Guo, Yanming and Guan, Charles and Ma, Jin},
  journal={arXiv preprint arXiv:2406.09046},
  year={2024}
}
```

## Source Data
Exiobase 3.8.2 is available via https://www.exiobase.eu/index.php/about-exiobase. The Exiobase developers provide the open-source `pymrio` toolkit for MRIO preprocessing: https://github.com/IndEcol/pymrio/tree/master.
