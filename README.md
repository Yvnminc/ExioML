# ExioML
Repository for paper ExioML: Eco-economic dataset for Machine Learning in Global Sectoral Sustainability, accepted at ICLR 2024 Climate Change AI workshop

## Introduction

ExioML is the first ML-ready benchmark dataset in Eco-economic research, named ExioML, for global sectoral sustainability analysis to fill the above research gaps. The overall architecture is illustrated in Figure: 

![Example Image](https://github.com/Yvnminc/ExioML/blob/main/visualisations/ExioML.png)

The ExioML is developed on top of the high-quality open-source EE-MRIO dataset ExioBase 3.8.2 with high spatiotemporal resolution, covering 163 sectors among 49 regions from 1995 to 2022, addressing the limitation in data inaccessibility. The EE-MRIO structure is described in following figure:

![Example Image](https://github.com/Yvnminc/ExioML/blob/main/visualisations/EE_MRIO.png)

Both factor accounting in tabular format and footprint network in graph structure are included in ExioML. We demonstrate a GHG emission regression task on a factor accounting table by comparing the performance between shallow and deep models. The result achieved the low Mean Squared Error (MSE). It quantified the sectoral GHG emission in terms of value-added, employment, and energy consumption, validating the proposed dataset's usability. The footprint network in ExioML is inherent in the multi-dimensional network structure of the MRIO framework and enables tracking resource flow between international sectors. Various promising research could be done by ExioML, such as predicting the embodied emission through international trade, estimation of regional sustainability transition, and the topological change of global trading networks based on historical trajectory. ExioML reduces the barrier and reduces the intensive data pre-processing for ML researchers with the ready-to-use features, simulates the corporation of ML and Eco-economic research for new algorithms, and provides analysis with new perspectives, contributing to making sound climate policy, and promotes global sustainable development.

## PyPI Package Usage

### Installation

```bash
pip install exioml
```

Development checkouts can be installed with `pip install -e .`, exposing the same API while reading the CSV assets from the repository `data/` directory or a custom path referenced through `EXIOML_DATA_DIR`.

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

`factor_value` mirrors the canonical greenhouse-gas column so downstream pipelines can rely on a stable field name regardless of the CSV header formatting.

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

The magnitude reflects kilograms of CO₂-equivalent; apply logarithmic transforms if you need to stabilize error scales. The `train` helper accepts `model="gdbt" | "random_forest" | "ridge"` or any scikit-learn estimator instance, and `param_grid` enables a GridSearchCV run before exporting a `TrainingResult` container with prediction helpers and cross-validation diagnostics.

### Command-line inspection

The CLI exposed through `python -m exioml` mirrors the Python API for quick checks:

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

Use the in-repo orchestration utilities to replicate the workshop benchmarks without leaving the CLI:

```bash
python - <<'PY'
from src.train import ShallowModel

model = ShallowModel(type="pxp", data="clean")
print(model.train(mode="val", iter=3))
PY
```

`ShallowModel` and `DeepModel` respect the PxP/IxI splits implemented in `src/data.py`, shuffle features with seeded reproducibility, and report wall-clock time plus MSE so you can compare against the published GBDT and GANDALF baselines.

## Dataset

ExioML supports graph and tabular structure learning algorithms by Footprint Network and Factor Accounting table. The factors included in PxP and IxI in ExioML are detailed:

- `Region (Categorical feature)`: 49 regions with region code (e.g. AU, US, CN)
- `Sector (Categorical feature)`: Product (200) or industry (163) (e.g. biogasoline, construction)
- `Value Added [M.EUR] (Numerical feature)`: Value added in million of Euros
- `Employment [1000 p.] (Numerical feature)`: Population engaged in thousands of persons
- `GHG emissions [kg CO2 eq.] (Numerical feature)`: GHG emissions in kilograms of CO$_2$ equivalent
- `Energy Carrier Net Total [TJ] (Numerical feature)`: Sum of all energy carriers in Terajoules
- `Year (Numerical feature)`: 28 Timesteps (e.g. 1995, 2022)

Due to size limited in the repository, the Footprint Network is not included in the dataset. The full dataset is hosted by Zendo at the link: (https://zenodo.org/records/10604610).

### Footprint Network

The Footprint Network models the high-dimensional global trading network, capturing its economic, social, and environmental impacts. This network is structured as a directed graph, where the directionality represents the sectoral input-output relationships, delineating sectors by their roles as sources (exporting) and targets (importing). The basic element in the ExioML Footprint Network is international trade across different sectors with different features such as value-added, emission amount, and energy input. The Footprint Network's potential pathway impact is learning the dependency of sectors in the global supply chain to identify critical sectors and paths for sustainability management and optimisation. 

![Example Image](https://github.com/Yvnminc/ExioML/blob/main/visualisations/footprint.png)

### Factor Accounting

The second part of ExioML is the Factor Accounting table, which shares the common features with the Footprint Network and summarises the total heterogeneous characteristics of various sectors.

![Example Image](https://github.com/Yvnminc/ExioML/blob/main/visualisations/boxplot.png)

![Example Image](https://github.com/Yvnminc/ExioML/blob/main/visualisations/pairplot.png)

## File structures
The file structure of this study is:

```bash
├── ExioML 
│   ├──data
│   │       ├── ExioML_factor_accounting_IxI.csv
│   │       └── ExioML_factor_accounting_IxI.csv
│   ├──src
│   │       ├── data.py
│   │       ├── model.py
│   │       └── train.py
│   │       ├── tune.py
│   │       └── requirement.txt
│   ├──supply_material
│   │       ├── ExioML_slide.pdf
│   │       └── ExioML-poster.pdf
│   │       └── ExioML.pdf
│   ├──notebooks
│   │       ├── EDA.ipynb
│   │       └── ExioML_toolkit.ipynb
│   │       └── ExioML_shallow.ipynb
│   │       └── ExioML_deep.ipynb
└───└─────────────────────

```

### data
- **ExioML_factor_accounting_PxP.csv:** Sector accounting table by Product.
- **ExioML_factor_accounting_IxI.csv:** Sector accounting table by Industry.

### src
- **data.py:** Data processing and loading.
- **model.py:** Model definition.
- **train.py:** Training script.
- **tune.py:** Hyperparameter tuning script.
- **requirement.txt:** Required packages.

### supply_material
- **ExioML_slide.pdf:** Presentation slides for ICLR 2024 Climate Change AI.
- **ExioML-poster.pdf:** Poster for ICLR 2024 Climate Change AI.
- **ExioML.pdf:** Paper for ICLR 2024 Climate Change AI.

### notebooks
- **EDA.ipynb:** Exploratory data analysis.
- **ExioML_toolkit.ipynb:** Toolkit for creating ExioML dataset.
- **ExioML_shallow.ipynb:** Shallow model training.
- **ExioML_deep.ipynb:** Deep model training.

## Additional Information
### Citation
More details of the dataset are introduced in our paper: ExioML.

```
@article{guo2024exioml,
  title={ExioML: Eco-economic dataset for Machine Learning in Global Sectoral Sustainability},
  author={Guo, Yanming and Guan, Charles and Ma, Jin},
  journal={arXiv preprint arXiv:2406.09046},
  year={2024}
}
```

### Source data
`Exiobase` 3.8.2 is available via the [link](https://www.exiobase.eu/index.php/about-exiobase).

The developers of `Exiobase` program proposed the `Pymrio` toolkit for pre-processing of MRIO table. It is the open source code could be accessed via the [link](https://github.com/IndEcol/pymrio/tree/master).
