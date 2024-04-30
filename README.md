# ExioML
Repository for paper ExioML: Eco-economic dataset for Machine Learning in Global Sectoral Sustainability, accepted at ICLR 2024 Climate Change AI workshop

## Introduction

ExioML is the first ML-ready benchmark dataset in Eco-economic research, named ExioML, for global sectoral sustainability analysis to fill the above research gaps. The overall architecture is illustrated in Figure: 

![Example Image](https://github.com/Yvnminc/ExioML/blob/main/visualisations/ExioML.png)

The ExioML is developed on top of the high-quality open-source EE-MRIO dataset ExioBase 3.8.2 with high spatiotemporal resolution, covering 163 sectors among 49 regions from 1995 to 2022, addressing the limitation in data inaccessibility. The EE-MRIO structure is described in following figure:

![Example Image](https://github.com/Yvnminc/ExioML/blob/main/visualisations/EE_MRIO.png)

 Both factor accounting in tabular format and footprint network in graph structure are included in ExioML. We demonstrate a GHG emission regression task on a factor accounting table by comparing the performance between shallow and deep models. The result achieved the low Mean Squared Error (MSE). It quantified the sectoral GHG emission in terms of value-added, employment, and energy consumption, validating the proposed dataset's usability. The footprint network in ExioML is inherent in the multi-dimensional network structure of the MRIO framework and enables tracking resource flow between international sectors. Various promising research could be done by ExioML, such as predicting the embodied emission through international trade, estimation of regional sustainability transition, and the topological change of global trading networks based on historical trajectory. ExioML reduces the barrier and reduces the intensive data pre-processing for ML researchers with the ready-to-use features, simulates the corporation of ML and Eco-economic research for new algorithms, and provides analysis with new perspectives, contributing to making sound climate policy, and promotes global sustainable development.

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
More details of the dataset are introduced in our paper: ExioML: Eco-economic dataset for Machine Learning in Global Sectoral Sustainability accepted by ICLR 2024 Climate Change AI workshop.

```
@inproceedings{guo2024exioml,
  title={ExioML: Eco-economic dataset for Machine Learning in Global Sectoral Sustainability},
  author={Yanming, Guo and Jin, Ma},
  booktitle={ICLR 2024 Workshop on Tackling Climate Change with Machine Learning},
  year={2024}
}
```

### Source data
`Exiobase` 3.8.2 is available via the [link](https://www.exiobase.eu/index.php/about-exiobase).

The developers of `Exiobase` program proposed the `Pymrio` toolkit for pre-processing of MRIO table. It is the open source code could be accessed via the [link](https://github.com/IndEcol/pymrio/tree/master).