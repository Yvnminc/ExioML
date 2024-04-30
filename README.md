# ExioML
Repository for paper ExioML: Eco-economic dataset for Machine Learning in Global Sectoral Sustainability, accepted at ICLR 2024 Climate Change AI workshop

## Introduction

ExioML is the first ML-ready benchmark dataset in Eco-economic research, named ExioML, for global sectoral sustainability analysis to fill the above research gaps. The overall architecture is illustrated in Figure: 

The ExioML is developed on top of the high-quality open-source EE-MRIO dataset ExioBase 3.8.2 with high spatiotemporal resolution, covering 163 sectors among 49 regions from 1995 to 2022, addressing the limitation in data inaccessibility. The EE-MRIO structure is described in following figure:


 Both factor accounting in tabular format and footprint network in graph structure are included in ExioML. We demonstrate a GHG emission regression task on a factor accounting table by comparing the performance between shallow and deep models. The result achieved the low Mean Squared Error (MSE). It quantified the sectoral GHG emission in terms of value-added, employment, and energy consumption, validating the proposed dataset's usability. The footprint network in ExioML is inherent in the multi-dimensional network structure of the MRIO framework and enables tracking resource flow between international sectors. Various promising research could be done by ExioML, such as predicting the embodied emission through international trade, estimation of regional sustainability transition, and the topological change of global trading networks based on historical trajectory. ExioML reduces the barrier and reduces the intensive data pre-processing for ML researchers with the ready-to-use features, simulates the corporation of ML and Eco-economic research for new algorithms, and provides analysis with new perspectives, contributing to making sound climate policy, and promotes global sustainable development.

## Dataset

ExioML supports graph and tabular structure learning algorithms by Footprint Network and Factor Accounting table. The factors included in PxP and IxI in ExioML
are detailed:

Region (Categorical feature)
Sector (Categorical feature)
Value Added [M.EUR] (Numerical feature)
Employment [1000 p.] (Numerical feature)
GHG emissions [kg CO2 eq.] (Numerical feature)
Energy Carrier Net Total [TJ] (Numerical feature)
Year (Numerical feature)

Due to size limited in the repository, the Footprint Network is not included in the dataset. The full dataset is hosted by Zendo at the link: (https://doi.org/10.5281/zenodo.10604610).

### Footprint Network

The Footprint Network models the high-dimensional global trading network, capturing its economic, social, and environmental impacts. This network is structured as a directed graph, where the directionality represents the sectoral input-output relationships, delineating sectors by their roles as sources (exporting) and targets (importing). The basic element in the ExioML Footprint Network is international trade across different sectors with different features such as value-added, emission amount, and energy input. The Footprint Network's potential pathway impact is learning the dependency of sectors in the global supply chain to identify critical sectors and paths for sustainability management and optimisation. 

### Factor Accounting

The second part of ExioML is the Factor Accounting table, which shares the common features with the Footprint Network and summarises the total heterogeneous characteristics of various sectors.

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
└───└─────────────────────

```

### Additional Information

More details of the dataset are introduced in our paper: ExioML: Eco-economic dataset for Machine Learning in Global Sectoral Sustainability accepted by ICLR 2024 Climate Change AI workshop.