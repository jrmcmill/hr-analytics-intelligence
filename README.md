# HR Analytics Intelligence

Work-in-progress exploratory analysis for the HRDataset_v14 employee dataset. The repo currently centers on a single notebook that cleans the raw CSV, produces descriptive workforce and attrition analysis, and writes a cleaned copy of the dataset to `data/processed/`.

## Current Structure

- `data/raw/HRDataset_v14.csv` - source dataset
- `data/processed/HRDataset_v14_cleaned.csv` - cleaned dataset exported by the notebook
- `notebooks/1_eda.ipynb` - main exploratory data analysis notebook
- `src/` - reserved for reusable Python code as the project grows

## Notebook Workflow

The current notebook covers:

- data loading and basic profiling
- string cleanup and date parsing
- workforce composition analysis
- attrition analysis
- engagement and performance analysis
- recruiting funnel and hiring outcome proxies
- manager effectiveness signals
- mobility and DEI analysis

Running the cleaning section will regenerate `data/processed/HRDataset_v14_cleaned.csv`.

## Getting Started

This project is notebook-first, so the fastest path is to open `notebooks/1_eda.ipynb` in VS Code and run the cells top to bottom.

The notebook currently uses these Python libraries:

- pandas
- numpy
- matplotlib
- geopandas
- scipy
- statsmodels

If you are using a virtual environment, activate it before launching VS Code or select it as the notebook kernel.

## Data Notes

- Keep the raw dataset under `data/raw/`.
- Keep derived or cleaned exports under `data/processed/`.
- Do not commit personal, confidential, or oversized intermediate files unless they are meant to be shared with the project.

## Roadmap

Likely next steps for the repo include:

- splitting repeated notebook logic into reusable functions under `src/`
- adding more notebooks for modeling or deeper analysis
- documenting dependencies in a requirements file
- adding tests for any reusable data-cleaning helpers
