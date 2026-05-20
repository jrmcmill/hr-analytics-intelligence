# HR Analytics Intelligence

This repository contains an HR analytics workflow built around the `HRDataset_v14` employee dataset. It started as a notebook-first exploration project and now includes a Streamlit dashboard for day-to-day review of the engineered data.

The current project emphasizes four things:

- understanding workforce composition and demographic mix
- tracking attrition, engagement, satisfaction, and performance
- comparing departments, managers, and recruiting sources
- surfacing employee segments that are useful for intervention planning

## Repository Layout

- `data/raw/HRDataset_v14.csv` - original source dataset
- `data/processed/HRDataset_v14_cleaned.csv` - cleaned export produced during exploratory work
- `data/processed/HRDataset_v14_engineered.csv` - feature-engineered dataset used by the dashboard
- `data/processed/HRDataset_v14_model_ready.csv` - modeling dataset prepared from the engineered data
- `notebooks/1_eda.ipynb` - exploratory data analysis and visual profiling
- `notebooks/2_engineering.ipynb` - feature engineering and derived variables
- `notebooks/3_stat_analysis.ipynb` - statistical tests and comparison analysis
- `notebooks/4_modeling.ipynb` - attrition and employment-status modeling
- `notebooks/5_segmentation.ipynb` - clustering and employee segmentation
- `streamlit_app.py` - Streamlit dashboard entry point
- `requirements.txt` - Python dependencies for the notebooks and dashboard

There is no separate `src/` package yet; the reusable logic currently lives in notebooks and the dashboard file.

## Data Pipeline

The working flow in this repo is:

1. Start from `data/raw/HRDataset_v14.csv`.
2. Clean and normalize fields in `1_eda.ipynb`, which produces `data/processed/HRDataset_v14_cleaned.csv`.
3. Engineer the main analysis features in `2_engineering.ipynb`, which produces `data/processed/HRDataset_v14_engineered.csv`.
4. Prepare model inputs in `4_modeling.ipynb`, which produces `data/processed/HRDataset_v14_model_ready.csv`.
5. Use `data/processed/HRDataset_v14_engineered.csv` as the source for the Streamlit dashboard.

## Notebook Map

### `notebooks/1_eda.ipynb`

The main exploratory notebook covers:

- initial data profiling and cleanup
- workforce composition by department, state, role, generation, and demographic groups
- attrition analysis
- engagement and satisfaction review
- performance analysis
- recruiting source and hiring outcome patterns
- manager effectiveness signals
- mobility and DEI-oriented analysis

This notebook is the broadest source of the project’s descriptive findings.

### `notebooks/2_engineering.ipynb`

This notebook creates the features used downstream, including:

- tenure and age-based features
- `Tenure_Years`, `MonthsSinceReview`, `Age`, `AgeAtHire`, and related time-derived fields
- `Generation`
- `Absent_Rate`, `Lateness_Flag`, and `High_Engagement_Flag`
- salary normalization and other standardization steps

### `notebooks/3_stat_analysis.ipynb`

This notebook focuses on targeted statistical comparisons, including:

- engagement differences by department
- attrition versus salary band
- diversity hiring and recruitment-source comparisons

### `notebooks/4_modeling.ipynb`

This notebook prepares and evaluates predictive models for HR outcomes, including:

- logistic regression as a baseline
- XGBoost classifier experiments
- attrition prediction
- employment-status classification
- confusion matrices, ROC curves, and classification reports

### `notebooks/5_segmentation.ipynb`

This notebook explores employee segmentation with clustering and dimensionality reduction, including:

- HDBSCAN clustering
- UMAP visualisation
- segment profiling and outcome analysis

## Streamlit Dashboard

The dashboard is implemented in [`streamlit_app.py`](streamlit_app.py) and reads the engineered dataset from [`data/processed/HRDataset_v14_engineered.csv`](data/processed/HRDataset_v14_engineered.csv).

It includes:

- an executive overview with headcount, attrition, engagement, performance, and tenure
- workforce profile views for generation, salary band, age, tenure, gender, and geography
- attrition and risk analysis by department, manager, recruitment source, salary band, generation, and behavioral flags
- engagement and performance views, including a numeric correlation heatmap
- diversity and hiring quality analysis by source, race, and gender mix
- segmentation profiles derived from engineered fields for practical HR intervention planning

Run it with:

```bash
streamlit run streamlit_app.py
```

Use the sidebar filters to scope the dashboard by department, salary band, generation, recruitment source, and employment status.

## Setup

The project has been exercised with the virtual environment already present in the workspace. If you are starting from a fresh checkout, create and activate a Python environment, then install the dependencies listed in `requirements.txt`.

Typical setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running The Project

Open the notebooks in VS Code if you want to inspect the analysis step by step.

For the dashboard:

```bash
streamlit run streamlit_app.py
```

For notebook work, start with `notebooks/1_eda.ipynb` and proceed in order through the engineering, statistics, modeling, and segmentation notebooks.

## Dependencies

The repository currently relies on:

- pandas
- numpy
- matplotlib
- seaborn
- scipy
- statsmodels
- geopandas
- streamlit

The notebooks also depend on the broader Jupyter stack that is already present in the virtual environment.

## Notes

- The engineered dataset is the best source for dashboard analysis because it already contains the derived tenure, engagement, lateness, and generation fields.
- The dashboard is intentionally read-only and does not write back to the processed data files.
- If you add reusable transformations later, `src/` is the natural place to move shared logic out of the notebooks.