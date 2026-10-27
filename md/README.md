# Pre-departure Flight Delay Classification

Predict departures delayed by at least 15 minutes at ATL, DFW and ORD, using information available under a simulated two-hour prediction cutoff. The project compares Logistic Regression, KNN, SVM and Random Forest with schedule-only and richer weather/history/traffic inputs.

**Current split:** train on 40,000 flights across January-December 2025; test on 10,000 July 2026 flights. Model selection and threshold calibration use chronological rounds within 2025.

## Start here

| Order | Notebook |
|---|---|
| 1 | [Cleaning and EDA](../notebooks/simple/01_cleaning_and_eda.ipynb) |
| 2 | [Feature Engineering](../notebooks/simple/02_feature_engineering.ipynb) |
| 3 | [Model Training](../notebooks/simple/03_model_training.ipynb) |
| 4 | [Evaluation and Findings](../notebooks/simple/04_evaluation_and_findings.ipynb) |

The four notebooks contain acquisition, cleaning, joins, aggregations, preprocessing, training and evaluation directly. Every code cell has at most ten lines, with one plot per cell. They do not import project helper scripts. See the [notebook guide](notebooks/simple/README.md) for feature formulas and outputs.

Use the existing environment with [requirements.txt](../requirements.txt). From the project root:

```sh
python3 scripts/run_simple_notebooks.py
```

The runner uses four fresh kernels in order. **This reproduces the completed July 2026 evaluation** using the frozen settings, 40,000 fitting IDs and 10,000 July flights. July is already exposed. Logistic Regression / B remains the validation-selected classifier; the simulated input-availability assumptions do not establish live performance.

## Results and deliverables

- [Final project report](../reports/final_report/output/pdf/predeparture_flight_delay_report.pdf) · [editable source and reproduction guide](reports/final_report/README.md)
- [Measured final findings](reports/final_models/RESULTS.md) · [reproduction verification](reports/simple_reproduction/README.md)

## Reference

[Objectives](OBJECTIVES.md) · [Required notebook data](NOTEBOOK_DATA_REQUIREMENTS.md) · [Dataset guide](DATASET_GUIDE.md) · [Schema and joins](SCHEMA_AND_JOINS.md) · [Scripts](scripts/README.md) · [Source inventory](../data/SOURCE_MANIFEST.json)

Run the notebooks in order. Outputs are written to `reports/simple_reproduction/`; the current settings and cohorts stay in `reports/final_models/`. Older experiment reports and bulky CSV exports have been removed.
