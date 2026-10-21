# Pre-departure Flight Delay Classification

Predict departures delayed by at least 15 minutes at ATL, DFW and ORD, using information available under a simulated two-hour prediction cutoff. The project compares Logistic Regression, KNN, SVM and Random Forest with schedule-only and richer weather/history/traffic inputs.

## Start here

| Order | Notebook |
|---|---|
| 1 | [Cleaning and EDA](notebooks/simple/01_cleaning_and_eda.ipynb) |
| 2 | [Feature Engineering](notebooks/simple/02_feature_engineering.ipynb) |
| 3 | [Model Training](notebooks/simple/03_model_training.ipynb) |
| 4 | [Evaluation and Findings](notebooks/simple/04_evaluation_and_findings.ipynb) |

The main sequence has **190 visible code lines, nine figures and a maximum of seven lines per code cell**. Shared Python modules handle repeated work; classifier definitions and `.fit()` calls stay visible. See the [notebook guide](notebooks/simple/README.md) for feature formulas, module explanations and outputs.

Use the existing environment with [requirements.txt](requirements.txt). From this project directory:

```sh
python3 scripts/run_simple_notebooks.py
```

The runner uses four fresh kernels in order. **This reproduces the completed July 2026 evaluation** using the frozen settings, 40,000 fitting IDs and 10,000 July flights. July is already exposed. Logistic Regression / B remains the validation-selected classifier; the simulated input-availability assumptions do not establish live performance.

## Results and deliverables

- [Interactive HTML dashboard](dashboard/dist/departure_delay_dashboard.html) · [usage guide](dashboard/README.md)
- [Final project report](reports/final_report/output/pdf/predeparture_flight_delay_report.pdf) · [editable source and reproduction guide](reports/final_report/README.md)
- [Measured final findings](reports/final_models/RESULTS.md) · [reproduction verification](reports/simple_reproduction/README.md)
- [Frozen evaluation exports](tableau/exports/final_july/README.md) · [Tableau workspace](tableau/workbooks/README.md)

## Reference

[Objectives](OBJECTIVES.md) · [Dataset guide](DATASET_GUIDE.md) · [Schema and joins](SCHEMA_AND_JOINS.md) · [Scripts](scripts/README.md) · [Source inventory](data/SOURCE_MANIFEST.json)

The four notebooks run independently. Raw data, processed datasets, fitted models, selection records and evaluation exports remain in their existing locations. Reproduction outputs and the completed checks are kept separately in `reports/simple_reproduction/`. The optional `scripts/verify_simple_reproduction.py` checks current results without any old notebook or backup.
