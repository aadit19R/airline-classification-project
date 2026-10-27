# Notebook guide

Run the four notebooks in order with **Restart Kernel and Run All**. Every code cell has at most ten lines, and each plot has its own cell. The EDA and feature notebooks use named intermediate variables, one purpose per cell, and plain-language explanations before the more involved calculations. History features use running totals and time joins: count before the end minus count before the start.

| Order | Notebook | Main steps |
|---|---|---|
| 1 | [Cleaning and EDA](../../../notebooks/simple/01_cleaning_and_eda.ipynb) | Read sources, clean types and labels, convert times, save Parquet, explore four charts |
| 2 | [Feature Engineering](../../../notebooks/simple/02_feature_engineering.ipynb) | Calendar features, weather join, earlier-delay aggregation, scheduled traffic and pressure join |
| 3 | [Model Training](../../../notebooks/simple/03_model_training.ipynb) | Reuse chosen settings, prepare inputs, train four models with A/B inputs |
| 4 | [Evaluation and Findings](../../../notebooks/simple/04_evaluation_and_findings.ipynb) | Baselines, five main metrics, model comparison, confusion matrix and precision–recall curves |

The notebooks contain their own calculation code and import no project helper scripts. The original feature formulas and model settings remain necessary to preserve the existing scores. The feature notebook is therefore still the most involved part.

## Running

Use [requirements.txt](../../../requirements.txt). Normal runs reuse the cached sources. Notebook 01's `DOWNLOAD_MISSING=False` prevents downloads; enable it only to obtain missing files. SVM fitting takes several minutes.

The optional runner uses a fresh kernel for each notebook:

```sh
python3 scripts/run_simple_notebooks.py
```

## Training and test data

**Train:** 40,000 flights across all of 2025. **Test:** 10,000 July 2026 flights.
Model settings and thresholds were chosen during earlier chronological development within 2025. The notebooks reuse those saved choices; they do not repeat the search or tune on July. July has already been evaluated.

- A has 26 schedule features.
- B has 78 inputs for Logistic Regression/KNN, 72 for Random Forest and 66 for SVM, before encoding.
- Imputation, scaling and category encoding learn from training rows only.
- Weather must precede the prediction cutoff by 20–180 minutes.
- Delay history uses outcomes from 35 to 7 days before the cutoff.

## Saved files

1. Notebook 01 saves `data/processed/flights_clean.parquet`: all 1,689,081 scoped source schedules through July.
2. Notebook 02 saves `reports/simple_reproduction/features.parquet`: the 50,000 selected training/test rows.
3. Notebook 03 saves `reports/simple_reproduction/models.joblib`: eight fitted A/B pipelines and their thresholds.
4. Notebook 04 saves `metrics.json` and `scores.npz` in the same results folder.

The older `data/processed/features.parquet` supports the published report; it is not an input to notebook training.
The [existing report](../../../reports/final_report/output/pdf/predeparture_flight_delay_report.pdf) retains the additional subgroup, review-budget and bootstrap analysis. Those sections are omitted from this shorter notebook walkthrough.

Optional independent checks cover feature values, learned preprocessing, scores, metrics and ranks:

```sh
python3 scripts/verify_simple_reproduction.py
```

[Execution](../../../reports/simple_reproduction/execution.json) · [Verification](../../../reports/simple_reproduction/verification.json) · [Refactor evidence](../../../reports/notebook_refactor.json)
