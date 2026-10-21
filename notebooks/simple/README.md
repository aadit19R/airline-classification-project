# Four short teaching notebooks

The main sequence has **190 visible code lines**. **Every code cell is seven lines or fewer**. There are **nine figures**; the remaining evidence uses tables and examples.

| Order | Notebook | Code lines | Figures |
|---|---|---:|---:|
| 1 | [Cleaning and EDA](01_cleaning_and_eda.ipynb) | 48 | 3 |
| 2 | [Feature Engineering](02_feature_engineering.ipynb) | 48 | 2 |
| 3 | [Model Training](03_model_training.ipynb) | 45 | 0 |
| 4 | [Evaluation and Findings](04_evaluation_and_findings.ipynb) | 49 | 4 |
| | **Total** | **190** | **9** |

The notebooks show the label and treatment rules, nonlinear feature formulas, four classifier definitions and explicit `.fit(X, y)` calls. Shared [module functions](../../scripts/simple_workflow.py) handle repeated decoding, joins, historical windows, preprocessing construction and result formatting. Each step has a short explanation of its inputs, purpose and output. No module fits or selects a model.

Scikit-learn supplies `Pipeline`, `ColumnTransformer`, `SimpleImputer`, `StandardScaler`, `OneHotEncoder`, `MissingIndicator`, the four classifiers, scoring functions, `classification_report`, `ConfusionMatrixDisplay` and `PrecisionRecallDisplay`. The exact frozen preprocessing recipes are reused from [final_modelling.py](../../scripts/final_modelling.py). Each pipeline learns its medians, scaling and categories only during the visible fitting calls.

**This reproduces the completed July 2026 evaluation. July is already exposed.** Model settings, feature order, fitting IDs, validation-derived thresholds and Logistic Regression / B selection remain unchanged. No feature search or test-driven adjustment is repeated.

## Run

Use the existing environment with [requirements.txt](../../requirements.txt). Open each notebook and run all cells in a fresh kernel, in order. Or, from the project root:

```sh
python3 scripts/run_simple_notebooks.py
```

The runner starts one fresh Jupyter kernel per notebook. Kernel sockets require localhost access. Runtime is dominated by the two SVM fits, which take several minutes. No downloads or new ML library are needed.

The completed [verification report](../../reports/simple_reproduction/verification.json) records the result checks. To repeat those optional checks, run `python3 scripts/verify_simple_reproduction.py`. Neither execution nor verification requires an earlier notebook or backup.

Outputs remain in `reports/simple_reproduction/`: one [Parquet handoff](../../reports/simple_reproduction/features.parquet), eight [fitted pipelines](../../reports/simple_reproduction/models.joblib), [metrics](../../reports/simple_reproduction/metrics.json), subgroup/priority JSON, compressed score arrays and [verification](../../reports/simple_reproduction/verification.json). No extra intermediate CSV or second new Parquet is created. [Execution records](../../reports/simple_reproduction/execution.json) retain actual outputs and timing.

## What the modules do

| Step | Inputs → output |
|---|---|
| `read_development_flights`, `decode_flights` | Archived raw files → scoped rows, typed dates/numbers, conversion report and identity checks |
| `add_timestamps`, `assign_periods` | Labeled rows → UTC schedules, two-hour cutoffs and original period flags |
| `load_context`, `shared_queries` | Existing context plus April–July raw rows → the same 40,000 fitting and 10,000 July queries |
| `calendar_features`, `match_weather` | Query rows and decoded reports → calendar/cyclic inputs and cutoff-matched weather |
| `history_features`, `traffic_features`, `pressure_features` | Queries and earlier events/schedules/reports → frozen history, traffic and pressure inputs |
| `model_input` | Fitting rows, visible classifier and A/B choice → X in the frozen order and an **unfitted** sklearn pipeline |
| `score_models`, `metric_table`, `error_summary`, `priority_table` | Fitted pipelines/scores → predictions, sklearn metrics, supported subgroup rates and fixed review quotas |
| `save_*`, `check_cleaning` | Explicit save/check operations → reproduction files or audit records |

These are separate stages, not one hidden end-to-end call. The module is ordinary readable Python. The original feature routines remain in [improvement_features.py](../../scripts/improvement_features.py) and [final_features.py](../../scripts/final_features.py).

## Feature formulas

- **Schedule/time:** daily sin/cos use angle 2π × minutes/1440; month uses 2π × (month−1)/12. Second harmonics and airport interactions retain the final schedule representation. Weekday remains categorical.
- **Weather matching:** latest observation ≤cutoff−20 minutes, with age ≤180 minutes at cutoff. Both versions at the ten approved conflicting station/time keys are excluded; new conflicts stop execution. Unmatched queries and unknown measurements remain missing.
- **Weather context:** dewpoint spread is temperature minus dewpoint. Wind components are knots × sin/cos(direction), with calm set to zero. Freezing/wet requires temperature ≤32°F and known wet evidence. Trace precipitation is zero with a separate flag. Unknown condition codes remain unknown.
- **Three-hour reports:** inclusive windows count reports and known precipitation, calculate the fraction of known amounts that are positive, minimum visibility, maximum gust, and visibility change from the first valid measurement. Overlapping precipitation totals are not summed.
- **History:** outcome events in [cutoff−35 days, cutoff−7 days). Origin rate = delayed/n; subgroup rate = (delayed + 20 × earlier origin rate)/(n + 20). Keep counts and no-history flags.
- **Traffic:** all origin schedules in [scheduled departure−60 minutes, scheduled departure), including later cancellations. Archived schedule availability is assumed.
- **Pressure:** parse a single METAR altimeter group before RMK (A2992 → 29.92 inHg). Inclusive three-hour change subtracts the first valid pressure; range is max−min. Keep variable wind, nonnegative gust excess and missing-pressure flags.
- **Nonlinear inputs:** the six square/log/inverse formulas remain directly visible in notebook 02 and preserve missing values.

A has 26 inputs. B has 66 for SVM, 72 for Random Forest and 78 for LR/KNN. The fitting sample spans all twelve months of 2025; the original 20k flags alone do not identify it. Development trains before April/July/October calibration and scores in the following two months. October thresholds transfer from 40k Jan–Sep fits to all-2025 refits without repair using July outcomes.

## Verification and preservation

The independent checker verifies all 78 selected features on all 50,000 rows, including missingness; sample IDs, labels and chronological boundaries; training-only preprocessing; all eight principal comparisons and four baselines; exact ranks, review quotas, subgroup support rules and dashboard summaries. It also checks the eight threshold derivations and confirms that the shared module never calls `.fit()`.

Current project raw data, frozen datasets/models/manifests/exports, published report artifacts and Tableau files are checked against [protected input hashes](../../reports/simple_reproduction/protected_checksums.json). The old notebook copies, archive bookkeeping and obsolete audit scripts have been removed after successful reproduction. Sakina's file and the reference notebook outside this project remain unchanged and are not dependencies. Original model-selection and export manifests retain their historical provenance. The pre-existing preparation-manifest hash difference from final selection remains documented in verification; both processed datasets still match the selection hashes.

[Published findings](../../reports/final_models/RESULTS.md) · [Dashboard](../../dashboard/README.md) · [Project report](../../reports/final_report/README.md).

Notebook sources are authored by [build_simple_notebooks.py](../../scripts/build_simple_notebooks.py). Rebuilding clears outputs and requires another execution.
