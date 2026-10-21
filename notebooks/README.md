# Notebook sequence and current results

The preparation notebooks have a documented readability update: **01 has 15 code cells, 02 has 29 across its three sections, and 02a has 10**. Longer technical routines live in [preparation_helpers.py](../scripts/preparation_helpers.py). The original analysis and all 41 saved figures remain. See the [changes, backups and verification guide](../reports/notebook_simplification/README.md). Counts and original-cell-preservation statements below describe the historical executions before this update; the original files remain in the refactor backup.

The original three preparation/exploration notebooks ran from fresh Python 3.13 kernels on 19 October 2026: **53 code cells, 33 inline figures, no errors or warnings**. Notebook 02 now has two executed extensions: **10 code cells/four figures** for the first candidates, then **13 code cells/four figures** for the broader feature experiments. [Original update and integrity checks](../reports/PREPARATION_TIMELINE_UPDATE.md).

1. [01_cleaning_and_eda.ipynb](01_cleaning_and_eda.ipynb): raw-source audit, label/time parsing and the six EDA stages; 22 code cells and 13 figures. Saves 1,339,044 schedule/context rows.
2. [02_feature_engineering.ipynb](02_feature_engineering.ipynb): original A/B preparation (20 code cells, 13 figures), the first candidate appendix (10 code cells, four figures), and broader experiments (13 code cells, four figures, 90 development fits). The original handoff retains all 1,158,165 eligible queries; the extensions retain all 938,417 development flights in memory.
3. [02a_deeper_training_exploration.ipynb](02a_deeper_training_exploration.ipynb): training-only composition/weather/traffic exploration and audits of the representation corrections; 11 code cells and seven figures. Reads handoffs and changes no input files.
4. [03_classification.ipynb](03_classification.ipynb) is executed: 19 code cells, 12 figures and no errors/warnings. It fits Logistic Regression, KNN, SVM and Random Forest, selects on validation, freezes settings/thresholds before test access, refits the eight A/B pipelines on the full-2025 sample, and exports real Tableau results. [Measured findings](../reports/MODELLING_RESULTS.md) · [Execution checks](../reports/MODEL_EXECUTION_REPORT.md).

| Role | Dates | Eligible rows | Shared sample flag |
|---|---|---:|---|
| Initial training / outcome EDA | Jan–Sep 2025 | 698,670 | model_sample: 20,000 |
| Validation / later selection | Oct–Dec 2025 | 239,747 | model_sample: 5,000 |
| Final refit after selection | Jan–Dec 2025 | 938,417 | refit_sample: 20,000 |
| Completed original test | Jan–Mar 2026 | 219,748 | model_sample: 10,000 |

The refit population overlaps development; it is not a fourth disjoint split. Sampling happens in notebook 02 under **Feature allowlists and shared sampling**. It is proportional month × outcome sampling with seed 42, not class balancing. The separate fixed-seed scatterplot samples only control plotting density. All comparisons use identical sample IDs.

Run 01 → 02 → 02a top-to-bottom from this directory or the project root, using ../requirements.txt. Only two datasets and one shared manifest are written:

| File | Purpose |
|---|---|
| data/processed/flights_clean.parquet | All scoped schedules and history context, identities and eligibility |
| data/processed/features.parquet | Unscaled A/B predictors, target, audit fields and both membership flags |
| reports/preparation_manifest.json | Current dates, treatment definitions, source lineage, sample fingerprints and checksums |

A has 10 predictors and B has 37; the initial-training-only B preprocessing preview produces 293 columns. Month uses fixed cyclic coordinates; raw month remains for reporting. Selective missingness indicators preserve wind/gust absence without repeating condition-code and ceiling flags. Cloud extraction reads all METAR body layers before RMK.

The teacher's methods remain: X/y orientation, variable roles/cardinality, percentages, skewness, three outlier diagnostics, and original/transformed plots. Distributions, scatterplots, boxplots, rate charts, missingness and correlation plots stay inline. No extra CSV datasets or explicit IPython imports were added.

A fresh 01 save invalidates the older feature-manifest entry: rerun 02 before consuming features. Raw source files remain unchanged. Parquet uses PyArrow; startup includes the enabled user package directory to support the installed engine.

Eight user-approved weather conflicts exclude both versions and use an earlier report within the existing age limit. New conflicts, repeated flight identities or unresolved required timestamps stop for review. Valid extremes remain; no automatic IQR deletion or capping is applied.

Q1-2026 outcomes are not explored for feature/model/threshold selection. Fixed label construction and stratified sampling use their labels without showing outcome rates. Frozen rolling history may use earlier 2026 outcomes after the seven-day lag; this assumes timely feeds and does not fit a classifier on test data. Historical schedule and weather/outcome delivery availability remain unverified simulation assumptions.

See [exploration findings](../reports/DEEPER_EXPLORATION_REVIEW.md), [the six-stage plan](../EDA_PLAN.md), [objectives](../OBJECTIVES.md) and [Tableau plan](../tableau/WORKBOOK_PLAN.md).

## Classification artifacts

Notebook 03 saves one fitted-model bundle at models/flight_delay_models.joblib, one validation decision record at reports/model_selection.json, the measured findings at reports/MODELLING_RESULTS.md, and the six predeclared Tableau CSVs with export_manifest.json. It creates no extra processed datasets. The earlier notebooks, both Parquet handoffs and the shared preparation manifest retain their checksums.

Run 03 after the preparation handoffs exist. It preserves the declared protocol on repeat execution; changing model search or source features after final-test exposure stops for discussion rather than treating the same test as newly unseen. The result tables identify the fitting period: validation uses January–September fits; test uses full-2025 refits.

## Completed improvement and confirmation phase

- [04_model_improvements.ipynb](04_model_improvements.ipynb): executed 13 code cells and 5 figures, with no errors or warnings.
- [05_fresh_test_evaluation.ipynb](05_fresh_test_evaluation.ipynb): executed 10 code cells and 5 figures, with no errors or warnings.

Notebook 04 uses three chronological validation blocks within 2025, compares shared nested 20k/40k samples, and freezes the time_flight_weather package and 40,000-flight final sample. Notebook 05 confirms the original and improved pipelines on the same 10,000 new Q2-2026 flights. It performs no training/tuning. Further development stops after fresh-test access has been recorded.

[Results](../reports/improvements/RESULTS.md) · [Versioned Tableau export guide](../tableau/exports/improved/README.md). The two original Parquet tables and shared preparation manifest remain unchanged; these notebooks export no extra intermediate datasets.

## Additional feature engineering at the end of notebook 02

Start at **Additional feature engineering · a new development experiment** in a fresh kernel. The appendix is self-contained, so reviewing it does not require rerunning the original save cells. The original 38 cells, outputs and notebook metadata remain intact.

It rebuilds the current 66 predictors and creates **28 additional candidate predictors**, plus four audit-only fields, for all **938,417 eligible 2025 development flights**. Groups cover relative hourly schedule demand, weather–traffic interactions, and 7-day/42-day histories ending seven days before cutoff. Local-hour references include zero-departure hours and account for daylight-saving transitions. Log-transformed traffic reduces extreme raw ratios; unknown weather stays unknown. Three constant columns remain audit-only.

The in-memory `candidate_feature_table` and `CANDIDATE_FEATURE_SETS` support separate comparisons with 66 / 71 / 75 / 85 / 94 predictors. Existing sample flags are retained. No additional datasets are saved, no classifier is fitted, no 2026 outcome exploration occurs, and F1 gains remain untested. The saved plots show substantial class overlap and 90.59% missingness in rain/thunder interactions under the existing condition-code policy.

[Extension audit](../reports/feature_engineering_extension.json) records original-cell checksums, input integrity and independent window checks. The repeat-scoring guard in notebook 05 and the standalone improvement audit now accept this documented appendix by checking every original cell and metadata entry; all other frozen files still require their original checksums. Neither model settings nor saved evaluation results were changed.

## Broader experiments inspired by similar projects

Continue from the preceding appendix into **Broader feature experiments inspired by other projects**. This adds 64 candidates across nine groups: calendar/holidays, categorical combinations, route geometry, schedule structure, departure waves, scheduled arrivals, pressure/wind, nonlinear weather and time-conditioned earlier histories. No individual-correlation filter is used. The combined package has 158 predictors, including the previous 28 candidates and current 66.

The section screens every package with Logistic Regression/Random Forest, then compares the two best pilot packages with all four models: **90 development fits** in total. Every comparison uses shared 20,000-flight training cohorts. A validation quarter's first month selects its threshold; the following two months score it. Imputation, scaling and encoding are fitted on training only. All previously selected model settings stay fixed.

Nonlinear weather gives LR its strongest observed improvement (mean F1 0.4718 → 0.4784). Pressure/wind helps RF most (0.4663 → 0.4816). KNN changes little; SVM's F1 does not improve. The combined 158-predictor package is not the strongest pilot package. These are exploratory 2025 development findings, not new final-test results or an exhaustive parameter search.

National scheduled arrivals come from 14 existing raw files. The approved rule retains all incoming schedules and flags clock/duration disagreements. No 2026 outcomes are read; January 2026 schedule covariates supply boundary context only. No processed datasets, model bundles or Tableau exports are written by this section. Only the new experiment/audit JSON reports are saved. The original 57 cells and outputs remain intact.

[Research and source notes](../reports/BROAD_FEATURE_RESEARCH.md) · [Full measured results](../reports/broad_feature_experiments.json) · [Execution and integrity audit](../reports/broad_feature_notebook_audit.json).
