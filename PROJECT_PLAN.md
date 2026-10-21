> **Current deliverable:** final model development, July confirmation, the [four-page offline HTML dashboard](dashboard/dist/departure_delay_dashboard.html) and the [nine-page report](reports/final_report/output/pdf/predeparture_flight_delay_report.pdf) are complete. The user replaced Tableau/Power BI with HTML. See [dashboard controls and verification](dashboard/README.md) and [report sources and reproduction](reports/final_report/README.md). The Tableau-specific steps and Q1/Q2 stages below are retained as historical planning; review the report against any course-specific submission requirements before submitting.

# Fresh project: pre-departure flight-delay classification

Updated 19 October 2026. Data preparation, 34 validation trials, the eight finalized A/B refits and frozen Q1-2026 testing are complete. The executed classification notebook and all six Tableau exports contain real results. [Measured findings](reports/MODELLING_RESULTS.md) · [Execution checks](reports/MODEL_EXECUTION_REPORT.md). The Tableau workbook and final submission remain.

**Title:** Pre-departure Flight Delay Risk Classification using Logistic Regression, KNN, SVM and Random Forest.

**Question:** In a historical simulation with a cutoff two hours before scheduled departure, how well can we identify flights that depart at least 15 minutes late, and does weather/history/traffic improve on schedules alone?

The governing [main and four supporting objectives](OBJECTIVES.md) are defined before implementation. The visual findings deliverable is a Tableau packaged workbook, with each supporting objective mapped to one dashboard.

Raw source locations and checksums are listed in [the raw-data guide](data/raw/README.md). The raw-only input rule overrides any earlier suggestion to reuse derived lookups or prepared weather/flight tables.

## 1. Keep the scope manageable

Use 2025 departures for development/final refit and Q1-2026 departures for final evaluation at ATL, ORD and DFW, across the reporting airlines in the data. This is a three-hub study, not a claim about all airports. The 2025 files contain 957,315 scheduled rows from those origins and 938,417 noncancelled flights with known departure delay. Of those, 241,232 (25.71%) departed at least 15 minutes late. The new Q1-2026 sources add 228,586 schedules and 219,748 eligible final-test queries. Required clock/timezone validation passed; 2026 delay rates remain unexplored.

Use the full initial-training population for descriptive analysis and all appropriately earlier raw-derived records for history/traffic context. Keep final-test outcome EDA reserved. Use a common sample for fitting and comparing models:

| Period | Role | Approved sample |
|---|---|---:|
| January–September 2025 | Initial training | 20,000 |
| October–December 2025 | Validation/tuning | 5,000 |
| January–December 2025 | Final refit after choices are frozen | Separate shared 20,000-flight sample |
| January–March 2026 | Final evaluation | 10,000 |

Sample within each block proportionally by month and outcome; retain natural validation/test prevalence. Save the seed and flight identities. All models and feature sets use the same query flights.

The earlier project already examined 2025 outcomes. The user approved Q1 2026 as the new final test; its outcomes remain outside exploration and selection. The structural source audit loads no outcome values. See [the acquisition and timeline report](reports/DATA_ACQUISITION_2026_Q1.md).

## 2. Define the new target and information cutoff

Set `delayed_departure = 1` when `DepDelay >= 15`; otherwise 0. This matches [BTS's DepDel15 definition](https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ). Departure delay constructs y and is excluded from X.

Exclude cancelled flights and unknown departure labels, recording counts and reasons. Later-diverted flights can remain if their departure outcome is known: an arrival-delay exclusion rule should not automatically carry over. State that this predicts departure delay conditional on operation, not cancellation risk for every scheduled flight.

Convert scheduled origin-local departure to UTC and subtract 120 minutes. Weather and earlier flight outcomes must precede that cutoff under the declared delivery assumptions. The archives do not verify actual live delivery times or reconstruct historical schedule revisions. The current flight's actual departure/arrival times, departure/arrival delay, taxi-out, actual duration and cause-of-delay minutes are excluded from predictors.

## 3. Reuse source data; rebuild the experiment

Use this new project's own raw flight CSVs, airport reference and original weather/station responses under data/raw. Only unchanged raw source bytes may be copied from earlier work. Rebuild cleaning, mappings, labels, features and model outputs here; do not read prior processed/interim/report/model datasets or generated timezone lookups.

Rebuild the departure label, eligible population, historical departure-delay rates, sampling, model pipelines and evaluation. The old severe-arrival-delay rates must not be renamed as departure-delay history.

Create a separate `departure-delay-classification` folder. The old folder can remain an archive. The new project does not require deleting data or copying every previous experiment.

## 4. Compare two feature sets

| Set | Inputs | Data work |
|---|---|---|
| A: schedule | Airline, origin, destination, month, weekday, scheduled departure time, distance, scheduled duration | Already available. One-hot encode airline/airports/weekday; use fixed cyclic month and clock coordinates. |
| B: A + weather/history/traffic | Origin wind/gusts, visibility, ceiling, rain/snow/thunderstorm or precipitation; earlier airline/origin/route departure-delay rates; scheduled departures around the query time | Weather is cached. Rebuild history for the new label and retain missing-value indicators. |

For B, join observations available by cutoff, or forecasts issued by then. Do not use full-day weather summaries or weather observed at scheduled departure. Existing observation joins use a 20-minute assumed delivery buffer and age limit; preserve and disclose those assumptions or define a documented replacement. An existing matched weather record does not establish that every weather measurement is complete.

A simple history policy is a 28-day window ending seven days earlier, with smoothing and missing/short-history flags. Use only earlier eligible outcomes; January needs earlier context or explicitly missing history. This simulates an outcome feed: monthly BTS releases alone do not establish live daily availability. Compute context from the full relevant source population before sampling model rows.

Aircraft rotation (previously C) is excluded from the agreed project and all four objectives. BTS contains final flown registrations, but not when those assignments became known. The complete raw files remain stored; do not use `Tail_Number`, incoming-aircraft outcomes or turnaround features linked through final assignments in A/B. The current flight's `LateAircraftDelay` cause minutes are also excluded.

O2 is fully answered by comparing A with B on identical flights. This measures the combined added information; isolated weather/history/traffic claims require additional predeclared comparisons. State the input-availability assumptions throughout the report and Tableau workbook. The results evaluate a historical simulation, not verified live warning performance.

## 5. Cleaning and storage

Follow the [six-stage EDA and preparation plan](EDA_PLAN.md): univariate analysis, bivariate analysis, missing value treatment, outlier treatment, variable creation and variable transformation. A source-quality audit, chronological split assignment and essential label/time parsing come first. Recheck distributions after treatment and after richer features are created.

Explore relationships and choose data-dependent treatments using training-period data. Learned imputation, clipping, encoding and scaling belong inside training-fitted pipelines; final-period outcomes must not drive feature or treatment choices. Valid severe delays are retained rather than removed automatically as outliers.

Audit schema, monthly coverage, missingness, duplicates and flight identity. Parse dates/numerics, handle 2400/overnight/DST cases, standardize airport/airline codes, and distinguish structural missingness from corrupt values. Preserve legitimate early departures and extreme operational delays unless a documented validity rule rejects them.

Validate join cardinality and row counts at each stage. Keep source-row identifiers. Store validated monthly Parquet partitions, a cleaning log and a feature dictionary. Small aggregate CSVs can serve Tableau.

## 6. Fit familiar models with a small search

Use training-only preprocessing in a scikit-learn Pipeline/ColumnTransformer: numeric median imputation; categorical imputation and one-hot encoding with a policy for unknown categories. Scale numeric features for Logistic Regression, KNN and SVM. Random Forest uses the same imputed numeric information and encoded categories without requiring numeric scaling. Do not assign arbitrary numeric airline/airport IDs as distances.

Fit logistic regression first, then KNN, SVM and Random Forest. Start with fixed settings and the same sample. A small later search can use C=0.1/1/10 for logistic/SVM and k=5/15/31 for KNN. LinearSVC is an SVM suitable for a larger runtime budget; RBF SVC is feasible to try on the fixed sample. [Official SVC documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) explains its scaling limits.

Random Forest is the fourth candidate. Start with 200 trees, depth 12, minimum leaf size 10 and balanced class weights, then compare a small validation-only depth/leaf search. These are initial settings, not measured best parameters. Use the same sampled flights and feature sets as the other classifiers. See the [official classifier documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html). No additional package beyond scikit-learn is needed.

Keep temporal training/validation/test blocks. If using cross-validation, expand earlier-date training blocks into later validation blocks; do not shuffle chronology. See [scikit-learn's time-series validation guidance](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split).

Compare an always-on-time baseline and optionally a training-derived hour/airline risk baseline. Class weights can be a training choice; validation and test retain natural prevalence. Select parameters and any thresholds on development data, freeze choices, then evaluate predeclared final comparisons once. LinearSVC scores can support ranking without a probability-calibration stage.

Report delayed-class precision, recall and F1; average precision; accuracy, ROC-AUC, confusion matrices and training/prediction time. Keep a 2-by-4 feature-set/model table and record coverage. Do not promise an accuracy threshold or claim new features help until measured.

## 7. Use a notebook-first structure

```text
departure-delay-classification/
  README.md
  requirements.txt
  config/project.json
  data/raw/                 # or documented immutable source paths
  data/processed/           # flights_clean.parquet and features.parquet
  notebooks/
    01_cleaning_and_eda.ipynb
    02_feature_engineering.ipynb
    02a_deeper_training_exploration.ipynb
    03_classification.ipynb
  models/                   # one fitted pipeline bundle
  reports/
  tableau/exports/          # six result CSVs and an export manifest
```

Notebook 1: extraction/audit, cleaning, missingness, label balance, airport/airline/time patterns and exclusions. Notebook 2: label/cutoff, A/B under the stated availability assumptions, chronological splits and validated feature data. Notebook 3: preprocessing, model comparisons, frozen decisions, final metrics and error analysis. All run top-to-bottom with plain-language explanations.

Core dependencies are pandas, NumPy, scikit-learn, PyArrow, Matplotlib, Seaborn, joblib and Jupyter; add download/timezone helpers when needed. CatBoost is not part of this design.

The Tableau workbook has four principal ML-results dashboards: model performance (O1), feature value (O2), prediction reliability (O3), and risk prioritization (O4). Raw delay summaries provide context; they do not replace answers to the objectives. Follow [the workbook specification](tableau/WORKBOOK_PLAN.md) and [export contract](tableau/export_contract.json).

## 8. Build in this order

1. Freeze scope, target, cutoff and population; create the independent folder/README.
2. Complete cleaning/EDA and save a labelled dataset.
3. Run schedule-only logistic regression end-to-end.
4. Add B and compare all four models on identical flights.
5. Check weather/history cutoff logic, record availability assumptions and confirm all A/B candidates use identical flights.
6. Freeze choices, evaluate final comparisons, and produce dashboard exports and the report.

The experiment now supplies real evidence for the four objectives through model comparisons, paired A/B changes, supported subgroup errors and review-quota rankings. Aircraft rotation remains outside scope. The chosen model is registered from validation; test results describe the fixed candidates without selecting a new winner. Use these outputs in the remaining Tableau workbook and final report.

Scope evidence: [verified monthly counts](data/scope_audit.json).

## Completed improvement confirmation

Notebooks 04 and 05 now implement the bounded improvement study and its frozen Q2-2026 confirmation. Q1 remains the completed original benchmark. The current result tables are versioned under tableau/exports/improved/; see reports/improvements/RESULTS.md. Building, packaging and reconciling the Tableau workbook remains the next stage.
