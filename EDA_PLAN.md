# Cleaning, EDA and feature preparation

Status: both core notebooks and the supplementary exploration have completed from fresh kernels under the approved 2025-development/Q1-2026-test chronology. Both handoffs are rebuilt; all 53 code cells and 33 figures are saved. Use only this project's raw sources. Keep original files unchanged and save new derived outputs here. The user requested frequent inline plots, minimal dataset exports and clarification of material ambiguities before treatment code is written. Routine documented preprocessing decisions are delegated.

The teacher's reference notebook has been reviewed and its applicable methods added to both core notebooks. Follow the inspect → summarize → visualize → diagnose → compare pattern described in [the reference methodology review](reports/REFERENCE_EDA_REVIEW.md), retaining the six-stage structure and project-specific cutoff rules.

## Before the six stages: establish the data and evaluation boundaries

- Audit file/schema coverage, types, units, duplicate records, stable flight identity and join cardinality. Reconcile every population exclusion and preserve source-row identifiers.
- Parse source missing codes, dates and clock values before plotting. Handle midnight/2400, overnight and daylight-saving cases explicitly. Invalid values must not silently become zeros.
- Approved chronology: initial training January–September 2025; validation October–December 2025; final refit on 2025 after selection; final evaluation January–March 2026. Late-2024 records warm up early-2025 history, and late-2025 records warm up early-2026 history. Current handoffs implement these roles. A separate 20,000-flight full-2025 refit flag is reserved for fitting after validation selection.
- Create the departure label and cutoff early because bivariate analysis needs them: departure delay >=15 minutes; scheduled departure minus two hours. Exclude cancellations and unknown labels without imputing the target; retain known departures even when the flight is later diverted.
- Use training-period data for exploratory relationships and data-dependent treatment choices. Validation supports the declared model/treatment selection. Final-period data receives fixed integrity checks and frozen transformations; do not use its outcome patterns to redesign features or cleaning. The prior project's exposure to 2025 remains disclosed.

The six stages are the reporting structure. Essential parsing, target creation and safe join keys occur first; richer variables are created later. Revisit distributions after treatment and after weather/history features exist, rather than forcing every operation into a single pass.

## 1. Univariate analysis

Inspect each relevant variable separately on training data:

- Numeric distributions: distance, scheduled duration and scheduled time; after safe feature joins, wind/gusts, visibility, precipitation, weather age and history support. Report count, missingness, median, quantiles and range with suitable histograms/boxplots.
- Categorical frequencies: airport, airline, destination, month, weekday, time band and later weather flags. Report counts, percentages, unique levels and intended variable roles, rather than relying on dtype alone.
- Add explicit mean/median/skewness summaries for suitable numerical variables. Binary flags and cyclic clock values need role-appropriate interpretation. Show X/y clearly in an exploration view; final model predictors still require the A/B allowlists.
- Target balance: delayed versus on-time flights and the signed departure-delay distribution. Departure delay is an outcome/audit field, not a predictor.

Interpret skew, sparsity, unusual codes and units. Sample large scatterplots reproducibly, and label their sample sizes; calculate reported summaries on the stated eligible population.

## 2. Bivariate analysis

- Compare delay rates across airlines, airports, weekdays, months and departure time bands, always showing flight counts.
- Compare candidate numerical distributions between delayed and on-time flights; use meaningful bins when showing delay rates against distance, duration or weather.
- After B features exist, examine weather/history/traffic versus the label. These analyses use cutoff-respecting joins, not weather at departure time or full-day summaries.
- Review numeric predictor correlations for redundancy. A correlation matrix is not suitable for arbitrary numeric airline/airport IDs. Mark small groups and avoid causal explanations from these plots.

These are descriptive findings and candidate-feature checks. O2 is answered later by retraining A/B models on identical flights, not by an attractive correlation chart. Keep additional interactions focused and motivated by the project.

## 3. Missing value treatment

Produce a missingness table by variable, origin and training month, including weather match rates and observation ages. Distinguish missing source measurements, structural absence, parsing errors and unmatched joins.

- Never impute missing departure labels or essential identities/timestamps needed to establish a valid query; log the applicable exclusion rule and count.
- Repair deterministic parsing/reference issues where supported by raw evidence. Review whether a weather value is genuinely absent before interpreting it as zero or an absent condition.
- Retain flights with missing richer covariates so A and B have identical query populations. Add meaningful missingness/support flags and predeclare any unusable-feature removal.
- Learn numeric imputation values and categorical fill policies inside the training pipeline. During development, fit them separately within each training fold; apply the frozen policies to validation/final data. Do not globally impute the dataset in EDA.

## 4. Outlier treatment

Use plots, quantiles and domain checks to flag extremes. Separate valid rare events from invalid records.

Compare the teacher's IQR, mean ± 3 SD and upper-99th-percentile rules on training predictors, reporting bounds and flagged counts. These diagnostic comparisons do not mutate data. For zero-heavy precipitation, an IQR of zero may flag positive observations without making them errors.

- Preserve legitimate early departures, severe delays, long routes and extreme weather. Do not delete flights merely because departure delay or a weather value lies outside an IQR rule; those events may be the phenomenon being predicted.
- Correct or mark impossible predictor values as missing only under a recorded validity rule. Exclude records only when essential query validity or labels cannot be established.
- Capping is not mandatory. If justified for a predictor, learn limits on training data and apply them consistently. Do not cap the label-defining delay or remove test extremes to improve scores.
- Keep full ranges in saved data. Any zoomed/clipped display must say so and report omitted counts. Compare pre/post-treatment summaries and record affected rows.

## 5. Variable creation

Create a feature dictionary with definition, unit, source, availability rule, feature set and missing-value policy.

- A: calendar features, scheduled departure bands/cyclic time, airline, origin/destination or a predeclared route representation, distance and scheduled duration. Avoid redundant categorical representations without a reason.
- B: cutoff-matched origin weather and missingness/age flags; freshly calculated earlier departure-delay rates with support; scheduled departure traffic. Initial history uses 28 days ending seven days earlier. Initial weather availability assumes a 20-minute buffer and a three-hour maximum age. Preserve and disclose the historical-feed and schedule-snapshot assumptions.
- Calculate history/traffic context before sampling query flights. Never use outcomes later than the declared historical window, current-flight actual departure/arrival data or delay-cause minutes as predictors. Frozen rolling history can include earlier 2026 outcomes after the seven-day lag during sequential test prediction; classifiers and settings are never fitted using 2026 outcomes. Any history smoothing parameter must be fixed or learned without future outcomes.
- Aircraft rotation, final tail assignments and their derived incoming-flight features remain excluded from the agreed project.

This stage spans notebooks 1 and 2: essential label/calendar variables support initial EDA; weather/history/traffic engineering is developed and audited in notebook 2, followed by its own univariate/bivariate checks.

## 6. Variable transformation

- Encode categories with a documented unknown-category policy; do not treat airline or airport codes as ordered numeric distances.
- Represent time cyclically or with declared bands so midnight is handled sensibly. Keep weekday categorical and represent month with a fixed 12-month sine/cosine cycle; retain ordinary month for audit and sampling.
- Scale numerical predictors for Logistic Regression, KNN and SVM. Random Forest uses the same predictor information without requiring numerical scaling.
- Consider log1p for suitable nonnegative, skewed predictors only when justified. Do not apply it blindly to signed delays or treat transformation as proof of predictive improvement.
- Include original-versus-log1p illustrations with matching samples, distributions/box plots and skewness summaries. Distance/traffic illustrations do not add model variables automatically. When demonstrating dtype conversion, assign the conversion and verify the resulting dtype.
- Fit all learned encoders, scalers, imputers and clipping limits on the appropriate training partition. Save them in reproducible model pipelines. Do not independently fit transforms on validation/test data.

Inspect transformations during preparation; finalize them in the classification notebook so they remain part of each saved estimator.

## Additional checks and outputs

- Record population/class balance, duplicates, reference coverage, feature coverage, join row counts and time-cutoff violations. Investigate leakage and keep identical A/B query IDs/labels.
- Preserve natural validation/test prevalence. Any class weighting or training resampling belongs to model development, not a balanced replacement evaluation dataset.
- Save a treatment ledger: variable/rule, rationale, fitted period, affected rows, before/after summary and output version. Also save exclusions, feature definitions and a short EDA interpretation with limitations.
- Keep plotting samples distinct from the shared modelling sample. Save charts and small aggregate tables with period, denominator and units for Tableau context. EDA charts complement the four ML-results dashboards; they do not substitute for model evidence.
- Do not fabricate results while creating notebook structure. Finish cleaning/feature checks before producing measured EDA findings, then train and export real predictions under the existing Tableau contract.

The final handoff comprises documented clean records, chronological roles, audited A/B features, treatment decisions, EDA figures/tables and reproducible preprocessing specifications. Model evaluation remains a separate stage.
