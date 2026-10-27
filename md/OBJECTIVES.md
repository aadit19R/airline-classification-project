# Project objectives and evidence

## Main objective

To develop and evaluate, through a historical simulation with a two-hour pre-departure cutoff, a classification framework using Logistic Regression, KNN, SVM and Random Forest, and assess whether its predictions can reliably identify and prioritize flights at risk of departing at least 15 minutes late.

The final conclusion must state what the measured models can support, where they fail, and whether richer inputs improve prediction. Positive results are not assumed.

This is an offline historical study. Weather delivery buffers and access to earlier flight outcomes are stated availability assumptions; the records do not verify a live feed. Aircraft rotation is excluded from the agreed objectives and model comparisons because the historical assignment time is unavailable. Its raw source fields remain stored, but no objective depends on using them.

## Supporting objectives

### O1: Select and verify a suitable classifier

To select the most suitable classifier through chronological validation and verify its delayed-flight detection performance and computational practicality on the final evaluation period.

- Required evidence: The predeclared validation metric for selection (AP in the original rounds; mean delayed-class F1 with AP tie-break in the final compact-feature round); delayed-class F1, precision, recall, average precision and runtime for confirmation.
- Dashboard page: `01_model_performance`. Model comparison bars, precision-recall curves, confusion matrix and selected-model cards.
- Why ML is necessary: Requires fitted model scores, frozen predictions and observed outcomes on later-period flights.

### O2: Measure the value of additional information

To assess whether adding origin weather, historical departure-delay patterns and scheduled airport traffic improves departure-delay classification compared with schedule features alone, using the same later-period flights and stated input-availability assumptions.

- Required evidence: Paired changes in average precision and F1 from A (schedule) to B (schedule plus weather/history/traffic), using identical flights.
- Dashboard page: `02_feature_value`. Feature-set comparison matrix, incremental-gain bars and feature-coverage cards.
- Why ML is necessary: Requires controlled model retraining and evaluation; raw correlations do not establish predictive improvement.

### O3: Identify prediction strengths and failure patterns

To evaluate the reliability of the selected model across airports, airlines and departure time bands by identifying missed delays and false alerts in sufficiently supported subgroups.

- Required evidence: Subgroup recall, false-negative rate, false-positive rate and precision, with denominator counts and support flags.
- Dashboard page: `03_prediction_reliability`. Error-rate heatmaps, missed-delay versus false-alert charts and supported-group detail tables.
- Why ML is necessary: Compares model predictions with actual outcomes; high observed delay frequency alone does not identify model failure.

### O4: Assess the usefulness of risk prioritization

To quantify how effectively the selected model concentrates actual delayed flights within predeclared review budgets of 5%, 10% and 20% of scored flights, compared with random prioritization and a simple historical-risk baseline.

- Required evidence: Delayed-flight recall, precision, lift and false alerts within the highest-ranked review quota.
- Dashboard page: `04_risk_prioritization`. Cumulative-gains curve, capacity selector, delay-capture/lift cards and ranked flight review list.
- Why ML is necessary: Requires model scores and a retrospective ranking evaluation; it cannot be answered by counting historical delays.

## Rules for answering the objectives

- O1 selects using validation and reports the frozen choice on the final period. Browsing the test comparison must not change model selection.
- O2 compares exactly the same flight IDs and target labels across feature sets. Retain flights with missing richer inputs using training-only imputation and missingness flags. A-to-B measures the combined weather/history/traffic addition; it cannot identify the separate contribution of each without additional ablations.
- O3 concerns error rates, with denominators. An airline's raw delay count is background information, not an answer to this objective.
- O4 is a retrospective ranking/capacity exercise. It does not establish actual money saved, delays prevented, or a deployable real-time alert policy. Use model scores, not unsupported probability labels.
- All four objectives are answerable using A/B without aircraft rotation. O2 can find improvement, no meaningful change or worse performance; none of these results is assumed in advance.
- Results describe the historical simulation under its input-availability assumptions. They do not verify actual two-hour live warning performance, or prove the historical schedule snapshot matched what was known at each cutoff.
- EDA supports cleaning, scope and feature choices and can appear as context in dashboard. The four objectives additionally require fitted models' validation/final-period outputs; descriptive charts remain part of the project. See [the explanation](EDA_AND_ML_EXPLAINED.md).

## Final compact-feature round

The final experiment uses F1-first selection on 2025 data and a frozen July-2026 confirmation. The four notebooks in `notebooks/simple/` reproduce its final fitting and evaluation. O2 uses matched-settings A/B controls on identical flights; the retuned 66-feature reference separately measures gains from compact additions. July covers one summer month at three hubs. See [final findings](reports/final_models/RESULTS.md) and the [July export guide](reports/evaluation_exports/README.md).
