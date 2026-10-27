# How EDA and ML fit this project

The objectives require model evidence because they assess prediction. This does not remove exploratory data analysis or make descriptive charts inappropriate. The project follows extraction, cleaning, exploratory analysis, feature engineering, model evaluation and visual communication.

## EDA establishes the context

Use EDA to audit missing values, joins and exclusions, understand the target balance, inspect departure patterns and choose features worth testing. Airport/time/weather charts belong in the notebook and provide context in the report. Keep outcome-driven feature choices within development periods; avoid using final evaluation findings to keep redesigning the model.

The agreed sequence includes univariate analysis, bivariate analysis, missing value treatment, outlier treatment, variable creation and variable transformation. See [the four-notebook guide](notebooks/simple/README.md) for examples and outputs. Source checks and essential parsing come first; valid extremes remain, and learned treatments are fitted on training data through the modelling pipelines.

## ML evaluates what can be predicted

| Supporting analysis | What the ML objective additionally requires |
|---|---|
| Compare observed delay rates in rain and dry weather | Fit comparable models with and without weather, then measure predictive changes on the same later-period flights. |
| Count delayed departures by airport | Compare the selected model's missed-delay and false-alert rates by airport, including denominator/support counts. |
| Show evening flights have more historical delays | Evaluate whether model scores concentrate actual delayed flights within a predeclared review quota better than historical-risk and random baselines. |
| Show which features correlate with delay | Compare fitted classifiers and feature sets using held-out scores; a correlation alone does not establish improved prediction. |

EDA describes records already observed. ML uses earlier examples to learn a rule, then is checked against flight outcomes excluded from fitting/tuning. Both appear in the project, with different claims supported by each.

## Modest or negative outcomes can still answer the objectives

The objectives ask to assess and compare, rather than promise that every model or additional feature will succeed. If Random Forest is strongest, that is a measured comparison. If it offers little gain over Logistic Regression, that is also useful evidence. If weather does not improve performance, report the result and relevant limitations rather than claiming it helped because its EDA chart looked convincing.

Similarly, weak ranking lift answers the prioritization objective negatively: the tested model has not demonstrated a useful review list in that evaluation. Such a result limits practical claims; it does not automatically invalidate the cleaning, experiment or analysis. A project grade cannot be guaranteed from model accuracy alone.

The important distinction is whether the work is reproducible and the claims follow the evidence: fair model/feature comparisons, a meaningful baseline, information available at the prediction time, honest final-period evaluation and correct metric denominators.

## Scope after adding Random Forest

There are four classifiers: Logistic Regression, KNN, SVM and Random Forest, with four supporting objectives. A/B comparisons contain eight candidate/feature combinations. Aircraft rotation (previously C) is excluded because assignment-at-cutoff evidence is unavailable. O2 remains a complete controlled A-to-B comparison; no replacement or fifth objective is needed.

The remaining weather/outcome timing assumptions apply to the historical simulation as a whole. Declaring them limits the live-use claim; it does not establish actual historical delivery times. Archived schedule revisions are not reconstructed either. Report the measured offline comparisons with these limits.
