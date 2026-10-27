# Measured model-improvement results

Completed 19 October 2026. All findings below come from executed notebooks and saved predictions. Original notebook 03, both prepared datasets, the shared preparation manifest, original fitted models and original dashboard exports retain their recorded checksums.

## What changed

- Expanded regularization search for Logistic Regression; larger KNN neighbourhoods and alternative distances; joint SVM C/gamma tuning; RF feature subsampling, depth, leaf size and class-weight comparisons.
- Compared the original preprocessing with continuous-only scaling plus unscaled binary/missing flags. The alternative is a complete preprocessing recipe, so its effect cannot be attributed solely to one scaler setting.
- Tested second time harmonics and airport × clock interactions, lagged flight-number history and richer origin-weather summaries. Compared feature packages cumulatively with model settings fixed.
- Compared nested shared samples of 20,000 and 40,000 flights. No eligible source rows were deleted and no intermediate split or transformed datasets were exported.
- Used three 2025 chronological validation blocks for development. Reserved newly acquired April–June 2026 for a frozen test, since Q1 had already been evaluated.

The common selected package is **time_flight_weather** and the common final fitting sample is **40,000 flights** across all twelve months of 2025. The eight final A/B pipelines use that same cohort. A contains schedule features; when the time package is retained, those time features appear in both A and B. B adds origin weather, earlier history and scheduled traffic, with any retained extensions.

The retained inputs contain 26 raw schedule predictors in A and 66 predictors in B before imputation/encoding. New weather inputs describe dew-point spread, wind-direction components, freezing/wet conditions and recent report-level weather summaries.

The development phase performed 252 chronological CV fits across 84 distinct configurations, followed by eight final refits. The search is staged and bounded, not exhaustive.

## Fresh test: compare models on identical flights

The raw Q2 scope contains 260,336 schedules, with 254,333 eligible labelled departures. The fixed sample contains 10,000 flights and 2,557 delays (25.57%). Month × label sampling preserves natural proportions.

| Model · B | Original AP | Improved AP | AP change | Original F1 | Improved F1 | Improved precision | Improved recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.4121 | 0.4570 | +0.0449 | 0.4443 | 0.4396 | 28.7% | 93.7% |
| KNN | 0.3910 | 0.4454 | +0.0544 | 0.4421 | 0.4634 | 32.5% | 80.8% |
| RBF SVM | 0.4098 | 0.4422 | +0.0324 | 0.4171 | 0.4500 | 32.8% | 71.9% |
| Random Forest | 0.4480 | 0.4719 | +0.0239 | 0.4591 | 0.4712 | 33.1% | 81.5% |

These are paired Q2 comparisons. Do not subtract the old Q1 score from a new Q2 score and call that difference a model improvement. AP measures ranking; F1/precision/recall also depend on the separately frozen warning thresholds.

The validation-selected candidate is **Logistic Regression / B**. It remains the selected candidate regardless of which fresh-test bar is highest. Its fresh-test AP is 0.4570, F1 0.4396, precision 28.7% and recall 93.7%. It produces 2,397 true delay warnings, 5,952 false warnings, 160 missed delays and 1,491 correct on-time predictions.

Random Forest has the highest Q2 AP and primary-policy F1 among the four improved B models. Logistic Regression was selected before this test because its mean rolling-validation AP was highest. The frozen selection is not changed using Q2 outcomes.

LR's primary F1 fell from 0.4443 to 0.4396 and false warnings increased from 5,518 to 5,952. Its ranking improvement therefore does not establish a better binary warning policy. The separately predeclared F0.5 alternative reaches F1 0.4664 with 4,377 false warnings, at the cost of 448 missed delays rather than 160. It remains a secondary policy comparison.

## Schedule-only results and simple baselines

| Model · A | Original AP | Improved AP | Original F1 | Improved F1 | Improved warned flights |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.4022 | 0.4163 | 0.4449 | 0.4293 | 8,606 |
| KNN | 0.3641 | 0.3768 | 0.4290 | 0.4443 | 6,622 |
| RBF SVM | 0.3359 | 0.3833 | 0.4081 | 0.4073 | 10,000 |
| Random Forest | 0.4128 | 0.4061 | 0.4510 | 0.4545 | 6,274 |

Schedule-only SVM remains weak: its F1 rule warned on virtually all latest-validation flights. Report its AP and final warning count alongside F1.

- always_on_time: AP 0.2557, F1 0.0000, accuracy 74.4%.
- historical_risk_ranking: AP 0.3440, F1 0.4135, accuracy 43.9%.

## Are the ranking gains clearly separated?

| Model · B | AP change | 95% paired day-bootstrap interval |
|---|---:|---:|
| Logistic Regression | +0.0449 | [+0.0347, +0.0567] |
| KNN | +0.0544 | [+0.0362, +0.0702] |
| RBF SVM | +0.0324 | [+0.0189, +0.0451] |
| Random Forest | +0.0239 | [+0.0127, +0.0339] |

The 300 bootstrap repetitions resample whole flight dates with matched weights for both versions. An interval crossing zero does not clearly separate the AP values on these Q2 dates. These intervals are conditional on the fitted models and do not include all training or future-period uncertainty.

## F1 and F0.5 are different warning policies

F1 remains primary, following the user’s preference for a familiar balanced project metric. F0.5 is shown as a secondary precision-prioritizing policy; it changes warnings, not ranking or AP. Neither policy is readjusted using Q2 outcomes.

| Selected model policy | Precision | Recall | F1 | False warnings | Missed delays |
|---|---:|---:|---:|---:|---:|
| F1 | 28.7% | 93.7% | 0.4396 | 5,952 | 160 |
| F0.5 | 32.5% | 82.5% | 0.4664 | 4,377 | 448 |

## Evidence for the four supporting objectives

- **O1, model suitability:** selection uses mean 2025 rolling-validation AP and confirmation uses Q2 scores. Runtime, precision, recall and confusion counts remain visible for all four models.
- **O2, value of additional information:** use the version-specific A/B paired changes in feature_comparison.csv. Notebook 04 additionally compares cumulative feature families. These do not establish causal effects or the benefit of every individual weather field.
- **O3, failure patterns:** the subgroup table records supported airport, airline and time-band recall, precision and false-positive rates. Unsupported rates remain blank. These are descriptive errors in the sampled future quarter.
- **O4, prioritization:** the selected model’s top 10% reviews 1,000 flights, captures 562 delays, has 56.2% review precision and captures 22.0% of all delays, at 2.20× prevalence. Compare with the historical-risk baseline at the same quota.

At the same 10% Q2 review budget, the original LR captured 474 delays; the improved LR captured 562. This is an 88-delay gain on matched flights, separate from its less successful primary warning threshold.

At 40,000 fitting rows, mean B fitting times in validation were about 0.32 seconds for LR, 7.07 seconds for RF and 191.27 seconds for SVM. KNN fits quickly but took 10.35 seconds to score the final 10,000 B test flights. These timings describe this machine/run, not universal speed benchmarks.

## Data quality, remaining limits and deferred work

Sampling limits classifier fitting only. Weather reports, scheduled traffic and historical counts use the full appropriate source population. The earlier feature handoff omits flight number and minute-of-day; notebook 04 explicitly retrieves those two fields from this project’s clean table. Airport interaction flags are converted to numeric before multiplying so Arrow-backed columns work correctly.

Three flight archives and three padded origin-weather responses were acquired directly from BTS/IEM. All source dates are present. Both versions at ORD 3 April 06:00 UTC and 10 June 20:55 UTC were excluded with user approval. The source audit found 95 schedules (0.04%) without a fresh report. The final 10,000-flight sample contains 4 such flights; they were retained with missingness-aware preprocessing.

A/B/history reconstruction matched the reviewed handoff on 200 audit flights. An independent output audit recomputed every exported score metric, confusion table, PR curve, subgroup count and capacity quota. The final notebooks contain no errors or stderr output.

Aircraft rotation remains excluded; the seven-day outcome lag is unchanged. Archived schedule availability and weather/outcome receipt timing remain assumptions. These sources do not show that all features would have reached a live prediction system by the cutoff. No accuracy target, operational savings or prevention of delays is claimed.

Pressure, relative traffic normalization, destination forecasts, a real-time disruption feed, further years and more encoding methods remain possible future experiments. They were not silently introduced, and none is required to report this completed comparison. The fresh test is now exposed and cannot be reused as untouched evidence for further tuning.

## Result files

- Development: reports/improvements/development_summary.json; frozen selection: reports/improvements/selection.json.
- Completed Q2 confirmation and independent checks: reports/improvements/test_results.json and execution_audit.json. The current [four-notebook workflow](../../notebooks/simple/README.md) reproduces the final July experiment.
- Fitted pipelines: models/improved_flight_delay_models.joblib.
- Six versioned evaluation tables: reports/evaluation_exports/improved/.
- Use F1 as the main threshold policy and keep model_version/candidate_id/cohort keys explicit.
