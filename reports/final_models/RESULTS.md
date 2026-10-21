# Final compact-model results · July 2026

The 2025 validation procedure selected **Logistic Regression**, using `combined`. Its mean development F1 was **0.483**. This choice was frozen before July outcomes were loaded; no test-driven model, feature, setting or threshold changes followed.

On the same **10,000 July flights**, 3,215 departed at least 15 minutes late (32.15%). The selected model achieved **F1 0.540**, AP 0.566, precision 0.381, and recall 0.924. It warned on 7,798 flights: 2,971 correct delay warnings and 4,827 false warnings; 244 actual delays were missed.

The strongest practical finding is risk prioritization: the selected model's top 10% review list contained **688 delays among 1,000 flights (68.8% precision; 2.14× random lift)**. The simple historical ranking found 442 delays in the same-size list. The full warning threshold flagged 7,798 flights and produced 4,827 false alerts, so its broad warning rate is a material limitation.

The compact additions raised Logistic Regression's July F1 by **0.015** against its equally retuned 66-feature reference. Its AP change was only +0.0006, with an interval spanning zero. KNN improved modestly on both metrics; Random Forest's pressure additions reduced AP by about 0.010, and SVM retained 66 features. Improvements were mixed.

## All four selected B models

| Model | Compact package | F1 | AP | Precision | Recall | Accuracy | False warnings |
|---|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | combined | 0.540 | 0.566 | 0.381 | 0.924 | 0.493 | 4,827 |
| KNN | combined | 0.545 | 0.532 | 0.401 | 0.853 | 0.542 | 4,104 |
| RBF SVM | current_66 | 0.534 | 0.513 | 0.435 | 0.692 | 0.612 | 2,893 |
| Random Forest | pressure_wind | 0.543 | 0.540 | 0.393 | 0.880 | 0.524 | 4,376 |

These are confirmations of fixed candidates. The largest July score does not redefine the validation-selected model. F1 is not accuracy, and 0.50 is not a universal coin-flip F1 threshold.

## Baselines

| Reference | F1 | AP | Accuracy |
|---|---:|---:|---:|
| always_on_time | 0.000 | 0.322 | 0.678 |
| always_delayed | 0.487 | 0.322 | 0.322 |
| fair_coin_seed42 | 0.398 | 0.325 | 0.509 |
| historical_risk_ranking | 0.481 | 0.396 | 0.373 |

At this prevalence, a fair coin's expected F1 is approximately 0.391; the table includes one reproducible seed-42 realization. Always-delayed F1 is 0.487. Always-on-time has F1 zero despite its higher-than-coin accuracy.

## Did the compact features help beyond retuning?

| Model | B minus retuned66 F1 | 95% day-bootstrap interval | AP change |
|---|---:|---|---:|
| Logistic Regression | +0.015 | [+0.011, +0.019] | +0.001 |
| KNN | +0.006 | [+0.002, +0.010] | +0.008 |
| RBF SVM | +0.000 | [+0.000, +0.000] | +0.000 |
| Random Forest | +0.001 | [-0.002, +0.004] | -0.010 |

An interval crossing zero does not establish a clear improvement in this month. A zero change can mean that validation retained the current 66 features. These are paired comparisons on identical flights, not differences between development and test periods.

## O2 · Matched-settings schedule A versus richer B

| Model | B minus A F1 | 95% day-bootstrap interval | AP change |
|---|---:|---|---:|
| Logistic Regression | +0.038 | [+0.033, +0.044] | +0.062 |
| KNN | +0.032 | [+0.027, +0.037] | +0.067 |
| RBF SVM | +0.048 | [+0.036, +0.060] | +0.031 |
| Random Forest | +0.002 | [-0.003, +0.008] | +0.048 |

A and B use the same classifier settings, fitting IDs and time harmonics. Their thresholds are independently calibrated using the same October procedure. This isolates richer information at matched settings; it does not claim an independently optimized A model. The combined addition cannot identify the individual causal contribution of weather, history or traffic.

## O3 · Supported prediction strengths and errors

| Dimension | Group | Flights | Precision | Recall | False-positive rate |
|---|---|---:|---:|---:|---:|
| origin | ATL | 3,180 | 0.413 | 0.911 | 0.647 |
| origin | DFW | 3,261 | 0.416 | 0.939 | 0.723 |
| origin | ORD | 3,559 | 0.322 | 0.921 | 0.755 |
| departure_time_band | 00–05 overnight | 101 | unsupported | unsupported | unsupported |
| departure_time_band | 06–11 morning | 3,663 | 0.269 | 0.685 | 0.424 |
| departure_time_band | 12–17 afternoon | 3,432 | 0.358 | 0.980 | 0.918 |
| departure_time_band | 18–23 evening | 2,804 | 0.481 | 1.000 | 0.997 |

False-positive rate is FP/(FP+TN), whereas the false-warning share among warnings is FP/(TP+FP). They have different denominators. Airline detail and support flags are in notebook 07 and subgroup_metrics.csv; unsupported values remain blank. These are descriptive error patterns in July, not causal airline/airport effects.

## O4 · Risk prioritization at fixed review quotas

| Ranking | Review quota | Reviewed | Delays captured | Review precision | Delay recall | Lift |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 5% | 500 | 359 | 0.718 | 0.112 | 2.23 |
| Logistic Regression | 10% | 1,000 | 688 | 0.688 | 0.214 | 2.14 |
| Logistic Regression | 20% | 2,000 | 1,222 | 0.611 | 0.380 | 1.90 |
| historical_risk_ranking | 5% | 500 | 221 | 0.442 | 0.069 | 1.37 |
| historical_risk_ranking | 10% | 1,000 | 442 | 0.442 | 0.137 | 1.37 |
| historical_risk_ranking | 20% | 2,000 | 918 | 0.459 | 0.286 | 1.43 |

Quotas and tie-breaking were fixed before July. This retrospective ranking analysis does not establish money saved, delays prevented or an operational real-time policy.

## Interpretation and limits

The selected model's false-warning share among warnings is 61.9%. Interpret that alongside recall and the review-quota results; a modest F1 does not justify claiming a reliable operational warning system.

The final classifiers fit a shared 40,000-flight sample covering all twelve months of 2025. October warning thresholds came from 40,000-flight January–September fits. Threshold transfer after refitting, seasonal changes and changes in airport/airline mix can alter warning rates; these results do not isolate their separate effects. No threshold was repaired using July outcomes.

The July population contained 89,701 schedules and 87,298 eligible operated/known-label flights. Cancelled and unknown-label rows follow the existing exclusion policy. Weather-missing sampled flights retained: 0.

July is one summer month at ATL/ORD/DFW, not a full-quarter or year-round estimate. The 300 paired day-block bootstrap intervals are conditional on the fitted models and only 31 observed date blocks. Reused development periods introduce selection optimism; only July is the new confirmation period.

The study remains a historical simulation: weather delivery uses an assumed 20-minute buffer and 180-minute age limit; outcome histories use 28-day windows ending seven days before each cutoff. Access to timely outcomes and archived schedule snapshots is assumed, not verified. Aircraft rotation and current-flight actual outcomes are excluded.

## Reviewable files

- notebooks/06_final_model_selection.ipynb: feature screen, bounded tuning, larger-fit check and final refits.
- notebooks/07_final_july_evaluation.ipynb: frozen confirmation, comparisons, error analysis and ranking plots.
- reports/final_models/development_audit.json: score, threshold, cohort and baseline-integrity checks.
- tableau/exports/final_july/: six reconciled Tableau sources and the exact-run manifest.
- models/final_flight_delay_models.joblib: final fitted pipelines.

The Tableau workbook is the next deliverable; these exports are its frozen data snapshot.
