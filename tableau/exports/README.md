# Model-output exports

Real outputs from run departure_20261018T201555701606Z. All are UTF-8 CSVs and share the frozen selection record. The selected candidate is logistic_regression / B.

| File | Rows | Purpose |
|---|---:|---|
| [flight_predictions.csv](flight_predictions.csv) | 150,000 | Flight scores, frozen warnings, TP/FP/FN/TN labels and deterministic review ranks |
| [model_metrics.csv](model_metrics.csv) | 20 | Global validation/test metrics, confusion counts and fit/score times |
| [feature_comparison.csv](feature_comparison.csv) | 8 | Paired A/B changes on identical flights |
| [subgroup_metrics.csv](subgroup_metrics.csv) | 380 | Airport, airline and time-band error rates with counts and support flags |
| [priority_curve.csv](priority_curve.csv) | 60 | 5%, 10% and 20% review-quota capture, precision, false alerts and lift |
| [pr_curve.csv](pr_curve.csv) | 132,279 | Precision–recall curves from risk scores |

export_manifest.json records checksums, fitting periods, cohort fingerprints and selected-model metadata. It is not a data table.

Rows contain multiple candidate predictions for each flight. Scope to one model, feature set and split before counting flights or showing a review list. Use exported risk_rank and rank_cohort_size to reproduce quota selections; tied scores use a seed/flight hash independent of outcomes.

Use the separate tables as appropriate data sources. Do not physically join global metric rows onto every flight and sum them. Global cards do not automatically recalculate when a flight list is filtered.

Undefined and unsupported rates are blank CSV fields, not zeros. SVM decision scores and other uncalibrated outputs must be labelled risk scores, not verified delay probabilities. Only the validation-selected candidate is marked is_validation_selected.

Validation predictions use January–September 2025 fits. Final-test predictions use full-2025 sample refits and the unchanged validation-selected thresholds. The study is a historical availability simulation at ATL/ORD/DFW; the target excludes cancellations.

The exported A/B feature coverage measures a valid weather report plus nonempty origin history; individual weather fields may remain missing.

The Tableau workbook itself has not been created. Follow [the workbook plan](../WORKBOOK_PLAN.md) and [field definitions](../export_contract.json).
