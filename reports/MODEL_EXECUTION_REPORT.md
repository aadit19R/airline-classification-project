# Classification execution and integrity checks

Completed 19 October 2026. Run: departure_20261018T201555701606Z.

The notebook completed from a fresh Python 3.13 kernel: 19 executed code cells, 12 inline figures, zero error outputs and zero stderr/warning outputs. All 34 declared parameter trials completed, and eight final A/B pipelines were refitted on the separate full-2025 20,000-flight cohort.

Selection was written at 2026-10-18T20:17:49.804348+00:00. The first test outcome load followed at 2026-10-18T20:18:11.900941+00:00. Parameter settings and thresholds were not changed after test access. The selected candidate is logistic_regression / B.

## Verification

- Independently recomputed all global confusion counts, AP and ROC-AUC from exported row-level predictions.
- Verified frozen thresholds reproduce every classification label; same IDs and labels across all candidates and baselines.
- Recomputed every review quota, capture count, precision and lift from exported ranks.
- Recomputed every precision–recall curve and subgroup confusion count; unsupported rates remain missing.
- Verified declared unique grains, run IDs, file counts/checksums, and deterministic outcome-independent score tie breaking.
- Reloaded the fitted joblib bundle and reproduced saved scores on 20 fixed test IDs chosen independently of outcomes for all eight pipelines.
- Verified original three preparation/exploration notebooks, both Parquet handoffs and the shared preparation manifest retain their original checksums.
- Preserved existing selective missingness indicators in all fitted final preprocessing pipelines.

## Files

- flight_predictions.csv: 150,000 rows.
- model_metrics.csv: 20 rows.
- feature_comparison.csv: 8 rows.
- subgroup_metrics.csv: 380 rows.
- priority_curve.csv: 60 rows.
- pr_curve.csv: 132,279 rows.

One fitted bundle is saved at models/flight_delay_models.joblib. No extra processed datasets or intermediate CSVs were created. Experiment/selection metadata is in reports/model_selection.json and tableau/exports/export_manifest.json. The notebook generates reports/MODELLING_RESULTS.md from actual results.

## Compatibility correction

The initial development run encountered scikit-learn 1.9's deprecation warning for an explicit SVC probability=False argument. It was stopped before a selection record or test access. Removing the redundant argument preserves default decision-score behaviour without fitting probability calibration. The full fresh-kernel run then completed without warnings. No model-search or evaluation policy was changed in response to test results.

## Interpretation boundaries

The validation AP winner remains the registered choice even if another test score is larger. Validation results are selection evidence; test results use refitted models and fixed thresholds. A/B measures the combined weather/history/traffic addition, not separate causal effects. Subgroup support rules do not establish statistical certainty.

This evaluates sampled noncancelled departures at three hubs in Q1 2026 under the documented input-availability assumptions. Timely weather/outcome delivery and archived schedule availability are not verified. Frozen rolling test histories may use earlier 2026 events after the seven-day lag; the classifiers never fit on 2026. Tableau workbook creation remains next.
