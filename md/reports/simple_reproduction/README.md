# Verified four-notebook workflow

The [four notebooks](../../notebooks/simple/README.md) run independently, with **190 visible code lines, 45 code cells and nine figures**. Every code cell has at most seven lines. Shared modules handle repeated work; classifier definitions and `.fit()` calls remain visible.

All four notebooks have been executed in fresh kernels, including the eight A/B model fits. This reproduces the completed July 2026 evaluation using the frozen settings, 40,000 fitting IDs, 10,000 July flights, feature order, thresholds and validation-selected Logistic Regression / B. July is already exposed.

[verification.json](verification.json) records the optional independent checks:

- All 78 selected features on all 50,000 rows match, including missing values and time cutoffs.
- Training-only preprocessing, chronological cohorts, fitting IDs and labels match.
- All eight principal models and four baselines reproduce saved confusion counts and metrics.
- 120,000 ranks, 36 review quotas, supported subgroup rates and dashboard summaries reconcile.
- Current protected inputs/results retain their hashes.

The old notebooks, duplicate versions, archive bookkeeping and obsolete audits have been removed. Neither notebook execution nor the optional checker needs them. Frozen machine-readable development and selection records remain available because the current workflow uses them.

## Run

From the project root, run `python3 scripts/run_simple_notebooks.py`. To repeat the optional result checks, run `python3 scripts/verify_simple_reproduction.py`.

Generated Parquet, joblib and score-array files stay local and are recreated by the notebook run. Git includes the executed notebooks, source code and JSON results/verification records.

## Included outputs

- [execution.json](execution.json): fresh-kernel timestamps, runtime, cell and figure counts.
- [verification.json](verification.json): current result checks and environment versions.
- [protected_checksums.json](protected_checksums.json): hashes of current raw inputs, data, model/export files and supporting code; no retired files.
- [features.parquet](features.parquet): one additional handoff containing the 40,000 fitting and 10,000 July rows.
- [models.joblib](models.joblib): eight refitted reproduction pipelines.
- [metrics.json](metrics.json), [subgroups.json](subgroups.json), [priority.json](priority.json) and [scores.npz](scores.npz): reproduced results.

The preparation manifest's pre-existing difference from the selection-time hash remains documented in verification. Both processed datasets still match the selection hashes; feature values and predictions are checked directly. Tableau, frozen model bundles, original result exports and the published report are preserved.
