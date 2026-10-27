# Notebook results

The [four notebooks](../../notebooks/simple/README.md) reproduce the completed July 2026 evaluation using 40,000 training flights, 10,000 test flights and the original model settings. Code cells have at most ten lines. Classifier definitions and fitting remain visible in notebook 03.

```sh
python3 scripts/run_simple_notebooks.py
python3 scripts/verify_simple_reproduction.py
```

Outputs in [reports/simple_reproduction/](../../../reports/simple_reproduction/):

- `features.parquet`: unscaled training and test inputs.
- `models.joblib`: eight A/B pipelines.
- `metrics.json`: accuracy, precision, recall, F1 and average precision for models and baselines.
- `scores.npz`: individual model and baseline scores.
- `cleaning_checks.json`: source-cleaning summary.
- `execution.json`: notebook execution times, outputs and errors.
- `verification.json`: independent comparison with the original features, preprocessing, predictions, metrics, ranks and quotas.
- `protected_checksums.json`: reference hashes for source data and published artifacts.

Subgroup, review-budget and bootstrap sections are omitted from the notebook walkthrough. Their published results remain in the [evaluation exports](../evaluation_exports/README.md) and report. The notebooks reuse the previously chosen thresholds; the optional verifier independently checks their derivation from the recorded calibration scores.

The checks compare 3,900,000 selected feature values, eight pipelines, four baselines, 120,000 ranks and 36 review quotas. The feature table uses one `split`: 40,000 all-2025 training flights and 10,000 July-2026 test flights.

[Label migration evidence](../../../reports/study_labels.json) records the earlier metadata changes.
[Refactor evidence](../../../reports/notebook_refactor.json) records source changes, removed-script hashes and pre-refactor feature fingerprints. Published model bundles and result exports remain the references; the verifier imports none of the removed modules.
