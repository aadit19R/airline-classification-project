# Evaluation exports

Model results are stored in `reports/evaluation_exports/`, independently of any dashboard software:

- `final_july/`: final July 2026 confirmation on 10,000 flights. Used by the HTML dashboard, report and independent notebook checks.
- `improved/`: earlier April-June 2026 comparison on 10,000 flights. Keep this cohort separate from July.

Each folder contains six CSVs and an `export_manifest.json` identifying its run, cohort, selected model and exact file hashes.

| File | Grain and purpose |
|---|---|
| `model_metrics.csv` | One candidate and threshold policy; performance and confusion counts |
| `feature_comparison.csv` | Paired A/B changes for each model |
| `subgroup_metrics.csv` | One candidate and supported subgroup; error rates and denominators |
| `priority_curve.csv` | One candidate and review quota; captured delays and false alerts |
| `pr_curve.csv` | One candidate and precision-recall point |
| `flight_predictions.csv` | One flight per candidate; actual outcome, prediction, score and fixed risk rank |

July defaults: `model_version=final`, `feature_set=B`, `threshold_policy=F1`. Logistic Regression B was selected before July; the highest July score does not change that choice. `reference66` is the equally retuned 66-feature comparison.

Filter predictions to one candidate before counting flights. Keep aggregate metrics separate from row-level predictions. Filtering a flight list does not recalculate global metrics or risk ranks. Blank subgroup rates mean insufficient support, not zero.

Review quotas are 5%, 10% and 20%, using the saved ranks and outcome-independent tie order. Scores are uncalibrated. July covers one summer month at three hubs under historical input-availability assumptions.

The CSV bytes and original run manifests are preserved. Historical experiment records may retain their original paths and display-metadata names as provenance.

[July files](../../../reports/evaluation_exports/final_july/) · [Q2 files](../../../reports/evaluation_exports/improved/) · [Final findings](../final_models/RESULTS.md) · [Dashboard](../../dashboard/README.md)
