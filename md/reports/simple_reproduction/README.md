# Current notebook outputs

The [four notebooks](../../notebooks/simple/README.md) use 40,000 training flights from 2025 and 10,000 July 2026 test flights.

| File | Created by | Purpose |
|---|---|---|
| `features.parquet` | Notebook 02 | Features and labels passed to training/evaluation |
| `models.joblib` | Notebook 03 | Eight fitted A/B pipelines, columns and thresholds |
| `metrics.json` | Notebook 04 | Accuracy, precision, recall, F1 and average precision |
| `scores.npz` | Notebook 04 | Individual scores for eight models and four baselines |
| `cleaning_checks.json` | Notebook 01 | Active source counts and cleaning summary |
| `execution.json` | Optional notebook runner | Execution times and errors |
| `verification.json` | Optional verifier | Comparison with the compact frozen reference |

```sh
python3 scripts/run_simple_notebooks.py
python3 scripts/verify_simple_reproduction.py
```

The verifier uses `reports/final_models/reference.json`, `reference_scores.npz` and the original final model bundle in `models/final_flight_delay_models.joblib`. It checks the selected feature values, training-only preprocessing, predictions, metrics, risk ranks and fixed review quotas. Older experiment reports and large CSV exports are no longer required.
