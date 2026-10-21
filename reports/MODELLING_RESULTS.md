# Classification results

Run: departure_20261018T201555701606Z

- Validation selected Logistic Regression / B with AP 0.4255.

- Frozen test performance: AP 0.4055, precision 32.7%, recall 76.1%, F1 0.4575, accuracy 54.3%, ROC-AUC 0.6684.

- On 10,000 test flights: 1,929 correctly warned delays, 3,967 false warnings, 607 missed delays and 3,497 correct on-time predictions.

- The largest descriptive test AP is 0.4055 for Logistic Regression / B; the registered validation choice remains unchanged.

- A-to-B test AP changes: Logistic Regression +0.0443; KNN +0.0530; RBF SVM +0.0742; Random Forest +0.0267.

- Lowest supported selected-model recall: departure_time_band=06–11 morning, 47.1% across 3,552 flights. This is descriptive, not an intrinsic quality ranking.

- The top 10% review list captures 471 delays (18.6% of actual delays), with 47.1% precision, lift 1.86 and 529 non-delayed reviewed flights.

- The same historical-risk review quota captures 12.3% of delays with 31.1% precision; random expected recall is 10.0%.

- A/B measures the combined additional feature package, not separate causal contributions from weather, history or traffic.

- Validation metrics used for selection are optimistic development evidence. Test results cover a proportional sample of Q1 2026 at three hubs; repeated weather reports and airport-days limit independence.

- Thresholds are retained after full-2025 refitting, without test recalibration. Score shifts can affect F1 even when ranking AP changes differently.

- Historical delivery and schedule availability are assumed, not verified live. The target excludes cancelled flights and aircraft rotation is outside scope.

- The six Tableau tables and one fitted-model bundle are saved. Creating and checking the Tableau workbook remains a separate stage.

See the executed notebook, reports/model_selection.json and tableau/exports/export_manifest.json for settings, thresholds, cohort fingerprints and source lineage.
