# Improved model bundle

`improved_flight_delay_models.joblib` contains eight standard scikit-learn pipelines, frozen A/B columns, F1 and secondary F0.5 thresholds, selection metadata and the frozen selection checksum. The original bundle is preserved.

Use `decision_function` for RBF SVM and positive-class `predict_proba` for the other models, then apply the saved policy threshold with score >= threshold. Calling Pipeline.predict uses the estimator default threshold and does not reproduce these exported warnings. Scores are uncalibrated; do not interpret SVM margins as probabilities.

All final fitting rows come from 2025. The new Q2 test only scores the saved pipelines; no model is fitted or tuned using Q2. [Measured results](../reports/improvements/RESULTS.md).
