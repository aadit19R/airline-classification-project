# Saved flight-delay pipelines

flight_delay_models.joblib contains all eight final full-2025 A/B pipelines, their frozen validation thresholds, score types, feature-set labels, selection metadata and the two baseline rules/objects. Read the exact input columns from selection.protocol.feature_sets. The export manifest records its checksum and package versions.

Use the saved preprocessing inside each pipeline; do not independently refit it for prediction. To reproduce this study's warnings, obtain the positive-class score and compare it with the saved threshold. Pipeline.predict uses the estimator's default cutoff, which can differ from the validation-selected cutoff.

For SVM, use pipeline.decision_function(X). For the other classifier candidates, use pipeline.predict_proba(X)[:, 1], with class 1 as the delayed class. Then calculate predicted = (score >= saved_threshold).astype(int).

Scores are not verified calibrated probabilities. This is a historical simulation, not a verified live data-feed implementation. The raw tail number/rotation features remain excluded.
