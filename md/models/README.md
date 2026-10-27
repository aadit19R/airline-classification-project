# Saved models

The current notebook bundle is `reports/simple_reproduction/models.joblib`, written by notebook 03 and loaded by notebook 04. It contains eight fitted scikit-learn A/B pipelines, input columns and chosen thresholds.

`models/final_flight_delay_models.joblib` is the original final-study reference used by the optional verifier. The other two bundles in this directory are legacy saved models; they are preserved but are not inputs to the current notebooks.

For SVM use `decision_function`; for the other classifiers use positive-class `predict_proba`. Compare the resulting score with the bundle's saved threshold. Calling `Pipeline.predict()` uses a default threshold and may produce different warnings.

Use each pipeline's fitted preprocessing. Scores are not verified calibrated probabilities.
