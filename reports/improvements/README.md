# Model-improvement experiment

Notebook 03, its model bundle and its Q1-2026 exports are preserved as the original experiment.

1. Run `notebooks/04_model_improvements.ipynb` for 2025 development only. It compares bounded parameter/scaling searches, cumulative feature packages, and nested shared 20k/40k training samples over three chronological validation blocks. It freezes choices before the new test.
2. Run `notebooks/05_fresh_test_evaluation.ipynb` to compare the original and improved pipelines on identical April–June 2026 flights. It performs no fitting or tuning.
3. Use `tableau/exports/improved/` for the new version. Keep `model_version`, `candidate_id`, evaluation period and threshold policy visible in Tableau.

F1 remains the primary warning-threshold objective. F0.5 is a secondary policy comparison. Model selection uses rolling-validation average precision, not the final test. A single common feature package and training size keep the all-model comparison consistent; individual models may benefit differently.

The original seven-day outcome lag, two-hour prediction horizon, weather freshness limits, natural evaluation prevalence and aircraft exclusion stay in force. No new external predictor feed is required. Q2 acquisition is stored in the new quarter-specific provenance files; original preparation handoffs are retained unchanged.

Files in this directory record the declared protocol, baseline checksums, frozen selection, execution summaries, approved Q2 weather conflicts and final test results. No intermediate split datasets or transformed matrices are exported. Once fresh-test access is recorded, further development requires a new explicitly separate experiment.
