# July final-model Tableau sources

These six CSVs are written by notebook 07 only after notebook 06 freezes its choices. Original Q1 sources and the `improved/` Q2 folder remain separate. Read `export_manifest.json` for the exact run, selected classifier, period and file hashes.

| File | Grain / purpose |
|---|---|
| model_metrics.csv | One candidate and threshold policy; O1 model cards, confusion counts and performance bars |
| feature_comparison.csv | One model's matched-settings A/B comparison; O2 paired feature-value charts |
| subgroup_metrics.csv | One candidate and supported subgroup; O3 error-rate charts with denominators |
| priority_curve.csv | One candidate and predeclared review quota; O4 capture, precision, lift and false alerts |
| pr_curve.csv | One candidate and precision–recall point; O1 ranking performance |
| flight_predictions.csv | One flight per candidate; error detail and reproducible ranked review lists |

Default filters: `model_version = final`, `feature_set = B`, `threshold_policy = F1`. The chosen classifier is registered in the manifest; the highest July score must not redefine it. The retuned current feature reference is `feature_set = reference66`. A contains schedule information with the same time harmonics, classifier settings and fitting IDs as B. A was not independently optimized in this round.

Keep aggregate sources separate from row-level predictions. Never sum a global metric duplicated across flight rows. Restrict the prediction list to exactly one candidate before counting flights. Global scorecards do not automatically recalculate when filtering a flight list.

The study period is `test_july_2026`, 10,000 natural-prevalence flights from ATL/ORD/DFW, departure >=15 minutes late, two-hour cutoff. The workbook must say **one-month summer confirmation**, not Q3 or year-round performance. The risk scores are uncalibrated. Weather/outcome availability and archived schedules remain historical-simulation assumptions.

For O4, `risk_rank` and `rank_cohort_size` are already calculated using an outcome-independent hash to break ties. Use K = ceiling(quota x cohort size), with quotas 5%, 10%, 20%. Reconcile review-list delays/false alerts against priority_curve.csv. Historical-risk ranking and random expectations are comparison baselines, not trained model choices.

For O3, unsupported subgroup rates are blank; do not replace them with zero. The original support rules require 200 group flights and 30 relevant actual/predicted positives/negatives for the rate being shown.

Paired F1/AP differences and 95% day-bootstrap limits are included on B rows in model_metrics.csv. `vs66_` compares compact B with the equally retuned 66-feature reference. `vsA_` compares richer B with the matched-settings schedule control. These intervals are conditional on the fitted models and only 31 observed day blocks.

Coverage: `added_feature_coverage` means a matched weather report and nonzero origin-history count. It does not mean every weather field is known. Label its card “Report & origin-history match”; use notebook 07’s missingness table for individual measurement coverage.
