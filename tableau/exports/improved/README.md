# Q2-2026 original-versus-improved Tableau results

Use this directory for the current four-dashboard workbook. These six CSVs contain actual results on the same 10,000 April–June 2026 flights. The original Q1 exports remain unchanged one directory above.

- `model_metrics.csv`: cards, bars and confusion counts. Main view: filter `threshold_policy = F1`. F0.5 rows are secondary policy comparisons for improved models only.
- `flight_predictions.csv`: one row per candidate and flight for the primary F1 rule. Filter to one candidate before counting/ranking flights. `risk_rank` is fixed within the complete cohort.
- `feature_comparison.csv`: paired A/B changes within each model version. These are different from original-versus-improved changes.
- `subgroup_metrics.csv`: supported error rates and denominators. Blank values mean insufficient support.
- `priority_curve.csv`: fixed 5%, 10%, 20% review quotas and matching ranking counts.
- `pr_curve.csv`: ranking curves; SVM margins are scores, not delay probabilities.

`model_version` is original, improved or baseline. `candidate_id` combines version/model/set. Use it with `cohort_id` and the appropriate curve, capacity or subgroup keys. Metrics also require `threshold_policy`; physically joining aggregate metrics to predictions can multiply counts.

`is_validation_selected` marks the candidate selected using 2025, regardless of the highest Q2 bar. The first test-access timestamp, frozen selection checksum, cohort and file checksums are in export_manifest.json. Use only this same run when packaging the workbook.

Classification thresholds do not change when a review-capacity control changes. Global metrics do not recalculate when a flight list is filtered by airport; use the precomputed subgroup table for those rates. Original models were reused without refitting, so their new fit_seconds values are blank; scoring times are measured here.

The subgroup/prediction tables use the F1 rule. F0.5 comparisons must use a separate metric-table view rather than driving those tables through a global policy filter. added_feature_coverage means a qualifying weather report plus origin history; it does not mean every weather measurement is nonmissing.

[Measured findings](../../../reports/improvements/RESULTS.md) · [Workbook plan](../../WORKBOOK_PLAN.md). The `.twbx` workbook itself has not been built yet.

See [the versioned export contract](../../improvement_export_contract.json) for table grains and required policy/version keys.
