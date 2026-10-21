# Preparation update for Q1-2026 testing

Completed 19 October 2026. All three notebooks were executed top-to-bottom from fresh local Python 3.13 kernels. Existing 2025/context raw sources and new Q1-2026 raw sources were used; no prior project's processed tables were used.

## Current timeline and samples

| Role | Period | Eligible query population | Shared sample |
|---|---|---:|---:|
| Initial training / EDA | January–September 2025 | 698,670 | 20,000 |
| Validation / model and threshold selection | October–December 2025 | 239,747 | 5,000 |
| Final refit after choices freeze | January–December 2025 | 938,417 | 20,000 |
| Final test | January–March 2026 | 219,748 | 10,000 |

Sampling is fixed-seed (42), proportional within month × outcome strata, without equalizing classes. A/B and all four classifiers share the IDs. The final-refit membership is a separate full-2025 sample, not the union of earlier samples; it can overlap them and never contains 2026 flights. Refit preprocessing and the chosen classifier from scratch only after validation choices are frozen. Keep the selected threshold; never optimize it on the test set.

All 1,158,165 eligible queries remain in features.parquet. flights_clean.parquet contains 1,339,044 schedules/context rows, including 153,143 late-2024 context schedules. No split-specific datasets were created.

## Applied corrections

- Replace categorical month in A/B with a fixed sine/cosine cycle. Keep ordinary month for EDA, sampling and Tableau. This avoids all-zero encoding of unseen calendar months, without proving seasonal generalization.
- Remove repeated automatic missingness indicators where explicit condition/cloud flags already describe the absence. Wind and gust retain automatic indicators. The numerical preview has 35 columns rather than the former recipe's 40 on the new sample.
- Read every cloud group in the METAR body before RMK, with structured sky-column fallback and unknown-height/coverage flags. This recovers known ceilings for 719 initial-training flights; the follow-up audit finds zero remaining known body-ceiling omissions.
- Apply the user-approved exclusion rule at eight conflicting weather timestamps: six in 2025 plus ATL 2026-01-06 08:50 UTC and ORD 2026-02-20 08:00 UTC. Both versions are removed from prepared weather; original raw payloads remain. All eligible flights still match an allowed earlier report.
- Refresh the exploratory supplement for January–September 2025 and correct its interpretation of the now-fixed representation issues. Additional weather-history, interaction and relative-traffic features remain candidates, not new automatic additions.

A now has 10 predictors and B has 37. Initial-training-only B preprocessing yields 293 encoded columns in both train and validation. No classifiers are trained.

## Execution evidence

| Notebook | Code cells passed | Inline figures | Errors / warnings |
|---|---:|---:|---|
| 01_cleaning_and_eda.ipynb | 22 | 13 | 0 / 0 |
| 02_feature_engineering.ipynb | 20 | 13 | 0 / 0 |
| 02a_deeper_training_exploration.ipynb | 11 | 7 | 0 / 0 |

Verification covers unique IDs, current date ranges, sample sizes and proportional strata, no test IDs in final refit, A/B allowlists and dictionary order, cyclic values, nanosecond UTC types, weather age/exclusion rules, finite preprocessing and current handoff checksums. An independent brute-force calculation matches 54 historical windows and traffic counts for 18 airport/period boundary queries.

The actual preprocessing function also passes A/B checks with scaling both enabled and disabled (264 columns for A, 293 for B). Focused weather checks cover extra cloud layers, ignored RMK layers, unknown ceiling heights, clear skies and missing coverage.

Every pre-existing source flight ID, label, eligibility flag and event/cutoff timestamp is unchanged. The same two Parquet paths and shared preparation manifest were rebuilt; no extra intermediate CSVs or explicit IPython imports were added. Representative saved figures were visually checked.

## Evaluation boundary and next stage

Outcome EDA is January–September 2025 only. Test labels are constructed and used by the frozen proportional sampling rule; no 2026 delay rates, predictor/outcome relationships or model scores were inspected for development.

Frozen rolling history can use earlier 2026 outcomes after the declared seven-day lag when scoring later test flights. This simulates sequential prediction under the existing timely-feed assumption. It does not train a classifier on 2026 or permit selection using test outcomes. Monthly BTS releases do not establish a live feed; weather receipt and archived schedule availability remain assumptions.

Notebook 03 remains to be built. It will select using 2025 validation, freeze model/features/settings/threshold, refit finalized pipelines on refit_sample, then evaluate on the 2026 model_sample and export real findings for the existing Tableau contract. No model performance or Tableau findings are claimed here.
