Current status: development and the completed July evaluation are recorded here. Read [RESULTS.md](RESULTS.md), or run the [four current notebooks](../../notebooks/simple/README.md) to reproduce the final features, model fits and findings. The six final_july export sources remain the frozen evaluation evidence.

# Final compact-feature experiment

This is a separate experiment authorized after notebook 02's exploratory feature trials. It preserves the original Q1 and Q2 results. Its development decisions and outputs are retained in the JSON/NPZ records listed below; the current four-notebook sequence reproduces the frozen final choices.

## Decision procedure

Development uses only 2025. Three expanding training periods end in March, June and September. April, July and October calibrate warning thresholds; May–June, August–September and November–December score them without refitting. A shared 20,000-flight fitting sample is used for selection, proportional by month and label. Evaluation prevalence stays natural.

Each model screens the current 66 features, six pressure/wind additions, six nonlinear weather additions, and their combination. Its strongest additional package and the 66-feature reference receive the same four-setting search. This is a bounded two-stage search. It does not exhaustively optimize every possible package.

The primary selection metric for this round is mean delayed-class F1, followed by AP. This updates the original AP-first selection rule for this experiment only. Choices are development evidence, with selection optimism, rather than new final-test estimates.

Fixed choices receive a latest-fold check with 40,000 fitting flights. October thresholds are frozen; November–December assesses the larger fit without changing the selection. The final refit uses a shared 40,000-flight sample spanning all twelve months of 2025. Transferring October thresholds to a new all-2025 fit can shift the warning rate; the fresh test must measure this, not conceal it.

A schedule-only A control uses each selected B model's exact settings, preprocessing recipe and flight IDs, including the same time harmonics. A/B measures additional weather/history/traffic information at matched settings, rather than claiming that A was independently optimized. The retuned 66-feature control separates additional-feature gains from hyperparameter changes.

## Fresh confirmation period

On 19 October 2026, the [BTS release history](https://transtats.bts.gov/releaseinfo.asp) and [reporting table](https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ) listed July 2026 as the latest month. August and September were not listed as available. July supplies a one-month fresh confirmation, not a full-quarter or year-round estimate.

The structural acquisition audit reads schedule and weather fields only. It found 631,970 national rows, 89,701 scoped schedules, all 31 days, no new conflicting weather timestamps and no missing weather-key matches. June/July overlapping payloads agree. Raw ZIP/CSV/weather responses, requests and checksums are saved under data/raw; the original execution deferred outcome loading until the model bundle was frozen.

The completed evaluation scored 10,000 shared natural-prevalence July flights. No features, settings, thresholds or model choice may change after access. Earlier April–June outcomes support lagged histories; July events can support later July queries only after the same seven-day lag. This is a sequential historical simulation under the existing availability assumptions, not evidence that BTS monthly publications could supply a live daily feed.

Compare all four models, the retuned 66-feature reference, matched-settings schedule controls, always-on-time, always-delayed, a fixed fair coin and historical-risk ranking. Paired day-block bootstrap intervals use 300 replicates, conditional on the fitted models and only 31 dates.

## Outputs

- protocol.json: predeclared decision procedure and input hashes.
- cohorts.json: shared development/calibration/scoring/refit identities.
- development.json / development_scores.npz: trial metrics and score audit arrays.
- selection.json: frozen model, features, settings, thresholds and bundle hash.
- models/final_flight_delay_models.joblib: final fitted pipeline bundle.
- test_access.json: first July outcome access and locked test procedure.
- test_results.json: fixed July results and paired comparisons.
- reports/evaluation_exports/final_july/: six versioned result CSVs and an export manifest.

The preparation Parquets have clearer period/sample column names; every nonmetadata value is unchanged, as recorded in `reports/study_labels.json`. The original preparation manifest, prior models and earlier exports are preserved. No new intermediate feature CSVs are required. The HTML dashboard uses the final July exports.
