# Broader feature-engineering experiment

Reviewed 19 October 2026. The new section at the end of notebook 02 implements the candidates below. Selection uses chronological 2025 development results, not individual correlations or either exposed 2026 test quarter.

| Source | Method that informed this experiment | Adaptation and limits |
|---|---|---|
| [Serena Fang and collaborators](https://github.com/Serena-Fang/Machine-Learning-Project-Flight-Delay-Prediction) | Categorical treatment of day variables despite weak numeric correlations; time/distance bands and alternative categorical encodings. | Try day-of-month/hour categories, fixed distance bands and route/airline combinations. Retain sparse, training-fitted one-hot encoding; do not copy their random-split target encoding. |
| [Duperier, Sauvestre and Leaf, Stanford CS229](https://cs229.stanford.edu/proj2016/report/DuperierSauvestreLeaf-ModelingFlightDelays-report.pdf) | Holidays, squared inputs, schedule demand and time dependence. | Try holiday proximity, nonlinear weather and scheduled arrival/departure waves. Their target is arrival delay at departure time; their reported scores are not a benchmark for our horizon. Their same-day/aircraft histories are not copied. |
| [Kim and Park, Journal of Big Data](https://link.springer.com/article/10.1186/s40537-023-00867-5) | Richer weather measurements, including pressure. | Extract altimeter setting and earlier changes from existing METARs. Their pressure variables, weather-delay labels and populations differ; we do not treat their feature rankings as proof of usefulness here. |
| [OPM federal-holiday calendar](https://www.opm.gov/policy-data-oversight/pay-leave/federal-holidays/#url=2025) | Recurring U.S. federal holidays. | Encode holiday day and proximity using pandas' recurring calendar. This is a demand proxy, not a passenger count; exceptional closures and school/local holidays are not covered. |
| [NWS METAR decoding](https://aviationweather.gov/help/data/) and [IEM fields](https://mesonet.agron.iastate.edu/request/download.phtml) | Altimeter and observation-code definitions. | Parse A groups before RMK, preserve unknown weather and apply the existing buffer/age rule. Altimeter setting is distinct from sea-level pressure. |
| [IANA timezone definitions](https://data.iana.org/time-zones/tzdb/northamerica) and [compatibility links](https://data.iana.org/time-zones/tzdb/backward) | Civil-time rules, including the U.S. Virgin Islands. | Join raw IATA/ICAO station metadata; use America/St_Thomas for STT/STX instead of the station catalog's Bermuda rule. Resolve repeated clock hours against the published schedule. |

Route geometry, daily airline schedule shares and time-of-day-conditioned histories are additional hypotheses developed for this dataset. They are not presented as exact reproductions of another project's implementation.

## Features and source handling

The experiment adds **64 candidates in nine groups** to the current 66 predictors. It separately compares the previous 28 candidates and a combined 158-predictor package. All 938,417 eligible 2025 development flights remain available. There is no correlation-based feature rejection.

The groups are calendar/holiday timing, categorical combinations, route geometry, schedule structure, departure waves, scheduled arrivals, pressure/wind, nonlinear weather, and departure-period-conditioned histories. All input data already exists in this project's raw sources or its own reviewed handoffs. No previous project's cleaned data is used.

National arrivals use only schedule columns from December 2024, all twelve 2025 months and January 2026. January schedule data supplies boundary context; no 2026 outcomes are read. The user approved using the arrival clock, inferring its date from departure plus duration, and retaining/flagging schedule discrepancies. Audit-only discrepancy counts accompany the new aggregate traffic features. Every schedule and query flight is retained. Any unresolved timezone/date ambiguity stops execution.

Airport geometry uses coordinates/elevation only. Current closure/operating status is not a historical predictor. Historical target rates retain the seven-day availability lag; weather retains the existing 20-minute buffer and 180-minute maximum age. Schedule revisions and report delivery remain historical-simulation assumptions.

## Development comparison

Use identical 20,000-flight training samples for all packages/models. For each validation quarter, use its first month to choose the F1 threshold, then score the following two months without refitting. Imputation, encoding and scaling are fitted on training only. All models keep their previously selected settings; this isolates feature changes rather than mixing them with another parameter search.

Screen every package with Logistic Regression and Random Forest, then compare the two best pilot packages against the same baseline with all four classifiers. F1 is primary; average precision, precision, recall, false warnings and runtime are also reported. The selection/scoring periods are development data, so package-selection optimism remains. An unexamined period is needed for unbiased final confirmation.

The machine-readable protocol and measured results are in `broad_feature_experiments.json`; execution and integrity checks are in `broad_feature_notebook_audit.json`. No processed/split datasets, fitted-model bundles or Tableau exports are replaced.
