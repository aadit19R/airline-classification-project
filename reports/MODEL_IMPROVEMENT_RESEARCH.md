# Research on improving the four flight-delay classifiers

Reviewed 19 October 2026. This is a literature/source review and a proposed experiment plan. No models were retrained, no new predictive performance is claimed, and the current frozen study and Tableau exports remain unchanged.

The current experiment is a useful initial benchmark, not an exhaustive search: three C values for Logistic Regression and SVM; six k/weight settings for KNN; five depth/leaf settings for Random Forest. SVM gamma stayed at scale, RF feature subsampling and class weights were fixed, and every classifier fitted 20,000 flights. Those decisions limit what can be concluded about each model's achievable performance.

## What similar projects actually show

| Primary source | What was done and reported | Transfer to this project |
|---|---|---|
| [Serena Fang and collaborators, BTS departure-delay project](https://github.com/Serena-Fang/Machine-Learning-Project-Flight-Delay-Prediction) | Used DepDel15, LR/DT/RF, categorical encoding comparisons, training upsampling and parameter search. The chosen RF with m-estimate encoding and training resampling reported test precision 0.44, recall 0.59 and accuracy 0.75. | Encoding and sampling deserve controlled comparisons. Their results are not a promised gain for our different period, airports and evaluation. Their report does not establish our chronological future-quarter protocol. |
| [Lemay and Bastin, Prediction of airport on-time performance, preprint](https://arxiv.org/html/2601.00875v1) | Uses historical flight-number, gate and airline punctuality; earlier records only for histories. Flight-number and airline history are important in their prediction models. The paper uses random train/validation/test sampling and has private airport operational variables. | Airline + flight-number + origin history is feasible with our existing raw-derived fields, using the existing lag, smoothing and sparse-history flags. Do not copy private gate features or treat their performance as a chronological benchmark. |
| [Rebollo and Balakrishnan, MIT, 2014](https://www.mit.edu/~hamsa/pubs/RebolloBalakrishnanTRC2014.pdf) | RF models use local and network delay states to predict 2–24 hours ahead. Their classification concerns aggregate origin–destination delay states and a 60-minute threshold. | Supports investigating current disruption context. It does not show that our individual-flight 15-minute problem should attain their reported error rate. Our seven-day history lag excludes the freshest state. |
| [Stanford CS229, Modeling Flight Delays, 2016](https://cs229.stanford.edu/proj2016/report/DuperierSauvestreLeaf-ModelingFlightDelays-report.pdf) | Builds time dependence, congestion, holidays and transformed inputs; also reports no improvement from SMOTE in its experiments. Despite its broad title, the introduction predicts arrival delay at departure time. | Interactions and temporal context are relevant ideas; the reported F1 is not a benchmark for our two-hour pre-departure target. No previous-aircraft feature should bypass our assignment-timing restriction. |
| [Kim and Park, Journal of Big Data, 2024](https://link.springer.com/article/10.1186/s40537-023-00867-5) | Compares familiar models with richer weather, including dew point and pressure. It downsamples weather-delay cases to 1:1; ICN labels use a delay greater than one hour. | Richer weather deserves testing, but the often-quoted high accuracies are not directly comparable with natural-prevalence, all-cause 15-minute delays. Retain our evaluation population. |

A further [PLOS One study](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0335141) lists actual departure/arrival times and delay amounts among selected features. Whatever their intended setup, that feature list is unsuitable for our two-hour cutoff. We should not reproduce a large score by adding information that becomes known after prediction.

The lesson is to copy defensible methods and check them on our task, rather than copy headline accuracy.

## Prioritized changes using the existing algorithms

All entries below are hypotheses to test on 2025 development data, not demonstrated improvements.

| Model | Limitation of the first run | Next controlled comparisons |
|---|---|---|
| Logistic Regression | Linear effects of the current encoded inputs; C=0.1 was best at the lower edge of the searched values. | Extend regularization below 0.1; add a small number of airport × time and weather × time interactions. Compare richer periodic time representation, such as an additional harmonic or periodic splines. |
| KNN | Standardized Euclidean distance across 293 encoded columns; best k=31 was the largest tried. Distance and duration are also highly correlated. | Compare k=31/61/101, Euclidean versus Manhattan distance, and removal of one redundant distance/duration input. Compare scaling continuous measurements separately from 0/1 flags. Check compact encodings or training-fitted feature reduction only if simpler changes do not help. |
| RBF SVM | Only C changed; gamma was fixed. | Joint C/gamma search on a logarithmic scale, with stronger regularization/smoother kernels among candidates. A linear SVM can be a separate, explicitly labelled SVM variant. Keep actual model identity clear in comparisons. |
| Random Forest | max_features stayed at sqrt and class_weight stayed balanced; only five depth/leaf combinations. | Compare feature-subsampling fractions, depth/leaf controls and None versus balanced weights. More trees can stabilize estimates, but should not be treated as a substitute for informative features. |

The [official RBF example](https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html) identifies C and gamma as interacting controls. The [neighbors guide](https://scikit-learn.org/stable/modules/neighbors.html) explains the noise/smoothing trade-off from larger k and the difficulty of high-dimensional neighbourhoods. The proposed scaling and redundant-feature ablations are our data-specific hypotheses.

The [time-feature example](https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html) demonstrates that linear models can benefit from richer periodic representations. Our existing sine/cosine month encoding correctly represents a cycle; one harmonic still limits the shapes a linear predictor can express. This is a possible extension, not a reason to revert to unseen one-hot months.

RF class weights already address imbalance in the initial experiment. The [RF documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) defines those controls. Testing alternative weights is reasonable; claiming imbalance was ignored would be incorrect.

## Variables to try, and whether more sources are required

A source-schema check confirmed that the saved clean table contains flight_number and the original weather responses contain dwpf, drct, p01i and full metar text. These are not all current B predictors.

| Candidate | Construction and cutoff rule | New source requirement |
|---|---|---|
| Flight-number punctuality | Smoothed history for airline + flight number + origin, optionally destination where support permits; only eligible earlier events within declared historical windows. Keep the seven-day lag initially. | None; flight number and source events are retained in the clean table. |
| Dew-point spread | Temperature minus dew point from the already matched report; a simple moisture-related predictor. | None; both measurements already exist in raw weather. |
| Wind direction | Encode degrees cyclically; consider direction with wind speed rather than treating 359 and 1 degrees as distant. | None; raw drct exists. This is not a crosswind estimate without runway information. |
| Earlier weather | Positive accumulation flag, visibility minimum or change, and gust maximum over a few hours of reports available by cutoff. Never sum overlapping p01i totals. | None for origin-station observations; construct from existing raw reports using the same exclusions and age/buffer rules. |
| Freezing/wet-weather interaction | A small, predeclared combination of temperature and observed precipitation/condition indicators. Unknown values remain explicit. | None; incremental benefit is unproven. |
| Relative scheduled traffic | Compare a known schedule count with its typical origin/weekday/hour level, learned from training only. | None for scoped departure counts. |
| Scheduled arrival demand | Add arrivals to hub traffic using national raw schedules and correctly reconstructed arrival dates/UTC. | Raw flights already exist, but broader timezone/overnight parsing needs an audit before coding. |
| Altimeter pressure and pressure trend | Audit A/Q groups in METAR body; derive consistently labelled altimeter settings rather than inventing sea-level pressure. | Full METAR is present; usable coverage must be checked. Otherwise request alti explicitly. |

[IEM documentation](https://mesonet.agron.iastate.edu/request/download.phtml) defines dew point, wind direction, pressure and precipitation fields. It notes that precipitation accumulations relate to observation/reset periods and that M has several possible meanings. Unknowns should not be blindly changed to zero.

Flight-number history, dew-point spread, prior weather and relative traffic are the first shortlist. The earlier exploration already found a broad association for prior wet weather, but the same-airport/day comparison was much smaller and had low overlap. It remains a candidate, not an established recovery effect.

More demanding candidates have different requirements:

- Same-day airport delay state: raw outcomes permit a retrospective calculation, but trustworthy real-time availability needs an outcome/status feed or a separately declared latency scenario. Do not silently replace the seven-day lag with minutes.
- Destination weather or forecasts: complete destination coverage for both years and archived forecast issuance times would need additional sourcing. The Q1-2026 weather acquisition covers the three origin stations only.
- Aircraft rotation: remains excluded until historical assignment-at-cutoff evidence is available.
- Additional years: might improve seasonal/rare-event coverage, but are not the first prerequisite. Existing unused raw variables and model settings should be investigated first.

## Thresholds, imbalance and sample size

Our original validation threshold maximizes delayed-class F1, and the selected LR threshold is approximately 0.191. Its low warning precision is partly an operating-point issue as well as limited predictive separation. A precision-prioritizing objective, such as F0.5 or maximum recall subject to a predeclared supported precision target, is a different decision policy. It requires discussion and validation; no precision target is guaranteed achievable.

Changing the threshold moves the precision/recall trade-off without improving the underlying score ranking or AP. [Threshold documentation](https://scikit-learn.org/stable/modules/classification_threshold.html) makes this distinction explicit. Raising the threshold does not guarantee every empirical precision step rises, and generally loses recall. We should report the trade-off instead of selecting an attractive accuracy in isolation.

Training resampling is worth a small comparison, but not an automatic fix for roughly 1:3 class imbalance. Current recall is already high and false warnings are the main concern. Plain SMOTE applied to one-hot identities can create fractional category combinations; [imbalanced-learn](https://imbalanced-learn.org/stable/over_sampling.html#smote-variants) provides mixed-type methods, but class weighting or simple training-only resampling is easier to explain first. Never balance validation/test or resample before splitting.

Use learning curves to decide whether 20,000 training rows are limiting the models. Compare shared cohorts at increasing sizes, for example 10k/20k/40k, with unchanged validation flights. Model fitting/runtime may differ greatly. LR/RF can additionally be examined at larger sizes in a clearly separate scalability analysis. The [learning-curve guide](https://scikit-learn.org/stable/modules/learning_curve.html) explains when more examples help; it does not guarantee that a larger sample improves this dataset.

## Proposed evaluation of an improvement phase

1. Preserve notebook 03, its selection record, bundle and current results as the baseline.
2. Declare a small set of changes, not a large collection of interactions chosen from test errors.
3. Use calendar-based rolling validation within 2025, with whole flight dates kept in one block. Fit imputers, encoders, feature selection, scaling and any resampling inside the fitting portion.
4. Compare tuning changes first, then add feature families one at a time on matched flights. Separate threshold-policy changes from AP/ranking improvements.
5. Check per-fold scores, runtime, false warnings, subgroup support and sample-size learning curves. Use several seed samples only as robustness checks, not to select a lucky result.
6. Freeze the improved procedure before a new independent confirmation period. Q1 2026 has already been inspected and cannot honestly be called an untouched test again. A later unexamined period would require source-availability checks and acquisition before a fresh final claim.

[Time-series validation guidance](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split) explains why ordinary shuffled folds are unsuitable for correlated temporal data. Calendar blocks are preferable here to mechanically splitting unequal numbers of sampled flights across dates.

The first improvement round needs no new external predictor source and can stay with the same four familiar model families. Additional source data becomes necessary for operationally timely disruption signals, complete destination forecasts, additional years, or a new untouched final evaluation period. No performance improvement is promised before these comparisons are measured.
