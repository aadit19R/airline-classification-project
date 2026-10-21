> **Superseded visual format (19 October 2026):** the user cancelled Tableau and Power BI and requested an offline HTML dashboard. The implemented deliverable is [Departure Insights](../dashboard/dist/departure_delay_dashboard.html), with [usage and verification](../dashboard/README.md). This file remains the historical Tableau design specification; no completed Tableau workbook is claimed.

> **Current data snapshot:** use [final_july/](exports/final_july/README.md) and its export_manifest.json. The frozen 2025 validation choice is Logistic Regression; July's largest test F1 does not select a new model. See [final findings](../reports/final_models/RESULTS.md). The workbook is still pending. The older Q2 packaging notes below are historical; dashboard logic and filter/grain rules continue to apply.

# Tableau workbook specification

**Deliverable:** `workbooks/departure_delay_findings.twbx`, with four dashboards and the supporting worksheets/data sources. **Status:** all six real model-result exports are available; building, packaging and visually checking the workbook is the next stage. Use the current [July final export guide](exports/final_july/README.md) and [final findings](../reports/final_models/RESULTS.md). The original Q1 sources remain available separately.

Tableau can [connect directly to CSV text files](https://help.tableau.com/current/pro/desktop/en-us/examples_text.htm). A [packaged workbook](https://help.tableau.com/current/pro/desktop/en-us/save_savework_packagedworkbooks.htm) includes local data sources and is the intended portable deliverable. Use UTF-8 CSV exports, optionally converted to extracts in Tableau. Keep the versioned ML data snapshot used in the submitted package.

## Workbook-wide design

- Every page states its objective ID, target (departure >=15 min late), prediction horizon (2 hours), airports, period, sample size and model/feature version.
- Default model is the frozen validation-selected candidate from A/B. Aircraft rotation (previously C) is excluded from the agreed scope. All labels distinguish validation evidence from final-period reporting. Do not mark a test-selected winner as the chosen model.
- Use four-page navigation, readable axes and consistent colours: blue for correctly identified delays, orange for false alerts, red for missed delays, grey for on-time/unsupported groups. Include exact counts in tooltips.
- Include descriptive context where it makes the story understandable: sample/class balance, observed group delay rates, weather coverage and review-list size. Principal objective answers additionally use predictions, evaluation, feature comparisons or ranking results. EDA is welcome; it does not replace the prediction evidence. See [EDA and ML explained](../EDA_AND_ML_EXPLAINED.md).
- Scores are labelled risk scores/ranks. SVM decision scores and uncalibrated model outputs must not be labelled calibrated delay probabilities.
- Include a visible study note: `Historical simulation; two-hour cutoff with assumed weather/outcome availability. Live performance not verified.` Archived schedules are used; historical schedule revisions are not reconstructed. No dashboard requires rotation results.

## 01 Model performance - O1

Purpose: show the validation-selected model and confirm its final-period performance versus the alternatives and baselines.

Compare Logistic Regression, KNN, SVM and Random Forest. Use `model_metrics.csv` for F1/AP/precision/recall cards, model bars and confusion counts; `pr_curve.csv` for precision-recall curves. Display train/predict time to explain model practicality. Default to the selected feature set; allow model/feature-set and validation/test viewing controls. Highlight the registered validation choice independently of whichever bars happen to be tallest in test.

The answer is a measured statement such as which candidate balanced delayed-flight detection and runtime. Fill it only after real results exist. Training accuracy is not the headline finding.

## 02 Feature value - O2

Purpose: assess whether B (schedule plus weather/history/traffic) improves on A (schedule alone) on identical later-period flights.

Use `feature_comparison.csv` for paired AP/F1 change bars, and `model_metrics.csv` for a model-by-feature-set comparison. Show common flight count, matching cohort fingerprint and coverage of added variables. Report negative or negligible changes as findings.

A-to-B measures the combined weather/history/traffic addition. Claiming separate benefits for weather, history or traffic requires additional predeclared ablations. F1 changes can reflect separately validation-selected operating thresholds; AP measures ranking without requiring a common raw threshold.

## 03 Prediction reliability - O3

Purpose: identify where the frozen selected model misses actual delays or produces false warnings.

Use `subgroup_metrics.csv` for airport/time-band heatmaps, airline precision/recall comparisons and a detail table with actual/predicted positives, true positives, false positives and false negatives. Controls choose the grouping dimension and error metric. Drill into actual sampled flight errors through `flight_predictions.csv`.

Support rules: at least 200 group rows; at least 30 actual delayed flights for recall/miss rate; 30 actual on-time flights for false-positive rate; 30 predicted delayed flights for precision. Show unsupported metrics as blank/grey with an explanation. These thresholds are project reporting choices, not proof of statistical certainty. Group findings remain descriptive within the evaluation period.

Plot prediction error rates, not just raw delay counts. Define miss rate as FN/(TP+FN) and false-positive rate as FP/(FP+TN); the denominators differ.

## 04 Risk prioritization - O4

Purpose: show how many actual delays appear in a limited high-risk review list.

Use `priority_curve.csv` for gains/lift and capacity cards and `flight_predictions.csv` for the corresponding review list. The capacity control offers the predeclared 5%, 10% and 20% quotas; default 10%. Compare the selected model with random expectation and a historical-risk ranking fitted from earlier data. Display delayed flights captured, recall, precision, false alerts and lift over cohort prevalence.

Ranking is within one fixed evaluation cohort, model and feature set. Export the completed risk rank and cohort size from Python so the Tableau review list uses exactly the same tie order and quota as the capacity metrics. Break tied scores using a reproducible outcome-independent hash of seed and flight ID. Let K=ceil(quota*N). Recall=delays_in_top_K/actual_delays, precision=delays_in_top_K/K, lift=precision/(actual_delays/N). Random expected recall=K/N; random expected precision=actual_delays/N. Undefined denominators produce nulls.

This is a retrospective cohort-ranking exercise. It does not establish a live alert threshold, savings, interventions or delays prevented. A capacity slider changes a display scenario, not the frozen classification threshold or chosen model.

## Export and filter rules

Follow [export_contract.json](export_contract.json). Each file has an explicit grain. Predictions contain multiple candidate rows per flight, so scope the view to exactly one model and feature set before counting flights; use unique flight IDs for flight counts.

Use separate data sources for row-level predictions, model metrics, paired comparisons, subgroup metrics and curves. Do not physically join aggregate metric rows onto every flight and then sum them. If relationships are used, define/test their keys and cardinalities against the [Tableau data-model guidance](https://help.tableau.com/current/pro/desktop/en-us/datasource_datamodel.htm).

Precomputed global metrics do not automatically recalculate when a flight list is filtered. Restrict filter actions to compatible sheets or precompute explicitly supported cohorts. Every chart states its own cohort and denominator; never make an airport-filtered flight list appear to validate an unchanged global scorecard.

## Completion checks

1. All four objectives have real measured evidence and an interpretation using A/B; no aircraft-rotation result is required.
2. Tableau cards reconcile to exported Python metrics and confusion counts for each candidate/cohort.
3. Candidate comparisons use identical flights, target definitions and periods.
4. Capacity cards and the review list reconcile at each quota, including tied scores.
5. Unsupported subgroup metrics remain blank, not zero.
6. The `.twbx` opens with its packaged data and displays correct fonts/labels/filters; export dashboard PDF/images as a viewing fallback.
7. Workbook snapshot, model-selection record, source/export manifest and report identify the same run.

## Current version for packaging

Use tableau/exports/final_july/ for the frozen July-2026 confirmation. Default model_version to final, feature_set to B, threshold_policy to F1, and the selected classifier to Logistic Regression. Keep the selected-model badge independent of July's highest F1. The retuned current-feature control is reference66; B's compact package differs by model and is recorded in model_metrics.csv. A/B uses matched settings and the same time harmonics/flight IDs. Do not mix Q1, Q2 or July observations in a paired chart.

Label the review-quota view as retrospective ranking. The selected model's top 10% review list has 688 actual delays among 1,000 flights; the full warning threshold produces 4,827 false alerts. The coverage card means matched weather report and available origin history, not complete weather measurements. Use the exact-run export manifest and final findings for all cards, captions and limits.
