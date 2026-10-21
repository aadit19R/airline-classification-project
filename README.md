> **Current modelling stage complete (19 October 2026):** [notebook 06](notebooks/06_final_model_selection.ipynb) selects compact features/settings using 2025; [notebook 07](notebooks/07_final_july_evaluation.ipynb) evaluates the frozen choices on 10,000 July-2026 flights. Read the [measured final findings](reports/final_models/RESULTS.md) and use [final July Tableau sources](tableau/exports/final_july/README.md). The visual deliverable is now the [offline interactive HTML dashboard](dashboard/dist/departure_delay_dashboard.html), replacing Tableau/Power BI at the user’s request. See its [controls and verification guide](dashboard/README.md). Older Q1/Q2 experiments below are retained as benchmarks. The frozen original config records the original round; the final round is governed by reports/final_models/protocol.json and selection.json.

# Pre-departure Flight Delay Risk Classification

A fresh project using Logistic Regression, KNN, SVM and Random Forest, with a **standalone interactive HTML dashboard** as the visual findings deliverable.

**Current stage:** frozen July-2026 model confirmation, the four-page HTML findings dashboard and the [nine-page project report](reports/final_report/output/pdf/predeparture_flight_delay_report.pdf) are complete. The report emphasizes extraction, cleaning, EDA and feature engineering, with six figures and model results near the end. An [editable source and reproduction guide](reports/final_report/README.md) is included. The original Q1 and improved Q2 experiments remain as historical benchmarks.

## Main objective

To develop and evaluate, through a historical simulation with a two-hour pre-departure cutoff, a classification framework using Logistic Regression, KNN, SVM and Random Forest, and assess whether its predictions can reliably identify and prioritize flights at risk of departing at least 15 minutes late.

## Supporting objectives

**O1 - Select and verify a suitable classifier.** To select the most suitable classifier through chronological validation and verify its delayed-flight detection performance and computational practicality on the final evaluation period.

**O2 - Measure the value of additional information.** To assess whether adding origin weather, historical departure-delay patterns and scheduled airport traffic improves departure-delay classification compared with schedule features alone, using the same later-period flights and stated input-availability assumptions.

**O3 - Identify prediction strengths and failure patterns.** To evaluate the reliability of the selected model across airports, airlines and departure time bands by identifying missed delays and false alerts in sufficiently supported subgroups.

**O4 - Assess the usefulness of risk prioritization.** To quantify how effectively the selected model concentrates actual delayed flights within predeclared review budgets of 5%, 10% and 20% of scored flights, compared with random prioritization and a simple historical-risk baseline.

## Agreed scope

- ATL, ORD and DFW departures: 2025 development/final refit. Q1 2026 is the completed original test; Q2 2026 is the completed fresh confirmation for improvements.
- Predict departure delay >=15 minutes, two hours before scheduled departure.
- Noncancelled flights with known departure outcomes; later-diverted flights can remain when departure is known.
- Original experiment: January–September training, October–December validation, all-2025 refit, Q1-2026 test. Improvement experiment: three rolling 2025 validation blocks, all-2025 refit, Q2-2026 test.
- Original shared initial-training / validation / test samples: 20,000 / 5,000 / 10,000; a separate 20,000-flight sample spans all 12 months of 2025 for final refit. Month × outcome stratification retains natural proportions; all eligible rows remain saved.
- A: schedule; B: A plus weather/history/traffic. Four models across two feature sets give eight candidate comparisons.
- Aircraft rotation (previously C) is outside the agreed project scope because historical assignment times are unavailable. Acquired tail-number fields remain stored without becoming model predictors.
- The experiment simulates a two-hour cutoff using archived records and stated weather/outcome delivery assumptions; it does not verify live availability. Historical schedule revisions are also not reconstructed.

The earlier project already examined 2025. Improvement research uses 2025 development only. Q1 results were already exposed; the newly sourced Q2 confirmation was evaluated only after the improved procedure was frozen.

The improvement phase compares nested 20k/40k fitting samples and selects 40,000 for all final models. Original configuration and preparation membership flags are preserved; the new frozen choices are recorded in reports/improvements/selection.json. New features are built in memory, with no extra processed dataset.

## Start here

1. [Objectives and the evidence needed to answer them](OBJECTIVES.md).
2. [Implementation plan](PROJECT_PLAN.md).
3. [Open the four-page findings dashboard](dashboard/dist/departure_delay_dashboard.html) · [usage and verification](dashboard/README.md). The earlier [Tableau specification](tableau/WORKBOOK_PLAN.md) is retained as design history.
4. [Export schemas for model results](tableau/export_contract.json).
5. [Project configuration](config/project.json) and [source inventory](data/SOURCE_MANIFEST.json).
6. [Data sources and exact variable checklist](DATA_REQUIREMENTS.md).
7. [Six-stage cleaning, EDA and feature preparation plan](EDA_PLAN.md).
8. [Complete dataset reference: sources, periods, record counts, variables and saved schemas](DATASET_GUIDE.md).
9. [Implemented schema, join keys, temporal matching and table relationships](SCHEMA_AND_JOINS.md).
10. [Shorter preparation notebooks: changes, original backups and execution verification](reports/notebook_simplification/README.md).

## EDA and ML each have a role

EDA remains part of cleaning, understanding the data and communicating context. The formal objectives additionally require model scores, prediction errors or controlled feature comparisons. The dashboard shows both context and model evidence. Modest or negative predictive results must be interpreted honestly; the objectives do not promise an accuracy target. Read [the explanation with examples](EDA_AND_ML_EXPLAINED.md).

## Current milestone

The final compact-feature round selected Logistic Regression / B on earlier validation and confirmed it on July 2026. The four HTML pages present model performance, paired feature gains, supported prediction errors and fixed-budget risk prioritization using the exact frozen July exports.

## Deliverables

Executed preparation, exploration and modelling notebooks; documented raw/processed data; fitted basic models; reproducible evaluation exports; a self-contained HTML dashboard covering all four supporting objectives; and a concise PDF report with editable source. The dashboard includes filtered CSV export and a browser print view. The report links the data preparation decisions to their findings and includes the final objective status, limitations and references.

All active raw source inputs now live inside this folder. Only original raw files were copied from the earlier project; 2025 flight archives were acquired directly from BTS. No prior cleaned/processed tables or generated timezone lookup is reused. Core dependencies are listed in requirements.txt.
