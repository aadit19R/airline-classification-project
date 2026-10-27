# Departure Insights · offline dashboard

The [four simple notebooks](../notebooks/simple/README.md) are the main explanation and reproduction sequence. The dashboard continues to read the same frozen July exports; a link in its footer opens the notebook guide when the project folder is available. Copying the HTML file alone still preserves all dashboard functionality.

Open **[departure_delay_dashboard.html](dist/departure_delay_dashboard.html)** in Safari, Chrome or Edge. Double-click the file in Finder. It is a single portable file: its charts, styling, code and study data are embedded. No account, internet, installation or local server is required.

The four pages of this HTML dashboard map to the agreed ML objectives:

1. **O1 · Model performance:** model/feature selectors, six comparison metrics, baseline references, confusion matrix, precision–recall curves, runtime and exact comparison table.
2. **O2 · Feature value:** schedule A → B and equally retuned 66 → B comparisons, paired 95% intervals, predictor counts and precisely defined source coverage.
3. **O3 · Prediction reliability:** airport/airline/time-band comparisons, supported-rate heatmap, group detail, FP/FN filters, searchable and sortable flight lists.
4. **O4 · Risk prioritization:** 5%, 10% and 20% quotas, capture/precision/lift, historical and random baselines, exact ranked review lists.

The global model selector changes the displayed model; Logistic Regression B remains the registered validation choice. The feature selector affects O1 only; O3/O4 use each model's selected B package. O2 compares all four classifiers.

Flight-list airport/search/outcome filters affect only the list. Selecting an O3 group updates its detail panel and list; the global scorecards remain on the full July sample. Global risk ranks and the review quota are never recomputed after filtering. CSV exports preserve the current scope; the top export button exports the current page's aggregate evidence, while Export list downloads all filtered rows, not just the visible 25-row page. Print page uses the browser print dialog and can be saved as PDF.

## Evidence and scope

The sole inputs are the six frozen files in `reports/evaluation_exports/final_july/`. Their hashes are checked against that run's manifest. The file embeds all 16 metric candidates, all 113,047 precision–recall points, the feature/subgroup/quota tables, and all 10,000 B predictions for each of the four classifiers. Flight identities are shared once to avoid duplication. No new fitting, sampling, threshold adjustment or test-driven selection occurs.

Default: Logistic Regression B, 10% review quota, July 2026, 10,000 natural-prevalence flights at ATL/DFW/ORD. This is one-month summer confirmation under historical input-availability assumptions. Unsupported subgroup rates stay null/blank; scores remain uncalibrated. Study notes contain objectives, metric definitions, assumptions and original source checksums.

## Rebuild and verification

From the project root:

```sh
python3 scripts/build_html_dashboard.py
node scripts/check_html_dashboard.mjs
```

The builder uses the project's existing pandas/NumPy environment. The dashboard itself uses plain HTML, CSS, JavaScript and responsive SVG, with no third-party runtime dependency.

- Builder verifies six input hashes and 52 confusion/quota reconciliations.
- JavaScript checks cover 287 assertions: all four models, all quotas, original risk order, list scopes, supported/null rates, feature deltas, export scopes, CSV escaping and chart geometry at narrow and desktop widths.
- Native Safari preview verifies all four pages, the feature contrast control, ATL subgroup selection, model switching, and 5%/10%/20% review totals.
- `build_manifest.json` records the exact run, source hashes and HTML digest; `verification.json` records the automated checks.

Editable sources: `template.html` (layout/styles) and `dashboard.js` (charts, interactions and export logic). The generated file in `dist/` is the file to share or submit.
