# Current experiment inputs and reference

| File | Needed by | Purpose |
|---|---|---|
| `selection.json` | Notebooks 02 and 03 | Feature lists, model settings, preprocessing recipes and thresholds |
| `cohorts.json` | Notebooks 02 and 03 | Exact 40,000 final fitting IDs and the current study's 2025 development cohorts |
| `reference.json` | Optional verifier | Frozen feature fingerprints, metrics, ranks and review quotas |
| `reference_scores.npz` | Optional verifier | Original scores for the eight final A/B models and four baselines |

The compact reference was extracted from the previously verified final July exports before the bulky exports were removed. It provides an independent original-results comparison for notebook reruns. It does not choose new settings or thresholds.

The older initial/Q2 experiment reports and trial logs have been removed. Current results are in `reports/simple_reproduction/metrics.json`; see [the findings](RESULTS.md).
