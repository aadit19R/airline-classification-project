# Acquisition and notebook helpers

`preparation_helpers.py` contains named technical routines extracted from notebooks 01, 02 and 02a to make their visible workflow shorter. Their original calculations, plots and checks are retained. `verify_preparation_notebooks.py` reruns all three in an isolated temporary copy, including notebook 02's 90-fit experiment, and checks the saved handoffs and measured results. See [the readability verification guide](../reports/notebook_simplification/README.md).

The acquisition tools preserve raw response bytes. The improvement helpers expose reusable parsing/features/exports; model fitting and experiment choices remain visible in notebooks 04 and 05.

- `acquire_raw_data.py --stage copy`: copies unchanged original raw files/request metadata from the earlier project's `data/raw` only. Source/destination SHA256 values are checked. No previous processed/interim/report/model/lookup input is allowed.
- `acquire_raw_data.py --stage download --workers 3`: downloads twelve full official 2025 BTS archives, including aircraft registrations, with provenance.
- `acquire_raw_data.py --stage extract`: extracts CSV members unchanged, validating archive/member integrity and matching source/extracted hashes.
- `acquire_2026_q1.py`: acquires or reuses verified Q1-2026 raw flights/weather; `--audit-only` verifies schedule structure and weather coverage without loading test outcome values.
- `audit_raw_data.py` (initial 2025 audit): scans the new project's raw sources for required fields, complete monthly flight coverage, project-station weather coverage and reference metadata. It writes fresh audit metadata only.
- `finalize_raw_setup.py` is the original 2025 setup publisher and writes the original configuration. The extended study is already published; rebuild current handoffs by running notebooks 01 then 02, rather than rerunning this legacy publisher.

Use Python 3.11+ with the packages in `requirements.txt`. A future modelling pipeline reads the local manifest; it does not require the earlier project's files. The one-time copy tool records the earlier raw source location only as acquisition provenance.

Do not rerun network downloads to change a source snapshot in the middle of a modelling experiment. Preserve the original source version and checksums for reproducibility.

- `acquire_2026_q2.py`: directly acquires and structurally audits Q2-2026 raw flight/weather sources, without examining test outcomes.
- `improvement_features.py`: reviewed weather decoding, extra weather/time features and cutoff-safe histories; no model fitting.
- `fresh_test_data.py`: applies the reviewed cleaning/feature rules to the new quarter after selection is frozen; returns one in-memory sample without saving split datasets.
- `improvement_exports.py`: builds the six versioned Tableau result tables from frozen scores.

- `audit_improvement_outputs.py`: recomputes exported metrics with independent sklearn scorers, checks curves/quotas/subgroups and reloads the saved models to verify scores. It does not refit classifiers.

## Notebook appendix integrity

`baseline_integrity.py` verifies the original baseline checksums, the earlier authorized notebook-02 appendix, and the exact verified preparation-notebook readability revisions. Readability acceptance requires intact original backups, unchanged saved outputs and matching support/verification hashes. Notebook 04's compatibility guards, notebook 05 and the final audit scripts use this check. The original fingerprint record is retained unchanged.

