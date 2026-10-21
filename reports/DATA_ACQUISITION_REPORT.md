# Raw data acquisition completed

See [the raw-data guide](../data/raw/README.md) for locations and provenance.

Downloaded all twelve full official 2025 BTS archives and extracted unchanged CSVs. Copied only original raw source files for weather, station metadata, airports and November/December 2024 context. No previous cleaned data was used.

Verified 7,001,619 raw 2025 rows and 1,165,985 context rows, all required source fields, source month coverage, three-station weather coverage and SHA256 identities. Raw storage is approximately 3.96 GiB. Required missing raw files: none.

Historical aircraft-assignment receipt timestamps are not recorded in BTS. Historical weather/outcome delivery receipts are also not established by these sources. The retrospective scenario/latency assumptions must remain explicit.

No cleaning or modelling was performed. The [machine-readable summary](DATA_ACQUISITION_SUMMARY.json) and [fresh raw validation](../data/raw/_provenance/raw_validation_report.json) contain the checked counts and limitations.
