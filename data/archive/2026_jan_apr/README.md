# January–April 2026 source archive

These months were acquired for earlier Q1/Q2 evaluation experiments. They do not contribute observations to the current final feature windows. Current notebooks train on 2025 and evaluate July 2026, using sufficiently earlier May–July outcomes as historical context.

This archive contains 24 original files: four monthly flight CSVs, four original ZIPs, four ZIP-metadata JSON files, four source readmes, four weather CSVs and four weather-request JSON files. All source bytes are preserved. `flights_clean.parquet` additionally preserves the 314,085 cleaned schedules removed from the active handoff.

`archive_manifest.json` records every original/current path, byte size and SHA256 checksum. The archived fingerprints were checked when the files were moved; current model verification does not require this archive. The current notebooks do not read this directory and their acquisition code skips these months.

The archive supports historical investigations; it is not required for running the current notebooks. See [current notebook data requirements](../../../md/NOTEBOOK_DATA_REQUIREMENTS.md).
