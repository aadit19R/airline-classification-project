# Active raw data and archived sources

Current notebooks read 17 flight months: November–December 2024, all 2025, and May–July 2026. January–April 2026 flight/weather payloads, request metadata and flight source readmes are archived in `data/archive/2026_jan_apr/`. They are not required by the notebook run.

See [exact notebook input requirements](../../NOTEBOOK_DATA_REQUIREMENTS.md), [the input manifest](../../../data/NOTEBOOK_INPUTS.json), and [archive mapping/checksums](../../../data/archive/2026_jan_apr/archive_manifest.json).

The acquisition records below describe the historical acquisition layout. Their original paths are preserved for provenance; use the archive mapping to locate moved files.

## Historical raw data: start here

All required raw files are now stored inside this project's own `data/raw` directory. They have not been cleaned, filtered by flight, imputed, merged or converted into model features. Original downloads and source response bytes are retained.

## January–March 2026 extension

The earlier Q1 evaluation sources are available: [flight CSVs](flights/bts/2026/csv), [original archives](flights/bts/2026/archives) and [origin weather](weather/observations/2026). They contain 1,671,142 national rows and 228,586 ATL/ORD/DFW schedules. Each origin has weather observations on all 90 Q1 days, with boundary padding. See [the dataset guide](../../DATASET_GUIDE.md).

The acquisition structural audit loaded no test outcome values. The user subsequently approved excluding both versions at the two ambiguous 2026 weather timestamps, consistent with the six earlier conflicts. Q1 and Q2 2026 are earlier evaluation periods. The current final model trains on 40,000 all-2025 flights and tests on 10,000 July 2026 flights. Source tables use explicit `preparation_period` names; final train/test membership is stored in the notebook feature handoff.

## Ready-to-open source files

| Data | Location | Files |
|---|---|---:|
| Full 2025 flight records, including aircraft registrations | [flights/bts/2025/csv](flights/bts/2025/csv) | 12 CSVs |
| Original 2025 downloads | [flights/bts/2025/archives](flights/bts/2025/archives) | 12 ZIPs + retrieval metadata |
| November/December 2024 context | [flights/bts/2024_context/csv](flights/bts/2024_context/csv) | 2 CSVs |
| Original context archives | [flights/bts/2024_context/archives](flights/bts/2024_context/archives) | 2 ZIPs |
| Raw weather response batches covering the three origin stations | [weather/observations/2025](weather/observations/2025) | 36 CSVs + original request metadata |
| Original station/network reference responses | [weather/stations](weather/stations) | 55 GeoJSONs + request metadata |
| Original airport reference | [reference/airports/airports.csv](reference/airports/airports.csv) | 1 CSV |

For example, [2025_01.csv](flights/bts/2025/csv/2025_01.csv) is the unchanged official January source. The complete files use 110 source columns, including `Tail_Number`; they are wider than the previous project's 28-column inputs.

## What was acquired and checked

- **7,001,619** raw 2025 flight rows across all 12 months.
- **1,165,985** raw late-2024 context rows.
- Required flight fields, including aircraft registrations, present in every month.
- Weather observations for ATL, ORD and DFW present in every 2025 calendar month.
- Raw station metadata contains an IANA timezone for each of the three origins; fresh mappings still need to be built.
- Original copied raw-file checksums, ZIP/member integrity and extracted CSV byte identity verified.
- Approximately **3.96 GiB** stored, including compressed originals and extracted CSVs.

Weather batches include other source-request stations and overlapping boundary observations. These remain unchanged; the executed notebooks perform fresh filtering and preparation in derived tables. Missing values and source formats are preserved. Reference metadata is a source snapshot rather than proven 2025 historical metadata.

## Provenance and validation

See the [source manifest](../SOURCE_MANIFEST.json), [file checksums](_provenance/file_inventory.json), [raw validation report](_provenance/raw_validation_report.json), [direct download log](_provenance/bts_download_log.json), [raw-copy provenance](_provenance/raw_copy_log.json), and [unchanged extraction log](_provenance/bts_extraction_log.json).

No file from the old project's processed/interim/model outputs or generated timezone lookup was used. Existing raw airport, weather, station and 2024 archive files were copied byte-for-byte. The full 2025 archives were downloaded directly from BTS.

## Required missing files and limitations

**Required raw files still missing: none.**

BTS does not include a historical timestamp of when each aircraft assignment became known. Aircraft rotation remains excluded; final aircraft registrations do not establish assignment availability at the prediction cutoff. Actual historical delivery/receipt timestamps for weather and flight outcome feeds are also not supplied. These are documented information limits, not silently invented columns.

Use the [current notebook guide](../../notebooks/simple/README.md) for cleaning, feature engineering, fitting and final July evaluation.
