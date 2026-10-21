# Q1 2026 raw data acquisition

Completed 2026-10-18T19:41:01.494550+00:00. This extends the project to 2025 development/final refit and January–March 2026 final testing. Sources are unmodified downloads from [BTS](https://transtats.bts.gov/PREZIP/) and [IEM](https://mesonet.agron.iastate.edu/request/download.phtml). No previous processed project data was reused.

## Ready-to-access files

| Data | Folder | Files |
|---|---|---:|
| Unmodified monthly flight CSVs | [data/raw/flights/bts/2026/csv](../data/raw/flights/bts/2026/csv) | 3 |
| Official compressed originals and request metadata | [data/raw/flights/bts/2026/archives](../data/raw/flights/bts/2026/archives) | 3 ZIPs + metadata |
| Origin weather responses and request metadata | [data/raw/weather/observations/2026](../data/raw/weather/observations/2026) | 3 CSVs + metadata |

Flight names: 2026_01.csv, 2026_02.csv and 2026_03.csv. Weather names: 2026_01_origins.csv, 2026_02_origins.csv and 2026_03_origins.csv. Each weather file contains ATL, ORD and DFW. These are natural monthly source requests, not processed intermediate tables.

## Verified coverage

| Month | National raw flight rows | Three-origin scheduled rows | Calendar days present |
|---|---:|---:|---:|
| 2026-01 | 544,003 | 73,955 | 31 |
| 2026-02 | 515,037 | 70,581 | 28 |
| 2026-03 | 612,102 | 84,050 | 31 |

- **1,671,142** national raw flight rows and **228,586** three-origin schedules, before outcome-based eligibility exclusions.
- All required source fields are present. Each month has 110 columns matching the 2025 schema.
- Archive CRC, SHA-256 and extracted-member byte-identity checks passed.
- **8,995** raw weather rows, including padding and overlaps; all **90 Q1 UTC days** have observations at each origin.
- UTC requests use report types 3/4 and preserve source M/T codes, existing weather variables and full METAR text.
- Each monthly request has two-day padding at both ends for local/UTC boundaries, earlier observations and the two-hour horizon.
- Qualifying weather keys exist for **228,586/228,586** schedules under the existing buffer/age rules. Ages range **20–139 minutes**. This is key existence only; no ambiguous payload version was selected.
- Repeated schedule identities: **0**; invalid clocks: **0**; unresolved UTC timestamps: **0**.
- Cross-year weather snapshots share **346** timestamp keys; **0** show changed/conflicting payloads.
- New raw payloads, metadata and source documentation use approximately **800.7 MiB**.

## Preparation review (resolved after acquisition)

- The two groups below were subsequently approved for the same exclusion policy as the six 2025 conflicts. Both versions are excluded from prepared weather; all raw versions remain stored.

| Station | Timestamp UTC | Raw report versions |
|---|---|---:|
| ATL | 2026-01-06 08:50 | 2 |
| ORD | 2026-02-20 08:00 | 2 |

As a diagnostic only, excluding ambiguous keys still leaves a qualifying earlier weather key for **228,586** schedules. The user subsequently approved that policy, and it is implemented in the updated feature notebook.

**Required source files missing: none.** No login, key or user-supplied file is needed. Existing full November/December 2025 records supply early-2026 history and traffic warm-up. Existing project raw airport/station/timezone references cover the three origins.

Historical weather/outcome receipt times and schedule availability remain the existing simulation assumptions. Aircraft rotation remains excluded.

## Holdout protection

The audit loads only schedule/date/identity fields from flight CSVs. Outcome columns are checked for presence, but **no test outcome values, delay rates, cancellation rates or predictive relationships are inspected**. The raw archives retain all official outcome fields for later frozen label construction and final evaluation.

## Approved timeline and preparation status

| Stage | Period |
|---|---|
| Initial fitting | January–September 2025 |
| Validation / model, feature and threshold selection | October–December 2025 |
| Final refit after choices are frozen | January–December 2025 |
| Final testing | January–March 2026 |

The approved timeline is implemented in split and evaluation_extension in [config/project.json](../config/project.json). Both notebooks were subsequently rerun and both Parquet handoffs rebuilt, with new shared sample membership and a separate 20,000-flight full-2025 refit flag. See [the preparation update](PREPARATION_TIMELINE_UPDATE.md). The acquisition-only no-outcome-values audit above describes the source audit before frozen label construction.

Learned preprocessing remains fitted only on the appropriate training cohort during selection. Finalized pipelines are refitted on the declared 2025 cohort after choices are frozen. Q1-2026 outcomes must not guide those choices; the final findings describe Q1 rather than every 2026 season.

See [acquisition provenance](../data/raw/_provenance/2026_q1_acquisition_log.json), [structural validation](../data/raw/_provenance/2026_q1_validation_report.json), [source manifest](../data/SOURCE_MANIFEST.json) and [checksums](../data/raw/_provenance/file_inventory.json).
