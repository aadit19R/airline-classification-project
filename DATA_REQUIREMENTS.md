# Data requirements and acquired sources

All required raw inputs are stored locally under [data/raw](data/raw/README.md). The 2025 BTS files were downloaded directly from the official source, including aircraft registrations. Only unmodified raw files and original source-request metadata were copied from the earlier project. No previous processed/interim/feature table, model output, combined station table or generated timezone lookup is used.

The approved scope uses 2025 development/final refit and January–March 2026 final testing for ATL, ORD and DFW, with a simulated two-hour cutoff. National flight records remain unfiltered raw sources; earlier departure outcomes provide history context. Aircraft rotation is outside the agreed model scope.

## 1. Flight records

Source: [BTS Reporting Carrier On-Time Performance](https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ). Files: [2025 CSVs](data/raw/flights/bts/2025/csv), [2025 archives](data/raw/flights/bts/2025/archives), [Q1-2026 CSVs](data/raw/flights/bts/2026/csv) and [Q1-2026 archives](data/raw/flights/bts/2026/archives).

| Raw source field | Role |
|---|---|
| FlightDate | Calendar, chronology, identity |
| Reporting_Airline | Airline predictor and identity |
| Flight_Number_Reporting_Airline | Identity and optional earlier punctuality |
| Origin, Dest | Airports, route and linkage |
| CRSDepTime | Scheduled departure predictor and prediction cutoff |
| CRSArrTime | Available schedule/audit field |
| CRSElapsedTime | Scheduled duration predictor |
| Distance | Distance predictor |
| DepDelay, DepDel15 | Outcome source/reconciliation: delayed departure >=15 minutes |
| Cancelled | Population/exclusion audit |
| Diverted | Status audit; known departure can remain |
| Tail_Number | Acquired source field; excluded from the agreed model predictors |
| DepTime, ArrTime, ArrDelay, ActualElapsedTime | Audit and earlier-event reconstruction only |

The source files contain the full official schema, rather than the earlier project's selected 28 aliases. Current-flight observed delays, actual times and cause minutes do not become pre-departure predictors. They are source records for labels or explicitly earlier events only.

November and December 2024 context is available in [this folder](data/raw/flights/bts/2024_context/csv), unchanged from original archives. The 28-day history ending seven days before a query needs context beginning 28 November 2024 for early January. Rebuild departure history from these raw sources in the new pipeline.

## 2. Origin weather observations

Source/units: [Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu/request/download.phtml). Files: [36 unmodified 2025 batches](data/raw/weather/observations/2025) and [3 Q1-2026 three-station responses](data/raw/weather/observations/2026), each with original request metadata and boundary padding.

| Raw fields | Use |
|---|---|
| station, valid | Station/time matching |
| sknt, gust | Wind speed/gust |
| vsby | Visibility |
| p01i | Precipitation, retaining source trace/missing codes |
| skyc1-skyc4, skyl1-skyl4 | Structured cloud-layer fallback; use all body layers when deriving ceiling |
| wxcodes | Derive rain/snow/fog/thunderstorm flags |
| metar | Full cloud extraction before RMK and raw-text auditing |
| tmpf | Optional temperature feature |
| dwpf, drct | Available optional dew point/wind direction |

The copied raw batches include ATL, ORD or DFW among their requested stations. Other stations, original missing values, boundary padding and overlapping observations remain untouched. The executed preparation notebooks filter these into new derived tables without altering the source files.

Select observations available by the prediction cutoff. Observation timestamps are known; actual historical receipt timestamps are not. A declared delivery-buffer scenario, initially 20 minutes and a maximum three-hour observation age, must be applied in the new pipeline. Archived forecasts are not required for the first observation-based B stage.

## 3. Airport/station reference and timezones

Source: [OurAirports](https://ourairports.com/data/). Local raw [airports.csv](data/raw/reference/airports/airports.csv) retains identifiers, name, coordinates, region/city, keywords and other original fields.

Original [IEM station/network GeoJSON responses](data/raw/weather/stations) retain station IDs, coordinates and `tzname`. The three query-origin stations all have raw IANA timezone values. Generate and verify airport/station/timezone mappings afresh from these raw reference inputs; OurAirports does not itself supply an IANA timezone column. No previous generated timezone cache is used. Airport reference snapshots and historical code changes require fresh join checks.

## 4. Derived features: freshly built in this project

- New binary departure label and query cutoff UTC.
- Month, weekday, route and departure time band/cyclic features.
- Earlier airline/origin/route departure-delay rates, support counts and missing/short-history indicators.
- Scheduled departure traffic around the query time.
- Weather ceiling/condition flags, observation age and missingness.
- Optional flight-number punctuality from earlier eligible outcomes.

These are newly built and checked in the executed preparation notebooks, rather than imported from the earlier project. History uses only outcomes within the declared earlier window under the documented retrospective-feed assumption. Optional flight-number punctuality is not included in A/B.

## 5. Aircraft rotation: excluded from the agreed project

`Tail_Number` exists in all twelve 2025 source files and both context months. These source bytes remain stored; excluding rotation requires no deletion or replacement download.

BTS records the final flown aircraft but not the historical time its assignment became known. Accordingly, do not build or use incoming-aircraft arrival status, previous-leg delays or turnaround features through that assignment in the agreed A/B experiment. No objective depends on them.

The current flight's `LateAircraftDelay` cause minutes are also excluded. Reopening aircraft rotation would require a separate scope decision and assignment-timing evidence. Weather/outcome receipt assumptions still need to be disclosed for B, and archived schedules do not reconstruct historical schedule revisions.

## 6. Tableau outputs: generate after modelling

Actual/predicted labels, scores/ranks, decision thresholds, confusion categories, algorithm/feature identifiers, model metrics, subgroup errors and capacity-ranking results are future pipeline outputs. No earlier predictions or cleaned export is reused. Follow [the export contract](tableau/export_contract.json).

Early-January 2026 rolling history uses existing late-2025 raw records. See [the Q1-2026 acquisition report](reports/DATA_ACQUISITION_2026_Q1.md). The notebooks and both handoffs have been rebuilt under the revised chronology, including a separate full-2025 refit sample. See [execution and integrity checks](reports/PREPARATION_TIMELINE_UPDATE.md).

## Status

Required raw files still missing: **none**. No user account/key/file is needed for the core raw sources. Historical assignment-time and feed-receipt evidence remains unavailable in these public source records and is documented rather than invented. See [the acquisition report](reports/DATA_ACQUISITION_REPORT.md) and [local source manifest](data/SOURCE_MANIFEST.json).

## Completed fresh confirmation sources

The improvement experiment adds untouched April–June 2026 BTS files in data/raw/flights/bts/2026/csv/2026_04.csv through 2026_06.csv, and three padded origin-weather responses in data/raw/weather/observations/2026/. Raw acquisition and structural coverage are recorded in data/raw/_provenance/2026_q2_acquisition_log.json and 2026_q2_validation_report.json. There are 260,336 scoped schedules with every source date present. Existing Q1 data supplies earlier history; final model fitting remains 2025 only. No additional intermediate datasets are saved.
