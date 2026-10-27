# Dataset reference: departure-delay classification

Source counts were audited against local schemas, acquisition records and final experiment records on 20 October 2026. Current split metadata is verified in `reports/simple_reproduction/verification.json` and its migration evidence in `reports/study_labels.json`. This guide describes the data itself, its coverage and its role. It contains no EDA patterns or model performance findings. Counts describe the locally acquired snapshots, not the continually updated source websites.

## 1. What the project uses

There are four raw source collections, from three providers. A flight record is the main observation; weather reports and reference records are linked to it. The model-ready rows are newly derived project data, not a downloaded ready-made classification dataset.

| Collection | Provider | One record represents | Local files |
| --- | --- | --- | --- |
| Flight schedules and outcomes | US Department of Transportation, Bureau of Transportation Statistics (BTS), Reporting Carrier On-Time Performance | One reported scheduled flight leg on a date, including cancelled flights | 21 monthly CSVs and their original ZIP archives |
| Airport weather observations | Iowa Environmental Mesonet (IEM), ASOS/AWOS/METAR archive | One station's observation/report at a timestamp | 43 CSV response batches |
| Weather-station reference | IEM network metadata | One weather station within a network response | 55 GeoJSON files; this means 55 response files, not 55 stations |
| Airport reference | OurAirports | One airport or other listed aviation facility | One CSV, 86,053 records and 19 columns |

The BTS source concerns reporting carriers' nonstop domestic services. It is not a worldwide flight census. Reporting-carrier identities concern the reporting operator; they are not automatically a complete marketed airline network. See the [BTS database profile](https://www.transtats.bts.gov/DatabaseInfo.asp?DB_URL=&QO_VQ=EFD).

The project's prediction scope is departures from Atlanta (ATL), Dallas Fort Worth (DFW) and Chicago O'Hare (ORD). The 2025 scheduled subset has 12 reporting-airline codes and 240 destination codes. Those counts describe this three-origin subset, not the whole national source. The airline codes are AA, AS, B6, DL, F9, MQ, NK, OH, OO, UA, WN and YX.

## 2. Flight periods and exact record counts

| Period | National raw rows | Schedules from ATL/ORD/DFW | Eligible departure-label population | Purpose |
| --- | ---: | ---: | ---: | --- |
| November-December 2024 | 1,165,985 | 153,143 | 151,884 history events | Earlier history context only |
| January-December 2025 | 7,001,619 | 957,315 | 938,417 | Development, validation and final fitting |
| January-March 2026 | 1,671,142 | 228,586 | 219,748 | Original evaluated test/benchmark |
| April-June 2026 | 1,817,231 | 260,336 | 254,333 | Evaluated improvement benchmark; later history context |
| July 2026 | 631,970 | 89,701 | 87,298 | Fresh final confirmation after the procedure was frozen |
| Total acquired | 12,287,947 | 1,689,081 | Roles differ; do not combine as one training set | 21 months of raw flight records |

National files are retained unchanged. Filtering to three origins happens in the project pipeline. A national file's row count is therefore different from the project population or model sample size. These raw snapshots cover November 2024 through July 2026; no August-December 2026 flight data is part of this project.

January 2025 needs earlier flight history. The history rule looks back over a 28-day window ending seven days before the prediction cutoff. Its earliest January windows reach late November 2024. These earlier months are context, not additional 2025 training examples.

**Current split: 40,000 training flights across January-December 2025 and 10,000 test flights from July 2026.** Model selection uses chronological rounds inside 2025. Q1 and Q2 2026 are earlier evaluated benchmarks, not the current test set. No 2026 flights enter model fitting. Earlier 2026 outcomes can enter later history features under the seven-day lag. July is now exposed and must not be used for new model choices.

## 3. Flight file structure, variables and units

The exact source name is **Reporting Carrier On-Time Performance (1987-present)**. The source database's long historical coverage does not mean all those years were downloaded here. Original ZIPs were obtained from [BTS PREZIP](https://transtats.bts.gov/PREZIP/), extracted without editing the CSV contents, and stored with retrieval metadata/checksums.

Each local monthly CSV has 110 header positions: **109 named fields and one blank trailing field** caused by the source's trailing delimiter. The blank field is not a real variable. A reader may call it `Unnamed: 109`. The full raw schema is preserved, although only a subset is used for the project.

| Variable family | Principal fields | Meaning and project role |
| --- | --- | --- |
| Date | FlightDate, Year, Quarter, Month, DayofMonth, DayOfWeek | Flight date/calendar and chronological membership |
| Carrier | Reporting_Airline, DOT_ID_Reporting_Airline, IATA_CODE_Reporting_Airline | Carrier identifiers; Reporting_Airline is the project's airline field |
| Flight and aircraft | Flight_Number_Reporting_Airline, Tail_Number | Flight service identifier and aircraft registration; registration is not aircraft model/type |
| Route and geography | Origin, Dest, airport/city/state identifiers | Origin-destination linkage and geographic context |
| Schedule | CRSDepTime, CRSArrTime, CRSElapsedTime | Scheduled local clock times and gate-to-gate duration in minutes |
| Distance | Distance, DistanceGroup | Airport distance in miles and a distance grouping |
| Departure outcome | DepTime, DepDelay, DepDelayMinutes, DepDel15 | Actual departure and departure delay; target/audit information, not current-flight predictors |
| Arrival and taxi outcomes | ArrTime, ArrDelay, TaxiOut, TaxiIn, WheelsOff, WheelsOn, AirTime, ActualElapsedTime | Realized operations; excluded as current-flight predictors |
| Status | Cancelled, CancellationCode, Diverted | Defines population and status audit; known-departure diverted flights can remain |
| Delay causes | CarrierDelay, WeatherDelay, NASDelay, SecurityDelay, LateAircraftDelay | Retrospective cause minutes; excluded as current-flight predictors |
| Gate return and diversion detail | FirstDepTime, TotalAddGTime, Div1...Div5 fields | Wider raw operational details; not used to predict this flight before departure |

`CRS` refers to the Computer Reservation System schedule. Clock fields are local HHMM values, not decimal hours: for example, 0930 means 09:30. Origin and destination clocks can be in different time zones. The project uses UTC internally for time comparisons, using America/New_York for ATL and America/Chicago for ORD/DFW, including daylight-saving changes.

`DepDelay` is signed: early departures can be negative. `DepDelayMinutes` instead sets early departure delay to zero. `DepDel15` is the source's at-least-15-minute departure indicator. Departure here concerns leaving the gate; `WheelsOff` is a separate takeoff event. See [BTS field definitions](https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ).

The classification label is rebuilt as `delayed_departure = 1` when `DepDelay >= 15`, and 0 otherwise, for noncancelled flights with a known departure outcome. Zero includes early departures and departures less than 15 minutes late. Cancelled/unknown-outcome flights do not receive an invented on-time label. The label is about departure delay conditional on operation, not a combined cancellation/disruption label. Cancelled schedules can still contribute schedule-context counts.

Flight number alone is not unique: it repeats on different dates and may repeat across carriers. The project uses a stable `flight_id` derived from source month and original row position, plus a separate schedule-identity check. Source rows are therefore traceable.

The prediction cutoff is two hours before scheduled departure. A raw column can exist in the archive without being a permissible predictor at that cutoff. Current-flight actual times, delays and cause fields are labels/audit material only. `Tail_Number` is retained raw but excluded: the archive does not say when the aircraft assignment became known. It cannot justify a verified predeparture rotation feature.

## 4. Weather observations

Weather comes from the [IEM ASOS/AWOS/METAR archive](https://mesonet.agron.iastate.edu/request/download.phtml), an archive of aviation weather reports. These are observations, not forecasts. IEM's [dataset documentation](https://mesonet.agron.iastate.edu/info/datasets/metar.html) explains its upstream sources and processing.

| Local weather collection | Files | Raw response rows | What the count includes |
| --- | ---: | ---: | --- |
| 2025 batches | 36 | 756,216 | Other requested stations as well as ATL/ORD/DFW, plus overlap/padding |
| January-March 2026 | 3 | 8,995 | The three origins with boundary padding and overlap |
| April-June 2026 | 3 | 8,706 | The three origins with boundary padding and overlap |
| July 2026 | 1 | 2,887 | The three origins with boundary padding |
| Total stored weather responses | 43 | 776,804 | Not a count of unique observations at the three origins |

There are 19 columns per weather CSV. The 2026 requests include routine and special reports, with two-day padding around month boundaries. Report spacing is irregular, not one guaranteed row per station-hour. Padding allows a flight near a month boundary to find an earlier observation; overlapping responses can contain the same report more than once. Flights and weather have different row meanings, so their row counts should not match.

| Raw column(s) | Meaning / units |
| --- | --- |
| station | Weather-station identifier |
| valid | Observation timestamp; downloaded UTC in these requests |
| tmpf | Air temperature, degrees Fahrenheit |
| dwpf | Dew-point temperature, degrees Fahrenheit |
| drct | Wind direction, degrees from true north |
| sknt | Wind speed, knots |
| gust | Wind gust, knots |
| vsby | Visibility, miles |
| p01i | Precipitation accumulation, inches; the interval ends at the report and starts at the previous hourly reset |
| skyc1, skyc2, skyc3, skyc4 | Reported cloud-layer coverage codes |
| skyl1, skyl2, skyl3, skyl4 | Corresponding cloud-layer heights, feet |
| wxcodes | Present-weather condition codes |
| metar | Full raw weather message as text |

`M` represents missing/unreported/quality-controlled values; `T` is the requested trace-precipitation code. These are source conventions, not numeric measurements. Detailed source definitions are on the IEM download page. Altimeter pressure was later extracted from `metar`; there is no separate downloaded `alti` column in these CSVs.

Weather is matched by origin station and time. The chosen report is the latest qualifying earlier observation: at least 20 minutes before the two-hour prediction cutoff and at most 180 minutes old relative to that cutoff. For an illustrative 14:00 departure, the cutoff is 12:00, so the report timestamp must be between 09:00 and 11:40. This illustrates the rule, not an observed flight.

The same weather report can serve several flights. Observation time is supplied, but actual historical receipt time is not; the 20-minute buffer is an explicit assumption. Historical weather revisions/overlapping response versions are preserved raw and resolved only in preparation under the documented rules.

## 5. Reference datasets

### IEM station/network metadata

The 55 GeoJSON responses are stored in `data/raw/weather/stations/`. A GeoJSON file contains station features, each with an identifier, properties and point geometry. Relevant fields include `sid`, `network`, station name, state/country, elevation, `tzname`, and longitude/latitude in the geometry. Networks for the three origins are GA_ASOS (ATL), IL_ASOS (ORD) and TX_ASOS (DFW). These source responses supplied fresh airport/station/timezone mappings; a previous project's generated timezone table was not reused.

Reference metadata is a downloaded snapshot. It does not prove every property had the same value on every historical flight date. Metadata files are not additional flight or weather-measurement examples.

### OurAirports airport reference

The local `data/raw/reference/airports/airports.csv` has **86,053 rows and 19 columns**. It is a broad airport/facility reference, not 86,053 airports included in this three-origin classification study. The snapshot contains identifiers, facility type, name, coordinates, elevation, country/region, municipality, scheduled-service status and reference links. Coordinates/elevation support wider route-feature candidates. It does not provide an IANA timezone column. Source: [OurAirports downloads](https://ourairports.com/data/).

All airport-reference columns, in file order:

`id`, `ident`, `type`, `name`, `latitude_deg`, `longitude_deg`, `elevation_ft`, `continent`, `iso_country`, `iso_region`, `municipality`, `scheduled_service`, `icao_code`, `iata_code`, `gps_code`, `local_code`, `home_link`, `wikipedia_link`, `keywords`.

## 6. Raw files, prepared datasets and model samples are different

| Item | Rows | Columns | Coverage and role |
| --- | ---: | ---: | --- |
| `data/processed/flights_clean.parquet` | 1,689,081 | 33 | Complete scoped schedules, November 2024-July 2026; saved by notebook 01 |
| `data/processed/features.parquet` | 1,158,165 | 55 | Eligible 2025 and Q1-2026 query flights; original predictors, label and traceability fields |
| `reports/simple_reproduction/features.parquet` | 50,000 | 109 | Final model inputs rebuilt by notebook 02; train=40,000 all-2025 flights, test=10,000 July flights |

The clean table includes 153,143 history-context schedules, 957,315 2025 schedules, 228,586 Q1-2026 schedules, 260,336 Q2 schedules and 89,701 July schedules. The features table contains 938,417 eligible 2025 flights plus 219,748 eligible Q1-2026 flights. It excludes history-only rows as prediction queries. These are two stages of the same project, not independent source datasets.

The cleaned schedule handoff covers every acquired month through July. `preparation_period` describes source periods, not final training/test membership. The older 55-column feature table is retained for published EDA/report reproduction; its preparation sample flags are historical. Notebook training reads only the new 50,000-row input table, with frozen cohort IDs from JSON records.

The final notebook handoff, `reports/simple_reproduction/features.parquet`, has one authoritative `split`: `train` for all 40,000 selected 2025 flights and `test` for all 10,000 selected July flights. It has no competing `role` field. Exact fitting IDs remain in `reports/final_models/cohorts.json`. The earlier label migration is recorded in `reports/study_labels.json`; the self-contained notebook refactor and pre-refactor feature fingerprints are recorded in `reports/notebook_refactor.json`. Selected predictor values, labels and row membership are unchanged.

The 55 stored feature-table columns are not 55 predictors: the original B set has 37 predictors, while the rest are labels, identifiers, sample flags and audit timestamps. The original A set has 10 predictors. After later additions, the final A set has 26 input columns. Selected B has 78 for Logistic Regression/KNN, 66 for SVM and 72 for Random Forest. These counts are before one-hot expansion; an encoded numeric matrix can have more columns. See Appendix C for exact final input lists.

The original January-September EDA population has 698,670 eligible flights. Initial fitting used a shared 20,000-flight sample. Final fitting uses a shared 40,000-flight sample spanning all twelve 2025 months, sampled proportionally by month and outcome. The July final evaluation uses a shared 10,000-flight sample from its 87,298 eligible departures. Sample sizes do not replace the full stored populations. All four models use identical fitting/evaluation IDs at the relevant comparison stage, and natural class proportions are retained. No 50/50 class balancing dataset was created.

Parquet is a typed, column-oriented binary table format: smaller/faster for these Python datasets than repeatedly saving wide CSVs. It is not a different source of data. The shared preparation JSON manifest is the log describing how handoffs were made, not another flight dataset.

## 7. Availability, provenance and boundaries

- All active raw inputs are inside this new project's `data/raw/` folder. Only unchanged raw files and original source-response metadata were copied from earlier work; no earlier cleaned, imputed, feature or prediction tables were imported.
- Full 2025 flight archives were independently downloaded from BTS. Later 2026 files were also acquired from BTS. Raw airport/weather/station/context files reused from the earlier project were copied as source bytes, then prepared afresh here.
- Original archives, extracted CSVs, request metadata, source URLs, retrieval timestamps and checksums are retained under the relevant raw folders and `data/raw/_provenance/`. They document these snapshots; future downloads can be revised by their providers.
- No extra required source file is missing for the agreed experiment. Historical aircraft-assignment timing, actual weather/outcome-feed receipt times and historical schedule revisions are not established by these public files.
- The sources do not contain ticket prices, passenger identities, bookings, passenger counts, live gate queues, verified real-time ATC restrictions or aircraft-model specifications for this project. Do not infer them from flight number, tail registration or airport identifiers.
- Weather is origin weather. It is not a complete en-route/destination forecast or an incoming-aircraft state feed. Traffic features come from archived schedules, rather than direct runway/terminal capacity sensors.

## 8. Where to open the data

All paths below are relative to the project root `departure-delay-classification/`.

| Purpose | Location |
| --- | --- |
| 2024 context CSVs | [data/raw/flights/bts/2024_context/csv/](../data/raw/flights/bts/2024_context/csv/) |
| Full 2025 flight CSVs | [data/raw/flights/bts/2025/csv/](../data/raw/flights/bts/2025/csv/) |
| January-July 2026 flight CSVs | [data/raw/flights/bts/2026/csv/](../data/raw/flights/bts/2026/csv/) |
| Original BTS field descriptions | [July 2026 source readme](../data/raw/flights/bts/2026/source_documentation/2026_07_readme.html) |
| Weather CSV responses | [data/raw/weather/observations/](../data/raw/weather/observations/) |
| Weather-station metadata | [data/raw/weather/stations/](../data/raw/weather/stations/) |
| Airport reference | [data/raw/reference/airports/airports.csv](../data/raw/reference/airports/airports.csv) |
| Original preparation handoffs | [data/processed/](../data/processed/) |
| Original preparation rules/log | [reports/preparation_manifest.json](../reports/preparation_manifest.json) |
| Original acquisition source registry | [data/SOURCE_MANIFEST.json](../data/SOURCE_MANIFEST.json) |
| Q2 acquisition details | [data/raw/_provenance/2026_q2_validation_report.json](../data/raw/_provenance/2026_q2_validation_report.json) |
| July acquisition details | [data/raw/_provenance/2026_july_validation_report.json](../data/raw/_provenance/2026_july_validation_report.json) |
| Final cohort definitions | [reports/final_models/cohorts.json](../reports/final_models/cohorts.json) |
| Final feature definitions | [Feature Engineering notebook](../notebooks/simple/02_feature_engineering.ipynb) |

Some older source guides/configuration notes describe the original Q1 experiment. This reference consolidates later Q2/July acquisitions without rewriting those frozen records.

## Appendix A. All 109 named raw BTS fields

The groups below list every named CSV field exactly once. The unnamed 110th header position is deliberately excluded. Full source descriptions are retained in the BTS source readme linked above.

**Calendar (6):** `Year`, `Quarter`, `Month`, `DayofMonth`, `DayOfWeek`, `FlightDate`.

**Airline and aircraft (5):** `Reporting_Airline`, `DOT_ID_Reporting_Airline`, `IATA_CODE_Reporting_Airline`, `Tail_Number`, `Flight_Number_Reporting_Airline`.

**Origin (9):** `OriginAirportID`, `OriginAirportSeqID`, `OriginCityMarketID`, `Origin`, `OriginCityName`, `OriginState`, `OriginStateFips`, `OriginStateName`, `OriginWac`.

**Destination (9):** `DestAirportID`, `DestAirportSeqID`, `DestCityMarketID`, `Dest`, `DestCityName`, `DestState`, `DestStateFips`, `DestStateName`, `DestWac`.

**Departure (9):** `CRSDepTime`, `DepTime`, `DepDelay`, `DepDelayMinutes`, `DepDel15`, `DepartureDelayGroups`, `DepTimeBlk`, `TaxiOut`, `WheelsOff`.

**Arrival (9):** `WheelsOn`, `TaxiIn`, `CRSArrTime`, `ArrTime`, `ArrDelay`, `ArrDelayMinutes`, `ArrDel15`, `ArrivalDelayGroups`, `ArrTimeBlk`.

**Cancellation and diversion status (3):** `Cancelled`, `CancellationCode`, `Diverted`.

**Duration and distance (6):** `CRSElapsedTime`, `ActualElapsedTime`, `AirTime`, `Flights`, `Distance`, `DistanceGroup`.

**Delay causes (5):** `CarrierDelay`, `WeatherDelay`, `NASDelay`, `SecurityDelay`, `LateAircraftDelay`.

**Gate return (3):** `FirstDepTime`, `TotalAddGTime`, `LongestAddGTime`.

**Diversion summary (5):** `DivAirportLandings`, `DivReachedDest`, `DivActualElapsedTime`, `DivArrDelay`, `DivDistance`.

**Diversion stops 1-5 (40):** `Div1Airport`, `Div1AirportID`, `Div1AirportSeqID`, `Div1WheelsOn`, `Div1TotalGTime`, `Div1LongestGTime`, `Div1WheelsOff`, `Div1TailNum`, `Div2Airport`, `Div2AirportID`, `Div2AirportSeqID`, `Div2WheelsOn`, `Div2TotalGTime`, `Div2LongestGTime`, `Div2WheelsOff`, `Div2TailNum`, `Div3Airport`, `Div3AirportID`, `Div3AirportSeqID`, `Div3WheelsOn`, `Div3TotalGTime`, `Div3LongestGTime`, `Div3WheelsOff`, `Div3TailNum`, `Div4Airport`, `Div4AirportID`, `Div4AirportSeqID`, `Div4WheelsOn`, `Div4TotalGTime`, `Div4LongestGTime`, `Div4WheelsOff`, `Div4TailNum`, `Div5Airport`, `Div5AirportID`, `Div5AirportSeqID`, `Div5WheelsOn`, `Div5TotalGTime`, `Div5LongestGTime`, `Div5WheelsOff`, `Div5TailNum`.

## Appendix B. Stored Parquet schemas

Actual stored column names and types. The 55-column table is a retained published reference.

### Cleaned source context: 1,689,081 rows, 33 columns

`data/processed/flights_clean.parquet`

| Column | Stored type |
|---|---|
| `flight_date` | `timestamp[ns]` |
| `airline` | `large_string` |
| `flight_number` | `large_string` |
| `origin` | `large_string` |
| `destination` | `large_string` |
| `scheduled_departure_hhmm` | `double` |
| `departure_delay_minutes` | `double` |
| `departure_delay_15_source` | `double` |
| `cancelled` | `double` |
| `diverted` | `double` |
| `scheduled_duration_minutes` | `double` |
| `distance_miles` | `double` |
| `source_file` | `large_string` |
| `source_row_number` | `int64` |
| `flight_id` | `large_string` |
| `label_eligible` | `bool` |
| `delayed_departure` | `int8` |
| `scheduled_departure_minutes` | `int16` |
| `scheduled_departure_utc` | `timestamp[ns, tz=UTC]` |
| `prediction_cutoff_utc` | `timestamp[ns, tz=UTC]` |
| `departure_outcome_event_utc` | `timestamp[ns, tz=UTC]` |
| `preparation_period` | `large_string` |
| `preparation_query_eligible` | `bool` |
| `distance_miles_invalid` | `int8` |
| `scheduled_duration_minutes_invalid` | `int8` |
| `month` | `int8` |
| `weekday` | `int8` |
| `departure_hour` | `int8` |
| `departure_time_band` | `large_string` |
| `departure_sin` | `double` |
| `departure_cos` | `double` |
| `month_sin` | `double` |
| `month_cos` | `double` |

### Published EDA reference: 1,158,165 rows, 55 columns

`data/processed/features.parquet`

| Column | Stored type |
|---|---|
| `flight_id` | `large_string` |
| `source_file` | `large_string` |
| `source_row_number` | `int64` |
| `flight_date` | `timestamp[us]` |
| `preparation_period` | `large_string` |
| `preparation_sample` | `bool` |
| `initial_refit_sample` | `bool` |
| `month` | `int8` |
| `scheduled_departure_utc` | `timestamp[ns, tz=UTC]` |
| `prediction_cutoff_utc` | `timestamp[ns, tz=UTC]` |
| `departure_time_band` | `large_string` |
| `delayed_departure` | `int8` |
| `weather_lookup_utc` | `timestamp[ns, tz=UTC]` |
| `weather_observed_utc` | `timestamp[ns, tz=UTC]` |
| `weather_source_file` | `large_string` |
| `history_origin_latest_event_utc` | `timestamp[ns, tz=UTC]` |
| `history_airline_origin_latest_event_utc` | `timestamp[ns, tz=UTC]` |
| `history_route_latest_event_utc` | `timestamp[ns, tz=UTC]` |
| `airline` | `large_string` |
| `origin` | `large_string` |
| `destination` | `large_string` |
| `weekday` | `int8` |
| `distance_miles` | `double` |
| `scheduled_duration_minutes` | `double` |
| `departure_sin` | `double` |
| `departure_cos` | `double` |
| `month_sin` | `double` |
| `month_cos` | `double` |
| `temperature_f` | `double` |
| `wind_knots` | `double` |
| `gust_knots` | `double` |
| `visibility_miles` | `double` |
| `precip_inches` | `double` |
| `precip_trace` | `double` |
| `ceiling_ft` | `double` |
| `no_ceiling_reported` | `double` |
| `cloud_coverage_missing` | `double` |
| `rain_reported` | `double` |
| `frozen_precip_reported` | `double` |
| `fog_or_mist_reported` | `double` |
| `thunder_reported` | `double` |
| `condition_codes_missing` | `double` |
| `ceiling_height_missing` | `double` |
| `weather_age_minutes` | `double` |
| `weather_report_missing` | `int8` |
| `history_origin_rate` | `double` |
| `history_origin_n` | `int32` |
| `history_origin_no_history` | `int8` |
| `history_airline_origin_rate` | `double` |
| `history_airline_origin_n` | `int32` |
| `history_airline_origin_no_history` | `int8` |
| `history_route_rate` | `double` |
| `history_route_n` | `int32` |
| `history_route_no_history` | `int8` |
| `airport_scheduled_departures_prior_60m` | `int32` |

### Current model inputs: 50,000 rows, 109 columns

`reports/simple_reproduction/features.parquet`

| Column | Stored type |
|---|---|
| `flight_date` | `timestamp[ns]` |
| `airline` | `large_string` |
| `flight_number` | `large_string` |
| `origin` | `large_string` |
| `destination` | `large_string` |
| `scheduled_departure_hhmm` | `double` |
| `departure_delay_minutes` | `double` |
| `departure_delay_15_source` | `double` |
| `cancelled` | `double` |
| `diverted` | `double` |
| `scheduled_duration_minutes` | `double` |
| `distance_miles` | `double` |
| `source_file` | `large_string` |
| `source_row_number` | `int64` |
| `flight_id` | `large_string` |
| `label_eligible` | `bool` |
| `delayed_departure` | `int8` |
| `scheduled_departure_minutes` | `int16` |
| `scheduled_departure_utc` | `timestamp[ns, tz=UTC]` |
| `prediction_cutoff_utc` | `timestamp[ns, tz=UTC]` |
| `departure_outcome_event_utc` | `timestamp[ns, tz=UTC]` |
| `preparation_period` | `large_string` |
| `preparation_query_eligible` | `bool` |
| `distance_miles_invalid` | `int8` |
| `scheduled_duration_minutes_invalid` | `int8` |
| `month` | `int8` |
| `weekday` | `int8` |
| `departure_hour` | `int8` |
| `departure_time_band` | `large_string` |
| `departure_sin` | `double` |
| `departure_cos` | `double` |
| `month_sin` | `double` |
| `month_cos` | `double` |
| `split` | `large_string` |
| `departure_sin2` | `double` |
| `departure_cos2` | `double` |
| `month_sin2` | `double` |
| `month_cos2` | `double` |
| `atl_departure_sin` | `double` |
| `atl_departure_cos` | `double` |
| `atl_departure_sin2` | `double` |
| `atl_departure_cos2` | `double` |
| `ord_departure_sin` | `double` |
| `ord_departure_cos` | `double` |
| `ord_departure_sin2` | `double` |
| `ord_departure_cos2` | `double` |
| `dfw_departure_sin` | `double` |
| `dfw_departure_cos` | `double` |
| `dfw_departure_sin2` | `double` |
| `dfw_departure_cos2` | `double` |
| `weather_lookup_utc` | `timestamp[ns, tz=UTC]` |
| `weather_observed_utc` | `timestamp[ns, tz=UTC]` |
| `dewpoint_spread_f` | `double` |
| `wind_east_knots` | `double` |
| `wind_north_knots` | `double` |
| `freezing_wet` | `double` |
| `prior3h_report_n` | `double` |
| `prior3h_precip_known_n` | `double` |
| `prior3h_wet_fraction` | `double` |
| `prior3h_visibility_min` | `double` |
| `prior3h_gust_max` | `double` |
| `prior3h_visibility_change` | `double` |
| `temperature_f` | `double` |
| `wind_knots` | `double` |
| `gust_knots` | `double` |
| `visibility_miles` | `double` |
| `precip_inches` | `double` |
| `precip_trace` | `double` |
| `ceiling_ft` | `double` |
| `no_ceiling_reported` | `double` |
| `cloud_coverage_missing` | `double` |
| `rain_reported` | `double` |
| `frozen_precip_reported` | `double` |
| `fog_or_mist_reported` | `double` |
| `thunder_reported` | `double` |
| `condition_codes_missing` | `double` |
| `ceiling_height_missing` | `double` |
| `weather_source_file` | `large_string` |
| `weather_age_minutes` | `double` |
| `weather_report_missing` | `int8` |
| `history_origin_rate` | `double` |
| `history_origin_n` | `int32` |
| `history_origin_no_history` | `int8` |
| `history_origin_latest_event_utc` | `timestamp[ns, tz=UTC]` |
| `history_airline_origin_rate` | `double` |
| `history_airline_origin_n` | `int32` |
| `history_airline_origin_no_history` | `int8` |
| `history_airline_origin_latest_event_utc` | `timestamp[ns, tz=UTC]` |
| `history_route_rate` | `double` |
| `history_route_n` | `int32` |
| `history_route_no_history` | `int8` |
| `history_route_latest_event_utc` | `timestamp[ns, tz=UTC]` |
| `history_flight_rate` | `double` |
| `history_flight_n` | `int32` |
| `history_flight_no_history` | `int8` |
| `history_flight_latest_event_utc` | `timestamp[ns, tz=UTC]` |
| `airport_scheduled_departures_prior_60m` | `int32` |
| `altimeter_inhg` | `double` |
| `altimeter_change3h_inhg` | `double` |
| `altimeter_range3h_inhg` | `double` |
| `variable_wind_flag` | `double` |
| `gust_excess_knots` | `double` |
| `altimeter_missing_flag` | `int8` |
| `temperature_centered_square` | `double` |
| `wind_square` | `double` |
| `dewpoint_spread_square` | `double` |
| `precip_log1p` | `double` |
| `ceiling_log1p` | `double` |
| `visibility_inverse` | `double` |

## Appendix C. Final input columns

A is the schedule control. B is the selected richer package for each algorithm; its additional columns are constructed from earlier raw information. These are predictors, not labels or output predictions. Counts precede one-hot encoding.

**A: 26 input columns.** `airline`, `origin`, `destination`, `weekday`, `distance_miles`, `scheduled_duration_minutes`, `departure_sin`, `departure_cos`, `month_sin`, `month_cos`, `departure_sin2`, `departure_cos2`, `month_sin2`, `month_cos2`, `atl_departure_sin`, `atl_departure_cos`, `atl_departure_sin2`, `atl_departure_cos2`, `ord_departure_sin`, `ord_departure_cos`, `ord_departure_sin2`, `ord_departure_cos2`, `dfw_departure_sin`, `dfw_departure_cos`, `dfw_departure_sin2`, `dfw_departure_cos2`.

**current_66: 66 input columns.** `airline`, `origin`, `destination`, `weekday`, `distance_miles`, `scheduled_duration_minutes`, `departure_sin`, `departure_cos`, `month_sin`, `month_cos`, `temperature_f`, `wind_knots`, `gust_knots`, `visibility_miles`, `precip_inches`, `precip_trace`, `ceiling_ft`, `no_ceiling_reported`, `cloud_coverage_missing`, `rain_reported`, `frozen_precip_reported`, `fog_or_mist_reported`, `thunder_reported`, `condition_codes_missing`, `ceiling_height_missing`, `weather_age_minutes`, `weather_report_missing`, `history_origin_rate`, `history_origin_n`, `history_origin_no_history`, `history_airline_origin_rate`, `history_airline_origin_n`, `history_airline_origin_no_history`, `history_route_rate`, `history_route_n`, `history_route_no_history`, `airport_scheduled_departures_prior_60m`, `departure_sin2`, `departure_cos2`, `month_sin2`, `month_cos2`, `atl_departure_sin`, `atl_departure_cos`, `atl_departure_sin2`, `atl_departure_cos2`, `ord_departure_sin`, `ord_departure_cos`, `ord_departure_sin2`, `ord_departure_cos2`, `dfw_departure_sin`, `dfw_departure_cos`, `dfw_departure_sin2`, `dfw_departure_cos2`, `history_flight_rate`, `history_flight_n`, `history_flight_no_history`, `dewpoint_spread_f`, `wind_east_knots`, `wind_north_knots`, `freezing_wet`, `prior3h_report_n`, `prior3h_precip_known_n`, `prior3h_wet_fraction`, `prior3h_visibility_min`, `prior3h_gust_max`, `prior3h_visibility_change`.

**pressure_wind: 72 input columns.** `airline`, `origin`, `destination`, `weekday`, `distance_miles`, `scheduled_duration_minutes`, `departure_sin`, `departure_cos`, `month_sin`, `month_cos`, `temperature_f`, `wind_knots`, `gust_knots`, `visibility_miles`, `precip_inches`, `precip_trace`, `ceiling_ft`, `no_ceiling_reported`, `cloud_coverage_missing`, `rain_reported`, `frozen_precip_reported`, `fog_or_mist_reported`, `thunder_reported`, `condition_codes_missing`, `ceiling_height_missing`, `weather_age_minutes`, `weather_report_missing`, `history_origin_rate`, `history_origin_n`, `history_origin_no_history`, `history_airline_origin_rate`, `history_airline_origin_n`, `history_airline_origin_no_history`, `history_route_rate`, `history_route_n`, `history_route_no_history`, `airport_scheduled_departures_prior_60m`, `departure_sin2`, `departure_cos2`, `month_sin2`, `month_cos2`, `atl_departure_sin`, `atl_departure_cos`, `atl_departure_sin2`, `atl_departure_cos2`, `ord_departure_sin`, `ord_departure_cos`, `ord_departure_sin2`, `ord_departure_cos2`, `dfw_departure_sin`, `dfw_departure_cos`, `dfw_departure_sin2`, `dfw_departure_cos2`, `history_flight_rate`, `history_flight_n`, `history_flight_no_history`, `dewpoint_spread_f`, `wind_east_knots`, `wind_north_knots`, `freezing_wet`, `prior3h_report_n`, `prior3h_precip_known_n`, `prior3h_wet_fraction`, `prior3h_visibility_min`, `prior3h_gust_max`, `prior3h_visibility_change`, `altimeter_inhg`, `altimeter_change3h_inhg`, `altimeter_range3h_inhg`, `variable_wind_flag`, `gust_excess_knots`, `altimeter_missing_flag`.

**combined: 78 input columns.** `airline`, `origin`, `destination`, `weekday`, `distance_miles`, `scheduled_duration_minutes`, `departure_sin`, `departure_cos`, `month_sin`, `month_cos`, `temperature_f`, `wind_knots`, `gust_knots`, `visibility_miles`, `precip_inches`, `precip_trace`, `ceiling_ft`, `no_ceiling_reported`, `cloud_coverage_missing`, `rain_reported`, `frozen_precip_reported`, `fog_or_mist_reported`, `thunder_reported`, `condition_codes_missing`, `ceiling_height_missing`, `weather_age_minutes`, `weather_report_missing`, `history_origin_rate`, `history_origin_n`, `history_origin_no_history`, `history_airline_origin_rate`, `history_airline_origin_n`, `history_airline_origin_no_history`, `history_route_rate`, `history_route_n`, `history_route_no_history`, `airport_scheduled_departures_prior_60m`, `departure_sin2`, `departure_cos2`, `month_sin2`, `month_cos2`, `atl_departure_sin`, `atl_departure_cos`, `atl_departure_sin2`, `atl_departure_cos2`, `ord_departure_sin`, `ord_departure_cos`, `ord_departure_sin2`, `ord_departure_cos2`, `dfw_departure_sin`, `dfw_departure_cos`, `dfw_departure_sin2`, `dfw_departure_cos2`, `history_flight_rate`, `history_flight_n`, `history_flight_no_history`, `dewpoint_spread_f`, `wind_east_knots`, `wind_north_knots`, `freezing_wet`, `prior3h_report_n`, `prior3h_precip_known_n`, `prior3h_wet_fraction`, `prior3h_visibility_min`, `prior3h_gust_max`, `prior3h_visibility_change`, `altimeter_inhg`, `altimeter_change3h_inhg`, `altimeter_range3h_inhg`, `variable_wind_flag`, `gust_excess_knots`, `altimeter_missing_flag`, `temperature_centered_square`, `wind_square`, `dewpoint_spread_square`, `precip_log1p`, `ceiling_log1p`, `visibility_inverse`.

Selected B packages: Logistic Regression and KNN use `combined`; SVM uses `current_66`; Random Forest uses `pressure_wind`. Full construction formulas and availability rules are in the linked feature scripts.
