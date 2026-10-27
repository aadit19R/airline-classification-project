# Schema and joins used in this project

The core pipeline code is visible in notebooks 01–04 and checked against the frozen July exports and pre-refactor feature fingerprints. The final section describes the current saved model outputs. This describes the implemented pipeline, not a proposed warehouse. It contains no EDA or model performance findings.

## 1. Physical storage and logical schema

The implementation is a file-based analytical pipeline using pandas DataFrames, CSV/GeoJSON raw sources and Parquet handoffs. There is no SQL database with declared primary/foreign-key constraints and no implemented star or snowflake warehouse. Keys and relationship rules are checked in Python. The final input is a wide flight-level feature table: one row per eligible query flight, with schedule, weather, history and traffic columns.

| Object | Grain: what one row means | Key used in this pipeline | Physical status |
| --- | --- | --- | --- |
| Raw BTS flight CSV | One reported scheduled flight leg on a date | Source month/file plus original CSV row position identifies the source record | Original monthly CSVs and ZIPs |
| Clean flights | One scoped scheduled flight, including context and ineligible schedules | `flight_id`, checked unique | `data/processed/flights_clean.parquet`: 1,374,996 rows, 33 columns |
| Prepared weather | One usable origin-station observation at a timestamp | Composite key: `origin`, `weather_observed_utc`, checked unique after overlap/conflict treatment | Built in memory from raw weather CSVs |
| IEM station reference | One station feature within a network response | Explicit network selection plus station `sid` for origin mappings | Raw GeoJSON responses |
| Airport reference | One airport/facility | `iata_code` for the required airport subset, checked unique before relevant lookups | Raw `airports.csv`: 86,053 rows, 19 columns |
| Published EDA feature reference | One eligible query flight | `flight_id`, checked unique | `data/processed/features.parquet`: 1,158,165 rows, 55 columns |
| Later model inputs | One selected query flight | `flight_id` and frozen cohort IDs | `reports/simple_reproduction/features.parquet`: 50,000 rows, saved by notebook 02 |
| Evaluation metrics | One model/input combination or baseline | `model_id`, `feature_set` | `reports/simple_reproduction/metrics.json` |

The cleaned source table uses `preparation_period` for source/EDA periods. The older EDA reference table retains `preparation_sample` and `initial_refit_sample` for historical provenance. The current 50,000-row notebook handoff uses only `split=train` (40,000 all-2025 flights) and `split=test` (10,000 July flights).

Notebook 01 saves 17 active months: November–December 2024, all of 2025, and May–July 2026. January–April 2026 sources and their 314,085 cleaned schedules are archived. Notebook 02 reads that handoff directly; it does not need a separate later-period loader. Feature counts differ from stored column counts because labels, identifiers, timestamps and audit flags are not predictors. The full stored schemas and final input lists are in [DATASET_GUIDE.md](DATASET_GUIDE.md).

## 2. Flight identity and reference lookups

`flight_id` has a form such as `2025_01:12345`, meaning the January source and original CSV row 12,345. This is an illustrative ID, not a claim about a specific flight. It is a project provenance key tied to the unchanged source ordering, not a universal industry flight ID.

A separate duplicate check uses the composite schedule identity:

`flight_date + airline + flight_number + origin + destination + scheduled_departure_hhmm`.

Flight number alone is insufficient. A repeated schedule identity triggers review rather than automatic deletion.

For the three origins, notebook 01 selects station metadata from GA_ASOS/ATL, IL_ASOS/ORD and TX_ASOS/DFW. Each station match must be unique and supply `tzname`. A loop applies that station's timezone to the origin's flights. The OurAirports snapshot is retained as a source reference; it is not used to create current model features.

**Do not equate identifiers from different providers.** BTS `OriginAirportID` and OurAirports `id` belong to different identifier systems. The implemented airport lookup uses the airport code. An airport's origin and destination roles are also distinct: origin-local timezone is used for departure-time conversion.

## 3. Monthly files are concatenated

`pd.concat(parts, ignore_index=True)` stacks monthly flight rows vertically after selecting the required columns/scope. Weather response batches are also concatenated before duplicate/conflict handling. Concatenation is not a relational join: January rows are followed by February rows rather than paired with them.

Notebook 02 also concatenates the two cloud-ceiling sources horizontally before taking their minimum. Both share the same weather-row index. This attaches values for existing observations; it does not match flights to weather. The actual relational joins follow below.

## 4. Main flight-weather join: backward as-of

The main weather join is `pd.merge_asof`, grouped by origin:

```python
weather_lookup_utc = prediction_cutoff_utc - pd.Timedelta(minutes=20)

# Both frames are sorted by the respective time column first.
features = pd.merge_asof(
    queries_sorted,
    weather_sorted,
    by='origin',
    left_on='weather_lookup_utc',
    right_on='weather_observed_utc',
    direction='backward',
    tolerance=pd.Timedelta(minutes=160),
)
```

This is an abbreviated explanation of the implemented call, not a standalone runnable cell. Notebook 02 also explicitly allows exact matches to the buffered lookup time.

For each flight, the operation chooses the latest report at the same origin no later than the buffered lookup time. The chosen report must be between 20 and 180 minutes old relative to the prediction cutoff. The code's tolerance is 160 minutes because the lookup was already shifted back by 20 minutes: 20 + 160 = 180. All these comparisons use UTC.

Example, with every time in UTC: a departure at 14:00 has a 12:00 cutoff. The buffered lookup is 11:40. A report at 11:30 is eligible; one at 11:50 is too recent under the buffer; one before 09:00 is too old. The latest eligible report is selected. This example illustrates the rule only.

The relationship is **many flights to one weather report**; a flight can also have no qualifying report. The operation retains the flight and leaves weather fields missing when no report qualifies. It does not create one flight row for every weather report that day. A simple merge on origin alone would multiply flight rows by the number of reports at that origin.

The 20-minute delivery buffer is a declared availability assumption. An observation timestamp is not a verified receipt timestamp.

## 5. Exact left join for additional weather fields

After the weather report is selected, later pressure/wind features are attached to that same report by an equality join:

```python
matched = frame[['origin', 'weather_observed_utc']].merge(
    reports[['origin', 'weather_observed_utc'] + pressure_columns],
    how='left',
    on=['origin', 'weather_observed_utc'],
    validate='many_to_one',
)
```

Several flights can share the same key on the left. The right side must have no more than one row per composite report key. `validate='many_to_one'` rejects an ambiguous right table. A row-count assertion checks that the enrichment has not multiplied flights. The exact merge is visible in [notebook 02](../notebooks/simple/02_feature_engineering.ipynb), in the pressure section.

## 6. Select flights by their IDs

Notebook 02 indexes the clean table on `flight_id` and selects the saved training IDs plus sampled July IDs using `.loc`. This retains schedule fields in the requested row order. It is an index lookup, not an additional merge. Notebook 01's EDA already has departure hour and observed delay, so it needs no extra flight-to-flight join.

## 7. Historical delay features: grouped time-range aggregation

History is conceptually a flight-to-earlier-flights relationship, but the code does not materialize a large self-join. It uses sorted outcomes, grouped running totals (`cumcount` and `cumsum`), and two backward `merge_asof` joins. One join finds the totals strictly before the window start; the other finds totals strictly before the end. Subtracting gives the flights and delays inside the window. Each query flight receives one set of summaries.

| History level | Matching group |
| --- | --- |
| Airport | `origin` |
| Airline within airport | `origin`, `airline` |
| Route | `origin`, `destination` |
| Flight-number service, later addition | `origin`, `airline`, `flight_number` |

For prediction cutoff T, events must satisfy **T - 35 days <= event time < T - 7 days**. The lower endpoint is included and the upper endpoint is excluded. The resulting features include past eligible-event count, delayed-event rate, no-history flag and latest contributing event timestamp. Subgroup rates use the documented smoothing rule.

Thus one query can summarize many past records while remaining one row. This avoids a many-to-many row explosion and excludes the query's own eventual outcome from its predictors. The seven-day gap assumes sufficiently timely access to earlier outcomes; the monthly BTS archive does not establish an operational daily feed.

## 8. Scheduled traffic: another grouped time-range aggregation

The core traffic feature counts schedules at the same origin in:

`[scheduled_departure_utc - 60 minutes, scheduled_departure_utc)`.

The upper endpoint is excluded, so the query's own scheduled departure is not counted. The code uses sorted schedule times and `np.searchsorted`, then assigns a single count to the query row. Later-cancelled schedules remain part of this schedule context.

For a 14:00 departure, this window is 13:00 through just before 14:00. It is based on published schedules, not the actual departures observed in that future hour. This is allowed under the archived-schedule-availability assumption, even though the prediction cutoff is 12:00. Broader candidate experiments also aggregate additional departure/arrival windows; those candidates are not all retained in final models.

## 9. Join-integrity checks

- `flight_id` must remain unique at the flight-feature stage.
- Prepared weather must be unique on `origin`, `weather_observed_utc`. Identical overlap payloads are collapsed; all versions of conflicting timestamps are excluded. The optional verifier compares rebuilt features with the saved reference fingerprints.
- Main feature row counts and flight order/IDs are checked against their query cohorts. The as-of join restores original query order after sorting for time matching.
- Exact weather enrichment validates `many_to_one`; selecting flight IDs starts from a table checked for unique IDs.
- Matched weather ages must obey the declared time limits. The optional verifier also checks contributing historical-event timestamps against the history window.
- Model fitting/evaluation feature sets share frozen query IDs and labels. Later July construction checks 10,000 unique query flights after enrichment.

These checks enforce the project rules in Python. They are not SQL foreign-key constraints, and no single assertion alone proves all forms of temporal availability.

## 10. Current saved evaluation outputs

`reports/simple_reproduction/metrics.json` has 12 rows: four classifiers with A/B inputs and four baselines. Its key is `model_id` plus `feature_set`.

`scores.npz` contains one 10,000-value score array per rule. Notebook 04 sorts July flights by `flight_id` before scoring, so that same order defines the array positions. A score array is not another flight source table.

The compact files in `reports/final_models/reference.json` and `reference_scores.npz` preserve original-results checks without the older bulky CSV exports. The feature table remains one row per selected flight; preprocessing and prediction do not multiply those rows.
