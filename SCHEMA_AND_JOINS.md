# Schema and joins used in this project

Verified against notebooks 01, 02 and 02a; `scripts/improvement_features.py`; `scripts/final_features.py`; and the frozen July export schemas on 21 October 2026. This describes the implemented pipeline, not a proposed warehouse. It contains no EDA or model performance findings.

## 1. Physical storage and logical schema

The implementation is a file-based analytical pipeline using pandas DataFrames, CSV/GeoJSON raw sources and Parquet handoffs. There is no SQL database with declared primary/foreign-key constraints and no implemented star or snowflake warehouse. Keys and relationship rules are checked in Python. The final input is a wide flight-level feature table: one row per eligible query flight, with schedule, weather, history and traffic columns.

| Object | Grain: what one row means | Key used in this pipeline | Physical status |
| --- | --- | --- | --- |
| Raw BTS flight CSV | One reported scheduled flight leg on a date | Source month/file plus original CSV row position identifies the source record | Original monthly CSVs and ZIPs |
| Clean flights | One scoped scheduled flight, including context and ineligible schedules | `flight_id`, checked unique | `data/processed/flights_clean.parquet`: 1,339,044 rows, 33 columns |
| Prepared weather | One usable origin-station observation at a timestamp | Composite key: `origin`, `weather_observed_utc`, checked unique after overlap/conflict treatment | Built in memory from raw weather CSVs |
| IEM station reference | One station feature within a network response | Explicit network selection plus station `sid` for origin mappings | Raw GeoJSON responses |
| Airport reference | One airport/facility | `iata_code` for the required airport subset, checked unique before relevant lookups | Raw `airports.csv`: 86,053 rows, 19 columns |
| Original flight features | One eligible query flight | `flight_id`, checked unique | `data/processed/features.parquet`: 1,158,165 rows, 55 columns |
| Later model inputs | One selected query flight | `flight_id` and frozen cohort IDs | Built in memory; later feature sets are not extra permanent Parquet handoffs |
| Prediction export | One flight scored by one candidate in one run/cohort | `run_id`, `candidate_id`, `split`, `cohort_id`, `flight_id` | Final CSV export |

The Parquet handoffs cover the original preparation stage through March 2026. Later scripts combine their records with raw April-July 2026 inputs as needed. Feature counts differ from stored column counts because labels, identifiers, timestamps and audit flags are not predictors. The full stored schemas and final input lists are in [DATASET_GUIDE.md](DATASET_GUIDE.md).

## 2. Flight identity and reference lookups

`flight_id` has a form such as `2025_01:12345`, meaning the January source and original CSV row 12,345. This is an illustrative ID, not a claim about a specific flight. It is a project provenance key tied to the unchanged source ordering, not a universal industry flight ID.

A separate duplicate check uses the composite schedule identity:

`flight_date + airline + flight_number + origin + destination + scheduled_departure_hhmm`.

Flight number alone is insufficient. A repeated schedule identity triggers review rather than automatic deletion.

For the three origins, notebook 01 selects station metadata from GA_ASOS/ATL, IL_ASOS/ORD and TX_ASOS/DFW. Each station match must be unique and have `tzname`. A dictionary lookup maps origin code to timezone. Airport reference records are filtered by `iata_code`; the three required records must also be unique. Broader route-feature experiments map required origin/destination codes to coordinates after checking the lookup index is unique. These are code-based lookups, often implemented with `.map()`, not necessarily calls to `.merge()`.

**Do not equate identifiers from different providers.** BTS `OriginAirportID` and OurAirports `id` belong to different identifier systems. The implemented airport lookup uses the airport code. An airport's origin and destination roles are also distinct: origin-local timezone is used for departure-time conversion.

## 3. Monthly files are concatenated

`pd.concat(parts, ignore_index=True)` stacks monthly flight rows vertically after selecting the required columns/scope. Weather response batches are also concatenated before duplicate/conflict handling. Concatenation is not a relational join: January rows are followed by February rows rather than paired with them.

There are also horizontal concatenations and index-based `.join()` operations in exploratory feature/plot cells. Those attach columns derived from the same base rows. Their correctness depends on preserving the original row index and flight order; they are not matches based on airline or airport. The feature appendices check index and `flight_id` alignment.

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

Several flights can share the same key on the left. The right side must have no more than one row per composite report key. `validate='many_to_one'` rejects an ambiguous right table. A row-count assertion checks that the enrichment has not multiplied flights. This is implemented in `add_compact` in [scripts/final_features.py](scripts/final_features.py). Similar exact report-key joins are used for raw-weather auditing in notebook 02a.

## 6. Flight-to-flight lookups and the EDA inner join

Later feature building recovers `flight_number` and `scheduled_departure_minutes` from clean flights by indexing that table on `flight_id` and using `.map()`. These are one-to-one key lookups for selected query IDs, without expanding rows.

Notebook 02a attaches departure hour and observed delay for exploration with:

```python
train.merge(
    scheduled[['flight_id', 'departure_hour', 'departure_delay_minutes']],
    on='flight_id',
    validate='one_to_one',
)
```

Because no `how` is supplied, this is an inner merge: it keeps matching IDs. Both tables come from the corresponding original training handoffs. `validate='one_to_one'` checks uniqueness on both sides. The observed delay attached here is for exploration; it does not become a predictor of that same flight.

## 7. Historical delay features: grouped time-range aggregation

History is conceptually a flight-to-earlier-flights relationship, but the code does not materialize a large self-join. It uses grouped, sorted event timestamps, `np.searchsorted` and cumulative delay counts. Each query flight receives one set of summaries.

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
- Prepared weather must be unique on `origin`, `weather_observed_utc`. Identical overlap payloads are collapsed; all versions of approved conflicting timestamps are excluded. Unreviewed conflicts stop processing.
- Main feature row counts and flight order/IDs are checked against their query cohorts. The as-of join restores original query order after sorting for time matching.
- Exact weather enrichment validates `many_to_one`; EDA flight enrichment validates `one_to_one`.
- Matched weather ages and contributing historical-event timestamps must obey the declared time limits.
- Model fitting/evaluation feature sets share frozen query IDs and labels. Later July construction checks 10,000 unique query flights after enrichment.

These checks enforce the project rules in Python. They are not SQL foreign-key constraints, and no single assertion alone proves all forms of temporal availability.

## 10. Evaluation-output relationships have a different grain

The final July export contains 10,000 distinct flights scored by 16 candidates, so `flight_predictions.csv` has 160,000 rows. Repeated `flight_id` values here are intentional. A candidate can represent a feature-set/model combination or a baseline.

| Export | Rows | Grain / verified unique key |
| --- | ---: | --- |
| `flight_predictions.csv` | 160,000 | Common candidate/cohort key + `flight_id` |
| `model_metrics.csv` | 16 | Common candidate/cohort key |
| `subgroup_metrics.csv` | 288 | Common key + `group_dimension`, `group_value` |
| `priority_curve.csv` | 48 | Common key + `capacity_fraction` |
| `pr_curve.csv` | 113,047 | Common key + `point_index` |

The common key is `run_id`, `candidate_id`, `split`, `cohort_id`. These compound keys were checked for duplicates while preparing this explanation; none were found. `feature_comparison.csv` separately describes one A-to-B comparison per model/cohort, with explicit base/added feature-set fields.

Joining detailed predictions to metrics on the full common key is many-to-one. Joining only on `model_id` can mix feature sets. If aggregate metrics are repeated onto prediction rows, summing those repeated aggregate fields would inflate totals. The HTML dashboard therefore keeps its metric, subgroup, priority, curve and flight data at their intended grains and selects records by identifiers rather than creating one giant joined table.

For a report, the accurate description is: **a pandas-based extraction and enrichment pipeline with temporal joins, key lookups and grouped window aggregation, producing a denormalized flight-level modeling table and separate evaluation tables.** A dimensional warehouse would be an additional implementation, not the schema currently built.
