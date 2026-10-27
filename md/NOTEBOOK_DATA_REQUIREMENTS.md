# Data required by the four notebooks

Run notebooks 01–04 in order. They use **17 active flight months**: November–December 2024, all of 2025, and May–July 2026. January–April 2026 is archived and is not an input to the current notebooks.

## Existing source files needed for a complete run

| Data | Active location | Files | Purpose |
|---|---|---:|---|
| November–December 2024 flights | `data/raw/flights/bts/2024_context/csv/2024_11.csv`, `2024_12.csv` | 2 | History for early-2025 training flights |
| January–December 2025 flights | `data/raw/flights/bts/2025/csv/2025_*.csv` | 12 | Training/development population and earlier history |
| May–July 2026 flights | `data/raw/flights/bts/2026/csv/2026_05.csv`, `2026_06.csv`, `2026_07.csv` | 3 | May/June history and July testing/traffic context |
| 2025 weather responses | `data/raw/weather/observations/2025/*.csv` | 36 | Weather for 2025 flights |
| May–July 2026 weather responses | `data/raw/weather/observations/2026/2026_0[5-7]_origins.csv` | 3 | Retained weather inputs, request padding and source-provenance ordering for the same feature build |
| Weather request metadata | A `.csv.json` beside each of the 39 weather CSVs | 39 | Notebook 01 reads saved requests and obtains a missing payload only when enabled |
| Origin station responses | `data/raw/weather/stations/GA_ASOS.geojson`, `IL_ASOS.geojson`, `TX_ASOS.geojson` | 3 | ATL, ORD and DFW timezone lookup |
| Station request metadata | A `.geojson.json` beside each of those three station files | 3 | Saved station requests |

The active weather inputs are retained as complete original monthly responses, including their padding. This is the exact current input set, rather than a claim that every observation contributes to a selected flight.

The normal run uses `DOWNLOAD_MISSING=False`. Missing required payloads stop the run. Original flight ZIPs and their download metadata are needed only if a CSV must be extracted or downloaded again; keep them as source provenance.

## Configuration and saved experiment records

Keep these four JSON files:

| File | Why it is needed |
|---|---|
| `config/project.json` | Project discovery, period definitions and preprocessing settings |
| `reports/final_models/cohorts.json` | Exact 40,000 training flight IDs |
| `reports/final_models/selection.json` | Feature lists, model settings and thresholds |
| `reports/model_selection.json` | Previously chosen historical-baseline threshold |

There are **105 existing input files** in total: 17 flight CSVs, 39 weather CSVs, 42 request-metadata files, three station GeoJSON files and four configuration/experiment JSON files. [The exact path list](../data/NOTEBOOK_INPUTS.json) records every file.

## Generated handoffs

These files are created when running the notebooks in order:

| Produced by | Output | Used by |
|---|---|---|
| Notebook 01 | `data/processed/flights_clean.parquet` — 1,374,996 active schedules | Notebook 02 |
| Notebook 02 | `reports/simple_reproduction/features.parquet` — 40,000 training and 10,000 July test flights | Notebooks 03 and 04 |
| Notebook 03 | `reports/simple_reproduction/models.joblib` — eight fitted A/B pipelines | Notebook 04 |
| Notebook 04 | `reports/simple_reproduction/metrics.json` and `scores.npz` | Saved evaluation results |

When starting from a later notebook, its preceding handoffs must already exist. The Python environment also needs the packages in [requirements.txt](../requirements.txt).

## Retained files outside the normal notebook dependency set

- [January–April 2026 archive](../data/archive/2026_jan_apr/README.md): original flight CSVs, ZIPs, metadata, source documentation, weather responses and the removed cleaned rows.
- Other station networks: 55 are retained; only GA_ASOS, IL_ASOS and TX_ASOS are required.
- OurAirports reference CSV: retained for provenance, not read by the notebooks.
- The older processed feature table and legacy saved model bundles are retained outside the notebook workflow. Older experiment reports and bulky evaluation exports have been removed.

The optional verifier uses the compact `reports/final_models/reference.json`, `reference_scores.npz` and original `models/final_flight_delay_models.joblib` bundle. These are dependencies of **verification**, not of a normal four-notebook run. Original acquisition inventories describe the earlier layout; the [archive manifest](../data/archive/2026_jan_apr/archive_manifest.json) maps every moved file from its former path to its current path.
