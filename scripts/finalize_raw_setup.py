"""Publish the inventory and user guide after the independent raw audit passes."""
from pathlib import Path
from datetime import datetime,timezone
import json

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw'
PROV=RAW/'_provenance'
report=json.loads((PROV/'raw_validation_report.json').read_text())
inventory=json.loads((PROV/'file_inventory.json').read_text())
if report['status']!='required_raw_inputs_available':raise ValueError('Raw audit not complete')
if json.loads((PROV/'bts_download_log.json').read_text())['failures']:raise ValueError('Required downloads failed')
flight_2025=[x for x in report['flight_files'] if x['source_month'].startswith('2025')]
context=[x for x in report['flight_files'] if x['source_month'].startswith('2024')]
assert len(flight_2025)==12 and len(context)==2

sources=[
{'id':'flights_2025','path':'raw/flights/bts/2025/csv','glob':'2025_*.csv','kind':'unmodified_official_csv','status':'available_local','expected_files':12,'source_url':'https://transtats.bts.gov/PREZIP/','contains_tail_number':True,'raw_rows':sum(x['rows'] for x in flight_2025),'purpose':'All 2025 schedule/outcome/aircraft fields, independently acquired; no previous selected-column or cleaned flight table needed'},
{'id':'flights_2025_archives','path':'raw/flights/bts/2025/archives','glob':'2025_*.zip','kind':'original_official_archive','status':'available_local','expected_files':12,'purpose':'Original source ZIPs and retrieval metadata'},
{'id':'history_context_2024','path':'raw/flights/bts/2024_context/csv','glob':'2024_*.csv','kind':'unmodified_official_csv','status':'available_local','expected_files':2,'contains_tail_number':True,'raw_rows':sum(x['rows'] for x in context),'purpose':'November/December history warm-up and optional aircraft-linkage prototype'},
{'id':'history_context_archives','path':'raw/flights/bts/2024_context/archives','glob':'2024_*.zip','kind':'byte_identical_original_raw_archive_copy','status':'available_local','expected_files':2},
{'id':'weather_observations_2025','path':'raw/weather/observations/2025','glob':'2025_*.csv','kind':'unaltered_raw_api_responses','status':'available_local','expected_files':36,'source_url':'https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py','purpose':'Original monthly batches whose queries include ATL, ORD or DFW; other stations and boundary overlaps remain unfiltered'},
{'id':'weather_station_reference','path':'raw/weather/stations','glob':'*.geojson','kind':'unaltered_raw_api_responses','status':'available_local','expected_files':len(list((RAW/'weather/stations').glob('*.geojson'))),'source_url':'https://mesonet.agron.iastate.edu/geojson/network.php','purpose':'Generate new airport/station/timezone mappings from source fields including tzname; no previous lookup imported'},
{'id':'airport_metadata','path':'raw/reference/airports/airports.csv','kind':'byte_identical_raw_source_copy','status':'available_local','source_url':'https://ourairports.com/data/','raw_rows':report['airport_reference']['rows'],'purpose':'Unmodified airport identifiers, names, coordinates and original metadata'},
]
manifest={'schema_version':'2.0','updated_at_utc':datetime.now(timezone.utc).isoformat(),'path_base':'data_directory','reuse_policy':'Only prior data/raw source bytes and original raw-response metadata may be copied. No prior processed/interim/feature/cleaned/model/timezone table is an input. All active input paths are inside this new project.','acquisition_status':'required_raw_inputs_available','sources':sources,'required_missing_files':[],'remaining_limitations':report['remaining_limitations'],'file_inventory':'raw/_provenance/file_inventory.json','raw_validation':'raw/_provenance/raw_validation_report.json','copy_provenance':'raw/_provenance/raw_copy_log.json','download_provenance':'raw/_provenance/bts_download_log.json','extraction_provenance':'raw/_provenance/bts_extraction_log.json','scope_audit':'scope_audit.json','variable_requirements':'../DATA_REQUIREMENTS.md'}
(ROOT/'data/SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
config_path=ROOT/'config/project.json'
c=json.loads(config_path.read_text())
c['status']='raw_data_acquired'
c['sources_manifest']='data/SOURCE_MANIFEST.json'
c['raw_only_input_policy']=True
c['feature_sets']['C']['source_status']='2025_tail_numbers_acquired_in_full_raw_files'
# Acquisition metadata must not reactivate the excluded rotation experiment.
c['feature_sets']['C']['availability_gate']='Not part of current objectives, training or Tableau findings. Reopening requires a separate scope decision and historical assignment-timing evidence.'
config_path.write_text(json.dumps(c,indent=2)+'\n')

readme=f'''# Raw data: start here

All required raw files are now stored inside this project's own `data/raw` directory. They have not been cleaned, filtered by flight, imputed, merged or converted into model features. Original downloads and source response bytes are retained.

## Ready-to-open source files

| Data | Location | Files |
|---|---|---:|
| Full 2025 flight records, including aircraft registrations | [flights/bts/2025/csv](flights/bts/2025/csv) | 12 CSVs |
| Original 2025 downloads | [flights/bts/2025/archives](flights/bts/2025/archives) | 12 ZIPs + retrieval metadata |
| November/December 2024 context | [flights/bts/2024_context/csv](flights/bts/2024_context/csv) | 2 CSVs |
| Original context archives | [flights/bts/2024_context/archives](flights/bts/2024_context/archives) | 2 ZIPs |
| Raw weather response batches covering the three origin stations | [weather/observations/2025](weather/observations/2025) | 36 CSVs + original request metadata |
| Original station/network reference responses | [weather/stations](weather/stations) | {len(list((RAW/'weather/stations').glob('*.geojson')))} GeoJSONs + request metadata |
| Original airport reference | [reference/airports/airports.csv](reference/airports/airports.csv) | 1 CSV |

For example, [2025_01.csv](flights/bts/2025/csv/2025_01.csv) is the unchanged official January source. The complete files use {flight_2025[0]['column_count']} source columns, including `Tail_Number`; they are wider than the previous project's 28-column inputs.

## What was acquired and checked

- **{sum(x['rows'] for x in flight_2025):,}** raw 2025 flight rows across all 12 months.
- **{sum(x['rows'] for x in context):,}** raw late-2024 context rows.
- Required flight fields, including aircraft registrations, present in every month.
- Weather observations for ATL, ORD and DFW present in every 2025 calendar month.
- Raw station metadata contains an IANA timezone for each of the three origins; fresh mappings still need to be built.
- Original copied raw-file checksums, ZIP/member integrity and extracted CSV byte identity verified.
- Approximately **{inventory['total_bytes']/1024**3:.2f} GiB** stored, including compressed originals and extracted CSVs.

Weather batches include other source-request stations and overlapping boundary observations. These are retained unchanged and need fresh cleaning later. Missing values and source formats are preserved. Reference metadata is a source snapshot rather than proven 2025 historical metadata.

## Provenance and validation

See the [source manifest](../SOURCE_MANIFEST.json), [file checksums](_provenance/file_inventory.json), [raw validation report](_provenance/raw_validation_report.json), [direct download log](_provenance/bts_download_log.json), [raw-copy provenance](_provenance/raw_copy_log.json), and [unchanged extraction log](_provenance/bts_extraction_log.json).

No file from the old project's processed/interim/model outputs or generated timezone lookup was used. Existing raw airport, weather, station and 2024 archive files were copied byte-for-byte. The full 2025 archives were downloaded directly from BTS.

## Required missing files and limitations

**Required raw files still missing: none.**

BTS does not include a historical timestamp of when each aircraft assignment became known. Aircraft rotation is excluded from the agreed A/B project because final `Tail_Number` cannot establish assignment availability at the cutoff. Actual historical delivery/receipt timestamps for weather and flight outcome feeds are also not supplied. These are documented information limits, not silently invented columns.

No classifiers, cleaned dataset, train/test feature matrix or Tableau findings workbook has been created during raw sourcing. The next stage starts fresh cleaning from these source files.
'''
(RAW/'README.md').write_text(readme)

# Bring current project instructions into line with the user's raw-only rule.
p=ROOT/'README.md';t=p.read_text()
t=t.replace('**Current stage:** project setup and objective definition. No new classifiers have been trained and no Tableau workbook with findings has been produced yet.','**Current stage:** raw data acquired and independently audited. Begin at [the raw-data guide](data/raw/README.md). No new classifiers or findings workbook have been produced yet.')
t=t.replace('- C requires supplementary 2025 tail-number data and is a separately labelled retrospective assignment scenario unless timing evidence exists.','- Full 2025 raw files now include tail numbers. C still requires a fresh linkage/coverage audit and is a separately labelled retrospective assignment scenario unless timing evidence exists.')
t=t.replace('Raw source data remains in the existing cache and is referenced through the source inventory. New code and outputs belong in this folder. Core dependencies are listed in requirements.txt; this project does not require the earlier fitted models.','All active raw source inputs now live inside this folder. Only original raw files were copied from the earlier project; 2025 flight archives were acquired directly from BTS. No prior cleaned/processed tables or generated timezone lookup is reused. Core dependencies are listed in requirements.txt.')
p.write_text(t)

req=ROOT/'requirements.txt';t=req.read_text()
if 'requests' not in t:t+='requests>=2.31,<3.0\n'
req.write_text(t)
ignore=ROOT/'.gitignore';t=ignore.read_text()
extra='\n# Large raw payloads stay local; acquisition scripts and guides stay versioned.\ndata/raw/**/*.csv\ndata/raw/**/*.zip\ndata/raw/**/*.part\ndata/raw/weather/stations/*.geojson\n'
if '# Large raw payloads' not in t:t+=extra
ignore.write_text(t)

# Old scope_audit was replaced by audit_raw_data.py using new raw data only.
summary={'status':'raw_acquisition_complete','raw_2025_rows':sum(x['rows'] for x in flight_2025),'raw_context_rows':sum(x['rows'] for x in context),'weather_csv_files':36,'station_geojson_files':len(list((RAW/'weather/stations').glob('*.geojson'))),'bytes':inventory['total_bytes'],'required_missing_files':[],'limitations':report['remaining_limitations'],'next_stage':'Fresh cleaning and transformations inside this project only.'}
(ROOT/'reports/DATA_ACQUISITION_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
(ROOT/'reports/DATA_ACQUISITION_REPORT.md').write_text(f'''# Raw data acquisition completed

See [the raw-data guide](../data/raw/README.md) for locations and provenance.\n\nDownloaded all twelve full official 2025 BTS archives and extracted unchanged CSVs. Copied only original raw source files for weather, station metadata, airports and November/December 2024 context. No previous cleaned data was used.\n\nVerified {summary['raw_2025_rows']:,} raw 2025 rows and {summary['raw_context_rows']:,} context rows, all required source fields, source month coverage, three-station weather coverage and SHA256 identities. Raw storage is approximately {summary['bytes']/1024**3:.2f} GiB. Required missing raw files: none.\n\nHistorical aircraft-assignment receipt timestamps are not recorded in BTS. Historical weather/outcome delivery receipts are also not established by these sources. The retrospective scenario/latency assumptions must remain explicit.\n\nNo cleaning or modelling was performed. The [machine-readable summary](DATA_ACQUISITION_SUMMARY.json) and [fresh raw validation](../data/raw/_provenance/raw_validation_report.json) contain the checked counts and limitations.\n''')
print('Published local-only raw manifest, guide, acquisition report and configuration status.')
