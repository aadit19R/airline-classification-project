"""Download untouched July-2026 test sources; audit structural coverage without outcomes.

Run from any directory:
    python scripts/acquire_2026_july.py
    python scripts/acquire_2026_july.py --audit-only

Existing 2025 data, notebooks and prepared tables are not rebuilt here.
"""
from __future__ import annotations

import argparse
import calendar
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import date, timedelta
import hashlib
import io
import json
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np
import pandas as pd
import requests

from acquire_raw_data import ROOT, RAW, PROV, now, digest, relative, save_json

YEAR = 2026
MONTHS = (7,)
ORIGINS = ('ATL', 'ORD', 'DFW')
WEATHER_FIELDS = ['tmpf', 'dwpf', 'drct', 'sknt', 'gust', 'vsby', 'p01i',
                  'skyc1', 'skyc2', 'skyc3', 'skyc4',
                  'skyl1', 'skyl2', 'skyl3', 'skyl4', 'wxcodes', 'metar']
WEATHER_COLUMNS = ['station', 'valid', *WEATHER_FIELDS]
FLIGHT_REQUIRED = {
    'FlightDate', 'Reporting_Airline', 'Flight_Number_Reporting_Airline',
    'Origin', 'Dest', 'CRSDepTime', 'CRSArrTime', 'CRSElapsedTime', 'Distance',
    'DepDelay', 'DepDel15', 'Cancelled', 'Diverted', 'Tail_Number',
}
# Deliberately excludes delay, cancellation, diversion and aircraft outcome values.
SCHEDULE_COLUMNS = ['FlightDate', 'Reporting_Airline', 'Flight_Number_Reporting_Airline',
                    'Origin', 'Dest', 'CRSDepTime', 'CRSElapsedTime', 'Distance']
ACQUISITION_LOG = PROV / '2026_july_acquisition_log.json'
VALIDATION_REPORT = PROV / '2026_july_validation_report.json'


def download(url, destination, *, kind, params=None):
    """Save the original response bytes plus verifiable request metadata."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = destination.with_suffix(destination.suffix + '.json')
    if destination.exists():
        if not metadata_path.is_file():
            raise ValueError(f'Existing source has no metadata: {destination}')
        metadata = json.loads(metadata_path.read_text())
        if digest(destination) != metadata['sha256']:
            raise ValueError(f'Existing source checksum differs: {destination}')
        print(f'Reused verified {relative(destination)}', flush=True)
        return metadata
    temporary = destination.with_suffix(destination.suffix + '.part')
    for attempt in range(1, 4):
        try:
            print(f'Downloading {destination.name}, attempt {attempt}', flush=True)
            started = now()
            with requests.get(url, params=params, stream=True, timeout=(20, 120)) as response:
                response.raise_for_status()
                headers = {k: response.headers[k] for k in
                           ['Content-Type', 'Content-Length', 'Last-Modified', 'ETag']
                           if k in response.headers}
                with temporary.open('wb') as stream:
                    for block in response.iter_content(1024 * 1024):
                        if block:
                            stream.write(block)
                resolved_url = response.url
            extra = {}
            if kind == 'flights':
                if not zipfile.is_zipfile(temporary):
                    raise ValueError('BTS response is not a ZIP archive')
                with zipfile.ZipFile(temporary) as archive:
                    members = [x for x in archive.infolist() if x.filename.lower().endswith('.csv')]
                    if len(members) != 1:
                        raise ValueError(f'Expected one CSV member, got {len(members)}')
                    with archive.open(members[0]) as stream:
                        header = next(csv.reader(io.TextIOWrapper(stream, encoding='utf-8-sig')))
                    if not FLIGHT_REQUIRED.issubset(header):
                        raise ValueError(f'Missing BTS fields: {FLIGHT_REQUIRED - set(header)}')
                    extra = {'archive_csv_member': members[0].filename,
                             'uncompressed_csv_bytes': members[0].file_size,
                             'column_count': len(header)}
            else:
                with temporary.open(encoding='utf-8-sig', newline='') as stream:
                    header = next(csv.reader(stream))
                if not set(WEATHER_COLUMNS).issubset(header):
                    raise ValueError(f'Weather response lacks fields: {set(WEATHER_COLUMNS) - set(header)}')
            metadata = {
                'path': relative(destination), 'role': f'raw_july_2026_{kind}',
                'method': 'direct_public_download_unmodified_response',
                'request': {'url': url, 'params': params or {}},
                'source_url': url, 'resolved_url': resolved_url,
                'started_at_utc': started, 'retrieved_at_utc': now(),
                'response_headers': headers, 'sha256': digest(temporary),
                'bytes': temporary.stat().st_size, **extra,
            }
            temporary.replace(destination)
            save_json(metadata_path, metadata)
            print(f'Saved {destination.name}: {metadata["bytes"] / 1024**2:.2f} MiB', flush=True)
            return metadata
        except Exception as error:
            print(f'{destination.name}: {type(error).__name__}: {error}', flush=True)
            if attempt == 3:
                raise
            time.sleep(attempt * 2)


def acquire_flight(month):
    filename = f'On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{YEAR}_{month}.zip'
    source = RAW / f'flights/bts/{YEAR}/archives/{YEAR}_{month:02d}.zip'
    metadata = download('https://transtats.bts.gov/PREZIP/' + filename, source, kind='flights')
    destination = RAW / f'flights/bts/{YEAR}/csv/{YEAR}_{month:02d}.csv'
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        member = archive.getinfo(metadata['archive_csv_member'])
        if not destination.exists():
            temporary = destination.with_suffix('.csv.part')
            with archive.open(member) as src, temporary.open('wb') as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
            temporary.replace(destination)
        with archive.open(member) as stream:
            archived_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
        actual_hash = digest(destination)
        if actual_hash != archived_hash:
            raise ValueError(f'Extracted CSV differs from archive: {destination}')
        for doc in archive.infolist():
            if doc.filename.lower().endswith('.html'):
                target = RAW / f'flights/bts/{YEAR}/source_documentation/{YEAR}_{month:02d}_{Path(doc.filename).name}'
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(doc))
    print(f'Extracted unchanged {destination.name}; CRC and member SHA-256 verified.', flush=True)
    return {'download': metadata, 'extracted_csv': {
        'path': relative(destination), 'sha256': actual_hash,
        'bytes': destination.stat().st_size, 'archive': relative(source),
        'member': member.filename, 'crc32': member.CRC}}


def acquire_weather(month):
    first = date(YEAR, month, 1)
    next_month = date(YEAR, month + 1, 1)
    # Pad for UTC/local boundaries, the two-hour horizon and earlier observations.
    params = {
        'station': list(ORIGINS), 'data': WEATHER_FIELDS,
        'sts': (first - timedelta(days=2)).isoformat() + 'T00:00Z',
        'ets': (next_month + timedelta(days=2)).isoformat() + 'T00:00Z',
        'tz': 'Etc/UTC', 'format': 'onlycomma', 'latlon': 'no', 'elev': 'no',
        'missing': 'M', 'trace': 'T', 'direct': 'no', 'report_type': [3, 4],
    }
    return download(
        'https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py',
        RAW / f'weather/observations/{YEAR}/{YEAR}_{month:02d}_origins.csv',
        kind='weather', params=params)


def acquire():
    log = {'started_at_utc': now(), 'flights': [], 'weather': [], 'failures': [],
           'policy': 'New direct raw sources only; no prior cleaned data or test outcome analysis.'}
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {pool.submit(acquire_flight, month): ('flights', month) for month in MONTHS}
        for future in as_completed(jobs):
            kind, month = jobs[future]
            try:
                log[kind].append(future.result())
            except Exception as error:
                log['failures'].append({'kind': kind, 'month': month, 'error': str(error)})
            save_json(ACQUISITION_LOG, log)
    # Keep IEM requests sequential.
    for month in MONTHS:
        try:
            log['weather'].append(acquire_weather(month))
        except Exception as error:
            log['failures'].append({'kind': 'weather', 'month': month, 'error': str(error)})
        save_json(ACQUISITION_LOG, log)
    log['completed_at_utc'] = now()
    save_json(ACQUISITION_LOG, log)
    if log['failures']:
        raise RuntimeError('Required downloads failed; see 2026_july_acquisition_log.json')
    return log


def boundary_weather_audit(current):
    """Compare overlapping June/July raw payloads without selecting versions."""
    parts = []
    for path in sorted((RAW / 'weather/observations/2026').glob('2026_06_origins.csv')):
        frame = pd.read_csv(path, usecols=WEATHER_COLUMNS, dtype='string', keep_default_na=False)
        frame = frame.loc[frame['station'].isin(ORIGINS)].copy()
        for column in WEATHER_COLUMNS:
            frame[column] = frame[column].str.strip()
        parts.append(frame)
    previous = pd.concat(parts, ignore_index=True).drop_duplicates(WEATHER_COLUMNS)
    common = previous[['station', 'valid']].drop_duplicates().merge(
        current[['station', 'valid']].drop_duplicates(), on=['station', 'valid'])
    overlapping = pd.concat([
        previous.merge(common, on=['station', 'valid']),
        current[WEATHER_COLUMNS].merge(common, on=['station', 'valid'])
    ], ignore_index=True).drop_duplicates(WEATHER_COLUMNS)
    differences = overlapping.loc[overlapping.duplicated(['station', 'valid'], keep=False)]
    keys = differences[['station', 'valid']].drop_duplicates()
    return {'shared_observation_keys': len(common),
            'keys_with_changed_or_conflicting_payloads': keys.to_dict('records'),
            'policy': 'Comparison only; both source snapshots remain unchanged.'}


def audit():
    flight_audits, schedules = [], []
    for month in MONTHS:
        path = RAW / f'flights/bts/{YEAR}/csv/{YEAR}_{month:02d}.csv'
        with path.open(encoding='utf-8-sig', newline='') as stream:
            header = next(csv.reader(stream))
        assert FLIGHT_REQUIRED.issubset(header)
        rows, day_counts, origin_counts = 0, {}, {}
        schema_2025 = RAW / 'flights/bts/2025/csv/2025_01.csv'
        with schema_2025.open(encoding='utf-8-sig', newline='') as stream:
            old_header = next(csv.reader(stream))
        for frame in pd.read_csv(path, usecols=SCHEDULE_COLUMNS, dtype='string', chunksize=150_000):
            assert frame['FlightDate'].str.startswith(f'{YEAR}-{month:02d}').all()
            rows += len(frame)
            for day, count in frame['FlightDate'].value_counts().items():
                day_counts[str(day)] = day_counts.get(str(day), 0) + int(count)
            scoped = frame.loc[frame['Origin'].isin(ORIGINS)].copy()
            for origin, count in scoped['Origin'].value_counts().items():
                origin_counts[str(origin)] = origin_counts.get(str(origin), 0) + int(count)
            schedules.append(scoped)
        expected_days = calendar.monthrange(YEAR, month)[1]
        assert len(day_counts) == expected_days
        flight_audits.append({
            'path': relative(path), 'month': f'{YEAR}-{month:02d}',
            'national_rows': rows, 'column_count': len(header),
            'same_columns_as_2025': header == old_header,
            'required_fields_present': True,
            'columns_added_since_2025': sorted(set(header) - set(old_header)),
            'columns_removed_since_2025': sorted(set(old_header) - set(header)),
            'date_min': min(day_counts), 'date_max': max(day_counts),
            'calendar_days_present': len(day_counts), 'scope_schedule_counts': origin_counts,
            'sha256': digest(path), 'outcome_values_loaded': False})
        print(f'Audited {path.name}: {rows:,} source rows, all {expected_days} days present.', flush=True)
    scoped = pd.concat(schedules, ignore_index=True)
    identity = ['FlightDate', 'Reporting_Airline', 'Flight_Number_Reporting_Airline',
                'Origin', 'Dest', 'CRSDepTime']
    repeated_schedules = int(scoped.duplicated(identity, keep=False).sum())
    missing_identity = scoped[identity].isna().sum().astype(int).to_dict()
    scoped['date'] = pd.to_datetime(scoped['FlightDate'])
    clock = pd.to_numeric(scoped['CRSDepTime'], errors='coerce')
    valid_clock = clock.notna() & clock.eq(np.floor(clock)) & clock.between(0, 2359) & clock.mod(100).lt(60)
    minutes = np.floor(clock / 100) * 60 + clock.mod(100)
    scoped['scheduled_utc'] = pd.Series(pd.NaT, index=scoped.index, dtype='datetime64[ns, UTC]')
    timezone_sources = {}
    for origin, network in [('ATL', 'GA_ASOS'), ('ORD', 'IL_ASOS'), ('DFW', 'TX_ASOS')]:
        path = RAW / f'weather/stations/{network}.geojson'
        station = next(x for x in json.loads(path.read_text())['features']
                       if x['properties'].get('sid') == origin)
        tz = station['properties']['tzname']
        timezone_sources[origin] = {'timezone': tz, 'source': relative(path)}
        mask = scoped['Origin'].eq(origin) & valid_clock
        local = scoped.loc[mask, 'date'] + pd.to_timedelta(minutes.loc[mask].astype(float), unit='m')
        scoped.loc[mask, 'scheduled_utc'] = local.dt.tz_localize(
            tz, ambiguous='NaT', nonexistent='NaT').dt.tz_convert('UTC').astype('datetime64[ns, UTC]')
    parts, weather_files = [], []
    for month in MONTHS:
        path = RAW / f'weather/observations/{YEAR}/{YEAR}_{month:02d}_origins.csv'
        metadata = json.loads(path.with_suffix('.csv.json').read_text())
        assert digest(path) == metadata['sha256']
        assert metadata['request']['params']['tz'] == 'Etc/UTC'
        frame = pd.read_csv(path, dtype='string', keep_default_na=False)
        assert set(WEATHER_COLUMNS).issubset(frame.columns)
        assert set(frame['station'].unique()) == set(ORIGINS)
        frame = frame[WEATHER_COLUMNS].copy()
        for column in WEATHER_COLUMNS:
            frame[column] = frame[column].str.strip()
        parts.append(frame)
        weather_files.append({'path': relative(path), 'rows': len(frame),
                              'sha256': metadata['sha256'], 'request': metadata['request']})
    weather = pd.concat(parts, ignore_index=True)
    raw_weather_rows = len(weather)
    weather = weather.drop_duplicates(WEATHER_COLUMNS).copy()
    weather['observed_utc'] = pd.to_datetime(weather['valid'], utc=True).astype('datetime64[ns, UTC]')
    assert weather['observed_utc'].notna().all()
    weather['utc_day'] = weather['observed_utc'].dt.strftime('%Y-%m-%d')
    q2 = weather.loc[weather['observed_utc'].ge('2026-07-01') & weather['observed_utc'].lt('2026-08-01')]
    station_coverage = {}
    for origin in ORIGINS:
        block = q2.loc[q2['station'].eq(origin)].sort_values('observed_utc')
        assert block['utc_day'].nunique() == 31
        station_coverage[origin] = {
            'july_utc_days_present': int(block['utc_day'].nunique()),
            'july_report_versions': len(block),
            'july_unique_observation_times': int(block['observed_utc'].nunique()),
            'maximum_gap_minutes_between_july_keys': float(block['observed_utc'].drop_duplicates().diff().dt.total_seconds().max() / 60)}
    conflict_mask = weather.duplicated(['station', 'observed_utc'], keep=False)
    conflicts = weather.loc[conflict_mask].groupby(['station', 'valid']).size().rename('versions').reset_index()
    # Coverage uses distinct timestamp keys only; no conflicting payload is selected.
    keys = weather[['station', 'observed_utc']].drop_duplicates().rename(columns={'station': 'origin'})
    keys['origin'] = keys['origin'].astype('string')
    query = scoped.loc[scoped['scheduled_utc'].notna(), ['Origin', 'scheduled_utc']].rename(columns={'Origin': 'origin'})
    query['origin'] = query['origin'].astype('string')
    query['cutoff_utc'] = query['scheduled_utc'] - pd.Timedelta(hours=2)
    query['lookup_utc'] = query['cutoff_utc'] - pd.Timedelta(minutes=20)
    matched = pd.merge_asof(query.sort_values('lookup_utc'), keys.sort_values('observed_utc'),
                           by='origin', left_on='lookup_utc', right_on='observed_utc',
                           direction='backward', tolerance=pd.Timedelta(minutes=160))
    ages = (matched['cutoff_utc'] - matched['observed_utc']).dt.total_seconds() / 60
    safe_keys = weather.loc[~conflict_mask, ['station', 'observed_utc']].drop_duplicates()
    safe_keys = safe_keys.rename(columns={'station': 'origin'})
    safe_keys['origin'] = safe_keys['origin'].astype('string')
    fallback = pd.merge_asof(query.sort_values('lookup_utc'), safe_keys.sort_values('observed_utc'),
                             by='origin', left_on='lookup_utc', right_on='observed_utc',
                             direction='backward', tolerance=pd.Timedelta(minutes=160))
    boundary = boundary_weather_audit(weather)
    report = {
        'created_at_utc': now(), 'status': 'raw_july_2026_sources_acquired_and_structurally_audited',
        'policy': 'Structural/raw integrity audit only; no test outcome values, delay rates or performance inspected.',
        'flight_files': flight_audits, 'national_flight_rows': sum(x['national_rows'] for x in flight_audits),
        'scope_scheduled_rows': len(scoped),
        'scope_origin_rows': scoped['Origin'].value_counts().astype(int).to_dict(),
        'scope_origin_day_counts': scoped.groupby('Origin')['FlightDate'].nunique().astype(int).to_dict(),
        'schedule_identity_repeated_rows': repeated_schedules, 'missing_identity_fields': missing_identity,
        'invalid_hhmm_rows': int((~valid_clock).sum()),
        'unresolved_scheduled_utc_rows': int(scoped['scheduled_utc'].isna().sum()),
        'timezone_sources_reused_from_project_raw': timezone_sources,
        'weather_files': weather_files, 'weather_raw_rows_including_padding_overlaps': raw_weather_rows,
        'identical_weather_overlap_versions': raw_weather_rows - len(weather),
        'weather_unique_payloads': len(weather), 'station_coverage': station_coverage,
        'weather_conflicting_station_times': conflicts.to_dict('records'),
        'cross_quarter_weather_boundary_audit': boundary,
        'conflict_policy': 'Retain raw sources; any payload selection/exclusion requires later review.',
        'cutoff_weather_key_coverage': {
            'scope_schedules_with_resolved_times': len(query),
            'matched_schedule_keys': int(matched['observed_utc'].notna().sum()),
            'unmatched_schedule_keys': int(matched['observed_utc'].isna().sum()),
            'min_age_minutes': float(ages.min()), 'max_age_minutes': float(ages.max()),
            'matches_if_ambiguous_keys_excluded': int(fallback['observed_utc'].notna().sum()),
            'fallback_check_is_diagnostic_only': True,
            'note': 'Key existence only; ambiguous payloads are not resolved by this check.'},
        'test_outcome_values_loaded': False, 'test_outcome_rates_computed': False,
        'required_missing_files': [],
        'history_context': 'Existing April-June 2026 schedules/outcomes support the declared rolling window; no extra 2024 downloads needed.',
        'remaining_assumptions': [
            'Historical weather/outcome delivery receipts and archived schedule availability remain unverified.',
            'Final aircraft assignments remain excluded from the agreed A/B experiment.'],
        'processed_handoffs_rebuilt': False,
    }
    save_json(VALIDATION_REPORT, report)
    print(f'Raw audit saved: {len(scoped):,} three-origin schedules; '
          f'{matched["observed_utc"].isna().sum():,} missing weather keys; '
          f'{len(conflicts)} ambiguous weather timestamps. No test outcome values loaded.', flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    if not args.audit_only:
        acquire()
    audit()
