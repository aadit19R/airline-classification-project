"""Only raw CSV decoding and local-time conversion for the simple notebooks.

Neither function creates labels, features, samples, models, plots or output files.
Weather decoding and efficient event-window counts reuse improvement_features.py.
"""
from pathlib import Path
import json
import pandas as pd

SOURCE_COLUMNS = {
    'FlightDate': 'flight_date', 'Reporting_Airline': 'airline',
    'Flight_Number_Reporting_Airline': 'flight_number', 'Origin': 'origin',
    'Dest': 'destination', 'CRSDepTime': 'scheduled_departure_hhmm',
    'CRSElapsedTime': 'scheduled_duration_minutes', 'Distance': 'distance_miles',
    'DepDelay': 'departure_delay_minutes', 'DepDel15': 'departure_delay_15_source',
    'Cancelled': 'cancelled', 'Diverted': 'diverted',
}


def read_flights(root, paths):
    """Paths -> three-origin string rows with stable file:CSV-line IDs, plus counts.

    Chunking avoids loading every national flight into memory. CSV row 1 is the
    header, so DataFrame index 0 becomes source row 2. Nothing is deduplicated.
    """
    parts, counts = [], []
    for path in paths:
        path = Path(path)
        total = retained = 0
        for chunk in pd.read_csv(path, usecols=SOURCE_COLUMNS, dtype='string', chunksize=150000):
            total += len(chunk)
            block = chunk.loc[chunk['Origin'].str.strip().isin(['ATL', 'ORD', 'DFW'])].copy()
            retained += len(block)
            block['source_file'] = path.relative_to(root).as_posix()
            block['source_row_number'] = block.index + 2
            block['flight_id'] = path.stem + ':' + block['source_row_number'].astype(str)
            parts.append(block.rename(columns=SOURCE_COLUMNS))
        counts.append({'month': path.stem, 'national_rows': total, 'scoped_rows': retained})
        print(f'{path.stem}: {retained:,} scoped schedules', flush=True)
    return pd.concat(parts, ignore_index=True), pd.DataFrame(counts)


def scheduled_utc(frame, root):
    """Local flight_date + scheduled_departure_minutes -> UTC Series and zone map.

    Read zones from archived station metadata. Unresolved DST times stop execution;
    choosing an offset would silently change the prediction cutoff.
    """
    zones = {}
    for origin, network in [('ATL', 'GA_ASOS'), ('ORD', 'IL_ASOS'), ('DFW', 'TX_ASOS')]:
        payload = json.loads((root / f'data/raw/weather/stations/{network}.geojson').read_text())
        hits = [f['properties']['tzname'] for f in payload['features'] if f['properties'].get('sid') == origin]
        assert len(hits) == 1
        zones[origin] = hits[0]
    local = frame['flight_date'] + pd.to_timedelta(frame['scheduled_departure_minutes'], unit='m')
    utc = pd.Series(pd.NaT, index=frame.index, dtype='datetime64[ns, UTC]')
    for origin, zone in zones.items():
        mask = frame['origin'].eq(origin)
        utc.loc[mask] = local.loc[mask].dt.tz_localize(zone, ambiguous='NaT', nonexistent='NaT').dt.tz_convert('UTC')
    assert utc.notna().all(), 'Unresolved local schedule time'
    return utc, zones
