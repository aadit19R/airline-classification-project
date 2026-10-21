"""Compact, cutoff-safe features reused by the final development and test notebooks."""
import hashlib
import json
import numpy as np
import pandas as pd
from improvement_features import (TIME_FEATURES, FLIGHT_FEATURES, WEATHER_NEW,
    load_weather, add_time, add_weather, add_histories, add_traffic)

PRESSURE = ['altimeter_inhg', 'altimeter_change3h_inhg', 'altimeter_range3h_inhg',
            'variable_wind_flag', 'gust_excess_knots', 'altimeter_missing_flag']
SHAPES = ['temperature_centered_square', 'wind_square', 'dewpoint_spread_square',
          'precip_log1p', 'ceiling_log1p', 'visibility_inverse']


def compact_columns(root):
    prep = json.loads((root/'reports/preparation_manifest.json').read_text())['feature_engineering']
    base = prep['feature_sets']['B'] + TIME_FEATURES + FLIGHT_FEATURES + WEATHER_NEW
    return {'current_66': base, 'pressure_wind': base+PRESSURE,
            'nonlinear_weather': base+SHAPES, 'combined': base+PRESSURE+SHAPES,
            'A': prep['feature_sets']['A']+TIME_FEATURES}


def pressure_reports(weather):
    weather = weather.copy()
    body = weather['metar'].str.split(r'\s+RMK(?:\s|$)', n=1, regex=True).str[0]
    groups = body.str.findall(r'(?<!\S)A(\d{4})(?=\s|=|$)')
    weather['altimeter_inhg'] = pd.to_numeric(groups.str[0], errors='coerce').div(100).where(groups.str.len().eq(1)).astype(float)
    wind = body.str.extract(r'(?<!\S)((?:\d{3}|VRB)\d{2,3}(?:G\d{2,3})?KT)(?=\s|=|$)', expand=False)
    weather['variable_wind_flag'] = wind.str.startswith('VRB').astype(float).where(wind.notna())
    weather['gust_excess_knots'] = (weather['gust_knots']-weather['wind_knots']).clip(lower=0)
    blocks = []
    for _, block in weather.groupby('origin', observed=True):
        block = block.sort_values('weather_observed_utc').set_index('weather_observed_utc')
        rolling = block['altimeter_inhg'].rolling('3h', closed='both')
        first = rolling.apply(lambda a: a[np.isfinite(a)][0] if np.isfinite(a).any() else np.nan, raw=True)
        block['altimeter_change3h_inhg'] = block['altimeter_inhg']-first
        block['altimeter_range3h_inhg'] = rolling.max()-rolling.min()
        blocks.append(block.reset_index())
    return pd.concat(blocks, ignore_index=True)


def add_compact(frame, weather):
    """Match pressure to the already cutoff-checked report; preserve row order/index."""
    frame = frame.copy()
    reports = pressure_reports(weather)
    matched = frame[['origin', 'weather_observed_utc']].merge(
        reports[['origin', 'weather_observed_utc']+PRESSURE[:-1]],
        how='left', on=['origin', 'weather_observed_utc'], validate='many_to_one')
    assert len(matched) == len(frame)
    for c in PRESSURE[:-1]:
        frame[c] = matched[c].to_numpy(dtype=float, na_value=np.nan)
    frame['altimeter_missing_flag'] = frame['altimeter_inhg'].isna().astype('int8')
    frame['temperature_centered_square'] = ((frame['temperature_f']-32)/10)**2
    frame['wind_square'] = frame['wind_knots']**2
    frame['dewpoint_spread_square'] = frame['dewpoint_spread_f']**2
    frame['precip_log1p'] = np.log1p(frame['precip_inches'])
    frame['ceiling_log1p'] = np.log1p(frame['ceiling_ft'])
    frame['visibility_inverse'] = 1/(1+frame['visibility_miles'])
    assert not np.isinf(frame[PRESSURE+SHAPES].to_numpy(dtype=float, na_value=np.nan)).any()
    return frame


def build_development(root, needed_ids, approved_keys):
    frame = pd.read_parquet(root/'data/processed/features.parquet', filters=[
        ('flight_date', '>=', pd.Timestamp('2025-01-01')), ('flight_date', '<', pd.Timestamp('2026-01-01'))])
    frame = frame.loc[frame['flight_id'].isin(needed_ids)].copy().reset_index(drop=True)
    context = pd.read_parquet(root/'data/processed/flights_clean.parquet', filters=[
        ('flight_date', '<', pd.Timestamp('2026-01-01'))])
    lookup = context.set_index('flight_id')
    for c in ['flight_number', 'scheduled_departure_minutes']:
        frame[c] = frame['flight_id'].map(lookup[c])
    weather, audit = load_weather(root, [2025], approved_keys)
    frame = add_time(add_histories(add_weather(frame, weather), context.loc[context['label_eligible']]))
    frame = add_compact(frame, weather)
    # Rebuild saved B features independently on a small structural audit cohort.
    check = frame.head(150).copy()
    rebuilt = add_traffic(add_histories(add_weather(check, weather, base=True),
                                      context.loc[context['label_eligible']], base=True), context)
    original = json.loads((root/'reports/preparation_manifest.json').read_text())['feature_engineering']['feature_sets']['B']
    for c in original[10:]:
        np.testing.assert_allclose(check[c].to_numpy(dtype=float), rebuilt[c].to_numpy(dtype=float), equal_nan=True, atol=1e-12, rtol=0)
    assert len(frame) == len(needed_ids) and frame['flight_id'].is_unique
    return frame.set_index('flight_id', drop=False), audit


def read_2026_raw(root, months):
    """Apply the existing raw cleaning policy; call for new test months only after freeze."""
    from fresh_test_data import SOURCE_COLUMNS
    parts, sources = [], []
    for month in months:
        path = root/f'data/raw/flights/bts/2026/csv/2026_{month:02d}.csv'
        for chunk in pd.read_csv(path, usecols=SOURCE_COLUMNS, dtype='string', chunksize=150000):
            block = chunk.loc[chunk['Origin'].str.strip().isin(['ATL', 'ORD', 'DFW'])].copy()
            block['flight_id'] = path.stem+':'+(block.index+2).astype(str)
            parts.append(block.rename(columns=SOURCE_COLUMNS))
        with path.open('rb') as stream:
            sources.append({'path':path.relative_to(root).as_posix(), 'sha256':hashlib.file_digest(stream,'sha256').hexdigest()})
    raw = pd.concat(parts, ignore_index=True)
    for c in ['airline', 'flight_number', 'origin', 'destination']:
        raw[c] = raw[c].str.strip().replace('', pd.NA)
    raw['flight_date'] = pd.to_datetime(raw['flight_date'], format='%Y-%m-%d').astype('datetime64[ns]')
    for c in list(SOURCE_COLUMNS.values())[5:]:
        raw[c] = pd.to_numeric(raw[c], errors='coerce').astype(float)
        raw[c] = raw[c].where(np.isfinite(raw[c]))
    assert raw['cancelled'].isin([0,1]).all()
    raw['label_eligible'] = raw['cancelled'].eq(0)&raw['departure_delay_minutes'].notna()
    raw['delayed_departure'] = raw['departure_delay_minutes'].ge(15).where(raw['label_eligible']).astype('Int8')
    known = raw['label_eligible']&raw['departure_delay_15_source'].notna()
    assert raw.loc[known,'departure_delay_15_source'].eq(raw.loc[known,'delayed_departure']).all()
    clock = raw['scheduled_departure_hhmm']
    assert (clock.notna()&clock.mod(1).eq(0)&clock.between(0,2359)&clock.mod(100).lt(60)).all()
    identity = ['flight_date','airline','flight_number','origin','destination','scheduled_departure_hhmm']
    assert raw[identity].notna().all().all() and not raw.duplicated(identity).any()
    raw['scheduled_departure_minutes'] = (clock//100*60+clock%100).astype('int16')
    raw['scheduled_departure_utc'] = pd.Series(pd.NaT, index=raw.index, dtype='datetime64[ns, UTC]')
    local = raw['flight_date']+pd.to_timedelta(raw['scheduled_departure_minutes'],unit='m')
    zones = json.loads((root/'reports/preparation_manifest.json').read_text())['cleaning']['origin_timezones_from_raw']
    for origin,tz in zones.items():
        mask = raw['origin'].eq(origin)
        raw.loc[mask,'scheduled_departure_utc'] = local.loc[mask].dt.tz_localize(tz,ambiguous='NaT',nonexistent='NaT').dt.tz_convert('UTC')
    assert raw['scheduled_departure_utc'].notna().all()
    raw['prediction_cutoff_utc'] = raw['scheduled_departure_utc']-pd.Timedelta(hours=2)
    raw['departure_outcome_event_utc'] = (raw['scheduled_departure_utc']+pd.to_timedelta(raw['departure_delay_minutes'],unit='m')).where(raw['label_eligible'])
    for c in ['distance_miles','scheduled_duration_minutes']:
        raw[c] = raw[c].where(raw[c].gt(0))
    raw['month'] = raw['flight_date'].dt.month.astype('int8')
    raw['weekday'] = raw['flight_date'].dt.dayofweek.astype('int8')
    raw['departure_hour'] = (raw['scheduled_departure_minutes']//60).astype('int8')
    raw['departure_time_band'] = pd.cut(raw['departure_hour'],[-1,5,11,17,23],labels=[
        '00–05 overnight','06–11 morning','12–17 afternoon','18–23 evening']).astype('string')
    angle = 2*np.pi*raw['scheduled_departure_minutes']/1440
    annual = 2*np.pi*(raw['month']-1)/12
    raw['departure_sin'],raw['departure_cos'] = np.sin(angle),np.cos(angle)
    raw['month_sin'],raw['month_cos'] = np.sin(annual),np.cos(annual)
    return raw,sources


def build_july_test(root, approved_keys):
    from final_modelling import sample
    raw,sources = read_2026_raw(root,[4,5,6,7])
    july = raw.loc[raw['month'].eq(7)]
    population = july.loc[july['label_eligible']].copy()
    query = sample(population,10000).reset_index(drop=True)
    cols = ['flight_id','origin','airline','flight_number','destination','label_eligible',
            'delayed_departure','departure_outcome_event_utc','scheduled_departure_utc']
    old = pd.read_parquet(root/'data/processed/flights_clean.parquet', columns=cols)
    context = pd.concat([old,raw[cols]],ignore_index=True)
    assert context['flight_id'].is_unique
    weather, audit = load_weather(root,[2026],approved_keys)
    query = add_time(add_traffic(add_histories(add_weather(query,weather,base=True),
                     context.loc[context['label_eligible']],base=True),context))
    query = add_compact(query,weather)
    assert len(query)==10000 and query['flight_id'].is_unique
    assert query['flight_date'].between('2026-07-01','2026-07-31').all()
    return query,dict(sources=sources,scheduled_july=len(july),eligible_july=len(population),
        cancelled_excluded=int(july['cancelled'].sum()),
        unknown_label_excluded=int((july['cancelled'].eq(0)&~july['label_eligible']).sum()),
        weather=audit,weather_unmatched=int(query['weather_report_missing'].sum()),
        policy='No retuning; sequential history windows retain the seven-day outcome lag.')
