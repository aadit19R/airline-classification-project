"""Cutoff-safe features for notebook 04; no model fitting or test selection here."""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd

WEATHER_COLUMNS = ['station','valid','tmpf','dwpf','drct','sknt','gust','vsby','p01i',
    'skyc1','skyc2','skyc3','skyc4','skyl1','skyl2','skyl3','skyl4','wxcodes','metar']
WEATHER_BASE = ['temperature_f','wind_knots','gust_knots','visibility_miles','precip_inches',
    'precip_trace','ceiling_ft','no_ceiling_reported','cloud_coverage_missing','rain_reported',
    'frozen_precip_reported','fog_or_mist_reported','thunder_reported',
    'condition_codes_missing','ceiling_height_missing']
TIME_FEATURES = ['departure_sin2','departure_cos2','month_sin2','month_cos2'] + [
    f'{o.lower()}_departure_{f}' for o in ['ATL','ORD','DFW'] for f in ['sin','cos','sin2','cos2']]
FLIGHT_FEATURES = ['history_flight_rate','history_flight_n','history_flight_no_history']
WEATHER_NEW = ['dewpoint_spread_f','wind_east_knots','wind_north_knots','freezing_wet',
    'prior3h_report_n','prior3h_precip_known_n','prior3h_wet_fraction',
    'prior3h_visibility_min','prior3h_gust_max','prior3h_visibility_change']

def ns_utc(series):
    return pd.to_datetime(series, utc=True).astype('datetime64[ns, UTC]')

def load_weather(root, years, approved_keys):
    paths = [p for year in years for p in sorted((root / f'data/raw/weather/observations/{year}').glob('*.csv'))]
    parts=[]
    for path in paths:
        metadata=json.loads(path.with_suffix('.csv.json').read_text())
        assert metadata['request']['params']['tz'] in ['Etc/UTC','UTC']
        raw=pd.read_csv(path,usecols=WEATHER_COLUMNS,dtype='string',keep_default_na=False)
        raw=raw.loc[raw['station'].str.strip().isin(['ATL','ORD','DFW'])].copy()
        raw[WEATHER_COLUMNS]=raw[WEATHER_COLUMNS].apply(lambda s:s.str.strip())
        raw['weather_source_file']=path.relative_to(root).as_posix()
        parts.append(raw)
    weather=pd.concat(parts,ignore_index=True).drop_duplicates(WEATHER_COLUMNS).copy()
    weather['weather_observed_utc']=ns_utc(weather['valid'])
    conflict=weather.duplicated(['station','weather_observed_utc'],keep=False)
    keys=weather.loc[conflict,['station','valid']].drop_duplicates()
    approved=pd.MultiIndex.from_tuples([tuple(x) for x in approved_keys])
    unknown=keys.loc[~pd.MultiIndex.from_frame(keys).isin(approved)]
    if len(unknown):
        raise ValueError('New weather conflicts require review: '+unknown.to_json(orient='records'))
    weather=weather.loc[~conflict].rename(columns={'station':'origin'}).copy()
    weather['precip_trace'] = weather['p01i'].eq('T').astype('float64')
    numeric_weather = ['tmpf', 'dwpf', 'drct', 'sknt', 'gust', 'vsby', 'p01i',
                       'skyl1', 'skyl2', 'skyl3', 'skyl4']
    nonnegative_weather = ['sknt', 'gust', 'vsby', 'p01i', 'skyl1', 'skyl2', 'skyl3', 'skyl4']
    weather_treatment_log = []
    for column in numeric_weather:
        original = weather[column]
        values = pd.to_numeric(original.mask(original.isin(['M', 'T', ''])), errors='coerce').astype('float64')
        values = values.where(np.isfinite(values))
        if column == 'p01i':
            values = values.mask(original.eq('T'), 0.0)
        negative = values.lt(0) if column in nonnegative_weather else pd.Series(False, index=weather.index)
        weather_treatment_log.append({'variable': column, 'negative_to_missing': int(negative.sum()),
                                      'capping': False})
        weather[column] = values.mask(negative)
    weather['precip_trace'] = weather['precip_trace'].where(weather['p01i'].notna())
    coverage = weather[[f'skyc{i}' for i in range(1, 5)]]
    heights = weather[[f'skyl{i}' for i in range(1, 5)]].set_axis(coverage.columns, axis=1)
    ceiling = coverage.isin(['BKN', 'OVC', 'VV'])
    coverage_known = coverage.isin(['CLR', 'SKC', 'FEW', 'SCT', 'BKN', 'OVC', 'VV', 'NSC', 'NCD']).any(axis=1)
    ceiling_height_missing = (ceiling & heights.isna()).any(axis=1)
    # Read all body layers: the structured source has only four cloud slots.
    body = weather['metar'].str.split(r'\s+RMK(?:\s|$)', n=1, regex=True).str[0]
    cloud_groups = body.str.findall(r'(?<!\S)(?:FEW|SCT|BKN|OVC|VV)(?:\d{3}|///)(?:CB|TCU)?(?=\s|=|$)')

    def parse_body_clouds(groups):
        ceilings = [g for g in groups if g.startswith(('BKN', 'OVC', 'VV'))]
        heights = [int(re.search(r'\d{3}', g).group()) * 100 for g in ceilings if '///' not in g]
        return (min(heights) if heights else np.nan, bool(ceilings),
                any('///' in g for g in ceilings))

    parsed_clouds = pd.DataFrame(cloud_groups.map(parse_body_clouds).tolist(),
                                columns=['body_ceiling_ft', 'body_has_ceiling', 'body_height_missing'],
                                index=weather.index)
    coverage_known |= cloud_groups.str.len().gt(0) | body.str.contains(r'\b(?:CLR|SKC|NSC|NCD)\b', regex=True)
    ceiling_height_missing |= parsed_clouds['body_height_missing']
    column_ceiling = heights.where(ceiling).min(axis=1)
    weather['ceiling_ft'] = pd.concat([column_ceiling, parsed_clouds['body_ceiling_ft']], axis=1).min(axis=1).mask(ceiling_height_missing)
    has_ceiling = ceiling.any(axis=1) | parsed_clouds['body_has_ceiling']
    weather['no_ceiling_reported'] = (~has_ceiling).astype(float).where(coverage_known)
    weather['cloud_coverage_missing'] = (~coverage_known).astype(float)
    weather['ceiling_height_missing'] = ceiling_height_missing.astype(float).where(coverage_known)
    assert weather.loc[ceiling_height_missing, 'ceiling_ft'].isna().all()
    assert weather.loc[~coverage_known, 'no_ceiling_reported'].isna().all()
    conditions = weather['wxcodes']
    condition_known = ~conditions.isin(['M', ''])
    for name, pattern in {'rain_reported': r'RA|DZ', 'frozen_precip_reported': r'SN|SG|PL|IC|GS|GR',
                           'fog_or_mist_reported': r'FG|BR', 'thunder_reported': r'TS'}.items():
        weather[name] = conditions.str.contains(pattern, regex=True, na=False).astype('float64').where(condition_known)
    weather['condition_codes_missing'] = (~condition_known).astype('float64')
    weather = weather.rename(columns={'tmpf': 'temperature_f', 'sknt': 'wind_knots',
                                       'gust': 'gust_knots', 'vsby': 'visibility_miles',
                                       'p01i': 'precip_inches'})
    WEATHER_FEATURES = ['temperature_f', 'wind_knots', 'gust_knots', 'visibility_miles',
        'precip_inches', 'precip_trace', 'ceiling_ft', 'no_ceiling_reported', 'cloud_coverage_missing',
        'rain_reported', 'frozen_precip_reported', 'fog_or_mist_reported', 'thunder_reported',
        'condition_codes_missing', 'ceiling_height_missing']
    direction=weather['drct'].where(weather['drct'].between(0,360))
    calm=weather['wind_knots'].eq(0)
    weather['dewpoint_spread_f']=weather['temperature_f']-weather['dwpf']
    weather['wind_east_knots']=(weather['wind_knots']*np.sin(np.deg2rad(direction))).mask(calm,0.)
    weather['wind_north_knots']=(weather['wind_knots']*np.cos(np.deg2rad(direction))).mask(calm,0.)
    # Unknown precip/condition is not silently interpreted as dry.
    wet=weather['precip_inches'].gt(0)|weather['precip_trace'].eq(1)|weather['rain_reported'].eq(1)|weather['frozen_precip_reported'].eq(1)
    wet_known=weather['precip_inches'].notna()|weather['rain_reported'].notna()
    weather['freezing_wet']=(weather['temperature_f'].le(32)&wet).astype(float).where(wet_known&weather['temperature_f'].notna())
    blocks=[]
    for _,block in weather.groupby('origin',observed=True):
        block=block.sort_values('weather_observed_utc').set_index('weather_observed_utc')
        wet_amount=block['precip_inches'].gt(0).astype(float).where(block['precip_inches'].notna())
        roll=block.rolling('3h',closed='both')
        block['prior3h_report_n']=pd.Series(1.,index=block.index).rolling('3h',closed='both').sum()
        block['prior3h_precip_known_n']=roll['precip_inches'].count()
        block['prior3h_wet_fraction']=wet_amount.rolling('3h',closed='both').mean()
        block['prior3h_visibility_min']=roll['visibility_miles'].min()
        block['prior3h_gust_max']=roll['gust_knots'].max()
        # Difference from first valid visibility in the window, not a fixed 3-hour slope.
        block['prior3h_visibility_change']=block['visibility_miles']-roll['visibility_miles'].apply(lambda a:a[~np.isnan(a)][0] if np.isfinite(a).any() else np.nan,raw=True)
        blocks.append(block.reset_index())
    weather=pd.concat(blocks,ignore_index=True)
    assert not weather.duplicated(['origin','weather_observed_utc']).any()
    return weather,{'source_files':[p.relative_to(root).as_posix() for p in paths],
                    'excluded_conflicts':keys.to_dict('records'),'usable_reports':len(weather)}

def add_time(frame):
    frame=frame.copy()
    angle=2*np.pi*frame['scheduled_departure_minutes']/1440
    month_angle=2*np.pi*(frame['month']-1)/12
    for name,values in [('departure_sin2',np.sin(2*angle)),('departure_cos2',np.cos(2*angle)),
                        ('month_sin2',np.sin(2*month_angle)),('month_cos2',np.cos(2*month_angle))]:
        frame[name]=values
    for origin in ['ATL','ORD','DFW']:
        for harmonic in ['sin','cos','sin2','cos2']:
            frame[f'{origin.lower()}_departure_{harmonic}']=frame['origin'].eq(origin).astype(float)*frame[f'departure_{harmonic}']
    return frame

def history_counts(query,events,groups,lag=7,days=28):
    n=np.zeros(len(query),dtype='int32');positive=n.copy()
    latest=np.full(len(query),np.iinfo('int64').min,dtype='int64')
    upper=ns_utc(query['prediction_cutoff_utc']).array.asi8-pd.Timedelta(days=lag).value
    lower=upper-pd.Timedelta(days=days).value
    eg=events.groupby(groups,observed=True,dropna=False).indices
    for key,pos in query.groupby(groups,observed=True,dropna=False).indices.items():
        if key not in eg: continue
        block=events.iloc[eg[key]].sort_values('departure_outcome_event_utc')
        times=ns_utc(block['departure_outcome_event_utc']).array.asi8
        cum=np.r_[0,block['delayed_departure'].to_numpy(dtype='int64').cumsum()]
        left=np.searchsorted(times,lower[pos],side='left');right=np.searchsorted(times,upper[pos],side='left')
        n[pos]=right-left;positive[pos]=cum[right]-cum[left]
        available=right>left;latest[pos[available]]=times[right[available]-1]
    assert np.all(latest[n>0]<upper[n>0]) and np.all(latest[n>0]>=lower[n>0])
    return n,positive,pd.to_datetime(latest,utc=True)

def add_histories(frame,events,base=False):
    frame=frame.copy()
    definitions=([('history_origin',['origin']),('history_airline_origin',['origin','airline']),
                  ('history_route',['origin','destination'])] if base else [])
    definitions+=[('history_flight',['origin','airline','flight_number'])]
    for prefix,groups in definitions:
        n,positive,latest=history_counts(frame,events,groups)
        prior=frame['history_origin_rate'].to_numpy() if prefix!='history_origin' else None
        frame[prefix+'_rate']=(np.divide(positive,n,out=np.full(len(frame),np.nan),where=n>0)
            if prior is None else (positive+20*prior)/(n+20))
        frame[prefix+'_n']=n
        frame[prefix+'_no_history']=(n==0).astype('int8')
        frame[prefix+'_latest_event_utc']=latest
    return frame

def add_weather(frame,weather,base=False):
    frame=frame.copy();frame['_order']=np.arange(len(frame))
    frame['weather_lookup_utc']=ns_utc(frame['prediction_cutoff_utc'])-pd.Timedelta(minutes=20)
    # Re-match by time using the identical rule; no dependency on a previously saved payload.
    cols=['origin','weather_observed_utc',*WEATHER_NEW]+(WEATHER_BASE+['weather_source_file'] if base else [])
    frame=frame.drop(columns=[c for c in cols if c!='origin' and c in frame])
    frame['origin']=frame['origin'].astype('string')
    right=weather[cols].copy();right['origin']=right['origin'].astype('string')
    frame=pd.merge_asof(frame.sort_values('weather_lookup_utc'),right.sort_values('weather_observed_utc'),
        by='origin',left_on='weather_lookup_utc',right_on='weather_observed_utc',direction='backward',
        tolerance=pd.Timedelta(minutes=160)).sort_values('_order').drop(columns='_order').reset_index(drop=True)
    age=(ns_utc(frame['prediction_cutoff_utc'])-frame['weather_observed_utc']).dt.total_seconds()/60
    assert age.dropna().between(20,180).all()
    if base:
        frame['weather_age_minutes']=age
        frame['weather_report_missing']=frame['weather_observed_utc'].isna().astype('int8')
    return frame

def add_traffic(frame,schedules):
    frame=frame.copy();counts=np.zeros(len(frame),dtype='int32')
    for origin in ['ATL','ORD','DFW']:
        pos=np.flatnonzero(frame['origin'].eq(origin).to_numpy())
        times=ns_utc(schedules.loc[schedules['origin'].eq(origin),'scheduled_departure_utc']).sort_values().array.asi8
        query=ns_utc(frame.iloc[pos]['scheduled_departure_utc']).array.asi8
        counts[pos]=np.searchsorted(times,query,side='left')-np.searchsorted(times,query-pd.Timedelta(hours=1).value,side='left')
    frame['airport_scheduled_departures_prior_60m']=counts
    return frame
