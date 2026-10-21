"""Apply the reviewed cleaning rules to Q2 raw sources after model selection."""
import json
import hashlib
import numpy as np
import pandas as pd
from improvement_features import ns_utc,add_time,add_weather,add_histories,add_traffic,load_weather

SOURCE_COLUMNS={'FlightDate':'flight_date','Reporting_Airline':'airline',
 'Flight_Number_Reporting_Airline':'flight_number','Origin':'origin','Dest':'destination',
 'CRSDepTime':'scheduled_departure_hhmm','CRSElapsedTime':'scheduled_duration_minutes',
 'Distance':'distance_miles','DepDelay':'departure_delay_minutes',
 'DepDel15':'departure_delay_15_source','Cancelled':'cancelled','Diverted':'diverted'}


def build_q2_test(root,approved_keys,seed=42,n=10000):
    from sklearn.model_selection import train_test_split
    parts=[];sources=[]
    for month in [4,5,6]:
        path=root/f'data/raw/flights/bts/2026/csv/2026_{month:02d}.csv'
        for chunk in pd.read_csv(path,usecols=SOURCE_COLUMNS,dtype='string',chunksize=150000):
            block=chunk.loc[chunk['Origin'].str.strip().isin(['ATL','ORD','DFW'])].copy()
            block['flight_id']=path.stem+':'+(block.index+2).astype(str)
            parts.append(block.rename(columns=SOURCE_COLUMNS))
        with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        sources.append({'path':path.relative_to(root).as_posix(),'sha256':digest})
    raw=pd.concat(parts,ignore_index=True)
    for c in ['airline','flight_number','origin','destination']:raw[c]=raw[c].str.strip().replace('',pd.NA)
    raw['flight_date']=pd.to_datetime(raw['flight_date'],format='%Y-%m-%d',errors='raise').astype('datetime64[ns]')
    for c in ['scheduled_departure_hhmm','scheduled_duration_minutes','distance_miles',
              'departure_delay_minutes','departure_delay_15_source','cancelled','diverted']:
        raw[c]=pd.to_numeric(raw[c],errors='coerce').astype(float)
        raw[c]=raw[c].where(np.isfinite(raw[c]))
    assert raw['cancelled'].isin([0,1]).all()
    raw['label_eligible']=raw['cancelled'].eq(0)&raw['departure_delay_minutes'].notna()
    raw['delayed_departure']=raw['departure_delay_minutes'].ge(15).where(raw['label_eligible']).astype('Int8')
    known=raw['label_eligible']&raw['departure_delay_15_source'].notna()
    assert raw.loc[known,'departure_delay_15_source'].eq(raw.loc[known,'delayed_departure']).all()
    clock=raw['scheduled_departure_hhmm']
    assert (clock.notna()&clock.mod(1).eq(0)&clock.between(0,2359)&clock.mod(100).lt(60)).all()
    required=['flight_date','airline','flight_number','origin','destination']
    assert raw[required].notna().all().all()
    assert not raw.duplicated(required+['scheduled_departure_hhmm']).any()
    raw['scheduled_departure_minutes']=(clock//100*60+clock%100).astype('int16')
    raw['scheduled_departure_utc']=pd.Series(pd.NaT,index=raw.index,dtype='datetime64[ns, UTC]')
    local=raw['flight_date']+pd.to_timedelta(raw['scheduled_departure_minutes'],unit='m')
    timezones=json.loads((root/'reports/preparation_manifest.json').read_text())['cleaning']['origin_timezones_from_raw']
    for origin,tz in timezones.items():
        mask=raw['origin'].eq(origin)
        raw.loc[mask,'scheduled_departure_utc']=local.loc[mask].dt.tz_localize(tz,ambiguous='NaT',nonexistent='NaT').dt.tz_convert('UTC')
    assert raw['scheduled_departure_utc'].notna().all()
    raw['prediction_cutoff_utc']=raw['scheduled_departure_utc']-pd.Timedelta(hours=2)
    raw['departure_outcome_event_utc']=(raw['scheduled_departure_utc']+pd.to_timedelta(raw['departure_delay_minutes'],unit='m')).where(raw['label_eligible'])
    for c in ['distance_miles','scheduled_duration_minutes']:raw[c]=raw[c].where(raw[c].gt(0))
    raw['month']=raw['flight_date'].dt.month.astype('int8')
    raw['weekday']=raw['flight_date'].dt.dayofweek.astype('int8')
    raw['departure_hour']=(raw['scheduled_departure_minutes']//60).astype('int8')
    raw['departure_time_band']=pd.cut(raw['departure_hour'],[-1,5,11,17,23],
        labels=['00–05 overnight','06–11 morning','12–17 afternoon','18–23 evening']).astype('string')
    angle=2*np.pi*raw['scheduled_departure_minutes']/1440;annual=2*np.pi*(raw['month']-1)/12
    raw['departure_sin']=np.sin(angle);raw['departure_cos']=np.cos(angle)
    raw['month_sin']=np.sin(annual);raw['month_cos']=np.cos(annual)
    population=raw.loc[raw['label_eligible']].copy().sort_values('flight_id')
    strata=population['month'].astype(str)+'|'+population['delayed_departure'].astype(str)
    ix,_=train_test_split(population.index,train_size=n,stratify=strata,random_state=seed)
    query=population.loc[ix].sort_values('flight_id').reset_index(drop=True)
    # Every group's history uses only events in its earlier window, even within Q2.
    old=pd.read_parquet(root/'data/processed/flights_clean.parquet',columns=[
        'flight_id','origin','airline','flight_number','destination','label_eligible','delayed_departure',
        'departure_outcome_event_utc','scheduled_departure_utc'])
    context=pd.concat([old,raw[old.columns]],ignore_index=True)
    assert context['flight_id'].is_unique
    events=context.loc[context['label_eligible']].copy()
    weather,weather_audit=load_weather(root,[2026],approved_keys)
    query=add_time(add_traffic(add_histories(add_weather(query,weather,base=True),events,base=True),context))
    assert query['flight_id'].is_unique and len(query)==n
    assert query['flight_date'].between('2026-04-01','2026-06-30').all()
    audit={'source_files':sources,'scheduled_scope_rows':len(raw),'eligible_population':len(population),
      'cancelled_excluded':int(raw['cancelled'].eq(1).sum()),
      'noncancelled_unknown_label_excluded':int((raw['cancelled'].eq(0)&raw['departure_delay_minutes'].isna()).sum()),
      'test_rows':len(query),'test_weather_unmatched':int(query['weather_report_missing'].sum()),
      'weather':weather_audit,'all_missing_weather_flights_retained':True,
      'policy':'Frozen sampling/cleaning only. No threshold or model retuning on Q2 outcomes.'}
    return query,audit
