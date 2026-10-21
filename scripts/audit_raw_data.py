"""Inspect source integrity/coverage without cleaning or writing model data."""
from __future__ import annotations
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw'
PROV=RAW/'_provenance'
ORIGINS={'ATL','ORD','DFW'}
REQUIRED={'FlightDate','Reporting_Airline','Flight_Number_Reporting_Airline','Origin','Dest','CRSDepTime','CRSArrTime','CRSElapsedTime','Distance','DepDelay','Cancelled','Diverted','Tail_Number','ArrTime','ArrDelay','ActualElapsedTime'}
WEATHER_REQUIRED={'station','valid','tmpf','dwpf','drct','sknt','gust','vsby','p01i','skyc1','skyc2','skyc3','skyc4','skyl1','skyl2','skyl3','skyl4','wxcodes','metar'}


def now():return datetime.now(timezone.utc).isoformat()
def save_json(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n')
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def audit():
    flight_files=[];scope=[]
    for folder in ['2024_context','2025']:
        paths=sorted((RAW/f'flights/bts/{folder}/csv').glob('*.csv'))
        if len(paths)!=(2 if folder=='2024_context' else 12):raise ValueError(f'Missing raw flight CSVs: {folder}')
        for path in paths:
            with path.open(encoding='utf-8-sig',newline='') as stream:header=next(csv.reader(stream))
            if not REQUIRED.issubset(header):raise ValueError(f'Missing fields in {path}: {REQUIRED-set(header)}')
            totals={'rows':0,'noncancelled_rows':0,'noncancelled_known_departure':0,'missing_tail_on_noncancelled':0}
            sc={'month':path.stem,'scheduled_rows':0,'eligible_departures':0,'cancelled':0,'unknown_departure_on_noncancelled':0,'diverted_with_known_departure':0,'delayed_15plus':0}
            dates_min=[];dates_max=[]
            cols=['FlightDate','Origin','DepDelay','Cancelled','Diverted','Tail_Number']
            for frame in pd.read_csv(path,usecols=cols,dtype={'FlightDate':'string','Origin':'string','Tail_Number':'string'},chunksize=150000):
                totals['rows']+=len(frame)
                cancelled=pd.to_numeric(frame.Cancelled,errors='coerce')
                delay=pd.to_numeric(frame.DepDelay,errors='coerce')
                operating=cancelled.eq(0)
                eligible=operating&delay.notna()
                totals['noncancelled_rows']+=int(operating.sum())
                totals['noncancelled_known_departure']+=int(eligible.sum())
                totals['missing_tail_on_noncancelled']+=int((operating&frame.Tail_Number.isna()).sum())
                dates_min.append(frame.FlightDate.min());dates_max.append(frame.FlightDate.max())
                if not frame.FlightDate.str.startswith(path.stem.replace('_','-')).all():raise ValueError(f'Flight dates outside source month: {path}')
                if folder=='2025':
                    q=frame.Origin.isin(ORIGINS)
                    sc['scheduled_rows']+=int(q.sum())
                    sc['eligible_departures']+=int((q&eligible).sum())
                    sc['cancelled']+=int((q&cancelled.eq(1)).sum())
                    sc['unknown_departure_on_noncancelled']+=int((q&operating&delay.isna()).sum())
                    sc['diverted_with_known_departure']+=int((q&eligible&frame.Diverted.eq(1)).sum())
                    sc['delayed_15plus']+=int((q&eligible&delay.ge(15)).sum())
            item={'path':str(path.relative_to(ROOT)),'source_month':path.stem,'column_count':len(header),'columns':header,'min_flight_date':str(min(dates_min)),'max_flight_date':str(max(dates_max)),**totals}
            flight_files.append(item)
            if folder=='2025':scope.append(sc)
            print(f"Audited {path.stem}: {totals['rows']:,} raw rows; {len(header)} columns; tail field available.",flush=True)
    weather=[];observed_months={s:set() for s in ORIGINS};raw_counts={s:0 for s in ORIGINS}
    for path in sorted((RAW/'weather/observations/2025').glob('*.csv')):
        with path.open(encoding='utf-8-sig',newline='') as stream:header=next(csv.reader(stream))
        if not WEATHER_REQUIRED.issubset(header):raise ValueError(f'Missing weather fields: {path}')
        frame=pd.read_csv(path,usecols=['station','valid'],dtype='string')
        counts={}
        for station in ORIGINS:
            values=frame.loc[frame.station.eq(station),'valid']
            counts[station]=int(len(values));raw_counts[station]+=len(values)
            observed_months[station].update(values.str.slice(0,7).dropna().unique())
        weather.append({'path':str(path.relative_to(ROOT)),'raw_rows':len(frame),'columns':header,'target_station_raw_rows':counts})
    if len(weather)!=36:raise ValueError(f'Unexpected raw weather-response count: {len(weather)}')
    expected_months={f'2025-{m:02d}' for m in range(1,13)}
    for station,months in observed_months.items():
        if not expected_months.issubset(months):raise ValueError(f'Missing observed weather months for {station}')
    references=[]
    for sid,network in [('ATL','GA_ASOS'),('ORD','IL_ASOS'),('DFW','TX_ASOS')]:
        path=RAW/f'weather/stations/{network}.geojson'
        features=json.loads(path.read_text())['features']
        hits=[x for x in features if x['properties'].get('sid')==sid]
        if len(hits)!=1 or not hits[0]['properties'].get('tzname'):raise ValueError(f'Missing raw station/timezone metadata: {sid}')
        references.append({'station':sid,'source':str(path.relative_to(ROOT)),'raw_iana_timezone':hits[0]['properties']['tzname'],'coordinates':hits[0]['geometry']['coordinates']})
    airport=RAW/'reference/airports/airports.csv'
    with airport.open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream);airport_fields=reader.fieldnames;count=0;found=set()
        if not {'iata_code','ident','name','latitude_deg','longitude_deg'}.issubset(airport_fields):raise ValueError('Airport schema incomplete')
        for row in reader:
            count+=1
            if row['iata_code'] in ORIGINS:found.add(row['iata_code'])
    if found!=ORIGINS:raise ValueError('Airport reference missing project origins')
    checks={'created_at_utc':now(),'status':'required_raw_inputs_available','policy':'Fresh audit of this project\'s local raw files; no prior cleaned/interim/feature/metric/timezone dataset was loaded. No source rows were changed.','flight_files':flight_files,'weather_files':weather,'weather_station_raw_row_counts_including_overlaps':raw_counts,'weather_station_month_coverage':{s:sorted(v) for s,v in observed_months.items()},'raw_station_timezone_fields':references,'airport_reference':{'path':str(airport.relative_to(ROOT)),'rows':count,'columns':airport_fields,'project_origins_found':sorted(found)},'raw_handling_notes':['Original ZIP members extracted byte-for-byte; archive CRC and SHA256 verified.','Weather responses retain other requested stations, missing values, original formats and overlaps; fresh cleaning is still required.','No target/predictor table has been created by this acquisition audit.'],'remaining_limitations':[{'item':'historical_aircraft_assignment_receipt_time','status':'not_recorded_in_BTS','impact':'Tail_Number provides the final flown aircraft, not when the assignment was known. C remains a separately labelled retrospective scenario unless assignment-time evidence is supplied.'},{'item':'historical_weather_and_outcome_feed_receipt_times','status':'not_in_monthly_flight_or_raw_observation_files','impact':'Use and disclose a delivery-latency scenario; actual historical receipt times have not been established.'}]}
    save_json(PROV/'raw_validation_report.json',checks)
    save_json(ROOT/'data/scope_audit.json',{'created_at_utc':now(),'source':'Fresh audit of locally acquired unmodified official raw CSVs; replaces prior-project planning estimates.','origins':sorted(ORIGINS),'year':2025,'monthly':scope})
    inventory=[]
    for path in sorted(RAW.rglob('*')):
        if not path.is_file() or '_provenance' in path.parts or path.name in ['README.md','.gitkeep']:continue
        if path.suffix=='.part' or path.name.endswith('.part'):raise ValueError(f'Unfinished source present: {path}')
        inventory.append({'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sha256':digest(path)})
    save_json(PROV/'file_inventory.json',{'created_at_utc':now(),'file_count':len(inventory),'total_bytes':sum(x['bytes'] for x in inventory),'files':inventory})
    print('All required raw source integrity/schema/month/station checks passed.',flush=True)
    print('2025 full-network rows:',sum(x['rows'] for x in flight_files if x['source_month'].startswith('2025')))
    print('New raw storage GiB:',round(sum(x['bytes'] for x in inventory)/1024**3,3))
    print('Scope eligible departures:',sum(x['eligible_departures'] for x in scope))


if __name__=='__main__':audit()
