"""Acquire independent, unmodified raw sources; never read prior cleaned data.

Usage: python scripts/acquire_raw_data.py --stage copy|download|extract|all
Public BTS files require network access. Existing raw source bytes are copied
without filtering records, renaming columns, imputing values or making features.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import zipfile
import requests

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw'
LEGACY_RAW=ROOT.parent/'airline-delay-project/data/raw'
PROV=RAW/'_provenance'
TARGET_STATIONS={'ATL','ORD','DFW'}
COPY_LOG=PROV/'raw_copy_log.json'


def now():return datetime.now(timezone.utc).isoformat()
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.writing')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)
def relative(path):return str(Path(path).relative_to(ROOT))


def copy_raw():
    records=[]
    def copy_one(source,dest,role,source_url=None,original_metadata=None):
        source=Path(source).resolve()
        if not source.is_relative_to(LEGACY_RAW.resolve()):
            raise ValueError(f'Refusing prior non-raw source: {source}')
        dest=RAW/dest;dest.parent.mkdir(parents=True,exist_ok=True)
        expected=digest(source)
        if dest.exists():
            if digest(dest)!=expected:raise ValueError(f'Existing destination differs: {dest}')
        else:shutil.copy2(source,dest)
        if digest(dest)!=expected:raise ValueError(f'Copy checksum mismatch: {dest}')
        records.append({'path':relative(dest),'role':role,'method':'byte_identical_raw_copy','copied_at_utc':now(),'original_raw_path':str(source),'source_url':source_url,'sha256':expected,'bytes':dest.stat().st_size,'original_metadata':original_metadata})
    copy_one(LEGACY_RAW/'airports/airports.csv','reference/airports/airports.csv','raw_airport_reference','https://ourairports.com/data/')
    for month in [11,12]:
        source=LEGACY_RAW/f'development_v3/bts/2024_{month:02d}.zip'
        metadata=json.loads(source.with_suffix('.json').read_text())
        if digest(source)!=metadata['sha256']:raise ValueError(f'Original archive checksum mismatch: {source}')
        copy_one(source,f'flights/bts/2024_context/archives/2024_{month:02d}.zip','raw_history_and_rotation_context',metadata['source_url'],metadata)
    weather=LEGACY_RAW/'weather_observations'
    selected=[]
    months={}; required_weather={'tmpf','dwpf','drct','sknt','gust','vsby','p01i','skyc1','skyc2','skyc3','skyc4','skyl1','skyl2','skyl3','skyl4','wxcodes','metar'}
    for meta_path in sorted(weather.glob('2025_*.csv.json')):
        meta=json.loads(meta_path.read_text())
        stations=set(meta['request']['params']['station'])
        hits=stations&TARGET_STATIONS
        if not hits:continue
        source=Path(str(meta_path)[:-5])
        if digest(source)!=meta['sha256']:raise ValueError(f'Original weather checksum mismatch: {source}')
        if not required_weather.issubset(set(meta['request']['params']['data'])):raise ValueError('Weather source missing requested variables')
        copy_one(source,f'weather/observations/2025/{source.name}','raw_weather_response',meta['request']['url'],meta)
        copy_one(meta_path,f'weather/observations/2025/{meta_path.name}','original_weather_request_metadata',meta['request']['url'])
        selected.append(source.name)
        month=source.name[:7]
        months.setdefault(month,set()).update(hits)
    if len(months)!=12 or any(s!=TARGET_STATIONS for s in months.values()):
        raise ValueError(f'Weather query coverage incomplete: {months}')
    # Copy only original network responses and their request metadata, not a
    # previous project's combined/cleaned station or timezone mapping.
    for source in sorted((weather/'stations').glob('*.geojson')):
        meta_path=source.with_suffix(source.suffix+'.json')
        if not meta_path.exists():raise ValueError(f'Missing raw station provenance: {source}')
        meta=json.loads(meta_path.read_text())
        if digest(source)!=meta['sha256']:raise ValueError(f'Original station checksum mismatch: {source}')
        copy_one(source,f'weather/stations/{source.name}','raw_station_network_response',meta['request']['url'],meta)
        copy_one(meta_path,f'weather/stations/{meta_path.name}','original_station_request_metadata',meta['request']['url'])
    save_json(COPY_LOG,{'created_at_utc':now(),'policy':'Only sources beneath previous data/raw; all copied data bytes unchanged. No prior processed/interim/reports/model dataset or timezone lookup reused.','target_stations':sorted(TARGET_STATIONS),'weather_response_files':len(selected),'weather_monthly_query_coverage':{m:sorted(v) for m,v in months.items()},'files':records})
    print(f'Copied and verified {len(records)} raw/reference/provenance files; {len(selected)} unchanged weather responses.',flush=True)


def download_month(month):
    name=f'On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2025_{month}.zip'
    url='https://transtats.bts.gov/PREZIP/'+name
    dest=RAW/f'flights/bts/2025/archives/2025_{month:02d}.zip'
    meta_path=dest.with_suffix('.zip.json')
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and meta_path.exists():
        meta=json.loads(meta_path.read_text())
        if digest(dest)==meta['sha256'] and zipfile.is_zipfile(dest):
            print(f'Reused verified 2025-{month:02d} archive.',flush=True);return meta
        raise ValueError(f'Existing archive mismatch: {dest}')
    temporary=dest.with_suffix('.zip.part')
    last=None
    for attempt in range(3):
        try:
            print(f'Downloading official BTS 2025-{month:02d} (attempt {attempt+1}).',flush=True)
            started=now()
            with requests.get(url,stream=True,timeout=(20,120)) as response:
                response.raise_for_status()
                headers={k:response.headers.get(k) for k in ['Content-Type','Content-Length','Last-Modified','ETag'] if response.headers.get(k)}
                with temporary.open('wb') as stream:
                    for block in response.iter_content(1024*1024):
                        if block:stream.write(block)
                resolved_url=response.url
            if not zipfile.is_zipfile(temporary):raise ValueError('Response is not a complete ZIP archive')
            with zipfile.ZipFile(temporary) as archive:
                csvs=[x for x in archive.infolist() if x.filename.lower().endswith('.csv')]
                if len(csvs)!=1:raise ValueError(f'Expected one official CSV, found {len(csvs)}')
                with archive.open(csvs[0]) as stream:header=stream.readline().decode('utf-8-sig')
                if 'Tail_Number' not in header:raise ValueError('Official source does not include Tail_Number')
            meta={'path':relative(dest),'role':'raw_official_2025_flights','method':'direct_public_download','source_url':url,'resolved_url':resolved_url,'started_at_utc':started,'retrieved_at_utc':now(),'response_headers':headers,'sha256':digest(temporary),'bytes':temporary.stat().st_size,'archive_csv_member':csvs[0].filename,'uncompressed_csv_bytes':csvs[0].file_size,'contains_tail_number':True}
            temporary.replace(dest);save_json(meta_path,meta)
            print(f'Complete 2025-{month:02d}: {dest.stat().st_size/1024**2:.1f} MiB, tail number present.',flush=True)
            return meta
        except Exception as error:
            last=error
            print(f'BTS 2025-{month:02d} attempt failed: {type(error).__name__}: {error}',flush=True)
            if attempt<2:time.sleep(2*(attempt+1))
    raise RuntimeError(f'Unable to obtain BTS 2025-{month:02d}: {last}')


def download_all(workers):
    results=[];failures=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_month={pool.submit(download_month,m):m for m in range(1,13)}
        for future in as_completed(future_to_month):
            month=future_to_month[future]
            try:results.append(future.result())
            except Exception as error:failures.append({'month':month,'error':str(error)})
    save_json(PROV/'bts_download_log.json',{'created_at_utc':now(),'completed':sorted(results,key=lambda x:x['path']),'failures':failures})
    if failures:raise RuntimeError(f'{len(failures)} required BTS downloads failed; see bts_download_log.json')


def extract_all():
    extracted=[]
    for year_folder in ['2024_context','2025']:
        archives=RAW/f'flights/bts/{year_folder}/archives'
        for source in sorted(archives.glob('*.zip')):
            with zipfile.ZipFile(source) as archive:
                entry=next(x for x in archive.infolist() if x.filename.lower().endswith('.csv'))
                dest=RAW/f'flights/bts/{year_folder}/csv/{source.stem}.csv'
                dest.parent.mkdir(parents=True,exist_ok=True)
                if not dest.exists() or dest.stat().st_size!=entry.file_size:
                    temporary=dest.with_suffix('.csv.part')
                    with archive.open(entry) as src, temporary.open('wb') as output:
                        shutil.copyfileobj(src,output,1024*1024)
                    temporary.replace(dest)
                # Freshly streamed content was CRC-checked by zipfile. Verify
                # byte identity against the archived member even on a reuse.
                with archive.open(entry) as src:
                    expected=hashlib.file_digest(src,'sha256').hexdigest()
                actual=digest(dest)
                if actual!=expected:raise ValueError(f'Extracted bytes mismatch: {dest}')
                extracted.append({'path':relative(dest),'method':'unchanged_zip_member_extraction','archive':relative(source),'original_member_name':entry.filename,'sha256':actual,'bytes':dest.stat().st_size,'zip_member_crc32':entry.CRC})
                docs=[x for x in archive.infolist() if x.filename.lower().endswith('.html')]
                for doc in docs:
                    doc_path=RAW/f'flights/bts/{year_folder}/source_documentation/{source.stem}_{Path(doc.filename).name}'
                    doc_path.parent.mkdir(parents=True,exist_ok=True)
                    doc_path.write_bytes(archive.read(doc))
                print(f'Extracted and verified {source.stem}: {dest.stat().st_size/1024**2:.1f} MiB CSV, unchanged source bytes.',flush=True)
    save_json(PROV/'bts_extraction_log.json',{'created_at_utc':now(),'files':extracted})
    expected={f'2025_{m:02d}.csv' for m in range(1,13)}
    actual={Path(x['path']).name for x in extracted if '/2025/' in x['path']}
    if actual!=expected:raise ValueError(f'Incomplete 2025 extraction: missing {expected-actual}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['copy','download','extract','all'],default='all')
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if args.stage in ['copy','all']:copy_raw()
    if args.stage in ['download','all']:download_all(args.workers)
    if args.stage in ['extract','all']:extract_all()
