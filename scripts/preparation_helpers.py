"""Technical routines for the short preparation notebooks.

Calculations, plots and safety checks are retained from the executed originals.
The notebooks keep the EDA sequence and the ordinary plotting/analysis steps visible.
These functions use explicit arguments/returns; no hidden execution of notebook cells.
"""
import site
if site.ENABLE_USER_SITE:
    site.addsitedir(site.getusersitepackages())
from datetime import datetime, timezone
from improvement_features import TIME_FEATURES, FLIGHT_FEATURES, WEATHER_NEW, load_weather, add_time, add_weather, add_histories
from matplotlib.ticker import PercentFormatter
from pandas.tseries.holiday import USFederalHolidayCalendar
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from time import perf_counter
import hashlib
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import platform
import re
import seaborn as sns
import site
import site, sys, json, hashlib


def show_table(data, **kwargs):
    print(data.to_string(**kwargs))


def show_plot():
    plt.tight_layout()
    plt.show()

def compare_outlier_rules(frame, columns):
    """Report training-only flags without changing the data."""
    rows = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors='coerce').dropna()
        values = values[np.isfinite(values)]
        if values.empty:
            rows.append([column, 0, 'Unavailable: no finite values', *([np.nan] * 4)])
            continue
        q1, q3 = values.quantile([.25, .75])
        iqr, mean, sd = q3 - q1, values.mean(), values.std(ddof=0)
        rules = [('IQR: Q1-1.5IQR to Q3+1.5IQR', q1 - 1.5 * iqr, q3 + 1.5 * iqr),
                 ('Mean +/- 3 SD (ddof=0)', mean - 3 * sd, mean + 3 * sd),
                 ('Above 99th percentile only', -np.inf, values.quantile(.99))]
        for method, lower, upper in rules:
            flagged = (values < lower) | (values > upper)
            rows.append([column, len(values), method, lower if np.isfinite(lower) else np.nan,
                         upper, int(flagged.sum()), 100 * flagged.mean()])
    fields = ['variable', 'n', 'method', 'lower_bound', 'upper_bound', 'flagged_n', 'flagged_percent']
    return pd.DataFrame(rows, columns=fields) if rows else pd.DataFrame()



def setup_cleaning():
    """Setup cleaning."""
    from pathlib import Path
    import json
    import site
    if site.ENABLE_USER_SITE:
        site.addsitedir(site.getusersitepackages())
    import platform
    from datetime import datetime, timezone

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    import seaborn as sns


    def show_table(data, **kwargs):
        print(data.to_string(**kwargs))

    def show_plot():
        plt.tight_layout()
        plt.show()

    ROOT = Path.cwd().resolve()
    if ROOT.name == 'notebooks':
        ROOT = ROOT.parent
    if (ROOT / 'departure-delay-classification/config/project.json').is_file():
        ROOT = ROOT / 'departure-delay-classification'
    if not (ROOT / 'config/project.json').is_file():
        raise FileNotFoundError('Open this notebook inside departure-delay-classification.')
    RAW = ROOT / 'data/raw'
    CONFIG = json.loads((ROOT / 'config/project.json').read_text())
    ORIGINS = CONFIG['scope']['origin_airports']
    SEED = CONFIG['sampling']['seed']
    WRITE_OUTPUTS = True
    PLOT_SAMPLE_SIZE = 10_000
    sns.set_theme(style='whitegrid', context='notebook', palette=['#247BA0', '#E08D3C', '#5C946E'])
    plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False})
    pd.set_option('display.max_columns', 35)

    show_table(pd.Series({'project_root': str(ROOT), 'python': platform.python_version(),
                       'pandas': pd.__version__, 'exploratory_period': CONFIG['split']['train_start'] + ' to ' + CONFIG['split']['train_end'],
                       'output_mode': 'one clean Parquet + shared JSON manifest'}).to_frame('setting'))
    return CONFIG, ORIGINS, PLOT_SAMPLE_SIZE, RAW, ROOT, SEED, WRITE_OUTPUTS



def read_flight_sources(ORIGINS, RAW, ROOT):
    """Read flight sources."""
    SOURCE_COLUMNS = {
        'FlightDate': 'flight_date', 'Reporting_Airline': 'airline',
        'Flight_Number_Reporting_Airline': 'flight_number', 'Origin': 'origin', 'Dest': 'destination',
        'CRSDepTime': 'scheduled_departure_hhmm', 'CRSElapsedTime': 'scheduled_duration_minutes',
        'Distance': 'distance_miles', 'DepDelay': 'departure_delay_minutes',
        'DepDel15': 'departure_delay_15_source', 'Cancelled': 'cancelled', 'Diverted': 'diverted',
    }
    source_paths = sorted((RAW / 'flights/bts/2024_context/csv').glob('2024_*.csv'))
    source_paths += sorted((RAW / 'flights/bts/2025/csv').glob('2025_*.csv'))
    source_paths += sorted((RAW / 'flights/bts/2026/csv').glob('2026_0[123].csv'))
    expected_months = pd.period_range('2024-11', '2026-03', freq='M').strftime('%Y_%m').tolist()
    assert [p.stem for p in source_paths] == expected_months, 'Missing or unexpected source months.'
    parts, source_log = [], []
    for path in source_paths:
        all_count = scope_count = 0
        for chunk in pd.read_csv(path, usecols=list(SOURCE_COLUMNS), dtype='string', chunksize=150_000):
            all_count += len(chunk)
            subset = chunk.loc[chunk['Origin'].str.strip().isin(ORIGINS)].copy()
            scope_count += len(subset)
            subset['source_row_number'] = subset.index.to_numpy() + 2  # header is CSV row 1
            subset['source_file'] = path.relative_to(ROOT).as_posix()
            subset['flight_id'] = path.stem + ':' + subset['source_row_number'].astype(str)
            parts.append(subset.rename(columns=SOURCE_COLUMNS))
        source_log.append({'source_file': path.relative_to(ROOT).as_posix(),
                           'national_rows_read': all_count, 'three_origin_rows': scope_count})
        print(f'{path.stem}: read {all_count:,}; retained {scope_count:,} scheduled rows.')

    flights = pd.concat(parts, ignore_index=True)
    del parts
    assert flights['flight_id'].is_unique
    show_table(pd.DataFrame(source_log))
    show_table(flights.head(5))
    return flights, source_log



def parse_and_label_flights(CONFIG, flights):
    """Parse and label flights."""
    text_columns = ['airline', 'flight_number', 'origin', 'destination']
    for column in text_columns:
        flights[column] = flights[column].str.strip().replace('', pd.NA)
    flights['flight_date'] = pd.to_datetime(flights['flight_date'], format='%Y-%m-%d', errors='coerce')
    numeric_columns = ['scheduled_departure_hhmm', 'scheduled_duration_minutes', 'distance_miles',
                       'departure_delay_minutes', 'departure_delay_15_source', 'cancelled', 'diverted']
    parse_log = []
    for column in numeric_columns:
        original = flights[column]
        numeric = pd.to_numeric(original, errors='coerce').astype('float64')
        parse_log.append({'variable': column,
                          'nonmissing_unparseable': int((original.notna() & numeric.isna()).sum()),
                          'nonfinite': int((numeric.notna() & ~np.isfinite(numeric)).sum())})
        flights[column] = numeric.where(np.isfinite(numeric))
    show_table(pd.DataFrame(parse_log))

    bad_status = ~flights['cancelled'].isin([0, 1])
    if bad_status.any():
        show_table(flights.loc[bad_status].head(10))
        raise ValueError('Cancellation status is unresolved. Clarify these records before proceeding.')

    # This flag is an audit/population rule, never a model predictor.
    flights['label_eligible'] = flights['cancelled'].eq(0) & flights['departure_delay_minutes'].notna()
    flights['delayed_departure'] = flights['departure_delay_minutes'].ge(15).where(flights['label_eligible']).astype('Int8')
    reconcile = flights['label_eligible'] & flights['departure_delay_15_source'].notna()
    label_mismatch = reconcile & flights['departure_delay_15_source'].ne(flights['delayed_departure'])
    if label_mismatch.fillna(False).any():
        show_table(flights.loc[label_mismatch.fillna(False)].head(10))
        raise ValueError('DepDelay and source DepDel15 disagree; investigate before choosing a label.')

    flights['split'] = 'context'
    for split_name in ['train', 'validation', 'test']:
        in_period = flights['flight_date'].between(
            CONFIG['split'][split_name + '_start'], CONFIG['split'][split_name + '_end'])
        flights.loc[in_period, 'split'] = split_name
    study = flights['split'].isin(['train', 'validation', 'test'])
    development = flights['split'].isin(['train', 'validation'])

    identity = ['flight_date', 'airline', 'flight_number', 'origin', 'destination', 'scheduled_departure_hhmm']
    duplicate_identity = flights.duplicated(identity, keep=False)
    if duplicate_identity.any():
        show_table(flights.loc[duplicate_identity, ['flight_id', *identity, 'departure_delay_minutes', 'cancelled']].head(20))
        raise ValueError('Repeated schedule identities require review. No duplicate flights were automatically deleted.')
    return development, parse_log, study



def add_prediction_timestamps(CONFIG, ORIGINS, RAW, flights, study):
    """Add prediction timestamps."""
    network_by_origin = {'ATL': 'GA_ASOS', 'ORD': 'IL_ASOS', 'DFW': 'TX_ASOS'}
    timezone_by_origin = {}
    for origin, network in network_by_origin.items():
        payload = json.loads((RAW / f'weather/stations/{network}.geojson').read_text())
        hits = [f['properties'] for f in payload['features'] if f['properties'].get('sid') == origin]
        assert len(hits) == 1 and hits[0].get('tzname'), f'Unresolved timezone for {origin}'
        timezone_by_origin[origin] = hits[0]['tzname']
    airport_reference = pd.read_csv(RAW / 'reference/airports/airports.csv',
        usecols=['iata_code', 'name', 'latitude_deg', 'longitude_deg'], dtype={'iata_code': 'string'})
    origin_reference = airport_reference.loc[airport_reference['iata_code'].isin(ORIGINS)].copy()
    assert origin_reference['iata_code'].is_unique and len(origin_reference) == len(ORIGINS)
    origin_reference['timezone'] = origin_reference['iata_code'].map(timezone_by_origin)
    show_table(origin_reference)

    clock = flights['scheduled_departure_hhmm']
    valid_clock = clock.notna() & clock.mod(1).eq(0) & clock.ge(0) & clock.lt(2400) & clock.mod(100).lt(60)
    required_keys = ['flight_date', 'airline', 'flight_number', 'origin', 'destination']
    bad_required = flights[required_keys].isna().any(axis=1) | ~valid_clock
    if bad_required.any():
        show_table(flights.loc[bad_required, ['flight_id', *required_keys, 'scheduled_departure_hhmm']].head(20))
        raise ValueError('Required identity/date/clock values need clarification, including any 2400 records.')

    flights['scheduled_departure_minutes'] = (clock // 100 * 60 + clock % 100).astype('int16')
    local_naive = flights['flight_date'] + pd.to_timedelta(flights['scheduled_departure_minutes'], unit='m')
    scheduled_utc = pd.Series(pd.NaT, index=flights.index, dtype='datetime64[ns, UTC]')
    for origin, tz in timezone_by_origin.items():
        indices = flights.index[flights['origin'].eq(origin)]
        localized = local_naive.loc[indices].dt.tz_localize(tz, ambiguous='NaT', nonexistent='NaT')
        scheduled_utc.loc[indices] = localized.dt.tz_convert('UTC')
    if scheduled_utc.isna().any():
        show_table(flights.loc[scheduled_utc.isna(), ['flight_id', 'flight_date', 'origin', 'scheduled_departure_hhmm']].head(20))
        raise ValueError('Unresolved DST/local-time records: clarify before deleting or assigning an offset.')
    flights['scheduled_departure_utc'] = scheduled_utc
    flights['prediction_cutoff_utc'] = scheduled_utc - pd.Timedelta(minutes=CONFIG['scope']['prediction_horizon_minutes'])

    # Outcome timestamps are history/audit context only, never current-query predictors.
    flights['departure_outcome_event_utc'] = (
        scheduled_utc + pd.to_timedelta(flights['departure_delay_minutes'], unit='m')
    ).where(flights['label_eligible'])
    flights['query_eligible'] = study & flights['label_eligible']
    population = flights.groupby('split', observed=True).agg(
        scheduled_rows=('flight_id', 'size'), known_operating_departures=('label_eligible', 'sum'),
        model_population=('query_eligible', 'sum'))
    show_table(population)
    assert flights.loc[flights['query_eligible'], 'split'].isin(['train', 'validation', 'test']).all()
    return population, timezone_by_origin



def training_eda_view(PLOT_SAMPLE_SIZE, SEED, flights):
    """Training eda view."""
    # Training-only EDA views. Validation/test outcomes are not explored to choose treatments.
    train = flights.loc[flights['query_eligible'] & flights['split'].eq('train')].copy()
    assert len(train) > 0
    train['target_label'] = train['delayed_departure'].map({0: 'On time / <15 min late', 1: '15+ min late'})
    train['month'] = train['flight_date'].dt.month
    train['weekday'] = train['flight_date'].dt.day_name()
    train['departure_hour'] = train['scheduled_departure_minutes'] // 60

    def plot_sample(frame, maximum=PLOT_SAMPLE_SIZE):
        return frame.sample(n=min(len(frame), maximum), random_state=SEED)

    def rate_table(frame, group):
        result = frame.groupby(group, observed=True)['delayed_departure'].agg(n='size', delayed='sum')
        result['delay_rate'] = result['delayed'] / result['n']
        return result.reset_index()

    print(f'Training EDA population: {len(train):,} eligible departures. Scatterplots use a fixed-seed sample; aggregate rates use all training rows.')
    return plot_sample, rate_table, train



def review_missing_flights(development, flights, plot_sample):
    """Review missing flights."""
    missing_columns = ['airline', 'destination', 'scheduled_duration_minutes', 'distance_miles',
                       'departure_delay_minutes', 'departure_delay_15_source', 'diverted']
    training_scheduled = flights.loc[flights['split'].eq('train')]
    missing_table = pd.DataFrame({'missing_n': training_scheduled[missing_columns].isna().sum(),
                                  'missing_fraction': training_scheduled[missing_columns].isna().mean()})
    show_table(missing_table.sort_values('missing_fraction', ascending=False))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    missing_table['missing_fraction'].sort_values().plot.barh(ax=axes[0])
    axes[0].xaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set(title='Missing fields · all scheduled training flights', xlabel='Fraction missing')
    missing_sample = plot_sample(training_scheduled, maximum=1_000)
    sns.heatmap(missing_sample[missing_columns].isna().T, cbar=False, xticklabels=False,
                cmap=['#E9F1F4', '#C44E52'], ax=axes[1])
    axes[1].set(title=f'Missingness pattern · sampled n={len(missing_sample):,}', xlabel='Scheduled flight sample')
    show_plot()

    # Describe exclusions in 2025 development only; no 2026 outcome exploration.
    exclusion_counts = pd.Series({
        '2025 scheduled scope rows': int(development.sum()),
        '2025 cancelled': int((development & flights['cancelled'].eq(1)).sum()),
        '2025 noncancelled, unknown departure label': int((development & flights['cancelled'].eq(0) & flights['departure_delay_minutes'].isna()).sum()),
        '2025 eligible query flights': int((development & flights['query_eligible']).sum()),
        '2025 eligible later-diverted flights retained': int((development & flights['query_eligible'] & flights['diverted'].eq(1)).sum()),
    })
    show_table(exclusion_counts.to_frame('rows'))
    assert exclusion_counts['2025 scheduled scope rows'] == (
        exclusion_counts['2025 cancelled'] + exclusion_counts['2025 noncancelled, unknown departure label']
        + exclusion_counts['2025 eligible query flights'])
    return exclusion_counts



def add_calendar_variables(flights, rate_table, train):
    """Add calendar variables."""
    flights['month'] = flights['flight_date'].dt.month.astype('int8')
    flights['weekday'] = flights['flight_date'].dt.dayofweek.astype('int8')
    flights['departure_hour'] = (flights['scheduled_departure_minutes'] // 60).astype('int8')
    flights['departure_time_band'] = pd.cut(flights['departure_hour'], bins=[-1, 5, 11, 17, 23],
        labels=['00–05 overnight', '06–11 morning', '12–17 afternoon', '18–23 evening']).astype('string')
    angle = 2 * np.pi * flights['scheduled_departure_minutes'] / 1440
    flights['departure_sin'] = np.sin(angle)
    flights['departure_cos'] = np.cos(angle)
    month_angle = 2 * np.pi * (flights['month'] - 1) / 12
    flights['month_sin'] = np.sin(month_angle)
    flights['month_cos'] = np.cos(month_angle)
    train_after = flights.loc[train.index].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    band_rates = rate_table(train_after, 'departure_time_band')
    axes[0].bar(band_rates['departure_time_band'], band_rates['delay_rate'])
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].tick_params(axis='x', rotation=20)
    axes[0].set(title='Delay rate by created time band · training', ylabel='15+ minute delay rate')
    clock_minutes = np.arange(1440)
    clock_angle = 2 * np.pi * clock_minutes / 1440
    points = axes[1].scatter(np.cos(clock_angle), np.sin(clock_angle), c=clock_minutes,
                              cmap='twilight', s=5)
    axes[1].set(aspect='equal', title='Cyclic clock representation · illustration',
                xlabel='Cosine coordinate', ylabel='Sine coordinate')
    fig.colorbar(points, ax=axes[1], label='Minutes after midnight')
    show_plot()
    return train_after



def save_clean_handoff(CONFIG, ROOT, WRITE_OUTPUTS, exclusion_counts, flights, parse_log, population, source_log, timezone_by_origin, treatment_log):
    """Save clean handoff."""
    import site
    if site.ENABLE_USER_SITE:
        site.addsitedir(site.getusersitepackages())  # Make the newly installed engine visible without restarting.

    assert flights['flight_id'].is_unique
    assert flights.loc[flights['query_eligible'], 'delayed_departure'].notna().all()
    assert (flights['prediction_cutoff_utc'] < flights['scheduled_departure_utc']).all()
    assert not {'Tail_Number', 'tail_number', 'LateAircraftDelay'}.intersection(flights.columns)

    cleaning_summary = {
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_files': source_log, 'parse_audit': parse_log,
        'population': population.reset_index().to_dict('records'),
        'exclusions': exclusion_counts.astype(int).to_dict(),
        'treatments': treatment_log, 'origin_timezones_from_raw': timezone_by_origin,
        'eda_period': CONFIG['split']['train_start'] + ' through ' + CONFIG['split']['train_end'],
        'split_definition': CONFIG['split'],
        'test_outcome_policy': CONFIG['evaluation_extension']['test_outcome_policy'],
        'outliers': 'valid extremes retained; no automatic capping or IQR row deletion',
        'availability_note': 'Historical simulation; weather/outcome delivery and archived schedule availability are assumptions.',
        'output': 'data/processed/flights_clean.parquet',
    }
    if WRITE_OUTPUTS:
        output = ROOT / 'data/processed/flights_clean.parquet'
        output.parent.mkdir(parents=True, exist_ok=True)
        flights.to_parquet(output, engine='pyarrow', index=False, compression='zstd')
        manifest_path = ROOT / 'reports/preparation_manifest.json'
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        manifest['cleaning'] = cleaning_summary
        manifest.pop('feature_engineering', None)  # Rebuild notebook 02 after every fresh clean handoff.
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + '\n')
        print(f'Saved {output.relative_to(ROOT)} ({len(flights):,} rows).')
        print('Next: review findings, then run 02_feature_engineering.ipynb.')
    else:
        print('WRITE_OUTPUTS=False: nothing saved. Notebook 02 needs the clean handoff before running.')



def setup_feature_engineering():
    """Setup feature engineering."""
    from pathlib import Path
    import site
    if site.ENABLE_USER_SITE:
        site.addsitedir(site.getusersitepackages())
    from datetime import datetime, timezone
    import hashlib
    import json

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    import seaborn as sns


    def show_table(data, **kwargs):
        print(data.to_string(**kwargs))

    def show_plot():
        plt.tight_layout()
        plt.show()

    ROOT = Path.cwd().resolve()
    if ROOT.name == 'notebooks':
        ROOT = ROOT.parent
    if (ROOT / 'departure-delay-classification/config/project.json').is_file():
        ROOT = ROOT / 'departure-delay-classification'
    if not (ROOT / 'config/project.json').is_file():
        raise FileNotFoundError('Open this notebook inside departure-delay-classification.')
    RAW = ROOT / 'data/raw'
    CONFIG = json.loads((ROOT / 'config/project.json').read_text())
    SEED = CONFIG['sampling']['seed']
    ORIGINS = CONFIG['scope']['origin_airports']
    WRITE_OUTPUTS = True
    WEATHER_BUFFER_MINUTES = 20
    WEATHER_MAX_AGE_MINUTES = 180
    HISTORY_DAYS = CONFIG['history']['window_days']
    HISTORY_LAG_DAYS = CONFIG['history']['lag_days']
    HISTORY_PRIOR_STRENGTH = 20  # Fixed pseudo-count, not selected from final-period results.
    sns.set_theme(style='whitegrid', context='notebook', palette=['#247BA0', '#E08D3C', '#5C946E'])
    plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False})
    pd.set_option('display.max_columns', 40)

    clean_path = ROOT / 'data/processed/flights_clean.parquet'
    manifest_path = ROOT / 'reports/preparation_manifest.json'
    missing = [p.name for p in (clean_path, manifest_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing handoff: {', '.join(missing)}. Run the final save cell in notebook 01.")
    manifest = json.loads(manifest_path.read_text())
    assert 'cleaning' in manifest, 'Missing notebook-01 provenance.'
    assert manifest['cleaning']['split_definition'] == CONFIG['split'], 'Rebuild notebook 01 with the current timeline.'
    flights = pd.read_parquet(clean_path, engine='pyarrow')
    assert flights['flight_id'].is_unique
    queries = flights.loc[flights['query_eligible']].copy().reset_index(drop=True)
    queries['delayed_departure'] = queries['delayed_departure'].astype('int8')
    assert queries['split'].isin(['train', 'validation', 'test']).all()
    assert len(queries) > 0
    show_table(queries.groupby('split', observed=True).size().to_frame('query_flights'))
    return CONFIG, HISTORY_DAYS, HISTORY_LAG_DAYS, HISTORY_PRIOR_STRENGTH, ORIGINS, RAW, ROOT, SEED, WEATHER_BUFFER_MINUTES, WEATHER_MAX_AGE_MINUTES, WRITE_OUTPUTS, clean_path, flights, manifest_path, queries



def read_weather_sources(CONFIG, ORIGINS, RAW, ROOT):
    """Read weather sources."""
    WEATHER_COLUMNS = ['station', 'valid', 'tmpf', 'dwpf', 'drct', 'sknt', 'gust', 'vsby', 'p01i',
                       'skyc1', 'skyc2', 'skyc3', 'skyc4', 'skyl1', 'skyl2', 'skyl3', 'skyl4',
                       'wxcodes', 'metar']
    weather_paths = sorted((RAW / 'weather/observations/2025').glob('2025_*.csv'))
    weather_paths += sorted((RAW / 'weather/observations/2026').glob('2026_0[123]_origins.csv'))
    assert len(weather_paths) == 39, 'Expected 36 raw 2025 batches and three Q1-2026 responses.'
    weather_parts, weather_source_log = [], []
    for path in weather_paths:
        metadata = json.loads(path.with_suffix('.csv.json').read_text())
        assert metadata['request']['params']['tz'] in ['Etc/UTC', 'UTC'], f'Non-UTC source: {path.name}'
        raw = pd.read_csv(path, usecols=WEATHER_COLUMNS, dtype='string', keep_default_na=False)
        raw = raw.loc[raw['station'].str.strip().isin(ORIGINS)].copy()
        for column in WEATHER_COLUMNS:
            raw[column] = raw[column].str.strip()
        raw['weather_source_file'] = path.relative_to(ROOT).as_posix()
        weather_parts.append(raw)
        weather_source_log.append({'source': path.relative_to(ROOT).as_posix(), 'scope_raw_rows': len(raw)})
    weather = pd.concat(weather_parts, ignore_index=True)
    del weather_parts
    weather_raw_rows = len(weather)
    weather = weather.drop_duplicates(subset=WEATHER_COLUMNS, keep='first').copy()
    exact_weather_duplicates = weather_raw_rows - len(weather)
    weather['weather_observed_utc'] = pd.to_datetime(weather['valid'], utc=True, errors='coerce').astype('datetime64[ns, UTC]')
    if weather['weather_observed_utc'].isna().any():
        show_table(weather.loc[weather['weather_observed_utc'].isna()].head(10))
        raise ValueError('Unparseable weather times require review before a cutoff join.')
    conflict = weather.duplicated(['station', 'weather_observed_utc'], keep=False)
    weather_conflicts = (weather.loc[conflict].groupby(['station', 'valid']).size()
                         .rename('reports_removed').reset_index())
    approved_conflicts = pd.MultiIndex.from_tuples([tuple(key) for key in CONFIG['weather_quality']['excluded_station_times']])
    if not pd.MultiIndex.from_frame(weather_conflicts[['station', 'valid']]).isin(approved_conflicts).all():
        show_table(weather_conflicts)
        raise ValueError('New weather conflicts need review; only explicitly approved timestamps may be excluded.')
    show_table(weather_conflicts)
    weather = weather.loc[~conflict].copy()  # Exclude both versions; retain every flight in the later join.
    assert not weather.duplicated(['station', 'weather_observed_utc']).any()
    weather = weather.rename(columns={'station': 'origin'})
    show_table(pd.Series({'scope_raw_weather_rows': weather_raw_rows,
                       'identical_overlap_rows_removed': exact_weather_duplicates,
                       'conflicting_station_times_excluded': len(weather_conflicts),
                       'conflicting_report_versions_excluded': int(conflict.sum()),
                       'unique_station_time_reports': len(weather)}).to_frame('count'))
    return conflict, exact_weather_duplicates, weather, weather_conflicts, weather_raw_rows, weather_source_log



def parse_weather_measurements(weather):
    """Parse weather measurements."""
    import re

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
    show_table(pd.DataFrame(weather_treatment_log))
    return WEATHER_FEATURES, weather, weather_treatment_log



def match_and_review_weather(WEATHER_BUFFER_MINUTES, WEATHER_FEATURES, WEATHER_MAX_AGE_MINUTES, queries, weather):
    """Match and review weather."""
    assert 0 <= WEATHER_BUFFER_MINUTES < WEATHER_MAX_AGE_MINUTES
    queries['origin'] = queries['origin'].astype('string')
    weather['origin'] = weather['origin'].astype('string')
    queries['_query_order'] = np.arange(len(queries))
    queries['weather_lookup_utc'] = queries['prediction_cutoff_utc'] - pd.Timedelta(minutes=WEATHER_BUFFER_MINUTES)
    right_columns = ['origin', 'weather_observed_utc', 'weather_source_file', *WEATHER_FEATURES]
    features = pd.merge_asof(
        queries.sort_values('weather_lookup_utc'),
        weather[right_columns].sort_values('weather_observed_utc'),
        left_on='weather_lookup_utc', right_on='weather_observed_utc', by='origin',
        direction='backward', allow_exact_matches=True,
        tolerance=pd.Timedelta(minutes=WEATHER_MAX_AGE_MINUTES - WEATHER_BUFFER_MINUTES),
    ).sort_values('_query_order').drop(columns='_query_order').reset_index(drop=True)
    features['weather_age_minutes'] = (features['prediction_cutoff_utc'] - features['weather_observed_utc']).dt.total_seconds() / 60
    features['weather_report_missing'] = features['weather_observed_utc'].isna().astype('int8')
    matched = features['weather_observed_utc'].notna()
    assert len(features) == len(queries) and features['flight_id'].equals(queries['flight_id'])
    assert features.loc[matched, 'weather_age_minutes'].between(WEATHER_BUFFER_MINUTES, WEATHER_MAX_AGE_MINUTES).all()

    train_features = features.loc[features['split'].eq('train')].copy()
    coverage = train_features.groupby('origin', observed=True).agg(
        n=('flight_id', 'size'), missing_reports=('weather_report_missing', 'sum'),
        median_age_minutes=('weather_age_minutes', 'median'))
    coverage['matched_fraction'] = 1 - coverage['missing_reports'] / coverage['n']
    show_table(coverage)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    coverage['matched_fraction'].plot.bar(ax=axes[0], ylim=(0, 1))
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set(title='Cutoff-weather matching · training flights', ylabel='Flights with a qualifying report')
    sns.histplot(data=train_features, x='weather_age_minutes', bins=35, ax=axes[1])
    axes[1].set(title='Matched observation age at prediction cutoff', xlabel='Age (minutes)')
    show_plot()
    return features, train_features



def add_earlier_delay_histories(HISTORY_DAYS, HISTORY_LAG_DAYS, HISTORY_PRIOR_STRENGTH, features, flights):
    """Add earlier delay histories."""
    events = flights.loc[flights['label_eligible'],
        ['origin', 'airline', 'destination', 'departure_outcome_event_utc', 'delayed_departure']].copy()
    assert events['departure_outcome_event_utc'].notna().all()
    events['delayed_departure'] = events['delayed_departure'].astype('int8')
    history_end = features['prediction_cutoff_utc'] - pd.Timedelta(days=HISTORY_LAG_DAYS)
    history_start = history_end - pd.Timedelta(days=HISTORY_DAYS)

    def historical_window_counts(query_frame, event_frame, group_columns):
        counts = np.zeros(len(query_frame), dtype=np.int32)
        positives = np.zeros(len(query_frame), dtype=np.int32)
        latest = np.full(len(query_frame), np.iinfo('int64').min, dtype='int64')
        query_groups = query_frame.groupby(group_columns, observed=True, dropna=False).indices
        event_groups = event_frame.groupby(group_columns, observed=True, dropna=False).indices
        lower = history_start.array.asi8
        upper = history_end.array.asi8
        for key, positions in query_groups.items():
            event_positions = event_groups.get(key)
            if event_positions is None:
                continue
            block = event_frame.iloc[event_positions].sort_values('departure_outcome_event_utc')
            times = block['departure_outcome_event_utc'].array.asi8
            cumulative = np.r_[0, block['delayed_departure'].to_numpy(dtype='int64').cumsum()]
            left = np.searchsorted(times, lower[positions], side='left')
            right = np.searchsorted(times, upper[positions], side='left')
            counts[positions] = right - left
            positives[positions] = cumulative[right] - cumulative[left]
            has_history = right > left
            latest[positions[has_history]] = times[right[has_history] - 1]
        return counts, positives, pd.to_datetime(latest, utc=True)

    history_definitions = [('history_origin', ['origin']),
                           ('history_airline_origin', ['origin', 'airline']),
                           ('history_route', ['origin', 'destination'])]
    for prefix, group_columns in history_definitions:
        n, positives, latest = historical_window_counts(features, events, group_columns)
        features[prefix + '_n'] = n
        raw_rate = np.divide(positives, n, out=np.full(len(features), np.nan), where=n > 0)
        if prefix == 'history_origin':
            features[prefix + '_rate'] = raw_rate
        else:
            prior = features['history_origin_rate'].to_numpy()
            features[prefix + '_rate'] = (positives + HISTORY_PRIOR_STRENGTH * prior) / (n + HISTORY_PRIOR_STRENGTH)
        features[prefix + '_no_history'] = (n == 0).astype('int8')
        features[prefix + '_latest_event_utc'] = latest
        known = n > 0
        assert (features.loc[known, prefix + '_latest_event_utc'] < history_end.loc[known]).all()
        assert (features.loc[known, prefix + '_latest_event_utc'] >= history_start.loc[known]).all()
        assert features[prefix + '_rate'].dropna().between(0, 1).all()
    return history_definitions, history_end



def plot_history_and_traffic(SEED, features):
    """Plot history and traffic."""
    train_features = features.loc[features['split'].eq('train')].copy()
    train_features['target_label'] = train_features['delayed_departure'].map({0: 'On time / <15 min late', 1: '15+ min late'})
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    history_plot_columns = ['history_origin_rate', 'history_airline_origin_rate', 'history_route_rate',
                             'history_route_n', 'history_airline_origin_n', 'airport_scheduled_departures_prior_60m']
    history_plot_labels = ['Origin delay rate', 'Airline/origin delay rate', 'Route delay rate',
                           'Earlier route flights (28 days)', 'Earlier airline/origin flights (28 days)',
                           'Scheduled departures in prior 60 min']
    for column, label, ax in zip(history_plot_columns, history_plot_labels, axes.flat):
        sns.histplot(data=train_features, x=column, bins=40, ax=ax)
        ax.set(title=f'{label} · training', xlabel=label)
    show_plot()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.boxplot(data=train_features, x='target_label', y='history_airline_origin_rate', ax=axes[0])
    axes[0].set(title='Earlier airline/origin delay rate by query outcome', xlabel='', ylabel='Past-window delay rate')
    sns.boxplot(data=train_features, x='target_label', y='airport_scheduled_departures_prior_60m', ax=axes[1])
    axes[1].set(title='Scheduled airport traffic by query outcome', xlabel='', ylabel='Prior 60-minute scheduled departures')
    show_plot()

    scatter = train_features.sample(n=min(10_000, len(train_features)), random_state=SEED)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.scatterplot(data=scatter, x='airport_scheduled_departures_prior_60m', y='history_airline_origin_rate',
                    hue='target_label', alpha=.25, s=14, ax=ax)
    ax.set(title=f'Traffic versus earlier delay risk · sampled n={len(scatter):,}',
           xlabel='Prior 60-minute scheduled departures', ylabel='Earlier airline/origin delay rate')
    show_plot()
    return train_features



def plot_history_relationships(train_features):
    """Plot history relationships."""
    history_bins = pd.qcut(train_features['history_airline_origin_rate'], q=8, duplicates='drop')
    binned = train_features.assign(history_bin=history_bins).groupby('history_bin', observed=True).agg(
        n=('flight_id', 'size'), delayed=('delayed_departure', 'sum'),
        mean_historical_rate=('history_airline_origin_rate', 'mean'))
    binned['query_delay_rate'] = binned['delayed'] / binned['n']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(binned['mean_historical_rate'], binned['query_delay_rate'], marker='o')
    axes[0].xaxis.set_major_formatter(PercentFormatter(1))
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set(title='Earlier risk bins versus observed query delay rate',
                xlabel='Mean historical rate in training bin', ylabel='Actual training-query delay rate')
    corr_columns = ['distance_miles', 'scheduled_duration_minutes', 'wind_knots', 'visibility_miles',
                     'history_origin_rate', 'history_airline_origin_rate', 'history_route_rate',
                     'airport_scheduled_departures_prior_60m']
    corr_labels = ['Distance', 'Scheduled duration', 'Wind', 'Visibility', 'Origin history',
                   'Airline/origin history', 'Route history', 'Scheduled traffic']
    corr = train_features[corr_columns].corr().set_axis(corr_labels).set_axis(corr_labels, axis=1)
    sns.heatmap(corr, cmap='vlag', vmin=-1, vmax=1,
                annot=True, fmt='.2f', ax=axes[1], annot_kws={'size': 8})
    axes[1].set_title('Predictor correlations · training queries')
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right')
    show_plot()
    show_table(binned)



def select_features_and_samples(CONFIG, SEED, WEATHER_FEATURES, features, history_definitions):
    """Select features and samples."""
    from sklearn.model_selection import train_test_split

    CATEGORICAL_FEATURES = ['airline', 'origin', 'destination', 'weekday']
    A_NUMERIC = ['distance_miles', 'scheduled_duration_minutes', 'departure_sin', 'departure_cos', 'month_sin', 'month_cos']
    HISTORY_FEATURES = [prefix + suffix for prefix, _ in history_definitions
                         for suffix in ['_rate', '_n', '_no_history']]
    B_NUMERIC = A_NUMERIC + WEATHER_FEATURES + ['weather_age_minutes', 'weather_report_missing'] + HISTORY_FEATURES + ['airport_scheduled_departures_prior_60m']
    FEATURE_SETS = {'A': CATEGORICAL_FEATURES + A_NUMERIC,
                    'B': CATEGORICAL_FEATURES + B_NUMERIC}
    FORBIDDEN = {'delayed_departure', 'departure_delay_minutes', 'departure_delay_15_source',
        'cancelled', 'diverted', 'departure_outcome_event_utc', 'flight_id', 'flight_number',
        'Tail_Number', 'tail_number', 'LateAircraftDelay', 'weather_source_file'}
    for name, columns in FEATURE_SETS.items():
        assert len(columns) == len(set(columns))
        assert not FORBIDDEN.intersection(columns)
        assert set(columns).issubset(features.columns)
        assert not features[columns].select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
    assert set(FEATURE_SETS['A']).issubset(FEATURE_SETS['B'])

    def sample_indices(population, desired):
        if len(population) < desired:
            raise ValueError(f'Population has fewer than {desired:,} rows; clarify sample size.')
        if desired == len(population):
            return population.index
        strata = population['month'].astype(str) + '|' + population['delayed_departure'].astype(str)
        if strata.value_counts().min() < 2:
            raise ValueError('Sparse month/outcome stratum; clarify sampling policy.')
        selected, _ = train_test_split(population.index, train_size=desired,
                                        random_state=SEED, stratify=strata)
        return selected

    features['model_sample'] = False
    sample_summary = []
    for split_name in ['train', 'validation', 'test']:
        population = features.loc[features['split'].eq(split_name)]
        selected = sample_indices(population, CONFIG['sampling'][split_name + '_rows'])
        features.loc[selected, 'model_sample'] = True
        sample_summary.append({'split': split_name, 'population_rows': len(population),
                               'sample_rows': len(selected), 'seed': SEED, 'same_A_B_ids': True})

    # A separate flag covers all 12 months for the final fit AFTER validation selection.
    refit_population = features.loc[features['flight_date'].between(
        CONFIG['split']['refit_start'], CONFIG['split']['refit_end'])]
    features['refit_sample'] = features.index.isin(
        sample_indices(refit_population, CONFIG['sampling']['refit_rows']))
    assert not features.loc[features['refit_sample'], 'split'].eq('test').any()
    assert set(features.loc[features['refit_sample'], 'month']) == set(range(1, 13))
    show_table(pd.DataFrame(sample_summary + [{
        'split': 'final refit (2025 only)', 'population_rows': len(refit_population),
        'sample_rows': int(features['refit_sample'].sum()), 'seed': SEED, 'same_A_B_ids': True}]))
    show_table(pd.DataFrame({'feature_set': ['A', 'B'],
                          'categorical_features': [len(CATEGORICAL_FEATURES)] * 2,
                          'numeric_features': [len(A_NUMERIC), len(B_NUMERIC)]}))
    print('Month/outcome stratification preserves proportions; it does not balance the delay classes.')
    print('Test labels are used only by the frozen sampling rule; no test delay rates or relationships are inspected.')
    return A_NUMERIC, B_NUMERIC, CATEGORICAL_FEATURES, FEATURE_SETS, sample_summary



def preview_feature_preprocessing(A_NUMERIC, B_NUMERIC, CATEGORICAL_FEATURES, CONFIG, FEATURE_SETS, features):
    """Preview feature preprocessing."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    def make_preprocessor(feature_set='B', scale_numeric=True):
        numeric = A_NUMERIC if feature_set == 'A' else B_NUMERIC
        if feature_set not in FEATURE_SETS:
            raise ValueError('Only A and B are in the agreed scope.')
        covered = CONFIG['preprocessing']['explicit_missingness_covered_columns']
        groups = [('numeric', [c for c in numeric if c not in covered], True),
                  ('explicit_flags', [c for c in numeric if c in covered], False)]
        transformers = []
        for name, columns, indicators in groups:
            if columns:
                steps = [('imputer', SimpleImputer(strategy='median', add_indicator=indicators,
                                                   keep_empty_features=True))]
                if scale_numeric:
                    steps.append(('scaler', StandardScaler()))
                transformers.append((name, Pipeline(steps), columns))
        categories = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown', keep_empty_features=True)),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=True, dtype=np.float64)),
        ])
        transformers.append(('categorical', categories, CATEGORICAL_FEATURES))
        return ColumnTransformer(transformers, remainder='drop', sparse_threshold=1.0)

    training_rows = features['split'].eq('train') & features['model_sample']
    validation_rows = features['split'].eq('validation') & features['model_sample']
    preview_preprocessor = make_preprocessor('B', scale_numeric=True)
    X_train_preview = preview_preprocessor.fit_transform(features.loc[training_rows, FEATURE_SETS['B']])
    X_validation_preview = preview_preprocessor.transform(features.loc[validation_rows, FEATURE_SETS['B']])
    show_table(pd.DataFrame({'partition': ['training preview', 'validation preview'],
                          'rows': [X_train_preview.shape[0], X_validation_preview.shape[0]],
                          'transformed_columns': [X_train_preview.shape[1], X_validation_preview.shape[1]]}))
    assert X_train_preview.shape[1] == X_validation_preview.shape[1]
    print('Preprocessing fitted on training only. No transformed matrix is exported.')

    names = preview_preprocessor.get_feature_names_out()
    assert not any('missingindicator_' + c in name for c in CONFIG['preprocessing']['explicit_missingness_covered_columns'] for name in names)
    for matrix in [X_train_preview, X_validation_preview]:
        assert np.isfinite(matrix.data if hasattr(matrix, 'toarray') else matrix).all()
    print('No repeated condition/ceiling indicators; transformed values are finite.')
    return training_rows



def plot_preprocessing_comparison(features, training_rows):
    """Plot preprocessing comparison."""
    # Before/after plots use a separate small numerical preview for readability.
    transform_plot_columns = ['distance_miles', 'wind_knots', 'airport_scheduled_departures_prior_60m']
    train_input = features.loc[training_rows, transform_plot_columns]
    plot_recipe = Pipeline([('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
                            ('scaler', StandardScaler())])
    train_standardized = pd.DataFrame(plot_recipe.fit_transform(train_input),
                                      columns=transform_plot_columns, index=train_input.index)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(data=train_input, ax=axes[0])
    axes[0].set(title='Before transformation · shared training sample', ylabel='Original units')
    sns.boxplot(data=train_standardized, ax=axes[1])
    axes[1].set(title='After training-only median fill + standardization', ylabel='Standardized units')
    for ax in axes:
        ax.tick_params(axis='x', rotation=20)
    show_plot()

    traffic_original = train_input['airport_scheduled_departures_prior_60m']
    traffic_log = np.log1p(traffic_original)
    show_table(pd.DataFrame({'original_traffic': traffic_original, 'log1p_traffic': traffic_log}).head())
    show_table(pd.DataFrame({'representation': ['original traffic', 'log1p traffic'],
                        'skewness': [traffic_original.skew(), traffic_log.skew()]}), index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    sns.histplot(traffic_original, bins=40, kde=traffic_original.nunique() > 1, ax=axes[0])
    axes[0].set(title='Scheduled traffic · original representation', xlabel='Prior 60-minute departures')
    sns.histplot(traffic_log, bins=40, kde=traffic_log.nunique() > 1, ax=axes[1])
    axes[1].set(title='log1p traffic · illustration only', xlabel='log(1 + prior 60-minute departures)')
    show_plot()
    print('Transformation decision: cyclic month/clock coordinates and one-hot weekday/airline/airport categories are the initial recipe. The log1p illustration is not added to A/B automatically. No target transformation or outlier capping is applied.')



def save_feature_handoff(CONFIG, FEATURE_SETS, HISTORY_DAYS, HISTORY_LAG_DAYS, HISTORY_PRIOR_STRENGTH, ROOT, WEATHER_BUFFER_MINUTES, WEATHER_MAX_AGE_MINUTES, WRITE_OUTPUTS, clean_path, conflict, exact_weather_duplicates, features, history_definitions, history_end, manifest_path, queries, sample_summary, weather_conflicts, weather_raw_rows, weather_source_log, weather_treatment_log):
    """Save feature handoff."""
    feature_dictionary = CONFIG['feature_dictionary']
    assert [item['feature'] for item in feature_dictionary] == FEATURE_SETS['B']
    show_table(pd.DataFrame(feature_dictionary))

    assert features['flight_id'].is_unique and len(features) == len(queries)
    assert features['flight_id'].equals(queries['flight_id'])
    assert features['delayed_departure'].equals(queries['delayed_departure'])
    assert not features.loc[features['model_sample'], 'split'].eq('context').any()
    assert (features['prediction_cutoff_utc'] < features['scheduled_departure_utc']).all()
    for prefix, _ in history_definitions:
        mask = features[prefix + '_n'].gt(0)
        assert (features.loc[mask, prefix + '_latest_event_utc'] < history_end.loc[mask]).all()

    def file_sha256(path):
        with path.open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()

    sample_cohort = features.loc[features['model_sample'], ['flight_id', 'delayed_departure']].sort_values('flight_id')
    cohort_fingerprint = hashlib.sha256(pd.util.hash_pandas_object(sample_cohort, index=False).values.tobytes()).hexdigest()
    audit_columns = ['flight_id', 'source_file', 'source_row_number', 'flight_date', 'split', 'model_sample', 'refit_sample', 'month',
        'scheduled_departure_utc', 'prediction_cutoff_utc', 'departure_time_band', 'delayed_departure',
        'weather_lookup_utc', 'weather_observed_utc', 'weather_source_file']
    audit_columns += [prefix + '_latest_event_utc' for prefix, _ in history_definitions]
    keep_columns = list(dict.fromkeys(audit_columns + FEATURE_SETS['B']))
    feature_output = features[keep_columns].copy()

    if WRITE_OUTPUTS:
        output = ROOT / 'data/processed/features.parquet'
        feature_output.to_parquet(output, engine='pyarrow', index=False, compression='zstd')
        manifest = json.loads(manifest_path.read_text())
        manifest['feature_engineering'] = {
            'completed_at_utc': datetime.now(timezone.utc).isoformat(),
            'input': 'data/processed/flights_clean.parquet', 'input_sha256': file_sha256(clean_path),
            'output': 'data/processed/features.parquet', 'output_sha256': file_sha256(output),
            'rows': len(feature_output), 'feature_sets': FEATURE_SETS, 'feature_dictionary': feature_dictionary,
            'split_definition': CONFIG['split'],
            'shared_samples': sample_summary, 'sample_cohort_fingerprint': cohort_fingerprint,
            'final_refit': {'rows': int(features['refit_sample'].sum()),
                'cohort_fingerprint': hashlib.sha256(pd.util.hash_pandas_object(
                    features.loc[features['refit_sample'], ['flight_id', 'delayed_departure']].sort_values('flight_id'),
                    index=False).values.tobytes()).hexdigest(),
                'policy': CONFIG['sampling']['refit_policy']},
            'preprocessing': CONFIG['preprocessing'],
            'test_outcome_policy': CONFIG['evaluation_extension']['test_outcome_policy'],
            'weather': {'source_batches': weather_source_log, 'raw_scope_rows': weather_raw_rows,
                'exact_duplicate_payloads_removed': exact_weather_duplicates,
                'conflicting_station_time_policy': CONFIG['weather_quality']['conflict_policy'],
                'excluded_conflicting_station_times': weather_conflicts.to_dict('records'),
                'conflicting_report_versions_excluded': int(conflict.sum()),
                'unreviewed_conflicts': 'stop and clarify',
                'buffer_minutes': WEATHER_BUFFER_MINUTES, 'max_age_minutes': WEATHER_MAX_AGE_MINUTES,
                'treatments': weather_treatment_log,
                'cloud_extraction': 'All METAR body layers before RMK, with structured sky-column fallback; unknown ceiling heights remain missing'}, 
            'history': {'window_days': HISTORY_DAYS, 'lag_days': HISTORY_LAG_DAYS,
                'subgroup_prior_strength': HISTORY_PRIOR_STRENGTH,
                'event_definition': 'scheduled departure UTC + reported departure delay',
                'scope': 'ATL/ORD/DFW; airline-origin and route-origin histories',
                'evaluation_policy': 'Frozen rolling rule; earlier 2026 outcomes may enter later test histories only after the seven-day lag. No classifier is refitted on 2026.'},
            'traffic': 'All scheduled origin departures in [query scheduled departure-60m, query scheduled departure)',
            'study_design': 'Historical simulation with unverified weather/outcome receipt and schedule-availability assumptions',
            'preprocessor_preview_only': True, 'classifiers_trained': False,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + '\n')
        print(f'Saved {output.relative_to(ROOT)}: {len(feature_output):,} queries, shared sample flags included.')
        print('Prepared data must be reviewed before notebook 03 model training.')
    else:
        print('WRITE_OUTPUTS=False: feature table and manifest were not written.')



def setup_candidate_experiment():
    """Setup candidate experiment."""
    from pathlib import Path
    import site, sys, json, hashlib
    if site.ENABLE_USER_SITE:
        site.addsitedir(site.getusersitepackages())
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    import seaborn as sns

    ROOT = Path.cwd().resolve()
    if ROOT.name == 'notebooks':
        ROOT = ROOT.parent
    if (ROOT / 'departure-delay-classification/config/project.json').is_file():
        ROOT /= 'departure-delay-classification'
    CONFIG = json.loads((ROOT / 'config/project.json').read_text())
    PREP = json.loads((ROOT / 'reports/preparation_manifest.json').read_text())['feature_engineering']
    FROZEN = json.loads((ROOT / 'reports/improvements/selection.json').read_text())
    SEED, LAG_DAYS, LONG_DAYS, PRIOR = 42, 7, 42, 20

    def extension_sha(path):
        with path.open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()

    protected_paths = ['data/processed/features.parquet', 'data/processed/flights_clean.parquet',
                      'reports/preparation_manifest.json', 'reports/improvements/selection.json',
                      'models/improved_flight_delay_models.joblib']
    protected_before = {p: extension_sha(ROOT / p) for p in protected_paths}
    assert protected_before['data/processed/features.parquet'] == FROZEN['feature_input_sha256']
    assert extension_sha(ROOT / 'scripts/improvement_features.py') == FROZEN['helper_sha256']
    sys.path.insert(0, str(ROOT / 'scripts'))
    from improvement_features import (TIME_FEATURES, FLIGHT_FEATURES, WEATHER_NEW,
                                      load_weather, add_time, add_weather, add_histories)

    development = pd.read_parquet(ROOT / PREP['output'], filters=[
        ('flight_date', '>=', pd.Timestamp('2025-01-01')),
        ('flight_date', '<', pd.Timestamp('2026-01-01'))]).reset_index(drop=True)
    context = pd.read_parquet(ROOT / 'data/processed/flights_clean.parquet', columns=[
        'flight_id', 'flight_date', 'origin', 'airline', 'destination', 'flight_number',
        'scheduled_departure_minutes', 'scheduled_departure_utc', 'label_eligible',
        'departure_outcome_event_utc', 'delayed_departure'],
        filters=[('flight_date', '<', pd.Timestamp('2026-01-01'))])
    assert development['flight_id'].is_unique and context['flight_id'].is_unique
    original_ids = development['flight_id'].copy()
    train_mask = development['split'].eq('train')
    assert development['flight_date'].dt.year.eq(2025).all()
    assert development['split'].isin(['train', 'validation']).all()
    assert context['flight_date'].max() < pd.Timestamp('2026-01-01')
    sns.set_theme(style='whitegrid', palette=['#247BA0', '#E08D3C', '#5C946E'])
    plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False})

    def extension_table(frame, **kwargs):
        print(frame.to_string(**kwargs))

    def extension_plot():
        plt.tight_layout()
        plt.show()

    extension_table(development.groupby('split', observed=True).size().to_frame('retained_flights'))
    return CONFIG, FROZEN, LAG_DAYS, LONG_DAYS, PREP, PRIOR, ROOT, SEED, context, development, extension_plot, extension_sha, extension_table, original_ids, protected_before, train_mask



def make_history_window_function(LAG_DAYS):
    """Make history window function."""
    def past_window_totals(query, reference, groups, time_column, value_column, days):
        end = (pd.to_datetime(query['prediction_cutoff_utc'], utc=True)
               .astype('datetime64[ns, UTC]') - pd.Timedelta(days=LAG_DAYS))
        start = end - pd.Timedelta(days=days)
        n = np.zeros(len(query), dtype='int32')
        total = np.zeros(len(query), dtype='float64')
        latest = np.full(len(query), np.iinfo('int64').min, dtype='int64')
        reference_groups = reference.groupby(groups, observed=True, dropna=False).indices
        for key, positions in query.groupby(groups, observed=True, dropna=False).indices.items():
            if key not in reference_groups:
                continue
            block = reference.iloc[reference_groups[key]].sort_values(time_column)
            times = pd.to_datetime(block[time_column], utc=True).astype('datetime64[ns, UTC]').array.asi8
            values = block[value_column].to_numpy(dtype='float64')
            assert np.isfinite(values).all()
            cumulative = np.r_[0., values.cumsum()]
            left = np.searchsorted(times, start.array.asi8[positions], side='left')
            right = np.searchsorted(times, end.array.asi8[positions], side='left')
            n[positions], total[positions] = right - left, cumulative[right] - cumulative[left]
            present = right > left
            latest[positions[present]] = times[right[present] - 1]
        latest = pd.Series(pd.to_datetime(latest, utc=True), index=query.index)
        present = n > 0
        assert (latest.loc[present] >= start.loc[present]).all()
        assert (latest.loc[present] < end.loc[present]).all()
        return n, total, latest
    return past_window_totals



def create_relative_traffic(LAG_DAYS, LONG_DAYS, ROOT, candidate_features, context, development, extension_table, past_window_totals, window_audit):
    """Create relative traffic."""
    hour_blocks = []
    traffic_query = development[['origin', 'prediction_cutoff_utc']].copy()
    traffic_query['local_weekday'] = 0
    traffic_query['local_hour'] = 0
    zones = json.loads((ROOT / 'reports/preparation_manifest.json').read_text())['cleaning']['origin_timezones_from_raw']
    for origin, zone in zones.items():
        schedules = context.loc[context['origin'].eq(origin)]
        start = schedules['flight_date'].min().tz_localize(zone).tz_convert('UTC')
        stop = (schedules['flight_date'].max() + pd.Timedelta(days=1)).tz_localize(zone).tz_convert('UTC')
        hours = pd.date_range(start, stop, freq='h', inclusive='left')
        counts = schedules.groupby(schedules['scheduled_departure_utc'].dt.floor('h')).size().reindex(hours, fill_value=0)
        local = hours.tz_convert(zone)
        hour_blocks.append(pd.DataFrame({'origin': origin, 'hour_utc': hours,
            'hour_end_utc': hours + pd.Timedelta(hours=1), 'local_weekday': local.dayofweek,
            'local_hour': local.hour, 'scheduled_departures': counts.to_numpy()}))
        positions = development['origin'].eq(origin)
        local_query = development.loc[positions, 'scheduled_departure_utc'].dt.tz_convert(zone)
        traffic_query.loc[positions, 'local_weekday'] = local_query.dt.dayofweek.to_numpy()
        traffic_query.loc[positions, 'local_hour'] = local_query.dt.hour.to_numpy()
    hour_reference = pd.concat(hour_blocks, ignore_index=True)
    assert not hour_reference.duplicated(['origin', 'hour_utc']).any()
    reference_start = hour_reference.groupby('origin', observed=True)['hour_utc'].min()
    needed_start = development['prediction_cutoff_utc'] - pd.Timedelta(days=LAG_DAYS + LONG_DAYS)
    assert (needed_start >= development['origin'].map(reference_start)).all(), 'Incomplete schedule warm-up.'

    n, total, latest = past_window_totals(traffic_query, hour_reference,
        ['origin', 'local_weekday', 'local_hour'], 'hour_end_utc', 'scheduled_departures', LONG_DAYS)
    mean = np.divide(total, n, out=np.full(len(n), np.nan), where=n > 0)
    keys = pd.MultiIndex.from_arrays([development['origin'], development['scheduled_departure_utc'].dt.floor('h')])
    current = hour_reference.set_index(['origin', 'hour_utc'])['scheduled_departures'].reindex(keys).to_numpy()
    assert np.isfinite(current).all()
    candidate_features['traffic_hour_departures'] = current
    candidate_features['traffic_expected_hour'] = mean
    candidate_features['traffic_relative_hour'] = np.divide(current, mean, out=np.full(len(n), np.nan), where=mean > 0)
    candidate_features['traffic_excess_hour'] = current - mean
    candidate_features['traffic_reference_n'] = n
    candidate_features['traffic_reference_missing_or_zero'] = ((n == 0) | (mean == 0)).astype('int8')
    candidate_features['traffic_relative_log1p'] = np.log1p(candidate_features['traffic_relative_hour'])
    TRAFFIC_CANDIDATES = ['traffic_hour_departures', 'traffic_expected_hour', 'traffic_relative_log1p',
                          'traffic_excess_hour', 'traffic_reference_missing_or_zero']
    window_audit['traffic_reference'] = latest
    extension_table(candidate_features.describe().T)
    print('Zero-traffic hours are retained:', int(hour_reference['scheduled_departures'].eq(0).sum()))
    return TRAFFIC_CANDIDATES, zones



def create_weather_traffic_interactions(SEED, candidate_features, development, extension_plot, extension_table, traffic_plot, train_mask):
    """Create weather traffic interactions."""
    relative = candidate_features['traffic_relative_log1p']
    weather_interactions = {
        'traffic_rain_interaction': relative * development['rain_reported'],
        'traffic_thunder_interaction': relative * development['thunder_reported'],
        'traffic_wind_interaction': relative * development['wind_knots'],
        'traffic_visibility_interaction': relative / (1 + development['visibility_miles']),
    }
    for column, values in weather_interactions.items():
        candidate_features[column] = values
    WEATHER_TRAFFIC_CANDIDATES = list(weather_interactions)
    weather_plot = traffic_plot.join(development.loc[train_mask, ['rain_reported', 'thunder_reported', 'visibility_miles', 'wind_knots']])
    known = weather_plot[['rain_reported', 'thunder_reported', 'visibility_miles']].notna().all(axis=1)
    weather_plot['weather_group'] = np.select([
        weather_plot['thunder_reported'].eq(1), weather_plot['rain_reported'].eq(1),
        weather_plot['visibility_miles'].lt(3), known],
        ['Thunder reported', 'Rain reported', 'Visibility <3 miles', 'No listed condition'], default='Incomplete weather')
    weather_plot['traffic_band'] = pd.cut(weather_plot['traffic_relative_hour'],
        [-np.inf, .75, 1, 1.25, np.inf], labels=['<0.75', '0.75–1', '1–1.25', '>1.25'])
    weather_by_traffic = weather_plot.groupby(['weather_group', 'traffic_band'], observed=True).agg(
        flights=('delayed_departure', 'size'), delay_rate=('delayed_departure', 'mean'))
    rates = weather_by_traffic['delay_rate'].unstack('traffic_band')
    counts = weather_by_traffic['flights'].unstack('traffic_band').reindex_like(rates)
    labels = rates.map(lambda v: f'{v:.1%}' if pd.notna(v) else '').astype(str)
    labels += counts.map(lambda v: f'\nn={int(v):,}' if pd.notna(v) else '')
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.heatmap(rates, mask=counts.lt(50) | counts.isna(), annot=labels, fmt='', cmap='YlOrRd',
                vmin=0, vmax=1, annot_kws={'fontsize': 8}, ax=axes[0], cbar_kws={'label': 'Observed delay rate'})
    axes[0].set(title='Weather × relative demand · training only', xlabel='Relative scheduled-demand band', ylabel='Reported weather grouping')
    scatter = weather_plot.sample(min(10_000, len(weather_plot)), random_state=SEED)
    sns.scatterplot(data=scatter, x='traffic_relative_log1p', y='wind_knots', hue='outcome', alpha=.25, s=12, ax=axes[1])
    axes[1].set(title='Wind and demand · 10,000 training flights', xlabel='log(1 + relative scheduled demand)', ylabel='Reported wind speed (knots)')
    axes[1].legend(title='', fontsize=8)
    extension_plot()
    extension_table(weather_by_traffic)
    return WEATHER_TRAFFIC_CANDIDATES



def create_alternative_histories(LAG_DAYS, LONG_DAYS, PRIOR, candidate_features, context, development, events, past_window_totals, window_audit):
    """Create alternative histories."""
    HISTORY_CANDIDATES = []
    history_groups = {'origin': ['origin'], 'airline_origin': ['origin', 'airline'],
                      'route': ['origin', 'destination']}
    earliest_context_date = context['flight_date'].min().tz_localize('UTC') + pd.Timedelta(days=1)
    assert (development['prediction_cutoff_utc'] - pd.Timedelta(days=LAG_DAYS + LONG_DAYS)
            > earliest_context_date).all(), 'Incomplete outcome warm-up.'
    for days in [7, LONG_DAYS]:
        for name, groups in history_groups.items():
            n, delayed, latest = past_window_totals(development, events, groups,
                'departure_outcome_event_utc', 'delayed_departure', days)
            prefix = f'history_{name}_{days}d'
            if name == 'origin':
                rate = np.divide(delayed, n, out=np.full(len(n), np.nan), where=n > 0)
            else:
                prior = candidate_features[f'history_origin_{days}d_rate'].to_numpy()
                rate = (delayed + PRIOR * prior) / (n + PRIOR)
            candidate_features[prefix + '_rate'] = rate
            candidate_features[prefix + '_n'] = n
            candidate_features[prefix + '_no_history'] = (n == 0).astype('int8')
            HISTORY_CANDIDATES += [prefix + suffix for suffix in ['_rate', '_n', '_no_history']]
            window_audit[prefix] = latest
            assert pd.Series(rate).dropna().between(0, 1).all()
    for name in history_groups:
        column = f'history_{name}_7d_minus_{LONG_DAYS}d'
        candidate_features[column] = (candidate_features[f'history_{name}_7d_rate']
                                      - candidate_features[f'history_{name}_{LONG_DAYS}d_rate'])
        HISTORY_CANDIDATES.append(column)
    AUDIT_ONLY = ['traffic_relative_hour', 'traffic_reference_n',
                  'history_origin_7d_no_history', 'history_origin_42d_no_history']
    HISTORY_CANDIDATES = [c for c in HISTORY_CANDIDATES if c not in AUDIT_ONLY]
    print(f'Created 21 history columns; {len(HISTORY_CANDIDATES)} are candidate predictors.')
    print('Origin no-history flags are audit-only: they are constant with the available complete warm-up.')
    return AUDIT_ONLY, HISTORY_CANDIDATES, history_groups



def review_candidate_features(AUDIT_ONLY, HISTORY_CANDIDATES, TRAFFIC_CANDIDATES, WEATHER_TRAFFIC_CANDIDATES, candidate_features, extension_plot, extension_table, train_mask):
    """Review candidate features."""
    CANDIDATE_GROUPS = {'relative_traffic': TRAFFIC_CANDIDATES,
        'weather_traffic': WEATHER_TRAFFIC_CANDIDATES, 'multiscale_history': HISTORY_CANDIDATES}
    NEW_FEATURES = [c for columns in CANDIDATE_GROUPS.values() for c in columns]
    candidate_profile = pd.DataFrame({
        'group': {c: group for group, columns in CANDIDATE_GROUPS.items() for c in columns},
        'training_missing_pct': 100 * candidate_features.loc[train_mask].isna().mean(),
        'training_unique': candidate_features.loc[train_mask].nunique()})
    candidate_profile.loc[AUDIT_ONLY, 'group'] = 'audit_only'
    extension_table(candidate_profile)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    candidate_profile['training_missing_pct'].sort_values().plot.barh(ax=axes[0], fontsize=8)
    axes[0].set(title='Candidate missingness · training only', xlabel='Missing (%)', ylabel='')
    correlation_columns = ['traffic_relative_log1p', 'traffic_wind_interaction', 'traffic_visibility_interaction',
        'history_origin_7d_rate', 'history_origin_42d_rate', 'history_airline_origin_7d_minus_42d',
        'history_route_7d_minus_42d']
    correlation_labels = ['log(1 + relative traffic)', 'Log traffic × wind', 'Log traffic / visibility',
        'Origin history 7d', 'Origin history 42d', 'Airline/origin change', 'Route change']
    corr = candidate_features.loc[train_mask, correlation_columns].corr()
    corr.index = corr.columns = correlation_labels
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='vlag', vmin=-1, vmax=1, square=True,
                annot_kws={'fontsize': 8}, ax=axes[1])
    axes[1].set_title('Candidate predictor correlations · training only')
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right')
    extension_plot()
    return NEW_FEATURES



def check_candidate_handoff(CURRENT_B, HISTORY_CANDIDATES, NEW_FEATURES, ROOT, TRAFFIC_CANDIDATES, WEATHER_TRAFFIC_CANDIDATES, candidate_features, development, extension_sha, extension_table, original_ids, protected_before):
    """Check candidate handoff."""
    assert len(NEW_FEATURES) == len(set(NEW_FEATURES)) == 28
    assert not set(NEW_FEATURES).intersection(development.columns)
    assert development['flight_id'].equals(original_ids)
    assert candidate_features.index.equals(development.index)
    assert not np.isinf(candidate_features.to_numpy(dtype=float)).any()
    for feature, source in [('traffic_rain_interaction', 'rain_reported'),
                            ('traffic_thunder_interaction', 'thunder_reported'),
                            ('traffic_wind_interaction', 'wind_knots'),
                            ('traffic_visibility_interaction', 'visibility_miles')]:
        assert candidate_features.loc[development[source].isna(), feature].isna().all()

    candidate_feature_table = pd.concat([development, candidate_features], axis=1)
    CANDIDATE_FEATURE_SETS = {'current_66': CURRENT_B,
        'current_plus_traffic': CURRENT_B + TRAFFIC_CANDIDATES,
        'current_plus_traffic_weather': CURRENT_B + TRAFFIC_CANDIDATES + WEATHER_TRAFFIC_CANDIDATES,
        'current_plus_history': CURRENT_B + HISTORY_CANDIDATES,
        'current_plus_all': CURRENT_B + NEW_FEATURES}
    for columns in CANDIDATE_FEATURE_SETS.values():
        assert len(columns) == len(set(columns))
        assert set(columns).issubset(candidate_feature_table.columns)
        assert not {'delayed_departure', 'flight_id', 'flight_number', 'departure_outcome_event_utc',
                    'prediction_cutoff_utc', 'tail_number', 'Tail_Number'}.intersection(columns)
        assert not candidate_feature_table[columns].select_dtypes(include=['datetime', 'datetimetz']).shape[1]
    assert all(extension_sha(ROOT / p) == checksum for p, checksum in protected_before.items())
    extension_table(pd.DataFrame({'predictors': {name: len(columns) for name, columns in CANDIDATE_FEATURE_SETS.items()}}))
    print(f'{len(candidate_feature_table):,} development flights retained; no 2026 rows, no saved datasets, no fitted classifiers.')
    print('Next: compare candidate groups on chronological 2025 validation. F1 improvement is not yet established.')
    return candidate_feature_table



def setup_broader_experiments(ROOT, candidate_feature_table, development, extension_sha, protected_before):
    """Setup broader experiments."""
    assert candidate_feature_table is not None, 'Run the preceding Additional feature engineering appendix first.'
    from pandas.tseries.holiday import USFederalHolidayCalendar
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer, MissingIndicator
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix
    from datetime import datetime, timezone
    from time import perf_counter

    broad_features = pd.DataFrame(index=development.index)
    BROAD_GROUPS = {}
    BROAD_BINARY = []
    broad_inputs_before = dict(protected_before)
    for relative in ['models/flight_delay_models.joblib', 'reports/model_selection.json',
                     'reports/improvements/test_results.json', 'tableau/exports/improved/export_manifest.json']:
        broad_inputs_before[relative] = extension_sha(ROOT / relative)
    BROAD_SEED = 42
    print('Feature groups will be compared on identical, natural-proportion 2025 flight samples.')
    return BROAD_BINARY, BROAD_GROUPS, BROAD_SEED, broad_features, broad_inputs_before



def create_calendar_candidates(BROAD_BINARY, BROAD_GROUPS, broad_features, development):
    """Create calendar candidates."""
    dates = development['flight_date'].dt.normalize()
    holidays = USFederalHolidayCalendar().holidays('2024-01-01', '2026-12-31')
    day_values = dates.to_numpy(dtype='datetime64[D]')
    holiday_values = holidays.to_numpy(dtype='datetime64[D]')
    before = np.searchsorted(holiday_values, day_values, side='right') - 1
    after = np.searchsorted(holiday_values, day_values, side='left')
    calendar_values = {
        'calendar_weekend': development['weekday'].ge(5).astype('int8'),
        'calendar_weekday_sin': np.sin(2*np.pi*development['weekday']/7),
        'calendar_weekday_cos': np.cos(2*np.pi*development['weekday']/7),
        'calendar_year_day_sin': np.sin(2*np.pi*(dates.dt.dayofyear-1)/365.2425),
        'calendar_year_day_cos': np.cos(2*np.pi*(dates.dt.dayofyear-1)/365.2425),
        'calendar_month_start': dates.dt.is_month_start.astype('int8'),
        'calendar_month_end': dates.dt.is_month_end.astype('int8'),
        'days_since_federal_holiday': (day_values-holiday_values[before]).astype(int),
        'days_until_federal_holiday': (holiday_values[after]-day_values).astype(int),
        'calendar_federal_holiday': dates.isin(holidays).astype('int8'),
    }
    calendar_values['calendar_holiday_within3d'] = (np.minimum(
        calendar_values['days_since_federal_holiday'], calendar_values['days_until_federal_holiday']) <= 3).astype('int8')
    for c, values in calendar_values.items():
        broad_features[c] = values
    BROAD_GROUPS['calendar_holidays'] = list(calendar_values)
    BROAD_BINARY += ['calendar_weekend', 'calendar_month_start', 'calendar_month_end',
                     'calendar_federal_holiday', 'calendar_holiday_within3d']
    period = (development['scheduled_departure_minutes']//240).astype('string')
    category_values = {
        'day_of_month_cat': dates.dt.day.astype('string'),
        'departure_hour_cat': (development['scheduled_departure_minutes']//60).astype('string'),
        'distance_band_cat': pd.cut(development['distance_miles'], [0,500,1000,np.inf],
                                    labels=['short_0_500', 'medium_500_1000', 'long_1000_plus']).astype('string'),
        'route_cat': development['origin'] + '>' + development['destination'],
        'airline_origin_cat': development['airline'] + '@' + development['origin'],
        'origin_period_cat': development['origin'] + '@' + period,
    }
    for c, values in category_values.items():
        broad_features[c] = values
    BROAD_GROUPS['categories_combinations'] = list(category_values)
    print('2025 recurring federal holidays:', holidays[holidays.year == 2025].strftime('%Y-%m-%d').tolist())
    return BROAD_BINARY



def create_route_and_schedule_candidates(BROAD_BINARY, BROAD_GROUPS, ROOT, broad_features, development, zones):
    """Create route and schedule candidates."""
    airports = pd.read_csv(ROOT / 'data/raw/reference/airports/airports.csv', dtype='string')
    needed_airports = set(development['origin']) | set(development['destination'])
    airports = airports.loc[airports['iata_code'].isin(needed_airports)].set_index('iata_code')
    assert airports.index.is_unique, 'Ambiguous airport identifiers require review.'
    coords = {}
    for name in ['latitude_deg', 'longitude_deg', 'elevation_ft']:
        values = pd.to_numeric(airports[name], errors='coerce')
        coords['origin_'+name] = development['origin'].map(values)
        coords['destination_'+name] = development['destination'].map(values)
    lat1, lon1, lat2, lon2 = [np.deg2rad(coords[k]) for k in [
        'origin_latitude_deg', 'origin_longitude_deg', 'destination_latitude_deg', 'destination_longitude_deg']]
    dlon = lon2-lon1
    angle = np.arctan2(np.sin(dlon)*np.cos(lat2),
                       np.cos(lat1)*np.sin(lat2)-np.sin(lat1)*np.cos(lat2)*np.cos(dlon))
    haversine = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    great_circle = 3958.7613 * 2 * np.arcsin(np.sqrt(haversine.clip(0,1)))
    route_values = {
        'route_direction_east': np.sin(angle).where(great_circle.gt(0)),
        'route_direction_north': np.cos(angle).where(great_circle.gt(0)),
        'route_great_circle_miles': great_circle,
        'route_published_distance_ratio': development['distance_miles']/great_circle.where(great_circle.gt(0)),
        'destination_latitude': coords['destination_latitude_deg'],
        'destination_longitude': coords['destination_longitude_deg'],
        'destination_elevation_ft': coords['destination_elevation_ft'],
        'scheduled_minutes_per100miles': 100*development['scheduled_duration_minutes']/development['distance_miles'].where(development['distance_miles'].gt(0)),
    }
    for c, values in route_values.items():
        broad_features[c] = values
    BROAD_GROUPS['route_geometry'] = list(route_values)

    schedule = pd.read_parquet(ROOT / 'data/processed/flights_clean.parquet', columns=[
        'flight_id','flight_date','origin','airline','destination','scheduled_departure_utc'],
        filters=[('flight_date','<',pd.Timestamp('2026-01-03'))])
    schedule['local_day'] = pd.NaT
    for origin, zone in zones.items():
        mask = schedule['origin'].eq(origin)
        schedule.loc[mask,'local_day'] = schedule.loc[mask,'scheduled_departure_utc'].dt.tz_convert(zone).dt.tz_localize(None).dt.normalize().to_numpy()
    day_groups = ['origin','local_day']
    airline_groups = day_groups+['airline']
    route_groups = airline_groups+['destination']
    schedule['origin_day_flights'] = schedule.groupby(day_groups,observed=True)['flight_id'].transform('size')
    schedule['airline_origin_day_flights'] = schedule.groupby(airline_groups,observed=True)['flight_id'].transform('size')
    schedule['airline_route_day_flights'] = schedule.groupby(route_groups,observed=True)['flight_id'].transform('size')
    schedule['airline_origin_day_share'] = schedule['airline_origin_day_flights']/schedule['origin_day_flights']
    schedule['airline_route_day_share'] = schedule['airline_route_day_flights']/schedule['origin_day_flights']
    rank = schedule.groupby(day_groups,observed=True)['scheduled_departure_utc'].rank(method='min')-1
    schedule['origin_day_schedule_progress'] = rank/(schedule['origin_day_flights']-1).clip(lower=1)
    first = schedule.groupby(route_groups,observed=True)['scheduled_departure_utc'].transform('min')
    schedule['airline_route_first_departure_flag'] = schedule['scheduled_departure_utc'].eq(first).astype('int8')
    schedule['minutes_since_airline_route_first'] = (schedule['scheduled_departure_utc']-first).dt.total_seconds()/60
    unique_times = schedule[route_groups+['scheduled_departure_utc']].drop_duplicates().sort_values('scheduled_departure_utc')
    unique_times['prior_same_route_schedule_gap_min'] = unique_times.groupby(route_groups,observed=True)['scheduled_departure_utc'].diff().dt.total_seconds()/60
    schedule = schedule.merge(unique_times,on=route_groups+['scheduled_departure_utc'],validate='many_to_one',how='left')
    structure = ['origin_day_flights','airline_origin_day_flights','airline_route_day_flights',
        'airline_origin_day_share','airline_route_day_share','origin_day_schedule_progress',
        'airline_route_first_departure_flag','minutes_since_airline_route_first','prior_same_route_schedule_gap_min']
    lookup = schedule.set_index('flight_id')
    assert lookup.index.is_unique
    for c in structure:
        broad_features[c] = development['flight_id'].map(lookup[c])
    BROAD_GROUPS['schedule_structure'] = structure
    BROAD_BINARY += ['airline_route_first_departure_flag']
    return BROAD_BINARY, schedule



def create_arrival_traffic_candidates(BROAD_GROUPS, ROOT, broad_features, development, zones):
    """Create arrival traffic candidates."""
    station_zones = {}
    for path in (ROOT/'data/raw/weather/stations').glob('*.geojson'):
        for feature in json.loads(path.read_text())['features']:
            item = feature['properties']
            if item.get('tzname'):
                station_zones[item['sid']] = item['tzname']
    reference_airports = pd.read_csv(ROOT/'data/raw/reference/airports/airports.csv',dtype='string')
    origin_zones = {}
    for _, airport in reference_airports.loc[reference_airports['iata_code'].notna()].iterrows():
        options = {station_zones[key] for c in ['iata_code','icao_code','gps_code'] if pd.notna(airport[c])
                   for key in [airport[c],airport[c][1:] if airport[c].startswith('K') else airport[c]] if key in station_zones}
        if len(options)==1:
            origin_zones[airport['iata_code']] = options.pop()
    origin_zones.update({'PBI':'America/New_York','STT':'America/St_Thomas','STX':'America/St_Thomas'})
    arrival_paths = [ROOT/'data/raw/flights/bts/2024_context/csv/2024_12.csv']
    arrival_paths += sorted((ROOT/'data/raw/flights/bts/2025/csv').glob('*.csv'))
    arrival_paths += [ROOT/'data/raw/flights/bts/2026/csv/2026_01.csv']
    incoming_parts, arrival_audit = [], []
    for path in arrival_paths:
        raw = pd.read_csv(path,usecols=['FlightDate','Origin','Dest','CRSDepTime','CRSArrTime','CRSElapsedTime'])
        raw = raw.loc[raw['Dest'].isin(zones)].copy()
        raw['origin_zone'] = raw['Origin'].map(origin_zones)
        assert raw['origin_zone'].notna().all(), raw.loc[raw['origin_zone'].isna(),'Origin'].unique()
        dep, arr, duration = [pd.to_numeric(raw[c],errors='coerce') for c in ['CRSDepTime','CRSArrTime','CRSElapsedTime']]
        for clock in [dep,arr]:
            assert ((clock.mod(1)==0)&clock.between(0,2400)&clock.mod(100).lt(60)).all()
        assert duration.gt(0).all()
        naive = pd.to_datetime(raw['FlightDate'])+pd.to_timedelta(dep//100*60+dep%100,unit='m')
        departure_options = []
        for ambiguous in [True,False]:
            utc = pd.Series(pd.NaT,index=raw.index,dtype='datetime64[ns, UTC]')
            for zone, positions in raw.groupby('origin_zone').groups.items():
                utc.loc[positions] = naive.loc[positions].dt.tz_localize(zone,ambiguous=ambiguous,
                    nonexistent='NaT').dt.tz_convert('UTC').astype('datetime64[ns, UTC]')
            assert utc.notna().all(), 'A nonexistent departure clock requires review.'
            departure_options.append(utc)
        best = pd.Series(pd.NaT,index=raw.index,dtype='datetime64[ns, UTC]')
        error = pd.Series(np.inf,index=raw.index)
        unresolved = pd.Series(False,index=raw.index)
        for departure in departure_options:
            estimate = departure+pd.to_timedelta(duration,unit='m')
            for origin, zone in zones.items():
                positions = raw['Dest'].eq(origin)
                local_day = estimate.loc[positions].dt.tz_convert(zone).dt.tz_localize(None).dt.normalize()
                for shift in [-1,0,1]:
                    local_clock = local_day+pd.Timedelta(days=shift)+pd.to_timedelta((arr.loc[positions]//100*60+arr.loc[positions]%100).mod(1440),unit='m')
                    for ambiguous in [True,False]:
                        candidate = local_clock.dt.tz_localize(zone,ambiguous=ambiguous,nonexistent='NaT').dt.tz_convert('UTC').astype('datetime64[ns, UTC]')
                        delta = (candidate-estimate.loc[positions]).abs().dt.total_seconds()/60
                        improve = delta.lt(error.loc[positions])
                        ties = delta.eq(error.loc[positions]) & candidate.ne(best.loc[positions]) & candidate.notna()
                        unresolved.loc[ties.index[ties]] = True
                        ix = improve.index[improve]
                        best.loc[ix], error.loc[ix], unresolved.loc[ix] = candidate.loc[ix], delta.loc[ix], False
        assert best.notna().all() and error.lt(720).all() and not unresolved.any(), 'Unresolved arrival-date ambiguity requires review.'
        flag = error.gt(0).astype('int8')
        incoming_parts.append(pd.DataFrame({'origin':raw['Dest'],'arrival_utc':best,'schedule_clock_disagreement':flag}))
        arrival_audit.append({'source':path.relative_to(ROOT).as_posix(),'rows':len(raw),
            'flagged_clock_duration_disagreements':int(flag.sum()),'max_discrepancy_minutes':float(error.max())})
        print(path.stem, 'incoming schedules:',len(raw),'flagged:',int(flag.sum()),flush=True)
    incoming = pd.concat(incoming_parts,ignore_index=True)
    arrival_counts = {}
    arrival_disagreement_audit = np.zeros(len(development),dtype='int32')
    for name, lower, upper in [('arrivals_prior60m',-60,0),('arrivals_next60m',0,60)]:
        counts = np.zeros(len(development),dtype='int32')
        for origin in zones:
            positions = np.flatnonzero(development['origin'].eq(origin).to_numpy())
            block = incoming.loc[incoming['origin'].eq(origin)].sort_values('arrival_utc')
            times = block['arrival_utc'].array.asi8
            query = development.iloc[positions]['scheduled_departure_utc'].array.asi8
            left = np.searchsorted(times,query+pd.Timedelta(minutes=lower).value,side='left')
            right = np.searchsorted(times,query+pd.Timedelta(minutes=upper).value,side='left')
            counts[positions] = right-left
            if lower<0:
                flagged = np.r_[0,block['schedule_clock_disagreement'].cumsum().to_numpy()]
                arrival_disagreement_audit[positions] = flagged[right]-flagged[left]
        arrival_counts[name] = counts
    arrival_counts['scheduled_movements_prior60m'] = arrival_counts['arrivals_prior60m']+development['airport_scheduled_departures_prior_60m']
    arrival_counts['arrival_departure_balance'] = (arrival_counts['arrivals_prior60m']-development['airport_scheduled_departures_prior_60m'])/(1+arrival_counts['scheduled_movements_prior60m'])
    for c, values in arrival_counts.items():
        broad_features[c] = values
    BROAD_GROUPS['scheduled_arrivals'] = list(arrival_counts)
    print('Development flights whose prior-hour arrivals include a flagged schedule:',int((arrival_disagreement_audit>0).sum()))
    return arrival_audit, arrival_disagreement_audit



def create_pressure_and_nonlinear_candidates(BROAD_BINARY, BROAD_GROUPS, CONFIG, ROOT, broad_features, development):
    """Create pressure and nonlinear candidates."""
    weather_broad, _ = load_weather(ROOT,[2025],CONFIG['weather_quality']['excluded_station_times'])
    body = weather_broad['metar'].str.split(r'\s+RMK(?:\s|$)',n=1,regex=True).str[0]
    groups = body.str.findall(r'(?<!\S)A(\d{4})(?=\s|=|$)')
    weather_broad['altimeter_inhg'] = pd.to_numeric(groups.str[0],errors='coerce').div(100).where(groups.str.len().eq(1))
    wind_group = body.str.extract(r'(?<!\S)((?:\d{3}|VRB)\d{2,3}(?:G\d{2,3})?KT)(?=\s|=|$)',expand=False)
    weather_broad['variable_wind_flag'] = wind_group.str.startswith('VRB').astype(float).where(wind_group.notna())
    weather_broad['gust_excess_knots'] = (weather_broad['gust_knots']-weather_broad['wind_knots']).clip(lower=0)
    pressure_blocks = []
    for _, block in weather_broad.groupby('origin',observed=True):
        block = block.sort_values('weather_observed_utc').set_index('weather_observed_utc')
        rolling = block['altimeter_inhg'].rolling('3h',closed='both')
        first = rolling.apply(lambda a:a[np.isfinite(a)][0] if np.isfinite(a).any() else np.nan,raw=True)
        block['altimeter_change3h_inhg'] = block['altimeter_inhg']-first
        block['altimeter_range3h_inhg'] = rolling.max()-rolling.min()
        pressure_blocks.append(block.reset_index())
    weather_broad = pd.concat(pressure_blocks,ignore_index=True)
    pressure_columns = ['altimeter_inhg','altimeter_change3h_inhg','altimeter_range3h_inhg',
                        'variable_wind_flag','gust_excess_knots']
    matched = development[['origin','weather_observed_utc']].merge(
        weather_broad[['origin','weather_observed_utc']+pressure_columns],
        how='left',on=['origin','weather_observed_utc'],validate='many_to_one')
    assert len(matched)==len(development)
    for c in pressure_columns:
        broad_features[c] = matched[c].to_numpy()
    broad_features['altimeter_missing_flag'] = broad_features['altimeter_inhg'].isna().astype('int8')
    BROAD_GROUPS['pressure_wind'] = pressure_columns+['altimeter_missing_flag']
    BROAD_BINARY += ['variable_wind_flag','altimeter_missing_flag']
    shape_values = {
        'temperature_centered_square': ((development['temperature_f']-32)/10)**2,
        'wind_square': development['wind_knots']**2,
        'dewpoint_spread_square': development['dewpoint_spread_f']**2,
        'precip_log1p': np.log1p(development['precip_inches']),
        'ceiling_log1p': np.log1p(development['ceiling_ft']),
        'visibility_inverse': 1/(1+development['visibility_miles']),
    }
    for c, values in shape_values.items():
        broad_features[c] = values
    BROAD_GROUPS['nonlinear_weather'] = list(shape_values)

    assert broad_features['altimeter_inhg'].notna().any(), 'No altimeter values parsed; review the source format.'
    return BROAD_BINARY



def create_conditional_histories(BROAD_BINARY, BROAD_GROUPS, broad_features, candidate_feature_table, development, events, extension_table, original_ids, past_window_totals, train_mask):
    """Create conditional histories."""
    conditional_query = development[['origin','airline','destination','prediction_cutoff_utc']].copy()
    conditional_query['departure_period'] = development['scheduled_departure_minutes']//240
    conditional_events = events.copy()
    conditional_events['departure_period'] = conditional_events['scheduled_departure_minutes']//240
    conditioned_columns = []
    conditional_latest = {}
    for name, keys in [('origin_period',['origin','departure_period']),
                       ('airline_origin_period',['origin','airline','departure_period']),
                       ('route_period',['origin','destination','departure_period'])]:
        count, delayed, latest = past_window_totals(conditional_query,conditional_events,keys,
            'departure_outcome_event_utc','delayed_departure',28)
        if name=='origin_period':
            rate = np.divide(delayed,count,out=np.full(len(count),np.nan),where=count>0)
        else:
            prior = broad_features['conditioned_origin_period_rate'].to_numpy()
            rate = (delayed+20*prior)/(count+20)
        prefix = 'conditioned_'+name
        for suffix, values in [('_rate',rate),('_n',count),('_no_history',(count==0).astype('int8'))]:
            broad_features[prefix+suffix] = values
            conditioned_columns.append(prefix+suffix)
        conditional_latest[name] = latest
    BROAD_GROUPS['conditioned_history'] = conditioned_columns
    BROAD_BINARY += [c for c in conditioned_columns if c.endswith('_no_history')]
    BROAD_NEW = [c for columns in BROAD_GROUPS.values() for c in columns]
    assert len(BROAD_NEW)==len(set(BROAD_NEW))
    assert not set(BROAD_NEW).intersection(candidate_feature_table.columns)
    numeric_broad = broad_features.select_dtypes(include='number').columns
    broad_features[numeric_broad] = broad_features[numeric_broad].astype('float64')
    broad_frame = pd.concat([candidate_feature_table,broad_features],axis=1)
    assert broad_frame['flight_id'].equals(original_ids)
    assert not np.isinf(broad_features.select_dtypes(include='number').to_numpy(dtype=float,na_value=np.nan)).any()
    broad_profile = pd.DataFrame({'group':{c:g for g,cols in BROAD_GROUPS.items() for c in cols},
        'training_missing_pct':100*broad_features.loc[train_mask].isna().mean(),
        'training_unique':broad_features.loc[train_mask].nunique()})
    extension_table(broad_profile)
    print(f'{len(BROAD_NEW)} additional candidates across {len(BROAD_GROUPS)} new groups; no correlation filter applied.')
    return BROAD_BINARY, BROAD_NEW, broad_frame, broad_profile



def prepare_broad_validation(BROAD_GROUPS, BROAD_NEW, BROAD_SEED, CURRENT_B, FROZEN, NEW_FEATURES, ROOT, broad_frame, broad_inputs_before, extension_table):
    """Prepare broad validation."""
    BROAD_PACKAGES = {'current_66':CURRENT_B, 'previous_28':CURRENT_B+NEW_FEATURES}
    BROAD_PACKAGES.update({name:CURRENT_B+columns for name,columns in BROAD_GROUPS.items()})
    BROAD_PACKAGES['all_candidates'] = CURRENT_B+NEW_FEATURES+BROAD_NEW
    FOLDS_BROAD = [('May–Jun','2025-03-31','2025-04-01','2025-06-30'),
                   ('Aug–Sep','2025-06-30','2025-07-01','2025-09-30'),
                   ('Nov–Dec','2025-09-30','2025-10-01','2025-12-31')]

    def fixed_sample(population,n):
        population = population.sort_values('flight_id')
        if len(population)==n:
            return population
        strata = population['month'].astype(str)+'|'+population['delayed_departure'].astype(str)
        selected,_ = train_test_split(population.index,train_size=n,random_state=BROAD_SEED,stratify=strata)
        return population.loc[selected].sort_values('flight_id')

    broad_cohorts = {}
    for name,end,start,stop in FOLDS_BROAD:
        fitting = broad_frame.loc[broad_frame['flight_date'].le(end)]
        validation = broad_frame.loc[broad_frame['flight_date'].between(start,stop)]
        if name=='Nov–Dec':
            fitting = fitting.loc[fitting['model_sample']]
            validation = validation.loc[validation['model_sample']]
        else:
            fitting,validation = fixed_sample(fitting,20000),fixed_sample(validation,5000)
        calibration = validation.loc[validation['flight_date'].dt.month.eq(pd.Timestamp(start).month)]
        scoring = validation.loc[~validation.index.isin(calibration.index)]
        assert len(fitting)==20000 and len(validation)==5000
        assert fitting['flight_date'].max()<calibration['flight_date'].min()<=calibration['flight_date'].max()<scoring['flight_date'].min()
        broad_cohorts[name] = {'train':fitting.index.to_list(),'calibrate':calibration.index.to_list(),'score':scoring.index.to_list()}

    BROAD_SETTINGS = {c['model_id']:c for c in FROZEN['choices'] if c['feature_set']=='B'}
    BROAD_REPORT = ROOT/'reports/broad_feature_experiments.json'
    broad_protocol = {'started_at_utc':datetime.now(timezone.utc).isoformat(),
        'query_period':'2025 only','history_lag_days':7,'package_columns':BROAD_PACKAGES,
        'model_settings':{m:{'recipe':c['recipe'],'params':c['params']} for m,c in BROAD_SETTINGS.items()},
        'folds':FOLDS_BROAD,'train_rows':20000,'quarter_calibration_scoring_rows':5000,
        'threshold':'Max F1 on first validation month, score subsequent two months; no refit',
        'package_selection':'Top two non-baseline mean-F1 packages averaged over LR and RF',
        'correlation_filter':False,'class_balancing':False,'new_final_test':False,
        'arrival_clock_policy':'User-approved arrival clock, inferred date, discrepancies retained and flagged',
        'input_sha256':broad_inputs_before}
    BROAD_REPORT.write_text(json.dumps({'status':'protocol_declared','protocol':broad_protocol},indent=2)+'\n')
    extension_table(pd.DataFrame({'predictors':{k:len(v) for k,v in BROAD_PACKAGES.items()}}))
    return BROAD_PACKAGES, BROAD_REPORT, BROAD_SETTINGS, broad_cohorts, broad_protocol



def prepare_broad_models(BROAD_BINARY, BROAD_PACKAGES, BROAD_REPORT, BROAD_SEED, BROAD_SETTINGS, CONFIG, NEW_FEATURES, PREP, broad_cohorts, broad_frame, broad_protocol):
    """Prepare broad models."""
    BINARY_BROAD = [c for c in PREP['feature_sets']['B']+TIME_FEATURES+FLIGHT_FEATURES+WEATHER_NEW+NEW_FEATURES
        if c.endswith('_no_history')] + ['precip_trace','no_ceiling_reported','cloud_coverage_missing',
        'rain_reported','frozen_precip_reported','fog_or_mist_reported','thunder_reported',
        'condition_codes_missing','ceiling_height_missing','weather_report_missing','freezing_wet',
        'traffic_reference_missing_or_zero'] + BROAD_BINARY
    COVERED_BROAD = CONFIG['preprocessing']['explicit_missingness_covered_columns']

    def broad_pipeline(model_id,columns):
        recipe = BROAD_SETTINGS[model_id]['recipe']
        params = BROAD_SETTINGS[model_id]['params'].copy()
        params.pop('drop_duration',None)
        categorical = [c for c in columns if c in ['airline','origin','destination','weekday'] or c.endswith('_cat')]
        numeric = [c for c in columns if c not in categorical]
        transforms = []
        if recipe=='original':
            for name,cols,indicators in [('numeric',[c for c in numeric if c not in COVERED_BROAD],True),
                                         ('explicit',[c for c in numeric if c in COVERED_BROAD],False)]:
                steps = [('imputer',SimpleImputer(strategy='median',add_indicator=indicators,keep_empty_features=True))]
                if model_id!='random_forest':
                    steps.append(('scaler',StandardScaler()))
                if cols:
                    transforms.append((name,Pipeline(steps),cols))
        else:
            continuous = [c for c in numeric if c not in BINARY_BROAD]
            flags = [c for c in numeric if c in BINARY_BROAD]
            steps = [('imputer',SimpleImputer(strategy='median',keep_empty_features=True))]
            if model_id!='random_forest':
                steps.append(('scaler',StandardScaler()))
            if continuous:
                transforms.append(('continuous',Pipeline(steps),continuous))
            if flags:
                transforms.append(('binary',SimpleImputer(strategy='constant',fill_value=0,keep_empty_features=True),flags))
            missing = [c for c in numeric if c not in COVERED_BROAD]
            if missing:
                transforms.append(('missing',MissingIndicator(features='all',error_on_new=False),missing))
        transforms.append(('category',Pipeline([
            ('imputer',SimpleImputer(strategy='constant',fill_value='Unknown',missing_values=pd.NA,keep_empty_features=True)),
            ('encoder',OneHotEncoder(handle_unknown='ignore',sparse_output=True))]),categorical))
        estimators = {
            'logistic_regression':LogisticRegression(max_iter=3000,random_state=BROAD_SEED),
            'knn':KNeighborsClassifier(algorithm='brute',n_jobs=2),
            'svm':SVC(kernel='rbf',cache_size=512,random_state=BROAD_SEED),
            'random_forest':RandomForestClassifier(n_estimators=200,max_features='sqrt',class_weight='balanced',random_state=BROAD_SEED,n_jobs=2)}
        return Pipeline([('preprocess',ColumnTransformer(transforms,sparse_threshold=1.0)),
                         ('classifier',estimators[model_id].set_params(**params))])

    def broad_scores(model,frame):
        return model.decision_function(frame) if isinstance(model.named_steps['classifier'],SVC) else model.predict_proba(frame)[:,1]

    def calibration_threshold(y,scores):
        precision,recall,thresholds = precision_recall_curve(y,scores)
        f1 = np.divide(2*precision[:-1]*recall[:-1],precision[:-1]+recall[:-1],
            out=np.zeros(len(thresholds)),where=(precision[:-1]+recall[:-1])>0)
        return float(thresholds[np.flatnonzero(np.isclose(f1,f1.max(),rtol=0,atol=1e-12))[-1]])

    broad_runs, broad_score_audit = [], []
    def evaluate_broad(model_id,package):
        columns = BROAD_PACKAGES[package]
        for fold,cohort in broad_cohorts.items():
            if any(r['model_id']==model_id and r['package']==package and r['fold']==fold for r in broad_runs):
                continue
            train,calibration,score = [broad_frame.loc[cohort[role]] for role in ['train','calibrate','score']]
            model = broad_pipeline(model_id,columns)
            started = perf_counter()
            model.fit(train[columns],train['delayed_departure'])
            fit_seconds = perf_counter()-started
            classifier = model.named_steps['classifier']
            if isinstance(classifier,SVC):
                assert classifier.fit_status_==0
            if isinstance(classifier,LogisticRegression):
                assert classifier.n_iter_.max()<classifier.max_iter
            cal_scores, scores = broad_scores(model,calibration[columns]),broad_scores(model,score[columns])
            assert np.isfinite(cal_scores).all() and np.isfinite(scores).all()
            threshold = calibration_threshold(calibration['delayed_departure'],cal_scores)
            y = score['delayed_departure'].to_numpy()
            tn,fp,fn,tp = confusion_matrix(y,scores>=threshold,labels=[0,1]).ravel()
            broad_runs.append({'model_id':model_id,'package':package,'fold':fold,
                'train_rows':len(train),'calibration_rows':len(calibration),'score_rows':len(score),
                'predictors':len(columns),'threshold':threshold,'f1':float(2*tp/(2*tp+fp+fn)),
                'precision':float(tp/(tp+fp)) if tp+fp else None,'recall':float(tp/(tp+fn)),
                'average_precision':float(average_precision_score(y,scores)),
                'warn_all_f1':float(2*y.mean()/(1+y.mean())),
                'tp':int(tp),'fp':int(fp),'fn':int(fn),'tn':int(tn),'fit_seconds':fit_seconds})
            broad_score_audit.append({'model_id':model_id,'package':package,'fold':fold,
                'y_true':y.tolist(),'score':scores.tolist(),'threshold':threshold})
            print(f'{model_id} / {package} / {fold}: F1={broad_runs[-1]["f1"]:.3f}, AP={broad_runs[-1]["average_precision"]:.3f}',flush=True)
            BROAD_REPORT.write_text(json.dumps({'status':'development_screening_running','protocol':broad_protocol,
                'runs':broad_runs},indent=2,allow_nan=False)+'\n')
    return broad_runs, evaluate_broad



def save_broad_experiment(BROAD_FINALISTS, BROAD_GROUPS, BROAD_NEW, BROAD_PACKAGES, BROAD_REPORT, NEW_FEATURES, ROOT, arrival_audit, arrival_disagreement_audit, broad_frame, broad_inputs_before, broad_profile, broad_protocol, broad_runs, comparison, development, extension_sha):
    """Save broad experiment."""
    assert len(broad_frame)==len(development)==938417
    assert broad_frame['flight_date'].dt.year.eq(2025).all()
    for columns in BROAD_PACKAGES.values():
        assert len(columns)==len(set(columns))
        assert not {'delayed_departure','flight_id','flight_number','departure_outcome_event_utc',
            'Tail_Number','tail_number','DepDelay','ArrDelay','LateAircraftDelay'}.intersection(columns)
    assert all(extension_sha(ROOT/p)==checksum for p,checksum in broad_inputs_before.items())
    broad_summary = {'status':'executed_pending_independent_audit','protocol':broad_protocol,
        'development_rows':len(broad_frame),'new_columns':len(BROAD_NEW),'new_groups':BROAD_GROUPS,
        'retained_previous_candidates':len(NEW_FEATURES),'all_candidate_predictors':len(BROAD_PACKAGES['all_candidates']),
        'models_compared':['logistic_regression','knn','svm','random_forest'],
        'training_missing_pct':broad_profile['training_missing_pct'].to_dict(),
        'arrival_source_audit':arrival_audit,
        'arrival_clock_disagreement_context_flights':int((arrival_disagreement_audit>0).sum()),
        'finalist_packages':BROAD_FINALISTS,'runs':broad_runs,
        'all_model_comparison':json.loads(comparison.to_json(orient='records')),
        'original_artifacts_unchanged':True,'new_2026_outcome_access':False,
        'no_processed_datasets_or_fitted_model_bundles_written':True}
    BROAD_REPORT.write_text(json.dumps(broad_summary,indent=2,allow_nan=False)+'\n')
    print(f'Completed {len(broad_runs)} development fits. Detailed results: reports/broad_feature_experiments.json')
    print('No correlation-based rejection. No claim of a new final-test F1 score.')



def setup_deeper_exploration():
    """Setup deeper exploration."""
    from pathlib import Path
    import site
    if site.ENABLE_USER_SITE:
        site.addsitedir(site.getusersitepackages())
    import hashlib
    import json
    import re

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    import seaborn as sns
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    ROOT = Path.cwd().resolve()
    if ROOT.name == 'notebooks':
        ROOT = ROOT.parent
    if (ROOT / 'departure-delay-classification/config/project.json').is_file():
        ROOT = ROOT / 'departure-delay-classification'
    CONFIG = json.loads((ROOT / 'config/project.json').read_text())
    MANIFEST = json.loads((ROOT / 'reports/preparation_manifest.json').read_text())
    INPUTS = ['notebooks/01_cleaning_and_eda.ipynb', 'notebooks/02_feature_engineering.ipynb',
              'data/processed/flights_clean.parquet', 'data/processed/features.parquet',
              'config/project.json', 'reports/preparation_manifest.json']

    def sha256(path):
        with path.open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()

    input_hashes = {name: sha256(ROOT / name) for name in INPUTS}
    for kind in ['input', 'output']:
        provenance = MANIFEST['feature_engineering']
        assert input_hashes[provenance[kind]] == provenance[kind + '_sha256'], 'Rebuild a stale handoff first.'
    train = pd.read_parquet(ROOT / 'data/processed/features.parquet', filters=[('split', '==', 'train')])
    scheduled = pd.read_parquet(ROOT / 'data/processed/flights_clean.parquet', filters=[('split', '==', 'train')])
    train = train.merge(scheduled[['flight_id', 'departure_hour', 'departure_delay_minutes']],
                        on='flight_id', validate='one_to_one')
    assert train['flight_id'].is_unique and train['split'].eq('train').all()
    assert train['flight_date'].between(CONFIG['split']['train_start'], CONFIG['split']['train_end']).all()
    TARGET = 'delayed_departure'
    BANDS = ['00–05 overnight', '06–11 morning', '12–17 afternoon', '18–23 evening']
    GROUPS = ['origin', 'month', 'departure_time_band']
    sns.set_theme(style='whitegrid', palette=['#247BA0', '#E08D3C', '#5C946E'])
    plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False})

    def show_table(data, **kwargs):
        print(data.to_string(**kwargs))

    def show_plot():
        plt.tight_layout()
        plt.show()

    def rates(frame, groups):
        return frame.groupby(groups, observed=True)[TARGET].agg(n='size', delay_rate='mean')

    print(f'{len(train):,} eligible training flights; {len(scheduled):,} scheduled flights.')
    print(f'Overall 15+ minute delay rate: {train[TARGET].mean():.2%}')
    return BANDS, CONFIG, GROUPS, INPUTS, MANIFEST, ROOT, TARGET, input_hashes, rates, scheduled, sha256, train



def audit_missingness_indicators(CONFIG, MANIFEST, sample, train):
    """Audit missingness indicators."""
    predictors = MANIFEST['feature_engineering']['feature_sets']['B']
    categorical = ['airline', 'origin', 'destination', 'weekday']
    numeric = [name for name in predictors if name not in categorical]
    recipe = make_pipeline(SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True),
                           StandardScaler())
    transformed = recipe.fit_transform(sample[numeric])
    names = list(recipe[0].get_feature_names_out(numeric))
    condition_flags = ['rain_reported', 'frozen_precip_reported', 'fog_or_mist_reported', 'thunder_reported']
    repeated = ['condition_codes_missing'] + ['missingindicator_' + name for name in condition_flags]
    values = transformed[:, [names.index(name) for name in repeated]]
    assert np.allclose(values, values[:, [0]])
    show_table(pd.Series(repeated, name='former_five_identical_columns').to_frame(), index=False)
    covered = CONFIG['preprocessing']['explicit_missingness_covered_columns']
    corrected_names, corrected_blocks = [], []
    for columns, add_indicators in [([c for c in numeric if c not in covered], True),
                                    ([c for c in numeric if c in covered], False)]:
        imputer = SimpleImputer(strategy='median', add_indicator=add_indicators, keep_empty_features=True)
        corrected_blocks.append(StandardScaler().fit_transform(imputer.fit_transform(sample[columns])))
        corrected_names.extend(imputer.get_feature_names_out(columns))
    assert not any('missingindicator_' + name in corrected_names for name in covered)
    for name in ['wind_knots', 'gust_knots']:
        if sample[name].isna().any():
            assert 'missingindicator_' + name in corrected_names
    assert np.isfinite(np.column_stack(corrected_blocks)).all()
    show_table(pd.DataFrame({'recipe': ['former', 'corrected'],
                            'numeric_columns': [len(names), len(corrected_names)],
                            'condition_absence_coordinates': [5, 1]}))
    print('Corrected automatic indicators:', [c for c in corrected_names if c.startswith('missingindicator_')])
    show_table(train[predictors].nunique().loc[lambda s: s <= 1].rename('distinct_training_values').to_frame())
    print('Ceiling missingness equals the no-ceiling flag in training:',
          train['ceiling_ft'].isna().eq(train['no_ceiling_reported'].eq(1)).all())
    print('Distance/duration correlation:', train['distance_miles'].corr(train['scheduled_duration_minutes']))
    show_table(train[['gust_knots', 'rain_reported', 'fog_or_mist_reported']].median().to_frame('nonmissing_median'))



def audit_full_weather_messages(CONFIG, ROOT, train):
    """Audit full weather messages."""
    columns = ['station', 'valid', 'metar', 'wxcodes', 'dwpf', 'p01i'] + [f'skyc{i}' for i in range(1, 5)] + [f'skyl{i}' for i in range(1, 5)]
    parts = []
    for source in sorted(train['weather_source_file'].dropna().unique()):
        raw = pd.read_csv(ROOT / source, usecols=columns, dtype='string', keep_default_na=False)
        raw = raw.loc[raw['station'].isin(CONFIG['scope']['origin_airports'])].copy()
        for column in columns:
            raw[column] = raw[column].str.strip()
        parts.append(raw)
    weather = pd.concat(parts, ignore_index=True).drop_duplicates(columns).rename(columns={'station': 'origin'})
    weather['weather_observed_utc'] = pd.to_datetime(weather['valid'], utc=True).astype('datetime64[ns, UTC]')
    approved = pd.MultiIndex.from_tuples([tuple(key) for key in CONFIG['weather_quality']['excluded_station_times']])
    weather = weather.loc[~pd.MultiIndex.from_frame(weather[['origin', 'valid']]).isin(approved)].copy()
    weather = weather.loc[weather['weather_observed_utc'].between(
        train['weather_observed_utc'].min() - pd.Timedelta(hours=3), train['weather_observed_utc'].max())].copy()
    assert not weather.duplicated(['origin', 'weather_observed_utc']).any()
    weather = weather.sort_values(['origin', 'weather_observed_utc']).reset_index(drop=True)
    body = weather['metar'].str.split(r'\s+RMK(?:\s|$)', n=1, regex=True).str[0]
    weather['cloud_groups'] = body.str.findall(r'(?<!\S)(?:FEW|SCT|BKN|OVC|VV)(?:\d{3}|///)(?:CB|TCU)?(?=\s|=|$)')

    def body_ceiling(groups):
        if any(g.startswith(('BKN', 'OVC', 'VV')) and '///' in g for g in groups):
            return np.nan
        heights = [int(re.search(r'\d{3}', group).group()) * 100 for group in groups
                   if group.startswith(('BKN', 'OVC', 'VV')) and '///' not in group]
        return min(heights) if heights else np.nan

    weather['body_ceiling_ft'] = weather['cloud_groups'].map(body_ceiling)
    coverage = weather[[f'skyc{i}' for i in range(1, 5)]]
    heights = weather[[f'skyl{i}' for i in range(1, 5)]].apply(pd.to_numeric, errors='coerce')
    heights.columns = coverage.columns
    ceiling_layers = coverage.isin(['BKN', 'OVC', 'VV'])
    weather['four_column_ceiling_ft'] = heights.where(ceiling_layers).min(axis=1).mask(
        (ceiling_layers & heights.isna()).any(axis=1))
    weather['explicit_status_flag'] = weather['metar'].str.contains(r'\b(?:PWINO|TSNO|FZRANO|FZNO|PNO)\b', regex=True)
    audit = train.merge(weather[['origin', 'weather_observed_utc', 'metar', 'wxcodes', 'dwpf',
                                'body_ceiling_ft', 'four_column_ceiling_ft', 'explicit_status_flag']],
                        on=['origin', 'weather_observed_utc'], how='left', validate='many_to_one', indicator=True)
    assert audit['_merge'].eq('both').all()
    missed_ceiling = audit['ceiling_ft'].isna() & audit['body_ceiling_ft'].notna()
    formerly_missed = audit['four_column_ceiling_ft'].isna() & audit['body_ceiling_ft'].notna()
    examples = audit.loc[formerly_missed, ['origin', 'weather_observed_utc', 'ceiling_ft',
                                        'body_ceiling_ft', 'no_ceiling_reported', 'metar']].drop_duplicates()
    print('Flights where the four-slot source would miss a body ceiling:', int(formerly_missed.sum()))
    assert not missed_ceiling.any(), 'Review any remaining known body ceiling omitted from the saved features.'
    print(f'{missed_ceiling.sum():,} flights ({missed_ceiling.mean():.3%}) still have an omitted known body ceiling.')
    print('Distinct recovered reports:', len(examples))
    show_table(examples.head(4), index=False)
    print('Recovered ceiling examples (ft):', sorted(examples['body_ceiling_ft'].unique()))
    unknown_conditions = audit['condition_codes_missing'].eq(1)
    print(f'No usable condition code: {unknown_conditions.sum():,} flights; '
          f'{(unknown_conditions & audit["explicit_status_flag"]).sum():,} also have explicit status tokens.')
    return audit, weather



def compare_airline_flight_mix(GROUPS, TARGET, rates, train):
    """Compare airline flight mix."""
    def shared_mix(frame, exposure, groups=GROUPS, min_n=30):
        grouped = frame.groupby(groups + [exposure], observed=True)[TARGET].agg(n='size', risk='mean')
        counts = grouped['n'].unstack(exposure).reindex(columns=[0, 1])
        risks = grouped['risk'].unstack(exposure).reindex(columns=[0, 1])
        keep = counts.ge(min_n).all(axis=1)
        counts, risks = counts.loc[keep], risks.loc[keep]
        assert len(counts) > 0
        weights = counts.sum(axis=1) / counts.sum().sum()
        means = risks.mul(weights, axis=0).sum()
        common = frame.loc[pd.MultiIndex.from_frame(frame[groups]).isin(counts.index)]
        raw = common.groupby(exposure, observed=True)[TARGET].mean()
        return {'shared_groups': len(counts), 'retained_rows': len(common),
                'retained_fraction': len(common) / len(frame),
                'raw_common_0': raw[0], 'raw_common_1': raw[1],
                'standardized_0': means[0], 'standardized_1': means[1],
                'difference_pp': (means[1] - means[0]) * 100}

    mix = pd.crosstab(train['airline'], train['origin'], normalize='index')
    airline_origin = rates(train, ['airline', 'origin'])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.heatmap(mix, annot=True, fmt='.0%', cmap='Blues', vmin=0, vmax=1, ax=axes[0], cbar=False)
    axes[0].set_title('Where each airline’s training flights depart')
    risk_map = airline_origin['delay_rate'].unstack().where(airline_origin['n'].unstack().ge(200))
    sns.heatmap(risk_map, annot=True, fmt='.1%', cmap='YlOrRd',
                vmin=0, vmax=.5, ax=axes[1], cbar=False)
    axes[1].set_title('Airline × origin delay rate · n ≥ 200')
    show_plot()
    pair = train.loc[train['airline'].isin(['F9', 'AA'])].copy()
    pair['is_F9'] = pair['airline'].eq('F9').astype(int)
    airline_comparison = shared_mix(pair, 'is_F9')
    show_table(rates(pair, 'airline'))
    show_table(pd.Series(airline_comparison).to_frame('F9_vs_AA_shared_mix'))
    show_table(pd.DataFrame([dict(min_per_group=n, **shared_mix(pair, 'is_F9', min_n=n))
                             for n in [10, 30, 50]]))
    return shared_mix



def compare_weather_and_departure_time(BANDS, rates, shared_mix, train):
    """Compare weather and departure time."""
    train['precip_state'] = np.select([train['precip_trace'].eq(1), train['precip_inches'].gt(0)],
                                     ['Trace', 'Positive'], default='Zero')
    interaction = rates(train, ['departure_time_band', 'precip_state'])
    contrast = interaction['delay_rate'].unstack().reindex(BANDS)
    contrast['positive_minus_zero_pp'] = (contrast['Positive'] - contrast['Zero']) * 100
    contrast['positive_to_zero_ratio'] = contrast['Positive'] / contrast['Zero']
    show_table(contrast[['positive_minus_zero_pp', 'positive_to_zero_ratio']])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    sns.heatmap(interaction['delay_rate'].unstack().reindex(BANDS)[['Zero', 'Trace', 'Positive']],
                annot=True, fmt='.1%', cmap='YlOrRd', ax=axes[0], cbar=False)
    axes[0].set(title='Delay rate · precipitation × departure time', xlabel='Latest accumulation')
    sns.heatmap(interaction['n'].unstack().reindex(BANDS)[['Zero', 'Trace', 'Positive']],
                annot=True, fmt=',.0f', cmap='Blues', ax=axes[1], cbar=False)
    axes[1].set(title='Corresponding flight counts', xlabel='Latest accumulation')
    show_plot()
    train['positive_accumulation'] = train['precip_inches'].gt(0).astype(int)
    show_table(pd.Series(shared_mix(train, 'positive_accumulation')).to_frame('positive_vs_nonpositive'))
    train['visibility_band'] = pd.cut(train['visibility_miles'], [-np.inf, 1, 3, 5, 10, np.inf],
                                     right=False, labels=['<1', '1–<3', '3–<5', '5–<10', '10+'])
    train['wind_band'] = pd.cut(train['wind_knots'], [-np.inf, 5, 10, 15, 20, np.inf],
                               right=False, labels=['<5', '5–<10', '10–<15', '15–<20', '20+'])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for column, ax, label in zip(['visibility_band', 'wind_band'], axes, ['Visibility (miles)', 'Wind (knots)']):
        table = rates(train, column)
        ax.plot(table.index.astype(str), table['delay_rate'], marker='o')
        ax.set(xlabel=label, ylabel='15+ minute delay rate', title='Training association')
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    show_plot()
    print(f'Visibility stored at exactly 10 miles: {train["visibility_miles"].eq(10).mean():.1%}')



def compare_earlier_weather(rates, shared_mix, train, weather):
    """Compare earlier weather."""
    weather['precip_numeric'] = pd.to_numeric(weather['p01i'].replace({'M': None, 'T': '0', '': None}), errors='coerce')
    weather['earlier_positive_3h'] = False
    weather['earlier_reports_3h'] = 0
    for origin, indices in weather.groupby('origin', observed=True).groups.items():
        block = weather.loc[indices]
        times = block['weather_observed_utc'].array.asi8
        left = np.searchsorted(times, times - pd.Timedelta(hours=3).value, side='left')
        end = np.arange(len(block))
        positives = np.r_[0, block['precip_numeric'].gt(0).to_numpy().cumsum()]
        weather.loc[indices, 'earlier_positive_3h'] = positives[end] - positives[left] > 0
        weather.loc[indices, 'earlier_reports_3h'] = end - left
    recent = train.merge(weather[['origin', 'weather_observed_utc', 'earlier_positive_3h', 'earlier_reports_3h']],
                         on=['origin', 'weather_observed_utc'], validate='many_to_one')
    recent['weather_history_state'] = np.select(
        [recent['precip_inches'].gt(0), recent['precip_trace'].eq(1),
         recent['precip_inches'].eq(0) & recent['earlier_positive_3h'], recent['earlier_reports_3h'].eq(0)],
        ['Latest positive', 'Latest trace', 'Latest zero; earlier positive', 'No earlier report'],
        default='Latest zero; no earlier positive')
    recent_rates = rates(recent, 'weather_history_state')
    show_table(recent_rates)
    fig, ax = plt.subplots(figsize=(10, 4))
    recent_rates['delay_rate'].plot.barh(ax=ax)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set(xlabel='15+ minute delay rate', ylabel='', title='Latest precipitation omits part of the history')
    show_plot()
    zero = recent.loc[recent['precip_inches'].eq(0) & recent['precip_trace'].eq(0) & recent['earlier_reports_3h'].gt(0)].copy()
    zero['earlier_positive_3h'] = zero['earlier_positive_3h'].astype(int)
    recent_comparison = shared_mix(zero, 'earlier_positive_3h')
    show_table(pd.Series(recent_comparison).to_frame('earlier_positive_vs_no_earlier_positive'))
    show_table(rates(zero, ['origin', 'earlier_positive_3h']))

    within_day_comparison = shared_mix(zero, 'earlier_positive_3h', ['origin', 'flight_date', 'departure_time_band'])
    show_table(pd.Series(within_day_comparison).to_frame('within_same_airport_day_and_time_band'))



def compare_daily_history(CONFIG, TARGET, scheduled, train):
    """Compare daily history."""
    daily = train.groupby(['origin', 'flight_date'], observed=True).agg(
        operated=(TARGET, 'size'), delayed=(TARGET, 'sum'), delay_rate=(TARGET, 'mean'),
        lagged_history=('history_origin_rate', 'mean'), positive_accumulation=('positive_accumulation', 'mean'))
    schedule_days = scheduled.groupby(['origin', 'flight_date'], observed=True).agg(
        scheduled=('flight_id', 'size'), cancelled=('cancelled', 'sum'))
    daily = daily.join(schedule_days)
    daily['cancel_rate'] = daily['cancelled'] / daily['scheduled']
    daily['history_gap'] = daily['delay_rate'] - daily['lagged_history']
    daily['cancelled_or_delayed_rate'] = (daily['cancelled'] + daily['delayed']) / daily['scheduled']
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True, sharey=True)
    for origin, ax in zip(CONFIG['scope']['origin_airports'], axes):
        line = daily.loc[origin]
        ax.plot(line.index, line['delay_rate'], alpha=.7, linewidth=1, label='Actual operated-delay rate')
        ax.plot(line.index, line['lagged_history'], linewidth=2, label='Existing lagged-history mean')
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set(title=origin, ylabel='Delay rate')
    axes[0].legend(ncol=2, loc='upper left')
    show_plot()
    show_table(daily.nlargest(6, 'history_gap'))
    show_table(daily.nlargest(5, 'cancel_rate'))
    print('Correlation across airport-days:', daily['delay_rate'].corr(daily['lagged_history']))
    worst = daily.nlargest(int(np.ceil(len(daily) * .1)), 'delay_rate')
    print(f'Top 10% airport-days by operated-delay rate: {worst["delayed"].sum()/daily["delayed"].sum():.1%} '
          f'of delays, {worst["operated"].sum()/daily["operated"].sum():.1%} of operated flights.')
    print('Cancelled training flights with a recorded departure-delay value:',
          int((scheduled['cancelled'].eq(1) & scheduled['departure_delay_minutes'].notna()).sum()))
    assert not scheduled.loc[scheduled['cancelled'].eq(1), 'query_eligible'].any()

