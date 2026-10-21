"""Shared steps for the short teaching notebooks; never fits or selects a model.

Source decoding, time joins, historical windows and repeated result formatting live
here. The notebook cells still create labels, apply physical-validity treatments,
show nonlinear formulas, instantiate sklearn estimators and call .fit explicitly.
The frozen feature/preprocessing implementations are reused without modification.
"""
from pathlib import Path
import hashlib
import json
from time import perf_counter
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
    f1_score, fbeta_score, precision_score, recall_score, roc_auc_score)
from simple_sources import read_flights, scheduled_utc
from improvement_features import (add_time, add_weather, add_histories, add_traffic,
    load_weather, WEATHER_BASE, WEATHER_NEW)
from final_features import pressure_reports, PRESSURE
from final_modelling import pipeline as frozen_pipeline, sample

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/simple_reproduction'
CONFIG = json.loads((ROOT/'config/project.json').read_text())
SELECTION = json.loads((ROOT/'reports/final_models/selection.json').read_text())
COHORTS = json.loads((ROOT/'reports/final_models/cohorts.json').read_text())
MODEL_NAMES = {'logistic_regression': 'Logistic Regression', 'knn': 'KNN',
               'svm': 'SVM', 'random_forest': 'Random Forest'}


def read_development_flights():
    """Read the same 17 raw files as historical cleaning, without transforming rows."""
    raw = ROOT/'data/raw/flights/bts'
    paths = sorted((raw/'2024_context/csv').glob('*.csv'))
    paths += sorted((raw/'2025/csv').glob('*.csv'))
    paths += sorted((raw/'2026/csv').glob('2026_0[123].csv'))
    assert len(paths) == 17
    return read_flights(ROOT, paths)


def decode_flights(frame):
    """Convert strings to dates/numbers and check natural identity; no label creation."""
    frame = frame.copy()
    for column in ['airline', 'flight_number', 'origin', 'destination']:
        frame[column] = frame[column].str.strip().replace('', pd.NA)
    frame['flight_date'] = pd.to_datetime(frame['flight_date'], format='%Y-%m-%d').astype('datetime64[ns]')
    conversions = []
    for column in ['scheduled_departure_hhmm', 'scheduled_duration_minutes', 'distance_miles',
                   'departure_delay_minutes', 'departure_delay_15_source', 'cancelled', 'diverted']:
        original = frame[column]
        values = pd.to_numeric(original, errors='coerce').astype(float)
        conversions.append({'variable': column, 'unparseable': int((original.notna() & values.isna()).sum())})
        frame[column] = values.where(np.isfinite(values))
    identity = ['flight_date', 'airline', 'flight_number', 'origin', 'destination', 'scheduled_departure_hhmm']
    assert frame['flight_id'].is_unique and not frame.duplicated(identity).any()
    assert frame[identity].notna().all().all() and frame['cancelled'].isin([0, 1]).all()
    return frame, pd.DataFrame(conversions)


def add_timestamps(frame):
    """HHMM/local schedule -> UTC schedule, two-hour cutoff and audit-only outcome time."""
    frame = frame.copy()
    clock = frame['scheduled_departure_hhmm']
    assert (clock.mod(1).eq(0) & clock.between(0, 2359) & clock.mod(100).lt(60)).all()
    frame['scheduled_departure_minutes'] = (clock//100*60+clock%100).astype('int16')
    frame['scheduled_departure_utc'], _ = scheduled_utc(frame, ROOT)
    frame['prediction_cutoff_utc'] = frame['scheduled_departure_utc']-pd.Timedelta(hours=2)
    frame['departure_outcome_event_utc'] = (frame['scheduled_departure_utc']
        + pd.to_timedelta(frame['departure_delay_minutes'], unit='m')).where(frame['label_eligible'])
    known = frame['label_eligible'] & frame['departure_delay_15_source'].notna()
    assert frame.loc[known, 'departure_delay_15_source'].eq(frame.loc[known, 'delayed_departure']).all()
    return frame


def assign_periods(frame):
    """Attach original context/train/validation/Q1 roles; final refit IDs are separate."""
    frame = frame.copy()
    frame['split'] = 'context'
    for name in ['train', 'validation', 'test']:
        mask = frame['flight_date'].between(CONFIG['split'][name+'_start'], CONFIG['split'][name+'_end'])
        frame.loc[mask, 'split'] = name
    frame['query_eligible'] = frame['label_eligible'] & frame['split'].ne('context')
    return frame


def outlier_summary(train, columns):
    """Three diagnostics, without deleting or capping any rows."""
    rows = []
    for column in columns:
        values = train[column].dropna()
        q1, q3 = values.quantile([.25, .75])
        iqr = q3-q1
        rows.append({'variable': column,
            'IQR_flags': int(((values < q1-1.5*iqr) | (values > q3+1.5*iqr)).sum()),
            'three_SD_flags': int((abs(values-values.mean()) > 3*values.std(ddof=0)).sum()),
            'above_p99': int((values > values.quantile(.99)).sum())})
    return pd.DataFrame(rows)


def check_cleaning(frame):
    """Check values/NA against the existing clean handoff, writing an audit only."""
    saved = pd.read_parquet(ROOT/'data/processed/flights_clean.parquet').set_index('flight_id')
    rebuilt = frame.set_index('flight_id').loc[saved.index]
    columns = [c for c in rebuilt if c in saved]
    pd.testing.assert_frame_equal(rebuilt[columns], saved[columns], check_dtype=False)
    OUT.mkdir(exist_ok=True, parents=True)
    (OUT/'cleaning_checks.json').write_text(json.dumps({'rows': len(saved),
        'columns_checked': columns, 'exact_values_and_missingness': True}, indent=2)+'\n')
    print(f'Cleaning matches the existing handoff: {len(saved):,} rows; {len(columns)} columns.')


def load_context():
    """Original clean context plus April–July, under the same reviewed cleaning policy."""
    old = pd.read_parquet(ROOT/'data/processed/flights_clean.parquet')
    paths = [ROOT/f'data/raw/flights/bts/2026/csv/2026_{m:02d}.csv' for m in [4, 5, 6, 7]]
    recent, _ = read_flights(ROOT, paths)
    recent, _ = decode_flights(recent)
    recent['label_eligible'] = recent['cancelled'].eq(0) & recent['departure_delay_minutes'].notna()
    recent['delayed_departure'] = recent['departure_delay_minutes'].ge(15).where(recent['label_eligible']).astype('Int8')
    recent = add_timestamps(recent)
    for column in ['distance_miles', 'scheduled_duration_minutes']:
        recent[column] = recent[column].where(recent[column].gt(0))
    context = pd.concat([old, recent], ignore_index=True)
    context['month'] = context['flight_date'].dt.month.astype('int8')
    assert context['flight_id'].is_unique
    return context


def shared_queries(context):
    """Frozen 40k fitting IDs + the original proportional seed-42 July 10k sample."""
    july = context.loc[context['flight_date'].between('2026-07-01', '2026-07-31') & context['label_eligible']]
    july_ids = sample(july, 10000)['flight_id'].tolist()
    frame = context.set_index('flight_id', drop=False).loc[COHORTS['final_refit_ids']+july_ids].reset_index(drop=True)
    frame['role'] = np.where(frame['flight_date'].dt.year.eq(2025), 'refit', 'july_reproduction')
    return frame


def calendar_features(frame):
    """Calendar fields, first/second harmonics and airport-specific daily harmonics."""
    frame = frame.copy()
    frame['month'] = frame['flight_date'].dt.month.astype('int8')
    frame['weekday'] = frame['flight_date'].dt.dayofweek.astype('int8')
    frame['departure_hour'] = (frame['scheduled_departure_minutes']//60).astype('int8')
    frame['departure_time_band'] = pd.cut(frame['departure_hour'], [-1, 5, 11, 17, 23],
        labels=['00–05 overnight', '06–11 morning', '12–17 afternoon', '18–23 evening']).astype('string')
    daily = 2*np.pi*frame['scheduled_departure_minutes']/1440
    annual = 2*np.pi*(frame['month']-1)/12
    frame['departure_sin'], frame['departure_cos'] = np.sin(daily), np.cos(daily)
    frame['month_sin'], frame['month_cos'] = np.sin(annual), np.cos(annual)
    return add_time(frame)


def read_weather():
    """Decode archived weather, excluding all versions at approved conflicting times."""
    quality = json.loads((ROOT/'reports/improvements/weather_quality.json').read_text())
    return load_weather(ROOT, [2025, 2026], quality['excluded_station_times'])


def match_weather(frame, weather):
    """Latest report 20–180 minutes before cutoff; keep unmatched flights."""
    return add_weather(frame, weather, base=True)


def history_features(frame, events):
    """Four earlier-event groups; 28-day window/7-day lag and strength-20 shrinkage."""
    return add_histories(frame, events, base=True)


def traffic_features(frame, schedules):
    """Count all origin schedules in [departure−60m, departure), including cancellations."""
    return add_traffic(frame, schedules)


def pressure_features(frame, weather):
    """Parse/roll pressure at report level, then join to the cutoff-checked weather row."""
    frame = frame.copy()
    reports = pressure_reports(weather)
    matched = frame[['origin', 'weather_observed_utc']].merge(
        reports[['origin', 'weather_observed_utc']+PRESSURE[:-1]],
        on=['origin', 'weather_observed_utc'], how='left', validate='many_to_one')
    for column in PRESSURE[:-1]:
        frame[column] = matched[column].to_numpy(dtype=float, na_value=np.nan)
    frame['altimeter_missing_flag'] = frame['altimeter_inhg'].isna().astype('int8')
    return frame


def save_features(frame, weather_audit):
    """Write the single additional Parquet and feature/source audit; no learned transforms."""
    columns = SELECTION['columns']['combined']
    assert len(frame) == 50000 and frame['flight_id'].is_unique
    numeric = [c for c in columns if c not in ['airline', 'origin', 'destination', 'weekday']]
    assert not np.isinf(frame[numeric].to_numpy(dtype=float)).any()
    OUT.mkdir(exist_ok=True, parents=True)
    frame.to_parquet(OUT/'features.parquet', index=False)
    (OUT/'feature_build.json').write_text(json.dumps({'rows': len(frame), 'fitting_rows': 40000,
        'july_rows': 10000, 'selected_features': columns, 'weather': weather_audit}, indent=2)+'\n')
    print('Saved unscaled features:', frame.groupby('role').size().to_dict())


def training_data():
    return pd.read_parquet(OUT/'features.parquet').set_index('flight_id', drop=False).loc[COHORTS['final_refit_ids']]


def cohort_table():
    metadata = pd.read_parquet(ROOT/'data/processed/features.parquet', columns=['flight_id', 'flight_date']).set_index('flight_id')
    rows = []
    for fold, groups in COHORTS['folds'].items():
        for role in ['train40', 'calibrate', 'score']:
            dates = metadata.loc[groups[role], 'flight_date']
            rows.append([fold, role, len(dates), dates.min(), dates.max()])
    return pd.DataFrame(rows, columns=['fold', 'role', 'rows', 'first_date', 'last_date'])


def development_table():
    return pd.DataFrame([{'model': MODEL_NAMES[m], 'B_package': p['package'],
        'mean_development_F1': p['mean_f1'], 'mean_development_AP': p['mean_ap'],
        'settings': str(p['params'])} for m, p in SELECTION['selected_by_model'].items()])


def threshold_table():
    return pd.DataFrame([{'model': MODEL_NAMES[m], 'inputs': fs, 'threshold': p['threshold']}
        for m, items in SELECTION['policies'].items() for fs, p in items.items() if fs in ['A', 'B']])


def model_input(train, estimator, model_id, feature_set):
    """Return X and an UNFITTED sklearn Pipeline using the exact frozen preprocessing.

    frozen_pipeline constructs ColumnTransformer/SimpleImputer/StandardScaler/
    OneHotEncoder/MissingIndicator. The notebook supplies the visible classifier.
    No .fit, scoring, sampling or tuning is performed here.
    """
    policy = SELECTION['policies'][model_id][feature_set]
    columns = SELECTION['columns'][policy['package']]
    covered = CONFIG['preprocessing']['explicit_missingness_covered_columns']
    model = frozen_pipeline(model_id, columns, policy['recipe'], policy['params'], covered)
    assert model.named_steps['classifier'].get_params() == estimator.get_params(), 'Use the frozen settings.'
    model.set_params(classifier=clone(estimator))
    return train[columns], model


def save_models(fitted, seconds):
    """Save already fitted notebook models and the frozen metadata in a separate bundle."""
    models = {}
    for key, model in fitted.items():
        m, fs = key.split('__')
        policy = SELECTION['policies'][m][fs]
        models[key] = dict(policy, columns=SELECTION['columns'][policy['package']],
                           pipeline=model, fit_seconds=seconds[key])
    joblib.dump({'models': models, 'fit_ids': COHORTS['final_refit_ids'],
        'selected_model_id': SELECTION['selected_model_id'], 'selection_run_id': SELECTION['run_id'],
        'purpose': 'reproduction of completed evaluation'}, OUT/'models.joblib', compress=3)
    return pd.Series(seconds, name='fit_seconds').rename_axis('model_inputs').to_frame()


def score_models(bundle, test):
    """Predict once per fitted pipeline; preserve SVM margins and frozen warning thresholds."""
    scores, thresholds, seconds = {}, {}, {}
    for key, item in bundle['models'].items():
        model, X = item['pipeline'], test[item['columns']]
        started = perf_counter()
        scores[key] = model.decision_function(X) if key.startswith('svm__') else model.predict_proba(X)[:, 1]
        seconds[key] = perf_counter()-started
        thresholds[key] = item['threshold']
    return scores, thresholds, pd.Series(seconds, name='prediction_seconds')


def add_baselines(scores, thresholds, test):
    """Add the same four original baselines to the score/threshold dictionaries."""
    old = joblib.load(ROOT/'models/flight_delay_models.joblib')
    historical = test['history_airline_origin_rate'].fillna(test['history_origin_rate']).fillna(old['historical_baseline_fallback'])
    scores.update(always_on_time=np.zeros(len(test)), always_delayed=np.ones(len(test)),
        fair_coin_seed42=np.random.default_rng(42).integers(0, 2, len(test)).astype(float),
        historical_risk_ranking=historical.to_numpy(dtype=float))
    thresholds.update({name: .5 for name in ['always_on_time', 'always_delayed', 'fair_coin_seed42']})
    thresholds['historical_risk_ranking'] = old['baseline_thresholds']['historical_risk_ranking']


def metric_table(y, scores, thresholds):
    """Use sklearn metrics for every model/baseline; zero-warning precision stays NaN."""
    rows = []
    for key, score in scores.items():
        assert np.isfinite(score).all()
        prediction = score >= thresholds[key]
        tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
        model, fs = key.split('__') if '__' in key else (key, 'baseline')
        rows.append({'model_id': model, 'feature_set': fs, 'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'precision': precision_score(y, prediction, zero_division=np.nan),
            'recall': recall_score(y, prediction), 'f1': f1_score(y, prediction),
            'f05': fbeta_score(y, prediction, beta=.5), 'accuracy': accuracy_score(y, prediction),
            'average_precision': average_precision_score(y, score), 'roc_auc': roc_auc_score(y, score)})
    return pd.DataFrame(rows)


def paired_gains():
    """Saved 300-repeat paired day-block intervals from the completed evaluation."""
    results = json.loads((ROOT/'reports/final_models/test_results.json').read_text())
    return pd.DataFrame(results['paired_comparisons']).query("comparison == 'B minus A'")


def error_summary(test, score, threshold):
    """Selected-model subgroup counts/rates, preserving the original support rules."""
    frame = test.rename(columns={'airline': 'reporting_airline'}).copy()
    y, predicted = frame['delayed_departure'].to_numpy(dtype=int), score >= threshold
    for name, mask in {'tp': (y == 1) & predicted, 'fp': (y == 0) & predicted,
                       'fn': (y == 1) & ~predicted, 'tn': (y == 0) & ~predicted}.items():
        frame[name] = mask.astype(int)
    settings, groups = CONFIG['evaluation'], []
    for dimension in settings['subgroup_dimensions']:
        block = frame.groupby(dimension, observed=True).agg(n=('flight_id', 'size'),
            tp=('tp', 'sum'), fp=('fp', 'sum'), fn=('fn', 'sum'), tn=('tn', 'sum'))
        supported = block['n'] >= settings['minimum_group_rows']
        for rate, numerator, denominator, minimum in [
            ('precision', block.tp, block.tp+block.fp, 'minimum_predicted_positives_for_precision'),
            ('recall', block.tp, block.tp+block.fn, 'minimum_actual_positives_for_recall'),
            ('false_positive_rate', block.fp, block.fp+block.tn, 'minimum_actual_negatives_for_fpr')]:
            block[rate] = (numerator/denominator).where(supported & denominator.ge(settings[minimum]))
        groups.append(block.rename_axis('group_value').reset_index().assign(dimension=dimension))
    return pd.concat(groups, ignore_index=True)


def priority_table(test, scores, chosen):
    """Selected model/history at 5/10/20%; same SHA256(seed:ID) tie order as publication."""
    y = test['delayed_departure'].to_numpy(dtype=int)
    tie = [hashlib.sha256(f'42:{fid}'.encode()).hexdigest() for fid in test['flight_id']]
    rows = []
    for key in [chosen, 'historical_risk_ranking']:
        ranked = pd.DataFrame({'score': scores[key], 'y': y, 'tie': tie}).sort_values(
            ['score', 'tie'], ascending=[False, True], kind='stable')
        for capacity in [.05, .10, .20]:
            k = int(np.ceil(capacity*len(y)))
            captured = int(ranked.head(k)['y'].sum())
            rows.append({'ranking': key.split('__')[0], 'budget': capacity, 'reviewed': k,
                'captured': captured, 'false_alerts': k-captured, 'recall': captured/y.sum(),
                'precision': captured/k, 'lift': captured/k/y.mean()})
    return pd.DataFrame(rows)


def save_evaluation(metrics, subgroups, priority, scores):
    """Persist reproduction results only; never changes published exports or thresholds."""
    for name, frame in [('metrics', metrics), ('subgroups', subgroups), ('priority', priority)]:
        frame.to_json(OUT/(name+'.json'), orient='records', indent=2, double_precision=15)
    np.savez_compressed(OUT/'scores.npz', **scores)
