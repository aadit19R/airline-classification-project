"""Check current notebook outputs against a compact frozen July reference.

Run after notebooks 01-04. Uses no older experiment reports or large CSV exports.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import ast
import joblib
import nbformat
import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, average_precision_score)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/simple_reproduction'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fingerprint(ids):
    return hashlib.sha256('\n'.join(sorted(map(str, ids))).encode()).hexdigest()


def column_hash(series):
    if series.name in ['airline', 'origin', 'destination', 'weekday', 'flight_id', 'split']:
        payload = series.astype('string').fillna('<missing>').to_json(orient='values').encode()
    else:
        values = series.to_numpy(dtype='<f8', na_value=np.nan).copy()
        values[np.isnan(values)] = np.nan
        payload = values.tobytes()
    return hashlib.sha256(payload).hexdigest()


def check_preprocessing(item, frozen, train):
    assert item['columns'] == frozen['columns']
    assert item['params'] == frozen['params'] and item['recipe'] == frozen['recipe']
    assert item['threshold'] == frozen['threshold']
    pipeline = item['pipeline']
    classifier = pipeline.named_steps['classifier']
    if hasattr(classifier, 'fit_status_'):
        assert classifier.fit_status_ == 0
    if classifier.__class__.__name__ == 'LogisticRegression':
        assert classifier.n_iter_.max() < classifier.max_iter
    assert pipeline.named_steps['classifier'].get_params() == frozen['pipeline'].named_steps['classifier'].get_params()
    prep, original = pipeline.named_steps['preprocess'], frozen['pipeline'].named_steps['preprocess']
    assert prep.feature_names_in_.tolist() == item['columns']
    assert len(prep.transformers_) == len(original.transformers_)
    learned = []
    for (name, transform, columns), (old_name, old_transform, old_columns) in zip(prep.transformers_, original.transformers_):
        assert name == old_name and columns == old_columns
        if name == 'remainder':
            continue
        steps = transform.named_steps if hasattr(transform, 'named_steps') else {'transform': transform}
        old_steps = old_transform.named_steps if hasattr(old_transform, 'named_steps') else {'transform': old_transform}
        for step, estimator in steps.items():
            old_estimator = old_steps[step]
            for attribute in ['statistics_', 'mean_', 'scale_', 'var_', 'features_']:
                if hasattr(estimator, attribute):
                    actual, expected = getattr(estimator, attribute), getattr(old_estimator, attribute)
                    if np.asarray(actual).dtype.kind in 'biufc':
                        np.testing.assert_allclose(actual, expected, equal_nan=True, rtol=0, atol=1e-12)
                    else:
                        np.testing.assert_array_equal(actual, expected)
                    learned.append(name+'/'+step+'/'+attribute)
            if hasattr(estimator, 'categories_'):
                for category, previous in zip(estimator.categories_, old_estimator.categories_):
                    np.testing.assert_array_equal(category, previous)
            if hasattr(estimator, 'strategy') and estimator.strategy == 'median':
                expected = train[columns].median().fillna(0).to_numpy(dtype=float)
                np.testing.assert_allclose(estimator.statistics_, expected, rtol=0, atol=1e-12)
        if 'scale' in steps:
            scaler = steps['scale']
            assert np.all(scaler.n_samples_seen_ == len(train))
            numeric = steps['impute'].transform(train[columns])
            np.testing.assert_allclose(scaler.mean_, numeric.mean(axis=0), rtol=0, atol=1e-10)
            np.testing.assert_allclose(scaler.var_, numeric.var(axis=0), rtol=1e-12, atol=1e-10)
        if 'encode' in steps:
            encoded_input = steps['impute'].transform(train[columns])
            for i, category in enumerate(steps['encode'].categories_):
                np.testing.assert_array_equal(category, np.unique(encoded_input[:, i]))
    assert not {'departure_delay_minutes', 'departure_delay_15_source', 'cancelled', 'diverted',
                'delayed_departure', 'departure_outcome_event_utc', 'flight_id'}.intersection(item['columns'])
    return len(learned)


def main():
    reference = json.loads((ROOT/'reports/final_models/reference.json').read_text())
    assert sha(ROOT/'reports/final_models/selection.json') == reference['selection_sha256']
    assert sha(ROOT/'reports/final_models/cohorts.json') == reference['cohorts_sha256']
    selection = json.loads((ROOT/'reports/final_models/selection.json').read_text())
    cohorts = json.loads((ROOT/'reports/final_models/cohorts.json').read_text())
    inputs = json.loads((ROOT/'data/NOTEBOOK_INPUTS.json').read_text())
    assert all((ROOT/item['path']).is_file() for item in inputs['files'])
    source = pd.read_parquet(ROOT/'data/processed/flights_clean.parquet')
    assert len(source) == 1374996
    assert not source['flight_date'].between('2026-01-01', '2026-04-30').any()
    assert sorted(source['flight_date'].dt.strftime('%Y-%m').unique()) == inputs['flight_months']
    dates = source.set_index('flight_id')['flight_date']
    for fold in cohorts['folds'].values():
        assert dates.loc[fold['train40']].max() < dates.loc[fold['calibrate']].min()
        assert dates.loc[fold['calibrate']].max() < dates.loc[fold['score']].min()
    data = pd.read_parquet(OUT/'features.parquet')
    assert len(data) == reference['rows'] == 50000
    for column, expected in reference['column_hashes'].items():
        assert column_hash(data[column]) == expected, 'Changed feature, ID or label: '+column
    train = data.loc[data['split'].eq('train')].copy()
    test = data.loc[data['split'].eq('test')].sort_values('flight_id').reset_index(drop=True)
    assert len(train) == 40000 and len(test) == 10000
    assert train['flight_date'].dt.year.eq(2025).all()
    assert test['flight_date'].between('2026-07-01', '2026-07-31').all()
    assert train['flight_id'].tolist() == cohorts['final_refit_ids']
    assert fingerprint(train['flight_id']) == reference['training_ids_sha256']
    assert fingerprint(test['flight_id']) == reference['test_ids_sha256']
    assert (data['scheduled_departure_utc']-data['prediction_cutoff_utc']).eq(pd.Timedelta(hours=2)).all()
    assert data['weather_age_minutes'].dropna().between(20, 180).all()
    for prefix in ['history_origin', 'history_airline_origin', 'history_route', 'history_flight']:
        latest = data[prefix+'_latest_event_utc']
        available = data[prefix+'_n'] > 0
        assert latest.loc[available].ge(data.loc[available, 'prediction_cutoff_utc']-pd.Timedelta(days=35)).all()
        assert latest.loc[available].lt(data.loc[available, 'prediction_cutoff_utc']-pd.Timedelta(days=7)).all()
    bundle = joblib.load(OUT/'models.joblib')
    frozen = joblib.load(ROOT/'models/final_flight_delay_models.joblib')
    assert bundle['fit_ids'] == cohorts['final_refit_ids']
    assert bundle['selected_model_id'] == selection['selected_model_id'] == reference['selected_model_id']
    assert len(bundle['models']) == 8
    train = train.set_index('flight_id', drop=False).loc[cohorts['final_refit_ids']]
    learned = sum(check_preprocessing(item, frozen['models'][key], train)
                  for key, item in bundle['models'].items())
    print('All 78 feature columns and eight training-only pipelines match the original reference.', flush=True)
    metrics = pd.read_json(OUT/'metrics.json').set_index(['model_id', 'feature_set'])
    scores = np.load(OUT/'scores.npz')
    original_scores = np.load(ROOT/'reports/final_models/reference_scores.npz')
    expected_keys = set(reference['score_keys'])
    assert set(scores.files) == set(original_scores.files) == expected_keys and len(metrics) == 12
    y = test['delayed_departure'].to_numpy(dtype=int)
    ties = [hashlib.sha256(f'42:{fid}'.encode()).hexdigest() for fid in test['flight_id']]
    baseline = json.loads((ROOT/'reports/model_selection.json').read_text())
    assert baseline['baselines']['historical_risk_ranking']['threshold'] == reference['historical_baseline']['threshold']
    max_difference = 0.
    for expected in reference['model_metrics']:
        name, feature_set = expected['model_id'], expected['feature_set']
        key = name+('__'+feature_set if feature_set != 'baseline' else '')
        score = scores[key]
        np.testing.assert_allclose(score, original_scores[key], rtol=0, atol=reference['score_tolerance'])
        max_difference = max(max_difference, float(np.max(abs(score-original_scores[key]))))
        threshold = bundle['models'][key]['threshold'] if feature_set != 'baseline' else .5
        if key == 'historical_risk_ranking':
            threshold = reference['historical_baseline']['threshold']
        prediction = score >= threshold
        original_prediction = original_scores[key] >= threshold
        np.testing.assert_array_equal(prediction, original_prediction)
        counts = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
        np.testing.assert_array_equal(counts, [expected[c] for c in ['tn', 'fp', 'fn', 'tp']])
        calculated = {'accuracy': accuracy_score(y, prediction),
                      'precision': precision_score(y, prediction, zero_division=np.nan),
                      'recall': recall_score(y, prediction), 'f1': f1_score(y, prediction),
                      'average_precision': average_precision_score(y, score)}
        for metric, value in calculated.items():
            wanted = np.nan if expected[metric] is None else expected[metric]
            np.testing.assert_allclose(value, wanted, rtol=0, atol=1e-10, equal_nan=True)
            np.testing.assert_allclose(metrics.loc[(name, feature_set), metric], wanted, rtol=0, atol=1e-10, equal_nan=True)
        order = np.lexsort((ties, -score))
        ranks = np.empty(len(score), dtype='<i4')
        ranks[order] = np.arange(1, len(score)+1)
        assert hashlib.sha256(ranks.tobytes()).hexdigest() == reference['rank_hashes'][key]
        for quota in [q for q in reference['review_quotas'] if q['key'] == key]:
            n = int(np.ceil(quota['budget']*len(y)))
            assert n == quota['reviewed'] and int(y[order[:n]].sum()) == quota['captured']
    cell_count = figures = 0
    for path in sorted((ROOT/'notebooks/simple').glob('*.ipynb')):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type != 'code':
                continue
            cell_count += 1
            assert len(cell.source.splitlines()) <= 10
            assert all(o.output_type != 'error' for o in cell.outputs)
            figures += sum('image/png' in o.get('data', {}) for o in cell.outputs)
            for node in ast.walk(ast.parse(cell.source)):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    assert node.func.id not in ['exec', 'eval', '__import__']
    result = {'status': 'passed', 'verified_at_utc': datetime.now(timezone.utc).isoformat(),
              'purpose': 'Verify current notebook data, trained pipelines and July results against a compact frozen reference.',
              'train_rows': len(train), 'test_rows': len(test), 'active_source_rows': len(source),
              'feature_columns_checked': len(reference['feature_columns']),
              'feature_values_checked': len(data)*len(reference['feature_columns']),
              'pipelines_checked': len(bundle['models']), 'baseline_rules_checked': 4,
              'learned_preprocessing_arrays_checked': learned,
              'risk_ranks_checked': len(y)*len(expected_keys),
              'review_quotas_checked': len(reference['review_quotas']),
              'maximum_absolute_score_difference': max_difference,
              'code_cells': cell_count, 'figures': figures,
              'older_experiment_reports_required': False,
              'large_evaluation_exports_required': False,
              'reference_files': ['reports/final_models/reference.json', 'reports/final_models/reference_scores.npz']}
    (OUT/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
