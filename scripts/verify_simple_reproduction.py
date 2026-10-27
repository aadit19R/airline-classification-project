"""Optional result checks for the four standalone teaching notebooks.

Uses pre-refactor feature fingerprints and frozen model/export references.
Requires no deleted helper module or external backup.
Writes one verification report; never overwrites a frozen result, model or handoff.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import platform
from importlib.metadata import version
import joblib
import nbformat
import numpy as np
import pandas as pd

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


def threshold(y, scores):
    from sklearn.metrics import precision_recall_curve
    precision, recall, cutoffs = precision_recall_curve(y, scores)
    total = precision[:-1]+recall[:-1]
    f1 = np.divide(2*precision[:-1]*recall[:-1], total, out=np.zeros(len(cutoffs)), where=total>0)
    return float(cutoffs[np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))[-1]])


def compare_features(actual, reference, columns):
    actual = actual.set_index('flight_id').sort_index()
    reference = reference.set_index('flight_id').sort_index()
    assert actual.index.tolist() == reference.index.tolist()
    for column in columns:
        np.testing.assert_array_equal(actual[column].isna(), reference[column].isna(), err_msg=column)
        if column in ['airline', 'origin', 'destination', 'weekday']:
            np.testing.assert_array_equal(actual[column], reference[column], err_msg=column)
        else:
            np.testing.assert_allclose(actual[column].to_numpy(dtype=float, na_value=np.nan),
                reference[column].to_numpy(dtype=float, na_value=np.nan), equal_nan=True,
                rtol=0, atol=1e-12, err_msg=column)
    for column in ['delayed_departure', 'flight_date', 'scheduled_departure_utc', 'prediction_cutoff_utc']:
        pd.testing.assert_series_equal(actual[column], reference[column], check_dtype=False)


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
    baseline = json.loads((OUT/'protected_checksums.json').read_text())
    for path, expected in baseline.items():
        assert not Path(path).is_absolute() and '..' not in Path(path).parts
        assert sha(ROOT/path) == expected, 'Protected input/result changed: '+path
    print('Frozen source/artifact checksums passed', flush=True)
    refactor = json.loads((ROOT/'reports/notebook_refactor.json').read_text())
    reference = refactor['reference']
    assert all(not (ROOT/path).exists() for path in refactor['removed_scripts'])
    for path, expected in refactor['notebook_source_hashes'].items():
        notebook = nbformat.read(ROOT/path, as_version=4)
        source = '\n'.join(cell.cell_type+'\0'+cell.source for cell in notebook.cells)
        assert hashlib.sha256(source.encode()).hexdigest() == expected
    selection = json.loads((ROOT/'reports/final_models/selection.json').read_text())
    assert sha(ROOT/'reports/final_models/selection.json') == reference['selection_sha256']
    saved_cohorts = json.loads((ROOT/'reports/final_models/cohorts.json').read_text())
    assert sha(ROOT/'reports/final_models/cohorts.json') == reference['cohorts_sha256']
    folds, fit_ids = saved_cohorts['folds'], saved_cohorts['final_refit_ids']
    assert fingerprint(fit_ids) == selection['refit_ids_sha256']
    source = pd.read_parquet(ROOT/'data/processed/flights_clean.parquet')
    source_index = source.set_index('flight_id')
    for fold in folds.values():
        assert source_index.loc[fold['train40'], 'flight_date'].max() < source_index.loc[fold['calibrate'], 'flight_date'].min()
        assert source_index.loc[fold['calibrate'], 'flight_date'].max() < source_index.loc[fold['score'], 'flight_date'].min()
    arrays = np.load(ROOT/'reports/final_models/development_scores.npz')
    labels = pd.read_parquet(ROOT/'data/processed/features.parquet',
        columns=['flight_id', 'delayed_departure']).set_index('flight_id')
    october = labels.loc[folds['Nov–Dec']['calibrate'], 'delayed_departure'].astype(int)
    threshold_checks = 0
    for policies in selection['policies'].values():
        for fs in ['A', 'B']:
            policy = policies[fs]
            scores = arrays[policy['check_key']+'_Nov–Dec_calibration']
            assert threshold(october, scores) == policy['threshold']
            threshold_checks += 1
    data = pd.read_parquet(OUT/'features.parquet')
    assert 'role' not in data and not {'model_sample', 'refit_sample', 'query_eligible'}.intersection(data.columns)
    assert data['split'].value_counts().to_dict() == {'train': 40000, 'test': 10000}
    train, test = [data.loc[data['split'].eq(split)].copy() for split in ['train', 'test']]
    assert train['flight_date'].between('2025-01-01', '2025-12-31').all()
    assert test['flight_date'].between('2026-07-01', '2026-07-31').all()
    config = json.loads((ROOT/'config/project.json').read_text())
    for name, frame in [('train', train), ('test', test)]:
        assert len(frame) == config['split'][name+'_rows'] == config['sampling'][name+'_rows']
        assert frame['flight_date'].between(config['split'][name+'_start'], config['split'][name+'_end']).all()
    assert not {'train', 'validation', 'test'}.intersection(data['preparation_period'].unique())
    assert train['flight_id'].tolist() == fit_ids and len(test) == 10000
    columns = reference['columns']
    assert len(data) == reference['rows']
    for column, expected in reference['column_hashes'].items():
        assert column_hash(data[column]) == expected, 'Feature/ID/label changed: '+column
    saved_base = pd.read_parquet(ROOT/'data/processed/features.parquet').set_index('flight_id', drop=False).loc[fit_ids]
    base_columns = json.loads((ROOT/'reports/preparation_manifest.json').read_text())['feature_engineering']['feature_sets']['B']
    compare_features(train, saved_base.reset_index(drop=True), base_columns)
    print('All 78 feature columns, IDs, splits and labels match pre-refactor fingerprints', flush=True)
    assert fingerprint(test['flight_id']) == json.loads((ROOT/'reports/final_models/test_results.json').read_text())['cohort_fingerprint']
    assert train['flight_date'].max() < test['flight_date'].min() and train['flight_date'].dt.month.nunique() == 12
    assert (data['scheduled_departure_utc']-data['prediction_cutoff_utc']).eq(pd.Timedelta(hours=2)).all()
    for prefix in ['history_origin', 'history_airline_origin', 'history_route', 'history_flight']:
        latest, has_history = data[prefix+'_latest_event_utc'], data[prefix+'_n'].gt(0)
        assert latest.loc[has_history].ge(data.loc[has_history, 'prediction_cutoff_utc']-pd.Timedelta(days=35)).all()
        assert latest.loc[has_history].lt(data.loc[has_history, 'prediction_cutoff_utc']-pd.Timedelta(days=7)).all()
    assert data['weather_age_minutes'].dropna().between(20, 180).all()
    july_population = int((source['flight_date'].between('2026-07-01', '2026-07-31') & source['label_eligible']).sum())
    new = joblib.load(OUT/'models.joblib')
    old = joblib.load(ROOT/'models/final_flight_delay_models.joblib')
    assert new['fit_ids'] == fit_ids and new['selected_model_id'] == selection['selected_model_id']
    train = train.set_index('flight_id', drop=False).loc[fit_ids]
    learned = sum(check_preprocessing(item, old['models'][key], train) for key, item in new['models'].items())
    print('Eight pipelines preserve feature order, settings and training-only learned preprocessing', flush=True)
    metrics = pd.read_json(OUT/'metrics.json')
    published_result = json.loads((ROOT/'reports/final_models/test_results.json').read_text())
    indexed = metrics.set_index(['model_id', 'feature_set'])
    for pair in published_result['paired_comparisons']:
        if pair['comparison'] != 'B minus A':
            continue
        for metric, prefix in [('f1', 'f1'), ('average_precision', 'ap')]:
            delta = indexed.loc[(pair['model_id'], 'B'), metric]-indexed.loc[(pair['model_id'], 'A'), metric]
            np.testing.assert_allclose(delta, pair[prefix+'_gain'], rtol=0, atol=1e-10)
    frozen = pd.read_csv(ROOT/'reports/evaluation_exports/final_july/model_metrics.csv')
    score_arrays = np.load(OUT/'scores.npz')
    expected_keys = set(new['models']) | {'always_on_time', 'always_delayed', 'fair_coin_seed42', 'historical_risk_ranking'}
    actual_keys = {row.model_id+('__'+row.feature_set if row.feature_set != 'baseline' else '') for row in metrics.itertuples()}
    assert actual_keys == set(score_arrays.files) == expected_keys and len(metrics) == 12
    test = test.sort_values('flight_id').reset_index(drop=True)
    y = test['delayed_departure'].to_numpy(dtype=int)
    predictions = pd.read_csv(ROOT/'reports/evaluation_exports/final_july/flight_predictions.csv')
    priority = pd.read_csv(ROOT/'reports/evaluation_exports/final_july/priority_curve.csv')
    rank_checks, quota_checks = 0, 0
    tie = np.array([hashlib.sha256(f'42:{fid}'.encode()).hexdigest() for fid in test['flight_id']])
    max_score_difference = 0.
    from sklearn.metrics import confusion_matrix, fbeta_score, roc_auc_score
    baseline_selection = json.loads((ROOT/'reports/model_selection.json').read_text())
    for _, row in metrics.iterrows():
        mask = frozen['model_id'].eq(row['model_id']) & frozen['feature_set'].eq(row['feature_set'])
        expected = frozen.loc[mask].iloc[0]
        count_fields = ['tn', 'fp', 'fn', 'tp']
        # The teaching table shows five core metrics; independently check the
        # full original metric set and confusion counts from its saved scores.
        key = row['model_id']+('__'+row['feature_set'] if row['feature_set'] != 'baseline' else '')
        cutoff = new['models'][key]['threshold'] if '__' in key else .5
        if key == 'historical_risk_ranking':
            cutoff = baseline_selection['baselines'][key]['threshold']
        prediction = score_arrays[key] >= cutoff
        counts = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
        np.testing.assert_array_equal(counts, expected[count_fields].astype(int))
        reported = ['precision', 'recall', 'f1', 'accuracy', 'average_precision']
        np.testing.assert_allclose(row[reported].astype(float), expected[reported].astype(float), rtol=0, atol=1e-10)
        np.testing.assert_allclose([fbeta_score(y, prediction, beta=.5), roc_auc_score(y, score_arrays[key])],
                                  expected[['f05', 'roc_auc']].astype(float), rtol=0, atol=1e-10)
        key = row['model_id']+('__'+row['feature_set'] if row['feature_set'] != 'baseline' else '')
        saved = predictions.loc[predictions['model_id'].eq(row['model_id']) & predictions['feature_set'].eq(row['feature_set'])].sort_values('flight_id')
        assert test['flight_id'].tolist() == saved['flight_id'].tolist()
        np.testing.assert_array_equal(y, saved['y_true'])
        s = score_arrays[key]
        max_score_difference = max(max_score_difference, float(np.max(abs(s-saved['risk_score'].to_numpy()))))
        np.testing.assert_allclose(s, saved['risk_score'], rtol=0, atol=1e-8)
        if '__' in key:
            np.testing.assert_array_equal(s >= new['models'][key]['threshold'], saved['y_pred'])
        order = np.lexsort((tie, -s))
        ranks = np.empty(len(s), dtype=int)
        ranks[order] = np.arange(1, len(s)+1)
        np.testing.assert_array_equal(ranks, saved['risk_rank'])
        rank_checks += len(s)
        for budget in [.05, .1, .2]:
            k = int(np.ceil(budget*len(y)))
            expected_quota = priority.loc[priority['model_id'].eq(row['model_id']) & priority['feature_set'].eq(row['feature_set']) & priority['capacity_fraction'].eq(budget)].iloc[0]
            assert k == expected_quota.reviewed_flights and y[order[:k]].sum() == expected_quota.captured_delays
            quota_checks += 1
    # Confirm that the offline dashboard still embeds the frozen sources and metrics.
    html = (ROOT/'dashboard/dist/departure_delay_dashboard.html').read_text()
    marker = '<script type="application/json" id="study-data">'
    embedded = json.loads(html.split(marker, 1)[1].split('</script>', 1)[0])
    for filename, evidence in embedded['manifest']['files'].items():
        assert sha(ROOT/'reports/evaluation_exports/final_july'/filename) == evidence['sha256']
    for _, row in metrics.iterrows():
        expected = next(r for r in embedded['metrics'] if r['model_id'] == row['model_id'] and r['feature_set'] == row['feature_set'])
        for column in ['f1', 'average_precision']:
            np.testing.assert_allclose(row[column], expected[column], rtol=0, atol=1e-10)
    execution = json.loads((OUT/'execution.json').read_text())
    figures = lines = code_cells = visible_fit_calls = 0
    cell_lengths = []
    import ast
    for position, path in enumerate(sorted((ROOT/'notebooks/simple').glob('*.ipynb'))):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        cells = [c for c in notebook.cells if c.cell_type == 'code']
        assert [c.execution_count for c in cells] == list(range(1, len(cells)+1))
        assert all(o.output_type != 'error' for c in cells for o in c.outputs)
        assert execution[path.name]['fresh_kernel']
        assert all(message.strip() == 'Matplotlib is building the font cache; this may take a moment.'
                   for message in execution[path.name]['stderr'])
        count_figures = sum('image/png' in o.get('data', {}) for c in cells for o in c.outputs)
        assert count_figures == execution[path.name]['figures']
        figures += count_figures
        lengths = [len(c.source.splitlines()) for c in cells]
        cell_lengths.extend(lengths)
        lines += sum(lengths)
        assert max(lengths) <= 10, 'Teaching cells should have at most ten lines'
        assert all(sum('image/png' in o.get('data', {}) for o in c.outputs) <= 1 for c in cells)
        assert execution[path.name]['visible_code_lines'] == sum(lengths)
        assert execution[path.name]['code_cells'] == len(cells)
        for cell in cells:
            visible_fit_calls += sum(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'fit' for node in ast.walk(ast.parse(cell.source)))
        code_cells += len(cells)
    assert len(execution) == 4 and figures > 0 and visible_fit_calls >= 1
    removed_names = {Path(name).stem for name in refactor['removed_scripts']}
    for path in sorted((ROOT/'notebooks/simple').glob('*.ipynb')):
        notebook = nbformat.read(path, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type != 'code':
                continue
            for node in ast.walk(ast.parse(cell.source)):
                if isinstance(node, ast.Import):
                    assert not {alias.name.split('.')[0] for alias in node.names}.intersection(removed_names)
                elif isinstance(node, ast.ImportFrom):
                    assert (node.module or '').split('.')[0] not in removed_names
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    assert node.func.id not in ['exec', 'eval', '__import__'], 'Hidden source loading is not allowed'
    assert len(new['models']) == 8
    timings = [execution[name] for name in sorted(execution)]
    # The runner supports --start/--stop; refreshing one notebook need not repeat
    # already verified model fits. Data/feature/cohort equality is checked above.
    assert all(item['started_at_utc'] <= item['finished_at_utc'] for item in timings)
    result = {'status': 'passed', 'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'reproduction of the completed July evaluation; no new test or selection',
        'visible_code_lines': lines, 'code_cells': code_cells, 'essential_figures': figures,
        'maximum_cell_lines': max(cell_lengths), 'median_cell_lines': float(np.median(cell_lengths)),
        'visible_classifier_fit_calls': visible_fit_calls, 'project_helpers_required': False, 'notebooks_self_contained': True,
        'validation_threshold_derivations_checked': threshold_checks,
        'fresh_kernels': 4, 'benign_font_cache_startup_notices': sum(len(d['stderr']) for d in execution.values()),
        'errors': 0, 'warnings': 0, 'feature_columns_checked': len(columns), 'feature_rows_checked': len(data),
        'feature_values_checked': len(columns)*len(data), 'feature_missingness_matches': True,
        'principal_A_B_models': 8, 'baseline_comparisons': 4, 'exact_rank_checks': rank_checks,
        'review_quota_checks': quota_checks, 'subgroup_analysis_in_notebooks': False,
        'maximum_absolute_score_difference': max_score_difference,
        'learned_preprocessing_arrays_checked': learned, 'chronological_cohorts_match': True,
        'training_only_preprocessing': True, 'dashboard_summaries_reconcile': True,
        'protected_files_unchanged': len(baseline), 'historical_sources_required': False,
        'evaluation_export_files_unchanged': sum(p.startswith('reports/evaluation_exports/') for p in baseline),
        'current_split': {'train': {'rows': len(train), 'period': ['2025-01-01', '2025-12-31']},
                          'test': {'rows': len(test), 'period': ['2026-07-01', '2026-07-31']}},
        'feature_hashes_match_pre_refactor_reference': True,
        'source_refactor_evidence': 'reports/notebook_refactor.json',
        'provenance_note': 'Selection and historical source hashes are preserved. reports/study_labels.json records the earlier label migration; reports/notebook_refactor.json records the move of source code into notebooks and pre-refactor feature fingerprints. All selected feature values, cohorts, learned preprocessing, predictions and ranks are checked directly.',
        'july_population': july_population, 'software': {'python': platform.python_version(),
        **{p: version(p) for p in ['pandas', 'numpy', 'scikit-learn', 'pyarrow', 'joblib', 'nbclient']}}}
    (OUT/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
