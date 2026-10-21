"""Optional result checks for the four standalone teaching notebooks.

Reads frozen reference builders only for verification, outside the teaching workflow.
Requires no retired notebook, archive, backup or historical audit script.
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
from final_features import build_development, build_july_test
from final_modelling import make_cohorts, fingerprint

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/simple_reproduction'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


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
    selection = json.loads((ROOT/'reports/final_models/selection.json').read_text())
    provenance_differences = {p: {'selection_sha256': old, 'current_sha256': sha(ROOT/p)}
        for p, old in selection['input_sha256'].items() if sha(ROOT/p) != old}
    assert set(provenance_differences) == {'reports/preparation_manifest.json'}
    saved_cohorts = json.loads((ROOT/'reports/final_models/cohorts.json').read_text())
    folds, fit_ids, _ = make_cohorts(ROOT)
    assert folds == saved_cohorts['folds'] and fit_ids == saved_cohorts['final_refit_ids']
    assert fingerprint(fit_ids) == selection['refit_ids_sha256']
    # Threshold derivation is outside the shortened main reading sequence.
    from final_modelling import threshold
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
    train, test = [data.loc[data['role'].eq(role)].copy() for role in ['refit', 'july_reproduction']]
    assert train['flight_id'].tolist() == fit_ids and len(test) == 10000
    approved = json.loads((ROOT/'reports/improvements/weather_quality.json').read_text())['excluded_station_times']
    expected_train, _ = build_development(ROOT, set(fit_ids), approved)
    columns = selection['columns']['combined']
    compare_features(train, expected_train.reset_index(drop=True), columns)
    saved_base = pd.read_parquet(ROOT/'data/processed/features.parquet').set_index('flight_id', drop=False).loc[fit_ids]
    base_columns = json.loads((ROOT/'reports/preparation_manifest.json').read_text())['feature_engineering']['feature_sets']['B']
    compare_features(train, saved_base.reset_index(drop=True), base_columns)
    print('Every selected training feature and missing value matches', flush=True)
    expected_test, test_audit = build_july_test(ROOT, approved)
    compare_features(test, expected_test, columns)
    assert fingerprint(test['flight_id']) == json.loads((ROOT/'reports/final_models/test_results.json').read_text())['cohort_fingerprint']
    assert train['flight_date'].max() < test['flight_date'].min()
    assert train['flight_date'].dt.month.nunique() == 12
    assert test['flight_date'].between('2026-07-01', '2026-07-31').all()
    assert (data['scheduled_departure_utc']-data['prediction_cutoff_utc']).eq(pd.Timedelta(hours=2)).all()
    for prefix in ['history_origin', 'history_airline_origin', 'history_route', 'history_flight']:
        latest = data[prefix+'_latest_event_utc']
        has_history = data[prefix+'_n'].gt(0)
        assert latest.loc[has_history].ge(data.loc[has_history, 'prediction_cutoff_utc']-pd.Timedelta(days=35)).all()
        assert latest.loc[has_history].lt(data.loc[has_history, 'prediction_cutoff_utc']-pd.Timedelta(days=7)).all()
    assert data['weather_age_minutes'].dropna().between(20, 180).all()
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
    frozen = pd.read_csv(ROOT/'tableau/exports/final_july/model_metrics.csv')
    score_arrays = np.load(OUT/'scores.npz')
    test = test.sort_values('flight_id').reset_index(drop=True)
    y = test['delayed_departure'].to_numpy(dtype=int)
    predictions = pd.read_csv(ROOT/'tableau/exports/final_july/flight_predictions.csv')
    priority = pd.read_csv(ROOT/'tableau/exports/final_july/priority_curve.csv')
    rank_checks, quota_checks = 0, 0
    tie = np.array([hashlib.sha256(f'42:{fid}'.encode()).hexdigest() for fid in test['flight_id']])
    max_score_difference = 0.
    for _, row in metrics.iterrows():
        mask = frozen['model_id'].eq(row['model_id']) & frozen['feature_set'].eq(row['feature_set'])
        expected = frozen.loc[mask].iloc[0]
        count_fields = ['tn', 'fp', 'fn', 'tp']
        metric_fields = ['precision', 'recall', 'f1', 'f05', 'accuracy', 'average_precision', 'roc_auc']
        np.testing.assert_array_equal(row[count_fields].astype(int), expected[count_fields].astype(int))
        np.testing.assert_allclose(row[metric_fields].astype(float), expected[metric_fields].astype(float), rtol=0, atol=1e-10)
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
    groups = pd.read_json(OUT/'subgroups.json')
    frozen_groups = pd.read_csv(ROOT/'tableau/exports/final_july/subgroup_metrics.csv')
    frozen_groups = frozen_groups.loc[frozen_groups['model_id'].eq(selection['selected_model_id']) & frozen_groups['feature_set'].eq('B')]
    for _, row in groups.iterrows():
        expected = frozen_groups.loc[frozen_groups['group_dimension'].eq(row['dimension']) & frozen_groups['group_value'].eq(str(row['group_value']))].iloc[0]
        assert row['n'] == expected['n_flights']
        for column in ['tn', 'fp', 'fn', 'tp', 'precision', 'recall', 'false_positive_rate']:
            np.testing.assert_allclose(row[column], expected[column], equal_nan=True, rtol=0, atol=1e-10)
    # Confirm that the offline dashboard still embeds the frozen sources and metrics.
    html = (ROOT/'dashboard/dist/departure_delay_dashboard.html').read_text()
    marker = '<script type="application/json" id="study-data">'
    embedded = json.loads(html.split(marker, 1)[1].split('</script>', 1)[0])
    for filename, evidence in embedded['manifest']['files'].items():
        assert sha(ROOT/'tableau/exports/final_july'/filename) == evidence['sha256']
    for _, row in metrics.iterrows():
        expected = next(r for r in embedded['metrics'] if r['model_id'] == row['model_id'] and r['feature_set'] == row['feature_set'])
        for column in ['tn', 'fp', 'fn', 'tp', 'f1', 'average_precision']:
            np.testing.assert_allclose(row[column], expected[column], rtol=0, atol=1e-10)
    execution = json.loads((OUT/'execution.json').read_text())
    figures = lines = code_cells = visible_fit_calls = 0
    cell_lengths = []
    expected_figures = [3, 2, 0, 4]
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
        assert count_figures == expected_figures[position]
        figures += count_figures
        lengths = [len(c.source.splitlines()) for c in cells]
        cell_lengths.extend(lengths)
        lines += sum(lengths)
        assert execution[path.name]['visible_code_lines'] == sum(lengths)
        assert execution[path.name]['code_cells'] == len(cells)
        for cell in cells:
            visible_fit_calls += sum(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'fit' for node in ast.walk(ast.parse(cell.source)))
        code_cells += len(cells)
    assert len(execution) == 4 and figures == 9 and 100 <= lines <= 250
    assert max(cell_lengths) <= 8 and visible_fit_calls == 4
    helper = ast.parse((ROOT/'scripts/simple_workflow.py').read_text())
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr in ['fit', 'fit_transform'] for n in ast.walk(helper))
    timings = [execution[name] for name in sorted(execution)]
    assert all(a['finished_at_utc'] <= b['started_at_utc'] for a, b in zip(timings, timings[1:]))
    result = {'status': 'passed', 'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'reproduction of the completed July evaluation; no new test or selection',
        'visible_code_lines': lines, 'code_cells': code_cells, 'essential_figures': figures,
        'maximum_cell_lines': max(cell_lengths), 'median_cell_lines': float(np.median(cell_lengths)),
        'visible_classifier_fit_calls': visible_fit_calls, 'helper_fits_models': False,
        'validation_threshold_derivations_checked': threshold_checks,
        'fresh_kernels': 4, 'benign_font_cache_startup_notices': sum(len(d['stderr']) for d in execution.values()),
        'errors': 0, 'warnings': 0, 'feature_columns_checked': len(columns), 'feature_rows_checked': len(data),
        'feature_values_checked': len(columns)*len(data), 'feature_missingness_matches': True,
        'principal_A_B_models': 8, 'baseline_comparisons': 4, 'exact_rank_checks': rank_checks,
        'review_quota_checks': quota_checks, 'subgroup_rows_checked': len(groups),
        'maximum_absolute_score_difference': max_score_difference,
        'learned_preprocessing_arrays_checked': learned, 'chronological_cohorts_match': True,
        'training_only_preprocessing': True, 'dashboard_summaries_reconcile': True,
        'protected_files_unchanged': len(baseline), 'historical_sources_required': False,
        'tableau_files_unchanged': sum(p.startswith('tableau/') for p in baseline),
        'preexisting_selection_hash_differences': provenance_differences,
        'provenance_note': 'The saved preparation manifest differs from its selection-time hash. Both processed datasets still match selection; every feature and principal prediction is checked directly here.',
        'july_population': test_audit['eligible_july'], 'software': {'python': platform.python_version(),
        **{p: version(p) for p in ['pandas', 'numpy', 'scikit-learn', 'pyarrow', 'joblib', 'nbclient']}}}
    (OUT/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
