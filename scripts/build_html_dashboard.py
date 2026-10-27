"""Package the frozen July ML results into one offline HTML dashboard."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'reports/evaluation_exports/final_july'
OUT = ROOT / 'dashboard/dist'
MODELS = ['logistic_regression', 'knn', 'svm', 'random_forest']


def records(frame):
    return json.loads(frame.to_json(orient='records', double_precision=15))


def build():
    manifest = json.loads((SOURCE / 'export_manifest.json').read_text())
    frames = {p.stem: pd.read_csv(p) for p in SOURCE.glob('*.csv')}
    for filename, expected in manifest['files'].items():
        assert hashlib.sha256((SOURCE / filename).read_bytes()).hexdigest() == expected['sha256'], filename
    metrics = frames['model_metrics']
    predictions = frames['flight_predictions']
    base = predictions.loc[(predictions.model_id == MODELS[0]) & (predictions.feature_set == 'B')].sort_values('flight_id')
    assert len(base) == 10000 and base.flight_id.is_unique
    fields = ['flight_id', 'flight_date', 'origin', 'destination', 'reporting_airline',
              'departure_time_band', 'y_true', 'scheduled_departure_utc', 'prediction_cutoff_utc']
    payload = {'manifest':manifest, 'metrics':records(metrics),
               'features':records(frames['feature_comparison']),
               'groups':records(frames['subgroup_metrics']),
               'priority':records(frames['priority_curve']),
               'flight_fields':fields, 'flights':base[fields].values.tolist(),
               'predictions':{}, 'curves':{}}
    checks = 0
    for model in MODELS:
        flight_rows = predictions.loc[(predictions.model_id == model) & (predictions.feature_set == 'B')].sort_values('flight_id')
        assert flight_rows.flight_id.tolist() == base.flight_id.tolist()
        assert flight_rows.y_true.tolist() == base.y_true.tolist()
        assert sorted(flight_rows.risk_rank.tolist()) == list(range(1,10001))
        payload['predictions'][model] = {'score':flight_rows.risk_score.tolist(),
            'predicted':flight_rows.y_pred.astype(int).tolist(), 'rank':flight_rows.risk_rank.astype(int).tolist()}
        metric = metrics.loc[(metrics.model_id == model) & (metrics.feature_set == 'B')].iloc[0]
        for label in ['TP','FP','FN','TN']:
            assert int((flight_rows.confusion_label == label).sum()) == int(metric[label.lower()])
            checks += 1
        for fraction in [.05,.1,.2]:
            p = frames['priority_curve'].query('model_id == @model and feature_set == "B" and capacity_fraction == @fraction').iloc[0]
            review = flight_rows.loc[flight_rows.risk_rank <= int(np.ceil(fraction * len(base)))]
            assert len(review) == p.reviewed_flights
            assert int(review.y_true.sum()) == p.captured_delays
            assert int((1-review.y_true).sum()) == p.false_alerts
            checks += 3
    # All precision–recall points are retained, in original point order.
    for candidate, curve in frames['pr_curve'].groupby('candidate_id',sort=False):
        payload['curves'][candidate] = curve.sort_values('point_index')[['recall','precision']].values.tolist()
    payload['audit'] = {'source_hashes_verified':6, 'flight_and_quota_checks':checks,
                        'pr_points_retained':sum(map(len,payload['curves'].values())),
                        'models_retrained':False}
    html = (ROOT/'dashboard/template.html').read_text()
    js = (ROOT/'dashboard/dashboard.js').read_text()
    data = json.dumps(payload, separators=(',',':'), ensure_ascii=False, allow_nan=False).replace('</','<\\/')
    html = html.replace('<!-- EMBEDDED_DATA -->', '<script type="application/json" id="study-data">'+data+'</script>')
    html = html.replace('<!-- EMBEDDED_APP -->', '<script id="dashboard-app">\n'+js+'\n</script>')
    OUT.mkdir(exist_ok=True,parents=True)
    target=OUT/'departure_delay_dashboard.html'
    target.write_text(html)
    audit={**payload['audit'],'run_id':manifest['run_id'],'html_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
           'bytes':target.stat().st_size,'sources':manifest['files'],'browser_qa':'pending'}
    (ROOT/'dashboard/build_manifest.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({'file':str(target),**payload['audit'],'bytes':target.stat().st_size},indent=2))


if __name__ == '__main__':
    build()
