from pathlib import Path
import json,hashlib,nbformat,joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score,precision_recall_curve,confusion_matrix,
    precision_score,recall_score,f1_score,fbeta_score,accuracy_score,roc_auc_score)
import sys
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'scripts'))
def metrics(y,s,t):
    predicted=np.asarray(s)>=t
    tn,fp,fn,tp=confusion_matrix(y,predicted,labels=[0,1]).ravel()
    return {'n_flights':len(y),'actual_positive':int(np.sum(y)),
      'predicted_positive':int(predicted.sum()),'tp':tp,'fp':fp,'fn':fn,'tn':tn,
      'prevalence':float(np.mean(y)),
      'precision':precision_score(y,predicted,zero_division=np.nan),
      'recall':recall_score(y,predicted), 'f1':f1_score(y,predicted),
      'f05':fbeta_score(y,predicted,beta=.5), 'accuracy':accuracy_score(y,predicted),
      'roc_auc':roc_auc_score(y,s),'average_precision':average_precision_score(y,s)}

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
r=root/'reports/improvements';out=root/'tableau/exports/improved'
protected=json.loads((r/'baseline_fingerprints.json').read_text())
from baseline_integrity import verify_baseline_files
baseline_integrity=verify_baseline_files(root,protected)
selection=json.loads((r/'selection.json').read_text())
access=json.loads((r/'test_access.json').read_text())
results=json.loads((r/'test_results.json').read_text())
manifest=json.loads((out/'export_manifest.json').read_text())
assert not selection['test_outcomes_loaded']
assert pd.Timestamp(selection['frozen_at_utc'])<pd.Timestamp(access['first_outcomes_load_started_at_utc'])
assert not results['settings_changed_after_test']
bundle=joblib.load(root/'models/improved_flight_delay_models.joblib')
assert bundle['selection_sha256']==sha(r/'selection.json')
assert len(bundle['models'])==8 and bundle['training_rows'] in [20000,40000]
assert sha(root/'scripts/improvement_features.py')==selection['helper_sha256']
notebooks={}
for name in ['04_model_improvements.ipynb','05_fresh_test_evaluation.ipynb']:
    nb=nbformat.read(root/'notebooks'/name,as_version=4)
    errors=[o for c in nb.cells for o in c.get('outputs',[]) if o.output_type=='error']
    warnings=[o.get('text','') for c in nb.cells for o in c.get('outputs',[]) if o.output_type=='stream' and o.get('name')=='stderr']
    assert not errors,errors
    assert not warnings,warnings
    assert all(c.execution_count for c in nb.cells if c.cell_type=='code')
    for c in nb.cells:
        if c.cell_type=='code':compile(c.source,name,'exec')
    notebooks[name]={'code_cells':sum(c.cell_type=='code' for c in nb.cells),
                     'figures':sum('image/png' in o.get('data',{}) for c in nb.cells for o in c.get('outputs',[])),
                     'errors':len(errors),'stderr':len(warnings)}
tables={name:pd.read_csv(out/name,float_precision='round_trip') for name in manifest['files']}
for name,info in manifest['files'].items():
    assert len(tables[name])==info['rows'] and sha(out/name)==info['sha256']
p=tables['flight_predictions.csv'];mm=tables['model_metrics.csv'];curves=tables['pr_curve.csv']
priorities=tables['priority_curve.csv'];subgroups=tables['subgroup_metrics.csv']
assert len(p)==180000 and len(mm)==26
truth=None
for candidate,block in p.groupby('candidate_id',sort=False):
    assert len(block)==10000 and block['flight_id'].is_unique
    ordered=block.sort_values('flight_id');current=ordered[['flight_id','y_true']].reset_index(drop=True)
    if truth is None:truth=current
    else:pd.testing.assert_frame_equal(truth,current)
    y=block['y_true'].to_numpy();s=block['risk_score'].to_numpy()
    np.testing.assert_array_equal(block['y_pred'],s>=block['decision_threshold'].iloc[0])
    assert block['risk_rank'].tolist()==list(range(1,10001))
    assert block['risk_score'].is_monotonic_decreasing
    assert block['flight_date'].between('2026-04-01','2026-06-30').all()
    for _,row in mm.loc[mm.candidate_id.eq(candidate)].iterrows():
        m=metrics(y,s,row['decision_threshold'])
        for field,value in m.items():np.testing.assert_allclose(row[field],value,rtol=0,atol=1e-12,equal_nan=True)
    q=curves.loc[curves.candidate_id.eq(candidate)].sort_values('point_index')
    pr,re,t=precision_recall_curve(y,s)
    np.testing.assert_allclose(q['precision'],pr,rtol=0,atol=1e-12)
    np.testing.assert_allclose(q['recall'],re,rtol=0,atol=1e-12)
    np.testing.assert_allclose(q['threshold'],np.r_[t,np.nan],rtol=0,atol=1e-12,equal_nan=True)
    for _,row in priorities.loc[priorities.candidate_id.eq(candidate)].iterrows():
        k=int(np.ceil(row['capacity_fraction']*len(block)));captured=int(block.head(k)['y_true'].sum())
        assert row['reviewed_flights']==k and row['captured_delays']==captured and row['false_alerts']==k-captured
    for _,row in subgroups.loc[subgroups.candidate_id.eq(candidate)].iterrows():
        b=block.loc[block[row['group_dimension']].astype(str).eq(str(row['group_value']))]
        tn,fp,fn,tp=confusion_matrix(b.y_true,b.y_pred,labels=[0,1]).ravel()
        assert [row.tp,row.fp,row.fn,row.tn]==[tp,fp,fn,tn] and row.n_flights==len(b)
        for rate,supported in [('precision','precision_supported'),('recall','recall_supported'),('false_positive_rate','false_positive_rate_supported')]:
            if not row[supported]:assert pd.isna(row[rate])
for _,row in tables['feature_comparison.csv'].iterrows():
    b=mm.loc[mm.model_version.eq(row.model_version)&mm.model_id.eq(row.model_id)&mm.threshold_policy.eq('F1')].set_index('feature_set')
    np.testing.assert_allclose(row.delta_average_precision,b.loc['B','average_precision']-b.loc['A','average_precision'],rtol=0,atol=1e-12)
    np.testing.assert_allclose(row.delta_f1,b.loc['B','f1']-b.loc['A','f1'],rtol=0,atol=1e-12)
from fresh_test_data import build_q2_test
from sklearn.svm import SVC
quality=json.loads((r/'weather_quality.json').read_text())
test_again,_=build_q2_test(root,quality['excluded_station_times'])
assert set(test_again.flight_id)==set(truth.flight_id)
missing=test_again.loc[test_again.weather_report_missing.eq(1)]
spots=pd.concat([missing,test_again.sample(20,random_state=1729)]).drop_duplicates('flight_id').head(20)
base_columns=json.loads((root/'reports/preparation_manifest.json').read_text())['feature_engineering']['feature_sets']
for version,loaded in [('original',joblib.load(root/'models/flight_delay_models.joblib')),('improved',bundle)]:
    for key,item in loaded['models'].items():
        model,fs=key.split('__');columns=base_columns[fs] if version=='original' else item['columns']
        pipeline=item['pipeline']
        if isinstance(pipeline.named_steps['classifier'],SVC):scores=pipeline.decision_function(spots[columns])
        else:scores=pipeline.predict_proba(spots[columns])[:,list(pipeline.named_steps['classifier'].classes_).index(1)]
        exported=p.loc[p.candidate_id.eq(version+'__'+key)].set_index('flight_id').loc[spots.flight_id]
        np.testing.assert_allclose(scores,exported.risk_score,rtol=0,atol=1e-12)

audit={'status':'passed','protected_baseline_files':len(protected),'notebooks':notebooks,
       'baseline_integrity':baseline_integrity,
       'selection_frozen_before_test':True,'all_candidate_ids_and_labels_match':True,
       'saved_bundles_reproduce_exported_scores':True,'reloaded_model_spot_checks':16*len(spots),
       'all_metrics_curves_confusions_and_quotas_recomputed':True,
       'table_rows':{name:len(frame) for name,frame in tables.items()}}
(r/'execution_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2))
