"""Independently reconcile frozen July exports and reload model predictions; no fitting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import nbformat
from final_features import build_july_test
from final_modelling import sha,save,fingerprint,metrics,scores
from baseline_integrity import verify_baseline_files


def audit(root):
    out=root/'reports/final_models'
    selection=json.loads((out/'selection.json').read_text())
    access=json.loads((out/'test_access.json').read_text())
    manifest=json.loads((root/'tableau/exports/final_july/export_manifest.json').read_text())
    assert pd.Timestamp(selection['bundle_frozen_at_utc'])<pd.Timestamp(access['first_outcome_access_at_utc'])
    assert sha(out/'selection.json')==access['protocol']['selection_sha256']==manifest['selection_sha256']
    assert sha(root/selection['bundle_path'])==selection['bundle_sha256']
    export=root/'tableau/exports/final_july'
    tables={name:pd.read_csv(export/name,float_precision='round_trip') for name in manifest['files']}
    for name,meta in manifest['files'].items():
        assert sha(export/name)==meta['sha256'] and len(tables[name])==meta['rows']
    mm=tables['model_metrics.csv'];pp=tables['flight_predictions.csv']
    checked=0
    for _,r in mm.iterrows():
        p=pp.loc[pp.candidate_id.eq(r.candidate_id)]
        assert len(p)==10000 and p.flight_id.nunique()==10000
        assert fingerprint(p.flight_id)==manifest['cohort_fingerprint']
        measured=metrics(p.y_true.to_numpy(),p.risk_score.to_numpy(),r.decision_threshold)
        for k in ['f1','precision','recall','average_precision','accuracy','roc_auc','tp','fp','fn','tn']:
            # Undefined precision for an always-on-time baseline is intentionally null.
            if k=='precision' and pd.isna(r[k]): assert measured['warnings']==0
            else: np.testing.assert_allclose(measured[k],r[k],rtol=0,atol=1e-10)
        assert np.array_equal(p.y_pred.to_numpy(),(p.risk_score>=r.decision_threshold).astype(int))
        assert sorted(p.risk_rank)==list(range(1,10001))
        checked+=1
    for _,r in tables['priority_curve.csv'].iterrows():
        p=pp.loc[pp.candidate_id.eq(r.candidate_id)].sort_values('risk_rank').head(int(r.reviewed_flights))
        assert len(p)==int(np.ceil(r.capacity_fraction*10000))
        assert int(p.y_true.sum())==r.captured_delays and len(p)-int(p.y_true.sum())==r.false_alerts
    chosen=mm.loc[mm.is_validation_selected]
    assert len(chosen)==1 and chosen.iloc[0].model_id==selection['selected_model_id'] and chosen.iloc[0].feature_set=='B'
    # Rebuild features and reload every fitted pipeline on a small common spot cohort.
    quality=json.loads((root/'reports/improvements/weather_quality.json').read_text())
    test,_=build_july_test(root,quality['excluded_station_times'])
    test=test.sort_values('flight_id').reset_index(drop=True)
    assert fingerprint(test.flight_id)==manifest['cohort_fingerprint']
    spot=test.iloc[np.linspace(0,9999,30,dtype=int)]
    bundle=joblib.load(root/selection['bundle_path'])
    spots=0
    for key,item in bundle['models'].items():
        m,fs=key.split('__')
        expected=pp.loc[pp.model_id.eq(m)&pp.feature_set.eq(fs)].set_index('flight_id')
        actual=scores(item['pipeline'],spot[item['columns']])
        np.testing.assert_allclose(actual,expected.loc[spot.flight_id,'risk_score'],rtol=0,atol=1e-12)
        assert np.array_equal(spot.delayed_departure.to_numpy(dtype=int),expected.loc[spot.flight_id,'y_true'])
        spots+=len(spot)
    notebooks={}
    for stem in ['06_final_model_selection','07_final_july_evaluation']:
        nb=nbformat.read(root/f'notebooks/{stem}.ipynb',as_version=4)
        nbformat.validate(nb)
        code=[c for c in nb.cells if c.cell_type=='code']
        assert all(c.execution_count is not None for c in code)
        assert not [o for c in code for o in c.outputs if o.output_type=='error']
        assert not [o for c in code for o in c.outputs if o.output_type=='stream' and o.name=='stderr']
        figures=sum('image/png' in o.get('data',{}) for c in code for o in c.outputs)
        notebooks[stem]=dict(code_cells=len(code),inline_figures=figures,errors=0,stderr=0)
    integrity=verify_baseline_files(root,json.loads((root/'reports/improvements/baseline_fingerprints.json').read_text()))
    return dict(status='passed',candidate_confusion_metric_reconciliations=checked,
        review_quota_reconciliations=len(tables['priority_curve.csv']),
        reloaded_pipeline_prediction_spots=spots,notebooks=notebooks,
        selection_frozen_before_july=True,no_test_driven_changes=True,baseline_integrity=integrity)


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    result=audit(root)
    save(root/'reports/final_models/execution_audit.json',result)
    print(json.dumps(result,indent=2))
