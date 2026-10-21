"""Verify the final development outputs without loading July outcome values."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import joblib
from final_modelling import sha, fingerprint, threshold, metrics, MODELS
from baseline_integrity import verify_baseline_files


def audit(root):
    out = root/'reports/final_models'
    selection = json.loads((out/'selection.json').read_text())
    assert 'bundle_frozen_at_utc' in selection, 'Final fit is incomplete.'
    protocol = json.loads((out/'protocol.json').read_text())
    assert sha(out/'protocol.json')==selection['protocol_sha256']
    assert all(sha(root/p)==v for p,v in protocol['input_sha256'].items())
    assert sha(out/'development.json')==selection['development_sha256']
    assert sha(out/'cohorts.json')==selection['cohorts_sha256']
    bundle = joblib.load(root/selection['bundle_path'])
    assert sha(root/selection['bundle_path'])==selection['bundle_sha256']
    d = json.loads((out/'development.json').read_text())
    cohorts = json.loads((out/'cohorts.json').read_text())
    f = pd.read_parquet(root/'data/processed/features.parquet',columns=[
        'flight_id','flight_date','delayed_departure'],filters=[
        ('flight_date','>=',pd.Timestamp('2025-01-01')),('flight_date','<',pd.Timestamp('2026-01-01'))]).set_index('flight_id')
    arrays = np.load(out/'development_scores.npz')
    checked = 0
    for r in d['runs']:
        for fold in r['folds']:
            c = cohorts['folds'][fold['fold']]
            train = c['train20' if r['size']==20000 else 'train40']
            assert set(train).isdisjoint(c['calibrate']+c['score'])
            assert set(c['calibrate']).isdisjoint(c['score'])
            assert f.loc[train,'flight_date'].max()<f.loc[c['calibrate'],'flight_date'].min()
            assert f.loc[c['calibrate'],'flight_date'].max()<f.loc[c['score'],'flight_date'].min()
            prefix = r['key']+'_'+fold['fold']
            cal = arrays[prefix+'_calibration'];s=arrays[prefix+'_score']
            assert np.isfinite(cal).all() and np.isfinite(s).all()
            assert threshold(f.loc[c['calibrate'],'delayed_departure'],cal)==fold['threshold']
            measured = metrics(f.loc[c['score'],'delayed_departure'].to_numpy(dtype=int),s,fold['threshold'])
            for field,v in measured.items():
                np.testing.assert_allclose(v,fold[field],rtol=0,atol=1e-12)
            checked += 1
    # Existing anchor controls must reproduce the notebook-02 broad experiment.
    broad = json.loads((root/'reports/broad_feature_experiments.json').read_text())
    broad_rows = broad.get('runs',[])
    reproductions = 0
    for r in d['runs']:
        if r['stage']!='feature_screen' or r['package']=='combined': continue
        old = [x for x in broad_rows if x['model_id']==r['model_id'] and x['package']==r['package']]
        for a in r['folds']:
            match = [x for x in old if x['fold']==a['fold']]
            if match:
                np.testing.assert_allclose([a['f1'],a['average_precision']],
                    [match[0]['f1'],match[0]['average_precision']],rtol=0,atol=1e-12)
                reproductions += 1
    final_ids = cohorts['final_refit_ids']
    assert len(final_ids)==len(set(final_ids))==40000
    assert f.loc[final_ids,'flight_date'].dt.month.nunique()==12
    assert fingerprint(final_ids)==selection['refit_ids_sha256']==bundle['refit_ids_sha256']
    for m in MODELS:
        a,b = [bundle['models'][m+'__'+s] for s in ['A','B']]
        assert a['params']==b['params'] and a['recipe']==b['recipe']
        assert a['pipeline'].named_steps['classifier'].get_params()==b['pipeline'].named_steps['classifier'].get_params()
        for label in ['A','reference66','B']:
            item = bundle['models'][m+'__'+label]
            assert item['pipeline'].n_features_in_==len(item['columns'])
            assert not {'Tail_Number','LateAircraftDelay','departure_delay_minutes'}.intersection(item['columns'])
            assert item['threshold']==selection['policies'][m][label]['threshold']
    integrity = verify_baseline_files(root,json.loads((root/'reports/improvements/baseline_fingerprints.json').read_text()))
    return dict(status='passed',development_fits_recomputed=checked,
        old_anchor_fold_reproductions=reproductions,final_refit_rows=40000,all_12_months=True,
        A_B_settings_matched=True,baseline_integrity=integrity,july_outcomes_loaded_by_audit=False)


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    result=audit(root)
    from final_modelling import save
    save(root/'reports/final_models/development_audit.json',result)
    print(json.dumps(result,indent=2))
