"""Build the six Tableau evaluation tables from frozen predictions."""
import hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score,confusion_matrix,precision_recall_curve


def metrics(y,s,t):
    tn,fp,fn,tp=confusion_matrix(y,np.asarray(s)>=t,labels=[0,1]).ravel()
    return {'n_flights':len(y),'actual_positive':int(tp+fn),'predicted_positive':int(tp+fp),
        'tp':int(tp),'fp':int(fp),'fn':int(fn),'tn':int(tn),'prevalence':float(np.mean(y)),
        'precision':float(tp/(tp+fp)) if tp+fp else np.nan,'recall':float(tp/(tp+fn)),
        'f1':float(2*tp/(2*tp+fp+fn)),'f05':float(1.25*tp/(1.25*tp+.25*fn+fp)),
        'average_precision':float(average_precision_score(y,s)),
        'roc_auc':float(roc_auc_score(y,s)),'accuracy':float((tp+tn)/len(y))}


def make_tables(test,evaluations,config,run_id,selected,cohort_sha,seed=42):
    predictions=[];metric_rows=[];curves=[];priority=[];subgroups=[]
    y=test['delayed_departure'].to_numpy(dtype=int)
    for e in evaluations:
        identity={'run_id':run_id,'model_version':e['version'],'model_id':e['model_id'],
                  'feature_set':e['feature_set'],'split':'test_q2_2026','cohort_id':cohort_sha}
        candidate=e['version']+'__'+e['model_id']+'__'+e['feature_set']
        identity['candidate_id']=candidate
        is_selected=(e['version'],e['model_id'],e['feature_set'])==selected
        metric_rows.append(dict(identity,threshold_policy='F1',**metrics(y,e['scores'],e['threshold']),
            decision_threshold=e['threshold'],is_validation_selected=is_selected,
            fit_seconds=e.get('fit_seconds',0.),prediction_seconds=e.get('prediction_seconds',0.)))
        if 'threshold_f05' in e:
            metric_rows.append(dict(identity,threshold_policy='F0.5',**metrics(y,e['scores'],e['threshold_f05']),
                decision_threshold=e['threshold_f05'],is_validation_selected=is_selected,
                fit_seconds=e.get('fit_seconds',0.),prediction_seconds=e.get('prediction_seconds',0.)))
        p=test[['flight_id','flight_date','scheduled_departure_utc','prediction_cutoff_utc',
                'origin','destination','airline','departure_time_band']].rename(columns={'airline':'reporting_airline'}).copy()
        p['flight_date']=p['flight_date'].dt.strftime('%Y-%m-%d')
        for c in ['scheduled_departure_utc','prediction_cutoff_utc']:p[c]=p[c].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        for c,v in identity.items():p[c]=v
        p['threshold_policy']='F1';p['y_true']=y;p['risk_score']=e['scores']
        p['decision_threshold']=e['threshold'];p['score_type']=e['score_type']
        p['y_pred']=(p['risk_score']>=e['threshold']).astype('int8')
        p['confusion_label']=np.select([p.y_true.eq(1)&p.y_pred.eq(1),p.y_true.eq(0)&p.y_pred.eq(1),
                                       p.y_true.eq(1)&p.y_pred.eq(0)],['TP','FP','FN'],default='TN')
        p['is_validation_selected']=is_selected
        p['ranking_tie_key']=[hashlib.sha256(f'{seed}:{fid}'.encode()).hexdigest() for fid in p['flight_id']]
        p=p.sort_values(['risk_score','ranking_tie_key'],ascending=[False,True],kind='stable').reset_index(drop=True)
        p['risk_rank']=np.arange(1,len(p)+1);p['rank_cohort_id']=cohort_sha;p['rank_cohort_size']=len(p)
        predictions.append(p)
        precision,recall,t=precision_recall_curve(y,e['scores'])
        c=pd.DataFrame({'point_index':np.arange(len(precision)),'threshold':np.r_[t,np.nan],
                        'precision':precision,'recall':recall})
        for field,value in identity.items():c[field]=value
        c['n_flights']=len(y);c['actual_positive']=int(y.sum());curves.append(c)
        for capacity in config['evaluation']['capacity_fractions']:
            k=int(np.ceil(capacity*len(p)));captured=int(p.head(k)['y_true'].sum())
            priority.append(dict(identity,capacity_fraction=capacity,n_flights=len(p),actual_positive=int(y.sum()),
                reviewed_flights=k,realized_review_fraction=k/len(p),captured_delays=captured,false_alerts=k-captured,
                capture_recall=captured/y.sum(),review_precision=captured/k,lift_over_prevalence=(captured/k)/y.mean(),
                random_expected_recall=k/len(p),random_expected_precision=float(y.mean()),cohort_fingerprint=cohort_sha))
        for dimension in config['evaluation']['subgroup_dimensions']:
            for group,block in p.groupby(dimension,observed=True):
                tn,fp,fn,tp=confusion_matrix(block.y_true,block.y_pred,labels=[0,1]).ravel()
                settings=config['evaluation'];supported=len(block)>=settings['minimum_group_rows']
                pr=supported and tp+fp>=settings['minimum_predicted_positives_for_precision']
                re=supported and tp+fn>=settings['minimum_actual_positives_for_recall']
                fr=supported and tn+fp>=settings['minimum_actual_negatives_for_fpr']
                subgroups.append(dict(identity,group_dimension=dimension,group_value=str(group),n_flights=len(block),
                    actual_positive=int(tp+fn),actual_negative=int(tn+fp),predicted_positive=int(tp+fp),
                    tp=int(tp),fp=int(fp),fn=int(fn),tn=int(tn),precision=tp/(tp+fp) if pr else np.nan,
                    recall=tp/(tp+fn) if re else np.nan,false_negative_rate=fn/(tp+fn) if re else np.nan,
                    false_positive_rate=fp/(tn+fp) if fr else np.nan,precision_supported=bool(pr),
                    recall_supported=bool(re),false_positive_rate_supported=bool(fr)))
    mm=pd.DataFrame(metric_rows);comparisons=[]
    for (model_version,model_id),pair in mm.loc[mm.threshold_policy.eq('F1')&mm.feature_set.isin(['A','B'])].groupby(['model_version','model_id']):
        a=pair.set_index('feature_set').loc['A'];b=pair.set_index('feature_set').loc['B']
        comparisons.append({'run_id':run_id,'model_version':model_version,'model_id':model_id,'split':'test_q2_2026',
            'cohort_id':cohort_sha,'base_feature_set':'A','added_feature_set':'B','n_common_flights':len(test),
            'identical_ids_and_labels':True,'cohort_fingerprint':cohort_sha,
            'base_average_precision':a.average_precision,'added_average_precision':b.average_precision,
            'delta_average_precision':b.average_precision-a.average_precision,
            'base_f1':a.f1,'added_f1':b.f1,'delta_f1':b.f1-a.f1,
            'added_feature_coverage':float((test['weather_report_missing'].eq(0)&test['history_origin_n'].gt(0)).mean()),
            'status':'completed','unavailable_reason':None})
    return {'flight_predictions.csv':pd.concat(predictions,ignore_index=True),
            'model_metrics.csv':mm,'pr_curve.csv':pd.concat(curves,ignore_index=True),
            'priority_curve.csv':pd.DataFrame(priority),'subgroup_metrics.csv':pd.DataFrame(subgroups),
            'feature_comparison.csv':pd.DataFrame(comparisons)}
