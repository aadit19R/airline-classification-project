"""Small shared utilities for notebooks 06/07; all learned preprocessing stays in pipelines."""
import hashlib
import json
from time import perf_counter
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, confusion_matrix, average_precision_score, roc_auc_score

MODELS = ['logistic_regression','knn','svm','random_forest']
NAMES = dict(zip(MODELS,['Logistic Regression','KNN','RBF SVM','Random Forest']))
FOLDS = [('May–Jun','2025-03-31','2025-04-01','2025-06-30'),
         ('Aug–Sep','2025-06-30','2025-07-01','2025-09-30'),
         ('Nov–Dec','2025-09-30','2025-10-01','2025-12-31')]
GRID = {
 'logistic_regression':[{'C':c} for c in [.03,.01,.1,.3]],
 'knn':[{'n_neighbors':k,'weights':w,'metric':'euclidean'} for k,w in [(101,'distance'),(51,'distance'),(201,'distance'),(101,'uniform')]],
 'svm':[{'C':1.,'gamma':'scale'},{'C':.3,'gamma':'scale'}, {'C':3.,'gamma':'scale'},{'C':1.,'gamma':.01}],
 'random_forest':[
     {'max_depth':16,'min_samples_leaf':5,'max_features':'sqrt','class_weight':'balanced'},
     {'max_depth':12,'min_samples_leaf':10,'max_features':'sqrt','class_weight':None},
     {'max_depth':16,'min_samples_leaf':20,'max_features':'sqrt','class_weight':None},
     {'max_depth':None,'min_samples_leaf':10,'max_features':'sqrt','class_weight':'balanced'}]}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def save(path,value):
    temporary = path.with_suffix(path.suffix+'.part')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def fingerprint(ids):
    return hashlib.sha256('\n'.join(sorted(map(str,ids))).encode()).hexdigest()


def sample(frame,n,seed=42):
    frame = frame.sort_values('flight_id')
    strata = frame['month'].astype(str)+'|'+frame['delayed_departure'].astype(str)
    ix,_ = train_test_split(frame.index,train_size=n,stratify=strata,random_state=seed)
    return frame.loc[ix].sort_values('flight_id')


def make_cohorts(root):
    features = pd.read_parquet(root/'data/processed/features.parquet',columns=[
        'flight_id','flight_date','month','delayed_departure','model_sample','refit_sample'],filters=[
        ('flight_date','>=',pd.Timestamp('2025-01-01')),('flight_date','<',pd.Timestamp('2026-01-01'))])
    cohorts = {}
    for name,end,start,stop in FOLDS:
        population = features.loc[features['flight_date'].le(end)]
        small = population.loc[population['model_sample']] if name=='Nov–Dec' else sample(population,20000)
        extra = sample(population.loc[~population['flight_id'].isin(small['flight_id'])],20000,43)
        validation = features.loc[features['flight_date'].between(start,stop)]
        validation = validation.loc[validation['model_sample']] if name=='Nov–Dec' else sample(validation,5000)
        calibration = validation.loc[validation['flight_date'].dt.month.eq(pd.Timestamp(start).month)]
        scoring = validation.loc[~validation.index.isin(calibration.index)]
        assert len(small)==20000 and len(extra)==20000 and len(validation)==5000
        assert population['flight_date'].max()<calibration['flight_date'].min()<=calibration['flight_date'].max()<scoring['flight_date'].min()
        cohorts[name] = {'train20':small['flight_id'].tolist(),
            'train40':sorted(small['flight_id'].tolist()+extra['flight_id'].tolist()),
            'calibrate':calibration['flight_id'].tolist(),'score':scoring['flight_id'].tolist()}
    first = features.loc[features['refit_sample']]
    extra = sample(features.loc[~features['flight_id'].isin(first['flight_id'])],20000,43)
    final_ids = sorted(first['flight_id'].tolist()+extra['flight_id'].tolist())
    assert len(final_ids)==40000
    needed = set(final_ids)
    for c in cohorts.values():
        for ids in c.values(): needed.update(ids)
    return cohorts,final_ids,needed


def pipeline(model_id,columns,recipe,params,covered):
    categorical = [c for c in columns if c in ['airline','origin','destination','weekday']]
    numeric = [c for c in columns if c not in categorical]
    flags = [c for c in numeric if c.endswith('_no_history') or c in [
        'precip_trace','no_ceiling_reported','cloud_coverage_missing','rain_reported',
        'frozen_precip_reported','fog_or_mist_reported','thunder_reported','condition_codes_missing',
        'ceiling_height_missing','weather_report_missing','freezing_wet',
        'variable_wind_flag','altimeter_missing_flag']]
    trans = []
    if recipe=='original':
        for name,cols,indicator in [('numeric',[c for c in numeric if c not in covered],True),
                                     ('explicit',[c for c in numeric if c in covered],False)]:
            steps = [('impute',SimpleImputer(strategy='median',add_indicator=indicator,keep_empty_features=True))]
            if model_id!='random_forest': steps.append(('scale',StandardScaler()))
            if cols: trans.append((name,Pipeline(steps),cols))
    else:
        continuous = [c for c in numeric if c not in flags]
        steps = [('impute',SimpleImputer(strategy='median',keep_empty_features=True))]
        if model_id!='random_forest': steps.append(('scale',StandardScaler()))
        if continuous: trans.append(('continuous',Pipeline(steps),continuous))
        if flags: trans.append(('binary',SimpleImputer(strategy='constant',fill_value=0,keep_empty_features=True),flags))
        missing = [c for c in numeric if c not in covered]
        if missing: trans.append(('missing',MissingIndicator(features='all',error_on_new=False),missing))
    trans.append(('category',Pipeline([
        ('impute',SimpleImputer(strategy='constant',fill_value='Unknown',missing_values=pd.NA,keep_empty_features=True)),
        ('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=True))]),categorical))
    estimators = {'logistic_regression':LogisticRegression(max_iter=3000,random_state=42),
        'knn':KNeighborsClassifier(algorithm='brute',n_jobs=2),
        'svm':SVC(kernel='rbf',cache_size=512,random_state=42),
        'random_forest':RandomForestClassifier(n_estimators=200,max_features='sqrt',class_weight='balanced',random_state=42,n_jobs=2)}
    return Pipeline([('preprocess',ColumnTransformer(trans,sparse_threshold=1.)),
                     ('classifier',estimators[model_id].set_params(**params))])


def scores(model,frame):
    return model.decision_function(frame) if isinstance(model.named_steps['classifier'],SVC) else model.predict_proba(frame)[:,1]


def threshold(y,s):
    p,r,t = precision_recall_curve(y,s)
    f = np.divide(2*p[:-1]*r[:-1],p[:-1]+r[:-1],out=np.zeros(len(t)),where=(p[:-1]+r[:-1])>0)
    return float(t[np.flatnonzero(np.isclose(f,f.max(),rtol=0,atol=1e-12))[-1]])


def metrics(y,s,t):
    tn,fp,fn,tp = confusion_matrix(y,np.asarray(s)>=t,labels=[0,1]).ravel()
    return dict(f1=float(2*tp/(2*tp+fp+fn)),precision=float(tp/(tp+fp)) if tp+fp else 0.,
        recall=float(tp/(tp+fn)),average_precision=float(average_precision_score(y,s)),
        accuracy=float((tp+tn)/len(y)),roc_auc=float(roc_auc_score(y,s)),
        tp=int(tp),fp=int(fp),fn=int(fn),tn=int(tn),warnings=int(tp+fp),n=len(y))


def fit(model_id,columns,recipe,params,covered,train):
    model = pipeline(model_id,columns,recipe,params,covered)
    started = perf_counter()
    model.fit(train[columns],train['delayed_departure'].astype(int))
    seconds = perf_counter()-started
    classifier = model.named_steps['classifier']
    if isinstance(classifier,SVC): assert classifier.fit_status_==0
    if isinstance(classifier,LogisticRegression): assert classifier.n_iter_.max()<classifier.max_iter
    return model,seconds
