"""Build the local Tableau deliverable from the six frozen July exports.
Requires lxml, pandas and Tableau's official tableauhyperapi. No model fitting.
"""
from pathlib import Path
import copy, hashlib, json, sys, uuid, zipfile
import pandas as pd
from lxml import etree as E
try:
    from tableauhyperapi import (HyperProcess, Telemetry, Connection, CreateMode,
                                TableDefinition, TableName, SqlType, Inserter)
except ImportError:
    sys.path.insert(0, '/private/tmp/flight-tableau-tools')
    from tableauhyperapi import (HyperProcess, Telemetry, Connection, CreateMode,
                                TableDefinition, TableName, SqlType, Inserter)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'tableau/exports/final_july'
OUT = ROOT / 'tableau/workbooks'
OUT.mkdir(exist_ok=True)
(OUT / 'Data').mkdir(exist_ok=True)
MODELS = {'logistic_regression':'Logistic Regression ★', 'knn':'KNN', 'svm':'SVM',
          'random_forest':'Random Forest', 'always_on_time':'Always on time',
          'always_delayed':'Always delayed', 'fair_coin_seed42':'Fair coin (seed 42)',
          'historical_risk_ranking':'Historical risk'}
raw = {p.stem: pd.read_csv(p) for p in SOURCE.glob('*.csv')}
for frame in raw.values():
    if 'model_id' in frame:
        frame['Model'] = frame.model_id.map(MODELS).fillna(frame.model_id)
metrics = raw['model_metrics']
print('Baseline IDs:', metrics.loc[~metrics.feature_set.isin(['A','B','reference66']), 'model_id'].tolist())
# Labels are presentation fields. The six source files are never modified.
main = metrics.loc[metrics.feature_set.eq('B')].copy()
assert len(main) == 4
for f in raw.values():
    if 'model_id' in f:
        f['Model'] = f.model_id.map(MODELS).fillna(f.model_id)

summary = []
for _, r in main.iterrows():
    summary.append({'Model':r.Model, 'Summary':
        f'{r.Model.replace(" ★", "")}  ·  package B ({int(r.n_predictors)} predictors)\n'
        f'F1 {r.f1:.3f}     AP {r.average_precision:.3f}     Precision {r.precision:.1%}     Recall {r.recall:.1%}\n'
        f'{int(r.tp):,} delays found  ·  {int(r.fp):,} false warnings  ·  {int(r.fn):,} missed delays\n'
        f'Fit {r.fit_seconds:.2f}s  ·  Predict {r.prediction_seconds:.3f}s / 10,000 flights'})
raw['summary'] = pd.DataFrame(summary)
# One row per confusion outcome / candidate, rather than joining totals to flights.
confusion = main.melt(id_vars=['Model'], value_vars=['tp','fp','fn','tn'],var_name='Outcome',value_name='Flights')
confusion['Outcome'] = confusion.Outcome.map({'tp':'TP · Delay found','fp':'FP · False warning','fn':'FN · Missed delay','tn':'TN · Correct on-time'})
raw['confusion'] = confusion
# Paired deltas and intervals are retained at model × contrast × metric grain.
gains=[]
for _,r in main.iterrows():
    for contrast,prefix in [('A → B','vsA'),('Retuned 66 → B','vs66')]:
        for metric,key in [('Average precision','ap'),('F1','f1')]:
            val,lo,hi = [float(r[f'{prefix}_{key}_{k}']) for k in ['gain','low','high']]
            gains.append({'Model':r.Model,'Contrast':contrast,'Metric':metric,'Gain':val,
                'Low':lo,'High':hi,'Interval':f'{val:+.4f}  |  95% paired interval [{lo:+.4f}, {hi:+.4f}]',
                'Evidence':'Interval crosses zero' if lo <= 0 <= hi else ('Positive interval' if lo>0 else 'Negative interval')})
raw['gains'] = pd.DataFrame(gains)
# Null subgroup rates remain null. A label explains why a bar is absent.
groups=[]
for _,r in raw['subgroup_metrics'].query('feature_set == "B"').iterrows():
    for label,field in [('Precision','precision'),('Recall','recall'),('Miss rate','false_negative_rate'),('False-positive rate','false_positive_rate')]:
        value = r[field]
        groups.append({'Model':r.Model,'Grouping':r.group_dimension,'Group':r.group_value,'Metric':label,'Rate':value,
            'Group label':f'{r.group_value} · n={int(r.n_flights):,}' + (' · low support' if pd.isna(value) else ''),
            'Flights':int(r.n_flights),'TP':int(r.tp),'FP':int(r.fp),'FN':int(r.fn),'TN':int(r.tn),
            'Support':'Insufficient support; rate withheld' if pd.isna(value) else 'Reporting support met'})
raw['group_rates'] = pd.DataFrame(groups)
priority=raw['priority_curve'].copy()
priority=priority[priority.feature_set.eq('B') | priority.model_id.eq('historical_risk_ranking')].copy()
for fraction in sorted(priority.capacity_fraction.unique()):
    r=priority.iloc[0].copy();r['Model']='Random expectation';r['model_id']='random_expectation';r['feature_set']='fixed';r['capacity_fraction']=fraction
    r['reviewed_flights']=int(round(fraction*10000));r['captured_delays']=fraction*3215;r['false_alerts']=fraction*6785
    r['capture_recall']=fraction;r['review_precision']=.3215;r['lift_over_prevalence']=1.
    priority=pd.concat([priority,pd.DataFrame([r])],ignore_index=True)
raw['review_comparison']=priority
quota=[]
for _,r in priority[priority.feature_set.eq('B')].iterrows():
    quota.append({'Model':r.Model,'Capacity':int(round(r.capacity_fraction*100)), 'Summary':
        f'Top {r.capacity_fraction:.0%}: {int(r.reviewed_flights):,} flights to review\n'
        f'{int(r.captured_delays):,} delays captured  ·  {int(r.false_alerts):,} false alerts\n'
        f'Precision {r.review_precision:.1%}  ·  Capture {r.capture_recall:.1%}  ·  Lift {r.lift_over_prevalence:.2f}×'})
raw['quota_summary']=pd.DataFrame(quota)
f=raw['flight_predictions']
f['Rank label']=f.risk_rank.map(lambda x:f'{int(x):05d}')
f['Flight detail'] = (f.flight_date.astype(str)+'  '+f.origin+' → '+f.destination+'  '+f.reporting_airline+
    '  |  actual delay: '+f.y_true.map({1:'yes',0:'no'})+'  |  '+f.confusion_label+'  |  score '+f.risk_score.map(lambda x:f'{x:.4f}'))
# Add calculated fields only to the Tableau data source, never to raw data.
PARAMS={'Model':('string','Logistic Regression ★',list(MODELS.values())[:4]),
        'Error metric':('string','False-positive rate',['Precision','Recall','Miss rate','False-positive rate']),
        'Review quota (%)':('integer',10,[5,10,20]),
        'List airport':('string','All',['All','ATL','DFW','ORD'])}
CALCS = {
 'summary': {'Keep':('[Model] = [Parameters].[Model]','boolean')},
 'confusion': {'Keep':('[Model] = [Parameters].[Model]','boolean')},
 'model_metrics': {'Compare':('[feature_set] = "B" OR NOT ([feature_set] = "A" OR [feature_set] = "reference66")','boolean')},
 'pr_curve': {'Keep':('[feature_set] = "B" AND [Model] = [Parameters].[Model]','boolean')},
 'group_rates': {'Keep':('[Model] = [Parameters].[Model] AND [Metric] = [Parameters].[Error metric]','boolean')},
 'review_comparison': {'Keep quota':('[capacity_fraction] = [Parameters].[Review quota (%)] / 100.0','boolean'),
   'Keep curve':('[Model] = [Parameters].[Model] OR [Model] = "Historical risk" OR [Model] = "Random expectation"','boolean')},
 'quota_summary': {'Keep':('[Model] = [Parameters].[Model] AND [Capacity] = [Parameters].[Review quota (%)]','boolean')},
 'flight_predictions': {'Keep review':('[feature_set] = "B" AND [Model] = [Parameters].[Model] AND [risk_rank] <= CEILING([Parameters].[Review quota (%)] / 100.0 * [rank_cohort_size]) AND ([Parameters].[List airport] = "All" OR [origin] = [Parameters].[List airport])','boolean'),
   'Keep errors':('[feature_set] = "B" AND [Model] = [Parameters].[Model] AND ([confusion_label] = "FP" OR [confusion_label] = "FN") AND ([Parameters].[List airport] = "All" OR [origin] = [Parameters].[List airport])','boolean')},
}

def sub(parent,tag,text=None,**attrs):
    e=E.SubElement(parent,tag,{k.replace('_','-'):str(v) for k,v in attrs.items()})
    if text is not None:e.text=str(text)
    return e

def dtype(series):
    if pd.api.types.is_bool_dtype(series):return 'boolean',SqlType.bool()
    if pd.api.types.is_integer_dtype(series):return 'integer',SqlType.big_int()
    if pd.api.types.is_numeric_dtype(series):return 'real',SqlType.double()
    return 'string',SqlType.text()

# Compact extracts replace duplicated CSV shipping; keep complete input tables.
schema={name:{c:dtype(df[c])[0] for c in df} for name,df in raw.items()}
if '--xml-only' not in sys.argv:
 with HyperProcess(Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU, parameters={'log_dir':str(OUT/'Data')}) as hp:
     for name,df in raw.items():
         types={c:dtype(df[c]) for c in df}
         schema[name]={c:t[0] for c,t in types.items()}
         path=OUT/f'Data/{name}.hyper'
         with Connection(hp.endpoint,str(path),CreateMode.CREATE_AND_REPLACE) as c:
             c.catalog.create_schema('Extract')
             table=TableDefinition(TableName('Extract','Extract'),[TableDefinition.Column(col,t[1]) for col,t in types.items()])
             c.catalog.create_table(table)
             rows=df.astype(object).where(pd.notna(df),None)
             with Inserter(c,table) as i:i.add_rows(rows.itertuples(index=False,name=None));i.execute()
             assert c.execute_scalar_query('SELECT COUNT(*) FROM "Extract"."Extract"') == len(df)
         print(name,len(df),flush=True)

r=E.Element('workbook',{'original-version':'18.1','source-platform':'mac','source-build':'2026.2.2 (20262.26.0819.2015)','version':'18.1'},nsmap={'user':'http://www.tableausoftware.com/xml/user'})
prefs=sub(r,'preferences');sub(prefs,'preference',name='ui.encoding.shelf.height',value='24')
dss=sub(r,'datasources');params=sub(dss,'datasource',name='Parameters',hasconnection='false',inline='true',version='18.1')
for name,(dt,default,options) in PARAMS.items():
    val=lambda x:'"'+str(x)+'"' if dt=='string' else str(x)
    col=sub(params,'column',caption=name,datatype=dt,name=f'[{name}]',param_domain_type='list',role='measure',type='nominal' if dt=='string' else 'quantitative',value=val(default))
    sub(col,'calculation',**{'class':'tableau','formula':val(default)})
    members=sub(col,'members')
    for opt in options:sub(members,'member',value=val(opt))
D={}
RATES={'f1','f05','average_precision','precision','recall','accuracy','roc_auc','prevalence','Rate','capture_recall','review_precision','capacity_fraction','Gain','Low','High'}
for name,df in raw.items():
    ds=sub(dss,'datasource',name=name,caption=name.replace('_',' ').title(),inline='true',version='18.1');D[name]=ds
    c=sub(ds,'connection',**{'class':'hyper','dbname':f'Data/{name}.hyper','schema':'Extract','tablename':'Extract','access_mode':'readonly','default-settings':'hyper'})
    sub(c,'relation',name='Extract',table='[Extract].[Extract]',type='table')
    metadata=sub(c,'metadata-records')
    for ordinal,(col,dt) in enumerate(schema[name].items()):
        rec=sub(metadata,'metadata-record',**{'class':'column'})
        for tag,txt in [('remote-name',col),('remote-type',{'string':129,'integer':20,'real':5,'boolean':11}[dt]),('local-name',f'[{col}]'),('parent-name','[Extract]'),('remote-alias',col),('ordinal',ordinal),('local-type',dt),('aggregation','Count' if dt in ['string','boolean'] else 'Sum'),('contains-null','true')]:sub(rec,tag,txt)
        attrs={'caption':col.replace('_',' ').title() if col not in ['f1','Model'] else ('F1' if col=='f1' else 'Model'), 'datatype':dt,'name':f'[{col}]','role':'dimension' if dt in ['string','boolean'] else 'measure','type':'nominal' if dt in ['string','boolean'] else 'quantitative'}
        if col in RATES:attrs['default_format']='p0.0%'
        if col=='f1' or col=='average_precision':attrs['default_format']='n0.000'
        if dt=='integer':attrs['default_format']='n#,##0'
        sub(ds,'column',**attrs)
    for col,(formula,dt) in CALCS.get(name,{}).items():
        e=sub(ds,'column',name=f'[{col}]',datatype=dt,role='dimension',type='nominal')
        sub(e,'calculation',**{'class':'tableau','formula':formula})
    sub(ds,'layout',dim_ordering='alphabetic',measure_ordering='alphabetic',show_structure='true',dim_percentage='0.5',measure_percentage='0.5')

worksheets=sub(r,'worksheets');dashboards=sub(r,'dashboards');windows=sub(r,'windows',source_height='30');S={}
COLORS={'TP · Delay found':'#247BA0','FP · False warning':'#E08D3C','FN · Missed delay':'#CC4C4C','TN · Correct on-time':'#A6ADB4',
 'Logistic Regression ★':'#247BA0','KNN':'#6B5DA8','SVM':'#5C946E','Random Forest':'#DB963F','Historical risk':'#8B8E94','Random expectation':'#BDC3CA'}

def sheet(name,source,rows=None,x=None,y=None,text=None,color=None,path=None,filters=None,mark='Bar',tooltip=None):
    ws=sub(worksheets,'worksheet',name=name);S[name]=source
    layout=sub(ws,'layout-options');title=sub(layout,'title');ft=sub(title,'formatted-text');sub(ft,'run',name,fontname='Arial',fontsize='12',bold='true',fontcolor='#20344D')
    t=sub(ws,'table');v=sub(t,'view');sub(sub(v,'datasources'),'datasource',name=source,caption=D[source].get('caption'));dep=sub(v,'datasource-dependencies',datasource=source)
    cols={c.get('name'):c for c in D[source].findall('column')}
    instances={}
    def field(col,quant=False,ordinal=False):
        der='Avg' if quant else 'None';suffix='qk' if quant else ('ok' if ordinal else 'nk');iname=f'[{der.lower()}:{col}:{suffix}]'
        if iname not in instances:
            if f'[{col}]' not in [c.get('name') for c in dep.findall('column')]:dep.append(copy.deepcopy(cols[f'[{col}]']))
            sub(dep,'column-instance',column=f'[{col}]',derivation=der,name=iname,pivot='key',type='quantitative' if quant else ('ordinal' if ordinal else 'nominal'))
            instances[iname]=True
        return f'[{source}].{iname}'
    refs={}
    for col in rows or []:refs[col]=field(col)
    if x:refs[x]=field(x,True)
    if y:refs[y]=field(y,True)
    if text:refs[text]=field(text,False if schema[source].get(text)=='string' else True)
    if color:refs[color]=field(color)
    if path:refs[path]=field(path,ordinal=True)
    pdp=sub(v,'datasource-dependencies',datasource='Parameters')
    for pcol in params.findall('column'):pdp.append(copy.deepcopy(pcol))
    for col,value in (filters or {}).items():
        ref=field(col).split('].',1)[1];filt=sub(v,'filter',**{'class':'categorical','column':ref})
        member='true' if value is True else ('false' if value is False else '"'+str(value)+'"')
        sub(filt,'groupfilter',function='member',level=ref,member=member)
    sub(v,'aggregation',value='true')
    style=sub(t,'style')
    for element,fmt in [('worksheet',{'font-family':'Arial','font-size':'10','color':'#3D4A5C'}),('pane',{'background-color':'#FFFFFF'}),('label',{'font-size':'9','color':'#3D4A5C'}),('gridline',{'line-visibility':'off'})]:
        rule=sub(style,'style-rule',element=element)
        for attr,val in fmt.items():sub(rule,'format',attr=attr,value=val)
    # Exact values are visible on bar labels; raw values remain in tooltips.
    pane=sub(sub(t,'panes'),'pane',selection_relaxation_option='selection-relaxation-allow');sub(sub(pane,'view'),'breakdown',value='auto');sub(pane,'mark',**{'class':mark})
    enc=sub(pane,'encodings')
    for tag,col in [('text',text),('color',color),('path',path)]:
        if col:sub(enc,tag,column=refs[col])
    if text and mark=='Bar':
        ms=pane.find('style')
        if ms is None:ms=sub(pane,'style')
        rule=sub(ms,'style-rule',element='mark');sub(rule,'format',attr='mark-labels-show',value='true');sub(rule,'format',attr='mark-labels-cull',value='true')
    if tooltip:
        tt=sub(pane,'customized-tooltip');ft=sub(tt,'formatted-text');sub(ft,'run',tooltip,fontname='Arial',fontsize='10')
    for ps in pane.findall('style'):pane.remove(ps);pane.append(ps)
    sub(t,'rows',' / '.join(refs[c] for c in rows) if rows else (refs[y] if y else ''))
    sub(t,'cols',refs[x] if x else '')
    return ws

sheet('F1 · four models and baselines','model_metrics',rows=['Model'],x='f1',text='f1',filters={'Compare':True})
sheet('AP · ranking quality','model_metrics',rows=['Model'],x='average_precision',text='average_precision',filters={'Compare':True})
sheet('Confusion counts · package B','confusion',rows=['Outcome'],x='Flights',text='Flights',color='Outcome',filters={'Keep':True})
sheet('Precision–recall · selected display model','pr_curve',x='recall',y='precision',color='Model',path='point_index',filters={'Keep':True},mark='Line')
sheet('Selected display model · July results','summary',text='Summary',filters={'Keep':True},mark='Text')
for contrast,prefix in [('A → B','A to B'),('Retuned 66 → B','66 to B')]:
    for metric in ['Average precision','F1']:
        sheet(f'{prefix} · {metric} gain','gains',rows=['Model'],x='Gain',text='Gain',filters={'Contrast':contrast,'Metric':metric},tooltip='Paired gain on the same 10,000 July flights. Hover / View Data for the full interval; paired 95% intervals are listed below.')
# CI details are a real data table, not an image; precision is preserved.
raw_ci=raw['gains']
# Use intervals as row headers: eight rows per contrast; scrollable, no rate summation.
sheet('Paired 95% intervals · A to B','gains',rows=['Model','Metric'],text='Interval',filters={'Contrast':'A → B'},mark='Text')
sheet('Paired 95% intervals · retuned 66 to B','gains',rows=['Model','Metric'],text='Interval',filters={'Contrast':'Retuned 66 → B'},mark='Text')
for grouping,title in [('origin','Airport'),('departure_time_band','Departure time'),('reporting_airline','Airline')]:
    sheet(f'{title} · error rate','group_rates',rows=['Group label'],x='Rate',text='Rate',filters={'Keep':True,'Grouping':grouping})
sheet('Sampled errors · flight detail','flight_predictions',rows=['Rank label'],text='Flight detail',filters={'Keep errors':True},mark='Text')
sheet('Review precision · fixed quota','review_comparison',rows=['Model'],x='review_precision',text='review_precision',filters={'Keep quota':True})
sheet('Delays captured · fixed quota','review_comparison',rows=['Model'],x='capture_recall',text='capture_recall',filters={'Keep quota':True})
sheet('Capture curve · retrospective ranking','review_comparison',x='capacity_fraction',y='capture_recall',color='Model',path='capacity_fraction',filters={'Keep curve':True},mark='Line')
sheet('Review budget · selected display model','quota_summary',text='Summary',filters={'Keep':True},mark='Text')
sheet('Ranked review list · fixed global cohort','flight_predictions',rows=['Rank label'],text='Flight detail',filters={'Keep review':True},mark='Text')

NOTE='Historical simulation; two-hour cutoff with assumed weather/outcome availability. Live performance not verified.'
COMMON='July 2026 · ATL / DFW / ORD · 10,000 sampled flights · 3,215 delays (32.15%) · departure ≥15 min late · final / B'
DASH_NAMES=['01 Model performance','02 Feature value','03 Prediction reliability','04 Risk prioritization']

def dashboard(index,title,subtitle):
    d=sub(dashboards,'dashboard',name=DASH_NAMES[index-1]);sub(d,'style');sub(d,'size',maxheight='780',maxwidth='1180',minheight='780',minwidth='1180')
    zones=sub(d,'zones');counter=[0]
    def zone(x,y,w,h,**a):
        counter[0]+=1;return sub(zones,'zone',id=counter[0],x=round(x/1180*100000),y=round(y/780*100000),w=round(w/1180*100000),h=round(h/780*100000),**a)
    def txt(x,y,w,h,text,size=10,bold=False,color='#3D4A5C',bg=None):
        z=zone(x,y,w,h,type_v2='text');ft=sub(z,'formatted-text');sub(ft,'run',text,fontname='Arial',fontsize=size,bold=str(bold).lower(),fontcolor=color)
        st=sub(z,'zone-style')
        for attr,val in [('margin','6'),('border-style','none')]+([('background-color',bg)] if bg else []):sub(st,'format',attr=attr,value=val)
        return z
    def view(name,x,y,w,h):
        z=zone(x,y,w,h,name=name);st=sub(z,'zone-style')
        for attr,val in [('border-style','solid'),('border-width','1'),('border-color','#E1E6ED'),('margin','6')]:sub(st,'format',attr=attr,value=val)
        return z
    def control(param,x,y,w):
        z=zone(x,y,w,48,type_v2='paramctrl',param=f'[Parameters].[{param}]',mode='compact',show_title='true')
        return z
    txt(12,8,1156,40,f'O{index}  /  {title}',22,True,'#20344D')
    txt(12,51,1156,29,COMMON,10)
    txt(12,83,1156,32,'★ Validation-selected model: Logistic Regression B. Controls change the display, never the frozen model choice.',10,True,'#247BA0','#EEF6FA')
    txt(12,119,1156,34,subtitle,10)
    txt(12,722,1156,28,NOTE,9,color='#68778A')
    txt(12,750,1156,24,'Use the four numbered dashboard tabs below. Scores are uncalibrated risk scores. Scope: one summer month; three hubs.',9,color='#68778A')
    win=sub(windows,'window',**{'class':'dashboard','name':DASH_NAMES[index-1],'maximized':'true'})
    vps=sub(win,'viewpoints')
    sub(win,'active',id='-1')
    return txt,view,control,vps

txt,view,ctrl,vps=dashboard(1,'MODEL PERFORMANCE','Compare four familiar classifiers with simple baselines. AP measures ranking; F1 uses each frozen validation-set threshold.')
ctrl('Model',15,157,280)
view('Selected display model · July results',315,154,850,88)
view('F1 · four models and baselines',15,250,365,230)
view('AP · ranking quality',390,250,365,230)
view('Confusion counts · package B',765,250,400,230)
view('Precision–recall · selected display model',15,491,735,216)
txt(765,495,400,212,'FINDING\nLR B: 92.4% recall, but 61.9% of its warnings are false.\n\nKNN has July’s highest F1; it does not replace the model selected on 2025 validation.\n\nAP random reference = 0.3215. Always-on-time accuracy is 67.85%, illustrating why accuracy alone is misleading.',11,bg='#F1F5F9')

txt,view,ctrl,vps=dashboard(2,'FEATURE VALUE','Matched A/B settings and identical flight IDs isolate the combined feature package. Zero and negative changes remain visible.')
view('A to B · Average precision gain',15,163,565,168)
view('A to B · F1 gain',595,163,570,168)
view('66 to B · Average precision gain',15,341,565,168)
view('66 to B · F1 gain',595,341,570,168)
view('Paired 95% intervals · A to B',15,518,565,131)
view('Paired 95% intervals · retuned 66 to B',595,518,570,131)
txt(15,654,1150,61,'All four improve AP over schedule-only A. Against retuned 66, gains are mixed: RF AP falls and SVM retains 66. RF’s A→B F1 interval crosses zero.\nCoverage: 100% matched weather reports and nonzero origin-history support; individual weather readings can still be missing. B packages: LR/KNN 78, SVM 66, RF 72 predictors. No separate weather/history/traffic causal claim.',10,bg='#F1F5F9')

txt,view,ctrl,vps=dashboard(3,'PREDICTION RELIABILITY','Rates use different denominators: miss = FN / actual delays; false-positive rate = FP / actual on-time flights.')
ctrl('Model',15,158,285);ctrl('Error metric',320,158,285);ctrl('List airport',895,158,265)
view('Airport · error rate',15,216,360,174)
view('Departure time · error rate',390,216,380,174)
view('Airline · error rate',785,216,380,343)
view('Sampled errors · flight detail',15,403,755,156)
txt(15,568,1150,142,'FINDING · FROZEN LR B\nEvening: 2,799 of 2,804 flights warned; recall 100%, but false-positive rate 99.7%. ORD false-positive rate 75.5%, versus ATL 64.7%. High recall alone hides the alert burden.\n\nBlank bars mean insufficient reporting support: group n≥200 and ≥30 relevant outcomes/warnings. AS, B6, WN and overnight have withheld rates. These cutoffs do not establish statistical certainty.\nThe airport control filters only the flight-detail list. Aggregate error-rate charts always show their full July groups. Hover / View Data for TP, FP, FN, TN.',10,bg='#F1F5F9')

txt,view,ctrl,vps=dashboard(4,'RISK PRIORITIZATION','Retrospective ranking across a fixed 10,000-flight cohort. Quotas alter the review list, not the classification threshold.')
ctrl('Model',15,156,265);ctrl('Review quota (%)',295,156,240);ctrl('List airport',895,156,265)
view('Review budget · selected display model',550,156,330,96)
view('Review precision · fixed quota',15,262,365,195)
view('Delays captured · fixed quota',395,262,365,195)
view('Capture curve · retrospective ranking',775,262,390,195)
view('Ranked review list · fixed global cohort',15,469,1150,157)
txt(15,635,1150,78,'AT 10% CAPACITY · LR B finds 688 delays in 1,000 reviews (68.8% precision; 21.4% capture; 2.14× lift). Historical risk finds 442; random expectation is 321.5.\nThe list airport control only narrows displayed rows; global ranks, quotas and scorecards stay fixed. Exact seed-42 hash tie order comes from Python. Random counts are expectations.\nNo measured costs, interventions, savings or prevented delays. A high score is not a calibrated probability.',10,bg='#F1F5F9')

for d in dashboards:
    win=next(w for w in windows if w.get('name')==d.get('name'))
    vp=win.find('viewpoints')
    for z in d.findall('./zones/zone'):
        if z.get('name'):sub(sub(vp,'viewpoint',name=z.get('name')),'zoom',type='entire-view' if 'list' not in z.get('name').lower() and 'detail' not in z.get('name').lower() and 'interval' not in z.get('name').lower() else 'fit-width')
# Put four dashboards first in the tab order; supporting sheets remain editable.
for ws in worksheets:
    win=sub(windows,'window',**{'class':'worksheet','name':ws.get('name'),'maximized':'true'})
    sub(win,'cards')
    sub(sub(win,'viewpoint'),'zoom',type='entire-view')
sub(r,'preferences') if False else None
workbook=OUT/'departure_delay_findings.twb'
E.ElementTree(r).write(str(workbook),pretty_print=True,xml_declaration=True,encoding='utf-8')
manifest={'source_run':str(metrics.run_id.iloc[0]),'cohort_id':str(metrics.cohort_id.iloc[0]),'test':'July 2026',
    'frozen_selection':'logistic_regression/B','source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SOURCE.glob('*.csv'))},
    'extract_rows':{n:len(df) for n,df in raw.items()},'models_retrained':False,'dashboard_count':4,
    'notes':'Hyper presentation fields are deterministic reshapes/labels. Unsupported subgroup values remain NULL.'}
(OUT/'workbook_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
with zipfile.ZipFile(OUT/'departure_delay_findings.twbx','w',zipfile.ZIP_DEFLATED) as z:
    z.write(workbook,workbook.name)
    for p in sorted((OUT/'Data').glob('*.hyper')):
        if p.stem in raw:z.write(p,'Data/'+p.name)
    z.write(OUT/'workbook_manifest.json','workbook_manifest.json')
    z.write(SOURCE/'export_manifest.json','source_export_manifest.json')
print('Built',OUT/'departure_delay_findings.twbx',flush=True)
