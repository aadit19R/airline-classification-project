"""Create the concise preparation-led report from executed project evidence.
Reads saved handoffs/results only; no notebook execution, refitting or retuning.
"""
from pathlib import Path
import hashlib
import json
import re
import os
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/departure-report-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FuncFormatter
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from xml.sax.saxutils import escape

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/final_report'
FIG=OUT/'figures'
PDF=OUT/'output/pdf/predeparture_flight_delay_report.pdf'
for p in [FIG,PDF.parent,OUT/'tmp/pdfs']:p.mkdir(exist_ok=True,parents=True)
BLUE,ORANGE,GRAY,RED,GREEN='#246b96','#d19347','#8290a1','#b5535a','#478776'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,
 'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,
 'axes.edgecolor':'#a9b3bf','axes.labelcolor':'#354457','xtick.color':'#354457','ytick.color':'#354457',
 'text.color':'#19293a','figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})
prep=json.loads((ROOT/'reports/preparation_manifest.json').read_text())
manifest=json.loads((ROOT/'tableau/exports/final_july/export_manifest.json').read_text())
selection=json.loads((ROOT/'reports/final_models/selection.json').read_text())
for f,meta in manifest['files'].items():
 assert hashlib.sha256((ROOT/'tableau/exports/final_july'/f).read_bytes()).hexdigest()==meta['sha256']
clean_columns=['flight_id','flight_date','origin','airline','destination','departure_time_band','departure_delay_minutes',
 'distance_miles','scheduled_duration_minutes','delayed_departure','departure_hour','month','cancelled','query_eligible']
schedules=pd.read_parquet(ROOT/'data/processed/flights_clean.parquet',columns=clean_columns,filters=[('split','==','train')])
train=schedules.loc[schedules.query_eligible].copy()
feature_columns=['flight_id','gust_knots','ceiling_ft','rain_reported','wind_knots','visibility_miles','temperature_f',
 'precip_inches','precip_trace','weather_age_minutes','weather_report_missing','condition_codes_missing','no_ceiling_reported']
weather=pd.read_parquet(ROOT/'data/processed/features.parquet',columns=feature_columns,filters=[('split','==','train')])
train=train.merge(weather,on='flight_id',how='left',validate='one_to_one')
assert len(train)==698670 and train.flight_id.is_unique
metrics=pd.read_csv(ROOT/'tableau/exports/final_july/model_metrics.csv')
priority=pd.read_csv(ROOT/'tableau/exports/final_july/priority_curve.csv')
model_ids=['logistic_regression','knn','svm','random_forest']
names=['Logistic Regression','KNN','RBF SVM','Random Forest']
b=metrics.loc[metrics.feature_set.eq('B')].set_index('model_id').loc[model_ids]

# Recompute the saved deeper-EDA comparison; keep raw and standardized populations explicit.
pair=train.loc[train.airline.isin(['AA','F9'])].copy()
groups=['origin','month','departure_time_band']
grouped=pair.groupby(groups+['airline'],observed=True).delayed_departure.agg(n='size',risk='mean')
counts=grouped['n'].unstack('airline').reindex(columns=['AA','F9'])
risks=grouped['risk'].unstack('airline').reindex(columns=['AA','F9'])
keep=counts.ge(30).all(axis=1);counts,risks=counts.loc[keep],risks.loc[keep]
weights=counts.sum(axis=1)/counts.sum().sum()
standardized=risks.mul(weights,axis=0).sum()
crude=pair.groupby('airline').delayed_departure.mean().reindex(['AA','F9'])
assert abs(standardized['F9']-.294364)<1e-6
stats={'eda_period':'January to September 2025','eda_scheduled':len(schedules),'eda_eligible':len(train),
 'eda_delayed':int(train.delayed_departure.sum()),'eda_delay_rate':float(train.delayed_departure.mean()),
 'departure_delay':{k:float(v) for k,v in train.departure_delay_minutes.describe().items()},
 'distance_duration_correlation':float(train.distance_miles.corr(train.scheduled_duration_minutes)),
 'airline_crude':crude.to_dict(),'airline_standardized':standardized.to_dict(),
 'airline_standardized_retained':int(counts.sum().sum()),'airline_pair_total':len(pair),
 'weather_measurement_missing_fraction':{k:float(v) for k,v in train[feature_columns[1:]].isna().mean().items()},
 'weather_age_quantiles':{str(k):float(v) for k,v in train.weather_age_minutes.quantile([0,.25,.5,.75,1]).items()},
 'delay_plot_outside_zoom':int((~train.departure_delay_minutes.between(-30,180)).sum()),
 'visibility_at_10_miles':float(train.visibility_miles.eq(10).mean())}
for col in ['origin','month','departure_time_band']:
 stats['rates_'+col]=json.loads(train.groupby(col,observed=True).delayed_departure.agg(n='size',rate='mean').reset_index().to_json(orient='records'))

def save(fig,name):
 fig.savefig(FIG/name,dpi=200,bbox_inches='tight',pad_inches=.12)
 plt.close(fig)

def fmtpct(ax):ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0));ax.grid(axis='y',alpha=.14);ax.set_axisbelow(True)

# Figure 1: actual records used, with context distinct from modeling.
fig,ax=plt.subplots(figsize=(7,1.13));ax.axis('off')
blocks=[('Nov-Dec 2024','History only',GRAY),('Jan-Sep 2025','Initial EDA',BLUE),('All 2025','Development and refit',BLUE),('Jan-Jun 2026','Earlier tests',GRAY),('July 2026','Fresh test',ORANGE)]
for i,(head,body,color) in enumerate(blocks):
 x=i/5;ax.text(x+.012,.79,head,weight='bold',fontsize=8.3,transform=ax.transAxes)
 ax.text(x+.012,.48,body.replace(' and ',' and\n'),fontsize=7.6,transform=ax.transAxes,va='top')
 ax.plot([x+.012,x+.182],[.19,.19],lw=5,color=color,solid_capstyle='round',transform=ax.transAxes)
ax.text(.012,-.03,'Periods overlap by purpose; the final refit covers all twelve 2025 months.',fontsize=7.5,transform=ax.transAxes,color='#5a6878')
save(fig,'01_study_periods.png')

# Figure 2: class balance, central distribution, complete tail, sampled scatter.
fig,axes=plt.subplots(2,2,figsize=(7.05,4.5),layout='constrained')
ax=axes[0,0];counts_y=train.delayed_departure.value_counts().reindex([0,1]);bars=ax.bar(['Under 15 min','15+ min late'],counts_y,color=[BLUE,ORANGE],width=.58)
ax.set(title='Class balance',ylabel='Eligible departures',ylim=(0,610000));ax.yaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x/1000:.0f}k'))
for bar,value in zip(bars,counts_y):ax.text(bar.get_x()+bar.get_width()/2,value+14000,f'{value:,}\n({value/len(train):.1%})',ha='center',fontsize=8)
ax=axes[0,1];delay=train.departure_delay_minutes;central=delay.between(-30,180);ax.hist(delay[central],bins=np.arange(-30,185,5),color=BLUE)
ax.axvline(15,color=RED,ls='--',lw=1);ax.set(title='Delay distribution with a readable zoom',xlabel='Departure delay (minutes)',ylabel='Flights');ax.yaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x/1000:.0f}k'))
ax.text(.97,.92,'Target starts at 15 min',ha='right',transform=ax.transAxes,fontsize=7.5,color=RED)
ax=axes[1,0];ax.boxplot(delay,orientation='horizontal',patch_artist=True,boxprops={'facecolor':'#b6d1e2'},flierprops={'markersize':1.5,'alpha':.2,'markeredgecolor':BLUE},medianprops={'color':RED});ax.set_xscale('symlog',linthresh=15);ax.set(title='Full delay range including valid extremes',xlabel='Minutes (symmetric log scale)',yticks=[]);ax.text(.02,.92,'Median -1 min; mean 17.1 min',transform=ax.transAxes,fontsize=7.7)
ax=axes[1,1];sample=train.sample(n=10000,random_state=42)
for label,color,name in [(0,BLUE,'Under 15 min'),(1,ORANGE,'15+ min late')]:
 d=sample[sample.delayed_departure.eq(label)];ax.scatter(d.distance_miles,d.scheduled_duration_minutes,s=3,alpha=.2,color=color,label=name,rasterized=True)
ax.set(title='Distance and scheduled duration move together',xlabel='Distance (miles)',ylabel='Scheduled duration (minutes)');ax.legend(fontsize=6.8,frameon=False,loc='upper left',markerscale=2)
save(fig,'02_univariate_and_scatter.png')

# Figure 3: associations use complete initial-training populations; adjustment as in 02a.
fig,axes=plt.subplots(2,2,figsize=(7.05,3.8),layout='constrained')
month=train.groupby('month',observed=True).delayed_departure.mean();axes[0,0].plot(month.index,month.values,'o-',color=BLUE,ms=3);axes[0,0].set(title='Summer months have higher observed risk',xticks=list(range(1,10)),xticklabels=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep'],ylabel='Delay rate',ylim=(0,.42));fmtpct(axes[0,0])
bands=train.groupby('departure_time_band',observed=True).delayed_departure.mean();bars=axes[0,1].bar(['00-05','06-11','12-17','18-23'],bands,color=BLUE);axes[0,1].set(title='Later scheduled departures have higher risk',xlabel='Local scheduled departure hour',ylabel='Delay rate',ylim=(0,.44));fmtpct(axes[0,1])
for bar,value in zip(bars,bands):axes[0,1].text(bar.get_x()+bar.get_width()/2,value+.012,f'{value:.1%}',ha='center',fontsize=8)
precip=train.assign(precip_state=np.select([train.precip_trace.eq(1),train.precip_inches.gt(0)],['Trace','Positive'],default='Zero'))
p=precip.groupby('precip_state',observed=True).delayed_departure.agg(n='size',rate='mean').reindex(['Zero','Trace','Positive'])
bars=axes[1,0].bar(p.index,p.rate,color=[BLUE,'#7fa9c3',ORANGE]);axes[1,0].set(title='Reported accumulation and delay are associated',ylabel='Delay rate',ylim=(0,.57));fmtpct(axes[1,0])
for bar,(_,r) in zip(bars,p.iterrows()):axes[1,0].text(bar.get_x()+bar.get_width()/2,r.rate+.012,f'{r.rate:.1%}\nn={int(r.n):,}',ha='center',fontsize=7.4)
ax=axes[1,1];x=np.arange(2);ax.bar(x-.17,crude.values,.32,label='Crude rate',color=GRAY);ax.bar(x+.17,standardized.values,.32,label='Common flight mix',color=BLUE);ax.set(title='Airline ranking changes after adjustment',xticks=x,xticklabels=['AA','F9'],ylabel='Delay rate',ylim=(0,.44));fmtpct(ax);ax.legend(fontsize=7,frameon=False,loc='upper center',ncol=2)
for j,(a,c) in enumerate(zip(crude,standardized)):
 ax.text(j-.17,a+.009,f'{a:.1%}',ha='center',fontsize=7);ax.text(j+.17,c+.009,f'{c:.1%}',ha='center',fontsize=7)
save(fig,'03_bivariate_findings.png')

# Figure 4: flight-weighted missingness, rather than conflating it with report coverage.
fig,axes=plt.subplots(1,2,figsize=(7.05,2.35),layout='constrained',gridspec_kw={'width_ratios':[1.15,1]})
cols=['rain_reported','gust_knots','ceiling_ft','wind_knots','weather_report_missing'];labels=['Condition flag','Gust speed','Ceiling height','Wind speed','Unmatched report']
missing=[train[c].isna().mean() if c!='weather_report_missing' else train[c].mean() for c in cols]
ax=axes[0];ax.barh(labels,missing,color=[ORANGE,ORANGE,BLUE,BLUE,GRAY]);ax.invert_yaxis();ax.set(xlim=(0,1.18),xlabel='Fraction of training flights',title='A matched report can have unknown fields');ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0))
for i,v in enumerate(missing):ax.text(v+.015,i,f'{v:.2%}' if 0<v<.001 else f'{v:.1%}',va='center',fontsize=8)
ax=axes[1];ax.hist(train.weather_age_minutes,bins=np.arange(20,185,5),color=BLUE);ax.axvline(180,color=RED,ls='--',lw=1);ax.set(title='Age of the matched report at cutoff',xlabel='Minutes before the prediction cutoff',ylabel='Flights',xlim=(15,185));ax.yaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x/1000:.0f}k'));ax.text(.97,.91,'180 min age limit',transform=ax.transAxes,ha='right',fontsize=7.5,color=RED)
save(fig,'04_weather_quality.png')

# Figure 5: schematic and clock representation are teaching diagrams, not measured effects.
fig,axes=plt.subplots(1,2,figsize=(7.05,2.1),layout='constrained',gridspec_kw={'width_ratios':[1.5,1]})
ax=axes[0];ax.set(xlim=(0,10),ylim=(0,3));ax.axis('off');ax.set_title('Keep a gap before using past outcomes',loc='left')
ax.plot([.4,9.5],[1.5,1.5],color=GRAY,lw=1)
ax.plot([.7,4],[1.5,1.5],color=BLUE,lw=6);ax.text(2.35,2.05,'28-day history window',ha='center',fontsize=8.5)
for x,label in [(.7,'35 days\nearlier'),(4,'7 days\nearlier'),(7,'Prediction\ncutoff'),(9.4,'Scheduled\ndeparture')]:
 ax.plot([x,x],[1.35,1.65],color=GRAY,lw=1);ax.text(x,.9,label,ha='center',va='top',fontsize=7.5)
ax.scatter([7,9.4],[1.5]*2,c=[ORANGE,GRAY],s=25,zorder=3)
ax.text(8.2,1.95,'2 hours',ha='center',fontsize=8)
ax.text(.4,.02,'Schematic; horizontal spacing is not to scale.',fontsize=7,color='#687586')
ax=axes[1];clock=np.linspace(0,2*np.pi,200);ax.plot(np.cos(clock),np.sin(clock),color='#c0cbd5',lw=1)
for minutes,label,offset in [(1435,'23:55',(.02,-.28)),(5,'00:05',(.02,.22)),(720,'12:00',(-.2,-.23))]:
 a=2*np.pi*minutes/1440;xx,yy=np.cos(a),np.sin(a);ax.scatter(xx,yy,s=28,color=BLUE);ax.annotate(label,(xx,yy),xytext=(xx+offset[0],yy+offset[1]),fontsize=8)
ax.set(aspect='equal',xlim=(-1.35,1.65),ylim=(-1.2,1.2),title='A clock should wrap at midnight',xlabel='Cosine coordinate',ylabel='Sine coordinate',xticks=[-1,0,1],yticks=[-1,0,1])
save(fig,'05_feature_logic.png')

# Figure 6: measured gains on identical final-period IDs, with saved paired intervals.
fig,axes=plt.subplots(1,2,figsize=(7.05,2.45),layout='constrained')
for ax,key,label in zip(axes,['ap','f1'],['Average precision gain','F1 gain']):
 for i,model in enumerate(model_ids):
  r=b.loc[model];v,lo,hi=[r[f'vsA_{key}_{k}']*100 for k in ['gain','low','high']]
  color=BLUE if lo>0 else GRAY
  ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color=color,capsize=3,ms=4,lw=1.7)
  ax.annotate(f'{v:+.2f}',(v,i),xytext=(0,9),textcoords='offset points',ha='center',fontsize=7)
 ax.axvline(0,color=GRAY,ls='--',lw=.8);ax.set(yticks=range(4),yticklabels=['Logistic Reg.','KNN','RBF SVM','Random Forest'],xlabel='B minus A (percentage points)',title=label);ax.invert_yaxis();ax.margins(x=.25,y=.2);ax.grid(axis='x',alpha=.13)
save(fig,'06_feature_gains.png')

# Declarative report content also supplies an editable Markdown copy.
PAGE=[]
def page(title):
 d={'title':title,'blocks':[]};PAGE.append(d);return d['blocks']
def p(blocks,text):blocks.append(('p',text))
def h(blocks,text):blocks.append(('h',text))
def image(blocks,name,caption,width=7.05):blocks.append(('image',name,caption,width))
def table(blocks,headers,rows,widths):blocks.append(('table',headers,rows,widths))
def bullet(blocks,text):blocks.append(('bullet',text))

q=page('Predeparture Flight Delay Classification')
p(q,'<b>Data extraction, cleaning, exploration and feature engineering</b><br/>ML and DW Project | 19 October 2026')
p(q,'This project studies whether a flight will depart at least 15 minutes late, using information that could be available two hours before its scheduled departure. The study covers Atlanta (ATL), Dallas Fort Worth (DFW) and Chicago O\'Hare (ORD). Most of this report explains how the data was understood and prepared. Model results follow in the final two pages.')
p(q,'Richer inputs improved risk ranking, but the selected warning rule generated many false alarms. Reviewing a small list of the highest-risk flights was more selective.')
h(q,'Project objectives')
p(q,'<b>Main objective:</b> develop and evaluate Logistic Regression, KNN, SVM and Random Forest in a historical two-hour prediction simulation, and assess whether their outputs can identify and prioritize delayed departures.')
for txt in ['<b>O1</b> Select a classifier using earlier validation and confirm its performance later.', '<b>O2</b> Measure the predictive value of weather, history and scheduled traffic beyond schedules alone.', '<b>O3</b> Identify supported patterns in missed delays and false warnings.', '<b>O4</b> Assess fixed review budgets of 5%, 10% and 20% against simple rankings.']:bullet(q,txt)
h(q,'Raw sources and extraction')
table(q,['Source','What was extracted'],[
 ['BTS monthly flight archives [3]','Flight dates, airlines, routes, scheduled times, distance, duration, cancellation flags and departure outcomes.'],
 ['IEM ASOS and METAR archive [4]','Origin weather observations, raw weather messages and station metadata used to resolve time zones.'],
 ['Raw airport reference [6]','Airport coordinates and elevation for wider route-feature experiments. These were candidates, not all final predictors.']], [1.75,5.3])
p(q,'All twelve 2025 BTS files contained 7,001,619 national records. Reading in chunks retained 957,315 scheduled departures at the three origins. Raw files were preserved; a stable flight ID used the source month and original row number. Only raw inputs were reused from the earlier unrelated project.')
image(q,'01_study_periods.png','Figure 1. EDA uses January-September 2025; final refits cover all 2025 months. Late-2024 records supply history. July 2026 is the fresh test.')

q=page('Cleaning the flight records')
p(q,'The shared preparation manifest is a log of the rules applied. The workflow covered univariate and bivariate analysis, missing values, outliers, variable creation and transformation. Source checks and the target definition came first so the plots used the correct population.')
h(q,'Parse fields and check identities')
p(q,'Codes were trimmed, dates parsed and numeric fields checked. The audited numeric fields had no non-missing parse failures. Flight identities were checked using date, airline, flight number, route and scheduled time; an unresolved duplicate would stop processing rather than be silently deleted.')
h(q,'Define the target and eligible population')
p(q,'The target is 1 when the signed departure delay is at least 15 minutes, and 0 otherwise. Early departures can legitimately have negative delay. The rebuilt label was checked against the BTS DepDel15 field [3]. Cancellation status was applied explicitly: a cancelled flight is not an on-time example.')
table(q,['2025 population decision','Flights'],[
 ['Scheduled departures at ATL DFW ORD','957,315'],['Cancelled and excluded from classification','18,898'],['Noncancelled flights with an unknown departure label','0'],['Eligible departures retained','938,417'],['Later-diverted flights included within those eligible rows','2,750']], [5.65,1.4])
p(q,'Cancelled records remain useful as schedule context for traffic counts. A later diversion does not automatically invalidate a known departure label. This means the target is delay among operated flights with known outcomes, rather than cancellation risk for every published schedule.')
h(q,'Missing values and impossible values')
p(q,'In the January-September training population, the core schedule predictors had no missing values. Missing outcomes among all schedules were handled through the eligibility rule, not filled with a guessed label. Missing weather predictors were retained for the model pipeline, as explained on page 5. Nonpositive distance or scheduled duration would become missing with an audit flag; neither rule changed a value in the audited handoff.')
h(q,'Review outliers without deleting valid events')
p(q,'Outlier checks used the middle-50% spread (IQR), distance from the mean, and the upper 1% of values. IQR flagged 20,787 training distances and 27,641 durations, but a long route or schedule can be valid. No automatic IQR deletion or percentile capping was applied. Outcome delays were not capped because they define the target.')
h(q,'Create a consistent clock')
p(q,'Scheduled HHMM values became minutes after midnight, then origin-local timestamps and UTC. Daylight-saving ambiguities were checked. The cutoff was set 120 minutes before scheduled departure. The current flight\'s actual departure and arrival information stayed outside its predictors.')

q=page('Understanding distributions before modeling')
p(q,f'Univariate analysis examines one variable at a time. The EDA population contains <b>{len(train):,} eligible departures</b> from January-September 2025. Of these, <b>{int(train.delayed_departure.sum()):,} ({train.delayed_departure.mean():.2%})</b> departed at least 15 minutes late. This imbalance makes an always-on-time classifier look reasonably accurate even though it detects no delays.')
image(q,'02_univariate_and_scatter.png','Figure 2. Complete training counts and delay distribution, plus a fixed-seed 10,000-flight scatter sample. The histogram zoom omits 12,380 observations from view only; the full-range box plot retains every delay. The 15-minute line defines the target, not an input.',7.05)
h(q,'What the graphs show')
p(q,'The typical flight is quite different from the average delay. The median departure delay is <b>-1 minute</b>, while the mean is <b>17.1 minutes</b>. A small number of very long disruptions pull the mean upward. Observed signed delays range from -60 to 2,364 minutes. The box plot uses a symmetric log scale to display both the early departures and the long positive tail without deleting them.')
p(q,'Distance and scheduled duration are also right-skewed. Their training medians are 666 miles and 126 minutes, compared with means of about 759 miles and 139 minutes. The scatter plot shows their strong relationship: correlation is <b>0.986</b>. They contain overlapping information, so retaining both is a modeling choice to evaluate, not evidence that both independently explain delays.')
p(q,'These findings guided the later use of median imputation, scale-aware preprocessing and compact feature comparisons. The observed departure delay is plotted only to understand the outcome; using that same flight\'s delay as an input would leak the answer.')

q=page('Exploring relationships and hidden patterns')
p(q,'Bivariate analysis compares variables with each other or with the target. The graphs below use rates rather than raw delay counts, because a busy airline can have more delays simply by operating more flights. These are descriptive associations, not causal explanations or proof of predictive improvement.')
image(q,'03_bivariate_findings.png','Figure 3. All panels describe January-September 2025. The airline adjustment gives AA and F9 the same shared origin-month-time-band mix, requiring at least 30 flights per airline in a group; it retains 183,589 flights, or 99.4% of their combined population.',7.05)
h(q,'Calendar and weather signals need context')
p(q,'Delay risk rose from <b>16.5% in the morning</b> to <b>35.5% in the evening</b>. June and July were higher-risk months than September in this single year. These patterns motivated cyclic calendar features and airport-specific time patterns, but they do not prove repeatable seasonality across years.')
p(q,'Flights matched to a positive precipitation-accumulation report had a <b>44.3%</b> delay rate, compared with <b>24.4%</b> for zero and <b>36.0%</b> for trace amounts. Accumulation describes an earlier reporting interval [4], not guaranteed rain at departure. Shared weather, airport conditions and the schedule can explain part of the difference.')
h(q,'A crude airline ranking can reverse')
p(q,'Frontier (F9) had a crude rate of 34.58%, slightly above American (AA) at 33.64%. After giving both airlines a common origin, month and departure-band mix, F9 was at 29.44% and AA at 33.37%. This reversal shows how composition affects the comparison. Route differences remain, so it does not establish which airline would perform better on another airline\'s flights.')
p(q,'The deeper EDA also found a sharp shock at ATL on 10 January: 643 of 764 schedules were cancelled, and 112 of the remaining 121 eligible departures were delayed. Slow historical averages can miss such sudden bad days. Cancellation exclusions therefore change the population being described, not merely the amount of data available.')

q=page('Extracting weather and treating missing values')
p(q,'Weather was joined by origin and time, using the most recent qualifying earlier report. A report had to be at least <b>20 minutes older than the prediction cutoff</b>, allowing an assumed delivery buffer, and no more than <b>180 minutes old</b>. The code used a backward as-of match, which means it searched backward in time instead of taking the nearest report on either side.')
image(q,'04_weather_quality.png','Figure 4. Flight-weighted training missingness and matched-report age. Every training flight matched a qualifying report, but condition flags were unknown for 90.6% and gust values for 80.3%. The observed report ages were 20-139 minutes, with a median of 49 minutes.',7.05)
h(q,'A weather report is not a complete set of measurements')
p(q,'IEM\'s M code can indicate an unreported, missing or quality-controlled value [4]. An absent gust was therefore left unknown rather than assumed to be zero. Trace precipitation became a numeric zero plus a separate trace flag, preserving the distinction from an ordinary zero. Negative physical quantities such as wind speed were treated as invalid; negative air temperature remained valid.')
p(q,'Ceiling height also needs a meaning-aware rule. No ceiling is required when only certain clear or scattered conditions are reported. The data therefore keeps separate indicators for no ceiling, unknown cloud coverage and unknown ceiling height. A missing ceiling number is not automatically a failed weather report.')
h(q,'Improve extraction before adding new data')
p(q,'The four structured sky-layer columns did not contain every layer in the raw METAR body. Parsing all body layers before the RMK remarks section recovered known ceilings for <b>719 training flights</b>. This corrected a representation problem using existing raw data, without changing flight identities or labels.')
p(q,'For the original 2025/Q1 weather handoff, 4,871 identical overlapping rows were collapsed. Both versions at eight conflicting station-time keys were excluded rather than choosing arbitrarily. Earlier qualifying reports still covered every training flight. The same approved rule was carried into later acquisitions; raw payloads remained unchanged.')
p(q,'Another source convention matters: visibility was stored at exactly 10 miles for 87.7% of training flights. NOAA defines 10SM as 10 miles or greater [5]. Treating all such observations as identical precise measurements would overstate the information in the source.')

q=page('Building useful features and a fair model input')
image(q,'05_feature_logic.png','Figure 5. The timing diagram is a schematic of the history and prediction rules. Clock coordinates are a mathematical representation, not measured effects. They keep 23:55 close to 00:05 and avoid treating midnight as a large numerical jump.',7.05)
table(q,['Feature family','Construction and reason'],[
 ['Calendar and schedule','Create weekday, month, departure bands and sine/cosine clock coordinates. Add richer time harmonics and airport-time interactions to represent different daily patterns.'],
 ['Earlier delay history','Use a 28-day outcome window ending seven days before cutoff. Build airport, airline-origin, route and later flight-number histories, with support counts and no-history flags.'],
 ['Scheduled traffic','Count origin schedules in the hour before scheduled departure. These are published schedules, including later cancellations; no actual future departures are used.'],
 ['Weather context','Combine temperature, visibility, wind and earlier weather-window summaries. Dew-point spread measures how close temperature is to the dew point. Avoid summing overlapping precipitation reports.'],
 ['Compact new candidates','Extract altimeter setting and its earlier three-hour change/range, variable wind and gust excess. Try squared weather terms, log transforms and inverse visibility.']], [1.35,5.7])
p(q,'History rates for small routes are pulled toward the airport\'s earlier average using a fixed prior strength of 20. This smoothing prevents a route with only a few past flights from being treated as equally reliable as a busy route. The seven-day gap is a declared availability assumption, not proof that BTS outcomes were published daily.')
p(q,'Broader experiments also tested holiday timing, route geometry, categorical combinations and scheduled arrival waves. Incoming schedules used published arrival clocks with inferred dates and discrepancy flags. These remained candidates; the final models did not automatically retain every new feature. Final schedule A has <b>26 input columns</b>, the retuned reference has 66, and selected B has 78 for LR/KNN, 66 for SVM and 72 for RF. Counts are before one-hot expansion.')
q=page('Preparing and sampling model inputs')
h(q,'Fit transformations on training data only')
p(q,'Category labels become yes/no columns (one-hot encoding), such as one column per airline. Missing continuous values use the training median, or middle value; the selected LR/KNN recipe handles binary flags separately with a zero placeholder plus missingness indicators. Unknown weather therefore remains identifiable. LR, KNN and SVM scale continuous inputs so units do not dominate; Random Forest does not require that scaling. Log transforms soften skewed magnitudes, and squared terms allow curved relationships.')
p(q,'All such fitted preprocessing stays inside the training pipeline. Models use the same fixed-seed flight IDs. Proportional month-by-outcome sampling preserves natural class proportions and keeps KNN/SVM practical; it does not balance the classes. Initial fits used 20,000 flights, followed by 20,000/40,000 comparisons; final fits used a shared 40,000-flight sample across all 2025 months. All eligible source rows remain stored.')

h(q,'Why a validation period is needed')
p(q,'Training lets a model learn patterns. Validation compares settings, feature packages and warning thresholds on later flights. Final testing checks the frozen choice on new outcomes. A random split can mix earlier and later conditions and is less realistic for this time-dependent problem. The final development procedure used the three chronological folds below.')
table(q,['Earlier training period','Threshold calibration','Later scoring period'],[
 ['January-March 2025','April 2025','May-June 2025'],
 ['January-June 2025','July 2025','August-September 2025'],
 ['January-September 2025','October 2025','November-December 2025']], [2.8,1.8,2.45])
h(q,'Sampling reduces computation rather than deleting the population')
table(q,['Stage','Eligible population','Shared model sample'],[
 ['Initial notebook training and EDA','698,670','20,000 for initial fitting'],
 ['Final all-2025 refit','938,417','40,000 across all twelve months'],
 ['Fresh July 2026 confirmation','87,298','10,000']], [3.0,1.7,2.35])
p(q,'EDA rates and most distributions used all eligible initial-training rows; scatter plots alone used a 10,000-row display sample. Model fitting samples are a separate decision. The same flight IDs and labels are used across classifiers and paired feature sets, which makes comparisons fair. Natural class proportions are retained, rather than creating a 50/50 class balance.')
p(q,'A small sample can miss rare categories: four destinations, representing 67 initial-training flights, were absent from the original 20,000-row fit sample. The encoder accepts unseen categories without crashing, although their category-specific information is then unavailable. Fixed month sine/cosine coordinates also avoid treating an unobserved calendar month as an ordinary unknown category.')

q=page('Models and measured predictive results')
p(q,'Logistic Regression combines weighted inputs; KNN looks for similar training flights; an RBF SVM learns a flexible class boundary; Random Forest combines decision trees. Their settings, feature packages and warning thresholds were selected using chronological 2025 development, then frozen before July outcomes were loaded.')
p(q,'Logistic Regression was selected by mean development F1 (0.483), with AP as tie-break. Final warning thresholds used October calibration from January-September fits; all-2025 refits kept those thresholds. Because Q1/Q2 2026 had already been evaluated, July was reserved for the new confirmation.')
table(q,['July model','F1','AP','Precision','Recall','ROC AUC','Accuracy'],[
 [name,f'{b.loc[mid,"f1"]:.3f}',f'{b.loc[mid,"average_precision"]:.3f}',f'{b.loc[mid,"precision"]:.1%}',f'{b.loc[mid,"recall"]:.1%}',f'{b.loc[mid,"roc_auc"]:.3f}',f'{b.loc[mid,"accuracy"]:.1%}']
 for mid,name in zip(model_ids,names)], [1.65,.58,.58,1.02,.88,1.0,1.04])
p(q,'The July sample has 3,215 delays among 10,000 flights (population: 87,298). Precision is the share of warnings that are correct; recall is the share of actual delays detected. F1 balances them; AP summarizes ranking across thresholds. Always-on-time accuracy is 67.85% but F1 is zero; always-delayed F1 is 0.487 and seeded fair-coin F1 is 0.398.')
p(q,'<b>Logistic Regression B remains the validation choice.</b> Its July F1 is 0.540 and recall is 92.4%, but precision is 38.1%. It produced 2,971 correct warnings, 4,827 false warnings and 244 missed delays. Thus 61.9% of its warnings were false. KNN\'s higher July F1 does not change the earlier selection. LR fit in 0.37 seconds and scored 10,000 flights in 0.023 seconds; SVM took about 250 seconds and 14 seconds respectively on this run.')
image(q,'06_feature_gains.png','Figure 6. Richer B minus matched-settings schedule A on identical July flights. Points are observed gains; whiskers are paired 95% day-bootstrap intervals (300 draws, 31 days). They are conditional on the fitted models, not guarantees about future years.',7.05)
p(q,'All four AP intervals are positive for A to B. Random Forest\'s F1 interval crosses zero. Against the equally retuned 66-input reference, compact additions were mixed: LR F1 rose by 1.50 points but its AP interval crossed zero; KNN improved modestly; RF AP fell by about 1.00 point; SVM retained 66 inputs. Adding features did not guarantee better prediction.')

q=page('Conclusions limitations and references')
h(q,'What the objectives established')
p(q,'<b>O1:</b> the chronological procedure selected Logistic Regression B. Its final scores demonstrate useful ranking but do not establish a reliable broad warning system. A high detection rate came with a large false-warning burden.')
p(q,'<b>O2:</b> weather, historical outcomes and scheduled traffic improved AP beyond schedule-only A on the same flights. Compact additions beyond retuned 66 inputs gave mixed results. The combined comparison cannot isolate the separate contribution of weather, history and traffic.')
p(q,'<b>O3:</b> many on-time flights were wrongly warned; the false-positive rate measures this share. For selected LR B, ORD\'s rate was 75.5%, versus ATL\'s 64.7%. In the evening it warned on 2,799 of 2,804 flights, giving 100% recall but a 99.7% false-positive rate. Unsupported small-group rates stay blank; the reporting rules require at least 200 group flights and 30 relevant outcomes or warnings.')
p(q,'<b>O4:</b> the top 10% LR review list contained <b>688 actual delays among 1,000 flights</b>: 68.8% precision, 21.4% of all delays captured and 2.14 times random lift. Historical risk found 442 delays in the same-size list. The model list still had 312 false alerts and left 2,527 delays outside it. At 5%/20% budgets it captured 359/1,222 delays. Ranks and outcome-independent tie order were fixed before evaluation.')
h(q,'Limits and practical meaning')
p(q,'The result applies to one summer month at three hubs and to departures with known operated outcomes. Weather receipt, timely access to earlier outcomes and archived schedule availability are assumptions. Many flights share the same weather report, so they are not independent weather experiments. Reused development periods can introduce selection optimism; threshold transfer after refitting and seasonal mix changes can alter warning rates.')
p(q,'False positives are unnecessary warnings; false negatives are missed delays. Their relative cost depends on the intended action, and no operational costs were measured. Scores are uncalibrated risk scores, not verified probabilities. Aircraft rotation was excluded because assignment timing was unavailable. The study does not establish prevented delays or money saved. Additional years, genuinely timestamped operational inputs and a fresh future evaluation would support stronger claims.')
h(q,'References and reproducibility')
p(q,'The organization follows the concise introduction, EDA, modeling and conclusion sequence in [1]. The related flight report [2] informed the emphasis on data preparation and feature explanations; it predicts arrival delay at departure and is not a comparable score benchmark. All numerical results in this report come from this project, not those examples.')
refs=[
 ('1','Carnegie Mellon University','Linear Regression Project Requirements','https://stat.cmu.edu/cmsac/sure/materials/lm_project_requirements.html'),
 ('2','Duperier Sauvestre and Leaf','Modeling Flight Delays CS229 report 2016','https://cs229.stanford.edu/proj2016/report/DuperierSauvestreLeaf-ModelingFlightDelays-report.pdf'),
 ('3','Bureau of Transportation Statistics','Reporting Carrier On Time Performance field definitions','https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ'),
 ('4','Iowa Environmental Mesonet','ASOS AWOS METAR archive and variable definitions','https://mesonet.agron.iastate.edu/request/download.phtml'),
 ('5','NOAA','ASOS User Guide visibility and cloud conventions','https://www.weather.gov/media/asos/aum-toc.pdf'),
 ('6','OurAirports','Raw airport reference downloads','https://ourairports.com/data/')]
for n,author,title,url in refs:q.append(('ref',f'[{n}] {author}. <link href="{escape(url)}" color="#245e87">{title}</link>. Accessed 19 October 2026.'))
q.append(('ref','Project evidence: notebooks 01 and 02 (cleaning and features), 02a (deeper EDA), 06 and 07 (final modeling and July confirmation); reports/preparation_manifest.json; reports/final_models/selection.json; tableau/exports/final_july/. The interactive HTML dashboard presents O1-O4. Report figures are regenerated from saved data, without refitting models.'))

# Build a plain, typeset PDF. A page break separates each purposeful report section.
fontdir=Path('/System/Library/Fonts/Supplemental')
for family,filename in [('Arial','Arial.ttf'),('Arial-Bold','Arial Bold.ttf'),('Arial-Italic','Arial Italic.ttf')]:
 path=fontdir/filename
 if path.exists():pdfmetrics.registerFont(TTFont(family,str(path)))
pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='Arial-Bold',italic='Arial-Italic',boldItalic='Arial-Bold')
styles={
 'title':ParagraphStyle('Title',fontName='Arial-Bold',fontSize=23,leading=27,textColor=colors.black,spaceAfter=12),
 'page':ParagraphStyle('Page',fontName='Arial-Bold',fontSize=18,leading=22,textColor=colors.black,spaceAfter=12),
 'h':ParagraphStyle('Heading',fontName='Arial-Bold',fontSize=11.5,leading=15,textColor=colors.black,spaceBefore=7,spaceAfter=4,keepWithNext=True),
 'p':ParagraphStyle('Body',fontName='Arial',fontSize=10.6,leading=14.25,textColor=colors.HexColor('#202a34'),spaceAfter=7),
 'bullet':ParagraphStyle('Bullet',fontName='Arial',fontSize=10.4,leading=13.3,textColor=colors.HexColor('#202a34'),spaceAfter=4,leftIndent=10,firstLineIndent=-10),
 'caption':ParagraphStyle('Caption',fontName='Arial',fontSize=8.3,leading=11,textColor=colors.HexColor('#596574'),spaceAfter=9),
 'cell':ParagraphStyle('Cell',fontName='Arial',fontSize=9.1,leading=11.6,textColor=colors.black),
 'ref':ParagraphStyle('Reference',fontName='Arial',fontSize=8.1,leading=10.8,textColor=colors.HexColor('#384553'),spaceAfter=4),
}
styles['headcell']=ParagraphStyle('Headcell',parent=styles['cell'],fontName='Arial-Bold',fontSize=9,leading=11.5,textColor=colors.white)
story=[]
for i,pageobj in enumerate(PAGE):
 if i:story.append(PageBreak())
 story.append(Paragraph(pageobj['title'],styles['title' if i==0 else 'page']))
 for block in pageobj['blocks']:
  if block[0] in ['p','h','ref']:story.append(Paragraph(block[1],styles[block[0]]))
  elif block[0]=='bullet':story.append(Paragraph('- '+block[1],styles['bullet']))
  elif block[0]=='image':
   _,name,caption,width=block;im=ImageReader(str(FIG/name));iw,ih=im.getSize();story.append(Image(str(FIG/name),width=width*72,height=width*72*ih/iw));story.append(Paragraph(caption,styles['caption']))
  elif block[0]=='table':
   _,headers,rows,widths=block
   cells=[[Paragraph(escape(str(x)),styles['headcell']) for x in headers]]+[[Paragraph(escape(str(x)),styles['cell']) for x in row] for row in rows]
   tab=Table(cells,colWidths=[x*72 for x in widths],repeatRows=1,hAlign='LEFT')
   tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#263d54')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('BOX',(0,0),(-1,-1),.5,colors.HexColor('#d9d9d9')),('INNERGRID',(0,0),(-1,-1),.4,colors.HexColor('#d9d9d9')),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),5.5),('BOTTOMPADDING',(0,0),(-1,-1),5.5)]+[('BACKGROUND',(0,j),(-1,j),colors.HexColor('#f3f6f9') if j%2==0 else colors.white) for j in range(1,len(cells))]))
   story.extend([tab,Spacer(1,9)])

def decorate(canvas,doc):
 canvas.saveState();canvas.setFont('Arial',8);canvas.setFillColor(colors.HexColor('#647181'))
 canvas.drawString(52,27,'ML and DW Project | Departure delay classification')
 canvas.drawRightString(560,27,str(doc.page));canvas.restoreState()

doc=SimpleDocTemplate(str(PDF),pagesize=letter,rightMargin=52,leftMargin=52,topMargin=43,bottomMargin=43,title='Predeparture Flight Delay Classification',author='',subject='Data extraction cleaning EDA feature engineering and final model findings')
doc.build(story,onFirstPage=decorate,onLaterPages=decorate)

# Editable source uses the same prose, tables and figures as the PDF.
def md(text):
 text=re.sub(r'<link href="([^"]+)"[^>]*>(.*?)</link>',r'[\2](\1)',text)
 return text.replace('<b>','**').replace('</b>','**').replace('<br/>','\n').replace('&amp;','&')
lines=[]
for i,pageobj in enumerate(PAGE):
 lines.append(('# ' if i==0 else '## ')+pageobj['title']+'\n')
 for block in pageobj['blocks']:
  if block[0] in ['p','ref']:lines.append(md(block[1])+'\n')
  elif block[0]=='h':lines.append('### '+block[1]+'\n')
  elif block[0]=='bullet':lines.append('- '+md(block[1])+'\n')
  elif block[0]=='image':lines.append(f'![{block[2]}](figures/{block[1]})\n\n{block[2]}\n')
  elif block[0]=='table':
   lines.append('| '+' | '.join(block[1])+' |');lines.append('| '+' | '.join(['---']*len(block[1]))+' |')
   for row in block[2]:lines.append('| '+' | '.join(map(str,row))+' |')
   lines.append('')
(OUT/'predeparture_flight_delay_report.md').write_text('\n'.join(lines)+'\n')
stats['source_run']=manifest['run_id'];stats['source_hashes']=manifest['files'];stats['model_selection_sha256']=hashlib.sha256((ROOT/'reports/final_models/selection.json').read_bytes()).hexdigest();stats['model_refitting_performed']=False
(OUT/'report_evidence.json').write_text(json.dumps(stats,indent=2)+'\n')
print(PDF)
print('Report sections:',len(PAGE),'| figures:',len(list(FIG.glob('*.png'))))
