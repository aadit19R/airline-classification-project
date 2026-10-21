import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const html=fs.readFileSync(path.join(root,'dashboard/dist/departure_delay_dashboard.html'),'utf8');
const data=JSON.parse(html.match(/<script type="application\/json" id="study-data">([\s\S]*?)<\/script>/)[1]);
const script=html.match(/<script id="dashboard-app">([\s\S]*?)<\/script>/)[1];
const context={window:{__DASHBOARD_DATA:data,__DASHBOARD_TEST_MODE:true},Intl,console,setTimeout,Map,Set,Number,Math};
vm.createContext(context);vm.runInContext(script,context);
const c=context.window.DashboardCore;
let checks=0;
function check(ok,message){assert.ok(ok,message);checks++;}
function reset(){Object.assign(c.state,c.DEFAULTS,{hiddenCurves:[]});}
function close(a,b){return Math.abs(a-b)<1e-10;}
check(data.flights.length===10000,'10,000 unique flights');
check(Object.keys(data.curves).length===16,'all 16 precision–recall candidates');
check(Object.values(data.curves).reduce((n,a)=>n+a.length,0)===113047,'all curve points retained');
check(!/<script[^>]+src=|<link[^>]+href=https?:/i.test(html),'no external assets');
check(!/\bfetch\s*\(|XMLHttpRequest|WebSocket/.test(script),'no network calls');
for(const id of c.MODEL_IDS){
 c.state.model=id;
 const full=c.allFlightRows(),m=c.metric(id,'B');
 check(new Set(full.map(r=>r.id)).size===10000,id+' unique row IDs');
 check(new Set(full.map(r=>r.rank)).size===10000,id+' unique original ranks');
 for(const type of ['TP','FP','FN','TN'])check(full.filter(r=>r.outcome===type).length===m[type.toLowerCase()],id+' '+type);
 for(const quota of [.05,.1,.2]){
  c.state.quota=quota;c.state.airport='All';c.state.search='';
  const q=c.getQuota(),rows=c.getFlightRows('review');
  check(rows.length===q.reviewed_flights,id+' quota count');
  check(rows.reduce((n,r)=>n+r.actual,0)===q.captured_delays,id+' captured delays');
  check(rows.filter(r=>!r.actual).length===q.false_alerts,id+' false alerts');
  check(rows.every((r,i)=>r.rank===i+1),id+' rank ordering');
  check(close(q.review_precision,rows.filter(r=>r.actual).length/rows.length),id+' precision');
  c.state.airport='ATL';const filtered=c.getFlightRows('review');
  check(filtered.every(r=>r.origin==='ATL'&&r.rank<=q.reviewed_flights),id+' airport scope');
  check(c.getQuota().reviewed_flights===q.reviewed_flights,id+' global quota unchanged');
 }
 c.state.airport='All';c.state.outcome='errors';c.state.group=null;c.state.quota=.1;
 check(c.getFlightRows('errors').length===m.fp+m.fn,id+' FP + FN');
 c.state.outcome='FN';check(c.getFlightRows('errors').length===m.fn,id+' missed-delay filter');
 c.state.outcome='errors';
 for(const grouping of ['origin','reporting_airline','departure_time_band']){
  c.state.grouping=grouping;
  const groups=c.groupData();
  check(groups.reduce((n,g)=>n+g.n_flights,0)===10000,id+' '+grouping+' denominator');
  for(const g of groups){
   c.state.group=g.group_value;const rows=c.getFlightRows('errors');
   check(rows.length===g.fp+g.fn,id+' '+grouping+' '+g.group_value+' error list');
  }
 }
 c.state.group=null;c.state.grouping='origin';c.state.airport='All';
 for(const feature of ['A','B','reference66']){
  c.state.feature=feature;c.state.page='performance';
  const output=c.performance();check(!/NaN|undefined/.test(output),id+' '+feature+' render');
  const row=c.metric();check(row.n_flights===10000,id+' identical feature cohorts');
 }
}
reset();
for(const contrast of ['A','66']){
 c.state.contrast=contrast;const feature=contrast==='A'?'A':'reference66';
 for(const row of c.featureData())for(const [key,column] of [['ap','average_precision'],['f1','f1']]){
  check(close(c.gainValue(row,key),row[column]-c.metric(row.model_id,feature)[column]),row.model_id+' paired '+contrast+' '+key);
 }
 check(!/NaN|undefined/.test(c.features()),'feature comparison render '+contrast);
}
reset();
check(c.groupData().find(g=>g.group_value==='ATL').false_positive_rate>.64,'ATL FPR');
c.state.grouping='reporting_airline';
for(const airline of ['AS','B6','WN'])check(c.groupData().find(g=>g.group_value===airline).precision===null,'unsupported '+airline+' stays null');
c.state.grouping='departure_time_band';check(c.groupData().find(g=>g.group_value.includes('overnight')).recall===null,'overnight stays null');
reset();
for(const page of ['performance','features','reliability','prioritization']){
 c.state.page=page;check(!/NaN|undefined/.test(c[page]()),page+' render');
 check(c.exportRows().length>0,page+' data export');
}
reset();c.state.page='prioritization';check(c.getQuota().captured_delays===688,'default 688 delays in 1000');
c.state.search='no-such-flight';check(c.flightTable('review').rows.length===0,'empty table handled');check(c.flightTable('review').html.includes('No flights match'),'empty-state message');
check(c.csvText([{name:'x,y',note:'a"b',value:1}]).includes('"x,y"'),'CSV commas escaped');
check(c.csvText([{name:'=FORMULA'}]).includes("'=FORMULA"),'CSV formula text protected');
check(c.nearestPoint([[1,.2],[.5,.4],[0,1]],.6)[0]===.5,'descending PR nearest point');
check(c.nearestPoint([[.05,.1],[.1,.2],[.2,.3]],.11)[0]===.1,'ascending quota nearest point');
// Exercise actual chart generators at desktop and narrow widths without a browser.
for(const width of [280,350,560,900]){
 reset();
 const el={clientWidth:width,innerHTML:'',dataset:{chart:'test'}};
 c.barChart(el,[{label:'Logistic Regression',value:.54},{label:'Unsupported group',value:null}]);
 check(!/NaN|undefined/.test(el.innerHTML),'bar chart geometry '+width);
 for(const key of ['ap','f1']){c.forestChart(el,key);check(!/NaN|undefined/.test(el.innerHTML),'forest chart '+key+' '+width);}
 c.lineChart(el,[{label:'Test',color:'#236b9f',points:[[0,1],[.5,.6],[1,.3]]}],{title:'Test',xLabel:'Recall (%)',yLabel:'Precision (%)'});
 check(!/NaN|undefined/.test(el.innerHTML),'line chart '+width);
 c.compositionChart(el);check(!/NaN|undefined/.test(el.innerHTML),'review composition '+width);
}
const result={checks,passed:true,models:4,source_files:6,flight_rows:10000,pr_points:113047};
fs.writeFileSync(path.join(root,'dashboard/verification.json'),JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify(result,null,2));
