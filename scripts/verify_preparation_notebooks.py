"""Verify the simplified preparation notebooks in an isolated project copy.

Runs all three notebooks, including the existing 90-fit development appendix.
Uses a fresh in-process Jupyter kernel in a new Python process per notebook.
It creates temporary validation files; original datasets/models/results are not rewritten.

Usage: python3 scripts/verify_preparation_notebooks.py
"""
import sys

if '--execute' in sys.argv:
    from pathlib import Path
    import hashlib,json,sys,time,traceback,os
    os.environ.setdefault("IPYTHONDIR","/private/tmp/preparation-ipython")
    os.environ.setdefault("MPLCONFIGDIR","/private/tmp/preparation-matplotlib")
    os.environ.setdefault("PYDEVD_DISABLE_FILE_VALIDATION","1")
    import nbformat
    from ipykernel.inprocess.ipkernel import InProcessKernel
    from IPython.utils.capture import capture_output

    stage=Path(sys.argv[2]);stem=sys.argv[3];original=Path(sys.argv[4])
    # One new Python process per notebook, with the normal inline plotting backend.
    kernel=InProcessKernel()
    shell=kernel.shell
    shell.run_line_magic('matplotlib','inline')
    nb=nbformat.read(stage/'notebooks'/f'{stem}.ipynb',as_version=4)
    manifest_original=(original/'reports/preparation_manifest.json').read_bytes()
    baseline=json.loads((original/'reports/notebook_simplification/baseline.json').read_text())
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    def cleaned(v):
     if isinstance(v,dict):return {k:cleaned(x) for k,x in v.items() if k not in ['completed_at_utc','started_at_utc','fit_seconds']}
     if isinstance(v,list):return [cleaned(x) for x in v]
     return v
    for i,cell in enumerate(nb.cells):
     if cell.cell_type!='code':continue
     with capture_output() as captured:
      executed=shell.run_cell(cell.source,store_history=True)
     cell.execution_count=shell.execution_count-1
     cell.outputs=[]
     for label,text in [('stdout',captured.stdout),('stderr',captured.stderr)]:
      if text:cell.outputs.append(nbformat.v4.new_output('stream',name=label,text=text))
     for output in captured.outputs:
      cell.outputs.append(nbformat.v4.new_output('display_data',data=output.data,metadata=output.metadata))
     nbformat.write(nb,stage/'verification_runs'/f'{stem}.ipynb')
     if not executed.success:
      raise RuntimeError(f'{stem} cell {i}: {executed.error_before_exec or executed.error_in_exec}\n{captured.stdout}\n{captured.stderr}')
     print(f'  finished cell {i}: '+cell.source.splitlines()[0][:90],flush=True)
     if 'prep.save_feature_handoff(' in cell.source:
      assert sha(stage/'data/processed/features.parquet')==baseline['protected_artifacts']['data/processed/features.parquet']
      assert cleaned(json.loads((stage/'reports/preparation_manifest.json').read_text()))==cleaned(json.loads(manifest_original))
      (stage/'reports/preparation_manifest.json').write_bytes(manifest_original)
      print('  MATCH: feature Parquet bytes and preparation rules; restored frozen timestamp snapshot for appendix checks',flush=True)
    raise SystemExit(0)

from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,time,traceback
import nbformat

ROOT=Path(__file__).resolve().parents[1]
import tempfile
STAGE=Path(tempfile.mkdtemp(prefix='departure-preparation-check-'))
OUT=ROOT/'reports/notebook_simplification'
BASE=json.loads((OUT/'baseline.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def log(message):print(message,flush=True)
def cleaned(value):
 if isinstance(value,dict):return {k:cleaned(v) for k,v in value.items() if k not in ['completed_at_utc','started_at_utc','fit_seconds']}
 if isinstance(value,list):return [cleaned(x) for x in value]
 return value

def setup():
 if STAGE.exists():shutil.rmtree(STAGE)
 STAGE.mkdir();(STAGE/'data').mkdir()
 (STAGE/'data/raw').symlink_to(ROOT/'data/raw',target_is_directory=True)
 (STAGE/'scripts').symlink_to(ROOT/'scripts',target_is_directory=True)
 for name in ['notebooks','config','models','reports']:
  shutil.copytree(ROOT/name,STAGE/name,ignore=shutil.ignore_patterns('final_report','tmp','__pycache__'))
 shutil.copytree(ROOT/'data/processed',STAGE/'data/processed')
 shutil.copytree(ROOT/'tableau/exports',STAGE/'tableau/exports')
 (STAGE/'verification_runs').mkdir()

result={'status':'running','started_at_utc':datetime.now(timezone.utc).isoformat(),'isolated_validation_root':str(STAGE),'notebooks':{}}
(OUT/'execution_progress.json').write_text(json.dumps(result,indent=2)+'\n')
setup()
manifest_original=(ROOT/'reports/preparation_manifest.json').read_bytes()

def check_handoff(name):
 p=STAGE/'data/processed'/name
 expected=BASE['protected_artifacts']['data/processed/'+name]
 assert sha(p)==expected, f'{name} bytes changed in isolated rerun'
 log('  MATCH: '+name+' exact original SHA-256')

try:
 for stem in ['01_cleaning_and_eda','02_feature_engineering','02a_deeper_training_exploration']:
  log('RUN '+stem);started=time.monotonic()
  nb=nbformat.read(STAGE/'notebooks'/f'{stem}.ipynb',as_version=4);nbformat.validate(nb)
  import subprocess,sys
  completed=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--execute',str(STAGE),stem,str(ROOT)],cwd=STAGE)
  assert completed.returncode==0,f'{stem} isolated execution failed; see log'
  nb=nbformat.read(STAGE/'verification_runs'/f'{stem}.ipynb',as_version=4)
  dest=STAGE/'verification_runs'/f'{stem}.ipynb';nbformat.write(nb,dest)
  code=[c for c in nb.cells if c.cell_type=='code']
  figures=[o.data['image/png'] for c in code for o in c.get('outputs',[]) if 'image/png' in o.get('data',{})]
  assert not [o for c in code for o in c.get('outputs',[]) if o.output_type=='error']
  result['notebooks'][stem]={'status':'passed','code_cells_executed':len(code),'figures':len(figures),'seconds':round(time.monotonic()-started,2),'executed_copy':str(dest)}
  if stem=='01_cleaning_and_eda':
   check_handoff('flights_clean.parquet')
   old=json.loads(manifest_original)['cleaning'];new=json.loads((STAGE/'reports/preparation_manifest.json').read_text())['cleaning']
   assert cleaned(new)==cleaned(old),'Cleaning manifest changed beyond runtime timestamps'
   (STAGE/'reports/preparation_manifest.json').write_bytes(manifest_original)
  if stem=='02_feature_engineering':
   fresh=json.loads((STAGE/'reports/broad_feature_experiments.json').read_text());old=json.loads((ROOT/'reports/broad_feature_experiments.json').read_text())
   post_audit = {'status','implementation_notes','implemented_candidates_without_correlation_filter','independent_audit','plot_refresh'}
   assert fresh['status']=='executed_pending_independent_audit' and old['status']=='passed'
   assert cleaned({k:v for k,v in fresh.items() if k not in post_audit})==cleaned({k:v for k,v in old.items() if k not in post_audit}), 'Broad-feature calculations changed beyond timing/audit metadata'
   result['broad_development_fits_checked']=len(fresh['runs']);log('  MATCH: all broad-experiment results excluding fit/run timing')
  (OUT/'execution_progress.json').write_text(json.dumps(result,indent=2)+'\n')
 result['status']='passed'
except Exception as error:
 result['status']='failed';result['error']=repr(error);result['traceback']=traceback.format_exc();log(result['traceback'])
finally:
 result['finished_at_utc']=datetime.now(timezone.utc).isoformat()
 (OUT/'execution_progress.json').write_text(json.dumps(result,indent=2)+'\n')
 log('VALIDATION STATUS: '+result['status'])
if result['status']!='passed':raise SystemExit(1)
