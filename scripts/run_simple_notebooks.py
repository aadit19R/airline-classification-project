"""Execute the teaching sequence in order, giving every notebook a fresh kernel.

    python3 scripts/run_simple_notebooks.py

--start / --stop permit resuming an interrupted run without repeating passed fits.
No historical notebook or frozen artifact is an execution target.
"""
from pathlib import Path
from datetime import datetime, timezone
from time import perf_counter
import argparse
import json
import os
import tempfile
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=int, default=1, choices=range(1, 5))
    parser.add_argument('--stop', type=int, default=4, choices=range(1, 5))
    args = parser.parse_args()
    assert args.start <= args.stop
    out = ROOT/'reports/simple_reproduction'
    out.mkdir(exist_ok=True, parents=True)
    log_path = out/'execution.json'
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    for path in sorted((ROOT/'notebooks/simple').glob('*.ipynb')):
        if not args.start <= int(path.name[:2]) <= args.stop:
            continue
        notebook = nbformat.read(path, as_version=4)
        started = perf_counter()
        began = datetime.now(timezone.utc).isoformat()
        print('Fresh kernel:', path.name, flush=True)
        with tempfile.TemporaryDirectory(prefix='flight-notebook-') as cache:
            env = dict(os.environ, MPLCONFIGDIR=cache, IPYTHONDIR=cache,
                       JUPYTER_PLATFORM_DIRS='1', PYTHONHASHSEED='0')
            client = NotebookClient(notebook, timeout=1200, kernel_name='python3',
                                    resources={'metadata': {'path': str(path.parent)}})
            client.on_cell_start = lambda cell, cell_index, **kw: print(
                f'  cell {cell_index+1}', flush=True) if cell.cell_type == 'code' else None
            try:
                client.execute(env=env)
            finally:
                nbformat.write(notebook, path)
        code_cells = [c for c in notebook.cells if c.cell_type == 'code']
        errors = [o for c in code_cells for o in c.outputs if o.output_type == 'error']
        stderr = [o.text for c in code_cells for o in c.outputs
                  if o.output_type == 'stream' and o.name == 'stderr']
        figures = sum('image/png' in o.get('data', {}) for c in code_cells for o in c.outputs)
        log[path.name] = {'started_at_utc': began, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
            'seconds': perf_counter()-started, 'fresh_kernel': True, 'code_cells': len(code_cells),
            'visible_code_lines': sum(len(c.source.splitlines()) for c in code_cells),
            'figures': figures, 'errors': len(errors), 'stderr': stderr}
        log_path.write_text(json.dumps(log, indent=2)+'\n')
        print('Completed:', path.name, log[path.name]['seconds'], 'seconds;', figures, 'figures', flush=True)


if __name__ == '__main__':
    main()
