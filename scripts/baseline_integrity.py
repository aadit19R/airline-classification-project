"""Verify frozen inputs and separately documented notebook-only revisions."""
import hashlib
import json

import nbformat


def _documented_readability_revision(root, relative, expected, actual):
    """Accept only a verified notebook revision with its original backup intact."""
    registry = root / 'reports/notebook_simplification/revision.json'
    if not registry.exists():
        return None
    revision = json.loads(registry.read_text())
    record = revision.get('notebooks', {}).get(relative)
    if record is None:
        return None
    assert revision['status'] == 'passed', 'Notebook simplification is not verified.'
    assert relative in {
        'notebooks/01_cleaning_and_eda.ipynb',
        'notebooks/02_feature_engineering.ipynb',
        'notebooks/02a_deeper_training_exploration.ipynb',
    }
    assert actual == record['after_sha256'], f'Notebook changed after verification: {relative}'
    backup = root / record['backup']
    backup_sha = hashlib.sha256(backup.read_bytes()).hexdigest()
    assert backup_sha == record['before_sha256'], 'Original notebook backup changed.'
    original = nbformat.read(backup, as_version=4)
    if backup_sha != expected:
        if relative == 'notebooks/01_cleaning_and_eda.ipynb':
            # A blank, output-free cell had been added before this refactor.
            # Removing only such cells must recover the exact frozen bytes.
            saved = json.loads(backup.read_text())
            saved['cells'] = [c for c in saved['cells'] if not (
                c['cell_type'] == 'code' and not ''.join(c['source']).strip() and not c.get('outputs'))]
            recovered = (json.dumps(saved, indent=1, ensure_ascii=False) + '\n').encode()
            assert hashlib.sha256(recovered).hexdigest() == expected, 'Original cleaning baseline changed.'
        else:
            # Notebook 02 already had an independently audited appendix before
            # this revision. Keep that earlier chain of evidence intact.
            assert relative == 'notebooks/02_feature_engineering.ipynb'
            appendix = json.loads((root / 'reports/feature_engineering_extension.json').read_text())
            assert appendix['status'] == 'passed'
            assert appendix['original_notebook_sha256'] == expected
            for value, field in [(original.cells[:appendix['original_cell_count']], 'original_cells_sha256'),
                                 (original.metadata, 'original_metadata_sha256')]:
                digest = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                assert digest == appendix[field], f'Original appendix baseline changed: {field}'
    for path, digest in revision['supporting_files'].items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest, f'Changed revision support: {path}'
    # Raw JSON retains the exact saved multiline representation. nbformat may
    # normalize lists of text lines into strings while reading a notebook.
    saved_books = [json.loads(backup.read_text()), json.loads((root / relative).read_text())]
    output_lists = [[o for c in nb['cells'] if c['cell_type'] == 'code' for o in c.get('outputs', [])]
                    for nb in saved_books]
    output_hashes = [hashlib.sha256(json.dumps(values, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                     for values in output_lists]
    assert output_hashes[0] == output_hashes[1] == record['outputs_sha256'], 'Saved notebook outputs changed.'
    return {'original_backup_verified': True, 'saved_outputs_unchanged': True,
            'fresh_execution_verified': True}


def verify_baseline_files(root, fingerprints):
    """Protect original notebook cells and metadata as well as all other files."""
    appended, revised = {}, {}
    for relative, expected in fingerprints.items():
        path = root / relative
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual == expected:
            continue
        documented = _documented_readability_revision(root, relative, expected, actual)
        if documented is not None:
            revised[relative] = documented
            continue
        assert relative == 'notebooks/02_feature_engineering.ipynb', f'Changed frozen input: {relative}'
        record = json.loads((root / 'reports/feature_engineering_extension.json').read_text())
        assert record['status'] == 'passed', 'The notebook-02 appendix has not passed its audit.'
        assert record['original_notebook_sha256'] == expected
        notebook = nbformat.read(path, as_version=4)
        count = record['original_cell_count']
        for value, field in [(notebook.cells[:count], 'original_cells_sha256'),
                             (notebook.metadata, 'original_metadata_sha256')]:
            digest = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            assert digest == record[field], f'Original notebook-02 content changed: {field}'
        assert len(notebook.cells) > count
        appended[relative] = {'original_cells_preserved': count, 'original_metadata_preserved': True}
    return {'checked_files': len(fingerprints), 'documented_notebook_appendices': appended,
            'documented_readability_revisions': revised}
