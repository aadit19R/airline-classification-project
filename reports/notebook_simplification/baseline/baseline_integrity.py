"""Verify frozen inputs, allowing the documented appendix to notebook 02."""
import hashlib
import json

import nbformat


def verify_baseline_files(root, fingerprints):
    """Protect original notebook cells and metadata as well as all other files."""
    appended = {}
    for relative, expected in fingerprints.items():
        path = root / relative
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual == expected:
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
    return {'checked_files': len(fingerprints), 'documented_notebook_appendices': appended}
