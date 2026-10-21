# Preparation notebook readability update

The three preparation notebooks keep the original analysis, explanatory notes, tables, figures, feature calculations and safety checks. Related cells are grouped, the empty cleaning cell is removed, and longer technical routines are named functions in [preparation_helpers.py](../../scripts/preparation_helpers.py). Familiar EDA and plotting code remains visible in the notebooks. There is no dynamic execution of hidden notebook-cell strings.

| Notebook | Code cells before | Code cells after | Visible nonblank code lines before | After |
| --- | ---: | ---: | ---: | ---: |
| [01: cleaning and EDA](../../notebooks/01_cleaning_and_eda.ipynb) | 23, including one empty cell | 15 | 443 | 200 |
| [02: features and both experiments](../../notebooks/02_feature_engineering.ipynb) | 43 | 29 | 1,353 | 353 |
| [02a: deeper exploration](../../notebooks/02a_deeper_training_exploration.ipynb) | 11 | 10 | 311 | 81 |

Most cells contain 20 lines or fewer. Some complete plotting cells and setup cells are slightly longer so a single step is not fragmented. All three sections of notebook 02 remain: original preparation, additional candidate features, and the broader 90-fit experiment. The additional-candidate section can still start in a fresh kernel.

## Outputs and verification

The outputs displayed in the edited notebooks are the original executed outputs, retained exactly. Their historical execution dates and counts refer to those original runs. Execution counters are cleared because cells have been regrouped; the output provenance is recorded in cell metadata. Fresh verification executes the revised code separately, rather than relabeling old outputs as new runs.

See [verification.json](verification.json) for the completed checks, [structure.json](structure.json) for each extracted function's source/AST record and cell mapping, and [revision.json](revision.json) for the hashes accepted by the project's integrity checker. Original notebook files and the earlier integrity checker are retained under [baseline/](baseline/); [baseline.json](baseline.json) fingerprints the protected artifacts.

The completed verification executed **54 code cells**, regenerated **41 byte-identical figures**, matched the non-timing text outputs, reproduced all **90 development-fit results**, and reproduced both Parquet handoffs byte for byte. All **39 protected data/model/result artifacts** retained their hashes. The independent [final-model audit](final_model_compatibility.json) and [July export/model-reload audit](july_export_compatibility.json) also passed. The [guard checks](integrity_guard_checks.json) confirm that unapproved edits to a notebook, its helper or the data configuration are still rejected.

Verification uses an isolated temporary project copy and a fresh in-process Jupyter kernel in a new Python process for each notebook. It runs the original save cells and the full feature experiment there. It compares both generated Parquet files to the original SHA-256 hashes, preparation rules excluding completion timestamps, and all 90 development-fit results excluding runtime timing fields. The existing frozen model bundles, selections, datasets and evaluation exports in the actual project are retained.

To repeat the isolated execution check from the project root:

```bash
python3 scripts/verify_preparation_notebooks.py
```

This includes the existing 90 model fits in notebook 02, so the experiment still takes several minutes even though the notebook is shorter. Temporary verification copies are not new project source datasets or replacement evaluation results.

## Integrity compatibility

The frozen original fingerprint registry is unchanged. `baseline_integrity.py` accepts only the exact verified notebook revisions when their original backups, original saved outputs and verification/supporting-file hashes match. Other frozen files must retain their original checksums. The earlier notebook-02 appendix chain is checked against the original backup.

Notebook 04 has a small compatibility-only adjustment: its direct old-byte assertions use the same documented revision verifier. It distinguishes unchanged file bytes from preserved baseline content. Its analysis, model settings and saved outputs remain intact. No experiment was retuned and no final-test choice changed.

Keep `scripts/preparation_helpers.py` with the project when sharing the notebooks. It contains the longer source-reading, weather/time parsing, history-window, validation and experiment routines; its names explain the corresponding notebook steps.
