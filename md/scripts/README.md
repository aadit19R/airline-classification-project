# Project scripts

The [four notebooks](../notebooks/simple/README.md) contain the analysis and visible classifier fitting calls. Their reusable modules are:

| Module | Purpose |
|---|---|
| `simple_sources.py` | Decode the raw flight files and convert schedule times |
| `simple_workflow.py` | Cleaning checks, feature stages, sklearn pipeline construction and result formatting; no model fitting |
| `improvement_features.py` | Time/weather/history/traffic calculations used by the selected features |
| `final_features.py` | Final pressure/wind features and independent feature reconstruction |
| `final_modelling.py` | Frozen preprocessing recipes, shared sampling and metric utilities |
| `fresh_test_data.py` | Reviewed source decoding and sampling used by feature reconstruction |

Run `python3 scripts/run_simple_notebooks.py` from the project root to execute all four notebooks in fresh kernels. Outputs go to `reports/simple_reproduction/`. No old notebook or archive is needed.

`verify_simple_reproduction.py` is an **optional** independent check of current features, fitted preprocessing, model results, ranks and dashboard totals. Its [latest result](../reports/simple_reproduction/verification.json) is included. It uses current raw/data/model/export files and `protected_checksums.json`; it does not depend on retired notebook copies.

`build_simple_notebooks.py` authors the notebook sources. Rebuilding clears their outputs, so execute the sequence again afterwards. `build_html_dashboard.py` and `build_final_report.py` regenerate the current dashboard/report from frozen results. `build_tableau_workbook.py` is retained unchanged with the Tableau workspace.

The `acquire_*` scripts, `audit_raw_data.py` and `finalize_raw_setup.py` preserve the source-acquisition history. The source files are already present; ordinary notebook runs need no downloads. The legacy setup publisher rewrites configuration and should not be used to rebuild the completed study. `improvement_exports.py` supports the saved historical export schema. Frozen machine-readable development, selection and evaluation records remain in `reports/`.
