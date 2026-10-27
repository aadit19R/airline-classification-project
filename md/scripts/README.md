# Optional project tools

The core acquisition, cleaning, feature engineering, training and evaluation code is in the [four notebooks](../notebooks/simple/README.md). They have no project-script imports.

| Tool | Purpose |
|---|---|
| `run_simple_notebooks.py` | Run the four notebooks in fresh kernels and record execution |
| `verify_simple_reproduction.py` | Compare notebook outputs with pre-refactor feature fingerprints, frozen models and published results |
| `build_html_dashboard.py` | Build the standalone HTML dashboard from frozen evaluation exports |
| `check_html_dashboard.mjs` | Check dashboard calculations and interactions |
| `build_final_report.py` | Regenerate the published report from existing handoffs and exports |
| `audit_raw_data.py` | Structural raw-source audit |
| `improvement_exports.py` | Export formatting for the earlier Q2 experiment |
| `finalize_raw_setup.py` | Historical setup publisher; it rewrites configuration and is not part of the current notebook run |

Notebook 01 has one optional acquisition section for all source months. Normal runs reuse local raw files; set `DOWNLOAD_MISSING=True` there only when sources need downloading.
