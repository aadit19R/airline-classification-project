# Notebook 02 execution and fixes

**Historical debugging record:** counts and dates below describe the earlier 2025-only build. The revised executed notebooks and current counts are documented in [the preparation update](PREPARATION_TIMELINE_UPDATE.md).

Notebook: `notebooks/02_feature_engineering.ipynb`  
Run date: 18 October 2026  
Result: all 20 code cells completed in order from a fresh Python 3.13 kernel, with 13 inline figures and no saved error outputs or stderr warnings.

## Why the errors happened

| Issue | Cause | Fix |
|---|---|---|
| Earlier `ImportError` saving Parquet | pandas needs a separate Parquet engine; neither PyArrow nor fastparquet was available to the active kernel. An already-running kernel also needed its newly created user package directory added to the search path. | Installed PyArrow 20.0.0 for the user's Python 3.13 environment, added the enabled user package directory and explicitly selected PyArrow for the handoffs. |
| Notebook 02 could not find its starting data | Notebook 01's failed final save meant that `flights_clean.parquet` and the shared preparation manifest had not been created. | Generated the handoff using the existing notebook-01 preparation code. The user's existing notebook-01 plots and tables were preserved. |
| `Conflicting station/time reports` in the next cell | After removing identical overlapping source rows, six airport/timestamp pairs still had two different weather report versions. Some disagree on visibility, wind or thunderstorms. The code intentionally stopped because no treatment had been agreed. | Following the user's explicit choice, excluded both report versions at these six timestamps. Recorded the approved keys in `config/project.json` and the actual exclusions in the shared manifest. Any new conflicts still stop for review. |
| Timestamp join incompatibility, reproduced during debugging | In the installed pandas 3 environment, the parsed weather timestamps defaulted to microseconds while flight timestamps were stored in nanoseconds. `merge_asof` requires matching units even when both sides use UTC. | Explicitly normalized parsed weather timestamps to `datetime64[ns, UTC]`. A small reproduction confirmed the mismatch and the corrected join before the full run. |

The earlier code-simplification checks used synthetic data. They established that the refactor preserved those outputs, but did not encounter the real weather conflicts or reproduce the user's pandas 3 environment. The fresh-kernel execution now covers the actual project inputs. The requirements range now admits the verified pandas 3 installation without downgrading the user's environment.

## Weather decision and its effect

The 36 raw source batches contain 34,915 in-scope weather rows. The audit removed 3,816 identical overlapping payloads, then excluded 12 conflicting report versions at six timestamps, leaving 31,087 unambiguous reports.

| Airport | Observation time (UTC) | Report versions excluded |
|---|---|---:|
| ATL | 2025-06-14 23:40 | 2 |
| DFW | 2025-05-06 16:50 | 2 |
| DFW | 2025-12-23 14:50 | 2 |
| ORD | 2025-07-20 09:10 | 2 |
| ORD | 2025-11-30 14:20 | 2 |
| ORD | 2025-12-28 16:50 | 2 |

Comparing matches using all distinct observation keys against matches excluding the six ambiguous keys identifies 190 affected flights: ATL 5, DFW 33 and ORD 152. Every affected flight has an earlier report within the existing age limit. No flight was removed. Source files remain unchanged.

All 938,417 eligible flights match a qualifying observation in the completed output. Ages at cutoff range from 20 to 139 minutes, within the declared 20–180 minute rule. Report coverage does not imply complete fields: gusts, condition codes and ceiling values remain missing where the source does not establish a value. Their existing missingness flags and train-fitted imputation rules remain in place.

## Completed checks and outputs

- Saved `data/processed/features.parquet`: 938,417 rows, retaining the clean handoff's exact eligible flight IDs, order and target labels.
- Updated `reports/preparation_manifest.json`, preserving its cleaning section and recording feature definitions, treatment counts, sample membership fingerprint and input/output SHA-256 checksums.
- Schedule set A has 9 predictors; expanded set B has 36. Both use identical sample membership.
- Population/sample sizes: training 618,942 / 20,000; validation 165,111 / 5,000; test 154,364 / 10,000.
- Preprocessing fitted on the training sample only. Training and validation previews both contain 302 transformed columns. The transformed matrices are not separately exported.
- No infinities among the nonmissing numerical predictors; unique IDs and cutoff/history assertions passed.
- Independent direct-filter counts agree with all three history aggregations and scheduled-traffic counts on 40 fixed-seed real flights.
- Verified the saved input/output checksums and reloaded feature-table IDs, targets and sample counts.
- Preserved all 13 figure blocks. Shortened long histogram/correlation labels after visual inspection to keep the figures readable.
- Created no intermediate CSV datasets. No classifiers are fitted by this notebook; model evaluation and Tableau findings remain the next project stage.

This remains the agreed historical simulation: the sources do not establish actual weather/outcome delivery times or historical schedule revision availability. The fixes do not change that study design.

## Verified environment

Python 3.13.6; pandas 3.0.3; NumPy 2.5.1; PyArrow 20.0.0; scikit-learn 1.9.0; Matplotlib 3.11.1; seaborn 0.13.2; nbclient 0.10.2; nbformat 5.10.4.
