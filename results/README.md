# Results

| Folder | Content | Paper |
|---|---|---|
| `a4000_cohort/` | 45 training runs (9 configurations × 5 seeds), analyzer outputs, and the scripts that recompute every number (`derive5.py`), table row (`make_rows5.py`) and data figure (`figs5.py`). See its README. | Tables 1–3, Figs. 2–3, Secs. 4.1–4.3, 4.5 |
| `a4000_nw4_control/` | 10 runs of the data-loading control (`num_workers = 4`) and `derive_nw4.py`. | Sec. 4.4 |
| `t4_cohort/` | Earlier Kaggle T4 cohort (five original configurations, seeds 42/123/2024): tables and the aggregated result bundle. | Table 4, Sec. 4.6 |
| `t4_supplementary/` | Exploratory single-seed T4 runs (learning-rate and recipe variants for the light models; learned-gate reruns). | Sec. 5 |

Per-run folders (`runs/<model>_<variant>_seed<seed>_train/`) contain:

* `result.json`: test top-1/top-5, training energy, three inference windows (clips, seconds, kWh,
  ms/clip, peak memory) and, for the hybrid, gate statistics and auxiliary-head diagnostics;
* `training_history.json`: per-epoch loss, accuracy and energy;
* `emissions.csv`: raw CodeCarbon 3.3.2 log (longitude/latitude columns removed);
* `provenance.json`: configuration, data-manifest hash, notebook and runner SHA-256, hardware, parameters;
* `signer_audit.json`: signers per split and pairwise overlap (zero); participant IDs removed.

Model checkpoints and test predictions are not included.
