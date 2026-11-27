# A4000 experiment code

All A4000 runs execute the model notebooks at the repository root: each runner extracts the notebook
cells (configuration, data pipeline, model, training loop, early stopping, energy tracking) and runs
them unchanged in a separate process per model and seed. Every run checks that the notebooks are
byte-identical to the reference versions (SHA-256) and that the data selection matches the pinned
manifest hash, and records the runner's own SHA-256 in `provenance.json`.

| Campaign | Package | Cohort id (in every `result.json`) | Runs | Paper |
|---|---|---|---|---|
| First (18–19 Nov 2026) | `gpu_package/run.py` | `local_pc_a4000_rebuiltcache_v1` | 18 (5 configurations + fixed fusion, seeds 42/123/2024) | Tables 2–3, Figs. 2–3 |
| Extension (22–24 Nov 2026) | `gpu_package_ext/run_ext.py` | `local_pc_a4000_rebuiltcache_v1` | 27 (seeds 7/999 of the six; 3 added models × 5 seeds) | Tables 1–3, Figs. 2–3 |
| Data-loading control (25–26 Nov 2026) | `gpu_package_nw4/run_ext.py` | `local_pc_a4000_numworkers4_v1` | 10 (ResNet-50, TimeSformer × 5 seeds) | Section 4.4 |

The first two campaigns form one pooled cohort (same host, packages, notebooks and data manifest,
verified before every run). The control cohort is reported side by side and never pooled.

* `local_setup/`: `build_cache_rebuilt.py` builds the 16-frame, 224×224 appearance cache from the
  official ASL Citizen videos; `prefetch_weights.py` downloads the pretrained weights of the five original
  models before any measured run; `NOTES.md` records setup decisions and deviations.
* Runner SHA-256 (as recorded in the run provenance): `gpu_package/run.py` 85789f86… (LF line endings),
  `gpu_package_ext/run_ext.py` eb9a5baf…, `gpu_package_nw4/run_ext.py` f7dfe3a7….

Paths in the `config*.json` files are those of the original machine (`D:/...`); edit them before rerunning.
Line endings: git stores text files with LF line endings. `gpu_package_ext/run_ext.py` was executed
with Windows (CRLF) line endings, which is the version whose SHA-256 (`eb9a5baf…`) is recorded in the
run provenance; the checked-out LF copy hashes to `ec7d4fbe…` and differs only in line endings
(verify with `tr -d '\r'` or by converting to CRLF). The same applies to `gpu_package_ext/SHA256SUMS.txt`.
