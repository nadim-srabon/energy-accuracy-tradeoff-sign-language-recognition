# Data-loading control (paper Sec. 4.4)

Cohort `local_pc_a4000_numworkers4_v1`: ResNet-50 and TimeSformer retrained at seeds
42/123/2024/7/999 on the same RTX A4000 host, data manifest
(`62334c35...`), notebooks, packages and recipe as the main cohort (`../a4000_cohort/`),
with one change: the DataLoader uses `num_workers=4` instead of 0. Run 25-26 Nov 2026
(15:18 to 05:07), runner `revision/gpu_package_nw4/run_ext.py` (sha256 `f7dfe3a7...`).
This cohort is never pooled with the main cohort; it is compared side by side only.

* `runs/<model>_baseline_seed<seed>_train/`: `result.json`, `training_history.json`,
  `emissions.csv` (longitude/latitude removed), `provenance.json`, `signer_audit.json` (counts only),
  and `nw4nbmod_<model>.py`
  (the notebook's own dataset/loader definitions written to an importable module so that
  Windows worker processes can load them; data processing is unchanged).
* `summary.csv`, `seed_results.csv`, `checks.txt`: analyzer output over all three
  campaigns (13 configuration keys x 5 seeds), cohort-keyed.
* `config.nw4.json`, `jobs.nw4.json`: run configuration and job list.
* `derive_nw4.py runs ../a4000_cohort/runs`: every number in the paper's Sec. 4.4.

Notes. A first attempt (25 Nov, 02:06) gave all five seeds the same initialization because
the seed did not reach the worker module; its 10 runs were quarantined on the GPU PC
(`D:\nsyss50_results_nw4\quarantined_seedflat_20261125\`) and are not used. In the
runs here, the seeds produce different models (pairwise top-1 prediction agreement
0.46-0.62). CPU power was higher for the runs started 17:50-23:11 on 25 Nov (seeds 123
and 2024 of both models), adding about 10% to those runs' energy. Inference with four
workers was slower than single-process inference for both models (worker start-up in
each pass), so the paper keeps the main cohort's inference measurements.
