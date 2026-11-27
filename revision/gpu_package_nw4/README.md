# Data-loading control (num_workers = 4)

Tests whether the training-energy ordering between ResNet-50 and TimeSformer depends on single-process
data loading (paper, Section 4.4). Both models are retrained at seeds 42, 123, 2024, 7 and 999 with four
DataLoader worker processes; everything else (host, packages, notebooks, data manifest, recipe) is
identical to the main A4000 cohort. Results form a separate cohort, `local_pc_a4000_numworkers4_v1`,
compared side by side and never pooled.

## How it differs from `../gpu_package_ext/run_ext.py`

1. `cfg.NUM_WORKERS = 4` (the main cohort uses 0).
2. On Windows, DataLoader workers are spawned processes that must import the dataset class. Classes
   executed into `__main__` cannot be pickled, so the runner writes the notebook's own imports, CONFIG and
   dataset definitions into an importable module (`nw4nbmod_<model>.py`, saved in every run folder) before
   the dataset cell runs. Data processing is unchanged; the notebooks stay byte-identical (SHA-256 checked).

## Files

| File | Purpose |
|---|---|
| `run_all_nw4.ps1` | Driver: `-Phase check` (preflight + one smoke run per model), `-Phase run` (10 runs, resumable). |
| `run_ext.py` | Control runner (SHA-256 `f7dfe3a7…`, recorded in every run's provenance). |
| `preflight_nw4.py`, `queue_ext.py` | Preflight checks and sequential queue. |
| `jobs.nw4.json`, `jobs.smoke_nw4.json`, `config.nw4.json` | Job lists and configuration. |
| `models_extra/`, `notebooks/` | Same files as in `../gpu_package_ext/` (notebooks hash-checked). |

## Note on a discarded first attempt

The first attempt (25 Nov 2026, 02:06) gave all five seeds the same initialization, because the seed did
not reach the worker module. Its runs were quarantined and are not used; the runner was fixed (seed
handling only) before the 10 reported runs, whose seeds produce different models. Results:
`../../results/a4000_nw4_control/`.
