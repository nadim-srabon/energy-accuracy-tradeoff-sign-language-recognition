# Local GPU-PC setup notes (NSySS 2026 paper 50 revision)

Recorded 2026-11-18 while preparing the copied repository
`quantifying_carbon_cost_in_slr_models` on the RTX A4000 workstation.
Every setup decision with a scientific implication is listed here. Nothing in
this file marks GPU validation as complete.

## Environment (new, isolated; existing environments untouched)

- New venv `\.venv-carbon` (repo root), Python 3.13.17.
  - Deviation from gpu_package README ("Python 3.11 or 3.12") and from the
    historical Kaggle environment (Python 3.12.13, paper Table 7). The machine
    has only Python 3.13.17; this is part of the new software cohort and is
    captured in each run's `provenance.json` / `pip-freeze.txt`.
- `torch==2.10.0+cu128`, `torchvision==0.25.0+cu128` — the historical
  torch/torchvision pair (paper Table 7: PyTorch 2.10.0, CUDA 12.8), verified
  working on the A4000 (sm_86), CUDA available, matmul on device OK.
- `revision/gpu_package/requirements.txt` installed unchanged (codecarbon,
  timm, transformers, thop, etc.). Additional setup-only package: `av`
  (PyAV, for video decoding in the cache builder only — not a training
  dependency), plus `pillow` (already a torchvision dependency) for resizing.
- Hardware: 1x NVIDIA RTX A4000 16 GB, 20 logical cores, 32 GB RAM, Windows 11.
  The historical runs reported 2x Tesla T4 on Kaggle; the adapter uses ONE
  device (gpu_id 0), no DataParallel. This is the defining hardware-cohort
  difference; no pooling of energy across cohorts.

## Data

- Official ASL Citizen split CSVs (`train.csv`, `val.csv`, `test.csv`,
  columns: Participant ID, Video file, Gloss, ASL-LEX Code) downloaded
  2026-11-18 from the official Microsoft download zip
  (`ASL_Citizen.zip`, SHA-visible remote size 45,924,134,223 bytes; only the
  split CSVs + use.txt license were fetched, via HTTP range requests) to
  `D:\asl_citizen_download\meta`. 83,399 rows, 2,731 glosses — matches the
  paper. Explicit `Participant ID` column present (runner requirement).
- The ORIGINAL appearance-cache shards (`/kaggle/input/notebooks/.../
  feature_cache_shard_{0,1,2}`, built by teammates' notebooks
  asl-shared-cache-building-id00/01/02) are NOT on this PC and are not
  publicly downloadable without Kaggle credentials. **Action available to the
  authors: download the three shard folders from the Kaggle Output tabs
  (login required) and re-point `config.json` at them; that restores the
  original cache.**
- Until then, a REBUILT cache is produced by
  `revision/local_setup/build_cache_rebuilt.py` from the official videos:
  - Only the 100-class pilot's videos (3,544 clips, ~1.8 GiB) were extracted
    from the official zip by the same range-request method, CRC-verified,
    manifest in `D:\asl_citizen_download\meta\extraction_manifest.json`.
  - Class selection: top-100 glosses by total row count across the three
    official split CSVs (`D:\asl_citizen_download\meta\top100_glosses.json`),
    matching the notebooks' "most_common over matched rows" rule assuming
    complete cache coverage. `expected_manifest_sha256` is deliberately NOT
    set in config.json (no verified reference manifest exists locally);
    every run records the manifest hash it used.
  - Cache layout: one .npz per clip, `appearance` uint8 (16, 1, 224, 224, 3).
    Documented deviations from the unknown original: full frame (no signer
    bbox crop), streams=1, uniform 16-frame sampling, PIL-bilinear 224x224.
  - Location: `D:\asl_cache_rebuilt\cache`, manifest
    `D:\asl_cache_rebuilt\cache_manifest.json`.
  - Scientific implication: accuracies are NOT comparable clip-for-clip with
    the historical Kaggle numbers; this cohort stands alone (its own
    manifest hash, cohort name, and energy accounting).

## Cohort

- `config.json` cohort: `local_pc_a4000_rebuiltcache_v1` (output root
  `D:/nsyss50_results`). Distinct from any historical/Kaggle cohort by name;
  the analyzer separates cohorts by (cohort, manifest hash, lr).

## Historical checkpoint evaluation (G1) status

- No best-checkpoint files, original `label_map.json` files, or
  `green_metrics_*.json` model cards exist on this PC (status.md already
  recorded this). Checkpoint-evaluation mode is therefore unusable until the
  authors supply trusted checkpoints + original class-index maps from Kaggle.
- The paper's archived result tables (from `Group_9_extracted.zip`,
  `metadata_results/`) are kept for comparison against the NEW cohort, never
  pooled with its energy.

## Prefetch (outside measured windows)

- `revision/local_setup/prefetch_weights.py` warms the torchvision/timm/HF
  caches with the exact identities extracted from the notebook sources:
  MobileNet_V3_Small IMAGENET1K_V1, ResNet50 IMAGENET1K_V2 (note the V2 —
  a recipe fact the paper should disclose), timm vit_tiny_patch16_224,
  facebook/timesformer-base-finetuned-k400, ConvNeXt_Tiny + Swin_T V1.
- All downloads/installs on 2026-11-18 happened before any measured run.

## Runtime defect found and fixed (2026-11-18; logged per the brief)

- **Defect:** `revision/gpu_package/queue.py` shadowed Python's standard-library
  `queue` module. With the script directory first on `sys.path`, any
  `import queue` performed inside scipy/pandas/sklearn (and other) import
  graphs resolved to the package's `queue.py`, whose module-level
  `parse_args()` then raised
  (`the following arguments are required: jobs`), crashing the import. First
  observed as 2 errors in `python test_package.py -v` on this PC; the same
  mechanism would have crashed every real `run.py` invocation at notebook
  import time. The planning workspace's CPU checks did not catch it.
- **Fix:** renamed to `queue_jobs.py` (no stdlib module of that name); all
  references updated (`run_all_experiments.ps1`, `gpu_package/README.md`).
  No scientific implication: process-queueing plumbing only.
- `test_package.py -v` now passes on this PC (3/3) — first run on this machine;
  previously "passed" only on the planning workspace.

## Resumability changes (2026-11-18, requested by the authors; logged per the brief)

- `run.py`: the executed training cell now calls an injected `_dump_epoch`
  once per epoch (patched at the unique anchor
  `is_best = val_metrics["top1"] > best_val_top1`, verified exactly once in
  each of the five notebooks). It appends one JSON line per epoch
  (loss/top-1/energy/best-val-so-far) to `epoch_log.jsonl` in the run
  directory. **Scientific implications:** no change to architecture,
  preprocessing, optimizer, precision, batch size, or the energy-measurement
  policy; the ~1 KB/epoch file write happens inside the measured window and is
  disclosed here. Best checkpoints were ALREADY saved to disk at every
  validation improvement (`best_model.pt`, one per run/seed directory).
- `queue.py --resume` (used by `run_all_experiments.ps1` for every queue):
  jobs whose output directory holds the completion marker
  (`result.json` / `diagnostic_only.json` / `validation_only.json`) are
  skipped; incomplete directories are MOVED ASIDE to
  `<run_id>.incomplete_<timestamp>` (failed runs stay visible and are never
  overwritten) and rerun FROM EPOCH 1.
- **Mid-run energy resume is deliberately NOT implemented**, following the
  notebook design recorded in the repo: resuming with a fresh CodeCarbon
  tracker would undercount total training energy, which is a headline
  measurement of this study. A crashed 3-hour run is therefore retrained from
  scratch; its salvaged `epoch_log.jsonl` quantifies the wasted energy so it
  can be reported separately, as the brief requires.

## Runner script

- `run_all_experiments.ps1` (repo root): preflight -> validate-source ->
  CPU tests -> smoke queue (6 configs) -> mobilenet/vivit overfit (100
  steps) -> baseline queue (5x3 seeds) -> [-IncludeTuning adds the
  validation-only LR sweep] -> fixed-fusion hybrid queue (3 seeds) ->
  analyze.py. Fail-fast at every phase; log dirs carry a timestamp suffix so
  reruns never overwrite refused directories. Smoke/overfit outputs are
  diagnostics and never enter scientific tables.
