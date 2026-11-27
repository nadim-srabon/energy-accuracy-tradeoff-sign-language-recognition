# Extension campaign: two more seeds and one more model per family

Runs on the same RTX A4000 host, Python environment, rebuilt cache and 100-class selection as the first
campaign (`../gpu_package/`), so its runs pool with the first 18 into one cohort
(`local_pc_a4000_rebuiltcache_v1`, five seeds per configuration).

| Part | Runs | Seeds |
|---|---|---|
| Existing six configurations (MobileNetV3-Small, ResNet-50, ViViT-Lite, TimeSformer, hybrid learned gate, hybrid fixed 50:50) | 12 | 7, 999 |
| Added models: R(2+1)D-18 (CNN), Video Swin-T (Transformer), MobileViT-S (hybrid) | 15 | 42, 123, 2024, 7, 999 |

## Added models

| Family | Model | Weights | Why |
|---|---|---|---|
| CNN | R(2+1)D-18 (31.4M) | torchvision, Kinetics-400 | Video-pretrained CNN, to separate "CNN vs Transformer" from "image vs video pretraining". |
| Transformer | Video Swin-T (27.9M) | torchvision, Kinetics-400 | Same pretraining data and similar size as R(2+1)D-18: a matched CNN–Transformer pair. |
| Hybrid | MobileViT-S (5.0M) | timm `mobilevit_s.cvnets_in1k`, ImageNet-1k | A light, single-backbone hybrid without a fusion gate. |

Each added model reuses one existing notebook's pipeline (data, sampler, batch 8, FP32, AdamW 1e-4,
≤30 epochs with early stopping, energy tracking, inference windows); only the architecture cell differs
(`models_extra/*.py`). The last stage is trainable, and inputs use each model's own pretraining
normalization. Pretraining inputs differ from the study's common input of 16 frames at 224 px:
R(2+1)D-18 was pretrained on 16 frames at 112 px, Video Swin-T on 32 frames at 224 px, MobileViT-S at 256 px.

## Files

| File | Purpose |
|---|---|
| `run_all_ext.ps1` | Driver. Phases: `check`, `newseeds`, `newmodels`, `analyze`, `collect`, `experiments`. |
| `run_ext.py` | Runner: identical to `../gpu_package/run.py` for the five original models, plus the three added models. |
| `models_extra/*.py` | Architecture cells of the added models. |
| `notebooks/` | Copies of the five notebooks (SHA-256 checked at every run). |
| `queue_ext.py` | Sequential queue with logs, fail-fast and `--resume`. |
| `jobs.*.json` | Job lists: smoke (9), overfit (3), newseeds (12), newmodels (15). |
| `config.ext.json` | Paths, cohort, pinned data-manifest hash, location of the first campaign's runs. |
| `preflight_ext.py` | Checks packages (against the first runs' `pip freeze`), GPU, notebook hashes, data, disk, planned runs. |
| `prefetch_ext.py` | Downloads/verifies all pretrained weights. |
| `test_ext.py`, `show_diagnostics.py`, `make_diag_config.py` | CPU tests and diagnostics helpers. |
| `analyze_ext.py` | Pools several cohort folders and writes summary statistics (mean, SD, 95% t-interval), Pareto front, crossovers and scenarios. |

## Running

```powershell
# from this folder, with the project's virtual environment at ..\..\.venv-carbon
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_all_ext.ps1 -Phase check        # preflight, weights, smoke, overfit
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_all_ext.ps1 -Phase experiments  # 27 runs, analysis, results zip
```

Interrupted queues resume with the same command: finished runs are skipped, and an interrupted run is
moved aside and restarted from epoch 1, so training energy is never under-counted. Edit the paths in
`config.ext.json` first.
