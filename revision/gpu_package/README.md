# Transfer and run on the GPU PC

This package adapts the repository's five notebook architectures and original single-seed training loop. It adds isolated seed output directories, explicit GPU measurement scope, strict data checks, per-checkpoint inference, and result aggregation. It has **not been GPU-validated on your target PC**. Run smoke checks before committing the experiment budget. The scientific validity/claim checklist is in `CLAIM_EVIDENCE_LEDGER.md`.

## 1. Transfer prerequisites

- Extract `nsyss50_gpu_package.zip` to an empty directory. Verify `SHA256.json` if transferring through another system.
- Transfer all three original `.npz` appearance-cache shards and original `train.csv`, `val.csv`, `test.csv`; access is external to this repository. Do not rebuild a different cache silently. Required CSV fields: `Video file`, `Gloss`, and an explicit participant/signer ID column. The runner rejects filename-only signer inference.
- Original cache layout is uint8 `appearance` with `(16, streams, H, W, 3)`. TimeSformer uses the original eight-frame subsampling and its processor normalization.
- For historical evaluation: transfer best checkpoint(s), the ORIGINAL index-to-gloss JSON (`{"0": "gloss", ...}`), original selected clips/splits, and model-card JSON/raw logs. **Never invent a label map to make a checkpoint load.** If missing, choose fresh training instead. Notebook extra-seed checkpoints may never have been persisted.
- Internet access is required on first use to fetch pretrained torchvision/timm/Hugging Face weights. Download once before timed jobs where possible; preserve caches and exact model identities. Model initialization/download is outside the original training-energy boundary.

## 2. Environment

Use a clean Python environment (Python 3.11 or 3.12), with an NVIDIA CUDA-capable GPU and supported driver. Use the official PyTorch selector for a matched `torch`/`torchvision` CUDA build: https://pytorch.org/get-started/locally/. Prefer the captured historical torch 2.10.0 / torchvision 0.25.0 stack if compatible and available; otherwise label a new software cohort and freeze it after smoke validation. Do not blindly install CPU-only torch.

PowerShell example:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# Install matching CUDA torch + torchvision using the official selector, then:
python -m pip install -r requirements.txt
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
Copy-Item config.example.json config.json
```

Edit `config.json`: actual cache paths, split directory, output root, descriptive cohort name, GPU ID. `num_workers=0` is intentionally fixed for Windows portability; report it as a local-PC recipe/environment difference. Notebook batch size 8, FP32, frames, freezing, sampler and early-stopping settings are preserved. Do not enable AMP or change batch size just to finish faster without creating a separate cohort. Dependency ranges are bootstrap constraints, **not a historical lock file**; each run saves `pip-freeze.txt`.

The tracker uses one specified GPU with machine-level CPU/RAM accounting and 1 s sampling. Use the host exclusively; close other GPU tasks. Record power limits, actual hardware, CPU/RAM method, PUE and CodeCarbon logs. This new tracking policy differs from old notebook defaults, so use a distinct cohort. Never pool old Kaggle energy values with local-PC energy.

## 3. Validate, then choose jobs

```powershell
python run.py --validate-source
python queue_jobs.py jobs.smoke.json --config config.json --log-dir logs_smoke
python run.py --config config.json --model mobilenet --mode overfit --steps 100
python run.py --config config.json --model vivit --mode overfit --steps 100
```

Inspect `training_input_rgb.png` for correct RGB/crop/stream, the trainable parameter list, losses, selected manifest and signer audit. A finite backward pass is an implementation check. Overfit should show a material decline in loss and high training-batch accuracy; dropout/label smoothing prevent literal zero loss. If this fails, investigate before expensive training; do not conclude the architecture is invalid based on one diagnostic.

Each output directory refuses overwrites. For reruns choose a new cohort/output root and document why. Failed runs remain visibly incomplete. There is no partial-epoch energy resume; restart a failed scientific run and report wasted experiment energy separately.

Checkpoint evaluation example:

```powershell
python run.py --config config.json --model resnet --mode evaluate --seed 42 --checkpoint D:/old/resnet42/best_model.pt --checkpoint-label-map D:/old/resnet42/label_map.json
```

The class map must exactly match current selection. Compare `selected_manifest.json` with the original manifest too; set optional `expected_manifest_sha256` in config after obtaining a verified reference hash. Label-map equality alone cannot certify identical historical test sets. Only load checkpoints from trusted project sources; the loader uses `weights_only=True` and strict state dict matching.

Fresh three-seed cohort, only if needed and budgeted:

```powershell
python queue_jobs.py jobs.baseline.json --config config.json --log-dir logs_baseline
```

The queue is sequential, saves stdout/stderr, and stops on error. It does not impose a 24 h deadline: use measured epoch times and stop adding runs at the freeze point. Do not run the full baseline and hybrid menus by default if the timing budget cannot cover both. Pending jobs can be removed from a copied job JSON, with the change recorded.

Validation-only tuning example (separate cohort recommended):

```powershell
python run.py --config config.json --model mobilenet --mode train --seed 42 --lr 0.0003 --validation-only
python run.py --config config.json --model vivit --mode train --seed 42 --lr 0.0003 --validation-only
```

Use baseline 1e-4 and candidates 3e-4/1e-3 specified before launch. Compare validation curves only. The validation-only run saves a checkpoint/history/energy and does not evaluate the test set. After selecting the recipe, evaluate that checkpoint once and repeat the selected recipe on seeds 123/2024 if possible. Its evaluation-only `train_kwh` is intentionally missing; join original training history energy with checkpoint hash manually in a reviewed analysis if needed. Do not impute energy or tune on returned test scores.

`jobs.tuning.json` contains this six-run validation-only sweep. Use it only after checking the timing budget, with a dedicated tuning cohort in a copied config.

Hybrid ablation, after same-PC baseline completion:

```powershell
python queue_jobs.py jobs.hybrid_fixed.json --config config.json --log-dir logs_hybrid_fixed
```

`fixed` replaces the gate with 0.5/0.5, freezes unused gate parameters, and preserves auxiliary CE loss, backbones and original budget. Training is from scratch. Each hybrid evaluation additionally records auxiliary-head accuracies and a checkpoint intervention using equal feature fusion; those diagnostics are NOT independently trained branch-only baselines. The CNN/Swin branches still both run during normal inference profiling. No energy advantage for branch pruning is measured by these interventions.

## 4. Return results and analyze

Every scientific run saves `provenance.json`, `label_map.json`, `selected_manifest.json`, `signer_audit.json`, environment freeze, raw `emissions.csv`, `result.json`, `predictions.npz`, and (training only) best checkpoint/history. Each seed has its own best checkpoint. Accuracy uses one deterministic test pass; energy uses three warmed full-test-loop windows of at least 60 seconds, with exact clip counts and CUDA synchronization. Timing is batch-amortized end-to-end time including data loading, **not batch-1 request latency**. Check raw logs and variability before accepting energy values.

```powershell
python analyze.py D:/nsyss50_results/local_pc_original_recipe_v1 --out analysis_baseline
```

The analyzer rejects duplicate seeds, separates data/recipe/hardware cohort keys, computes SD and t CIs over seeds, creates a descriptive mean Pareto table/plot, and produces illustrative intensity/volume scenarios. It does not run significance tests or certify model convergence. Negative CI bounds remain visible; do not silently clip them. Independent replication and small-sample assumptions need interpretation. Check plot labels visually before publication. Scenario intensities 50/200/412.72/800 are illustrative, not verified named regions.

Return the entire cohort folder, queue logs, analysis outputs and config, plus a human note with machine location (if known), failures, changes and GPU exclusivity. Checkpoints may be transferred separately if large. Preserve source JSON/CSV alongside the final paper, and document preprocessing, external pretraining and embodied/facility costs omitted from operational accounting.

CodeCarbon source: https://docs.codecarbon.io/3.3/explanation/methodology/ and https://docs.codecarbon.io/latest/explanation/power-estimation/ . GPU telemetry is combined with CPU/RAM measurement or estimation; the tool does not directly measure carbon or complete facility energy.
