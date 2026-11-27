# NSySS 2026 paper-50 revision: sequential experiment pipeline (fail-fast).
# Run from the repository root:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\run_all_experiments.ps1
# Optional: -IncludeTuning   (adds the six validation-only LR-sweep runs AFTER the baseline;
#                            brief says run tuning only if the time budget allows)
# Every phase aborts the whole pipeline on the first non-zero exit, like queue.py.
param(
    [switch]$IncludeTuning
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$Py = Join-Path $Root '.venv-carbon\Scripts\python.exe'
$Pkg = Join-Path $Root 'revision\gpu_package'
$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'

function Say($msg) { Write-Host ("[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $msg) }

if (-not (Test-Path $Py)) { throw "venv python not found: $Py" }
if (-not (Test-Path "$Pkg\config.json")) { throw "config.json missing in $Pkg (copy config.example.json and edit paths first)" }

Say "Phase 0: environment and data preflight"
& $Py -c "import torch, torchvision, timm, transformers, codecarbon, numpy, pandas, scipy, sklearn; print('torch', torch.__version__, 'torchvision', torchvision.__version__); print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0)); assert torch.cuda.is_available()"
if ($LASTEXITCODE -ne 0) { throw 'Preflight failed: CUDA stack check' }
$cfg = Get-Content "$Pkg\config.json" -Raw | ConvertFrom-Json
foreach ($d in $cfg.cache_dirs) { if (-not (Test-Path $d)) { throw "cache dir missing: $d" } }
if (-not (Test-Path (Join-Path $cfg.splits_dir 'train.csv'))) { throw "splits missing: $($cfg.splits_dir)" }
$cacheNpz = 0; foreach ($d in $cfg.cache_dirs) { $cacheNpz += (Get-ChildItem $d -Filter *.npz -File).Count }
Say "preflight ok: $cacheNpz cache npz files, splits at $($cfg.splits_dir), cohort '$($cfg.cohort)'"

Say "Phase 1: source validation (--validate-source)"
& $Py (Join-Path $Pkg 'run.py') --validate-source
if ($LASTEXITCODE -ne 0) { throw 'Phase 1 failed' }

Say "Phase 2: CPU package tests (test_package.py -v)"
& $Py (Join-Path $Pkg 'test_package.py') -v
if ($LASTEXITCODE -ne 0) { throw 'Phase 2 failed' }

Say "Phase 3: smoke queue (six configurations)"
& $Py (Join-Path $Pkg 'queue_jobs.py') (Join-Path $Pkg 'jobs.smoke.json') --config "$Pkg\config.json" --log-dir "logs_smoke_$Stamp" --resume
if ($LASTEXITCODE -ne 0) { throw 'Phase 3 failed: inspect logs_smoke_*; smoke/overfit outputs are diagnostics only' }
Say "INSPECT NOW: training_input_rgb.png + provenance.json of each smoke run in $($cfg.output_dir)\$($cfg.cohort)"
Say "Phase 4: weak-model overfit diagnostics (mobilenet, vivit; 100 steps)"
& $Py (Join-Path $Pkg 'run.py') --config "$Pkg\config.json" --model mobilenet --mode overfit --steps 100
if ($LASTEXITCODE -ne 0) { throw 'Phase 4 failed (mobilenet overfit)' }
& $Py (Join-Path $Pkg 'run.py') --config "$Pkg\config.json" --model vivit --mode overfit --steps 100
if ($LASTEXITCODE -ne 0) { throw 'Phase 4 failed (vivit overfit)' }

Say "Phase 5: baseline queue (5 models x 3 seeds; the long job)"
$sw = [Diagnostics.Stopwatch]::StartNew()
& $Py (Join-Path $Pkg 'queue_jobs.py') (Join-Path $Pkg 'jobs.baseline.json') --config "$Pkg\config.json" --log-dir "logs_baseline_$Stamp" --resume
if ($LASTEXITCODE -ne 0) { throw ('Phase 5 failed after {0:N1} min; inspect logs_baseline_*' -f $sw.Elapsed.TotalMinutes) }
Say ("baseline queue finished in {0:N1} min" -f $sw.Elapsed.TotalMinutes)

if ($IncludeTuning) {
    Say "Phase 5b: validation-only tuning queue (6 runs, separate energy accounting)"
    & $Py (Join-Path $Pkg 'queue_jobs.py') (Join-Path $Pkg 'jobs.tuning.json') --config "$Pkg\config.json" --log-dir "logs_tuning_$Stamp" --resume
    if ($LASTEXITCODE -ne 0) { throw 'Phase 5b failed: inspect logs_tuning_*' }
}

Say "Phase 6: fixed-fusion hybrid queue (3 seeds; after same-PC learned-gate baseline)"
& $Py (Join-Path $Pkg 'queue_jobs.py') (Join-Path $Pkg 'jobs.hybrid_fixed.json') --config "$Pkg\config.json" --log-dir "logs_hybrid_fixed_$Stamp" --resume
if ($LASTEXITCODE -ne 0) { throw 'Phase 6 failed: inspect logs_hybrid_fixed_*' }

Say "Phase 7: analysis"
$AnalysisOut = Join-Path $Root ("analysis_" + $cfg.cohort + "_$Stamp")
& $Py (Join-Path $Pkg 'analyze.py') (Join-Path $cfg.output_dir $cfg.cohort) --out $AnalysisOut
if ($LASTEXITCODE -ne 0) { throw 'Phase 7 failed' }

Say "ALL PHASES COMPLETE. Cohort output: $($cfg.output_dir)\$($cfg.cohort); analysis: $AnalysisOut"
Say "Reminder: do not mark the queue 'complete' if any job failed; partial runs are not independent seeds."
