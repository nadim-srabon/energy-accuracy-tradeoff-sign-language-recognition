<#
.SYNOPSIS
  Extension campaign for NSysS '26 paper 50 on the RTX A4000 PC.
  (1) two new seeds (7, 999) for the six configurations already run, and
  (2) one new model per family (R(2+1)D-18, Video Swin-T, MobileViT-S) x 5 seeds.

.DESCRIPTION
  Phases (run them in this order; every phase stops at the first failure):
    check        preflight + weight download + source/CPU tests + smoke + overfit  (~20-40 min)
    newseeds     12 runs: 6 existing configurations x seeds 7, 999                  (~17 h)
    newmodels    15 runs: 3 new models x seeds 42, 123, 2024, 7, 999                (~15-25 h)
    analyze      pool with the first 18 runs and write the 5-seed summary
    collect      zip everything except checkpoints for transfer back
    experiments  newseeds + newmodels + analyze + collect in one go
  Queues use --resume: rerunning a phase skips finished runs and restarts an
  interrupted run from epoch 1 (its partial folder is kept, renamed .incomplete_*).

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File .\run_all_ext.ps1 -Phase check
  powershell -NoProfile -ExecutionPolicy Bypass -File .\run_all_ext.ps1 -Phase experiments
#>
param(
    [ValidateSet('check', 'newseeds', 'newmodels', 'analyze', 'collect', 'experiments')]
    [string]$Phase = 'check',
    [string]$Python = ''
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
New-Item -ItemType Directory -Force -Path 'logs_ext' | Out-Null
Start-Transcript -Path "logs_ext\driver_${Phase}_$stamp.log" | Out-Null

if (-not $Python) {
    $repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
    $Python = Join-Path $repo '.venv-carbon\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $Python)) {
    Stop-Transcript | Out-Null
    throw "Python not found at $Python. Run with -Python <full path to .venv-carbon\Scripts\python.exe>"
}
Write-Host "Python: $Python"

# Keep the PC awake while this script runs (released automatically when it exits).
Add-Type -Namespace Nsyss -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[void][Nsyss.Power]::SetThreadExecutionState([uint32]2147483649)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

$cfg = Get-Content -Raw -LiteralPath 'config.ext.json' | ConvertFrom-Json
$extRoot = (Join-Path $cfg.output_dir $cfg.cohort).Replace('/', '\')
$refRoot = ($cfg.reference_results_dir).Replace('/', '\')
$diagBase = (($cfg.output_dir.TrimEnd('/', '\')) + '_diagnostics').Replace('/', '\')

function Invoke-Py {
    param([string]$Name, [string[]]$Arguments)
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $Python @Arguments
    $code = $LASTEXITCODE
    $ErrorActionPreference = $old
    if ($code -ne 0) {
        Stop-Transcript | Out-Null
        throw "$Name FAILED (exit $code). Read the message above and the queue log in logs_ext\. Nothing after this step was run."
    }
}

function Run-Check {
    Invoke-Py 'Preflight' @('-u', 'preflight_ext.py', '--config', 'config.ext.json')
    Invoke-Py 'Download / verify pretrained weights' @('-u', 'prefetch_ext.py')
    Invoke-Py 'Source validation (8 models)' @('-u', 'run_ext.py', '--validate-source')
    Invoke-Py 'CPU tests' @('-u', 'test_ext.py')
    Invoke-Py 'Diagnostics config' @('make_diag_config.py', 'config.ext.json', 'config.diag.json', $stamp)
    Invoke-Py 'Smoke tests (9 configurations)' @('-u', 'queue_ext.py', 'jobs.smoke_ext.json', '--config', 'config.diag.json', '--log-dir', "logs_ext\smoke_$stamp")
    Invoke-Py 'Overfit diagnostics (3 new models, 100 steps)' @('-u', 'queue_ext.py', 'jobs.overfit_ext.json', '--config', 'config.diag.json', '--log-dir', "logs_ext\overfit_$stamp")
    Invoke-Py 'Diagnostics summary' @('show_diagnostics.py', (Join-Path $diagBase "diagnostics_$stamp"))
    Write-Host "`nCHECK PHASE PASSED. Look at the table above and one training_input_rgb.png, then run -Phase experiments." -ForegroundColor Green
}

function Run-Queue([string]$Jobs, [string]$Tag) {
    Invoke-Py "Preflight ($Tag)" @('-u', 'preflight_ext.py', '--config', 'config.ext.json')
    Invoke-Py "Queue $Tag" @('-u', 'queue_ext.py', $Jobs, '--config', 'config.ext.json', '--log-dir', "logs_ext\${Tag}_$stamp", '--resume')
}

function Run-Analyze {
    $out = "analysis_ext_$stamp"
    Invoke-Py 'Analysis (first 18 runs + extension)' @('-u', 'analyze_ext.py', $refRoot, $extRoot, '--out', $out)
    Write-Host "Analysis folder: $(Join-Path $PSScriptRoot $out)" -ForegroundColor Green
}

function Run-Collect {
    $stage = Join-Path $env:TEMP "nsyss50_ext_$stamp"
    New-Item -ItemType Directory -Force -Path $stage | Out-Null
    # Run folders without checkpoints (*.pt are large; keep them on the GPU PC).
    robocopy $extRoot (Join-Path $stage 'runs') /E /XF *.pt /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy of $extRoot failed ($LASTEXITCODE)" }
    if (Test-Path -LiteralPath $diagBase) {
        robocopy $diagBase (Join-Path $stage 'diagnostics') /E /XF *.pt /NFL /NDL /NJH /NJS /NP | Out-Null
        if ($LASTEXITCODE -ge 8) { throw "robocopy of $diagBase failed ($LASTEXITCODE)" }
    }
    $global:LASTEXITCODE = 0
    Copy-Item -Recurse -Path 'logs_ext' -Destination (Join-Path $stage 'logs_ext')
    Get-ChildItem -Directory -Filter 'analysis_ext_*' | ForEach-Object {
        Copy-Item -Recurse -Path $_.FullName -Destination (Join-Path $stage $_.Name)
    }
    foreach ($f in @('config.ext.json', 'preflight_report.json')) {
        if (Test-Path -LiteralPath $f) { Copy-Item -LiteralPath $f -Destination $stage }
    }
    $zip = Join-Path $PSScriptRoot "results_ext_$stamp.zip"
    Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $zip
    Remove-Item -Recurse -Force -LiteralPath $stage
    Write-Host "`nResults zip (no checkpoints): $zip" -ForegroundColor Green
}

switch ($Phase) {
    'check'       { Run-Check }
    'newseeds'    { Run-Queue 'jobs.newseeds.json' 'newseeds' }
    'newmodels'   { Run-Queue 'jobs.newmodels.json' 'newmodels' }
    'analyze'     { Run-Analyze }
    'collect'     { Run-Collect }
    'experiments' {
        Run-Queue 'jobs.newseeds.json' 'newseeds'
        Run-Queue 'jobs.newmodels.json' 'newmodels'
        Run-Analyze
        Run-Collect
    }
}
Write-Host "`nPhase '$Phase' finished $(Get-Date -Format 'yyyy-MM-dd HH:mm')." -ForegroundColor Green
Stop-Transcript | Out-Null
