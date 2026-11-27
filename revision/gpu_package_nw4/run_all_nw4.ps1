<#
.SYNOPSIS
  NW4 control campaign: ResNet-50 and TimeSformer re-trained with num_workers=4
  (parallel data loading), everything else identical to the A4000 cohort recipe.
  Answers referee major comment 3: does the training-energy ordering hold with a
  faster input pipeline? Separate cohort (local_pc_a4000_numworkers4_v1), never
  pooled with the main one.

.DESCRIPTION
  Phases (each stops at the first failure):
    check  preflight + one smoke run per model (exercises Windows spawn workers)
    run    the 10 training runs (2 configurations x 5 seeds), --resume safe

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File .\run_all_nw4.ps1 -Phase check
  powershell -NoProfile -ExecutionPolicy Bypass -File .\run_all_nw4.ps1 -Phase run
#>
param(
    [ValidateSet('check', 'run')]
    [string]$Phase = 'check',
    [string]$Python = ''
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
New-Item -ItemType Directory -Force -Path 'logs_nw4' | Out-Null
Start-Transcript -Path "logs_nw4\driver_${Phase}_$stamp.log" | Out-Null

if (-not $Python) {
    $repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
    $Python = Join-Path $repo '.venv-carbon\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $Python)) {
    Stop-Transcript | Out-Null
    throw "Python not found at $Python"
}
Write-Host "Python: $Python"

Add-Type -Namespace Nsyss -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[void][Nsyss.Power]::SetThreadExecutionState([uint32]2147483649)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

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
        throw "$Name FAILED (exit $code). Read the message above and the queue log in logs_nw4\. Nothing after this step was run."
    }
}

switch ($Phase) {
    'check' {
        Invoke-Py 'Preflight (nw4)' @('-u', 'preflight_nw4.py', '--config', 'config.nw4.json')
        Invoke-Py 'Smoke tests (2 models, num_workers=4)' @('-u', 'queue_ext.py', 'jobs.smoke_nw4.json', '--config', 'config.nw4.json', '--log-dir', "logs_nw4\smoke_$stamp")
        Write-Host "`nCHECK PHASE PASSED. Smoke logs contain one training step per model through real" -ForegroundColor Green
        Write-Host "DataLoader workers; inspect, then run -Phase run." -ForegroundColor Green
    }
    'run' {
        Invoke-Py 'Preflight (nw4)' @('-u', 'preflight_nw4.py', '--config', 'config.nw4.json')
        Invoke-Py 'Queue nw4 (10 runs)' @('-u', 'queue_ext.py', 'jobs.nw4.json', '--config', 'config.nw4.json', '--log-dir', "logs_nw4\run_$stamp", '--resume')
        Write-Host "`nNW4 RUN PHASE FINISHED. Compare D:\nsyss50_results_nw4\local_pc_a4000_numworkers4_v1" -ForegroundColor Green
        Write-Host "train_kwh against the main cohort (D:\nsyss50_results_ext\...\summary basis)." -ForegroundColor Green
    }
}
Write-Host "`nPhase '$Phase' finished $(Get-Date -Format 'yyyy-MM-dd HH:mm')." -ForegroundColor Green
Stop-Transcript | Out-Null
