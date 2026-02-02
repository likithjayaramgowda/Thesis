Param()

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$steps = @(
    ".\\scripts\\01_setup_env.ps1",
    ".\\scripts\\02_get_data.ps1",
    ".\\scripts\\03_preprocess.ps1",
    ".\\scripts\\04_train.ps1",
    ".\\scripts\\05_export.ps1",
    ".\\scripts\\06_benchmark.ps1",
    ".\\scripts\\07_make_reports.ps1"
)

foreach ($s in $steps) {
    Write-Host "Running $s ..."
    & $s
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "DEFENSE PACK READY: docs\\defense_pack"
