Param()

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
Set-Location $root

$venvPython = Join-Path $root "venv\\Scripts\\python.exe"
if (-not (Test-Path $venvPython)) { throw "venv not found. Run scripts\\01_setup_env.ps1 first." }

$env:PYTHONPATH = $root
& $venvPython -m src.models.prune_models
& $venvPython -m src.export.export_models
