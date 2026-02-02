Param()

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
Set-Location $root

$venvPython = Join-Path $root "venv\\Scripts\\python.exe"
if (-not (Test-Path $venvPython)) { throw "venv not found. Run scripts\\01_setup_env.ps1 first." }

$env:PYTHONPATH = $root
& $venvPython -m src.train.train_classification
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# YOLO prints warnings to stderr; don't treat warnings as terminating errors
$ErrorActionPreference = "Continue"
& $venvPython -m src.train.train_yolo
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
