Param()

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
Set-Location $root

$venvPython = Join-Path $root "venv\\Scripts\\python.exe"
$venvStreamlit = Join-Path $root "venv\\Scripts\\streamlit.exe"
if (-not (Test-Path $venvPython)) { throw "venv not found. Run scripts\\01_setup_env.ps1 first." }
if (-not (Test-Path $venvStreamlit)) { throw "streamlit not found in venv. Re-run setup to install dependencies." }

$env:PYTHONPATH = $root
& $venvStreamlit run src/infer/streamlit_app.py
