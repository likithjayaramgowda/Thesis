Param()

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
Set-Location $root

$venvPython = Join-Path $root "venv\\Scripts\\python.exe"
if (-not (Test-Path $venvPython)) { throw "venv not found. Run scripts\\01_setup_env.ps1 first." }

& $venvPython -m pip install --upgrade kaggle | Out-Null
$kaggleExe = Join-Path $root "venv\\Scripts\\kaggle.exe"
if (-not (Test-Path $kaggleExe)) {
    throw "Kaggle CLI not available in venv. Run scripts\\01_setup_env.ps1 again."
}

$kaggleDir = Join-Path $env:USERPROFILE ".kaggle"
$kaggleJson = Join-Path $kaggleDir "kaggle.json"
$repoKaggleJson = Join-Path $root "kaggle.json"
if (-not (Test-Path $kaggleJson)) {
    if (-not (Test-Path $kaggleDir)) { New-Item -ItemType Directory -Force -Path $kaggleDir | Out-Null }
    if (Test-Path $repoKaggleJson) {
        Copy-Item -Force $repoKaggleJson $kaggleJson
    } elseif (Test-Path (Join-Path $root "scripts\\kaggle.json.template")) {
        Copy-Item -Force (Join-Path $root "scripts\\kaggle.json.template") $kaggleJson
        Write-Host "Fill in your Kaggle token at $kaggleJson before rerun."
        exit 1
    } else {
        throw "Kaggle token not found. Place kaggle.json at repo root or in $kaggleJson."
    }
}

$config = & $venvPython -c "import yaml;print(yaml.safe_load(open('configs/experiment_default.yaml'))['data']['datasets'])"

function Download-KaggleDataset {
    param([string]$slug, [string]$dest)
    if (-not $slug) { return }
    if (Test-Path $dest) { return }
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    & $kaggleExe datasets download -d $slug -p $dest --unzip
    if ($LASTEXITCODE -ne 0) {
        throw "Kaggle download failed for $slug"
    }
}

$datasetsJson = & $venvPython -c "import yaml, json; cfg=yaml.safe_load(open('configs/experiment_default.yaml')); print(json.dumps(cfg['data']['datasets']))"
$data = ConvertFrom-Json $datasetsJson

foreach ($name in $data.PSObject.Properties.Name) {
    $d = $data.$name
    $rawDir = Join-Path $root $d.raw_dir
    if (-not (Test-Path $rawDir)) {
        Write-Host "Downloading $name from Kaggle..."
        Download-KaggleDataset -slug $d.kaggle_slug -dest $rawDir
    } else {
        $files = Get-ChildItem -Path $rawDir -Recurse -File -ErrorAction SilentlyContinue
        if (-not $files) {
            Write-Host "Downloading $name from Kaggle..."
            Download-KaggleDataset -slug $d.kaggle_slug -dest $rawDir
        } else {
            Write-Host "$name already present at $rawDir"
        }
    }
    $filesPost = Get-ChildItem -Path $rawDir -Recurse -File -ErrorAction SilentlyContinue
    if (-not $filesPost) {
        Write-Host "ERROR: $name download produced no files in $rawDir. Check Kaggle access/slug."
        exit 1
    }
}

Write-Host "Data download step complete."
