Param()

$ErrorActionPreference = "Stop"

function Get-Python311 {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $probe = & py -3.11 -c "import sys; print(sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -eq 0) { return @("py","-3.11") }
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        $ver = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($LASTEXITCODE -eq 0 -and $ver -eq "3.11") { return @("python") }
    }
    throw "Python 3.11 not found. Install Python 3.11 and retry."
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
Set-Location $root

$pythonCmd = Get-Python311

if (Test-Path "venv") {
    Remove-Item -Recurse -Force "venv"
}

& $pythonCmd[0] @($pythonCmd[1..($pythonCmd.Length-1)] + @("-m","venv","venv"))

$venvPython = Join-Path $root "venv\\Scripts\\python.exe"
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r "requirements.txt"

Write-Host "Environment ready."
