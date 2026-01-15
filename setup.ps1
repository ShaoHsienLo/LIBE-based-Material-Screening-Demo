$ErrorActionPreference = "Stop"

Set-Location -Path $PSScriptRoot

$venvPath = Join-Path $PSScriptRoot ".venv"
$pythonExe = Join-Path $venvPath "Scripts\python.exe"

if (-not (Test-Path $venvPath)) {
    python -m venv $venvPath
}

& $pythonExe -m pip install -U pip
& $pythonExe -m pip install -r (Join-Path $PSScriptRoot "requirements.txt")

Write-Host "Setup complete."
Write-Host "Next: .\run_all.ps1"
