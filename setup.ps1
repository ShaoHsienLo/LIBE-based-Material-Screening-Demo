$ErrorActionPreference = "Stop"

$venvPath = ".venv"
$pythonExe = Join-Path $venvPath "Scripts\\python.exe"

if (-not (Test-Path $venvPath)) {
    python -m venv $venvPath
}

& $pythonExe -m pip install -U pip
& $pythonExe -m pip install -r requirements.txt

Write-Host "Setup complete."
Write-Host "Next: .\\run_all.ps1"
