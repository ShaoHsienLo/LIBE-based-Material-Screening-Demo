$ErrorActionPreference = "Stop"

# Ensure we run relative to the script folder (repo root)
Set-Location -Path $PSScriptRoot

$pythonExe = Join-Path -Path $PSScriptRoot -ChildPath ".venv\Scripts\python.exe"
if (-not (Test-Path -Path $pythonExe)) {
    throw "Missing $pythonExe. Run .\setup.ps1 first."
}

$srcDir = Join-Path -Path $PSScriptRoot -ChildPath "src"

$files = @(
    "00_inspect_libe.py",
    "00b_inspect_nested.py",
    "01_build_table.py",
    "02_train_rank.py",
    "03_explain_and_export.py",
    "04_family_ranking.py",
    "05_rule_feedback_retrain.py"
)

foreach ($f in $files) {
    $scriptPath = Join-Path -Path $srcDir -ChildPath $f
    Write-Host "==> Running $f"
    & $pythonExe $scriptPath
}

Write-Host "All scripts finished."
