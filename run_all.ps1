$ErrorActionPreference = "Stop"

$pythonExe = ".venv\\Scripts\\python.exe"
if (-not (Test-Path $pythonExe)) {
    throw "Missing $pythonExe. Run .\\setup.ps1 first."
}

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
    Write-Host "==> Running $f"
    & $pythonExe (Join-Path "src" $f)
}

Write-Host "All scripts finished."
