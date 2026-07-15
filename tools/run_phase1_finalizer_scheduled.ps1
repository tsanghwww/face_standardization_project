param(
    [string]$ProjectRoot = "D:\face_standardization_project"
)

$ErrorActionPreference = "Continue"
Set-Location $ProjectRoot
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$runtimeTemp = Join-Path $ProjectRoot ".runtime_tmp\finalizer"
New-Item -ItemType Directory -Force $runtimeTemp, (Join-Path $ProjectRoot "results\phase1_parity") | Out-Null
$env:TEMP = $runtimeTemp
$env:TMP = $runtimeTemp

& $python tools\finalize_phase1_parity.py *> results\phase1_parity\finalizer.log
exit $LASTEXITCODE
