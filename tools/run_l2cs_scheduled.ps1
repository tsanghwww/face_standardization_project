param(
    [string]$ProjectRoot = "D:\face_standardization_project"
)

$ErrorActionPreference = "Continue"
Set-Location $ProjectRoot
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$log = Join-Path $ProjectRoot "results\gaze_10k_l2cs_rebuilt\scheduled.log"
$runtimeTemp = Join-Path $ProjectRoot ".runtime_tmp\l2cs"
$torchCache = Join-Path $ProjectRoot ".runtime_tmp\torchinductor"
New-Item -ItemType Directory -Force $runtimeTemp, $torchCache | Out-Null
$env:TEMP = $runtimeTemp
$env:TMP = $runtimeTemp
$env:TORCHINDUCTOR_CACHE_DIR = $torchCache

& $python tools\run_l2cs_batch.py `
    --images-dir archive\generated_yellow-stylegan2 `
    --weights models\l2cs\L2CSNet_gaze360.pkl `
    --output-dir results\gaze_10k_l2cs_rebuilt `
    --device cuda `
    --confidence-threshold 0.5 `
    --resume *> $log

exit $LASTEXITCODE
