param(
    [string]$ProjectRoot = "D:\face_standardization_project"
)

$ErrorActionPreference = "Stop"
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$start = (Get-Date).AddMinutes(1)

$arcfaceArgs = "tools\extract_arcface_embeddings.py --images-dir archive\generated_yellow-stylegan2 " +
    "--screening-report results\screening_p95\screening_report.json " +
    "--output-dir results\arcface_p95_rebuilt --model-name buffalo_l --det-size 640 " +
    "--det-thresh 0.1 --ctx-id -1 --save-aligned --exclude-ids configs\phase1_eye_invalid_ids.txt"

$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 12) `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$trigger = New-ScheduledTaskTrigger -Once -At $start

$l2csWrapper = Join-Path $ProjectRoot "tools\run_l2cs_scheduled.ps1"
$finalizerWrapper = Join-Path $ProjectRoot "tools\run_phase1_finalizer_scheduled.ps1"
$l2csAction = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File $l2csWrapper" -WorkingDirectory $ProjectRoot
$arcfaceAction = New-ScheduledTaskAction -Execute $python -Argument $arcfaceArgs -WorkingDirectory $ProjectRoot
$finalizerAction = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File $finalizerWrapper" -WorkingDirectory $ProjectRoot

Register-ScheduledTask -TaskName "FacePhase1-L2CS" -Action $l2csAction -Trigger $trigger `
    -Settings $settings -Description "Rebuild 10K L2CS gaze features" -Force | Out-Null
Register-ScheduledTask -TaskName "FacePhase1-ArcFace" -Action $arcfaceAction -Trigger $trigger `
    -Settings $settings -Description "Rebuild p95 ArcFace identity features" -Force | Out-Null
Register-ScheduledTask -TaskName "FacePhase1-Finalizer" -Action $finalizerAction -Trigger $trigger `
    -Settings $settings -Description "Retry ArcFace failures and build Phase1 master manifest" -Force | Out-Null

Start-ScheduledTask -TaskName "FacePhase1-L2CS"
Start-ScheduledTask -TaskName "FacePhase1-ArcFace"
Start-ScheduledTask -TaskName "FacePhase1-Finalizer"
Start-Sleep -Seconds 3
Get-ScheduledTask -TaskName "FacePhase1-L2CS", "FacePhase1-ArcFace", "FacePhase1-Finalizer" |
    Select-Object TaskName, State | Format-Table -AutoSize
