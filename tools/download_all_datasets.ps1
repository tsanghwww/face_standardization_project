# Download all external evaluation datasets on win-lenovo using aria2c.
# Robust: resumable (-c), multi-connection (-x 8), retries, torrent support.
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File download_all_datasets.ps1
# Logs to datasets\external\download_log.txt

$ErrorActionPreference = 'Continue'
$aria2 = 'D:\face_standardization_project\tools\aria2\aria2c.exe'
$base  = 'D:\face_standardization_project\datasets\external'
$log   = Join-Path $base 'download_log.txt'

function Log($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
    Add-Content -Path $log -Value $line
    Write-Output $line
}

function Aria($url, $destDir, $outName, $desc) {
    New-Item -ItemType Directory -Force -Path $destDir | Out-Null
    $dest = Join-Path $destDir $outName
    $ariaLog = Join-Path $destDir ("aria2_" + $outName + ".log")
    Log "START: $desc"
    & $aria2 @('-c', '-x', '8', '-s', '8', '--max-tries', '30', '--retry-wait', '8',
               '--connect-timeout', '30', '--timeout', '120',
               '--file-allocation', 'none', '--console-log-level', 'warn',
               '--log', $ariaLog, '--log-level', 'notice', '--summary-interval', '0',
               '-d', $destDir, '-o', $outName, $url) 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0 -and (Test-Path $dest)) {
        $size = (Get-Item $dest).Length
        Log "DONE ($size bytes): $desc"
    } else {
        Log "FAILED (exit $LASTEXITCODE): $desc"
        Log "  aria2 log tail:"
        if (Test-Path $ariaLog) { Get-Content $ariaLog -Tail 5 | ForEach-Object { Log ("    " + $_) } }
    }
}

function Aria-Torrent($torrentPath, $destDir, $desc) {
    New-Item -ItemType Directory -Force -Path $destDir | Out-Null
    $ariaLog = Join-Path $destDir 'aria2_torrent.log'
    Log "START (torrent): $desc"
    & $aria2 @('--seed-time=0', '--max-tries', '15', '--retry-wait', '10',
               '--enable-dht=true', '--dht-listen-port', '6881-6999', '--bt-enable-lpd=true',
               '--file-allocation', 'none', '--console-log-level', 'warn',
               '--log', $ariaLog, '--log-level', 'notice', '--summary-interval', '0',
               '-d', $destDir, $torrentPath) 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Log "DONE (torrent): $desc"
    } else {
        Log "FAILED (torrent, exit $LASTEXITCODE): $desc"
        if (Test-Path $ariaLog) { Get-Content $ariaLog -Tail 5 | ForEach-Object { Log ("    " + $_) } }
    }
}

New-Item -ItemType Directory -Force -Path $base | Out-Null
Log '================ download session (aria2 v2) start ================'

# --- WIDER FACE (HuggingFace official mirror: CUHK-CSE/wider_face) ---
$hf = 'https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data'
$wider = Join-Path $base 'WIDER_FACE'
Aria "$hf/WIDER_train.zip"      $wider 'WIDER_train.zip'     'WIDER FACE train (~1.38GB)'
Aria "$hf/WIDER_val.zip"        $wider 'WIDER_val.zip'       'WIDER FACE val (~234MB)'
Aria "$hf/WIDER_test.zip"       $wider 'WIDER_test.zip'      'WIDER FACE test (~445MB)'
Aria "$hf/wider_face_split.zip" $wider 'wider_face_split.zip' 'WIDER FACE annotations'

# --- 300W-LP (Google Drive, confirm=t flow verified) ---
Aria 'https://drive.usercontent.google.com/download?id=0B7OEHD3T4eCkVGs0TkhUWFN6N1k&export=download&confirm=t' (Join-Path $base '300W-LP') '300W_LP.zip' '300W-LP (~8GB, Google Drive)'

# --- COFW Color: handled by resume_cofw.ps1 (single-connection, detached) ---
Log 'COFW Color handled separately by resume_cofw.ps1 (skipped here)'

# --- AFLW2000-3D: images (hailo gdrive mirror) + annotations (3DDFA GitHub) ---
$afDir = Join-Path $base 'AFLW2000-3D'
Aria 'https://drive.usercontent.google.com/download?id=1r_ciJ1M0BSRTwndIBt42GlPFRv6CvvEP&export=download&confirm=t' (Join-Path $afDir 'images') 'test.data.zip' 'AFLW2000-3D images (2000 cropped, ~159MB, hailo gdrive)'
$cfgUrl = 'https://raw.githubusercontent.com/cleardusk/3DDFA/master/test.configs'
foreach ($f in @('AFLW2000-3D.pose.npy','AFLW2000-3D.pts68.npy','AFLW2000-3D-Reannotated.pts68.npy','AFLW2000-3D_crop.roi_box.npy')) {
    Aria "$cfgUrl/$f" (Join-Path $afDir 'annotations') $f "AFLW2000-3D annotation $f (GitHub 3DDFA)" $null
}

# --- AFLW2000-3D full zip with .mat via torrent (optional; trackers mostly dead) ---
$torrent = 'D:\face_standardization_project\tools\aflw2000-3d.torrent'
if (Test-Path $torrent) {
    Aria-Torrent $torrent (Join-Path $base 'AFLW2000-3D') 'AFLW2000-3D full zip w/ .mat (~87MB, torrent + DHT)'
} else {
    Log 'AFLW2000-3D torrent file missing - skipped'
}

Log '================ download session end ================'
