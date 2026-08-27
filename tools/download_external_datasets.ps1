# Download external evaluation datasets onto win-lenovo.
# Usage: powershell -ExecutionPolicy Bypass -File download_external_datasets.ps1
# Designed to be run detached; progress is logged to datasets\external\download_log.txt

$ErrorActionPreference = 'Continue'
$logPath = 'D:\face_standardization_project\datasets\external\download_log.txt'
$base = 'D:\face_standardization_project\datasets\external'

function Log($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
    Add-Content -Path $logPath -Value $line
    Write-Output $line
}

function Download-File($url, $dest, $desc) {
    if (Test-Path $dest) {
        $size = (Get-Item $dest).Length
        Log "SKIP (exists, $size bytes): $desc -> $dest"
        return
    }
    Log "START: $desc"
    Log "  url: $url"
    $dir = Split-Path $dest -Parent
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $attempt = 0
    while ($attempt -lt 6) {
        $attempt++
        # curl.exe: -C - resumes, -L follows redirects, --retry for transient failures
        & curl.exe -sS -L -C - --retry 3 --retry-delay 5 --connect-timeout 20 --max-time 7200 -o $dest $url 2>>$logPath
        if ($LASTEXITCODE -eq 0) {
            $size = (Get-Item $dest).Length
            Log "DONE (attempt $attempt, $size bytes): $desc"
            return
        }
        Log "RETRY attempt $attempt failed (exit $LASTEXITCODE): $desc"
        Start-Sleep -Seconds 10
    }
    Log "FAILED after retries: $desc"
}

New-Item -ItemType Directory -Force -Path $base | Out-Null
Log "============ download session start ============"

# --- 1. WIDER FACE (HuggingFace official mirror: CUHK-CSE/wider_face) ---
$hfRoot = 'https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data'
Download-File "$hfRoot/WIDER_train.zip" "$base\WIDER_FACE\WIDER_train.zip" "WIDER FACE train (~1.38GB)"
Download-File "$hfRoot/WIDER_val.zip"   "$base\WIDER_FACE\WIDER_val.zip"   "WIDER FACE val (~234MB)"
Download-File "$hfRoot/WIDER_test.zip"  "$base\WIDER_FACE\WIDER_test.zip"  "WIDER FACE test (~445MB)"
Download-File "$hfRoot/wider_face_split.zip" "$base\WIDER_FACE\wider_face_split.zip" "WIDER FACE annotations"

# --- 2. 300W-LP (Google Drive, file 0B7OEHD3T4eCkVGs0TkhUWFN6N1k) ---
# gdrive large-file flow: first GET to obtain confirm token, then direct download.
$gdId = '0B7OEHD3T4eCkVGs0TkhUWFN6N1k'
$gdDest = "$base\300W-LP\300W_LP.zip"
if (-not (Test-Path $gdDest)) {
    Log "START: 300W-LP (~8GB) via Google Drive"
    $cookieJar = "$base\300W-LP\gd_cookies.txt"
    New-Item -ItemType Directory -Force -Path "$base\300W-LP" | Out-Null
    $confirm = 't'
    try {
        & curl.exe -sS -L -c $cookieJar --max-time 60 "https://drive.google.com/uc?export=download&id=$gdId" -o "$base\300W-LP\gd_first.html" 2>>$logPath
        $html = Get-Content "$base\300W-LP\gd_first.html" -Raw -ErrorAction SilentlyContinue
        $m = [regex]::Match($html, 'name="confirm" value="([^"]+)"')
        if ($m.Success) { $confirm = $m.Groups[1].Value }
        Log "gdrive confirm token: $confirm"
    } catch { Log "gdrive token step warning: $_" }
    Download-File "https://drive.usercontent.google.com/download?id=$gdId&export=download&confirm=$confirm" $gdDest "300W-LP (~8GB)"
} else {
    Log "SKIP (exists): 300W-LP"
}

Log "============ download session end ============"
