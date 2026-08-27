# Resume COFW Color with a SINGLE connection (multi-connection pieces were
# inconsistent across Wayback snapshot servers). Run standalone, detached.
$ErrorActionPreference = 'Continue'
$aria2 = 'D:\face_standardization_project\tools\aria2\aria2c.exe'
$destDir = 'D:\face_standardization_project\datasets\external\COFW_Color'
$url = 'http://web.archive.org/web/20211104050812/http://www.vision.caltech.edu/xpburgos/ICCV13/Data/COFW_color.zip'
$log = Join-Path $destDir 'cofw_resume.log'
& $aria2 @('-c', '-x', '1', '-s', '1', '--max-tries', '60', '--retry-wait', '10',
           '--connect-timeout', '30', '--timeout', '180',
           '--file-allocation', 'none', '--console-log-level', 'notice',
           '--log', $log, '--log-level', 'notice', '--summary-interval', '0',
           '-d', $destDir, '-o', 'COFW_color.zip', $url) 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) {
    Add-Content -Path (Join-Path $destDir 'cofw_resume_done.txt') -Value ((Get-Date).ToString() + ' DONE size=' + (Get-Item (Join-Path $destDir 'COFW_color.zip')).Length)
} else {
    Add-Content -Path (Join-Path $destDir 'cofw_resume_done.txt') -Value ((Get-Date).ToString() + ' FAILED exit=' + $LASTEXITCODE)
}
