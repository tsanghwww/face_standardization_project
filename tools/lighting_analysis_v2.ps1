param(
    [string]$Root = "H:\face_standardization_project\results\screening_v2",
    [int]$Resize = 64
)

Add-Type -AssemblyName System.Drawing

$outCsv = Join-Path $Root "lighting_analysis.csv"
$outSummary = Join-Path $Root "lighting_summary.csv"
$outMd = Join-Path $Root "lighting_summary.md"

function Measure-ImageLight {
    param(
        [string]$Path,
        [string]$Group,
        [int]$Resize
    )

    $src = $null
    $bmp = $null
    $g = $null
    try {
        $src = [System.Drawing.Image]::FromFile($Path)
        $bmp = New-Object System.Drawing.Bitmap $Resize, $Resize, ([System.Drawing.Imaging.PixelFormat]::Format24bppRgb)
        $g = [System.Drawing.Graphics]::FromImage($bmp)
        $g.DrawImage($src, 0, 0, $Resize, $Resize)

        $n = $Resize * $Resize
        $sum = 0.0
        $sumSq = 0.0
        $dark = 0
        $veryDark = 0
        $bright = 0
        $over = 0
        $leftSum = 0.0
        $rightSum = 0.0
        $leftN = 0
        $rightN = 0

        $rect = New-Object System.Drawing.Rectangle 0, 0, $Resize, $Resize
        $data = $bmp.LockBits($rect, [System.Drawing.Imaging.ImageLockMode]::ReadOnly, $bmp.PixelFormat)
        try {
            $stride = [Math]::Abs($data.Stride)
            $bytes = New-Object byte[] ($stride * $Resize)
            [System.Runtime.InteropServices.Marshal]::Copy($data.Scan0, $bytes, 0, $bytes.Length)

            for ($y = 0; $y -lt $Resize; $y++) {
                $row = $y * $stride
                for ($x = 0; $x -lt $Resize; $x++) {
                    $idx = $row + ($x * 3)
                    $b = $bytes[$idx]
                    $gch = $bytes[$idx + 1]
                    $r = $bytes[$idx + 2]
                    $lum = 0.299 * $r + 0.587 * $gch + 0.114 * $b
                $sum += $lum
                $sumSq += $lum * $lum
                if ($lum -lt 40) { $dark++ }
                if ($lum -lt 20) { $veryDark++ }
                if ($lum -gt 220) { $bright++ }
                if ($lum -gt 245) { $over++ }
                    if ($x -lt ($Resize / 2)) {
                        $leftSum += $lum
                        $leftN++
                    } else {
                        $rightSum += $lum
                        $rightN++
                    }
                }
            }
        } finally {
            $bmp.UnlockBits($data)
        }

        $mean = $sum / $n
        $var = [Math]::Max(($sumSq / $n) - ($mean * $mean), 0)
        $std = [Math]::Sqrt($var)
        $leftMean = $leftSum / $leftN
        $rightMean = $rightSum / $rightN
        $lrDiff = [Math]::Abs($leftMean - $rightMean)

        [pscustomobject]@{
            image_id = [System.IO.Path]::GetFileNameWithoutExtension($Path)
            group = $Group
            path = $Path
            mean_luma = [Math]::Round($mean, 4)
            std_luma = [Math]::Round($std, 4)
            dark_ratio = [Math]::Round($dark / $n, 6)
            very_dark_ratio = [Math]::Round($veryDark / $n, 6)
            bright_ratio = [Math]::Round($bright / $n, 6)
            overexposed_ratio = [Math]::Round($over / $n, 6)
            left_mean_luma = [Math]::Round($leftMean, 4)
            right_mean_luma = [Math]::Round($rightMean, 4)
            lr_abs_diff = [Math]::Round($lrDiff, 4)
        }
    } finally {
        if ($g -ne $null) { $g.Dispose() }
        if ($bmp -ne $null) { $bmp.Dispose() }
        if ($src -ne $null) { $src.Dispose() }
    }
}

function Summarize-Group {
    param(
        [object[]]$Rows,
        [string]$Group
    )

    $rs = @($Rows | Where-Object { $_.group -eq $Group })
    $count = $rs.Count
    if ($count -eq 0) {
        return [pscustomobject]@{
            group = $Group; count = 0; mean_luma_avg = ""; std_luma_avg = "";
            dark_ratio_avg = ""; bright_ratio_avg = ""; overexposed_ratio_avg = "";
            lr_abs_diff_avg = ""; dark_image_count = ""; bright_image_count = "";
            imbalanced_light_count = ""
        }
    }

    [pscustomobject]@{
        group = $Group
        count = $count
        mean_luma_avg = [Math]::Round(($rs | Measure-Object mean_luma -Average).Average, 4)
        std_luma_avg = [Math]::Round(($rs | Measure-Object std_luma -Average).Average, 4)
        dark_ratio_avg = [Math]::Round(($rs | Measure-Object dark_ratio -Average).Average, 6)
        bright_ratio_avg = [Math]::Round(($rs | Measure-Object bright_ratio -Average).Average, 6)
        overexposed_ratio_avg = [Math]::Round(($rs | Measure-Object overexposed_ratio -Average).Average, 6)
        lr_abs_diff_avg = [Math]::Round(($rs | Measure-Object lr_abs_diff -Average).Average, 4)
        dark_image_count = @($rs | Where-Object { $_.mean_luma -lt 80 }).Count
        bright_image_count = @($rs | Where-Object { $_.mean_luma -gt 180 }).Count
        imbalanced_light_count = @($rs | Where-Object { $_.lr_abs_diff -gt 25 }).Count
    }
}

$allRows = New-Object System.Collections.Generic.List[object]
foreach ($group in @("warn", "pass")) {
    $dirName = if ($group -eq "warn") { "warn_images" } else { "pass_images" }
    $dir = Join-Path $Root $dirName
    $files = @(Get-ChildItem $dir -File -Include *.png, *.jpg, *.jpeg)
    if ($files.Count -eq 0) {
        $files = @(Get-ChildItem $dir -File | Where-Object { $_.Extension -match "^\.(png|jpg|jpeg)$" })
    }
    $i = 0
    foreach ($file in $files) {
        $i++
        if (($i % 500) -eq 0) {
            Write-Host "$group $i / $($files.Count)"
        }
        $allRows.Add((Measure-ImageLight -Path $file.FullName -Group $group -Resize $Resize))
    }
}

$allRows | Export-Csv $outCsv -NoTypeInformation -Encoding UTF8

$summary = @(
    Summarize-Group -Rows $allRows -Group "warn"
    Summarize-Group -Rows $allRows -Group "pass"
)
$summary | Export-Csv $outSummary -NoTypeInformation -Encoding UTF8

$warn = $summary | Where-Object { $_.group -eq "warn" }
$pass = $summary | Where-Object { $_.group -eq "pass" }
$md = @"
# Lighting Analysis Summary

- Source: `$Root`
- Method: resized each image to ${Resize}x${Resize}; computed luminance statistics.
- Metrics:
  - `mean_luma`: average brightness.
  - `std_luma`: brightness contrast.
  - `dark_ratio`: fraction of pixels with luminance < 40.
  - `bright_ratio`: fraction of pixels with luminance > 220.
  - `overexposed_ratio`: fraction of pixels with luminance > 245.
  - `lr_abs_diff`: absolute left/right brightness imbalance.

## Group Summary

| group | count | mean_luma_avg | std_luma_avg | dark_ratio_avg | bright_ratio_avg | overexposed_ratio_avg | lr_abs_diff_avg | dark_image_count | bright_image_count | imbalanced_light_count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| warn | $($warn.count) | $($warn.mean_luma_avg) | $($warn.std_luma_avg) | $($warn.dark_ratio_avg) | $($warn.bright_ratio_avg) | $($warn.overexposed_ratio_avg) | $($warn.lr_abs_diff_avg) | $($warn.dark_image_count) | $($warn.bright_image_count) | $($warn.imbalanced_light_count) |
| pass | $($pass.count) | $($pass.mean_luma_avg) | $($pass.std_luma_avg) | $($pass.dark_ratio_avg) | $($pass.bright_ratio_avg) | $($pass.overexposed_ratio_avg) | $($pass.lr_abs_diff_avg) | $($pass.dark_image_count) | $($pass.bright_image_count) | $($pass.imbalanced_light_count) |

## Interpretation Guide

- Higher `dark_ratio_avg` means more under-lit face images.
- Higher `bright_ratio_avg` or `overexposed_ratio_avg` means stronger highlights or possible glare.
- Higher `lr_abs_diff_avg` means stronger side-lighting imbalance.
- If Warn is higher than Pass on these metrics, lighting can be discussed as one recurring factor in the Warn set.
"@
$md | Set-Content $outMd -Encoding UTF8

Write-Host "CSV: $outCsv"
Write-Host "Summary CSV: $outSummary"
Write-Host "Summary MD: $outMd"
