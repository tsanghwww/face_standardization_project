# Connectivity test for dataset download planning
$targets = @(
    @{ Name = 'google';  Url = 'https://www.google.com' },
    @{ Name = 'github';  Url = 'https://github.com' },
    @{ Name = 'hf';      Url = 'https://huggingface.co' },
    @{ Name = 'baidu';   Url = 'https://www.baidu.com' },
    @{ Name = 'cbsr';    Url = 'http://www.cbsr.ia.ac.cn' },
    @{ Name = 'ibug';    Url = 'https://www.ibug.doc.ic.ac.uk' },
    @{ Name = 'caltech'; Url = 'http://www.vision.caltech.edu' },
    @{ Name = 'scface';  Url = 'http://www.scface.org' }
)
foreach ($t in $targets) {
    try {
        $code = & curl.exe -s -o NUL -w '%{http_code}' --max-time 15 --connect-timeout 8 $t.Url 2>$null
        Write-Output ("{0,-10} => {1}" -f $t.Name, $code)
    } catch {
        Write-Output ("{0,-10} => ERROR {1}" -f $t.Name, $_.Exception.Message)
    }
}
Write-Output '--- DNS ---'
foreach ($h in @('www.google.com','github.com','huggingface.co','www.cbsr.ia.ac.cn')) {
    try {
        $ips = [System.Net.Dns]::GetHostAddresses($h) | ForEach-Object { $_.IPAddressToString }
        Write-Output ("{0} => {1}" -f $h, ($ips -join ', '))
    } catch {
        Write-Output ("{0} => DNS FAIL" -f $h)
    }
}
