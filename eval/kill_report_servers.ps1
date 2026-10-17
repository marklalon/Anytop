# Stop detached checkpoint review servers, including portable report copies.
$ErrorActionPreference = 'Stop'
$serverPattern = '(?:^|\s)(?:"[^"]*[\\/]serve_report\.py"|[^\s"]*[\\/]serve_report\.py|"?serve_report\.py"?)(?=\s|$)'
$servers = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(?:w|\d+(?:\.\d+)*)?\.exe$' -and
    $_.CommandLine -match $serverPattern
})

$stopped = 0
$failed = 0
foreach ($server in $servers) {
    try {
        Stop-Process -Id $server.ProcessId -ErrorAction Stop
        Write-Host "Stopped report server PID $($server.ProcessId)"
        $stopped++
    } catch {
        # A server may exit between enumeration and termination.
        if (Get-Process -Id $server.ProcessId -ErrorAction SilentlyContinue) {
            Write-Warning "Could not stop report server PID $($server.ProcessId): $_"
            $failed++
        }
    }
}
Write-Host "Report servers stopped: $stopped; failed: $failed"
if ($failed -gt 0) { exit 1 }
exit 0
