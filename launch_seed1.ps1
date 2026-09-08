$datasets = @("halfcheetah-medium-v2", "hopper-medium-v2", "walker2d-medium-v2")
$seeds = @(1, 2)

foreach ($seed in $seeds) {
    Write-Host "Launching seed=$seed"
    $jobs = @()
    foreach ($ds in $datasets) {
        $log = "C:\Users\krish\AppData\Local\Temp\opencode\m4_seed${seed}_${ds}.log"
        $err = "C:\Users\krish\AppData\Local\Temp\opencode\m4_seed${seed}_${ds}.log.err"
        $psi = Start-Process -FilePath ".\.venv\Scripts\python.exe" -ArgumentList "-m", "adaptive_plasticity.train_iql", "-dataset", $ds, "-seed", "$seed", "-steps", "200000", "-device", "cuda", "-output-root", "." -RedirectStandardOutput $log -RedirectStandardError $err -PassThru
        $jobs += $psi.Id
    }
    Write-Host "Waiting for seed=$seed jobs: $($jobs -join ',')"
    # Wait for all jobs to complete
    while ($jobs) {
        $remaining = @()
        foreach ($pid in $jobs) {
            $proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
            if ($proc) {
                $remaining += $pid
            }
        }
        if (-not $remaining) { break }
        Start-Sleep -Seconds 60
    }
    Write-Host "Seed $seed all complete"
}
Write-Host "All seeds complete"