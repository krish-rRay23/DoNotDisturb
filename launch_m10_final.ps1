# Final M10 launcher: 36 runs = 3 datasets x 2 controllers x 2 shift conditions x 3 seeds.
# Batched concurrent execution: 4 batches x 9 PowerShell background jobs.
# Uses the project's CUDA-enabled interpreter explicitly. Does not run on import.
$PYTHON = ".venv\Scripts\python.exe"
$WORKDIR = $PWD.Path

$datasets = @("halfcheetah-medium-v2", "hopper-medium-v2", "walker2d-medium-v2")
$controllers = @("fixed", "adaptive")
$shiftSteps = @(5000, 0)
$seeds = @(0, 1, 2)
$BATCH_SIZE = 9

# Build all 36 run combinations (order preserved).
$runs = @()
foreach ($ds in $datasets) {
    foreach ($ctrl in $controllers) {
        foreach ($shiftStep in $shiftSteps) {
            foreach ($seed in $seeds) {
                if ($shiftStep -gt 0) { $shiftLabel = "shift" } else { $shiftLabel = "noshift" }
                $runs += [pscustomobject]@{
                    Dataset    = $ds
                    Controller = $ctrl
                    ShiftStep  = $shiftStep
                    ShiftLabel = $shiftLabel
                    Seed       = $seed
                    Log        = "logs/M10-$ctrl-$shiftLabel-seed$seed-$ds.log"
                    Err        = "logs/M10-$ctrl-$shiftLabel-seed$seed-$ds.err"
                    Summary    = "results/M10-$ds-$ctrl-$shiftLabel-seed$seed/summary.json"
                }
            }
        }
    }
}
Write-Host "Total runs: $($runs.Count)"

$failed = @()
$batchCount = [math]::Ceiling($runs.Count / $BATCH_SIZE)
for ($b = 0; $b -lt $batchCount; $b++) {
    $batch = @($runs | Select-Object -Skip ($b * $BATCH_SIZE) -First $BATCH_SIZE)
    Write-Host "Starting batch $($b + 1)/$batchCount ($($batch.Count) jobs)"
    $jobs = @()
    foreach ($r in $batch) {
        $jobName = "M10-$($r.Controller)-$($r.ShiftLabel)-seed$($r.Seed)-$($r.Dataset)"
        Write-Host "Starting job $jobName"
        $jobs += Start-Job -Name $jobName -ScriptBlock {
            param($workdir, $py, $ds, $ctrl, $shiftStep, $seed, $log, $err)
            Set-Location -LiteralPath $workdir
            & $py -m adaptive_plasticity.m5 `
                --dataset $ds `
                --seed "$seed" `
                --online-steps 20000 `
                --update-freq 1000 `
                --online-ratio 0.5 `
                --warmup-steps 6000 `
                --eval-interval 5000 `
                --eval-episodes 5 `
                --device cuda `
                --controller-type $ctrl `
                --shift-type obs_noise `
                --shift-step "$shiftStep" `
                --severity 0.1 1>> $log 2>> $err
            exit $LASTEXITCODE
        } -ArgumentList $WORKDIR, $PYTHON, $r.Dataset, $r.Controller, $r.ShiftStep, $r.Seed, $r.Log, $r.Err
    }
    Wait-Job -Job $jobs | Out-Null
    $batchFailed = $false
    foreach ($job in $jobs) {
        Receive-Job -Job $job | Out-Null
        if ($job.State -eq "Failed") {
            Write-Host "FAILED $($job.Name) (see logs/$($job.Name).err)"
            $failed += $job.Name
            $batchFailed = $true
        } else {
            Write-Host "Finished $($job.Name)"
        }
        Remove-Job -Job $job
    }
    if ($batchFailed) {
        Write-Host "Stopping: failures in batch $($b + 1); not starting next batch."
        break
    }
    Write-Host "Batch $($b + 1)/$batchCount complete"
}

# Final verification: 36 successful runs/summaries.
$ok = @($runs | Where-Object { Test-Path -LiteralPath $_.Summary })
Write-Host "Successful summaries: $($ok.Count)/36"
if ($failed.Count -gt 0 -or $ok.Count -ne 36) {
    Write-Host "M10 runs INCOMPLETE: $($failed.Count) failed, $($ok.Count)/36 summaries present."
    exit 1
}
Write-Host "All 36 M10 runs complete"
