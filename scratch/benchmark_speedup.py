"""Benchmark measuring runtime per step before and after optimization."""

from __future__ import annotations

import os
import time
import torch
import numpy as np

# Apply CPU thread capping
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

from adaptive_plasticity.final_runner import run_final_cell

def run_bench():
    t0 = time.time()
    res = run_final_cell(
        dataset_id="halfcheetah-medium-v2",
        controller="none",
        shift="none",
        severity=0.0,
        seed=42,
        online_steps=500,
        warmup_updates=100,
        eval_interval=250,
        outdir="scratch/bench_out",
        use_shared_offline_checkpoints=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    elapsed = time.time() - t0
    print(f"500 online steps + 100 warmup updates took {elapsed:.2f}s ({elapsed/600*1000:.2f} ms/step)")

if __name__ == "__main__":
    run_bench()
