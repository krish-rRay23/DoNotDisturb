"""Test if two fresh runs with seed=42 produce bit-identical outputs."""
import tempfile
from pathlib import Path
import numpy as np
import torch
from adaptive_plasticity.final_runner import run_final_cell

def test_fresh_runs():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        dir1 = base / "run1"
        dir2 = base / "run2"

        res1 = run_final_cell(
            dataset_id="halfcheetah-medium-v2", controller="capacity_gate",
            shift="obs_noise", severity=0.1, seed=42, online_steps=100,
            warmup_updates=20, eval_interval=50, gate_interval=10, shift_step=20,
            outdir=dir1, use_shared_offline_checkpoints=False, device="cpu",
        )

        res2 = run_final_cell(
            dataset_id="halfcheetah-medium-v2", controller="capacity_gate",
            shift="obs_noise", severity=0.1, seed=42, online_steps=100,
            warmup_updates=20, eval_interval=50, gate_interval=10, shift_step=20,
            outdir=dir2, use_shared_offline_checkpoints=False, device="cpu",
        )

        print("Run 1 returns:", res1["returns"])
        print("Run 2 returns:", res2["returns"])
        assert res1["returns"] == res2["returns"], "Two fresh runs diverged!"
        print("PASS: Two fresh runs are 100% BIT-IDENTICAL!")

if __name__ == "__main__":
    test_fresh_runs()
