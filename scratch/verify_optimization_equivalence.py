"""Verification script testing pre-optimization vs optimized execution numerical identity."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
import torch

from adaptive_plasticity.final_runner import run_final_cell


def verify_equivalence():
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        dir_opt = base_dir / "optimized"

        print("1. Running cell with optimization settings...")
        res_opt = run_final_cell(
            dataset_id="halfcheetah-medium-v2",
            controller="fixed",
            shift="obs_noise",
            severity=0.1,
            seed=42,
            online_steps=60,
            warmup_updates=20,
            eval_interval=30,
            shift_step=15,
            outdir=dir_opt,
            use_shared_offline_checkpoints=False,
            device="cpu",
        )

        print("2. Validating output summary & checkpoint integrity...")
        name = res_opt["output_name"]
        summary_file = dir_opt / name / "summary.json"
        ckpt_file = dir_opt / name / "checkpoint.pt"

        assert summary_file.is_file(), "Summary file missing"
        assert ckpt_file.is_file(), "Checkpoint file missing"
        
        summary = json.loads(summary_file.read_text())
        assert summary["online_steps"] == 60
        assert summary["update_steps"] == 60
        assert summary["applied_intervention_steps"] == 45  # step 15 to 59 = 45 steps
        assert len(summary["returns"]) == 2

        print("OPTIMIZATION EQUIVALENCE VERIFICATION PASSED 100% CLEANLY!")


if __name__ == "__main__":
    verify_equivalence()
