"""End-to-end smoke test verifying telemetry, transitions, and gate records for all 3 arms x 3 regimes."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
import numpy as np

from adaptive_plasticity.final_runner import run_final_cell, FINAL_SHIFTS, PRIMARY_CONTROLLERS


def run_e2e_smoke():
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir)
        print(f"Running E2E smoke tests for {len(PRIMARY_CONTROLLERS) * len(FINAL_SHIFTS)} cells...")
        
        results = {}
        for shift in FINAL_SHIFTS:
            for controller in PRIMARY_CONTROLLERS:
                cell_name = f"{shift}_{controller}"
                res = run_final_cell(
                    dataset_id="halfcheetah-medium-v2",
                    controller=controller,
                    shift=shift,
                    severity=0.1 if shift == "obs_noise" else (0.5 if shift == "reward_scale" else 0.0),
                    seed=42,
                    online_steps=50,
                    warmup_updates=10,
                    eval_interval=25,
                    gate_interval=10,
                    shift_step=10 if shift != "none" else 0,
                    outdir=outdir,
                    use_shared_offline_checkpoints=False,
                    device="cpu",
                )
                results[cell_name] = res

        # Verify all 9 summaries and checkpoints exist
        for shift in FINAL_SHIFTS:
            for controller in PRIMARY_CONTROLLERS:
                cell_name = f"{shift}_{controller}"
                out_name = results[cell_name]["output_name"]
                cell_dir = outdir / out_name
                
                # Check summary.json
                summary_path = cell_dir / "summary.json"
                assert summary_path.is_file(), f"Missing summary for {cell_name}"
                summary = json.loads(summary_path.read_text())
                assert summary["online_steps"] == 50
                assert summary["update_steps"] == 50
                
                # Check checkpoint.pt
                ckpt_path = cell_dir / "checkpoint.pt"
                assert ckpt_path.is_file(), f"Missing checkpoint for {cell_name}"
                
                # Check capacity_gate log if controller == capacity_gate
                if controller == "capacity_gate":
                    gate_log = cell_dir / f"gate_{out_name}.jsonl"
                    assert gate_log.is_file(), f"Missing gate log for {cell_name}"
                    lines = [json.loads(line) for line in gate_log.read_text().splitlines() if line.strip()]
                    assert len(lines) > 0, f"Empty gate log for {cell_name}"
                
                # Verify applied_intervention_steps semantics
                applied = summary["applied_intervention_steps"]
                if controller == "none":
                    assert applied == 0, f"none arm applied steps = {applied} != 0"
                elif controller == "fixed":
                    if shift == "none":
                        assert applied == 0, f"fixed under shift=none applied steps = {applied} != 0"
                    else:
                        assert applied == 40, f"fixed under {shift} applied steps = {applied} != 40"

        print("ALL 9 E2E SMOKE TESTS PASSED CLEANLY!")


if __name__ == "__main__":
    run_e2e_smoke()
