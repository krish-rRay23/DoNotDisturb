"""Step-by-step debug of verify_resume.py."""
import tempfile
from pathlib import Path
import numpy as np
import torch
import adaptive_plasticity.final_runner as fr
from adaptive_plasticity.final_runner import run_final_cell

def debug_verify():
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        dir_uninterrupted = base_dir / "uninterrupted"
        dir_interrupted = base_dir / "interrupted"

        print("1. Running uninterrupted cell (100 online steps)...")
        res_un = run_final_cell(
            dataset_id="halfcheetah-medium-v2", controller="capacity_gate",
            shift="obs_noise", severity=0.1, seed=42, online_steps=100,
            warmup_updates=20, eval_interval=50, gate_interval=10, shift_step=20,
            outdir=dir_uninterrupted, use_shared_offline_checkpoints=False, device="cpu",
        )

        print("2. Simulating crash at step 51...")
        original_add = fr.OnlineBuffer.add
        def mock_add(self, *args, **kwargs):
            if len(self) == 51:
                raise InterruptedError("Simulated sudden crash at step 51!")
            return original_add(self, *args, **kwargs)

        fr.OnlineBuffer.add = mock_add

        try:
            run_final_cell(
                dataset_id="halfcheetah-medium-v2", controller="capacity_gate",
                shift="obs_noise", severity=0.1, seed=42, online_steps=100,
                warmup_updates=20, eval_interval=50, gate_interval=10, shift_step=20,
                outdir=dir_interrupted, use_shared_offline_checkpoints=False, device="cpu",
            )
        except InterruptedError:
            print("Caught crash at step 51!")
        finally:
            fr.OnlineBuffer.add = original_add

        print("3. Resuming interrupted cell from step 50 checkpoint...")
        res_re = run_final_cell(
            dataset_id="halfcheetah-medium-v2", controller="capacity_gate",
            shift="obs_noise", severity=0.1, seed=42, online_steps=100,
            warmup_updates=20, eval_interval=50, gate_interval=10, shift_step=20,
            outdir=dir_interrupted, use_shared_offline_checkpoints=False, device="cpu",
        )

        print("\n--- RESULTS ---")
        print("Uninterrupted returns: ", res_un["returns"])
        print("Resumed returns:       ", res_re["returns"])

        print("\nUninterrupted applied: ", res_un["applied_intervention_steps"])
        print("Resumed applied:       ", res_re["applied_intervention_steps"])

        name_un = res_un["output_name"]
        name_in = "FINAL-halfcheetah-medium-v2-capacity_gate-obs_noise-sev0.1-seed42"

        gate_un = (dir_uninterrupted / name_un / f"gate_{name_un}.jsonl").read_text()
        gate_in = (dir_interrupted / name_in / f"gate_{name_in}.jsonl").read_text()

        print("\nGate log match:", gate_un == gate_in)
        if gate_un != gate_in:
            print("Gate UN:\n", gate_un)
            print("Gate IN:\n", gate_in)

if __name__ == "__main__":
    debug_verify()
