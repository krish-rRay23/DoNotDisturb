"""Detailed diagnostic to trace exact point of divergence between uninterrupted and resumed runs."""
import tempfile
from pathlib import Path
import numpy as np
import torch
import adaptive_plasticity.final_runner as fr
from adaptive_plasticity.final_runner import run_final_cell

def trace_divergence():
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        dir_uninterrupted = base_dir / "uninterrupted"
        dir_interrupted = base_dir / "interrupted"

        print("1. Running uninterrupted cell...")
        res_un = run_final_cell(
            dataset_id="halfcheetah-medium-v2",
            controller="capacity_gate",
            shift="obs_noise",
            severity=0.1,
            seed=42,
            online_steps=100,
            warmup_updates=20,
            eval_interval=50,
            gate_interval=10,
            shift_step=20,
            outdir=dir_uninterrupted,
            use_shared_offline_checkpoints=False,
            device="cpu",
        )

        print("2. Simulating crash at step 51...")
        original_add = fr.OnlineBuffer.add
        def mock_add(self, *args, **kwargs):
            if len(self) == 51:
                raise InterruptedError("Crash at step 51!")
            return original_add(self, *args, **kwargs)

        fr.OnlineBuffer.add = mock_add

        try:
            run_final_cell(
                dataset_id="halfcheetah-medium-v2",
                controller="capacity_gate",
                shift="obs_noise",
                severity=0.1,
                seed=42,
                online_steps=100,
                warmup_updates=20,
                eval_interval=50,
                gate_interval=10,
                shift_step=20,
                outdir=dir_interrupted,
                use_shared_offline_checkpoints=False,
                device="cpu",
            )
        except InterruptedError:
            pass
        finally:
            fr.OnlineBuffer.add = original_add

        print("3. Resuming interrupted cell...")
        res_re = run_final_cell(
            dataset_id="halfcheetah-medium-v2",
            controller="capacity_gate",
            shift="obs_noise",
            severity=0.1,
            seed=42,
            online_steps=100,
            warmup_updates=20,
            eval_interval=50,
            gate_interval=10,
            shift_step=20,
            outdir=dir_interrupted,
            use_shared_offline_checkpoints=False,
            device="cpu",
        )

        print("Uninterrupted returns:", res_un["returns"])
        print("Resumed returns:      ", res_re["returns"])

        # Check weights at end
        name_un = res_un["output_name"]
        name_in = "FINAL-halfcheetah-medium-v2-capacity_gate-obs_noise-sev0.1-seed42"
        ckpt_un = torch.load(dir_uninterrupted / name_un / "checkpoint.pt", map_location="cpu", weights_only=False)
        ckpt_re = torch.load(dir_interrupted / name_in / "checkpoint.pt", map_location="cpu", weights_only=False)

        for k in ckpt_un:
            if isinstance(ckpt_un[k], torch.Tensor):
                diff = torch.max(torch.abs(ckpt_un[k] - ckpt_re[k])).item()
                if diff > 0:
                    print(f"Weight diff in {k}: {diff}")

if __name__ == "__main__":
    trace_divergence()
