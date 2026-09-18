"""Detailed diagnostic of return calculation during verify_resume."""
import tempfile
from pathlib import Path
import numpy as np
import torch
import adaptive_plasticity.final_runner as fr
from adaptive_plasticity.final_runner import run_final_cell

def trace_returns():
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        dir_uninterrupted = base_dir / "uninterrupted"
        dir_interrupted = base_dir / "interrupted"

        orig_eval = fr._evaluate
        def debug_eval(agent, env, *, seed, episodes, device):
            ret = orig_eval(agent, env, seed=seed, episodes=episodes, device=device)
            print(f"[DEBUG _evaluate] seed={seed}, ret={ret}")
            return ret

        fr._evaluate = debug_eval

        print("--- RUN 1: Uninterrupted ---")
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

        print("--- RUN 2: Interrupt at step 51 ---")
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

        print("--- RUN 3: Resume from step 50 ---")
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

        fr._evaluate = orig_eval

        print("Uninterrupted returns:", res_un["returns"])
        print("Resumed returns:      ", res_re["returns"])

if __name__ == "__main__":
    trace_returns()
