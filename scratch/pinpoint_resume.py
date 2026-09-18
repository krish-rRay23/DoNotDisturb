"""Pinpoint the exact variable that differs at step 51 between uninterrupted and resumed runs."""
import tempfile
from pathlib import Path
import numpy as np
import torch
import adaptive_plasticity.final_runner as fr
from adaptive_plasticity.final_runner import run_final_cell

def pinpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        dir_uninterrupted = base_dir / "uninterrupted"
        dir_interrupted = base_dir / "interrupted"

        step51_data = {}

        orig_add = fr.OnlineBuffer.add
        current_run = [1]

        def hook_add(self, obs, action, reward, next_obs, done):
            if len(self) == 50:  # This is step 51 (since buffer index 50 is 51st transition)
                step51_data[current_run[0]] = {
                    "obs": obs.copy(),
                    "action": action.copy(),
                    "reward": float(reward),
                    "next_obs": next_obs.copy(),
                    "done": float(done),
                }
            return orig_add(self, obs, action, reward, next_obs, done)

        fr.OnlineBuffer.add = hook_add

        print("--- RUN 1: Uninterrupted ---")
        run_final_cell(
            dataset_id="halfcheetah-medium-v2", controller="capacity_gate",
            shift="obs_noise", severity=0.1, seed=42, online_steps=100,
            warmup_updates=20, eval_interval=50, gate_interval=10, shift_step=20,
            outdir=dir_uninterrupted, use_shared_offline_checkpoints=False, device="cpu",
        )

        print("--- RUN 2: Crash at step 51 ---")
        current_run[0] = 2
        def crash_add(self, *args, **kwargs):
            if len(self) == 51:
                raise InterruptedError("Crash at step 51!")
            return orig_add(self, *args, **kwargs)

        fr.OnlineBuffer.add = crash_add

        try:
            run_final_cell(
                dataset_id="halfcheetah-medium-v2", controller="capacity_gate",
                shift="obs_noise", severity=0.1, seed=42, online_steps=100,
                warmup_updates=20, eval_interval=50, gate_interval=10, shift_step=20,
                outdir=dir_interrupted, use_shared_offline_checkpoints=False, device="cpu",
            )
        except InterruptedError:
            pass

        print("--- RUN 3: Resume from step 50 ---")
        current_run[0] = 3
        fr.OnlineBuffer.add = hook_add
        run_final_cell(
            dataset_id="halfcheetah-medium-v2", controller="capacity_gate",
            shift="obs_noise", severity=0.1, seed=42, online_steps=100,
            warmup_updates=20, eval_interval=50, gate_interval=10, shift_step=20,
            outdir=dir_interrupted, use_shared_offline_checkpoints=False, device="cpu",
        )

        fr.OnlineBuffer.add = orig_add

        print("\n--- STEP 51 COMPARISON ---")
        d1 = step51_data[1]
        d3 = step51_data[3]

        for k in d1:
            if isinstance(d1[k], np.ndarray):
                diff = np.max(np.abs(d1[k] - d3[k]))
                print(f"{k}: max diff = {diff}")
            else:
                print(f"{k}: RUN 1={d1[k]}, RUN 3={d3[k]}")

if __name__ == "__main__":
    pinpoint()
