"""Check if obs at step 50 matches current_obs restored in RUN 3."""
import tempfile
from pathlib import Path
import numpy as np
import torch
import adaptive_plasticity.final_runner as fr
from adaptive_plasticity.final_runner import run_final_cell

def check_obs_match():
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        dir_uninterrupted = base_dir / "uninterrupted"
        dir_interrupted = base_dir / "interrupted"

        run1_step50_obs = []

        orig_add = fr.OnlineBuffer.add
        def hook_add(self, obs, action, reward, next_obs, done):
            if len(self) == 49: # Step 50 input obs
                run1_step50_obs.append(obs.copy())
            if len(self) == 50: # Step 50 next_obs (which becomes step 51 input obs)
                run1_step50_obs.append(next_obs.copy())
            return orig_add(self, obs, action, reward, next_obs, done)

        fr.OnlineBuffer.add = hook_add

        print("--- RUN 1 ---")
        run_final_cell(
            dataset_id="halfcheetah-medium-v2", controller="capacity_gate",
            shift="obs_noise", severity=0.1, seed=42, online_steps=100,
            warmup_updates=20, eval_interval=50, gate_interval=10, shift_step=20,
            outdir=dir_uninterrupted, use_shared_offline_checkpoints=False, device="cpu",
        )

        print("--- RUN 2 (Crash) ---")
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
        finally:
            fr.OnlineBuffer.add = orig_add

        state_path = dir_interrupted / "FINAL-halfcheetah-medium-v2-capacity_gate-obs_noise-sev0.1-seed42" / "state.pt"
        saved_state = torch.load(state_path, map_location="cpu", weights_only=False)
        restored_obs = saved_state["env_state"]["current_obs"]

        print("RUN 1 step 50 input obs: ", run1_step50_obs[0][:4])
        print("RUN 1 step 50 next_obs:  ", run1_step50_obs[1][:4])
        print("Restored current_obs:    ", restored_obs[:4])

        diff = np.max(np.abs(run1_step50_obs[1] - restored_obs))
        print("Diff between RUN 1 step 50 next_obs and restored current_obs:", diff)

if __name__ == "__main__":
    check_obs_match()
