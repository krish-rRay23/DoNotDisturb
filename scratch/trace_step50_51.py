"""Trace every object at step 50 and step 51 in RUN 1 vs RUN 3."""
import tempfile
from pathlib import Path
import numpy as np
import torch
import adaptive_plasticity.final_runner as fr
from adaptive_plasticity.final_runner import run_final_cell

def trace_all():
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        dir_uninterrupted = base_dir / "uninterrupted"
        dir_interrupted = base_dir / "interrupted"

        run1_log = []
        run3_log = []
        current_run = [1]

        orig_step = fr.make_env

        # Hook inside final_runner by wrapping apply_regime_transform and env.step
        orig_trans = fr.apply_regime_transform
        def hook_trans(obs, reward, *, shift, severity, noise_rng):
            res = orig_trans(obs, reward, shift=shift, severity=severity, noise_rng=noise_rng)
            rng_sample = noise_rng.get_state()[1][0] # First int of state array
            entry = f"trans: in_obs={obs[:2]}, out_obs={res[0][:2]}, noise_rng_sample={rng_sample}"
            if current_run[0] == 1:
                run1_log.append(entry)
            elif current_run[0] == 3:
                run3_log.append(entry)
            return res

        fr.apply_regime_transform = hook_trans

        print("--- RUN 1 ---")
        run_final_cell(
            dataset_id="halfcheetah-medium-v2", controller="capacity_gate",
            shift="obs_noise", severity=0.1, seed=42, online_steps=100,
            warmup_updates=20, eval_interval=50, gate_interval=10, shift_step=20,
            outdir=dir_uninterrupted, use_shared_offline_checkpoints=False, device="cpu",
        )

        print("--- RUN 2 (Crash) ---")
        current_run[0] = 2
        orig_add = fr.OnlineBuffer.add
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

        print("--- RUN 3 (Resume) ---")
        current_run[0] = 3
        run_final_cell(
            dataset_id="halfcheetah-medium-v2", controller="capacity_gate",
            shift="obs_noise", severity=0.1, seed=42, online_steps=100,
            warmup_updates=20, eval_interval=50, gate_interval=10, shift_step=20,
            outdir=dir_interrupted, use_shared_offline_checkpoints=False, device="cpu",
        )

        fr.apply_regime_transform = orig_trans

        print(f"\nTotal transform calls: RUN 1 = {len(run1_log)}, RUN 3 = {len(run3_log)}")

        # Print transform calls around step 50-55
        # In RUN 1, shift starts at step 20.
        print("\nRUN 1 transform calls [28:35]:")
        for i, entry in enumerate(run1_log[28:35], 29):
            print(f"  Step {i}: {entry}")

        print("\nRUN 3 transform calls [0:7]:")
        for i, entry in enumerate(run3_log[0:7], 1):
            print(f"  Call {i}: {entry}")

if __name__ == "__main__":
    trace_all()
