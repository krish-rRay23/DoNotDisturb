"""Comparative step-by-step runner to find exact step where uninterrupted and resumed runs diverge."""
import tempfile
from pathlib import Path
import numpy as np
import torch
import adaptive_plasticity.final_runner as fr
from adaptive_plasticity.final_runner import run_final_cell

def step_compare():
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        dir_uninterrupted = base_dir / "uninterrupted"
        dir_interrupted = base_dir / "interrupted"

        print("--- RUN 1: Uninterrupted ---")
        un_actions = []
        un_obs = []
        un_weights = []

        # Hook into online step in final_runner
        orig_update = fr.IQLAgent.update
        def hook_un_update(self, batch):
            res = orig_update(self, batch)
            un_weights.append(self.policy.mean_head.weight.data.clone())
            return res

        fr.IQLAgent.update = hook_un_update

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
        re_weights = []
        def hook_re_update(self, batch):
            res = orig_update(self, batch)
            re_weights.append(self.policy.mean_head.weight.data.clone())
            return res

        fr.IQLAgent.update = hook_re_update

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
        re_weights.clear() # clear warmup weights from before crash
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

        fr.IQLAgent.update = orig_update

        print(f"Total updates recorded: Uninterrupted={len(un_weights)}, Resumed (after resume)={len(re_weights)}")
        # Note: uninterrupted had 20 warmup + 100 online = 120 updates.
        # Resumed after step 50 should have 50 updates (steps 51 to 100).
        # un_weights[70] corresponds to step 51 update (20 warmup + 50 online = 70).

        un_online_weights = un_weights[70:] # updates 70 to 119
        print(f"Comparing {len(un_online_weights)} vs {len(re_weights)} post-resume updates...")
        for idx, (w_un, w_re) in enumerate(zip(un_online_weights, re_weights)):
            step_num = 51 + idx
            diff = torch.max(torch.abs(w_un - w_re)).item()
            if diff > 0:
                print(f"DIVERGENCE at step {step_num} update! Max diff: {diff}")
                break
            else:
                if idx < 5 or idx > 45:
                    print(f"Step {step_num} update MATCH: diff = 0.0")

if __name__ == "__main__":
    step_compare()
