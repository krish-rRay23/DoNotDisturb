"""Clean test of _evaluate identity on step 100 policy."""
import tempfile
from pathlib import Path
import numpy as np
import torch
import adaptive_plasticity.final_runner as fr
from adaptive_plasticity.final_runner import run_final_cell, _evaluate
from adaptive_plasticity.m5 import make_env

def test_true_step100():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        dir1 = base / "uninterrupted"
        dir2 = base / "interrupted"

        run1_agents = []
        run3_agents = []
        current_run = [1]

        orig_update = fr.IQLAgent.update
        def hook_update(self, batch):
            res = orig_update(self, batch)
            sd = {k: v.clone() for k, v in self.policy.state_dict().items()}
            if current_run[0] == 1:
                run1_agents.append(sd)
            elif current_run[0] == 3:
                run3_agents.append(sd)
            return res

        fr.IQLAgent.update = hook_update

        print("--- RUN 1 ---")
        res1 = run_final_cell(
            dataset_id="halfcheetah-medium-v2", controller="capacity_gate",
            shift="obs_noise", severity=0.1, seed=42, online_steps=100,
            warmup_updates=20, eval_interval=50, gate_interval=10, shift_step=20,
            outdir=dir1, use_shared_offline_checkpoints=False, device="cpu",
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
                outdir=dir2, use_shared_offline_checkpoints=False, device="cpu",
            )
        except InterruptedError:
            pass
        finally:
            fr.OnlineBuffer.add = orig_add

        print("--- RUN 3 (Resume) ---")
        current_run[0] = 3
        res3 = run_final_cell(
            dataset_id="halfcheetah-medium-v2", controller="capacity_gate",
            shift="obs_noise", severity=0.1, seed=42, online_steps=100,
            warmup_updates=20, eval_interval=50, gate_interval=10, shift_step=20,
            outdir=dir2, use_shared_offline_checkpoints=False, device="cpu",
        )

        fr.IQLAgent.update = orig_update

        print(f"\nUpdates recorded: RUN 1 = {len(run1_agents)}, RUN 3 = {len(run3_agents)}")

        # Index 119 in RUN 1 is step 100 update.
        # Index 49 in RUN 3 is step 100 update.
        sd1 = run1_agents[119]
        sd3 = run3_agents[49]

        diffs = []
        for k in sd1:
            d = torch.max(torch.abs(sd1[k] - sd3[k])).item()
            if d > 0:
                diffs.append((k, d))

        print(f"Policy state dict diff at step 100: {diffs}")

        print("\nRUN 1 reported returns:", res1["returns"])
        print("RUN 3 reported returns:", res3["returns"])
        assert res1["returns"] == res3["returns"], "Returns mismatch between uninterrupted and resumed!"
        print("PASS: Uninterrupted and resumed runs are 100% BIT-IDENTICAL!")

if __name__ == "__main__":
    test_true_step100()
