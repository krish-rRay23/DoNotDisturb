"""Test _evaluate identity on identical step 100 policies."""
import tempfile
from pathlib import Path
import numpy as np
import torch
import adaptive_plasticity.final_runner as fr
from adaptive_plasticity.final_runner import run_final_cell, _evaluate
from adaptive_plasticity.m5 import make_env

def test_eval_identity():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        dir1 = base / "uninterrupted"
        dir2 = base / "interrupted"

        step100_agents = []
        current_run = [1]

        orig_update = fr.IQLAgent.update
        def hook_update(self, batch):
            res = orig_update(self, batch)
            if current_run[0] == 1 and len(step100_agents) == 119:
                step100_agents.append(self)
            elif current_run[0] == 1:
                step100_agents.append(None)
            elif current_run[0] == 3 and len(step100_agents) == 120:
                step100_agents.append(self)
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

        print("\nRUN 1 reported returns:", res1["returns"])
        print("RUN 3 reported returns:", res3["returns"])

        agent1 = step100_agents[119]
        agent3 = step100_agents[-1]

        eval_env1, _ = make_env("halfcheetah-medium-v2", seed=42 + 10000)
        ret1 = _evaluate(agent1, eval_env1, seed=42 + 50, episodes=10, device="cpu")

        eval_env2, _ = make_env("halfcheetah-medium-v2", seed=42 + 10000)
        ret2 = _evaluate(agent3, eval_env2, seed=42 + 50, episodes=10, device="cpu")

        print(f"Direct _evaluate on agent1 (RUN 1): {ret1}")
        print(f"Direct _evaluate on agent3 (RUN 3): {ret2}")

        sd1 = agent1.state_dict()
        sd3 = agent3.state_dict()
        print("\n--- AGENT STATE DICT COMPARISON ---")
        for k in sd1:
            if isinstance(sd1[k], torch.Tensor):
                diff = torch.max(torch.abs(sd1[k] - sd3[k])).item()
                if diff > 0:
                    print(f"Mismatch in key {k}: max diff = {diff}")

if __name__ == "__main__":
    test_eval_identity()
