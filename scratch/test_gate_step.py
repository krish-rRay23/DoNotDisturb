"""Print weight norm and update step around step 50-60 in UN vs RESUMED."""
import tempfile
from pathlib import Path
import numpy as np
import torch
import adaptive_plasticity.final_runner as fr
from adaptive_plasticity.final_runner import run_final_cell

def test_steps():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        dir1 = base / "uninterrupted"
        dir2 = base / "interrupted"

        un_norms = []
        re_norms = []
        mode = [1]

        orig_update = fr.IQLAgent.update
        def hook_update(self, batch):
            res = orig_update(self, batch)
            norm = float(torch.norm(self.policy.mean_head.weight).item())
            if mode[0] == 1:
                un_norms.append(norm)
            elif mode[0] == 3:
                re_norms.append(norm)
            return res

        fr.IQLAgent.update = hook_update

        print("--- RUN 1 ---")
        run_final_cell(
            dataset_id="halfcheetah-medium-v2", controller="capacity_gate",
            shift="obs_noise", severity=0.1, seed=42, online_steps=100,
            warmup_updates=20, eval_interval=50, gate_interval=10, shift_step=20,
            outdir=dir1, use_shared_offline_checkpoints=False, device="cpu",
        )

        print("--- RUN 2 (Crash) ---")
        mode[0] = 2
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
        mode[0] = 3
        run_final_cell(
            dataset_id="halfcheetah-medium-v2", controller="capacity_gate",
            shift="obs_noise", severity=0.1, seed=42, online_steps=100,
            warmup_updates=20, eval_interval=50, gate_interval=10, shift_step=20,
            outdir=dir2, use_shared_offline_checkpoints=False, device="cpu",
        )

        fr.IQLAgent.update = orig_update

        print(f"\nTotal updates recorded: RUN 1 = {len(un_norms)}, RUN 3 = {len(re_norms)}")

        # Print norms around update 65-75 in RUN 1 vs 0-15 in RUN 3
        print("\nRUN 1 norms around update 70 (online step 50):")
        for i in range(65, 75):
            print(f"  Update {i} (online {i-20}): {un_norms[i]:.8f}")

        print("\nRUN 3 norms (first 10 post-resume updates):")
        for i in range(10):
            print(f"  Post-resume update {i+1} (online {51+i}): {re_norms[i]:.8f}")

if __name__ == "__main__":
    test_steps()
