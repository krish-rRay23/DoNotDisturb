import torch
import numpy as np
import tempfile
from pathlib import Path
import adaptive_plasticity.final_runner as fr

def compare_step_by_step():
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        dir_un = base_dir / "un"
        dir_in = base_dir / "in"

        print("--- UNINTERRUPTED RUN (up to step 60) ---")
        fr.run_final_cell(
            dataset_id="halfcheetah-medium-v2",
            controller="capacity_gate",
            shift="obs_noise",
            severity=0.1,
            seed=42,
            online_steps=60,
            warmup_updates=20,
            eval_interval=50,
            gate_interval=10,
            shift_step=20,
            outdir=dir_un,
            use_shared_offline_checkpoints=False,
            device="cpu",
        )

        # Now simulate run up to 50, then resume to 60
        print("--- INTERRUPTED RUN (up to step 50) ---")
        saved_evals = {"count": 0}
        orig_eval = fr._evaluate
        def mock_eval(*args, **kwargs):
            res = orig_eval(*args, **kwargs)
            saved_evals["count"] += 1
            return res
        fr._evaluate = mock_eval

        orig_add = fr.OnlineBuffer.add
        def mock_add(self, *args, **kwargs):
            if saved_evals["count"] == 1:
                raise InterruptedError("crash!")
            return orig_add(self, *args, **kwargs)
        fr.OnlineBuffer.add = mock_add

        try:
            fr.run_final_cell(
                dataset_id="halfcheetah-medium-v2",
                controller="capacity_gate",
                shift="obs_noise",
                severity=0.1,
                seed=42,
                online_steps=60,
                warmup_updates=20,
                eval_interval=50,
                gate_interval=10,
                shift_step=20,
                outdir=dir_in,
                use_shared_offline_checkpoints=False,
                device="cpu",
            )
        except InterruptedError:
            pass
        finally:
            fr._evaluate = orig_eval
            fr.OnlineBuffer.add = orig_add

        print("--- RESUMED RUN (from step 50 to 60) ---")
        fr.run_final_cell(
            dataset_id="halfcheetah-medium-v2",
            controller="capacity_gate",
            shift="obs_noise",
            severity=0.1,
            seed=42,
            online_steps=60,
            warmup_updates=20,
            eval_interval=50,
            gate_interval=10,
            shift_step=20,
            outdir=dir_in,
            use_shared_offline_checkpoints=False,
            device="cpu",
        )

        # Compare gate logs
        un_log = list(dir_un.glob("*/gate_*.jsonl"))[0].read_text().strip().split("\n")
        in_log = list(dir_in.glob("*/gate_*.jsonl"))[0].read_text().strip().split("\n")

        print("\nUN LOG:")
        for line in un_log:
            print(" ", line)

        print("\nIN LOG:")
        for line in in_log:
            print(" ", line)

if __name__ == "__main__":
    compare_step_by_step()
