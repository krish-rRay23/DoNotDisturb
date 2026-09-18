import tempfile
from pathlib import Path
import torch
import numpy as np
import adaptive_plasticity.final_runner as fr
from adaptive_plasticity.capacity_gate import compute_capacity_metrics, build_fixed_probe

def trace_diff():
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        dir_un = base_dir / "un"
        dir_in = base_dir / "in"

        # Track effective rank per step
        un_ranks = {}
        in_ranks = {}

        probe = build_fixed_probe(np.zeros((100, 17), dtype=np.float32), n=256, seed=42)

        orig_update = fr.IQLAgent.update
        mode = ["un"]

        def hook_update(self, batch):
            res = orig_update(self, batch)
            m = compute_capacity_metrics(self.policy, probe)
            erank = m["effective_rank"]
            if mode[0] == "un":
                un_ranks[len(un_ranks)] = erank
            else:
                in_ranks[len(in_ranks)] = erank
            return res

        fr.IQLAgent.update = hook_update

        print("Running UN...")
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

        mode[0] = "in"
        print("Running IN interrupt...")
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

        print("Running IN resume...")
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

        fr.IQLAgent.update = orig_update

        print("\nUN Ranks len:", len(un_ranks))
        print("IN Ranks len:", len(in_ranks))
        print("\nStep-by-step Effective Rank after warmup:")
        # Warmup is 20 updates (indices 0..19)
        # Online updates 1..50 are indices 20..69
        # Online updates 51..60 are indices 70..79
        for i in range(20, min(len(un_ranks), len(in_ranks))):
            step = i - 19
            r_un = un_ranks[i]
            r_in = in_ranks[i]
            diff = abs(r_un - r_in)
            print(f"Update {i} (online step {step}): UN={r_un:.6f}, IN={r_in:.6f}, diff={diff:.6f}")

if __name__ == "__main__":
    trace_diff()
