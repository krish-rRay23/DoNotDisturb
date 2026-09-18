import tempfile
from pathlib import Path
import torch
import adaptive_plasticity.final_runner as fr

def test_ckpt_diff():
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        dir_un = base_dir / "un"
        dir_in = base_dir / "in"

        print("1. Running UN (100 steps)...")
        res_un = fr.run_final_cell(
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
            outdir=dir_un,
            use_shared_offline_checkpoints=False,
            device="cpu",
        )

        print("2. Running IN (crash at step 51)...")
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
                online_steps=100,
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

        print("3. Resuming IN (from step 50 to 100)...")
        res_in = fr.run_final_cell(
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
            outdir=dir_in,
            use_shared_offline_checkpoints=False,
            device="cpu",
        )

        ckpt_un = torch.load(dir_un / res_un["output_name"] / "checkpoint.pt", map_location="cpu", weights_only=False)
        ckpt_in = torch.load(dir_in / res_in["output_name"] / "checkpoint.pt", map_location="cpu", weights_only=False)

        print("\nComparing Checkpoint Weights:")
        diffs = []
        for k in ckpt_un:
            t1 = ckpt_un[k]
            t2 = ckpt_in[k]
            if isinstance(t1, torch.Tensor):
                d = torch.max(torch.abs(t1 - t2)).item()
                if d > 0:
                    diffs.append((k, d))
        print("Param diffs:", diffs)
        print("Returns UN:", res_un["returns"])
        print("Returns IN:", res_in["returns"])

if __name__ == "__main__":
    test_ckpt_diff()
