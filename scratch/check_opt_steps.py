import tempfile
from pathlib import Path
import torch
import adaptive_plasticity.final_runner as fr

def test_opt_steps():
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        dir_un = base_dir / "un"
        dir_in = base_dir / "in"

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

        step_un = list(ckpt_un["actor_optimizer"]["state"].values())[0]["step"]
        step_in = list(ckpt_in["actor_optimizer"]["state"].values())[0]["step"]

        print(f"UN optimizer step at step 100: {step_un}")
        print(f"IN optimizer step at step 100: {step_in}")

if __name__ == "__main__":
    test_opt_steps()
