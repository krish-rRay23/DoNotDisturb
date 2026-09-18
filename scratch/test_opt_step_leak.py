import tempfile
from pathlib import Path
import torch
import adaptive_plasticity.final_runner as fr

def test_leak():
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir)
        dir_in = outdir / "in"

        print("--- PART 1: Crash mock ---")
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

        state_file = list(dir_in.rglob("state.pt"))[0]
        st = torch.load(state_file, map_location="cpu", weights_only=False)
        opt_st_50 = list(st["agent_state"]["actor_optimizer"]["state"].values())[0]["step"]
        print(f"State.pt saved at step 50 has optimizer step = {opt_st_50}")

        print("--- PART 2: Resume ---")
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

        ckpt_in = torch.load(dir_in / res_in["output_name"] / "checkpoint.pt", map_location="cpu", weights_only=False)
        opt_st_100 = list(ckpt_in["actor_optimizer"]["state"].values())[0]["step"]
        print(f"Checkpoint saved at step 100 after resume has optimizer step = {opt_st_100}")

if __name__ == "__main__":
    test_leak()
