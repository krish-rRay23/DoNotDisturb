import tempfile
from pathlib import Path
import torch
import adaptive_plasticity.final_runner as fr

def test_state_pt():
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        dir_in = base_dir / "in"

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
        opt_state = st["agent_state"]["actor_optimizer"]["state"]
        param_0_step = list(opt_state.values())[0]["step"]
        print("At step 50 state.pt: online_step =", st["online_step"], "update_step =", st["update_step"])
        print("At step 50 state.pt: actor_optimizer step =", param_0_step)

if __name__ == "__main__":
    test_state_pt()
