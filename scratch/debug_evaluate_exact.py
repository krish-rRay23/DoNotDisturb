import torch
import adaptive_plasticity.final_runner as fr
from adaptive_plasticity.iql import IQLAgent

def test_eval_on_checkpoints():
    ckpt_un = torch.load("c:/Users/krish/OneDrive/Desktop/adaptive-plasticity-rl/scratch/test_un_ckpt.pt", map_location="cpu", weights_only=False) if False else None

    # Let's run UN and IN and immediately test evaluate on loaded checkpoint dicts
    import tempfile
    from pathlib import Path

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

        un_path = dir_un / res_un["output_name"] / "checkpoint.pt"
        in_path = dir_in / res_in["output_name"] / "checkpoint.pt"

        ckpt_un_data = torch.load(un_path, map_location="cpu", weights_only=False)
        ckpt_in_data = torch.load(in_path, map_location="cpu", weights_only=False)

        agent_un = IQLAgent(17, 6, device="cpu")
        agent_un.load_state_dict(ckpt_un_data)

        agent_in = IQLAgent(17, 6, device="cpu")
        agent_in.load_state_dict(ckpt_in_data)

        print("\nEvaluating UN agent on seed=92...")
        ret_un = fr._evaluate(agent_un, "halfcheetah-medium-v2", seed=92, episodes=10, device="cpu")

        print("Evaluating IN agent on seed=92...")
        ret_in = fr._evaluate(agent_in, "halfcheetah-medium-v2", seed=92, episodes=10, device="cpu")

        print(f"\nFinal Offline Eval Comparison: UN={ret_un}, IN={ret_in}, diff={abs(ret_un - ret_in)}")

if __name__ == "__main__":
    test_eval_on_checkpoints()
