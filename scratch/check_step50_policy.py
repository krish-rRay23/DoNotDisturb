"""Compare policy state dict at step 50 in UN vs state.pt in IN."""
import tempfile
from pathlib import Path
import torch
import adaptive_plasticity.final_runner as fr
from adaptive_plasticity.final_runner import run_final_cell

def check_step50_policy():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        dir1 = base / "uninterrupted"
        dir2 = base / "interrupted"

        step50_policy_un = []

        orig_eval = fr._evaluate
        def hook_eval(agent, env, *, seed, episodes, device):
            res = orig_eval(agent, env, seed=seed, episodes=episodes, device=device)
            # Eval runs at step 50
            if seed == 42:
                step50_policy_un.append({k: v.clone() for k, v in agent.policy.state_dict().items()})
            return res

        fr._evaluate = hook_eval

        print("--- RUN 1 ---")
        run_final_cell(
            dataset_id="halfcheetah-medium-v2", controller="capacity_gate",
            shift="obs_noise", severity=0.1, seed=42, online_steps=100,
            warmup_updates=20, eval_interval=50, gate_interval=10, shift_step=20,
            outdir=dir1, use_shared_offline_checkpoints=False, device="cpu",
        )

        print("--- RUN 2 (Crash) ---")
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
            fr._evaluate = orig_eval

        state_path = dir2 / "FINAL-halfcheetah-medium-v2-capacity_gate-obs_noise-sev0.1-seed42" / "state.pt"
        saved = torch.load(state_path, map_location="cpu", weights_only=False)
        step50_policy_in = saved["agent_state"]["policy"]

        print("Comparing RUN 1 step 50 policy vs state.pt saved policy...")
        diffs = []
        for k in step50_policy_un[0]:
            d = torch.max(torch.abs(step50_policy_un[0][k] - step50_policy_in[k])).item()
            if d > 0:
                diffs.append((k, d))

        if diffs:
            print("POLICY DIFFERENCE AT STEP 50:", diffs)
        else:
            print("PASS: Policy at step 50 in RUN 1 and state.pt are 100% BIT-IDENTICAL!")

if __name__ == "__main__":
    check_step50_policy()
