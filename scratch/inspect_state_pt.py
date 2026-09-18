"""Inspect saved state.pt contents."""
import tempfile
from pathlib import Path
import torch
import adaptive_plasticity.final_runner as fr
from adaptive_plasticity.final_runner import run_final_cell

def inspect_state():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        outdir = base / "interrupted"

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
                outdir=outdir, use_shared_offline_checkpoints=False, device="cpu",
            )
        except InterruptedError:
            pass
        finally:
            fr.OnlineBuffer.add = orig_add

        state_path = outdir / "FINAL-halfcheetah-medium-v2-capacity_gate-obs_noise-sev0.1-seed42" / "state.pt"
        saved = torch.load(state_path, map_location="cpu", weights_only=False)

        print("Keys in state.pt:", list(saved.keys()))
        print("online_step:", saved["online_step"])
        print("update_step:", saved["update_step"])
        print("applied_count:", saved["applied_count"])
        print("last_eval:", saved["last_eval"])
        print("returns:", saved["returns"])
        print("gate_records count:", len(saved["gate_records"]))
        print("gate_records steps:", [r["step"] for r in saved["gate_records"]])

if __name__ == "__main__":
    inspect_state()
