import math
import torch
import tempfile
from pathlib import Path
import adaptive_plasticity.final_runner as fr

def debug():
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir)
        print("--- RUN 1 ---", flush=True)
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
            outdir=outdir,
            use_shared_offline_checkpoints=False,
            device="cpu",
        )
        
        print("Outdir files:", list(outdir.rglob("*")), flush=True)
        state_files = list(outdir.rglob("state.pt"))
        if not state_files:
            print("No state.pt found!", flush=True)
            return
        state_file = state_files[0]
        st = torch.load(state_file, map_location="cpu", weights_only=False)
        print("Saved state rank_anchor:", st["rank_anchor"], flush=True)
        print("Saved state gate_records len:", len(st["gate_records"]), flush=True)
        for r in st["gate_records"]:
            print("  record step", r["step"], "rank_anchor:", r["rank_anchor"], flush=True)

        gate_log_file = list(outdir.glob("*/gate_*.jsonl"))[0]
        print("\nGate log file content BEFORE resume:", flush=True)
        print(gate_log_file.read_text(), flush=True)

        print("--- RESUME TO 70 STEPS ---", flush=True)
        fr.run_final_cell(
            dataset_id="halfcheetah-medium-v2",
            controller="capacity_gate",
            shift="obs_noise",
            severity=0.1,
            seed=42,
            online_steps=70,
            warmup_updates=20,
            eval_interval=50,
            gate_interval=10,
            shift_step=20,
            outdir=outdir,
            use_shared_offline_checkpoints=False,
            device="cpu",
        )

        print("\nGate log file content AFTER resume:", flush=True)
        print(gate_log_file.read_text(), flush=True)

if __name__ == "__main__":
    debug()
