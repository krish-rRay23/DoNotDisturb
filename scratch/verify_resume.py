"""Comprehensive verification of interrupted-vs-uninterrupted run resumability."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
import numpy as np
import torch

from adaptive_plasticity.final_runner import run_final_cell, _evaluate
import adaptive_plasticity.final_runner as fr


def verify_resumability():
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        dir_uninterrupted = base_dir / "uninterrupted"
        dir_interrupted = base_dir / "interrupted"

        print("1. Running uninterrupted cell (100 online steps)...")
        res_uninterrupted = run_final_cell(
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
            outdir=dir_uninterrupted,
            use_shared_offline_checkpoints=False,
            device="cpu",
        )

        print("2. Simulating crash right after step 50 checkpoint is written...")
        original_eval = fr._evaluate
        saved_evals = {"count": 0}

        def mock_eval(*args, **kwargs):
            res = original_eval(*args, **kwargs)
            saved_evals["count"] += 1
            return res

        fr._evaluate = mock_eval

        original_add = fr.OnlineBuffer.add
        def mock_add(self, *args, **kwargs):
            if saved_evals["count"] == 1:
                raise InterruptedError("Simulated sudden crash right after step 50 checkpoint was written!")
            return original_add(self, *args, **kwargs)

        fr.OnlineBuffer.add = mock_add

        try:
            run_final_cell(
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
                outdir=dir_interrupted,
                use_shared_offline_checkpoints=False,
                device="cpu",
            )
        except InterruptedError:
            print("Successfully caught simulated crash after step 50 checkpoint!")
        finally:
            fr._evaluate = original_eval
            fr.OnlineBuffer.add = original_add

        # Verify state.pt exists for interrupted run
        name_in = "FINAL-halfcheetah-medium-v2-capacity_gate-obs_noise-sev0.1-seed42"
        state_path = dir_interrupted / name_in / "state.pt"
        assert state_path.is_file(), "state.pt was not saved before crash!"

        print("3. Resuming interrupted run from step 50 checkpoint to 100 steps...")
        res_resumed = run_final_cell(
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
            outdir=dir_interrupted,
            use_shared_offline_checkpoints=False,
            device="cpu",
        )

        print("4. Comparing uninterrupted vs resumed cell outputs...")
        
        # Check summary returns and metrics
        assert res_uninterrupted["applied_intervention_steps"] == res_resumed["applied_intervention_steps"], \
            f"Applied intervention steps mismatch!"
        assert res_uninterrupted["online_steps"] == res_resumed["online_steps"] == 100
        assert res_uninterrupted["update_steps"] == res_resumed["update_steps"] == 100

        # Check checkpoints (checkpoint.pt)
        name_un = res_uninterrupted["output_name"]
        ckpt_un = torch.load(dir_uninterrupted / name_un / "checkpoint.pt", map_location="cpu", weights_only=False)
        ckpt_in = torch.load(dir_interrupted / name_in / "checkpoint.pt", map_location="cpu", weights_only=False)

        print("\n--- CHECKPOINT WEIGHT COMPARISON ---")
        for key in ckpt_un:
            assert key in ckpt_in, f"Missing key {key} in resumed checkpoint"
            t1 = ckpt_un[key]
            t2 = ckpt_in[key]
            if isinstance(t1, torch.Tensor):
                diff = torch.max(torch.abs(t1 - t2)).item()
                print(f"Checkpoint key {key} diff: {diff}")
                assert torch.equal(t1, t2), f"Tensor mismatch for key {key}"

        print("\n--- GATE LOG SCIENTIFIC COMPARISON ---")
        gate_un = (dir_uninterrupted / name_un / f"gate_{name_un}.jsonl").read_text()
        gate_in = (dir_interrupted / name_in / f"gate_{name_in}.jsonl").read_text()
        recs_un = [json.loads(l) for l in gate_un.strip().split("\n")]
        recs_in = [json.loads(l) for l in gate_in.strip().split("\n")]
        assert len(recs_un) == len(recs_in), f"Gate record count mismatch: {len(recs_un)} vs {len(recs_in)}"

        for idx, (r1, r2) in enumerate(zip(recs_un, recs_in)):
            # Compare all non-wallclock fields
            for k in r1:
                if k in ("timestamp", "eval_cost_seconds"):
                    continue
                assert r1[k] == r2[k], f"Mismatch at record {idx} for key {k}: {r1[k]} vs {r2[k]}"
        print("Gate records match 100% bit-identically across all scientific metrics!")

        assert res_uninterrupted["returns"] == res_resumed["returns"], \
            f"Returns mismatch: {res_uninterrupted['returns']} vs {res_resumed['returns']}"
        assert res_uninterrupted["normalized"] == res_resumed["normalized"], \
            f"Normalized returns mismatch!"

        print("RESUMABILITY VERIFICATION PASSED 100% BIT-IDENTICAL!")


if __name__ == "__main__":
    import traceback
    try:
        verify_resumability()
    except Exception as e:
        print("EXCEPTION OCCURRED:", e)
        traceback.print_exc()
        raise
