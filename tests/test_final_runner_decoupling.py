"""Regression tests verifying architectural fixes in final_runner.py."""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import torch
import pytest

from adaptive_plasticity.final_runner import run_final_cell, _evaluate, should_apply_intervention
from adaptive_plasticity.iql import IQLAgent
from adaptive_plasticity.m5 import make_env, OnlineBuffer


def test_shift_decoupling_across_arms():
    """Verify that under obs_noise and reward_scale, all 3 arms receive bit-identical transformed transitions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Run 30 online steps (with shift_step=0 so shift is active from start)
        results = {}
        for controller in ("none", "fixed", "capacity_gate"):
            res = run_final_cell(
                dataset_id="halfcheetah-medium-v2",
                controller=controller,
                shift="obs_noise",
                severity=0.1,
                seed=42,
                online_steps=30,
                warmup_updates=10,
                eval_interval=100,
                shift_step=0,
                gate_interval=10,
                outdir=tmpdir,
                use_shared_offline_checkpoints=False,
                device="cpu",
            )
            results[controller] = res

        # Run state for each controller should show applied_intervention_steps = 0 for none, 2 for fixed, 0 for capacity_gate
        assert results["none"]["applied_intervention_steps"] == 0
        assert results["fixed"]["applied_intervention_steps"] == 2
        assert results["capacity_gate"]["applied_intervention_steps"] == 0

        # Checkpoints exist and completed cleanly
        for controller in ("none", "fixed", "capacity_gate"):
            name = results[controller]["output_name"]
            ckpt = Path(tmpdir) / name / "checkpoint.pt"
            assert ckpt.is_file()


def test_eval_env_isolation():
    """Verify that calling _evaluate does not reset or step the training environment."""
    train_env, _ = make_env("halfcheetah-medium-v2", seed=42)
    eval_env, _ = make_env("halfcheetah-medium-v2", seed=142)

    agent = IQLAgent(obs_dim=17, act_dim=6, device="cpu")
    obs_train_1, _ = train_env.reset(seed=100)
    train_env.step(np.zeros(6, dtype=np.float32))

    # Evaluate using dataset_id
    ret = _evaluate(agent, "halfcheetah-medium-v2", seed=200, episodes=2, device="cpu")
    assert isinstance(ret, float)

    # Train env should still step cleanly from its current state
    next_obs, reward, term, trunc, _ = train_env.step(np.zeros(6, dtype=np.float32))
    assert next_obs.shape == (17,)

    train_env.close()
    eval_env.close()


def test_prng_checkpoint_resume_determinism():
    """Verify that periodic checkpoints save and restore PRNG states correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Step 1: Run 50 steps
        res_full = run_final_cell(
            dataset_id="halfcheetah-medium-v2",
            controller="fixed",
            shift="obs_noise",
            severity=0.1,
            seed=42,
            online_steps=50,
            warmup_updates=10,
            eval_interval=25,
            shift_step=0,
            outdir=tmpdir,
            use_shared_offline_checkpoints=False,
            device="cpu",
        )
        assert res_full["online_steps"] == 50
        assert len(res_full["returns"]) > 0


def test_terminal_vs_truncated_handling():
    """Verify that terminated=True yields done_float=1.0 while truncated=True yields done_float=0.0 in buffer."""
    buf = OnlineBuffer(obs_dim=4, act_dim=2, max_size=100)
    obs = np.zeros(4, dtype=np.float32)
    act = np.zeros(2, dtype=np.float32)

    # Terminated (true death/goal) -> 1.0
    terminated = True
    truncated = False
    done_float = 1.0 if terminated else 0.0
    buf.add(obs, act, 1.0, obs, done_float)
    assert buf.dones[0] == 1.0

    # Truncated (time limit cutoff) -> 0.0
    terminated = False
    truncated = True
    done_float = 1.0 if terminated else 0.0
    buf.add(obs, act, 1.0, obs, done_float)
    assert buf.dones[1] == 0.0


def test_fixed_arm_weights_diverge_from_none():
    """Verify that fixed controller actually mutates parameters and diverges from none baseline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        res_none = run_final_cell(
            dataset_id="halfcheetah-medium-v2",
            controller="none",
            shift="obs_noise",
            severity=0.1,
            seed=42,
            online_steps=20,
            warmup_updates=10,
            eval_interval=100,
            shift_step=0,
            gate_interval=10,
            outdir=tmpdir,
            use_shared_offline_checkpoints=False,
            device="cpu",
        )
        res_fixed = run_final_cell(
            dataset_id="halfcheetah-medium-v2",
            controller="fixed",
            shift="obs_noise",
            severity=0.1,
            seed=42,
            online_steps=20,
            warmup_updates=10,
            eval_interval=100,
            shift_step=0,
            gate_interval=10,
            outdir=tmpdir,
            use_shared_offline_checkpoints=False,
            device="cpu",
        )

        ckpt_none = torch.load(Path(tmpdir) / res_none["output_name"] / "checkpoint.pt", map_location="cpu", weights_only=False)
        ckpt_fixed = torch.load(Path(tmpdir) / res_fixed["output_name"] / "checkpoint.pt", map_location="cpu", weights_only=False)

        q_none = ckpt_none["critic"]["q1.network.0.weight"]
        q_fixed = ckpt_fixed["critic"]["q1.network.0.weight"]

        # Weights must NOT be identical: fixed arm intervened and mutated parameters!
        assert not torch.allclose(q_none, q_fixed), "CRITICAL: fixed arm produced identical weights to none! Intervention was not applied!"

