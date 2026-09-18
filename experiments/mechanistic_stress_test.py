"""Exploratory Mechanistic Stress Test for CapacityGate System.

Verifies that the CapacityGate mechanism, hysteresis logic, telemetry logging,
and intervention invocation function correctly when genuine representation-capacity
degradation is explicitly induced.

Label: Exploratory Mechanistic Stress Test (separate from 135-run confirmatory study).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from adaptive_plasticity.capacity_gate import (
    CapacityGateConfig,
    CapacityGateController,
    build_fixed_probe,
)
from adaptive_plasticity.distribution_shifts import apply_observation_noise
from adaptive_plasticity.final_runner import (
    _REF_RETURNS,
    _evaluate,
    apply_regime_transform,
    sample_mixed_batch_fast,
    should_apply_intervention,
)
from adaptive_plasticity.iql import IQLAgent
from adaptive_plasticity.m5 import OnlineBuffer, load_offline_data, make_env
from adaptive_plasticity.offline_cache import get_offline_checkpoint_path
from adaptive_plasticity.reproducibility import seed_everything


def run_stress_cell(
    *,
    dataset_id: str = "hopper-medium-v2",
    seed: int = 0,
    arm: str = "capacity_gate",  # "capacity_gate" or "none" (passive)
    online_steps: int = 10_000,
    stress_step: int = 2_500,
    dormancy_stress_fraction: float = 0.35,  # Force 35% dormancy at stress_step
    eval_interval: int = 1_000,
    eval_episodes: int = 5,
    gate_interval: int = 500,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    outdir: Path = Path("results/stress_test"),
) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    device_obj = torch.device(device)

    seed_everything(seed, deterministic=True)
    torch.use_deterministic_algorithms(True, warn_only=True)

    offline_data = load_offline_data(dataset_id)
    obs_dim = int(offline_data["observations"].shape[1])
    act_dim = int(offline_data["actions"].shape[1])
    offline_count = int(offline_data["observations"].shape[0])

    agent = IQLAgent(obs_dim, act_dim, device=device)
    env, rng = make_env(dataset_id, seed=seed)
    noise_rng = np.random.RandomState(seed + 7919)
    online_buf = OnlineBuffer(obs_dim=obs_dim, act_dim=act_dim)
    probe = build_fixed_probe(offline_data["observations"], n=256, seed=seed)

    # Load shared pre-trained offline checkpoint if available
    shared_ckpt = get_offline_checkpoint_path(dataset_id, seed)
    if shared_ckpt.is_file():
        ckpt_data = torch.load(shared_ckpt, map_location=device_obj, weights_only=False)
        agent.load_state_dict(ckpt_data["agent_state"])

    gate = CapacityGateController(config=CapacityGateConfig(active_severity=0.1))
    rank_anchor = float(gate.capture_anchor_from_policy(agent.policy, probe))

    offline_storage: dict[str, Any] = offline_data
    if device.startswith("cuda"):
        offline_storage = {k: torch.as_tensor(v, device=device_obj) for k, v in offline_data.items()}

    returns: list[float] = []
    normalized: list[float] = []
    applied_count = 0
    gate_records: list[dict[str, Any]] = []

    obs, _ = env.reset(seed=seed)
    t0 = time.time()

    for step in range(1, online_steps + 1):
        # Introduce explicit capacity stress at stress_step
        if step == stress_step:
            print(f"[{arm}] Inducing explicit representation capacity stress at step {step}...", flush=True)
            with torch.no_grad():
                # Zero out 35% of first-layer backbone neurons and apply negative bias
                n_units = int(agent.policy.backbone.network[0].weight.shape[0] * dormancy_stress_fraction)
                agent.policy.backbone.network[0].weight.data[:n_units] = 0.0
                agent.policy.backbone.network[0].bias.data[:n_units] = -100.0

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device_obj).unsqueeze(0)
        with torch.no_grad():
            action = agent.policy.act(obs_t, deterministic=True).squeeze(0).cpu().numpy()
        next_obs_raw, reward_raw, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)

        # Trigger logic
        trigger = False
        if arm == "capacity_gate":
            trigger = should_apply_intervention(
                controller="capacity_gate", shift_active=(step >= stress_step), gate_on=gate.gate_on
            )

        if trigger:
            next_obs, reward_used = apply_regime_transform(
                next_obs_raw, float(reward_raw), shift="obs_noise", severity=0.1, noise_rng=noise_rng
            )
            applied_count += 1
        else:
            next_obs, reward_used = np.asarray(next_obs_raw).copy(), float(reward_raw)

        online_buf.add(np.asarray(obs).copy(), action, reward_used, np.asarray(next_obs).copy(), 1.0 if done else 0.0)
        obs = next_obs
        if done:
            obs, _ = env.reset(seed=seed + step)

        # Update gradient
        mixed, _, _ = sample_mixed_batch_fast(
            online_buf=online_buf,
            offline_data=offline_storage,
            offline_count=offline_count,
            batch_size=256,
            online_ratio=0.5,
            rng=rng,
            device=device_obj,
        )
        agent.update(mixed)

        # Evaluate gate
        if step % gate_interval == 0:
            rec = gate.update_from_policy(agent.policy, probe, step=step)
            rec["step"] = step
            rec["arm"] = arm
            rec["shift_active"] = step >= stress_step
            gate_records.append(rec)

        # Evaluate policy
        if step % eval_interval == 0:
            ret = _evaluate(agent, env, seed=seed + step, episodes=eval_episodes, device=device_obj)
            r_min, r_max = _REF_RETURNS[dataset_id]
            norm = 100.0 * (ret - r_min) / (r_max - r_min)
            returns.append(ret)
            normalized.append(norm)

    env.close()
    runtime = time.time() - t0

    return {
        "arm": arm,
        "dataset_id": dataset_id,
        "seed": seed,
        "online_steps": online_steps,
        "stress_step": stress_step,
        "applied_count": applied_count,
        "returns": returns,
        "normalized": normalized,
        "rank_anchor": rank_anchor,
        "gate_records": gate_records,
        "gate_on_activations": sum(1 for r in gate_records if r.get("gate_on")),
        "runtime_seconds": round(runtime, 2),
    }


def main():
    print("=== Running Exploratory Mechanistic Stress Test ===", flush=True)
    outdir = Path("results/stress_test")
    outdir.mkdir(parents=True, exist_ok=True)

    res_gate = run_stress_cell(arm="capacity_gate", outdir=outdir)
    res_none = run_stress_cell(arm="none", outdir=outdir)

    print("\n--- STRESS TEST RESULTS ---")
    print(f"CapacityGate Arm:")
    print(f"  Applied intervention steps: {res_gate['applied_count']}")
    print(f"  Gate ON evaluations: {res_gate['gate_on_activations']} / {len(res_gate['gate_records'])}")
    print(f"  Final Normalized Score: {res_gate['normalized'][-1]:.2f}")

    print(f"\nNone (Passive) Arm:")
    print(f"  Applied intervention steps: {res_none['applied_count']}")
    print(f"  Final Normalized Score: {res_none['normalized'][-1]:.2f}")

    # Log telemetry
    summary = {
        "test_name": "exploratory_mechanistic_stress_test",
        "condition": "Explicit 35% neuron dormancy injection at step 2500",
        "capacity_gate": res_gate,
        "none_baseline": res_none,
    }
    (outdir / "stress_test_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    # Plot trajectories
    steps_eval = list(range(1000, 10001, 1000))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(steps_eval, res_gate["normalized"], label="CapacityGate", color="blue", marker="o")
    axes[0].plot(steps_eval, res_none["normalized"], label="None (Passive)", color="gray", linestyle="--", marker="x")
    axes[0].axvline(2500, color="red", linestyle=":", label="Capacity Stress Event")
    axes[0].set_title("Performance Recovery Under Capacity Stress")
    axes[0].set_xlabel("Online Step")
    axes[0].set_ylabel("Normalized Score")
    axes[0].legend()
    axes[0].grid(True)

    gate_steps = [r["step"] for r in res_gate["gate_records"]]
    dorms = [r["dormancy"] for r in res_gate["gate_records"]]
    gate_ons = [1.0 if r["gate_on"] else 0.0 for r in res_gate["gate_records"]]

    axes[1].plot(gate_steps, dorms, label="Dormancy D_t", color="orange", marker="s")
    axes[1].axhline(0.15, color="red", linestyle="--", label="dorm_on threshold (0.15)")
    axes[1].fill_between(gate_steps, 0, gate_ons, alpha=0.2, color="yellow", label="Gate ON")
    axes[1].set_title("CapacityGate Activation & Dormancy Telemetry")
    axes[1].set_xlabel("Online Step")
    axes[1].set_ylabel("Dormancy / Gate State")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    plots_dir = Path("plots/stress_test")
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / "mechanistic_stress_test.png"
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"\nSaved stress test plot to {plot_path}")
    print(f"Saved stress test summary to {outdir / 'stress_test_summary.json'}")


if __name__ == "__main__":
    main()
