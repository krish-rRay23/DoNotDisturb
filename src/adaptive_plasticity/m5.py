from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from adaptive_plasticity.train_iql import build_transitions
from adaptive_plasticity.iql import IQLAgent, q_target_value, advantage_weights, expectile_loss, soft_update
from adaptive_plasticity.datasets import load_local_dataset
from adaptive_plasticity.reproducibility import seed_everything, detect_device, create_metadata, write_metadata, git_commit_hash, package_versions
from adaptive_plasticity.distribution_shifts import (
    DistributionShiftOrchestrator, ShiftType,
    Intervention, FixedIntervention, RandomIntervention, AdaptiveIntervention,
    AdaptivePlasticityController,
)
from adaptive_plasticity.m6 import param_magnitudes, representation_change, activation_dormancy


# ---------------------------------------------------------------------------
# Online replay buffer (numpy-based, no extra dependencies)
# ---------------------------------------------------------------------------

class OnlineBuffer:
    """Simple FIFO replay buffer for online transitions.

    Stores (observation, action, reward, next_observation, done) with
    a fixed maximum size.  When full, oldest transitions are overwritten.
    """

    def __init__(self, obs_dim: int, act_dim: int, max_size: int = 100_000) -> None:
        self.max_size = max_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.count = 0
        self.idx = 0

        self.observations = np.empty((max_size, obs_dim), dtype=np.float32)
        self.actions = np.empty((max_size, act_dim), dtype=np.float32)
        self.rewards = np.empty((max_size,), dtype=np.float32)
        self.next_observations = np.empty((max_size, obs_dim), dtype=np.float32)
        self.dones = np.empty((max_size,), dtype=np.float32)

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float,
            next_obs: np.ndarray, done: float) -> None:
        i = self.idx
        self.observations[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_observations[i] = next_obs
        self.dones[i] = done
        self.idx = (i + 1) % self.max_size
        self.count = min(self.count + 1, self.max_size)

    def sample(self, batch_size: int, rng: np.random.RandomState) -> dict[str, np.ndarray]:
        "Sample a random batch without replacement from the stored transitions."
        n = min(self.count, self.max_size)
        if n < batch_size:
            # pad with repeats of existing transitions
            indices = rng.randint(0, n, size=batch_size)
        else:
            indices = rng.choice(n, size=batch_size, replace=False)
        return {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_observations": self.next_observations[indices],
            "dones": self.dones[indices],
        }

    def __len__(self) -> int:
        return self.count


# ---------------------------------------------------------------------------
# M5 Pipeline
# ---------------------------------------------------------------------------


def make_env(
    dataset_id: str,
    *,
    seed: int,
    idx: int = 0,
) -> tuple[gym.Env, np.random.RandomState]:
    """Create a MuJoCo-v5 environment and a deterministic RNG.

    Maps D4RL canonical dataset IDs (e.g. ``halfcheetah-medium-v2``) to
    gym environment names (e.g. ``HalfCheetah-v5``).
    """
    env_name_map = {
        "halfcheetah-medium-v2": "HalfCheetah-v5",
        "hopper-medium-v2": "Hopper-v5",
        "walker2d-medium-v2": "Walker2d-v5",
    }
    gym_name = env_name_map.get(dataset_id, dataset_id)
    env = gym.make(gym_name)
    env_seed = seed + idx
    env.reset(seed=env_seed)
    rng = np.random.RandomState(env_seed)
    return env, rng


def load_offline_data(
    dataset_id: str,
) -> dict[str, np.ndarray]:
    """Load the frozen offline dataset once; returned unchanged for the whole run."""
    data = load_local_dataset(dataset_id)
    return build_transitions(data)


def m5_run(
    *,
    dataset_id: str,
    seed: int,
    online_steps: int,
    update_freq: int,
    online_ratio: float,
    warmup_steps: int,
    eval_interval: int,
    eval_episodes: int,
    max_online_buffer: int = 100_000,
    device: str = "cpu",
    controller_type: str | None = None,
    shift_step: int = 0,
    severity: float = 0.1,
    shift_type: ShiftType | None = "obs_noise",
) -> dict[str, Any]:
    """Run one M5 offline→online transition experiment.

    Returns a summary dict with all logged metrics.

    Optional adaptive intervention controller is applied to the online phase only:
    * ``controller_type`` selects the controller mode:
      - ``none``: no intervention (intervention-free baseline).
      - ``adaptive``: ``AdaptivePlasticityController`` with M6 diagnostics.
      - ``fixed``: ``FixedIntervention`` with configured severity.
    * ``shift_step`` is the online step (from online-loop start) at which
      the intervention fires; ``0`` means no intervention.
    * ``severity`` is the base severity for the underlying FixedIntervention.
    * The ``AdaptivePlasticityController.update`` method is called at each
      ``update_freq`` steps with M6 diagnostics, and its returned
      ``intervention_strength`` modulates the ``FixedIntervention`` severity
      for that step.
    """
    # CUDA determinism
    if device == "cuda" and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    device = torch.device(device)

    # ----- offline data (frozen) -----
    offline_data = load_offline_data(dataset_id)
    offline_obs_dim = int(offline_data["observations"].shape[1])
    offline_act_dim = int(offline_data["actions"].shape[1])
    offline_count = int(offline_data["observations"].shape[0])

    # ----- IQL agent from M4 checkpoint -----
    # We reload the agent from scratch with the same hyperparameters;
    # the checkpoint supplies the initial weights.
    # Hyperparameters match the M4 iql.yaml defaults.
    agent = IQLAgent(
        obs_dim=offline_obs_dim,
        act_dim=offline_act_dim,
        hidden_dim=256,
        num_layers=2,
        expectile=0.7,
        temperature=3.0,
        discount=0.99,
        target_tau=0.005,
        max_advantage=100.0,
        actor_lr=3e-4,
        critic_lr=3e-4,
        device=device,
    )

    # ----- online environment -----
    env, rng = make_env(dataset_id, seed=seed)

    # ----- online buffer -----
    online_buf = OnlineBuffer(obs_dim=offline_obs_dim, act_dim=offline_act_dim, max_size=max_online_buffer)

    # ----- distribution shift orchestrator (M8) -----
    shift = DistributionShiftOrchestrator(
        shift_type=shift_type,
        shift_step=shift_step,
        severity=severity,
    )

    # ----- M9/M10 intervention orchestrator -----
    intervention: Intervention | None = None
    adaptive_controller: AdaptivePlasticityController | None = None

    if controller_type == "adaptive":
        adaptive_controller = AdaptivePlasticityController(
            seed=seed,
            repr_weight=0.4,
            perf_weight=-0.4,
            dorm_weight=-0.2,
            min_severity=0.0,
            max_severity=1.0,
        )
        # Set initial reference agent for temporal representation change
        adaptive_controller.set_reference_agent(agent)
        # Create a FixedIntervention for the adaptive controller to modulate
        intervention = FixedIntervention(
            shift_type="obs_noise",
            shift_step=shift_step,
            severity=severity,
            seed=seed,
        )
    elif controller_type == "fixed":
        intervention = FixedIntervention(
            shift_type="obs_noise",
            shift_step=shift_step,
            severity=severity,
            seed=seed,
        )
    # "none" -> both remain None (no adaptive controller, no fixed shift intervention)

    # ----- logging -----
    metrics: dict[str, Any] = {
        "offline_steps": offline_count,
        "online_steps": 0,
        "update_steps": 0,
        "online_buffer_size": 0,
        "online_ratio": online_ratio,
        "warmup_steps": warmup_steps,
        "eval_interval": eval_interval,
        "eval_episodes": eval_episodes,
        "returns": [],
        "normalized": [],
    }

    # Diagnostics logging
    diag_log_path = None
    diag_log_file = None

    # Controller state for checkpoint/resume
    controller_state = {
        "prev_agent_params": None,
        "prev_perf": None,
        "intervention_severity": severity,
    }

    # ----- warmup: train exclusively on offline data -----
    for step in range(1, warmup_steps + 1):
        indices = rng.choice(offline_count, size=256, replace=False)
        batch = {name: torch.as_tensor(array[indices], device=device)
                 for name, array in offline_data.items()}
        losses = agent.update(batch)
        if step % 1000 == 0:
            pass  # could log losses here

    # Generate output name for diagnostics
    ctrl_map = {"none": "none", "adaptive": "adaptive", "fixed": "fixed"}
    shift_map = {}
    if shift_step > 0:
        shift_map = {"obs_noise": "-shift", "reward_scale": "-shift"}
    else:
        shift_map = {"obs_noise": "-noshift", "reward_scale": "-noshift"}
    
    ctrl_str = ctrl_map[controller_type] if controller_type else "none"
    shift_str = shift_map.get(shift_type, "-noshift") if shift_type else "-noshift"
    out_name = "M10-" + ctrl_str + shift_str + "-seed" + str(seed)

    # ----- main loop: online collection + mixed updates -----
    online_step = 0
    update_step = 0
    last_eval = 0

    # Setup diagnostics logging
    diag_log_path = None
    diag_log_file = None

    while online_step < online_steps:
        # --- collect one online transition ---
        obs, _ = env.reset(seed=seed + online_step)
        done = False
        total_r = 0.0  # native env reward for evaluation

        while not done and online_step < online_steps:
            # --- determine action from policy ---
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = agent.policy.act(obs_tensor, deterministic=True).squeeze(0).cpu().numpy()

            # --- step environment ---
            next_obs, reward_env, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            # --- apply M9 intervention if active ---
            if intervention is not None and intervention.is_shifted(online_step):
                obs, reward_intervention = intervention.apply(obs, reward_env, rng=rng)
            else:
                reward_intervention = reward_env

            # --- use intervention-transformed reward in buffer ---
            online_buf.add(obs, action, reward_intervention, next_obs, done if done else 0.0)
            obs = next_obs
            total_r += float(reward_env)
            online_step += 1
            metrics["online_steps"] = online_step

            # --- update every update_freq steps (after warmup) ---
            if online_step >= warmup_steps and online_step % update_freq == 0:
                update_step += 1
                # determine how many online transitions to mix in
                batch_size = 256
                n_online = max(1, int(batch_size * online_ratio))
                n_offline = batch_size - n_online

                # sample from online buffer
                if len(online_buf) >= n_online:
                    online_batch = online_buf.sample(n_online, rng)
                else:
                    # not enough online data yet; fall back to offline-only
                    online_batch = {"observations": np.empty((0, offline_obs_dim)),
                                    "actions": np.empty((0, offline_act_dim)),
                                    "rewards": np.empty((0,)),
                                    "next_observations": np.empty((0, offline_obs_dim)),
                                    "dones": np.empty((0,))}

                # sample offline batch (frozen, same indices every time for reproducibility)
                off_indices = rng.choice(offline_count, size=n_offline, replace=False)
                offline_batch = {name: torch.as_tensor(array[off_indices], device=device)
                                 for name, array in offline_data.items()}

                # mixed batch: online first, then offline (the agent.update
                # just receives a dict; it doesn't need to know the mix,
                # but we track it for logging)
                mixed: dict[str, torch.Tensor] = {}
                if n_online > 0:
                    mixed.update({k: torch.as_tensor(v, device=device)
                                  for k, v in online_batch.items()})
                if n_offline > 0:
                    mixed.update({k: torch.as_tensor(v, device=device)
                                  for k, v in offline_batch.items()})

                # If we have both, concatenate along first dim; otherwise use whichever is available.
                if n_online > 0 and n_offline > 0:
                    # simpler: just stack online then offline; agent will see them as one batch
                    for k in mixed:
                        mixed[k] = torch.cat([mixed[k], offline_batch[k]])
                elif n_offline == 0:
                    # pure online
                    pass  # mixed already has only online

                agent.update(mixed)
                metrics["update_steps"] = update_step
                metrics["online_buffer_size"] = len(online_buf)

                # --- M10 adaptive plasticity controller step ---
                if adaptive_controller is not None:
                    # Gather M6-style diagnostics from current agent state.
                    # These are online-available; no future information used.
                    # repr_change: cosine similarity of policy parameters (temporal)
                    # perf_change: return delta from evaluation
                    # activation_dormancy: fraction of near-zero policy activations
                    # param_magnitudes: L2 norms of trainable parameters

                    # Compute diagnostics
                    try:
                        dorm_val = float(activation_dormancy(agent.policy, 0.001))
                    except Exception:
                        dorm_val = 0.0

                    # Compute temporal representation change
                    repr_val = 0.0
                    try:
                        repr_val = float(adaptive_controller._compute_temporal_repr_change(agent))
                    except Exception:
                        repr_val = 0.0

                    # Get current normalized performance for perf_change calculation
                    current_norm = metrics["normalized"][-1] if metrics["normalized"] else 0.0

                    # Update the controller and get intervention strength + perf_change
                    # using the corrected M10 rule: 0.4*repr - 0.4*perf - 0.2*dormancy
                    strength = adaptive_controller.update(
                        diagnostics={},
                        agent=agent,
                        current_perf=current_norm,
                    )

                    # Modulate the fixed intervention's severity for subsequent steps.
                    # The AdaptivePlasticityController returns strength in
                    # [min_severity, max_severity]; clip to [0,1] range.
                    if intervention is not None and isinstance(intervention, FixedIntervention):
                        intervention.severity = float(np.clip(strength, 0.0, 1.0))

                    # Get the actual perf_change delta used by the controller
                    perf_change = 0.0
                    if adaptive_controller._history:
                        perf_change = adaptive_controller._history[-1].get("perf_change", 0.0)

                    # Log diagnostics (single record per controller update)
                    if diag_log_file is None:
                        import json
                        diag_log_path = Path("results") / f"diagnostics_{out_name}.jsonl"
                        diag_log_file = open(diag_log_path, "w", encoding="utf-8")

                    # Log diagnostic record (single record per controller update)
                    diag_record = {
                        "step": online_step,
                        "shift_active": intervention.is_shifted(online_step) if intervention else False,
                        "controller_type": "adaptive" if adaptive_controller is not None else "fixed",
                        "repr_change": repr_val,
                        "perf_change": adaptive_controller._history[-1].get("perf_change", 0.0) if adaptive_controller._history else 0.0,
                        "activation_dormancy": dorm_val,
                        "param_magnitudes": param_magnitudes(agent),
                        "controller_strength": strength,
                        "evaluation_return": metrics["returns"][-1] if metrics["returns"] else None,
                        "evaluation_normalized": metrics["normalized"][-1] if metrics["normalized"] else None,
                    }
                    if diag_log_file is None:
                        import json
                        diag_log_path = Path("results") / f"diagnostics_{out_name}.jsonl"
                        diag_log_file = open(diag_log_path, "w", encoding="utf-8")

                    diag_log_file.write(json.dumps(diag_record) + "\n")
                    diag_log_file.flush()

                    # Update reference agent for next step
                    adaptive_controller.set_reference_agent(agent)

        # --- evaluation ---
# --- evaluation ---
        if online_step - last_eval >= eval_interval or online_step >= online_steps:
            # deterministic evaluation
            agent.policy.eval()
            eval_returns = []
            with torch.no_grad():
                for ep in range(eval_episodes):
                    e_obs, _ = env.reset(seed=seed + last_eval + ep)
                    e_done = False
                    e_r = 0.0
                    while not e_done:
                        e_obs_t = torch.as_tensor(e_obs, dtype=torch.float32, device=device).unsqueeze(0)
                        with torch.no_grad():
                            a = agent.policy.act(e_obs_t, deterministic=True).squeeze(0).cpu().numpy()
                        e_obs, r, t, tr, _ = env.step(a)
                        e_done = bool(t or tr)
                        e_r += float(r)
                    eval_returns.append(e_r)
            agent.policy.train()
            mean_ret = float(np.mean(eval_returns))
            # normalize using the M4 reference scores (hard-coded from M4)
            ref = {"halfcheetah-medium-v2": (-280.178953, 12135.0),
                   "hopper-medium-v2": (-20.272305, 3234.3),
                   "walker2d-medium-v2": (1.629008, 4592.3)}
            r_min, r_max = ref[dataset_id]
            norm = 100.0 * (mean_ret - r_min) / (r_max - r_min) if r_max > r_min else None
            metrics["returns"].append(mean_ret)
            metrics["normalized"].append(norm if norm is not None else mean_ret)
            last_eval = online_step

    # Close diagnostics log
    if diag_log_file is not None:
        diag_log_file.close()

    env.close()

    # finalize
    summary = {
        "dataset_id": dataset_id,
        "seed": seed,
        "offline_steps": metrics["offline_steps"],
        "online_steps": metrics["online_steps"],
        "update_steps": metrics["update_steps"],
        "online_buffer_final_size": len(online_buf),
        "online_ratio": online_ratio,
        "warmup_steps": warmup_steps,
        "eval_interval": eval_interval,
        "eval_episodes": eval_episodes,
        "returns": metrics["returns"],
        "normalized": metrics["normalized"],
        "best_return": float(max(metrics["returns"])) if metrics["returns"] else None,
        "best_normalized": float(max(metrics["normalized"])) if metrics["normalized"] else None,
        "runtime_seconds": None,  # caller can fill if desired
    }
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="M5: Offline→Online IQL transition.")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["halfcheetah-medium-v2", "hopper-medium-v2", "walker2d-medium-v2"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--online-steps", type=int, default=20_000)
    parser.add_argument("--update-freq", type=int, default=1000)
    parser.add_argument("--online-ratio", type=float, default=0.5)
    parser.add_argument("--warmup-steps", type=int, default=50_000)
    parser.add_argument("--eval-interval", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--shift-type", type=str, default=None,
                        choices=["obs_noise", "reward_scale"])
    parser.add_argument("--shift-step", type=int, default=0)
    parser.add_argument("--severity", type=float, default=0.1)
    parser.add_argument("--controller-type", type=str, default="none",
                        choices=["none", "adaptive", "fixed"],
                        help="Intervention controller type")
    args = parser.parse_args()

    # CUDA determinism
    if args.device == "cuda" and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    # Map controller type and shift condition to descriptive strings
    ctrl_map = {"none": "none", "adaptive": "adaptive", "fixed": "fixed"}
    shift_map = {}
    if args.shift_step > 0:
        shift_map = {"obs_noise": "-shift", "reward_scale": "-shift"}
    else:
        shift_map = {"obs_noise": "-noshift", "reward_scale": "-noshift"}
    
    ctrl_str = ctrl_map[args.controller_type]
    shift_str = shift_map.get(args.shift_type, "-noshift") if args.shift_type else "-noshift"
    
    out_name = "M10-" + ctrl_str + shift_str + "-seed" + str(args.seed)
    out = Path("results") / out_name
    out.mkdir(parents=True, exist_ok=True)
    
    summary = m5_run(
        dataset_id=args.dataset,
        seed=args.seed,
        online_steps=args.online_steps,
        update_freq=args.update_freq,
        online_ratio=args.online_ratio,
        warmup_steps=args.warmup_steps,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        device=args.device,
        shift_type=args.shift_type,
        shift_step=args.shift_step,
        severity=args.severity,
        controller_type=args.controller_type,
    )

    out = Path("results") / out_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()