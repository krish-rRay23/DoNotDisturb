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
from adaptive_plasticity.m6 import (
    param_magnitudes, representation_change, activation_dormancy,
    activation_dormancy_from_probe, build_probe_batch,
    ACTIVATION_DORMANCY_KIND, DIAGNOSTICS_SCHEMA_VERSION, DORM_TAU,
)


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


def sample_mixed_batch(
    *,
    online_buf: OnlineBuffer,
    offline_data: dict[str, np.ndarray],
    offline_count: int,
    batch_size: int,
    online_ratio: float,
    rng: np.random.RandomState,
    device: torch.device | str,
) -> tuple[dict[str, torch.Tensor], int, int]:
    """Sample one mixed offline/online batch with exact size preservation.

    * ``online_ratio=1.0`` → 100% online (``n_online == batch_size``).
    * ``online_ratio=0.5`` → 50% online + 50% offline.
    * ``online_ratio=0.0`` → 100% offline (``n_online == 0``).
    * ``n_online = int(batch_size * ratio)`` (clamped to [0, 1]),
      ``n_offline = batch_size - n_online``, so ``len == batch_size`` exactly.
    * Deterministic under the caller's ``rng``: online is sampled first via
      ``OnlineBuffer.sample`` (consuming ``rng``), then offline via
      ``rng.choice`` — the same order as the training loop has always used.
    * If the online buffer holds fewer than ``n_online`` transitions, fall
      back to offline-only with the full ``batch_size`` (preserves size and
      matches the historical "not enough online data" behaviour).
    * Returns ``(mixed, n_online, n_offline)`` where ``mixed`` maps each key
      to a ``torch.Tensor`` of leading dim ``batch_size`` (online rows first,
      then offline rows). IQL update semantics are unchanged: the caller
      passes ``mixed`` straight to ``agent.update``.
    """
    ratio = min(1.0, max(0.0, float(online_ratio)))
    n_online = int(batch_size * ratio)
    n_offline = batch_size - n_online

    if n_online > 0 and len(online_buf) < n_online:
        # Not enough online data yet; fall back to offline-only at full size.
        n_online = 0
        n_offline = batch_size

    online_t: dict[str, torch.Tensor] | None = None
    if n_online > 0:
        online_np = online_buf.sample(n_online, rng)
        online_t = {k: torch.as_tensor(v, device=device) for k, v in online_np.items()}

    offline_t: dict[str, torch.Tensor] | None = None
    if n_offline > 0:
        if offline_count >= n_offline:
            off_indices = rng.choice(offline_count, size=n_offline, replace=False)
        else:
            # Tiny synthetic datasets in tests; sample with replacement.
            off_indices = rng.randint(0, offline_count, size=n_offline)
        offline_t = {
            name: torch.as_tensor(array[off_indices], device=device)
            for name, array in offline_data.items()
        }

    if online_t is not None and offline_t is not None:
        assert set(online_t) == set(offline_t)
        mixed: dict[str, torch.Tensor] = {
            k: torch.cat([online_t[k], offline_t[k]], dim=0) for k in online_t
        }
    elif online_t is not None:
        mixed = online_t
    else:
        assert offline_t is not None
        mixed = offline_t

    return mixed, n_online, n_offline


# ---------------------------------------------------------------------------
# M5 Pipeline
# ---------------------------------------------------------------------------


def m10_output_name(
    *,
    dataset_id: str,
    controller_type: str | None,
    shift_type: ShiftType | str | None,
    shift_step: int,
    severity: float,
    seed: int,
) -> str:
    """Phase 4C output identity for every M10/Phase-4 artifact.

    Severity-qualified so different severities can never overwrite/collide:
    ``M10-<dataset>-<ctrl><-shift|-noshift>-sev<severity>-seed<seed>``,
    e.g. ``M10-hopper-medium-v2-adaptive-shift-sev0.5-seed1``.
    ``severity`` is formatted with ``%g`` (``0.1`` -> ``0.1``, ``0.0`` ->
    ``0``) for a stable, filesystem-safe identity. The run directory and
    the ``diagnostics_<name>.jsonl`` file both derive from this single
    helper, so they always agree.
    """
    ctrl_map = {"none": "none", "adaptive": "adaptive", "fixed": "fixed"}
    shift_map: dict[str, str]
    if shift_step > 0:
        shift_map = {"obs_noise": "-shift", "reward_scale": "-shift"}
    else:
        shift_map = {"obs_noise": "-noshift", "reward_scale": "-noshift"}

    ctrl_str = ctrl_map[controller_type] if controller_type else "none"
    shift_str = shift_map.get(shift_type, "-noshift") if shift_type else "-noshift"
    sev_str = "-sev%g" % float(severity)
    return "M10-" + dataset_id + "-" + ctrl_str + shift_str + sev_str + "-seed" + str(seed)


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

    # ----- Phase 3 fixed dormancy probe (cached once per run) -----
    # 256 frozen offline observations, seeded by run seed; reused verbatim for
    # every neuronal-dormancy evaluation. No learning, no online probe.
    dormancy_probe = build_probe_batch(
        offline_data["observations"], n=256, seed=seed)
    dormancy_reference: float | None = None  # anchor D_0, set post-warmup

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
        # Capture the FIXED anchor for representation change (Phase 1: set
        # once here; never refreshed per update so repr measures drift vs
        # the run-start anchor, not a rolling per-step delta).
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

    # Anchor dormancy reference on the fixed probe post-warmup (log-only delta).
    try:
        _d0, _ = activation_dormancy_from_probe(agent.policy, dormancy_probe, tau=DORM_TAU)
        import math as _math_anchor
        dormancy_reference = float(_d0) if _math_anchor.isfinite(_d0) else None
    except Exception:
        dormancy_reference = None

    # Generate output name for diagnostics (Phase 4C severity-qualified identity).
    out_name = m10_output_name(
        dataset_id=dataset_id,
        controller_type=controller_type,
        shift_type=shift_type,
        shift_step=shift_step,
        severity=severity,
        seed=seed,
    )

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

            # --- evaluation during training at eval_interval ---
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

            # --- update every update_freq steps (after warmup) ---
            if online_step >= warmup_steps and online_step % update_freq == 0:
                update_step += 1
                batch_size = 256
                mixed, n_online, n_offline = sample_mixed_batch(
                    online_buf=online_buf,
                    offline_data=offline_data,
                    offline_count=offline_count,
                    batch_size=batch_size,
                    online_ratio=online_ratio,
                    rng=rng,
                    device=device,
                )

                agent.update(mixed)
                metrics["update_steps"] = update_step
                metrics["online_buffer_size"] = len(online_buf)

                # --- M10 adaptive plasticity controller step (Phase 4E live path) ---
                # Live-path signal contract (M10 archived results untouched):
                # * perf: a FRESH mean state-value on the fixed 256-obs probe,
                #   recomputed at EVERY controller update (never a stale
                #   carried-forward evaluation). Its EMA delta feeds the
                #   controller; raw eval returns stay in the log for audit.
                # * repr/dormancy: equation inputs are CENTERED deltas vs
                #   stable pre-shift anchors captured once at the first
                #   controller update (warmup < shift_step keeps this
                #   pre-shift); raw values stay in the log for audit.
                # * scales: NORMALIZED z-scores vs pre-shift scales estimated
                #   from finite pre-shift observations only (Phase 4G; gated
                #   by shift_active so post-shift samples never leak in).
                #   z = delta/scale per channel; raw+centered stay in the log.
                # * coefficients 0.4/-0.4/-0.2 and clip [0,1] unchanged.
                if adaptive_controller is not None:
                    # Gather M6-style diagnostics from current agent state.
                    # These are online-available; no future information used.
                    # repr_change: weight-based 1-cos distance vs FIXED anchor
                    #   (anchor set once at run start; not refreshed per update)
                    # perf_value_proxy: fresh mean V(s) on the FIXED probe,
                    #   recomputed every controller update (Phase 4E; the eval
                    #   return below is log-only and may be stale between evals)
                    # activation_dormancy: Phase 3 neuronal dormancy D_t on the
                    #   FIXED 256-obs probe (policy backbone ReLU layers only;
                    #   absolute D_t feeds the log; the CENTERED delta vs the
                    #   pre-shift anchor feeds the controller) -- see m6.py
                    # param_magnitudes: L2 norms of trainable parameters

                    # Compute diagnostics (absolute D_t on cached fixed probe)
                    import math as _math_dorm
                    try:
                        _dorm_raw, dorm_per_layer = activation_dormancy_from_probe(
                            agent.policy, dormancy_probe, tau=DORM_TAU)
                        dorm_val = float(_dorm_raw)
                        if dormancy_reference is None and _math_dorm.isfinite(dorm_val):
                            dormancy_reference = dorm_val
                        if _math_dorm.isfinite(dorm_val) and dormancy_reference is not None and _math_dorm.isfinite(dormancy_reference):
                            dorm_delta = float(dorm_val - dormancy_reference)
                        else:
                            dorm_delta = float("nan")
                    except Exception:
                        dorm_val = float("nan")
                        dorm_per_layer = {}
                        dorm_delta = float("nan")

                    # Representation change vs the FIXED anchor (NaN if missing)
                    import math as _math
                    repr_val = float("nan")
                    try:
                        repr_val = float(adaptive_controller._compute_temporal_repr_change(agent))
                    except Exception:
                        repr_val = float("nan")
                    repr_for_log = repr_val  # NaN preserved to mark missing anchor

                    # Phase 4E: capture the stable pre-shift centering anchors
                    # once (first controller update wins; requires
                    # warmup_steps < shift_step so this is pre-shift).
                    # Non-finite candidates are ignored inside the setter,
                    # deferring anchoring to the next update.
                    if not adaptive_controller.centering_anchors_set():
                        adaptive_controller.set_centering_anchors(
                            repr_anchor=repr_val,
                            dormancy_anchor=dorm_val,
                        )

                    # Phase 4E: FRESH performance signal at every controller
                    # update -- mean state-value on the fixed probe under
                    # no_grad (deterministic; no env interaction). Replaces
                    # the stale carried-forward evaluation return, whose raw
                    # values remain logged below for auditability.
                    perf_value_proxy = float("nan")
                    try:
                        with torch.no_grad():
                            _probe_t = torch.as_tensor(
                                dormancy_probe, dtype=torch.float32, device=device)
                            perf_value_proxy = float(
                                agent.value(_probe_t).detach().float().mean().item())
                    except Exception:
                        perf_value_proxy = float("nan")

                    # Update the controller and get intervention strength.
                    # Equation inputs (Phase 4G): normalized z-scores of
                    # delta_repr, EMA-smoothed perf_value_proxy delta, and
                    # delta_dormancy vs PRE-SHIFT scales with the preserved
                    # M10 rule: 0.4*z_repr - 0.4*z_perf - 0.2*z_dorm.
                    # Raw/centered diagnostics stay in history/JSONL for audit.
                    # Phase 4G: pre_shift gates scale estimation, so post-shift
                    # samples never leak into the scales (same flag drives the
                    # shift_active field below to keep gating and log in sync).
                    shift_active_now = intervention.is_shifted(online_step) if intervention else False
                    strength = adaptive_controller.update(
                        diagnostics={"activation_dormancy": dorm_val},
                        agent=agent,
                        current_perf=perf_value_proxy,
                        pre_shift=not shift_active_now,
                    )

                    # Modulate the fixed intervention's severity for subsequent steps.
                    # The AdaptivePlasticityController returns strength in
                    # [min_severity, max_severity]; clip to [0,1] range.
                    if intervention is not None and isinstance(intervention, FixedIntervention):
                        intervention.severity = float(np.clip(strength, 0.0, 1.0))

                    # Read back the centered equation inputs used by the
                    # controller (single source of truth for the log below).
                    perf_change = 0.0
                    if adaptive_controller._history:
                        perf_change = adaptive_controller._history[-1].get("perf_change", 0.0)

                    # Log diagnostics (single record per controller update)
                    if diag_log_file is None:
                        import json
                        diag_log_path = Path("results") / f"diagnostics_{out_name}.jsonl"
                        diag_log_file = open(diag_log_path, "w", encoding="utf-8")

                    # Log diagnostic record (single record per controller update)
                    # Phase 1: anchor is FIXED (set once at run start), so it is
                    # intentionally NOT refreshed here. repr NaN = missing anchor.
                    # Phase 4C: run metadata persisted in every record so the
                    # severity/shift condition is recoverable without the filename.
                    # Phase 4E: raw diagnostics preserved verbatim; centered
                    # equation inputs explicit (delta_repr/delta_dormancy) with
                    # their pre-shift anchors; perf_value_proxy is the fresh
                    # per-update signal (perf_change* derive from it), while
                    # evaluation_* remain the sparse eval returns for audit.
                    # Phase 4G: normalized z-scores and the pre-shift scales
                    # used explicit (z_repr/z_perf/z_dorm + *_scale).
                    _hist_last = adaptive_controller._history[-1] if adaptive_controller._history else {}
                    diag_record = {
                        "step": online_step,
                        "shift_active": shift_active_now,
                        "controller_type": "adaptive" if adaptive_controller is not None else "fixed",
                        "severity": float(severity),
                        "shift_type": shift_type,
                        "shift_step": int(shift_step),
                        "repr_change": repr_for_log,
                        "repr_anchor": _hist_last.get("repr_anchor"),
                        "delta_repr": _hist_last.get("delta_repr"),
                        "repr_scale": _hist_last.get("repr_scale", 1.0),
                        "z_repr": _hist_last.get("z_repr"),
                        "perf_value_proxy": perf_value_proxy,
                        "perf_change": _hist_last.get("perf_change", 0.0),
                        "perf_change_raw": _hist_last.get("perf_change_raw", 0.0),
                        "perf_scale": _hist_last.get("perf_scale", 1.0),
                        "z_perf": _hist_last.get("z_perf"),
                        "activation_dormancy": dorm_val,
                        "dormancy_anchor": _hist_last.get("dormancy_anchor"),
                        "delta_dormancy": _hist_last.get("delta_dormancy"),
                        "dormancy_scale": _hist_last.get("dormancy_scale", 1.0),
                        "z_dorm": _hist_last.get("z_dorm"),
                        "activation_dormancy_per_layer": dorm_per_layer,
                        "activation_dormancy_kind": ACTIVATION_DORMANCY_KIND,
                        "dormancy_delta_vs_anchor": dorm_delta,
                        "diagnostics_schema_version": DIAGNOSTICS_SCHEMA_VERSION,
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

    # Close diagnostics log
    if diag_log_file is not None:
        diag_log_file.close()

    env.close()

    # finalize (Phase 4C: full experiment metadata persisted for auditability)
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
        "severity": float(severity),
        "shift_type": shift_type,
        "shift_step": int(shift_step),
        "controller_type": controller_type,
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

    # Map controller type and shift condition to the Phase 4C
    # severity-qualified identity (single helper shared with m5_run).
    out_name = m10_output_name(
        dataset_id=args.dataset,
        controller_type=args.controller_type,
        shift_type=args.shift_type,
        shift_step=args.shift_step,
        severity=args.severity,
        seed=args.seed,
    )
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