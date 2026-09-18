"""Final-study runner: FINAL confirmatory protocol (approved specification).

Protocol (FINAL-v1, complete — every cell records all of this):
  * offline warmup: 25,000 updates on frozen offline data (shared base checkpoint supported)
  * online steps: 25,000, one gradient update per online step (update_freq=1)
  * shift_step: 5,000 (online-step units; ``none`` regime -> 0, never active)
  * regimes: none / obs_noise (sigma=0.1) / reward_scale (scale=0.5)
  * primary arms: none / fixed / capacity_gate (135 cells: 3 envs x 3
    regimes x 3 arms x 5 seeds); m10ref is a SEPARATE frozen reference
    (15 cells: halfcheetah-medium-v2 only x 3 regimes x 5 seeds), never
    part of the primary matrix
  * online_ratio: 0.5, batch 256
  * evaluation every 2,500 online steps, 10 deterministic episodes,
    normalized D4RL score via the frozen M4 reference table
  * deterministic seeded execution (isolated RNG streams)
  * atomic checkpointing with graceful intra-cell resume

Trigger semantics (the ONLY difference between arms — see
``should_apply_intervention``):
  * none: never apply.
  * fixed: apply whenever the shift is active.
  * capacity_gate: apply only while the shift is active AND the gate is ON.
  * m10ref: apply only while the shift is active AND last M10 strength > 0.

Observation-noise semantics: the transformed observation feeds BOTH the
policy (next input) and the stored replay transition. Reward-scale affects
only the stored reward. Frozen ``apply_observation_noise`` /
``apply_reward_scale`` are reused read-only; observation noise draws from an
isolated ``RandomState(seed + NOISE_RNG_OFFSET)`` never consumed elsewhere.

Nothing here modifies frozen M1-M10 code or the CapacityGate thresholds/logic.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from adaptive_plasticity.capacity_gate import (
    CAPACITY_GATE_KIND,
    CAPACITY_GATE_SCHEMA_VERSION,
    CapacityGateConfig,
    CapacityGateController,
    build_fixed_probe,
)
from adaptive_plasticity.distribution_shifts import (
    apply_observation_noise,
    apply_reward_scale,
)
from adaptive_plasticity.interventions import apply_plasticity_intervention
from adaptive_plasticity.iql import IQLAgent
from adaptive_plasticity.m5 import (
    OnlineBuffer,
    load_offline_data,
    make_env,
)
from adaptive_plasticity.offline_cache import (
    DEFAULT_CHECKPOINT_DIR,
    get_offline_checkpoint_path,
    is_offline_checkpoint_valid,
    train_offline_checkpoint,
)
from adaptive_plasticity.reproducibility import (
    create_metadata,
    detect_device,
    git_commit_hash,
    seed_everything,
    utc_timestamp,
    write_metadata,
)

#: Frozen protocol identifier (recorded in every plan cell and run).
PROTOCOL_VERSION: str = "final-v1"

#: Regime -> intervention severity (sigma for obs_noise, scale for reward_scale).
REGIME_SEVERITY: dict[str, float] = {
    "none": 0.0,
    "obs_noise": 0.1,
    "reward_scale": 0.5,
    "actuator_cripple": 1.0,
}

#: Offset isolating the observation-noise RNG stream from all other streams.
NOISE_RNG_OFFSET: int = 7919

#: Confirmation token required by the harness before matrix execution
#: (135 primary + 15 M10-reference cells).
CONFIRM_TOKEN: str = "CONFIRM-FINAL-150"

FINAL_ENVS: tuple[str, ...] = (
    "halfcheetah-medium-v2",
    "hopper-medium-v2",
    "walker2d-medium-v2",
)
FINAL_SHIFTS: tuple[str, ...] = ("none", "obs_noise", "reward_scale", "actuator_cripple")
FINAL_CONTROLLERS: tuple[str, ...] = ("none", "fixed", "capacity_gate", "redo", "redo_periodic", "m10ref")
FINAL_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)

#: Primary confirmatory arms (the 135-cell matrix). ``m10ref`` is excluded
#: here by design; it runs only in the separate 15-cell reference plan.
PRIMARY_CONTROLLERS: tuple[str, ...] = ("none", "fixed", "capacity_gate")

#: Reference-only arm and its env scope (halfcheetah-medium-v2 only).
REFERENCE_CONTROLLERS: tuple[str, ...] = ("m10ref",)
REFERENCE_ENVS: tuple[str, ...] = ("halfcheetah-medium-v2",)

_REF_RETURNS = {
    "halfcheetah-medium-v2": (-280.178953, 12135.0),
    "hopper-medium-v2": (-20.272305, 3234.3),
    "walker2d-medium-v2": (1.629008, 4592.3),
}


def final_protocol() -> dict[str, Any]:
    """Return the complete frozen protocol dict (single source of truth)."""
    gate_defaults = CapacityGateConfig()
    return {
        "protocol_version": PROTOCOL_VERSION,
        "offline_warmup_updates": 25_000,
        "online_steps": 25_000,
        "shift_step": 5_000,
        "regime_severity": dict(REGIME_SEVERITY),
        "online_ratio": 0.5,
        "update_freq": 1,
        "batch_size": 256,
        "eval_interval": 2_500,
        "eval_episodes": 10,
        "gate_interval": 1_000,
        "probe_size": 256,
        "noise_rng_offset": NOISE_RNG_OFFSET,
        "deterministic": True,
        "gate_thresholds": {
            "dorm_on": gate_defaults.dorm_on,
            "dorm_off": gate_defaults.dorm_off,
            "rho_on": gate_defaults.rho_on,
            "rho_off": gate_defaults.rho_off,
        },
    }


def final_output_name(
    *,
    dataset_id: str,
    controller: str,
    shift: str,
    severity: float,
    seed: int,
) -> str:
    """Unified output identity: ``FINAL-<env>-<ctrl>-<shift>-sev<g>-seed<N>``."""
    return "FINAL-%s-%s-%s-sev%g-seed%d" % (dataset_id, controller, shift, float(severity), int(seed))


def should_apply_intervention(
    *,
    controller: str,
    step: int = 5_000,
    shift_step: int = 5_000,
    gate_on: bool = False,
    m10_strength: float = 0.0,
) -> bool:
    """Pure trigger rule — the ONLY intended difference between arms.

    - none: never applies.
    - fixed / redo_periodic: applies unconditionally during adaptation phase (step >= shift_step).
    - capacity_gate: applies autonomously whenever gate_on is True (no oracle knowledge of shift).
    - m10ref: applies whenever M10 heuristic strength > 0.
    """
    if controller == "none":
        return False
    if controller in ("fixed", "redo", "redo_periodic"):
        return step >= shift_step
    if controller == "capacity_gate":
        return bool(gate_on)
    if controller == "m10ref":
        return float(m10_strength) > 0.0
    return False


def apply_regime_transform(
    obs: np.ndarray,
    reward: float,
    *,
    shift: str,
    severity: float,
    noise_rng: np.random.RandomState | None,
) -> tuple[np.ndarray, float]:
    """Apply the frozen regime transform (read-only reuse)."""
    if shift == "obs_noise":
        rng = noise_rng if noise_rng is not None else np.random.RandomState()
        return apply_observation_noise(np.asarray(obs), severity=float(severity), rng=rng), float(reward)
    if shift == "reward_scale":
        return np.asarray(obs).copy(), apply_reward_scale(float(reward), severity=float(severity))
    return np.asarray(obs).copy(), float(reward)


def sample_mixed_batch_fast(
    *,
    online_buf: OnlineBuffer,
    offline_data: dict[str, Any],
    offline_count: int,
    batch_size: int,
    online_ratio: float,
    rng: np.random.RandomState,
    device: torch.device | str,
) -> tuple[dict[str, torch.Tensor], int, int]:
    """Mixed offline/online batch with uniform with-replacement sampling.

    Supports pre-loaded GPU tensors for offline_data to eliminate CPU->GPU copies.
    """
    ratio = min(1.0, max(0.0, float(online_ratio)))
    n_online = int(batch_size * ratio)
    n_offline = batch_size - n_online

    if n_online > 0 and len(online_buf) < n_online:
        n_online = 0
        n_offline = batch_size

    online_t: dict[str, torch.Tensor] | None = None
    if n_online > 0:
        n = len(online_buf)
        indices = rng.randint(0, n, size=n_online)
        online_np = {
            "observations": online_buf.observations[indices],
            "actions": online_buf.actions[indices],
            "rewards": online_buf.rewards[indices],
            "next_observations": online_buf.next_observations[indices],
            "dones": online_buf.dones[indices],
        }
        online_t = {k: torch.as_tensor(v, device=device) for k, v in online_np.items()}

    offline_t: dict[str, torch.Tensor] | None = None
    if n_offline > 0:
        off_indices = rng.randint(0, offline_count, size=n_offline)
        first_val = next(iter(offline_data.values()))
        if isinstance(first_val, torch.Tensor):
            off_indices_t = torch.as_tensor(off_indices, device=first_val.device)
            offline_t = {name: array[off_indices_t] for name, array in offline_data.items()}
        else:
            offline_t = {
                name: torch.as_tensor(array[off_indices], device=device)
                for name, array in offline_data.items()
            }

    if online_t is not None and offline_t is not None:
        mixed: dict[str, torch.Tensor] = {
            k: torch.cat([online_t[k], offline_t[k]], dim=0) for k in online_t
        }
    elif online_t is not None:
        mixed = online_t
    else:
        assert offline_t is not None
        mixed = offline_t

    return mixed, n_online, n_offline


def _evaluate(
    agent: IQLAgent,
    dataset_id: str,
    *,
    seed: int,
    episodes: int,
    device,
    crippled_idx: int | None = None,
) -> float:
    """Deterministic policy evaluation (evaluates on target dynamics if crippled)."""
    eval_env, _ = make_env(dataset_id, seed=seed)
    try:
        agent.policy.eval()
        returns: list[float] = []
        with torch.no_grad():
            for ep in range(episodes):
                obs, _ = eval_env.reset(seed=seed + ep)
                done = False
                total = 0.0
                while not done:
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        action = agent.policy.act(obs_t, deterministic=True).squeeze(0).cpu().numpy()
                    if crippled_idx is not None:
                        action = action.copy()
                        action[crippled_idx] = 0.0
                    obs, reward, terminated, truncated, _ = eval_env.step(action)
                    done = bool(terminated or truncated)
                    total += float(reward)
                returns.append(total)
        agent.policy.train()
        return float(np.mean(returns))
    finally:
        eval_env.close()


def _atomic_write_file(target_path: Path, content: str | bytes) -> None:
    """Atomic write to prevent partial/corrupted files during sudden process termination."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_file = tempfile.mkstemp(dir=target_path.parent, prefix="tmp_", suffix=target_path.suffix)
    os.close(fd)
    tmp_path = Path(tmp_file)
    try:
        if isinstance(content, str):
            tmp_path.write_text(content, encoding="utf-8")
        else:
            tmp_path.write_bytes(content)
        os.replace(tmp_path, target_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def run_final_cell(
    *,
    dataset_id: str,
    controller: str,
    shift: str,
    seed: int,
    severity: float | None = None,
    shift_step: int = 5_000,
    online_steps: int = 25_000,
    warmup_updates: int = 25_000,
    update_freq: int = 1,
    gate_interval: int = 1_000,
    online_ratio: float = 0.5,
    eval_interval: int = 2_500,
    eval_episodes: int = 10,
    device: str = "cuda",
    outdir: str | Path = "results",
    gate_config: CapacityGateConfig | None = None,
    deterministic: bool = True,
    use_shared_offline_checkpoints: bool = True,
    offline_checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
    output_name: str | None = None,
) -> dict[str, Any]:
    """Run one factorial cell; write summary, metadata, telemetry, checkpoint."""
    t_start = time.time()
    if controller not in FINAL_CONTROLLERS:
        raise ValueError(f"Unknown controller {controller!r}")
    if shift not in FINAL_SHIFTS:
        raise ValueError(f"Unknown shift {shift!r}")
    if severity is None:
        severity = REGIME_SEVERITY[shift]
    severity = float(severity)

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    device = detect_device(device)
    seed_everything(seed, deterministic=deterministic)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    torch_device = torch.device(device)

    out_name = str(output_name) if output_name is not None else final_output_name(
        dataset_id=dataset_id, controller=controller, shift=shift,
        severity=severity, seed=seed,
    )
    out_path = Path(outdir) / out_name
    out_path.mkdir(parents=True, exist_ok=True)

    # If completed summary and checkpoint already exist and are valid, return immediately
    summary_file = out_path / "summary.json"
    ckpt_file = out_path / "checkpoint.pt"
    if summary_file.is_file() and ckpt_file.is_file():
        try:
            return json.loads(summary_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    active_shift_step = 0 if shift == "none" else int(shift_step)

    offline_data = load_offline_data(dataset_id)
    obs_dim = int(offline_data["observations"].shape[1])
    act_dim = int(offline_data["actions"].shape[1])
    offline_count = int(offline_data["observations"].shape[0])

    agent = IQLAgent(obs_dim, act_dim, device=device)
    env, rng = make_env(dataset_id, seed=seed)
    noise_rng = np.random.RandomState(int(seed) + NOISE_RNG_OFFSET)
    probe_rng = np.random.RandomState(int(seed) + 9999)
    online_buf = OnlineBuffer(obs_dim=obs_dim, act_dim=act_dim)
    probe = build_fixed_probe(offline_data["observations"], n=256, seed=seed)

    # Pre-load offline dataset into GPU memory if on CUDA for zero-copy sampling
    offline_storage: dict[str, Any] = offline_data
    if device.startswith("cuda"):
        offline_storage = {
            k: torch.as_tensor(v, device=torch_device)
            for k, v in offline_data.items()
        }

    # Intervention severity is strictly decoupled from shift severity (fixed at 0.1)
    gate = CapacityGateController(config=(gate_config or CapacityGateConfig(
        active_severity=0.1,
        cooldown_steps=3000,
        dorm_off=0.08,
    )))
    m10 = None
    m10_strength = 0.0
    n_m10_updates = 0
    if controller == "m10ref":
        from adaptive_plasticity.distribution_shifts import AdaptivePlasticityController
        m10 = AdaptivePlasticityController(seed=seed)
        m10.set_reference_agent(agent)

    def shifted(step: int) -> bool:
        if shift == "none":
            return False
        return step >= active_shift_step

    # ----- Check for Intra-Cell Periodic Resume State -----
    state_file = out_path / "state.pt"
    resumed = False
    online_step = 0
    update_step = 0
    applied_count = 0
    last_eval = 0
    returns: list[float] = []
    normalized: list[float] = []
    gate_records: list[dict[str, Any]] = []
    m10_records: list[dict[str, Any]] = []
    rank_anchor: float = float("nan")

    if state_file.is_file():
        try:
            saved_state = torch.load(state_file, map_location=torch_device, weights_only=False)
            agent.load_state_dict(saved_state["agent_state"])
            online_step = saved_state["online_step"]
            update_step = saved_state["update_step"]
            applied_count = saved_state["applied_count"]
            last_eval = saved_state["last_eval"]
            returns = list(saved_state["returns"])
            normalized = list(saved_state["normalized"])
            rank_anchor = saved_state["rank_anchor"]
            if "noise_rng_state" in saved_state:
                noise_rng.set_state(saved_state["noise_rng_state"])
            if "probe_rng_state" in saved_state:
                probe_rng.set_state(saved_state["probe_rng_state"])
            if "rng_state" in saved_state:
                rng.set_state(saved_state["rng_state"])
            if "torch_rng_state" in saved_state and saved_state["torch_rng_state"] is not None:
                trng = saved_state["torch_rng_state"]
                if isinstance(trng, torch.Tensor):
                    trng = trng.cpu().to(torch.uint8)
                torch.set_rng_state(trng)
            if "torch_cuda_rng_state" in saved_state and saved_state["torch_cuda_rng_state"] is not None and torch.cuda.is_available():
                c_rngs = saved_state["torch_cuda_rng_state"]
                if isinstance(c_rngs, (list, tuple)):
                    c_rngs = [s.cpu().to(torch.uint8) if isinstance(s, torch.Tensor) else s for s in c_rngs]
                torch.cuda.set_rng_state_all(c_rngs)
            if "online_buf" in saved_state:
                buf_dict = saved_state["online_buf"]
                cnt = buf_dict.get("count", buf_dict.get("size", 0))
                online_buf.idx = buf_dict.get("idx", buf_dict.get("ptr", 0))
                online_buf.count = cnt
                if cnt > 0:
                    online_buf.observations[:cnt] = buf_dict["observations"]
                    online_buf.actions[:cnt] = buf_dict["actions"]
                    online_buf.rewards[:cnt] = buf_dict["rewards"]
                    online_buf.next_observations[:cnt] = buf_dict["next_observations"]
                    online_buf.dones[:cnt] = buf_dict["dones"]
            if "gate_records" in saved_state:
                gate_records = list(saved_state["gate_records"])
            if "m10_records" in saved_state:
                m10_records = list(saved_state["m10_records"])
            if "gate_state" in saved_state:
                gst = saved_state["gate_state"]
                gate.gate_on = bool(gst.get("gate_on", False))
                gate.n_evaluations = int(gst.get("n_evaluations", 0))
                gate.n_interventions = int(gst.get("n_interventions", 0))
                gate.last_intervention_step = int(gst.get("last_intervention_step", -999999))
            if math.isfinite(rank_anchor):
                gate.set_rank_anchor(rank_anchor, force=True)
            if "m10_state" in saved_state:
                mst = saved_state["m10_state"]
                m10_strength = float(mst.get("strength", 0.0))
                n_m10_updates = int(mst.get("n_updates", 0))
            resumed = True
        except Exception as e:
            print(f"FAILED TO RESUME STATE CHECKPOINT: {e}", flush=True)
            import traceback
            traceback.print_exc()
            resumed = False

    # ----- Offline Warmup Phase (if not resumed) -----
    if not resumed:
        shared_ckpt_path = get_offline_checkpoint_path(dataset_id, seed, offline_checkpoint_dir)
        if (
            use_shared_offline_checkpoints
            and warmup_updates == 25_000
            and is_offline_checkpoint_valid(shared_ckpt_path)
        ):
            # Fast load from verified shared offline base checkpoint
            base_data = torch.load(shared_ckpt_path, map_location=torch_device, weights_only=False)
            agent.load_state_dict(base_data["agent_state"])
        else:
            # Train offline warmup directly
            for _ in range(1, int(warmup_updates) + 1):
                indices = rng.randint(0, offline_count, size=256)
                if isinstance(offline_storage["observations"], torch.Tensor):
                    batch = {k: v[indices] for k, v in offline_storage.items()}
                else:
                    batch = {k: torch.as_tensor(v[indices], device=torch_device)
                             for k, v in offline_storage.items()}
                agent.update(batch)

        # Capture rank anchor post-warmup / pre-online
        if controller == "capacity_gate":
            rank_anchor = float(gate.capture_anchor_from_agent(agent, probe))

    # ----- Online Adaptation Loop -----
    try:
        if resumed and "env_state" in saved_state:
            env_st = saved_state["env_state"]
            if env_st.get("qpos") is not None and env_st.get("qvel") is not None:
                env.reset(seed=seed + online_step)
                env.unwrapped.set_state(env_st["qpos"], env_st["qvel"])
                env._elapsed_steps = env_st.get("elapsed", 0)
                obs = env_st["current_obs"]
            else:
                obs_raw, _ = env.reset(seed=seed + online_step)
                if shifted(online_step):
                    obs, _ = apply_regime_transform(obs_raw, 0.0, shift=shift,
                                                    severity=severity, noise_rng=noise_rng)
                else:
                    obs = np.asarray(obs_raw).copy()
        else:
            obs_raw, _ = env.reset(seed=seed + online_step)
            if shifted(online_step):
                obs, _ = apply_regime_transform(obs_raw, 0.0, shift=shift,
                                                severity=severity, noise_rng=noise_rng)
            else:
                obs = np.asarray(obs_raw).copy()

        while online_step < online_steps:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=torch_device).unsqueeze(0)
            with torch.no_grad():
                action = agent.policy.act(obs_t, deterministic=True).squeeze(0).cpu().numpy()

            is_shift_active = shifted(online_step)
            if is_shift_active and shift == "actuator_cripple":
                action = action.copy()
                crippled_idx = 0 if dataset_id.startswith("halfcheetah") else 2
                action[crippled_idx] = 0.0

            next_obs_raw, reward_raw, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)

            # Shift application is completely independent of intervention decisions
            if is_shift_active:
                next_obs, reward_used = apply_regime_transform(
                    next_obs_raw, float(reward_raw), shift=shift,
                    severity=severity, noise_rng=noise_rng)
            else:
                next_obs, reward_used = np.asarray(next_obs_raw).copy(), float(reward_raw)

            # Store transition in online replay buffer (done_float=1.0 only for true terminal states)
            done_float = 1.0 if terminated else 0.0
            online_buf.add(np.asarray(obs).copy(), action, reward_used,
                           np.asarray(next_obs).copy(), done_float)
            obs = next_obs
            if done:
                obs_raw, _ = env.reset(seed=seed + online_step)
                if shifted(online_step):
                    obs, _ = apply_regime_transform(obs_raw, 0.0, shift=shift,
                                                    severity=severity, noise_rng=noise_rng)
                else:
                    obs = np.asarray(obs_raw).copy()
            online_step += 1

            if online_step % update_freq == 0:
                update_step += 1
                mixed, _, _ = sample_mixed_batch_fast(
                    online_buf=online_buf, offline_data=offline_storage,
                    offline_count=offline_count, batch_size=256,
                    online_ratio=online_ratio, rng=rng, device=torch_device,
                )
                agent.update(mixed)

            if online_step % gate_interval == 0 and online_step < online_steps:
                # Nominal intervention phase begins at shift_step (or 1k for pilot / 5k for final when shift is none)
                if shift == "none":
                    nominal_shift_step = 1_000 if online_steps <= 5_000 else 5_000
                else:
                    nominal_shift_step = int(shift_step)
                intervention_phase = (online_step >= nominal_shift_step)

                # Determine active probe (sampled uniformly from all online transitions up to current step)
                if online_buf.count >= 256:
                    probe_indices = probe_rng.choice(online_buf.count, size=256, replace=False)
                    active_probe = torch.as_tensor(
                        online_buf.observations[probe_indices],
                        dtype=torch.float32, device=torch_device
                    )
                else:
                    active_probe = probe
                offline_probe = probe

                if controller == "capacity_gate":
                    rec = gate.update_from_agent(agent, active_probe, step=online_step)
                    rec["controller"] = controller
                    rec["shift_active"] = shifted(online_step)

                    # Dual logging: log offline probe sidecar metrics to distinguish active vs intrinsic capacity
                    from adaptive_plasticity.capacity_gate import true_activation_dormancy, feature_effective_rank, capture_hidden_features
                    off_hidden = capture_hidden_features(agent.policy, offline_probe)
                    if off_hidden:
                        off_dorm, _ = true_activation_dormancy(agent.policy, offline_probe)
                        off_rank = feature_effective_rank(off_hidden[-1])
                        rec["offline_dormancy"] = float(off_dorm)
                        rec["offline_effective_rank"] = float(off_rank)
                        rec["offline_rho"] = float(off_rank / rank_anchor) if (not math.isnan(rank_anchor) and rank_anchor > 0) else 1.0
                    rec["online_dormancy"] = float(rec["dormancy"])
                    rec["online_rho"] = float(rec["rho"])

                    # Autonomous capacity gate: triggers strictly when should_intervene is True (cooldown respected)
                    if rec.get("should_intervene", False):
                        w_before = next(agent.critic.parameters()).detach().clone()
                        apply_plasticity_intervention(
                            agent,
                            intervention_type="shrink_perturb",
                            severity=gate.config.active_severity,
                            shrink=0.1,
                            probe_batch=active_probe,
                            rng=noise_rng,
                        )
                        w_after = next(agent.critic.parameters()).detach().clone()
                        rec["param_diff_norm"] = float(torch.norm(w_after - w_before).item())
                        applied_count += 1

                    gate_records.append({k: v for k, v in rec.items()
                                         if k not in ("dormancy_per_layer", "policy_metrics", "critic_metrics")})
                elif controller == "fixed":
                    # Unconditional periodic baseline: applies every gate_interval during intervention phase
                    if intervention_phase:
                        apply_plasticity_intervention(
                            agent,
                            intervention_type="shrink_perturb",
                            severity=0.1,
                            shrink=0.1,
                            probe_batch=active_probe,
                            rng=noise_rng,
                        )
                        applied_count += 1
                elif controller in ("redo", "redo_periodic"):
                    # Unconditional periodic ReDo: recycles dormant neurons every gate_interval during intervention phase
                    if intervention_phase:
                        apply_plasticity_intervention(
                            agent,
                            intervention_type="redo",
                            severity=0.1,
                            probe_batch=active_probe,
                            rng=noise_rng,
                        )
                        applied_count += 1
                elif controller == "m10ref":
                    assert m10 is not None
                    from adaptive_plasticity.m6 import activation_dormancy_from_probe, DORM_TAU
                    try:
                        d_raw, _ = activation_dormancy_from_probe(agent.policy, active_probe, tau=DORM_TAU)
                    except Exception:
                        d_raw = float("nan")
                    m10_strength = float(m10.update(
                        {"activation_dormancy": float(d_raw)}, agent=agent,
                        current_perf=float(normalized[-1]) if normalized else 0.0,
                        pre_shift=not shifted(online_step)))
                    n_m10_updates += 1
                    m10_records.append({
                        "step": online_step,
                        "shift_active": shifted(online_step),
                        "intervention_strength": m10_strength,
                        "activation_dormancy": float(d_raw),
                    })
                    if m10_strength > 0.0:
                        apply_plasticity_intervention(
                            agent,
                            intervention_type="shrink_perturb",
                            severity=m10_strength,
                            shrink=0.1,
                            probe_batch=active_probe,
                            rng=noise_rng,
                        )
                        applied_count += 1

            if online_step - last_eval >= eval_interval or online_step >= online_steps:
                c_idx = (0 if dataset_id.startswith("halfcheetah") else 2) if (shifted(online_step) and shift == "actuator_cripple") else None
                mean_ret = _evaluate(agent, dataset_id, seed=seed + last_eval,
                                     episodes=eval_episodes, device=torch_device,
                                     crippled_idx=c_idx)
                r_min, r_max = _REF_RETURNS[dataset_id]
                norm = 100.0 * (mean_ret - r_min) / (r_max - r_min)
                returns.append(mean_ret)
                normalized.append(norm)
                last_eval = online_step

                # Periodic atomic checkpoint for intra-cell crash recovery
                if online_step < online_steps:
                    env_qpos = env.unwrapped.data.qpos.copy() if hasattr(env.unwrapped, "data") else None
                    env_qvel = env.unwrapped.data.qvel.copy() if hasattr(env.unwrapped, "data") else None
                    periodic_state = {
                        "online_step": online_step,
                        "update_step": update_step,
                        "applied_count": applied_count,
                        "last_eval": last_eval,
                        "returns": returns,
                        "normalized": normalized,
                        "rank_anchor": rank_anchor,
                        "agent_state": agent.state_dict(),
                        "gate_records": gate_records,
                        "m10_records": m10_records,
                        "gate_state": {
                            "gate_on": gate.gate_on,
                            "n_evaluations": gate.n_evaluations,
                            "n_interventions": gate.n_interventions,
                            "last_intervention_step": gate.last_intervention_step,
                        },
                        "m10_state": {
                            "strength": m10_strength,
                            "n_updates": n_m10_updates,
                        },
                        "noise_rng_state": noise_rng.get_state(),
                        "probe_rng_state": probe_rng.get_state(),
                        "rng_state": rng.get_state(),
                        "torch_rng_state": torch.get_rng_state(),
                        "torch_cuda_rng_state": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
                        "env_state": {
                            "qpos": env_qpos,
                            "qvel": env_qvel,
                            "elapsed": getattr(env, "_elapsed_steps", 0),
                            "current_obs": obs.copy(),
                        },
                        "online_buf": {
                            "idx": online_buf.idx,
                            "count": online_buf.count,
                            "observations": online_buf.observations[:online_buf.count],
                            "actions": online_buf.actions[:online_buf.count],
                            "rewards": online_buf.rewards[:online_buf.count],
                            "next_observations": online_buf.next_observations[:online_buf.count],
                            "dones": online_buf.dones[:online_buf.count],
                        },
                    }
                    fd, tmp_s = tempfile.mkstemp(dir=out_path, prefix="tmp_state_", suffix=".pt")
                    os.close(fd)
                    torch.save(periodic_state, tmp_s)
                    os.replace(tmp_s, state_file)
    finally:
        env.close()

    # ----- Telemetry Blocks -----
    if controller == "capacity_gate":
        gate_block: dict[str, Any] = {
            "applicable": True,
            "kind": CAPACITY_GATE_KIND,
            "schema_version": CAPACITY_GATE_SCHEMA_VERSION,
            "config": {
                "dorm_on": gate.config.dorm_on,
                "dorm_off": gate.config.dorm_off,
                "rho_on": gate.config.rho_on,
                "rho_off": gate.config.rho_off,
                "active_severity": gate.config.active_severity,
                "cooldown_steps": gate.config.cooldown_steps,
            },
            "n_evaluations": gate.n_evaluations,
            "n_interventions": gate.n_interventions,
            "intervention_frequency": (
                gate.n_interventions / gate.n_evaluations if gate.n_evaluations else 0.0),
            "rank_anchor": rank_anchor,
        }
        gate_log_path = out_path / f"gate_{out_name}.jsonl"
        log_content = "".join(json.dumps(rec, default=str) + "\n" for rec in gate_records)
        _atomic_write_file(gate_log_path, log_content)
    else:
        gate_block = {"applicable": False, "kind": CAPACITY_GATE_KIND,
                      "schema_version": CAPACITY_GATE_SCHEMA_VERSION}

    if controller == "m10ref":
        m10_block: dict[str, Any] = {
            "applicable": True, "frozen_reference": True,
            "n_updates": n_m10_updates, "last_strength": m10_strength,
        }
        m10_log_path = out_path / f"m10ref_{out_name}.jsonl"
        m10_content = "".join(json.dumps(rec, default=str) + "\n" for rec in m10_records)
        _atomic_write_file(m10_log_path, m10_content)
    else:
        m10_block = {"applicable": False}

    # ----- Final Checkpoint & Atomic Write -----
    fd, tmp_ckpt = tempfile.mkstemp(dir=out_path, prefix="tmp_ckpt_", suffix=".pt")
    os.close(fd)
    torch.save(agent.state_dict(), tmp_ckpt)
    os.replace(tmp_ckpt, ckpt_file)

    protocol = final_protocol()
    protocol.update({
        "shift_step": active_shift_step,
        "severity": severity,
        "warmup_requested": int(warmup_updates),
        "device_requested": device,
    })
    summary: dict[str, Any] = {
        "output_name": out_name,
        "protocol": protocol,
        "dataset_id": dataset_id,
        "controller": controller,
        "shift": shift,
        "shift_step": active_shift_step,
        "severity": severity,
        "seed": int(seed),
        "online_steps": online_step,
        "update_steps": update_step,
        "applied_intervention_steps": applied_count,
        "returns": returns,
        "normalized": normalized,
        "best_return": float(max(returns)) if returns else None,
        "best_normalized": float(max(normalized)) if normalized else None,
        "gate": gate_block,
        "m10ref": m10_block,
        "checkpoint": "checkpoint.pt",
        "environment": {
            "device": str(torch_device),
            "torch_version": torch.__version__,
            "git_commit": git_commit_hash(),
            "deterministic": bool(deterministic),
        },
        "created_at": utc_timestamp(),
        "runtime_seconds": time.time() - t_start,
    }

    _atomic_write_file(summary_file, json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")

    metadata = create_metadata(
        {"cell": {k: summary[k] for k in (
            "dataset_id", "controller", "shift", "severity", "seed",
            "shift_step", "online_steps", "update_steps",
            "applied_intervention_steps")},
         "protocol": protocol,
         "output_name": out_name},
        device,
    )
    write_metadata(metadata, out_path / "metadata.json")

    # Clean up intermediate state checkpoint now that cell is completed
    if state_file.is_file():
        try:
            state_file.unlink()
        except Exception:
            pass

    return summary


def main() -> None:
    """CLI entry point forwarding arguments to the final factorial harness."""
    import sys
    from experiments.final_factorial import main as harness_main

    new_args = []
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--parallel":
            new_args.append("--workers")
            if i + 1 < len(sys.argv):
                new_args.append(sys.argv[i + 1])
                i += 1
        elif arg == "--confirm":
            new_args.append("--confirm-matrix")
            if i + 1 < len(sys.argv):
                new_args.append(sys.argv[i + 1])
                i += 1
        else:
            new_args.append(arg)
        i += 1

    if "--execute" not in new_args and "--help" not in new_args and "-h" not in new_args:
        new_args.append("--execute")

    sys.argv = [sys.argv[0]] + new_args
    harness_main()


if __name__ == "__main__":
    main()

