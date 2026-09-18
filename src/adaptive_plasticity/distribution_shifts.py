from __future__ import annotations

import math
from typing import Literal, Any, Optional

import numpy as np
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from adaptive_plasticity.iql import IQLAgent

ShiftType = Literal["obs_noise", "reward_scale"]

#: Phase 4C: EMA smoothing factor for the performance-change signal.
#: perf_ema(alpha=0.5) halves the influence of a single noisy evaluation
#: vs the old single-step delta, while remaining a pure deterministic
#: function of the observed evaluation sequence (no RNG, no future data).
PERF_EMA_ALPHA = 0.5

#: Phase 4G: numerically safe floor for pre-shift diagnostic scales.
#: A channel whose pre-shift variation is below this floor normalizes
#: against the floor (never divides by zero or a denormal). Channels with
#: fewer than 2 pre-shift samples use the neutral scale 1.0 instead, so
#: early updates preserve raw magnitudes until variation is observed.
DIAG_SCALE_FLOOR = 1e-6


class InterventionType:
    """Enumeration of intervention types for M9."""
    FIXED = "fixed"
    RANDOM = "random"
    ADAPTIVE = "adaptive"


def apply_observation_noise(
    obs: np.ndarray,
    *,
    severity: float = 0.1,
    max_magnitude: float = None,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    if severity <= 0:
        return obs.copy()
    if rng is None:
        rng = np.random.RandomState()
    if max_magnitude is not None:
        noise = rng.randn(*obs.shape) * severity * max_magnitude
        noisy = obs + noise
        noisy = np.clip(noisy, -max_magnitude - severity * max_magnitude,
                        max_magnitude + severity * max_magnitude)
    else:
        noise = rng.randn(*obs.shape) * severity
        noisy = obs + noise
    return noisy


def apply_reward_scale(
    reward: float,
    *,
    severity: float = 0.5,
    min_reward: float = None,
    max_reward: float = None,
    rng: np.random.RandomState | None = None,
) -> float:
    if severity <= 0:
        return reward
    scaled = reward * severity
    if min_reward is None:
        min_reward = -100.0
    if max_reward is None:
        max_reward = 100.0
    return float(np.clip(scaled, min_reward, max_reward))


class DistributionShiftOrchestrator:
    def __init__(self, *, shift_type, shift_step, severity=0.1):
        if shift_type not in ("obs_noise", "reward_scale"):
            raise ValueError(f"Unknown shift_type {shift_type!r}")
        if shift_step < 0:
            raise ValueError(f"shift_step must be non-negative, got {shift_step}")
        if severity < 0:
            raise ValueError(f"severity must be non-negative, got {severity}")
        self.shift_type = shift_type
        self.shift_step = shift_step
        self.severity = severity
        self._logged = False

    def is_shifted(self, online_step):
        return self.shift_step > 0 and online_step >= self.shift_step

    def apply_to_observation(self, obs, rng=None):
        if self.shift_type == "obs_noise":
            return apply_observation_noise(obs, severity=self.severity, rng=rng)
        return obs

    def apply_to_reward(self, reward):
        if self.shift_type == "reward_scale":
            return apply_reward_scale(reward, severity=self.severity)
        return reward

    def snapshot(self):
        return {"shift_type": self.shift_type, "shift_step": self.shift_step, "severity": self.severity}


class Intervention:
    def __init__(self, intervention_type, shift_step, severity=0.1, seed=None):
        self.intervention_type = intervention_type
        self.shift_step = shift_step
        self.severity = severity
        self._rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        self._logged = False
        self._adaptive_params = {}

    def is_shifted(self, online_step):
        return self.shift_step > 0 and online_step >= self.shift_step

    def apply(self, obs, reward, rng=None):
        raise NotImplementedError

    def diagnostics(self):
        return {
            "intervention_type": self.intervention_type,
            "shift_step": self.shift_step,
            "severity": self.severity,
        }


class FixedIntervention(Intervention):
    intervention_type = InterventionType.FIXED

    def __init__(self, shift_type, shift_step, severity=0.1, seed=None):
        super().__init__(self.intervention_type, shift_step, severity, seed)
        self._shift_type = shift_type

    def apply(self, obs, reward, rng=None):
        transformed_obs = apply_observation_noise(obs, severity=self.severity) if self._shift_type == "obs_noise" else obs.copy()
        transformed_reward = apply_reward_scale(reward, severity=self.severity) if self._shift_type == "reward_scale" else reward
        return transformed_obs, transformed_reward

    def diagnostics(self):
        base = super().diagnostics()
        base["intervention"] = self.intervention_type
        return base


class RandomIntervention(Intervention):
    intervention_type = InterventionType.RANDOM

    def __init__(self, shift_type, shift_step, severity=0.1, seed=None):
        super().__init__(self.intervention_type, shift_step, severity, seed)
        self._shift_type = shift_type
        if severity > 0:
            self._fire_prob = min(1.0, severity)
        else:
            self._fire_prob = 0.0
        self._fired_this_run = False

    def apply(self, obs, reward, rng=None):
        after_shift = self.is_shifted(0)
        if after_shift and not self._fired_this_run:
            self._fired_this_run = True
        transformed_obs = apply_observation_noise(obs, severity=self.severity) if self._shift_type == "obs_noise" else obs.copy()
        transformed_reward = apply_reward_scale(reward, severity=self.severity) if self._shift_type == "reward_scale" else reward
        return transformed_obs, transformed_reward

    def diagnostics(self):
        base = super().diagnostics()
        base["intervention"] = self.intervention_type
        base["fire_prob"] = self._fire_prob
        return base


class AdaptiveIntervention(Intervention):
    intervention_type = InterventionType.ADAPTIVE

    def __init__(self, shift_type, shift_step, severity=0.1, seed=None,
                 repr_weight=0.4, perf_weight=0.4, dorm_weight=-0.2, mag_weight=-0.1,
                 min_severity=0.0, max_severity=1.0):
        super().__init__(self.intervention_type, shift_step, severity, seed)
        self._shift_type = shift_type
        self.repr_weight = repr_weight
        self.perf_weight = perf_weight
        self.dorm_weight = dorm_weight
        self.mag_weight = mag_weight
        self.min_severity = min_severity
        self.max_severity = max_severity
        self._history = [{}]

    def update(self, diagnostics):
        repr_val = diagnostics.get("repr_change", 0.0)
        perf_val = diagnostics.get("perf_change", 0.0)
        dorm_val = diagnostics.get("activation_dormancy", 0.0)
        mag_dict = diagnostics.get("param_magnitudes")
        mag_val = 0.0
        if mag_dict is not None:
            finite_mags = [v for v in mag_dict.values() if isinstance(v, (int, float)) and math.isfinite(v)]
            mag_val = float(np.mean(finite_mags)) if finite_mags else 0.0
        if math.isnan(repr_val):
            repr_val = 0.0
        if math.isnan(perf_val):
            perf_val = 0.0
        if math.isnan(dorm_val):
            dorm_val = 0.0
        raw = (self.repr_weight * repr_val
               + self.perf_weight * perf_val
               + self.dorm_weight * dorm_val
               + self.mag_weight * mag_val)
        strength = max(self.min_severity, min(self.max_severity, raw))
        self._history.append({
            "step": 0, "repr_change": repr_val, "perf_change": perf_val,
            "activation_dormancy": dorm_val, "param_magnitude": mag_val,
            "intervention_strength": strength
        })
        return strength

    def apply(self, obs, reward, rng=None):
        return obs, reward

    def diagnostics(self):
        base = super().diagnostics()
        base["adaptive_params"] = {
            "repr_weight": self.repr_weight,
            "perf_weight": self.perf_weight,
            "dorm_weight": self.dorm_weight,
            "min_severity": self.min_severity,
            "max_severity": self.max_severity,
            "history_len": len(self._history),
        }
        if self._history:
            last = self._history[-1]
            base.update({
                k: v for k, v in last.items() if k != "step"
            })
        return base


class AdaptivePlasticityController:
    """M10 Adaptive Plasticity Controller (Phase 1).

    Rule-based controller that dynamically adjusts intervention strength
    based on online diagnostics from the M6 diagnostics module. Uses only
    information available at decision time, has explicit bounds [0,1], and
    is deterministic given a fixed seed.

    Diagnostic input contract (explicit):
    - ``repr_change``: weight-based 1 - cosine distance of current params vs
      the FIXED anchor captured by ``set_reference_agent()`` at run start
      (Phase 1: not refreshed every update). NaN = missing/invalid anchor,
      distinct from genuine 0.0 (identical params). Used with
      ``repr_weight=+0.4`` (drift up -> strength up).
    - ``perf_change``: Phase 4C EMA-smoothed temporal delta of the
      normalized evaluation return (unbounded). Update rule when
      ``current_perf`` is given (the m5.py path): ``raw = current - prev``
      is logged as ``perf_change_raw``; the control input is the EMA delta
      ``smooth = ema_new - ema_prev`` with
      ``ema_new = alpha*current + (1-alpha)*ema_prev`` (``alpha=0.5`` by
      default, ``perf_ema_alpha`` tunable). First call -> 0.0 for both.
      A single noisy evaluation therefore moves the control input by
      ``alpha`` times its raw jump instead of the full jump. NaN/missing
      -> smoothed input treated as 0.0 for the strength computation but
      the ORIGINAL raw value is preserved in history. When ``current_perf``
      is NOT given, ``diagnostics["perf_change"]`` is used directly
      (legacy direct-input path, unchanged). Used with
      ``perf_weight=-0.4`` (degradation up -> strength up).
    - ``activation_dormancy``: Phase 3 neuronal dormancy D_t in [0, 1]
      (fraction of dormant policy-backbone ReLU hidden units on the fixed
      256-obs probe, tau=0.1, see m6.py; kind="neuronal-v1"). Phase 4E: when
      centering anchors are set (live m5.py path), the equation is fed the
      centered delta ``delta_dormancy = D_t - dormancy_anchor`` while the
      absolute D_t is preserved in history/JSONL for auditability. With no
      anchors set (legacy path), the absolute value feeds the equation
      directly (unchanged). Used with ``dorm_weight=-0.2``.
    - ``param_magnitudes`` is logged for provenance but NOT used in the
      control equation (raw magnitudes are scale-dependent).
    - No input rescaling in Phase 1 (known limitation: unbounded perf deltas
      can saturate the output; documented, not rescaled yet).
    - Phase 4E centering: ``delta_repr = repr_change - repr_anchor`` and
      ``delta_dormancy = activation_dormancy - dormancy_anchor``, where the
      anchors are a stable pre-shift reference captured once via
      ``set_centering_anchors()`` (first-wins; explicit reset only).
      Anchors are stored as finite floats; non-finite candidate values are
      ignored so anchoring defers to the next update. History always logs
      both raw inputs (``repr_change``, ``activation_dormancy``) and the
      centered equation inputs (``delta_repr``, ``delta_dormancy``) plus the
      anchors, so raw vs centered is explicit.
    - Phase 4G normalization: pre-shift scale estimates per channel
      (``repr_scale``, ``perf_scale``, ``dormancy_scale``) from finite
      pre-shift equation inputs only (population std via Welford;
      ``pre_shift=True`` gates accumulation, post-shift samples never
      enter). Normalized inputs ``z_repr = delta_repr / repr_scale``,
      ``z_perf = perf_change / perf_scale``,
      ``z_dorm = delta_dormancy / dormancy_scale`` feed the equation.
      Scales used for the current update derive strictly from prior
      pre-shift observations (use-then-update); <2 samples -> neutral 1.0;
      otherwise ``max(std, DIAG_SCALE_FLOOR)``. History logs the z-values
      and the scales used, so normalized vs raw/centered is explicit.

    Control equation: ``raw = 0.4*z_repr - 0.4*z_perf - 0.2*z_dorm``,
    ``strength = clip(raw, [min_severity, max_severity])``.
    """

    def __init__(
        self,
        *,
        seed: int | None = None,
        repr_weight: float = 0.4,
        perf_weight: float = -0.4,
        dorm_weight: float = -0.2,
        min_severity: float = 0.0,
        max_severity: float = 1.0,
        perf_ema_alpha: float = PERF_EMA_ALPHA,
    ) -> None:
        self._rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        self.repr_weight = repr_weight
        self.perf_weight = perf_weight
        self.dorm_weight = dorm_weight
        self.min_severity = min_severity
        self.max_severity = max_severity
        self.perf_ema_alpha = float(perf_ema_alpha)
        self._history: list[dict[str, Any]] = []
        # Fixed anchor for representation-change diagnostics (Phase 1).
        # Set once via set_reference_agent(); NOT refreshed per update.
        self._prev_agent_params: dict[str, torch.Tensor] | None = None
        self._prev_perf: float | None = None
        # Phase 4C: EMA level of the evaluation return. None until the first
        # finite current_perf is observed; then a deterministic float.
        self._perf_ema: float | None = None
        # Phase 4E: stable pre-shift centering anchors for repr/dormancy.
        # None until set_centering_anchors() captures finite values; first
        # capture wins, explicit reset only (mirror of set_reference_agent).
        self._repr_anchor: float | None = None
        self._dormancy_anchor: float | None = None
        # Phase 4G: Welford running stats per channel over FINITE pre-shift
        # equation inputs only: {"repr"|"perf"|"dorm": [count, mean, M2]}.
        # Post-shift samples never enter (gated by pre_shift=True).
        self._scale_state: dict[str, list[float]] = {
            "repr": [0.0, 0.0, 0.0],
            "perf": [0.0, 0.0, 0.0],
            "dorm": [0.0, 0.0, 0.0],
        }
        self._step: int = 0

    def set_reference_agent(self, agent: IQLAgent, *, force: bool = False) -> None:
        """Capture the FIXED anchor for representation-change comparison.

        Call once at run start (e.g. post-warmup). Subsequent calls are
        ignored unless ``force=True`` (or via ``reset_anchor``), so the
        anchor is never silently replaced after each update. This keeps
        ``repr_change`` a drift-vs-anchor signal instead of a per-step delta.
        """
        if self._prev_agent_params is not None and not force:
            return
        self._prev_agent_params = {
            "policy": self._flatten_module_params(agent.policy),
            "critic": self._flatten_module_params(agent.critic),
            "value": self._flatten_module_params(agent.value),
        }

    def reset_anchor(self, agent: IQLAgent) -> None:
        """Explicitly replace the fixed anchor (use only for a new run/phase)."""
        self.set_reference_agent(agent, force=True)

    def set_centering_anchors(
        self,
        *,
        repr_anchor: float | None = None,
        dormancy_anchor: float | None = None,
        force: bool = False,
    ) -> None:
        """Capture the stable pre-shift reference for centered control inputs.

        First finite capture wins; subsequent calls are ignored unless
        ``force=True`` (or via ``reset_centering_anchors``). Non-finite
        candidates (NaN/inf/None) never overwrite a set anchor and never
        create one, so anchoring deterministically defers to the first
        update with finite diagnostics. Only ever call with pre-shift
        values (live path: first controller update, warmup < shift_step).
        """
        if repr_anchor is not None:
            r = float(repr_anchor)
            if math.isfinite(r) and (self._repr_anchor is None or force):
                self._repr_anchor = r
        if dormancy_anchor is not None:
            d = float(dormancy_anchor)
            if math.isfinite(d) and (self._dormancy_anchor is None or force):
                self._dormancy_anchor = d

    def reset_centering_anchors(
        self,
        *,
        repr_anchor: float | None = None,
        dormancy_anchor: float | None = None,
    ) -> None:
        """Explicitly replace the centering anchors (use only for a new run/phase)."""
        self.set_centering_anchors(
            repr_anchor=repr_anchor, dormancy_anchor=dormancy_anchor, force=True
        )

    def centering_anchors_set(self) -> bool:
        """True once both pre-shift anchors are captured (both finite)."""
        return self._repr_anchor is not None and self._dormancy_anchor is not None

    def _accumulate_scale(self, channel: str, value: float) -> None:
        """Fold one finite pre-shift observation into a channel's Welford stats.

        Non-finite values are ignored (finite-only estimation). Deterministic
        pure arithmetic; no RNG.
        """
        if not math.isfinite(value):
            return
        st = self._scale_state[channel]
        st[0] += 1.0
        delta = value - st[1]
        st[1] += delta / st[0]
        st[2] += delta * (value - st[1])

    def channel_scale(self, channel: str) -> float:
        """Current pre-shift scale for a channel (``repr``/``perf``/``dorm``).

        Population std of the finite pre-shift observations seen so far;
        <2 samples -> neutral 1.0; otherwise ``max(std, DIAG_SCALE_FLOOR)``.
        Deterministic given the call history.
        """
        count, _mean, m2 = self._scale_state[channel]
        if count < 2.0:
            return 1.0
        var = m2 / count
        if not math.isfinite(var) or var < 0.0:
            return DIAG_SCALE_FLOOR
        return max(math.sqrt(var), DIAG_SCALE_FLOOR)
    
    def _flatten_module_params(self, module: torch.nn.Module) -> torch.Tensor:
        """Flatten all trainable parameters of a module."""
        parts: list[torch.Tensor] = []
        for p in module.parameters():
            if p.requires_grad:
                parts.append(p.detach().data.flatten())
        if parts:
            return torch.cat(parts)
        return torch.tensor([], dtype=torch.float32)
    
    def _compute_temporal_repr_change(self, agent: IQLAgent) -> float:
        """Weight-based 1 - cosine distance vs the FIXED anchor.

        Returns math.nan when the anchor is missing/invalid (never 0.0),
        distinguishing "no data" from genuine zero change. Per-module
        (policy/critic/value kept separate) flattened distances, averaged;
        clipped to [0, 1] (0 = identical, 1 = orthogonal-or-opposite).
        """
        if self._prev_agent_params is None:
            return math.nan

        current_params = {
            "policy": self._flatten_module_params(agent.policy),
            "critic": self._flatten_module_params(agent.critic),
            "value": self._flatten_module_params(agent.value),
        }

        sims = []
        for key in ("policy", "critic", "value"):
            old = self._prev_agent_params.get(key)
            cur = current_params.get(key)
            if old is None or cur is None or old.numel() == 0 or cur.numel() == 0:
                continue
            if old.norm().item() > 0 and cur.norm().item() > 0:
                sim = float(torch.dot(old, cur) / (old.norm().item() * cur.norm().item()))
                sims.append(max(0.0, min(1.0, 1.0 - sim)))  # 1 - cosine, clipped to [0,1]

        if not sims:
            return math.nan
        return float(np.mean(sims))

    def update(
        self,
        diagnostics: dict[str, Any],
        agent: IQLAgent | None = None,
        current_perf: float | None = None,
        pre_shift: bool | None = None,
    ) -> float:
        """Compute intervention strength from M6 diagnostics (explicit contract).

        ``pre_shift`` (Phase 4G): True marks this observation as a
        pre-shift sample whose finite equation inputs enter the running
        scale estimates; False/None never accumulate (post-shift samples
        and legacy callers are excluded from scale estimation). Scales used
        for the current update always derive from strictly prior pre-shift
        observations (use-then-update), so a sample never normalizes
        itself. Legacy callers omitting ``pre_shift`` keep scale 1.0 and
        therefore bit-identical outputs.

        Input precedence (all online-available, no future info):
        - repr: from FIXED anchor via ``_compute_temporal_repr_change(agent)``
          when ``agent`` is given; else ``diagnostics["repr_change"]``.
          (Phase 4C: unchanged fixed-anchor semantics.) Phase 4E: the
          equation input is ``delta_repr = repr_raw - repr_anchor`` once
          centering anchors are set, else the raw value (passthrough).
        - perf: when ``current_perf`` is given, the control input is the
          EMA-smoothed delta (see class docstring); the single-step raw
          delta is logged as ``perf_change_raw``. First call -> 0.0.
          When ``current_perf`` is NOT given, ``diagnostics["perf_change"]``
          is used directly (legacy path); ``perf_change_raw`` mirrors it.
          Phase 4E live path: ``current_perf`` is a fresh value-probe
          signal recomputed at every controller update (never a stale
          carried-forward evaluation), so ``perf_change_raw`` is nonzero
          whenever the probe value moves between consecutive updates.
        - dormancy: ``diagnostics["activation_dormancy"]`` (Phase 1 neuronal
          dormancy D_t on the fixed probe; Phase 4C: unchanged).
          Phase 4E: the equation input is
          ``delta_dormancy = dorm_raw - dormancy_anchor`` once anchors are
          set, else the raw value (passthrough).
          ``dorm_weight=-0.2`` preserved.
        - ``param_magnitudes`` logged only, never in the equation.

        NaN/inf inputs contribute 0.0 to ``raw`` but the ORIGINAL values are
        preserved in history, so missing/invalid (NaN) stays distinguishable
        from genuine 0.0. Centered deltas of non-finite raws stay non-finite
        in history (sanitized to 0.0 only for the equation math).
        Normalized z-values of non-finite inputs stay non-finite in history
        (sanitized to 0.0 only for the equation math).
        No rescaling in Phase 1. Weights preserved:
        repr=+0.4, perf=-0.4, dorm=-0.2.

        Returns:
            float in [min_severity, max_severity] subset of [0, 1].
        """
        # Representation change: fixed-anchor recompute wins when agent given
        if agent is not None:
            repr_raw = self._compute_temporal_repr_change(agent)
        else:
            repr_raw = float(diagnostics.get("repr_change", 0.0))

        # Performance change: EMA-smoothed temporal delta wins when
        # current_perf is given; else the legacy direct diagnostics input.
        perf_raw: float  # value entering the control equation
        perf_raw_single: float  # single-step delta, always logged
        # Phase 4G: whether this update's perf observation itself is finite
        # (a non-finite current_perf sanitizes to perf_raw=0.0 for the
        # equation but must NOT enter scale estimation).
        perf_obs_finite = False
        if current_perf is not None:
            cur = float(current_perf)
            prev_before = self._prev_perf
            perf_obs_finite = (
                isinstance(cur, float)
                and math.isfinite(cur)
                and (prev_before is None
                     or (isinstance(prev_before, float)
                         and math.isfinite(prev_before)))
            )
            if self._prev_perf is not None:
                perf_raw_single = cur - self._prev_perf
            else:
                perf_raw_single = 0.0  # First evaluation, no previous to compare
            self._prev_perf = cur
            if self._perf_ema is None:
                # Anchor the EMA level on the first finite observation.
                if isinstance(cur, float) and (math.isnan(cur) or math.isinf(cur)):
                    perf_raw = 0.0
                    # _perf_ema stays None until a finite level is seen.
                else:
                    self._perf_ema = cur
                    perf_raw = 0.0
            else:
                if isinstance(perf_raw_single, float) and (
                    math.isnan(perf_raw_single) or math.isinf(perf_raw_single)
                ):
                    perf_raw = 0.0  # non-finite jump: no EMA update
                elif isinstance(cur, float) and (math.isnan(cur) or math.isinf(cur)):
                    perf_raw = 0.0
                else:
                    alpha = self.perf_ema_alpha
                    ema_new = alpha * cur + (1.0 - alpha) * self._perf_ema
                    perf_raw = ema_new - self._perf_ema
                    self._perf_ema = ema_new
        else:
            perf_raw = float(diagnostics.get("perf_change", 0.0))
            perf_raw_single = perf_raw
            perf_obs_finite = isinstance(perf_raw, float) and math.isfinite(perf_raw)

        dorm_raw = float(diagnostics.get("activation_dormancy", 0.0))
        # param_magnitudes is logged for provenance but NOT used in control equation;
        # raw magnitudes are scale-dependent and not a direct distribution-shift signal.

        # Phase 4E centering: equation inputs are deltas vs the stable
        # pre-shift anchors once captured; otherwise raw passthrough
        # (legacy behavior bit-identical). Non-finite raws stay non-finite
        # in the logged deltas and contribute 0.0 to the equation math.
        if self._repr_anchor is not None:
            try:
                delta_repr = float(repr_raw - self._repr_anchor)
            except (TypeError, ValueError, OverflowError):
                delta_repr = float("nan")
            if isinstance(repr_raw, float) and (
                math.isnan(repr_raw) or math.isinf(repr_raw)
            ):
                delta_repr = float(repr_raw)
        else:
            delta_repr = float(repr_raw)
        if self._dormancy_anchor is not None:
            try:
                delta_dormancy = float(dorm_raw - self._dormancy_anchor)
            except (TypeError, ValueError, OverflowError):
                delta_dormancy = float("nan")
            if isinstance(dorm_raw, float) and (
                math.isnan(dorm_raw) or math.isinf(dorm_raw)
            ):
                delta_dormancy = float(dorm_raw)
        else:
            delta_dormancy = float(dorm_raw)

        # Sanitize for math only; history keeps the original (NaN-preserving)
        def _for_math(v: float) -> float:
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return 0.0
            return v

        # Phase 4G normalization: z-scores vs pre-shift scales estimated from
        # strictly prior finite pre-shift observations (<2 samples: 1.0).
        repr_scale = self.channel_scale("repr")
        perf_scale = self.channel_scale("perf")
        dorm_scale = self.channel_scale("dorm")
        z_repr = float(delta_repr / repr_scale)
        z_perf = float(perf_raw / perf_scale)
        z_dorm = float(delta_dormancy / dorm_scale)
        if isinstance(delta_repr, float) and (
            math.isnan(delta_repr) or math.isinf(delta_repr)
        ):
            z_repr = float(delta_repr)
        if isinstance(perf_raw, float) and (
            math.isnan(perf_raw) or math.isinf(perf_raw)
        ):
            z_perf = float(perf_raw)
        if isinstance(delta_dormancy, float) and (
            math.isnan(delta_dormancy) or math.isinf(delta_dormancy)
        ):
            z_dorm = float(delta_dormancy)

        raw = (
            self.repr_weight * _for_math(z_repr)
            + self.perf_weight * _for_math(z_perf)
            + self.dorm_weight * _for_math(z_dorm)
        )

        # Clip to explicit bounds [min_severity, max_severity] subset of [0,1]
        strength = max(self.min_severity, min(self.max_severity, raw))

        # Phase 4G: fold finite pre-shift equation inputs into the running
        # scales AFTER computing this update (use-then-update). Post-shift
        # (pre_shift=False/None) never accumulates: no post-shift leakage.
        # Non-finite channels are skipped even when pre_shift is True.
        if pre_shift is True:
            self._accumulate_scale("repr", delta_repr)
            if perf_obs_finite:
                self._accumulate_scale("perf", perf_raw)
            self._accumulate_scale("dorm", delta_dormancy)

        # Log the decision (param_magnitude retained in history for diagnostics only).
        # Raw inputs preserved verbatim; centered equation inputs explicit;
        # normalized z-values and the scales used explicit.
        self._history.append({
            "step": 0,  # step will be set by caller
            "repr_change": repr_raw,
            "repr_anchor": self._repr_anchor,
            "delta_repr": delta_repr,
            "perf_change": perf_raw,
            "perf_change_raw": perf_raw_single,
            "activation_dormancy": dorm_raw,
            "dormancy_anchor": self._dormancy_anchor,
            "delta_dormancy": delta_dormancy,
            "repr_scale": repr_scale,
            "perf_scale": perf_scale,
            "dormancy_scale": dorm_scale,
            "z_repr": z_repr,
            "z_perf": z_perf,
            "z_dorm": z_dorm,
            "pre_shift": pre_shift,
            "param_magnitude": 0.0,  # logged only; not used in control equation
            "intervention_strength": strength,
        })

        return strength

    def apply(self, obs: np.ndarray, reward: float, rng: np.random.RandomState | None = None) -> tuple[np.ndarray, float]:
        """No-op apply - controller only modulates severity, doesn't transform data.

        This enforces the separation of inputs (diagnostics), control action
        (strength), and measured outcomes (obs/reward).
        """
        return obs, reward

    def diagnostics(self) -> dict[str, Any]:
        """Return full diagnostics provenance for this controller."""
        base = {
            "repr_weight": self.repr_weight,
            "perf_weight": self.perf_weight,
            "dorm_weight": self.dorm_weight,
            "min_severity": self.min_severity,
            "max_severity": self.max_severity,
        }
        # Merge history entries
        history_entries: list[dict[str, Any]] = []
        for entry in self._history:
            d = {"repr_change": entry["repr_change"],
                 "repr_anchor": entry.get("repr_anchor"),
                 "delta_repr": entry.get("delta_repr", entry["repr_change"]),
                 "perf_change": entry["perf_change"],
                 "perf_change_raw": entry.get("perf_change_raw", entry["perf_change"]),
                 "activation_dormancy": entry["activation_dormancy"],
                 "dormancy_anchor": entry.get("dormancy_anchor"),
                 "delta_dormancy": entry.get("delta_dormancy", entry["activation_dormancy"]),
                 "repr_scale": entry.get("repr_scale", 1.0),
                 "perf_scale": entry.get("perf_scale", 1.0),
                 "dormancy_scale": entry.get("dormancy_scale", 1.0),
                 "z_repr": entry.get("z_repr", entry.get("delta_repr", entry["repr_change"])),
                 "z_perf": entry.get("z_perf", entry["perf_change"]),
                 "z_dorm": entry.get("z_dorm", entry.get("delta_dormancy", entry["activation_dormancy"])),
                 "pre_shift": entry.get("pre_shift"),
                 "param_magnitude": entry["param_magnitude"],
                 "intervention_strength": entry["intervention_strength"]}
            history_entries.append(d)
        base["history"] = history_entries
        return base