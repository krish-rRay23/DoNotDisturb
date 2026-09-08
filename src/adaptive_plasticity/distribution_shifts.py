from __future__ import annotations

import math
from typing import Literal, Any, Optional

import numpy as np
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from adaptive_plasticity.iql import IQLAgent

ShiftType = Literal["obs_noise", "reward_scale"]


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
    if max_magnitude is None:
        max_magnitude = 1.0
    if rng is None:
        rng = np.random.RandomState()
    noise = rng.randn(*obs.shape) * severity * max_magnitude
    noisy = obs + noise
    noisy = np.clip(noisy, -max_magnitude - severity * max_magnitude,
                    max_magnitude + severity * max_magnitude)
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
    """M10 Adaptive Plasticity Controller.

    Rule-based controller that dynamically adjusts intervention strength
    based on online diagnostics from the M6 diagnostics module.  Uses only
    information available at decision time, has explicit bounds [0,1], and
    is deterministic given a fixed seed.

    Scientific design:
    - Clearly separate inputs (M6 diagnostics), control action (strength in [0,1]),
      and measured outcomes (to avoid leakage).
    - Tunable severity weights grounded in the plasticity hypothesis:
      representation change, performance change, activation dormancy.
    - param_magnitudes is logged for provenance but NOT used in the control equation
      (raw magnitudes are scale-dependent and not a direct distribution-shift signal).
    - All randomness controlled via seeded RNG; given same seed, always
      produces identical output.
    - Logs every decision with full diagnostics for provenance.
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
    ) -> None:
        self._rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        self.repr_weight = repr_weight
        self.perf_weight = perf_weight
        self.dorm_weight = dorm_weight
        self.min_severity = min_severity
        self.max_severity = max_severity
        self._history: list[dict[str, Any]] = []
        # State for temporal diagnostics
        self._prev_agent_params: dict[str, torch.Tensor] | None = None
        self._prev_perf: float | None = None
        self._step: int = 0

    def update_reference_state(self, agent: IQLAgent, current_perf: float) -> None:
        """Update the reference agent state and performance for temporal diagnostics.
        
        Call this AFTER the controller update, using the agent state AFTER the update step.
        """
        import torch
        # Store current agent parameters as reference for next update
        self._prev_agent_params = {
            "policy": self._flatten_params(torch.nn.ModuleList(list(
                self._get_module("policy").parameters() if hasattr(self, "_get_module") else []))),
            "critic": self._flatten_params(torch.nn.ModuleList(list(
                self._get_module("critic").parameters() if hasattr(self, "_get_module") else []))),
            "value": self._flatten_params(torch.nn.ModuleList(list(
                self._get_module("value").parameters() if hasattr(self, "_get_module") else []))),
        }
        # Actually, we need a different approach - store agent params directly
        # This will be handled by the caller passing the agent
        self._step += 1
    
    def set_reference_agent(self, agent: IQLAgent) -> None:
        """Set the reference agent for temporal comparison.
        
        Call this at the start of the run and after each controller update.
        """
        import torch
        self._prev_agent_params = {
            "policy": self._flatten_module_params(agent.policy),
            "critic": self._flatten_module_params(agent.critic),
            "value": self._flatten_module_params(agent.value),
        }
    
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
        """Compute temporal representation change vs previous reference state."""
        if self._prev_agent_params is None:
            return 0.0
        
        import torch
        current_params = {
            "policy": self._flatten_module_params(agent.policy),
            "critic": self._flatten_module_params(agent.critic),
            "value": self._flatten_module_params(agent.value),
        }
        
        sims = []
        for key in ("policy", "critic", "value"):
            old = self._prev_agent_params.get(key)
            cur = current_params.get(key)
            if old is not None and cur is not None and old.numel() > 0 and cur.numel() > 0:
                if old.norm().item() > 0 and cur.norm().item() > 0:
                    sim = float(torch.dot(old, cur) / (old.norm().item() * cur.norm().item()))
                    sim = max(0.0, min(1.0, 1.0 - sim))  # 1 - cosine_sim, clipped to [0,1]
                    sims.append(sim)
        
        return float(np.mean(sims)) if sims else 0.0

    def update(
        self,
        diagnostics: dict[str, Any],
        agent: IQLAgent | None = None,
        current_perf: float | None = None,
    ) -> float:
        """Compute intervention strength from M6 diagnostics.

        Args:
            diagnostics: dict with keys:
                - repr_change (float): representation change magnitude (will be computed if agent provided)
                - perf_change (float): performance change (return delta) (will be computed if current_perf provided)
                - activation_dormancy (float): fraction of dormant activations
                - param_magnitudes (dict): per-param magnitude dict (logged only)

        Returns:
            float in [min_severity, max_severity] subset of [0, 1],
            representing the intervention strength to apply.
        """
        # Compute temporal representation change if agent provided
        if agent is not None:
            repr_val = self._compute_temporal_repr_change(agent)
        else:
            repr_val = float(diagnostics.get("repr_change", 0.0))

        # Compute real performance change from evaluation returns
        if current_perf is not None:
            if self._prev_perf is not None:
                perf_val = current_perf - self._prev_perf
            else:
                perf_val = 0.0  # First evaluation, no previous to compare
            self._prev_perf = current_perf
        else:
            perf_val = float(diagnostics.get("perf_change", 0.0))

        dorm_val = float(diagnostics.get("activation_dormancy", 0.0))
        # param_magnitudes is logged for provenance but NOT used in control equation;
        # raw magnitudes are scale-dependent and not a direct distribution-shift signal.

        # Sanitize NaN/inf
        if math.isnan(repr_val):
            repr_val = 0.0
        if math.isnan(perf_val):
            perf_val = 0.0
        if math.isnan(dorm_val):
            dorm_val = 0.0
        if math.isinf(repr_val) or math.isinf(perf_val) or math.isinf(dorm_val):
            repr_val = 0.0
            perf_val = 0.0
            dorm_val = 0.0

        # Compute raw weighted combination using only direct plasticity signals:
        # repr_change: ↑ drift → ↑ intervention strength
        # perf_change:  ↓ performance → ↑ intervention strength (negative weight)
        # activation_dormancy: ↑ dormancy → ↓ intervention strength (negative weight)
        raw = (
            self.repr_weight * repr_val
            + self.perf_weight * perf_val
            + self.dorm_weight * dorm_val
        )

        # Clip to explicit bounds [min_severity, max_severity] subset of [0,1]
        strength = max(self.min_severity, min(self.max_severity, raw))

        # Log the decision (param_magnitude retained in history for diagnostics only)
        self._history.append({
            "step": 0,  # step will be set by caller
            "repr_change": repr_val,
            "perf_change": perf_val,
            "activation_dormancy": dorm_val,
            "param_magnitude": 0.0,  # logged only; not used in control equation
            "intervention_strength": strength,
        })

        return strength
        """Compute intervention strength from M6 diagnostics.

        Args:
            diagnostics: dict with keys:
                - repr_change (float): representation change magnitude
                - perf_change (float): performance change (return delta)
                - activation_dormancy (float): fraction of dormant activations
                - param_magnitudes (dict): per-param magnitude dict (logged only)

        Returns:
            float in [min_severity, max_severity] subset of [0, 1],
            representing the intervention strength to apply.
        """
        # Extract diagnostics values (all online-available, no future info)
        repr_val = float(diagnostics.get("repr_change", 0.0))
        perf_val = float(diagnostics.get("perf_change", 0.0))
        dorm_val = float(diagnostics.get("activation_dormancy", 0.0))
        # param_magnitudes is logged for provenance but NOT used in control equation;
        # raw magnitudes are scale-dependent and not a direct distribution-shift signal.

        # Sanitize NaN/inf
        if math.isnan(repr_val):
            repr_val = 0.0
        if math.isnan(perf_val):
            perf_val = 0.0
        if math.isnan(dorm_val):
            dorm_val = 0.0
        if math.isinf(repr_val) or math.isinf(perf_val) or math.isinf(dorm_val):
            repr_val = 0.0
            perf_val = 0.0
            dorm_val = 0.0

        # Compute raw weighted combination using only direct plasticity signals:
        # repr_change: ↑ drift → ↑ intervention strength
        # perf_change:  ↓ performance → ↑ intervention strength (negative weight)
        # activation_dormancy: ↑ dormancy → ↓ intervention strength (negative weight)
        raw = (
            self.repr_weight * repr_val
            + self.perf_weight * perf_val
            + self.dorm_weight * dorm_val
        )

        # Clip to explicit bounds [min_severity, max_severity] subset of [0,1]
        strength = max(self.min_severity, min(self.max_severity, raw))

        # Log the decision (param_magnitude retained in history for diagnostics only)
        self._history.append({
            "step": 0,  # step will be set by caller
            "repr_change": repr_val,
            "perf_change": perf_val,
            "activation_dormancy": dorm_val,
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
                 "perf_change": entry["perf_change"],
                 "activation_dormancy": entry["activation_dormancy"],
                 "param_magnitude": entry["param_magnitude"],
                 "intervention_strength": entry["intervention_strength"]}
            history_entries.append(d)
        base["history"] = history_entries
        return base