"""Capacity-aware intervention gate (final proposed method).

Intervene only when capacity degradation is detected. This module is
self-contained and does NOT import or reuse the frozen M10 controller logic
(``AdaptivePlasticityController``) or the legacy ``AdaptiveIntervention``.

Signals (computed on a fixed deterministic probe batch, eval/no_grad):
  1. True activation dormancy: fraction of hidden units that never fire on
     the probe. A unit is dormant iff its mean activation over the probe is
     ``<= 0``. Overall dormancy is the width-weighted mean over hidden ReLU
     layers. Healthy nets score ~0.0; collapsed layers score 1.0.
  2. Relative rank retention ``rho_t = sRank_t / sRank_anchor``, where sRank
     is the entropy-based effective rank (Roy & Vetterli) of last-hidden-
     layer features and ``sRank_anchor`` is captured exactly once at the
     start of online adaptation from the same fixed probe. Healthy nets sit
     at rho ~ 1.0 regardless of environment (absolute sRank varies 40-140
     across envs on real data); collapse drives rho toward 0.

Gate:
  Simple threshold gate with hysteresis (no learned controller, fixed global
  thresholds, no per-environment tuning). The intervention itself is kept
  identical to the existing fixed intervention; only the trigger changes:
  when the gate is ON the caller applies the fixed intervention at the
  configured severity, otherwise it passes observations/rewards through.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

#: Module/schema version for gate logs (independent of M10/M6 versions).
#: v2: relative rank retention (rank_anchor/rho/rho_on/rho_off) replaces
#: absolute sRank thresholds.
CAPACITY_GATE_SCHEMA_VERSION: int = 2
CAPACITY_GATE_KIND: str = "capacity-gate-v1"

#: Default probe size (matches the frozen offline probe convention).
PROBE_SIZE: int = 256

__all__ = [
    "CAPACITY_GATE_SCHEMA_VERSION",
    "CAPACITY_GATE_KIND",
    "PROBE_SIZE",
    "CapacityGateConfig",
    "CapacityGateController",
    "build_fixed_probe",
    "capture_hidden_features",
    "true_activation_dormancy",
    "feature_effective_rank",
    "feature_numerical_rank",
    "compute_capacity_metrics",
]


# ---------------------------------------------------------------------------
# Probe batch
# ---------------------------------------------------------------------------

def build_fixed_probe(
    observations: np.ndarray,
    n: int = PROBE_SIZE,
    seed: int = 0,
) -> torch.Tensor:
    """Build a fixed deterministic probe batch from frozen offline observations.

    Deterministic given ``(observations, n, seed)`` via
    ``np.random.RandomState``. No learning, no fitting. Returns a float32 CPU
    tensor of shape ``[min(n, N), obs_dim]``. Empty input yields an empty
    ``[0, ...]`` tensor (callers map to NaN).
    """
    obs = np.asarray(observations)
    if obs.size == 0 or obs.shape[0] == 0:
        dim = int(obs.shape[1]) if obs.ndim == 2 else 0
        return torch.empty((0, dim), dtype=torch.float32)
    count = int(obs.shape[0])
    rng = np.random.RandomState(seed)
    indices = rng.choice(count, size=min(n, count), replace=(count < n))
    return torch.as_tensor(np.asarray(obs[indices], dtype=np.float32))


# ---------------------------------------------------------------------------
# Hidden-feature capture (eval / no_grad)
# ---------------------------------------------------------------------------

def _resolve_policy_module(policy: torch.nn.Module) -> torch.nn.Module:
    mod = policy
    if hasattr(policy, "policy") and hasattr(policy.policy, "parameters"):
        mod = policy.policy
    return mod


def _hidden_relus(mod: torch.nn.Module) -> list[torch.nn.Module]:
    """Hidden post-activation (ReLU) layers of the policy backbone only."""
    backbone = getattr(mod, "backbone", None)
    if backbone is not None and hasattr(backbone, "network"):
        return [m for m in backbone.network if isinstance(m, torch.nn.ReLU)]
    return [m for m in mod.modules() if isinstance(m, torch.nn.ReLU)]


def capture_hidden_features(
    policy: torch.nn.Module,
    probe_batch: torch.Tensor | np.ndarray | None,
) -> list[torch.Tensor]:
    """Run the fixed probe through the policy backbone, return per-layer ReLU outputs.

    Eval mode + ``no_grad``; the caller's training mode is restored. Returns
    ``[]`` when there is no usable probe or no hidden ReLU layers. Each entry
    is a detached float32 CPU tensor of shape ``[N, H_l]``.
    """
    if probe_batch is None:
        return []
    probe = (
        probe_batch
        if isinstance(probe_batch, torch.Tensor)
        else torch.as_tensor(np.asarray(probe_batch, dtype=np.float32))
    )
    if probe.numel() == 0 or probe.shape[0] == 0:
        return []
    mod = _resolve_policy_module(policy)
    relus = _hidden_relus(mod)
    if not relus:
        return []
    try:
        ref_param = next(mod.parameters())
        target_device = ref_param.device
    except StopIteration:
        target_device = torch.device("cpu")
    flat = probe.detach().to(dtype=torch.float32, device=target_device)
    captured: dict[int, torch.Tensor] = {}
    handles = []
    try:
        for idx, relu in enumerate(relus):
            handles.append(
                relu.register_forward_hook(
                    lambda _m, _inp, out, _i=idx: captured.__setitem__(_i, out.detach())
                )
            )
        was_training = mod.training
        mod.eval()
        try:
            with torch.no_grad():
                _ = mod(flat)
        finally:
            if was_training:
                mod.train()
    finally:
        for h in handles:
            h.remove()
    if len(captured) != len(relus):
        return []
    out: list[torch.Tensor] = []
    for idx in range(len(relus)):
        h = captured[idx].detach().to("cpu").float()
        if h.dim() != 2 or h.shape[0] == 0:
            return []
        out.append(h)
    return out


# ---------------------------------------------------------------------------
# Metric 1: true activation dormancy
# ---------------------------------------------------------------------------

def true_activation_dormancy(
    policy: torch.nn.Module,
    probe_batch: torch.Tensor | np.ndarray | None,
) -> tuple[float, dict[int, float]]:
    """Fraction of hidden units that never fire on the fixed probe batch.

    A unit ``i`` in layer ``l`` is dormant iff its mean activation over the
    probe ``a_i^l = mean_n h[n, i] <= 0`` (exact zero for ReLU means the unit
    was off for every probe sample). Overall dormancy is the width-weighted
    mean of per-layer dormant fractions. Range ``[0, 1]``; empty/missing
    probe or no hidden layers yields ``(NaN, {})``. Deterministic:
    eval/no_grad with a cached probe gives bit-identical results.
    """
    feats = capture_hidden_features(policy, probe_batch)
    if not feats:
        return math.nan, {}
    per_layer: dict[int, float] = {}
    total_dormant = 0
    total_units = 0
    for idx, h in enumerate(feats):
        mean_act = h.mean(dim=0)  # [H_l]
        if not torch.isfinite(mean_act).all():
            return math.nan, {}
        n_dormant = int((mean_act <= 0).sum().item())
        width = int(mean_act.numel())
        per_layer[idx] = float(n_dormant) / float(width) if width else math.nan
        total_dormant += n_dormant
        total_units += width
    if total_units == 0:
        return math.nan, {}
    return float(total_dormant) / float(total_units), per_layer


# ---------------------------------------------------------------------------
# Metric 2: effective rank / numerical rank
# ---------------------------------------------------------------------------

def feature_effective_rank(
    features: torch.Tensor | np.ndarray | None,
    *,
    center: bool = True,
) -> float:
    """Entropy-based effective rank (Roy & Vetterli, 2007) of a feature matrix.

    ``erank(X) = exp(H(p))`` with ``p_i = s_i / sum(s)`` over the singular
    values ``s`` of (optionally mean-centered) ``X``. Range ``[1, rank(X)]``
    for non-degenerate input; ``1.0`` for rank-1; ``NaN`` for empty input,
    all-zero/constant features, or non-finite values. Deterministic pure
    arithmetic (CPU SVD); no learning.
    """
    if features is None:
        return math.nan
    x = (
        features.detach().float().to("cpu")
        if isinstance(features, torch.Tensor)
        else torch.as_tensor(np.asarray(features, dtype=np.float32))
    )
    if x.numel() == 0 or x.dim() != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        return math.nan
    if not torch.isfinite(x).all():
        return math.nan
    with torch.no_grad():
        xc = x - x.mean(dim=0, keepdim=True) if center else x
        if not torch.isfinite(xc).all():
            return math.nan
        try:
            s = torch.linalg.svdvals(xc)
        except Exception:
            return math.nan
    s = s[torch.isfinite(s)]
    s = s[s > 0]
    total = float(s.sum().item()) if s.numel() else 0.0
    if s.numel() == 0 or not math.isfinite(total) or total <= 0.0:
        return math.nan
    p = s / total
    entropy = float((-(p * torch.log(p + 1e-12)).sum()).item())
    if not math.isfinite(entropy):
        return math.nan
    return float(math.exp(entropy))


def feature_numerical_rank(
    features: torch.Tensor | np.ndarray | None,
    *,
    center: bool = True,
    tol: float | None = None,
) -> float:
    """Numerical rank: count of singular values above tolerance.

    Default tolerance follows the standard matrix-rank convention
    ``max(N, H) * eps * s_max`` (float32 eps). Returns ``NaN`` for empty,
    all-zero, or non-finite input. Deterministic pure arithmetic.
    """
    if features is None:
        return math.nan
    x = (
        features.detach().float().to("cpu")
        if isinstance(features, torch.Tensor)
        else torch.as_tensor(np.asarray(features, dtype=np.float32))
    )
    if x.numel() == 0 or x.dim() != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        return math.nan
    if not torch.isfinite(x).all():
        return math.nan
    with torch.no_grad():
        xc = x - x.mean(dim=0, keepdim=True) if center else x
        try:
            s = torch.linalg.svdvals(xc)
        except Exception:
            return math.nan
    s = s[torch.isfinite(s)]
    if s.numel() == 0:
        return math.nan
    smax = float(s[0].item())
    if not math.isfinite(smax) or smax <= 0.0:
        return math.nan
    if tol is None:
        tol = float(max(x.shape) * float(torch.finfo(torch.float32).eps) * smax)
    return float(int((s > tol).sum().item()))


def compute_capacity_metrics(
    policy: torch.nn.Module,
    probe_batch: torch.Tensor | np.ndarray | None,
) -> dict[str, Any]:
    """Compute both capacity signals in one eval/no_grad pass.

    Returns ``{"dormancy", "dormancy_per_layer", "effective_rank",
    "numerical_rank", "rank_layer", "hidden_widths"}``. Rank is measured on
    the LAST hidden ReLU layer (closest to the output head); dormancy is the
    width-weighted overall fraction. Degenerate inputs yield NaN ranks with
    ``rank_layer=None``.
    """
    feats = capture_hidden_features(policy, probe_batch)
    if not feats:
        return {
            "dormancy": math.nan,
            "dormancy_per_layer": {},
            "effective_rank": math.nan,
            "numerical_rank": math.nan,
            "rank_layer": None,
            "hidden_widths": [],
        }
    total_dormant = 0
    total_units = 0
    per_layer: dict[int, float] = {}
    for idx, h in enumerate(feats):
        mean_act = h.mean(dim=0)
        if not torch.isfinite(mean_act).all():
            return {
                "dormancy": math.nan,
                "dormancy_per_layer": {},
                "effective_rank": math.nan,
                "numerical_rank": math.nan,
                "rank_layer": None,
                "hidden_widths": [int(f.shape[1]) for f in feats],
            }
        n_dormant = int((mean_act <= 0).sum().item())
        width = int(mean_act.numel())
        per_layer[idx] = float(n_dormant) / float(width) if width else math.nan
        total_dormant += n_dormant
        total_units += width
    dormancy = float(total_dormant) / float(total_units) if total_units else math.nan
    rank_layer = len(feats) - 1
    return {
        "dormancy": dormancy,
        "dormancy_per_layer": per_layer,
        "effective_rank": feature_effective_rank(feats[rank_layer]),
        "numerical_rank": feature_numerical_rank(feats[rank_layer]),
        "rank_layer": rank_layer,
        "hidden_widths": [int(f.shape[1]) for f in feats],
    }


def compute_agent_capacity_metrics(
    agent: Any,
    probe_batch: torch.Tensor | np.ndarray | None,
) -> dict[str, Any]:
    """Compute capacity metrics for both Critic (primary locus) and Policy."""
    p_metrics = compute_capacity_metrics(agent.policy, probe_batch) if hasattr(agent, "policy") else {}

    c_metrics: dict[str, Any] = {}
    if hasattr(agent, "critic") and probe_batch is not None:
        q1 = getattr(agent.critic, "q1", agent.critic)
        probe_t = (
            probe_batch if isinstance(probe_batch, torch.Tensor)
            else torch.as_tensor(np.asarray(probe_batch), dtype=torch.float32)
        )
        try:
            ref_param = next(q1.parameters())
            target_device = ref_param.device
        except StopIteration:
            target_device = torch.device("cpu")
        probe_t = probe_t.to(device=target_device, dtype=torch.float32)
        if probe_t.ndim == 2 and probe_t.shape[0] > 0:
            first_linear = next((m for m in q1.modules() if isinstance(m, torch.nn.Linear)), None)
            if first_linear is not None and first_linear.in_features > probe_t.shape[-1]:
                diff = first_linear.in_features - probe_t.shape[-1]
                zeros_a = torch.zeros((probe_t.shape[0], diff), dtype=torch.float32, device=target_device)
                qa_probe = torch.cat([probe_t, zeros_a], dim=-1)
            else:
                qa_probe = probe_t
            c_metrics = compute_capacity_metrics(q1, qa_probe)

    c_erank = c_metrics.get("effective_rank", math.nan)
    p_erank = p_metrics.get("effective_rank", math.nan)
    c_dorm = c_metrics.get("dormancy", math.nan)
    p_dorm = p_metrics.get("dormancy", math.nan)

    primary_erank = c_erank if math.isfinite(c_erank) else p_erank
    primary_dorm = max(
        c_dorm if math.isfinite(c_dorm) else 0.0,
        p_dorm if math.isfinite(p_dorm) else 0.0,
    ) if (math.isfinite(c_dorm) or math.isfinite(p_dorm)) else math.nan

    return {
        "effective_rank": primary_erank,
        "dormancy": primary_dorm,
        "critic_effective_rank": c_erank,
        "critic_dormancy": c_dorm,
        "policy_effective_rank": p_erank,
        "policy_dormancy": p_dorm,
        "policy_metrics": p_metrics,
        "critic_metrics": c_metrics,
    }


# ---------------------------------------------------------------------------
# Threshold / hysteresis gate
# ---------------------------------------------------------------------------

@dataclass
class CapacityGateConfig:
    """Fixed global thresholds (no per-environment tuning).

    Rank is RELATIVE: ``rho_t = sRank_t / sRank_anchor`` with the anchor
    captured once at the start of online adaptation, so identical retention
    means identical decisions in every environment despite different absolute
    sRank scales. Dormancy stays absolute (already in [0, 1]).

    Gate turns ON when degradation is detected and stays ON until recovery:

    * ON if ``dormancy >= dorm_on`` OR ``rho <= rho_on``.
    * While ON, turns OFF only when ``dormancy <= dorm_off``
      AND ``rho >= rho_off`` (hysteresis band).
    * NaN/degenerate metrics never trip the gate; observations without a
      usable signal hold the previous state.
    """

    dorm_on: float = 0.15
    dorm_off: float = 0.05
    rho_on: float = 0.70
    rho_off: float = 0.85
    # Intervention strength applied while the gate is ON. Kept identical to
    # the fixed intervention's configured severity by the caller.
    active_severity: float = 0.1
    # Principled refractory cooldown period: minimum online steps between parameter interventions
    cooldown_steps: int = 0

    def __post_init__(self) -> None:
        if not (0.0 <= self.dorm_off <= self.dorm_on <= 1.0):
            raise ValueError(
                f"Require 0 <= dorm_off <= dorm_on <= 1, got {self.dorm_off}, {self.dorm_on}"
            )
        if not (0.0 < self.rho_on <= self.rho_off):
            raise ValueError(
                f"Require 0 < rho_on <= rho_off, got {self.rho_on}, {self.rho_off}"
            )
        if not (0.0 <= self.active_severity <= 1.0):
            raise ValueError(f"active_severity must be in [0,1], got {self.active_severity}")
        if self.cooldown_steps < 0:
            raise ValueError(f"cooldown_steps must be non-negative, got {self.cooldown_steps}")


@dataclass
class CapacityGateController:
    """Threshold gate with hysteresis + full audit logging.

    State: ``gate_on`` (bool), ``history`` (one record per ``update``),
    intervention counters. ``update()`` is deterministic given the metric
    sequence; timestamps use wall-clock only for logging (never for logic).
    """

    config: CapacityGateConfig = field(default_factory=CapacityGateConfig)
    gate_on: bool = False
    history: list[dict[str, Any]] = field(default_factory=list)
    n_interventions: int = 0  # number of updates with gate ON
    n_evaluations: int = 0
    last_intervention_step: int = -999999
    # Rank anchor (raw sRank at the start of online adaptation). None until
    # captured; first finite capture wins, explicit reset only.
    _rank_anchor: float | None = None

    # -- anchor ----------------------------------------------------------

    def set_rank_anchor(self, value: float | None, *, force: bool = False) -> None:
        """Capture ``sRank_anchor`` exactly once (first finite value wins).

        Non-finite values (NaN/inf/None) never create or overwrite the
        anchor. Subsequent calls are ignored unless ``force=True`` (or via
        ``reset_rank_anchor``).
        """
        if value is None:
            return
        v = float(value)
        if math.isfinite(v) and v > 0.0 and (self._rank_anchor is None or force):
            self._rank_anchor = v

    def reset_rank_anchor(self, value: float | None = None) -> None:
        """Explicitly replace (or clear, when None) the rank anchor."""
        self._rank_anchor = None
        self.set_rank_anchor(value, force=True)

    def capture_anchor_from_policy(
        self,
        policy: torch.nn.Module,
        probe_batch: torch.Tensor | np.ndarray | None,
    ) -> float:
        """Capture the anchor from a policy + fixed probe (eval/no_grad).

        Returns the captured raw sRank, or NaN when the probe yields no
        usable rank (anchor left unset).
        """
        metrics = compute_capacity_metrics(policy, probe_batch)
        self.set_rank_anchor(metrics["effective_rank"])
        return self._rank_anchor if self._rank_anchor is not None else math.nan

    @property
    def rank_anchor(self) -> float | None:
        return self._rank_anchor

    @staticmethod
    def retention(effective_rank: float, rank_anchor: float | None) -> float:
        """Relative rank retention ``rho = sRank / anchor`` (NaN if unusable)."""
        if rank_anchor is None:
            return math.nan
        try:
            rho = float(effective_rank) / float(rank_anchor)
        except (TypeError, ValueError, OverflowError, ZeroDivisionError):
            return math.nan
        if not (math.isfinite(float(effective_rank)) and math.isfinite(float(rank_anchor))):
            return math.nan
        if not (float(rank_anchor) > 0.0):
            return math.nan
        return rho if math.isfinite(rho) else math.nan

    # -- core logic ------------------------------------------------------

    def _should_turn_on(self, dormancy: float, rho: float) -> bool:
        dorm_trip = math.isfinite(dormancy) and dormancy >= self.config.dorm_on
        rank_trip = math.isfinite(rho) and rho <= self.config.rho_on
        return bool(dorm_trip or rank_trip)

    def _should_turn_off(self, dormancy: float, rho: float) -> bool:
        dorm_ok = math.isfinite(dormancy) and dormancy <= self.config.dorm_off
        rank_ok = math.isfinite(rho) and rho >= self.config.rho_off
        return bool(dorm_ok and rank_ok)

    def update(
        self,
        *,
        dormancy: float | None = None,
        effective_rank: float | None = None,
        numerical_rank: float | None = None,
        step: int | None = None,
        timestamp: float | None = None,
    ) -> dict[str, Any]:
        """Evaluate the gate on one observation; return the decision record.

        Rank is relative: the first finite ``effective_rank`` observed
        captures ``sRank_anchor`` (exactly once; explicit
        ``set_rank_anchor``/``capture_anchor_from_policy`` preferred) and
        yields ``rho=1.0`` for that update — a fresh anchor never trips the
        gate. Later updates use ``rho = sRank / anchor`` against
        ``rho_on``/``rho_off``. NaN/degenerate rank (or a missing anchor)
        yields NaN rho, which never trips; observations without a usable
        signal hold the previous state. Returned record includes raw sRank,
        anchor, rho, dormancy, thresholds, gate state, intervention
        strength/count/frequency, and timestamp.
        """
        curr_step = self.n_evaluations if step is None else int(step)
        dorm = float(dormancy) if dormancy is not None else math.nan
        erank = float(effective_rank) if effective_rank is not None else math.nan
        nrank = float(numerical_rank) if numerical_rank is not None else math.nan

        # Anchor exactly once: first finite sRank wins (explicit capture
        # preferred; this fallback keeps direct update() callers usable).
        if self._rank_anchor is None and math.isfinite(erank) and erank > 0.0:
            self._rank_anchor = erank
        rho = self.retention(erank, self._rank_anchor)

        prev = self.gate_on
        if not prev:
            if self._should_turn_on(dorm, rho):
                self.gate_on = True
        else:
            if self._should_turn_off(dorm, rho):
                self.gate_on = False

        in_cooldown = (curr_step - self.last_intervention_step < self.config.cooldown_steps)
        should_intervene = bool(self.gate_on and not in_cooldown)

        if should_intervene:
            self.last_intervention_step = curr_step
            self.n_interventions += 1

        strength = float(self.config.active_severity) if should_intervene else 0.0
        self.n_evaluations += 1

        record: dict[str, Any] = {
            "step": curr_step,
            "timestamp": time.time() if timestamp is None else float(timestamp),
            "dormancy": dorm,
            "effective_rank": erank,
            "numerical_rank": nrank,
            "rank_anchor": self._rank_anchor,
            "rho": rho,
            "dorm_on": self.config.dorm_on,
            "dorm_off": self.config.dorm_off,
            "rho_on": self.config.rho_on,
            "rho_off": self.config.rho_off,
            "gate_prev": prev,
            "gate_on": self.gate_on,
            "should_intervene": should_intervene,
            "in_cooldown": in_cooldown,
            "cooldown_steps": self.config.cooldown_steps,
            "last_intervention_step": self.last_intervention_step,
            "intervention_strength": strength,
            "intervention_active": bool(self.gate_on),
            "intervention_count": self.n_interventions,
            "intervention_frequency": self.n_interventions / self.n_evaluations,
            "gate_kind": CAPACITY_GATE_KIND,
            "schema_version": CAPACITY_GATE_SCHEMA_VERSION,
        }
        self.history.append(record)
        return dict(record)

    def update_from_policy(
        self,
        policy: torch.nn.Module,
        probe_batch: torch.Tensor | np.ndarray | None,
        *,
        step: int | None = None,
    ) -> dict[str, Any]:
        """Convenience: compute metrics from a policy + probe, then gate."""
        t0 = time.perf_counter()
        metrics = compute_capacity_metrics(policy, probe_batch)
        record = self.update(
            dormancy=metrics["dormancy"],
            effective_rank=metrics["effective_rank"],
            numerical_rank=metrics["numerical_rank"],
            step=step,
        )
        record["eval_cost_seconds"] = time.perf_counter() - t0
        self.history[-1]["eval_cost_seconds"] = record["eval_cost_seconds"]
        record["dormancy_per_layer"] = metrics["dormancy_per_layer"]
        record["rank_layer"] = metrics["rank_layer"]
        self.history[-1]["dormancy_per_layer"] = metrics["dormancy_per_layer"]
        self.history[-1]["rank_layer"] = metrics["rank_layer"]
        return record

    def capture_anchor_from_agent(
        self,
        agent: Any,
        probe_batch: torch.Tensor | np.ndarray | None,
    ) -> float:
        """Capture rank anchor from both critic and policy."""
        metrics = compute_agent_capacity_metrics(agent, probe_batch)
        self.set_rank_anchor(metrics["effective_rank"])
        return self._rank_anchor if self._rank_anchor is not None else math.nan

    def update_from_agent(
        self,
        agent: Any,
        probe_batch: torch.Tensor | np.ndarray | None,
        *,
        step: int | None = None,
    ) -> dict[str, Any]:
        """Evaluate the gate from an agent's critic + policy capacity."""
        t0 = time.perf_counter()
        metrics = compute_agent_capacity_metrics(agent, probe_batch)
        record = self.update(
            dormancy=metrics["dormancy"],
            effective_rank=metrics["effective_rank"],
            step=step,
        )
        record["eval_cost_seconds"] = time.perf_counter() - t0
        record["critic_effective_rank"] = metrics["critic_effective_rank"]
        record["critic_dormancy"] = metrics["critic_dormancy"]
        record["policy_effective_rank"] = metrics["policy_effective_rank"]
        record["policy_dormancy"] = metrics["policy_dormancy"]
        self.history[-1]["eval_cost_seconds"] = record["eval_cost_seconds"]
        self.history[-1]["critic_effective_rank"] = metrics["critic_effective_rank"]
        self.history[-1]["critic_dormancy"] = metrics["critic_dormancy"]
        self.history[-1]["policy_effective_rank"] = metrics["policy_effective_rank"]
        self.history[-1]["policy_dormancy"] = metrics["policy_dormancy"]
        return record

    # -- logging ---------------------------------------------------------

    def diagnostics(self) -> dict[str, Any]:
        return {
            "gate_on": self.gate_on,
            "rank_anchor": self._rank_anchor,
            "n_evaluations": self.n_evaluations,
            "n_interventions": self.n_interventions,
            "intervention_frequency": (
                self.n_interventions / self.n_evaluations if self.n_evaluations else 0.0
            ),
            "config": {
                "dorm_on": self.config.dorm_on,
                "dorm_off": self.config.dorm_off,
                "rho_on": self.config.rho_on,
                "rho_off": self.config.rho_off,
                "active_severity": self.config.active_severity,
            },
            "history": [dict(h) for h in self.history],
        }

    def write_jsonl(self, path: str | Path) -> Path:
        """Append-free full dump of history as JSONL (one record per line)."""
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for h in self.history:
                f.write(json.dumps(h, default=str) + "\n")
        return path
