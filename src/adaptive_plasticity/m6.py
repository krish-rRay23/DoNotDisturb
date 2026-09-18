from __future__ import annotations

import math
import numpy as np
import torch
from typing import Dict, List, Any

# Phase 3: activation-based neuronal dormancy (single source of truth).
# DORM_EPS is the legacy Phase-1 policy weight-sparsity threshold, kept only
# for backward compatibility (see legacy_weight_sparsity). The live metric is
# neuronal dormancy on a fixed probe batch (see activation_dormancy_from_probe)
# with normalized-activation threshold DORM_TAU.
DORM_EPS: float = 1e-4
DORM_TAU: float = 0.1
DORM_PROBE_SIZE: int = 256
DORM_FLOOR: float = 1e-8
ACTIVATION_DORMANCY_KIND: str = "neuronal-v1"
# Diagnostics schema version: 1 = legacy weight-sparsity era, 2 = neuronal-v1.
# Bump so old logs are never confused with new logs.
DIAGNOSTICS_SCHEMA_VERSION: int = 2


def _cosine_distance(old: torch.Tensor, cur: torch.Tensor) -> float:
    """Weight-based 1 - cosine distance between flattened param vectors.

    Returns math.nan when either vector is empty or has zero norm
    (missing/invalid reference), distinguishing "no data" from genuine
    zero change (identical params -> 0.0). Result clipped to [0, 1] so
    identical params give 0.0 and orthogonal-or-opposite params give 1.0.
    """
    if old.numel() == 0 or cur.numel() == 0:
        return math.nan
    old_norm = float(old.norm().item())
    cur_norm = float(cur.norm().item())
    if not (old_norm > 0 and cur_norm > 0):
        return math.nan
    sim = float(torch.dot(old, cur) / (old_norm * cur_norm))
    return max(0.0, min(1.0, 1.0 - sim))

def param_magnitudes(agent: IQLAgent) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, param in agent.policy.named_parameters():
        if param.requires_grad:
            out[f"policy.{name}"] = float(param.detach().data.norm().item())
    for name, param in agent.critic.named_parameters():
        if param.requires_grad:
            out[f"critic.{name}"] = float(param.detach().data.norm().item())
    for name, param in agent.value.named_parameters():
        if param.requires_grad:
            out[f"value.{name}"] = float(param.detach().data.norm().item())
    return out


def critic_param_magnitudes(agent: IQLAgent) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, param in agent.critic.named_parameters():
        if param.requires_grad:
            out[f"critic.{name}"] = float(param.detach().data.norm().item())
    for name, param in agent.value.named_parameters():
        if param.requires_grad:
            out[f"value.{name}"] = float(param.detach().data.norm().item())
    return out


def policy_param_magnitudes(agent: IQLAgent) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, param in agent.policy.named_parameters():
        if param.requires_grad:
            out[f"policy.{name}"] = float(param.detach().data.norm().item())
    return out


def gradient_magnitudes(agent: IQLAgent) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, param in agent.policy.named_parameters():
        if param.requires_grad and param.grad is not None:
            out[f"policy.{name}"] = float(param.grad.detach().data.norm().item())
    for name, param in agent.critic.named_parameters():
        if param.requires_grad and param.grad is not None:
            out[f"critic.{name}"] = float(param.grad.detach().data.norm().item())
    for name, param in agent.value.named_parameters():
        if param.requires_grad and param.grad is not None:
            out[f"value.{name}"] = float(param.grad.detach().data.norm().item())
    return out


def representation_change(agent: IQLAgent, old_agent: IQLAgent) -> Dict[str, float]:
    """Weight-based 1 - cosine distance per module (policy/critic/value kept separate).

    Reference is the explicit ``old_agent`` argument; callers needing a
    distribution-shift signal should pass a *fixed anchor* (e.g. post-warmup
    checkpoint), not a rolling predecessor. Missing/zero-norm modules yield
    math.nan (not 0.0) to distinguish "no data" from genuine zero change.
    """
    sims: Dict[str, List[float]] = {"policy": [], "critic": [], "value": []}
    for key, mod_old, mod_new in (
        ("policy", old_agent.policy, agent.policy),
        ("critic", old_agent.critic, agent.critic),
        ("value", old_agent.value, agent.value),
    ):
        for old_pm, new_pm in zip(mod_old.parameters(), mod_new.parameters()):
            if old_pm.requires_grad and new_pm.requires_grad and old_pm.dtype.is_floating_point:
                o = old_pm.detach().data.flatten()
                n = new_pm.detach().data.flatten()
                if o.norm().item() > 0 and n.norm().item() > 0:
                    sims[key].append(float(torch.dot(o, n) / (o.norm().item() * n.norm().item())))
    out: Dict[str, float] = {}
    for key in ("policy", "critic", "value"):
        vals = sims[key]
        mean_val = float(np.mean(vals)) if vals else math.nan
        # Use 1 - cosine distance; clip to [0, 1] so identical params give 0.
        # NaN is preserved for missing/zero-norm modules.
        if math.isfinite(mean_val):
            out[key] = max(0.0, min(1.0, 1.0 - mean_val))
        else:
            out[key] = math.nan
    return out


def performance_change(
    returns_before: List[float],
    normalized_before: List[float],
    returns_after: List[float],
    normalized_after: List[float],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if returns_before and returns_after:
        out["return_mean_delta"] = float(np.mean(np.array(returns_after, dtype=np.float64) - np.array(returns_before, dtype=np.float64)))
        out["return_first_delta"] = float(np.array(returns_after, dtype=np.float64)[0] - np.array(returns_before, dtype=np.float64)[0])
    else:
        out["return_mean_delta"] = math.nan
        out["return_first_delta"] = math.nan
    if normalized_before and normalized_after:
        out["normalized_mean_delta"] = float(np.mean(np.array(normalized_after, dtype=np.float64) - np.array(normalized_before, dtype=np.float64)))
        out["normalized_first_delta"] = float(np.array(normalized_after, dtype=np.float64)[0] - np.array(normalized_before, dtype=np.float64)[0])
    else:
        out["normalized_mean_delta"] = math.nan
        out["normalized_first_delta"] = math.nan
    return out


def legacy_weight_sparsity(policy: torch.nn.Module, eps: float = DORM_EPS) -> float:
    """Legacy Phase-1 policy weight sparsity proxy (kept for backward compat).

    Returns the fraction of trainable *policy weights* with |w| < eps.
    NOT neuronal dormancy. Retained so old logs/thresholds stay interpretable;
    new code must use activation_dormancy()/activation_dormancy_from_probe().
    """
    mod = policy
    if hasattr(policy, "policy") and hasattr(policy.policy, "parameters"):
        mod = policy.policy
    total_params = 0
    near_zero = 0
    for p in mod.parameters():
        if p.requires_grad:
            total_params += p.numel()
            near_zero += (p.detach().abs() < eps).sum().item()
    if total_params == 0:
        return math.nan
    return float(near_zero) / float(total_params)


def _resolve_policy_module(policy: torch.nn.Module) -> torch.nn.Module:
    """Unwrap an IQLAgent (or similar) to its policy module when needed."""
    mod = policy
    if hasattr(policy, "policy") and hasattr(policy.policy, "parameters"):
        mod = policy.policy
    return mod


def _policy_hidden_relus(mod: torch.nn.Module) -> List[torch.nn.Module]:
    """Return hidden post-activation (ReLU) layers of the policy backbone only.

    Covers policy.backbone ReLU outputs; excludes the final backbone Linear
    embedding and mean/log_std heads (output layers, not hidden).
    """
    backbone = getattr(mod, "backbone", None)
    if backbone is not None and hasattr(backbone, "network"):
        return [m for m in backbone.network if isinstance(m, torch.nn.ReLU)]
    # Fallback for non-standard policies: ReLUs directly under the module.
    return [m for m in mod.modules() if isinstance(m, torch.nn.ReLU)]


def build_probe_batch(
    observations: np.ndarray,
    n: int = DORM_PROBE_SIZE,
    seed: int = 0,
) -> torch.Tensor:
    """Build a fixed probe batch from frozen offline observations.

    Deterministic given (observations, n, seed) via np.random.RandomState.
    No learning, no fitting. Returns float32 CPU tensor [min(n,N), obs_dim].
    Empty input yields an empty [0, ...] tensor (callers map to NaN).
    """
    obs = np.asarray(observations)
    if obs.size == 0 or obs.shape[0] == 0:
        dim = int(obs.shape[1]) if obs.ndim == 2 else 0
        return torch.empty((0, dim), dtype=torch.float32)
    count = int(obs.shape[0])
    rng = np.random.RandomState(seed)
    indices = rng.choice(count, size=min(n, count), replace=(count < n))
    return torch.as_tensor(np.asarray(obs[indices], dtype=np.float32))


def activation_dormancy_from_probe(
    policy: torch.nn.Module,
    probe_batch: torch.Tensor | np.ndarray | None,
    tau: float = DORM_TAU,
    floor: float = DORM_FLOOR,
) -> tuple[float, Dict[int, float]]:
    """Neuronal dormancy from hidden-unit activations on a fixed probe batch.

    Per-neuron mean-abs activation a_i^l over the probe, normalized by the
    neuron's own layer mean (s_i^l = a_i^l / abar^l); dormant iff s_i^l <= tau.
    Degenerate layer (abar^l <= floor) => D_l = 1.0. Empty/missing probe or no
    hidden ReLU layers => (NaN, {}). Deterministic: eval() + no_grad(), cached
    probe reuse gives bit-identical results. No learned probe model.
    """
    if probe_batch is None:
        return math.nan, {}
    probe = probe_batch if isinstance(probe_batch, torch.Tensor) else torch.as_tensor(
        np.asarray(probe_batch, dtype=np.float32))
    if probe.numel() == 0 or probe.shape[0] == 0:
        return math.nan, {}
    mod = _resolve_policy_module(policy)
    relus = _policy_hidden_relus(mod)
    if not relus:
        return math.nan, {}
    # Match probe device/dtype to the policy parameters.
    try:
        ref_param = next(mod.parameters())
        target_device = ref_param.device
    except StopIteration:
        target_device = torch.device("cpu")
    flat = probe.detach().to(dtype=torch.float32, device=target_device)
    captured: Dict[int, torch.Tensor] = {}
    handles = []
    try:
        for idx, relu in enumerate(relus):
            handles.append(relu.register_forward_hook(
                lambda _m, _inp, out, _i=idx: captured.__setitem__(_i, out.detach())))
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
        return math.nan, {}
    per_layer: Dict[int, float] = {}
    total_dormant = 0
    total_units = 0
    for idx in range(len(relus)):
        h = captured[idx].detach().float()
        if h.dim() != 2 or h.shape[0] == 0:
            return math.nan, {}
        act = h.abs().mean(dim=0)  # [H_l]
        layer_mean = float(act.mean().item())
        width = int(act.numel())
        if not math.isfinite(layer_mean) or layer_mean <= floor:
            d_l = 1.0
            n_dormant = width
        else:
            scores = act / layer_mean
            n_dormant = int((scores <= tau).sum().item())
            d_l = float(n_dormant) / float(width) if width else math.nan
        per_layer[idx] = d_l
        total_dormant += n_dormant
        total_units += width
    if total_units == 0:
        return math.nan, {}
    return float(total_dormant) / float(total_units), per_layer


def activation_dormancy(
    policy: torch.nn.Module,
    probe_batch: torch.Tensor | np.ndarray | None = None,
    tau: float = DORM_TAU,
    eps: float = DORM_EPS,
) -> float:
    """Neuronal activation dormancy (overall fraction, Phase 3 live metric).

    Returns width-weighted overall dormancy D in [0, 1] measured from
    hidden-unit activations on the fixed probe batch (see
    activation_dormancy_from_probe). ``probe_batch=None`` (no probe) yields
    NaN to distinguish "no data" from genuine 0.0. ``eps`` is a deprecated
    legacy signature placeholder (kept so old call sites/signatures still
    import); it is ignored by the neuronal computation -- use
    legacy_weight_sparsity() for the old weight-sparsity proxy. The
    controller keeps dorm_weight = -0.2: higher dormancy -> lower strength.
    """
    _ = eps  # deprecated placeholder; neuronal metric does not use it.
    overall, _ = activation_dormancy_from_probe(policy, probe_batch, tau=tau)
    return overall


class DiagnosticsLogger:
    def __init__(
        self,
        log_interval: int = 100,
        probe_batch: torch.Tensor | np.ndarray | None = None,
        tau: float = DORM_TAU,
    ) -> None:
        if log_interval <= 0:
            raise ValueError("log_interval must be a positive integer")
        self.log_interval = log_interval
        self.probe_batch = probe_batch
        self.tau = tau
        self.step = 0
        self._predecessor_params: Dict[str, torch.Tensor] | None = None
        self.param_magnitudes_history: List[Dict[str, float]] = []
        self.gradient_magnitudes_history: List[Dict[str, float]] = []
        self.representation_change_history: Dict[str, float] = {
            "policy": math.nan, "critic": math.nan, "value": math.nan
        }
        self.activation_dormancy_history: List[float] = []
        self.activation_dormancy_per_layer_history: List[Dict[int, float]] = []
        self._dormancy_reference: float | None = None
        self.performance_change_history: Dict[str, List[float]] = {
            "return_mean_delta": [], "return_first_delta": [],
            "normalized_mean_delta": [], "normalized_first_delta": []
        }

    def _flatten_params(self, module: torch.nn.Module) -> torch.Tensor:
        parts: List[torch.Tensor] = []
        for p in module.parameters():
            if p.requires_grad:
                parts.append(p.detach().data.flatten())
        if parts:
            return torch.cat(parts)
        return torch.tensor([], dtype=torch.float32)

    def _compute_representation_change(self, agent: IQLAgent) -> Dict[str, float]:
        if self._predecessor_params is None:
            return {"policy": math.nan, "critic": math.nan, "value": math.nan}
        out: Dict[str, float] = {}
        for key in ("policy", "critic", "value"):
            old = self._predecessor_params[key]
            if key == "policy":
                cur = self._flatten_params(agent.policy)
            elif key == "critic":
                cur = self._flatten_params(agent.critic)
            else:
                cur = self._flatten_params(agent.value)
            if old.norm().item() > 0 and cur.norm().item() > 0:
                sim = float(torch.dot(old, cur) / (old.norm().item() * cur.norm().item()))
                # Use 1 - cosine similarity; clip to [0, 1] so identical
                # params give 0
                sim = max(0.0, min(1.0, 1.0 - sim))
            else:
                sim = math.nan
            out[key] = sim
        return out

    def maybe_log(
        self,
        step: int,
        agent: IQLAgent,
        returns_before: List[float] | None = None,
        normalized_before: List[float] | None = None,
        returns_after: List[float] | None = None,
        normalized_after: List[float] | None = None,
        probe_batch: torch.Tensor | np.ndarray | None = None,
    ) -> Dict[str, Any] | None:
        if step % self.log_interval != 0:
            return None
        self.step = step
        pm = param_magnitudes(agent)
        self.param_magnitudes_history.append(pm)
        gm = gradient_magnitudes(agent)
        self.gradient_magnitudes_history.append(gm)
        rc = self._compute_representation_change(agent)
        self.representation_change_history = rc
        probe = probe_batch if probe_batch is not None else self.probe_batch
        ad, per_layer = activation_dormancy_from_probe(agent, probe, tau=self.tau)
        self.activation_dormancy_history.append(ad)
        self.activation_dormancy_per_layer_history.append(per_layer)
        # Reference = first finite D (log-only delta; never fed to controller).
        if self._dormancy_reference is None and math.isfinite(ad):
            self._dormancy_reference = ad
        if math.isfinite(ad) and self._dormancy_reference is not None and math.isfinite(self._dormancy_reference):
            dorm_delta = float(ad - self._dormancy_reference)
        else:
            dorm_delta = math.nan
        if returns_before is not None and returns_after is not None and normalized_before is not None and normalized_after is not None:
            pc = performance_change(returns_before, normalized_before, returns_after, normalized_after)
            for k, v in pc.items():
                self.performance_change_history[k].append(v)
        self._predecessor_params = {
            "policy": self._flatten_params(agent.policy),
            "critic": self._flatten_params(agent.critic),
            "value": self._flatten_params(agent.value),
        }
        diagnostics: Dict[str, Any] = {
            "step": step,
            "param_magnitudes": pm,
            "gradient_magnitudes": gm,
            "representation_change": rc,
            "activation_dormancy": ad,
            "activation_dormancy_per_layer": per_layer,
            "activation_dormancy_kind": ACTIVATION_DORMANCY_KIND,
            "dormancy_delta_vs_anchor": dorm_delta,
            "diagnostics_schema_version": DIAGNOSTICS_SCHEMA_VERSION,
            "performance_return_mean_delta": self.performance_change_history["return_mean_delta"][-1:] if self.performance_change_history["return_mean_delta"] else math.nan,
            "performance_return_first_delta": self.performance_change_history["return_first_delta"][-1:] if self.performance_change_history["return_first_delta"] else math.nan,
            "performance_normalized_mean_delta": self.performance_change_history["normalized_mean_delta"][-1:] if self.performance_change_history["normalized_mean_delta"] else math.nan,
            "performance_normalized_first_delta": self.performance_change_history["normalized_first_delta"][-1:] if self.performance_change_history["normalized_first_delta"] else math.nan,
        }
        return diagnostics


def snapshot_diagnostics(
    agent: IQLAgent,
    old_agent: IQLAgent | None = None,
    returns_before: List[float] | None = None,
    normalized_before: List[float] | None = None,
    returns_after: List[float] | None = None,
    normalized_after: List[float] | None = None,
    eps: float = DORM_EPS,
    probe_batch: torch.Tensor | np.ndarray | None = None,
    tau: float = DORM_TAU,
    dormancy_reference: float | None = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["param_magnitudes"] = param_magnitudes(agent)
    out["gradient_magnitudes"] = gradient_magnitudes(agent)
    if old_agent is not None:
        out["representation_change"] = representation_change(agent, old_agent)
    else:
        out["representation_change"] = {"policy": math.nan, "critic": math.nan, "value": math.nan}
    ad, per_layer = activation_dormancy_from_probe(agent, probe_batch, tau=tau)
    out["activation_dormancy"] = ad
    out["activation_dormancy_per_layer"] = per_layer
    out["activation_dormancy_kind"] = ACTIVATION_DORMANCY_KIND
    if dormancy_reference is not None and math.isfinite(ad) and math.isfinite(dormancy_reference):
        out["dormancy_delta_vs_anchor"] = float(ad - dormancy_reference)
    else:
        out["dormancy_delta_vs_anchor"] = math.nan
    out["diagnostics_schema_version"] = DIAGNOSTICS_SCHEMA_VERSION
    returns_before = returns_before or []
    normalized_before = normalized_before or []
    returns_after = returns_after or []
    normalized_after = normalized_after or []
    out["performance_return_mean_delta"] = (
        float(np.mean(np.array(returns_after, dtype=np.float64) - np.array(returns_before, dtype=np.float64)))
        if returns_before and returns_after else math.nan
    )
    out["performance_normalized_mean_delta"] = (
        float(np.mean(np.array(normalized_after, dtype=np.float64) - np.array(normalized_before, dtype=np.float64)))
        if normalized_before and normalized_after else math.nan
    )
    out["performance_return_first_delta"] = (
        float(np.array(returns_after, dtype=np.float64)[0] - np.array(returns_before, dtype=np.float64)[0])
        if returns_before and returns_after else math.nan
    )
    out["performance_normalized_first_delta"] = (
        float(np.array(normalized_after, dtype=np.float64)[0] - np.array(normalized_before, dtype=np.float64)[0])
        if normalized_before and normalized_after else math.nan
    )
    return out