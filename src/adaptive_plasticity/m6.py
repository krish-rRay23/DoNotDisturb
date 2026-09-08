from __future__ import annotations

import math
import numpy as np
import torch
from typing import Dict, List, Any

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
    sims: Dict[str, List[float]] = {"policy": [], "critic": [], "value": []}
    old_p = list(old_agent.policy.parameters())
    new_p = list(agent.policy.parameters())
    for old_pm, new_pm in zip(old_p, new_p):
        if old_pm.requires_grad and new_pm.requires_grad and old_pm.dtype.is_floating_point:
            o = old_pm.detach().data.flatten()
            n = new_pm.detach().data.flatten()
            if o.norm().item() > 0 and n.norm().item() > 0:
                sims["policy"].append(float(torch.dot(o, n) / (o.norm().item() * n.norm().item())))
    old_c = list(old_agent.critic.parameters()) + list(old_agent.value.parameters())
    new_c = list(agent.critic.parameters()) + list(agent.value.parameters())
    for old_pm, new_pm in zip(old_c, new_c):
        if old_pm.requires_grad and new_pm.requires_grad and old_pm.dtype.is_floating_point:
            o = old_pm.detach().data.flatten()
            n = new_pm.detach().data.flatten()
            if o.norm().item() > 0 and n.norm().item() > 0:
                sims["critic"].append(float(torch.dot(o, n) / (o.norm().item() * n.norm().item())))
    out: Dict[str, float] = {}
    for key in ("policy", "critic", "value"):
        vals = sims[key]
        mean_val = float(np.mean(vals)) if vals else math.nan
        # Use 1 - cosine similarity; clip to [0, 1] so identical params give 0
        if math.isfinite(mean_val):
            out[key] = max(0.0, min(1.0, 1.0 - mean_val))
        else:
            out[key] = 0.0  # Return 0.0 instead of NaN for missing/zero-norm modules
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


def activation_dormancy(policy: torch.nn.Module, eps: float = 1e-4) -> float:
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


class DiagnosticsLogger:
    def __init__(self, log_interval: int = 100) -> None:
        if log_interval <= 0:
            raise ValueError("log_interval must be a positive integer")
        self.log_interval = log_interval
        self.step = 0
        self._predecessor_params: Dict[str, torch.Tensor] | None = None
        self.param_magnitudes_history: List[Dict[str, float]] = []
        self.gradient_magnitudes_history: List[Dict[str, float]] = []
        self.representation_change_history: Dict[str, float] = {
            "policy": math.nan, "critic": math.nan, "value": math.nan
        }
        self.activation_dormancy_history: List[float] = []
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
        ad = activation_dormancy(agent)
        self.activation_dormancy_history.append(ad)
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
    eps: float = 1e-4,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["param_magnitudes"] = param_magnitudes(agent)
    out["gradient_magnitudes"] = gradient_magnitudes(agent)
    if old_agent is not None:
        out["representation_change"] = representation_change(agent, old_agent)
    else:
        out["representation_change"] = {"policy": math.nan, "critic": math.nan, "value": math.nan}
    out["activation_dormancy"] = activation_dormancy(agent, eps)
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
    out["performance_normalized_first_delta"] = (
        float(np.array(normalized_after, dtype=np.float64)[0] - np.array(normalized_before, dtype=np.float64)[0])
        if normalized_before and normalized_after else math.nan
    )
    out["performance_normalized_first_delta"] = (
        float(np.array(normalized_after, dtype=np.float64)[0] - np.array(normalized_before, dtype=np.float64)[0])
        if normalized_before and normalized_after else math.nan
    )
    return out