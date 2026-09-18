"""Principled neural network parameter interventions for plasticity preservation.

Grounds all interventions in peer-reviewed literature:
1. Shrink-and-Perturb (Ash & Adams, ICLR 2020; Nikishin et al., ICML 2022):
   W <- (1 - lambda) * W + eta * sigma_init * Z, Z ~ N(0, I)
   where sigma_init = sqrt(2 / fan_in) (Kaiming standard deviation for ReLU).
   Adam first and second moment buffers (exp_avg, exp_avg_sq) are reset for
   perturbed parameters.
2. Dormant Neuron Recycling / ReDo (Sokar et al., ICML 2023):
   Identifies dormant units whose normalized activation score over a probe
   batch satisfies:
       s_i = mean_n(|h_{n,i}|) / [ (1/H) sum_j mean_n(|h_{n,j}|) ] < tau
   For each dormant unit i:
       Incoming weights and biases are re-initialized from Kaiming uniform.
       Outgoing weights are set to 0 (temporarily preserving function output).
       Adam optimizer states for those parameter slices are cleared to zero.
3. Head / Critic Reset (Nikishin et al., 2022; Schwarzer et al., 2023):
   Re-initializes the output projection head or penultimate layer, clearing
   optimizer state.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def reset_optimizer_state_for_param(
    optimizer: torch.optim.Optimizer | None,
    param: torch.nn.Parameter,
    mask: torch.Tensor | None = None,
) -> None:
    """Reset Adam moment accumulators (exp_avg, exp_avg_sq) for a parameter.

    If ``mask`` is None, the entire moment tensor is cleared.
    If ``mask`` is provided (e.g. for individual neurons), only the selected
    indices in the moment tensor are zeroed out.
    """
    if optimizer is None:
        return
    state = optimizer.state.get(param)
    if not state:
        return
    for key in ("exp_avg", "exp_avg_sq"):
        buf = state.get(key)
        if buf is not None:
            if mask is None:
                buf.zero_()
            else:
                buf[mask] = 0.0


def shrink_and_perturb_module(
    module: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    shrink: float = 0.1,
    severity: float = 0.1,
    rng: np.random.RandomState | None = None,
) -> dict[str, Any]:
    """Apply Shrink-and-Perturb (Ash & Adams 2020) to all Linear layers of a module.

    W <- (1 - shrink) * W + severity * sigma_init * Z, Z ~ N(0, I)
    sigma_init is Kaiming He normal std: sqrt(2 / fan_in).
    Biases are shrunk toward 0.
    Adam moments are reset to zero for all perturbed weights.
    """
    shrink = float(np.clip(shrink, 0.0, 1.0))
    severity = float(np.clip(severity, 0.0, 1.0))
    if shrink == 0.0 and severity == 0.0:
        return {"layers_perturbed": 0, "params_perturbed": 0}

    device = next(module.parameters()).device
    layers_count = 0
    params_count = 0

    with torch.no_grad():
        for layer in module.modules():
            if isinstance(layer, nn.Linear):
                weight = layer.weight
                fan_in = weight.shape[1]
                sigma_init = math.sqrt(2.0 / float(fan_in))

                if rng is not None:
                    noise_np = rng.randn(*weight.shape).astype(np.float32)
                    noise = torch.as_tensor(noise_np, device=device)
                else:
                    noise = torch.randn_like(weight)

                # W <- (1 - shrink) * W + severity * sigma_init * noise
                weight.data.mul_(1.0 - shrink)
                weight.data.add_(noise * (severity * sigma_init))

                if layer.bias is not None:
                    layer.bias.data.mul_(1.0 - shrink)
                    if rng is not None:
                        b_noise_np = rng.randn(*layer.bias.shape).astype(np.float32)
                        b_noise = torch.as_tensor(b_noise_np, device=device)
                    else:
                        b_noise = torch.randn_like(layer.bias)
                    layer.bias.data.add_(b_noise * (severity * sigma_init))
                    reset_optimizer_state_for_param(optimizer, layer.bias)

                reset_optimizer_state_for_param(optimizer, weight)
                layers_count += 1
                params_count += weight.numel() + (layer.bias.numel() if layer.bias is not None else 0)

    return {
        "layers_perturbed": layers_count,
        "params_perturbed": params_count,
        "shrink": shrink,
        "severity": severity,
    }


def redo_recycle_linear_layer(
    linear_prev: nn.Linear,
    linear_next: nn.Linear,
    dormant_indices: torch.Tensor,
    optimizer_prev: torch.optim.Optimizer | None = None,
    optimizer_next: torch.optim.Optimizer | None = None,
) -> int:
    """Recycle dormant neurons between consecutive linear layers (Sokar et al., 2023).

    For dormant unit indices i in hidden layer:
    1. Re-initialize incoming weights linear_prev.weight[i, :] with Kaiming uniform.
    2. Reset incoming bias linear_prev.bias[i] to 0.
    3. Zero outgoing weights linear_next.weight[:, i] = 0.
    4. Reset Adam moments for modified parameters.
    """
    if dormant_indices.numel() == 0:
        return 0

    n_recycled = int(dormant_indices.numel())
    device = linear_prev.weight.device
    idx = dormant_indices.to(device)

    with torch.no_grad():
        fan_in = linear_prev.weight.shape[1]
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0

        # 1. Re-initialize incoming weights
        new_in = torch.empty((n_recycled, fan_in), device=device, dtype=linear_prev.weight.dtype)
        nn.init.uniform_(new_in, -bound, bound)
        linear_prev.weight.data[idx, :] = new_in

        # 2. Reset incoming bias
        if linear_prev.bias is not None:
            linear_prev.bias.data[idx] = 0.0
            reset_optimizer_state_for_param(optimizer_prev, linear_prev.bias, mask=idx)

        reset_optimizer_state_for_param(optimizer_prev, linear_prev.weight, mask=idx)

        # 3. Zero outgoing weights to preserve function output at reset moment
        linear_next.weight.data[:, idx] = 0.0
        # For outgoing weight matrix of shape [out_dim, in_dim], column idx is modified
        col_mask = torch.zeros_like(linear_next.weight.data, dtype=torch.bool)
        col_mask[:, idx] = True
        reset_optimizer_state_for_param(optimizer_next, linear_next.weight, mask=col_mask)

    return n_recycled


def extract_sequential_linear_layers(mod: nn.Module) -> list[nn.Linear]:
    """Extract Linear layers in feed-forward sequence."""
    linears = []
    backbone = getattr(mod, "backbone", None)
    if backbone is not None and hasattr(backbone, "network"):
        for m in backbone.network:
            if isinstance(m, nn.Linear):
                linears.append(m)
    elif hasattr(mod, "network"):
        for m in mod.network:
            if isinstance(m, nn.Linear):
                linears.append(m)
    else:
        for m in mod.modules():
            if isinstance(m, nn.Linear):
                linears.append(m)
    return linears


def apply_redo_to_module(
    module: nn.Module,
    hidden_features: list[torch.Tensor],
    optimizer: torch.optim.Optimizer | None = None,
    tau: float = 0.1,
) -> dict[str, Any]:
    """Apply ReDo recycling to all hidden layers of a module.

    hidden_features contains activation tensors [N, H_l] for each hidden layer.
    """
    linears = extract_sequential_linear_layers(module)
    if len(linears) < 2 or len(hidden_features) < 1:
        return {"recycled_neurons": 0, "dormant_fraction": 0.0}

    total_units = 0
    total_recycled = 0

    with torch.no_grad():
        for l_idx, h in enumerate(hidden_features):
            if l_idx + 1 >= len(linears):
                break
            prev_lin = linears[l_idx]
            next_lin = linears[l_idx + 1]

            mean_acts = h.abs().mean(dim=0)  # [H_l]
            layer_mean = mean_acts.mean() + 1e-12
            scores = mean_acts / layer_mean  # [H_l] relative score

            dormant_mask = (scores < tau)
            dormant_idx = torch.where(dormant_mask)[0]

            width = int(scores.numel())
            total_units += width

            if dormant_idx.numel() > 0:
                n_rec = redo_recycle_linear_layer(
                    prev_lin, next_lin, dormant_idx,
                    optimizer_prev=optimizer, optimizer_next=optimizer
                )
                total_recycled += n_rec

    return {
        "recycled_neurons": total_recycled,
        "total_hidden_units": total_units,
        "recycled_fraction": total_recycled / total_units if total_units > 0 else 0.0,
    }


def apply_plasticity_intervention(
    agent: Any,
    *,
    intervention_type: str = "shrink_perturb",
    severity: float = 0.1,
    shrink: float = 0.1,
    probe_batch: torch.Tensor | np.ndarray | None = None,
    tau_redo: float = 0.1,
    target_components: tuple[str, ...] = ("critic", "policy"),
    rng: np.random.RandomState | None = None,
) -> dict[str, Any]:
    """Apply a verified parameter intervention to the IQLAgent in-place.

    Parameters:
        agent: IQLAgent instance (has policy, critic, value, actor_optimizer, critic_optimizer, value_optimizer).
        intervention_type: 'shrink_perturb' | 'redo' | 'head_reset'.
        severity: noise/perturbation severity factor in [0, 1].
        shrink: weight shrinkage factor in [0, 1] for shrink_perturb.
        probe_batch: deterministic probe batch (required for ReDo activation scores).
        tau_redo: relative activation threshold for ReDo (default: 0.1).
        target_components: which networks to perturb ('critic', 'policy', 'value').
        rng: optional seeded NumPy RandomState for determinism.

    Returns:
        Summary dict of intervention metrics.
    """
    records: dict[str, Any] = {
        "intervention_type": intervention_type,
        "severity": severity,
        "shrink": shrink,
        "target_components": list(target_components),
    }

    if intervention_type == "none" or severity <= 0.0:
        return records

    if intervention_type == "shrink_perturb":
        # Apply Shrink-and-Perturb to Critic
        if "critic" in target_components and hasattr(agent, "critic"):
            c_rec = shrink_and_perturb_module(
                agent.critic, optimizer=getattr(agent, "critic_optimizer", None),
                shrink=shrink, severity=severity, rng=rng,
            )
            records["critic_perturb"] = c_rec
            # Target critic Polyak target is also synced to prevent wild TD divergence
            if hasattr(agent, "target_critic"):
                agent.target_critic.load_state_dict(agent.critic.state_dict())

        # Apply Shrink-and-Perturb to Policy
        if "policy" in target_components and hasattr(agent, "policy"):
            p_rec = shrink_and_perturb_module(
                agent.policy, optimizer=getattr(agent, "actor_optimizer", None),
                shrink=shrink, severity=severity, rng=rng,
            )
            records["policy_perturb"] = p_rec

        # Apply to Value network if requested
        if "value" in target_components and hasattr(agent, "value"):
            v_rec = shrink_and_perturb_module(
                agent.value, optimizer=getattr(agent, "value_optimizer", None),
                shrink=shrink, severity=severity, rng=rng,
            )
            records["value_perturb"] = v_rec

    elif intervention_type == "redo":
        from adaptive_plasticity.capacity_gate import capture_hidden_features

        # ReDo on Critic
        if "critic" in target_components and hasattr(agent, "critic") and probe_batch is not None:
            # For twin critic, we evaluate and recycle on q1
            q1 = getattr(agent.critic, "q1", None)
            if q1 is not None:
                # Probe needs state-action; for critic probe, append zeros as actions
                probe_t = (
                    probe_batch if isinstance(probe_batch, torch.Tensor)
                    else torch.as_tensor(np.asarray(probe_batch), dtype=torch.float32)
                )
                ref_p = next(agent.critic.parameters())
                probe_t = probe_t.to(device=ref_p.device, dtype=torch.float32)
                act_dim = getattr(agent.policy, "act_dim", 6)
                zeros_a = torch.zeros((probe_t.shape[0], act_dim), device=probe_t.device, dtype=torch.float32)
                qa_probe = torch.cat([probe_t, zeros_a], dim=-1)

                feats_q1 = capture_hidden_features(q1, qa_probe)
                redo_c = apply_redo_to_module(
                    q1, feats_q1, optimizer=getattr(agent, "critic_optimizer", None), tau=tau_redo
                )
                records["critic_redo"] = redo_c
                # Sync q2 and target critic
                if hasattr(agent, "target_critic"):
                    agent.target_critic.load_state_dict(agent.critic.state_dict())

        # ReDo on Policy
        if "policy" in target_components and hasattr(agent, "policy") and probe_batch is not None:
            feats_p = capture_hidden_features(agent.policy, probe_batch)
            redo_p = apply_redo_to_module(
                agent.policy, feats_p, optimizer=getattr(agent, "actor_optimizer", None), tau=tau_redo
            )
            records["policy_redo"] = redo_p

    return records
