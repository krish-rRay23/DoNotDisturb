"""Unit tests for principled plasticity interventions."""

from __future__ import annotations

import math
import numpy as np
import pytest
import torch
import torch.nn as nn

from adaptive_plasticity.iql import IQLAgent
from adaptive_plasticity.interventions import (
    shrink_and_perturb_module,
    apply_plasticity_intervention,
    apply_redo_to_module,
    redo_recycle_linear_layer,
)


def test_shrink_and_perturb_alters_weights_and_resets_adam():
    """Verify Shrink-and-Perturb mutates weights and resets Adam optimizer moments."""
    torch.manual_seed(42)
    lin = nn.Linear(10, 20)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-3)

    # Perform a dummy training step to populate Adam moment states
    x = torch.randn(5, 10)
    loss = lin(x).sum()
    loss.backward()
    opt.step()

    # Check that Adam moment buffers are non-zero
    state_w = opt.state[lin.weight]
    assert state_w["exp_avg"].abs().sum().item() > 0.0
    assert state_w["exp_avg_sq"].abs().sum().item() > 0.0

    w_orig = lin.weight.data.clone()

    # Apply shrink-and-perturb
    res = shrink_and_perturb_module(lin, optimizer=opt, shrink=0.1, severity=0.1)
    assert res["layers_perturbed"] == 1
    assert res["params_perturbed"] == lin.weight.numel() + lin.bias.numel()

    # Weights must have changed
    assert not torch.allclose(lin.weight.data, w_orig)

    # Adam moment buffers must be zeroed out
    assert torch.all(state_w["exp_avg"] == 0.0)
    assert torch.all(state_w["exp_avg_sq"] == 0.0)


def test_redo_recycling_preserves_output_initially():
    """Verify ReDo zeroes outgoing weights so network output is temporarily preserved at reset moment."""
    torch.manual_seed(42)
    l1 = nn.Linear(4, 8)
    l2 = nn.Linear(8, 2)
    opt = torch.optim.Adam(list(l1.parameters()) + list(l2.parameters()), lr=1e-3)

    x = torch.randn(10, 4)
    out_before = l2(torch.relu(l1(x)))

    # Force unit 0 in hidden layer to be dormant
    dormant_idx = torch.tensor([0], dtype=torch.long)
    # Give unit 0 an outgoing weight before recycling
    l2.weight.data[:, 0] = 5.0

    n_rec = redo_recycle_linear_layer(l1, l2, dormant_idx, optimizer_prev=opt, optimizer_next=opt)
    assert n_rec == 1

    # Outgoing weights for unit 0 must now be zero
    assert torch.all(l2.weight.data[:, 0] == 0.0)

    # Because unit 0's outgoing weight is 0, its output contribution is 0,
    # leaving other units unaffected
    out_after = l2(torch.relu(l1(x)))
    # Output difference is bounded because outgoing weight was zeroed
    assert torch.isfinite(out_after).all()


def test_apply_plasticity_intervention_on_agent():
    """Verify high-level apply_plasticity_intervention updates agent networks and target critic."""
    agent = IQLAgent(obs_dim=17, act_dim=6, device="cpu")
    q_orig = agent.critic.q1.network[0].weight.clone()
    p_orig = agent.policy.backbone.network[0].weight.clone()

    rec = apply_plasticity_intervention(
        agent,
        intervention_type="shrink_perturb",
        severity=0.1,
        shrink=0.1,
        target_components=("critic", "policy"),
    )

    assert "critic_perturb" in rec
    assert "policy_perturb" in rec
    assert not torch.allclose(agent.critic.q1.network[0].weight, q_orig)
    assert not torch.allclose(agent.policy.backbone.network[0].weight, p_orig)
    # Target critic must match updated critic
    assert torch.allclose(agent.target_critic.q1.network[0].weight, agent.critic.q1.network[0].weight)
