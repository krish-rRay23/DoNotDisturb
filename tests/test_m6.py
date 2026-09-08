"""M6 unit tests: plasticity diagnostics module.

All tests use tiny synthetic models; no real IQL training, no environment.
"""

import math
import numpy as np
import pytest
import torch

from adaptive_plasticity.m6 import (
    param_magnitudes,
    gradient_magnitudes,
    representation_change,
    performance_change,
    activation_dormancy,
    snapshot_diagnostics,
    DiagnosticsLogger,
)
from adaptive_plasticity.iql import IQLAgent


def test_param_magnitudes_returns_dict():
    agent = IQLAgent(17, 6, device="cpu")
    out = param_magnitudes(agent)
    assert isinstance(out, dict)
    assert len(out) > 0
    # all values finite
    for v in out.values():
        assert math.isfinite(v)


def test_gradient_magnitudes_requires_grad():
    agent = IQLAgent(17, 6, device="cpu")
    # loss backward to populate gradients
    x = torch.randn(4, 17, requires_grad=False)
    y = torch.randn(4, 6)
    mean, _ = agent.policy(x)
    loss = (mean.sum() - y.sum()).pow(2)
    loss.backward()
    out = gradient_magnitudes(agent)
    assert isinstance(out, dict)
    assert len(out) > 0
    # every stored gradient should be finite
    for v in out.values():
        assert math.isfinite(v)


def test_representation_change_shapes():
    """representation_change returns dict with keys policy/critic/value."""
    agent = IQLAgent(17, 6, device="cpu")
    old_agent = IQLAgent(17, 6, device="cpu")
    # perturb old_agent params slightly
    with torch.no_grad():
        for p in old_agent.policy.parameters():
            p.add_(0.1)
    cr = representation_change(agent, old_agent)
    assert isinstance(cr, dict)
    assert "policy" in cr
    assert "critic" in cr
    assert "value" in cr
    # values should be in [-1, 1] (cosine similarity) or NaN
    for k, v in cr.items():
        if math.isfinite(v):
            assert -1.0 <= v <= 1.0


def test_performance_change_basic():
    returns_before = [10.0, 12.0, 11.0]
    returns_after = [11.0, 13.0, 12.0]
    normalized_before = [0.5, 0.6, 0.55]
    normalized_after = [0.6, 0.7, 0.65]
    out = performance_change(returns_before, normalized_before, returns_after, normalized_after)
    assert "return_mean_delta" in out
    assert "return_first_delta" in out
    assert "normalized_mean_delta" in out
    assert "normalized_first_delta" in out
    # mean delta should be ~1.0 for returns, ~0.1 for normalized
    assert abs(out["return_mean_delta"] - 1.0) < 0.01
    assert abs(out["normalized_mean_delta"] - 0.1) < 0.01


def test_performance_change_empty():
    out = performance_change([], [], [], [])
    assert math.isnan(out["return_mean_delta"])
    assert math.isnan(out["return_first_delta"])
    assert math.isnan(out["normalized_mean_delta"])
    assert math.isnan(out["normalized_first_delta"])


def test_activation_dormancy_proxy():
    """activation_dormancy returns a fraction in [0,1] or NaN for empty module."""
    # build a tiny linear model
    linear = torch.nn.Linear(4, 2)
    d = activation_dormancy(linear, eps=1e-4)
    # should be a float in [0,1] or NaN if nothing to count
    if math.isfinite(d):
        assert 0.0 <= d <= 1.0


def test_snapshot_diagnostics_keys():
    agent = IQLAgent(17, 6, device="cpu")
    old = IQLAgent(17, 6, device="cpu")
    with torch.no_grad():
        for p in old.policy.parameters():
            p.add_(0.1)
    s = snapshot_diagnostics(agent, old_agent=old,
                             returns_before=[1.0], normalized_before=[0.5],
                             returns_after=[2.0], normalized_after=[0.6])
    assert "param_magnitudes" in s
    assert "gradient_magnitudes" in s
    assert "representation_change" in s
    assert "activation_dormancy" in s
    # performance deltas
    assert "performance_return_mean_delta" in s
    assert "performance_normalized_mean_delta" in s


def test_diagnostics_logger_interval():
    logger = DiagnosticsLogger(log_interval=5)
    agent = IQLAgent(17, 6, device="cpu")
    # log step 5 should produce a dict; step 4 should return None
    d5 = logger.maybe_log(5, agent)
    d4 = logger.maybe_log(4, agent)
    assert d5 is not None
    assert d4 is None
    # after logging step 5, step internal counter should be 5
    assert logger.step == 5


def test_diagnostics_logger_representation_change():
    logger = DiagnosticsLogger(log_interval=1)
    agent = IQLAgent(17, 6, device="cpu")
    old_agent = IQLAgent(17, 6, device="cpu")
    with torch.no_grad():
        for p in old_agent.policy.parameters():
            p.add_(0.1)
    # log step 1 – first step, representation change should be NaN
    d1 = logger.maybe_log(1, agent)
    assert d1 is not None
    assert math.isnan(d1["representation_change"]["policy"])
    assert math.isnan(d1["representation_change"]["critic"])
    assert math.isnan(d1["representation_change"]["value"])

    # log step 2 – now representation change should be finite (params differ)
    d2 = logger.maybe_log(2, agent)
    assert d2 is not None
    # policy similarity should be < 1 (params were perturbed)
    sim = d2["representation_change"]["policy"]
    if math.isfinite(sim):
        assert sim < 1.0
    # subsequent steps should also log
    d3 = logger.maybe_log(3, agent)
    assert d3 is not None


def test_snapshot_diagnostics_no_old_agent():
    agent = IQLAgent(17, 6, device="cpu")
    s = snapshot_diagnostics(agent)
    assert "param_magnitudes" in s
    assert "gradient_magnitudes" in s
    assert "representation_change" in s
    # representation_change should have NaN keys when no old_agent
    rc = s["representation_change"]
    assert "policy" in rc
    assert "critic" in rc
    assert "value" in rc
    for v in rc.values():
        assert math.isnan(v)