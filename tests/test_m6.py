"""M6 unit tests: plasticity diagnostics module.

All tests use tiny synthetic models; no real IQL training, no environment.
"""

import math
import numpy as np
import pytest
import torch

from adaptive_plasticity.m6 import (
    DORM_EPS,
    DORM_TAU,
    DORM_PROBE_SIZE,
    DORM_FLOOR,
    ACTIVATION_DORMANCY_KIND,
    DIAGNOSTICS_SCHEMA_VERSION,
    param_magnitudes,
    gradient_magnitudes,
    representation_change,
    performance_change,
    activation_dormancy,
    activation_dormancy_from_probe,
    legacy_weight_sparsity,
    build_probe_batch,
    snapshot_diagnostics,
    DiagnosticsLogger,
)
from adaptive_plasticity.iql import IQLAgent


def _synthetic_probe(obs_dim: int, n: int = 256, seed: int = 0) -> torch.Tensor:
    rng = np.random.RandomState(seed)
    return torch.as_tensor(rng.randn(n, obs_dim).astype(np.float32))


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
    """representation_change returns weight-based 1-cos distance dict."""
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
    # Phase 1: values are 1 - cosine DISTANCE in [0, 1] (0 = identical,
    # 1 = orthogonal-or-opposite) or NaN for missing/zero-norm modules.
    for k, v in cr.items():
        if math.isfinite(v):
            assert 0.0 <= v <= 1.0
    # perturbed policy module must show positive distance
    assert math.isfinite(cr["policy"]) and cr["policy"] > 0.0


def test_representation_change_identical_is_zero():
    """Identical agents give genuine ~0.0 distance (not NaN)."""
    agent = IQLAgent(17, 6, device="cpu")
    cr = representation_change(agent, agent)
    for k, v in cr.items():
        assert math.isfinite(v), f"{k} should be finite for identical agents"
        assert abs(v) < 1e-6, f"{k} should be ~0.0 for identical agents, got {v}"


def test_representation_change_critic_value_separated():
    """Critic/value buckets are computed separately (Phase 1 fix).

    Perturbing only the value head must move the value entry while leaving
    the critic entry at 0.0 -- the old bug concatenated critic+value params
    into the critic bucket and left value stuck at 0.0.
    """
    import copy
    agent = IQLAgent(17, 6, device="cpu")
    old_agent = copy.deepcopy(agent)
    with torch.no_grad():
        for p in old_agent.value.parameters():
            p.add_(0.5)
    cr = representation_change(agent, old_agent)
    assert math.isfinite(cr["value"]) and cr["value"] > 0.0
    assert math.isfinite(cr["critic"]) and abs(cr["critic"]) < 1e-6
    assert math.isfinite(cr["policy"]) and abs(cr["policy"]) < 1e-6


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
    """activation_dormancy is a policy weight sparsity proxy in [0,1]."""
    # build a tiny linear model
    linear = torch.nn.Linear(4, 2)
    d = activation_dormancy(linear, eps=DORM_EPS)
    # should be a float in [0,1] or NaN if nothing to count
    if math.isfinite(d):
        assert 0.0 <= d <= 1.0
    # shared threshold is the single source of truth across M5/M6
    assert DORM_EPS == 1e-4
    # default eps must equal the shared constant
    import inspect
    assert inspect.signature(activation_dormancy).parameters["eps"].default == DORM_EPS


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

    # log step 2 -- same agent object, so distance vs predecessor is 0.0
    d2 = logger.maybe_log(2, agent)
    assert d2 is not None
    # Phase 1: weight-based 1-cos DISTANCE; unchanged params -> 0.0
    dist = d2["representation_change"]["policy"]
    assert math.isfinite(dist)
    assert dist == 0.0
    # perturb the agent, then log step 3 -- distance must become positive
    with torch.no_grad():
        for p in agent.policy.parameters():
            p.add_(0.25)
    d3 = logger.maybe_log(3, agent)
    assert d3 is not None
    dist3 = d3["representation_change"]["policy"]
    assert math.isfinite(dist3)
    assert 0.0 < dist3 <= 1.0


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


# ---------------------------------------------------------------------------
# Phase 3: activation-based neuronal dormancy (neuronal-v1)
# ---------------------------------------------------------------------------

def test_neuronal_dormancy_deterministic():
    """1. Repeated evaluation on the same fixed probe is deterministic."""
    agent = IQLAgent(8, 2, device="cpu")
    probe = _synthetic_probe(8, n=256, seed=0)
    d1, per1 = activation_dormancy_from_probe(agent.policy, probe)
    d2, per2 = activation_dormancy_from_probe(agent.policy, probe)
    assert math.isfinite(d1) and math.isfinite(d2)
    assert d1 == d2
    assert per1 == per2


def test_neuronal_dormancy_fixed_probe_reuse():
    """2. build_probe_batch is deterministic; cached-probe reuse is exact."""
    rng_obs = np.random.RandomState(0).randn(2000, 8).astype(np.float32)
    p1 = build_probe_batch(rng_obs, n=DORM_PROBE_SIZE, seed=123)
    p2 = build_probe_batch(rng_obs, n=DORM_PROBE_SIZE, seed=123)
    p3 = build_probe_batch(rng_obs, n=DORM_PROBE_SIZE, seed=999)
    assert p1.shape == (DORM_PROBE_SIZE, 8)
    assert torch.equal(p1, p2)
    assert not torch.equal(p1, p3)
    agent = IQLAgent(8, 2, device="cpu")
    assert activation_dormancy(agent.policy, p1) == activation_dormancy(agent.policy, p1)


def test_neuronal_dormancy_all_dead_layer():
    """3. A fully zeroed hidden layer reports D_l ~= 1.0."""
    agent = IQLAgent(8, 2, device="cpu")
    probe = _synthetic_probe(8, n=256, seed=1)
    # Zero the Linear feeding the second backbone ReLU (network[2] -> ReLU[3]).
    with torch.no_grad():
        agent.policy.backbone.network[2].weight.zero_()
        agent.policy.backbone.network[2].bias.zero_()
    overall, per_layer = activation_dormancy_from_probe(agent.policy, probe)
    assert set(per_layer.keys()) == {0, 1}
    assert per_layer[1] == pytest.approx(1.0)
    assert overall == pytest.approx((per_layer[0] + 1.0) / 2.0)


def test_neuronal_dormancy_healthy_network_finite():
    """4. A freshly initialized network gives finite D in [0, 1]."""
    agent = IQLAgent(8, 2, device="cpu")
    probe = _synthetic_probe(8, n=256, seed=2)
    overall, per_layer = activation_dormancy_from_probe(agent.policy, probe)
    assert math.isfinite(overall) and 0.0 <= overall <= 1.0
    assert set(per_layer.keys()) == {0, 1}
    for v in per_layer.values():
        assert math.isfinite(v) and 0.0 <= v <= 1.0


def test_neuronal_dormancy_tau_monotonicity():
    """5. Larger tau cannot report less dormancy on a fixed policy+probe."""
    agent = IQLAgent(8, 2, device="cpu")
    probe = _synthetic_probe(8, n=256, seed=3)
    d0, _ = activation_dormancy_from_probe(agent.policy, probe, tau=0.0)
    d1, _ = activation_dormancy_from_probe(agent.policy, probe, tau=DORM_TAU)
    d2, _ = activation_dormancy_from_probe(agent.policy, probe, tau=0.5)
    assert d0 <= d1 <= d2


def test_neuronal_dormancy_overall_per_layer_consistency():
    """6. Overall D equals the width-weighted mean of per-layer D_l."""
    agent = IQLAgent(8, 2, hidden_dim=32, num_layers=2, device="cpu")
    probe = _synthetic_probe(8, n=256, seed=4)
    overall, per_layer = activation_dormancy_from_probe(agent.policy, probe)
    widths = [32, 32]
    expected = sum(per_layer[i] * w for i, w in enumerate(widths)) / sum(widths)
    assert overall == pytest.approx(expected)


def test_neuronal_dormancy_empty_probe_nan():
    """7. Empty/missing probe yields NaN (distinguished from genuine 0.0)."""
    agent = IQLAgent(8, 2, device="cpu")
    empty = torch.empty((0, 8), dtype=torch.float32)
    overall, per_layer = activation_dormancy_from_probe(agent.policy, empty)
    assert math.isnan(overall) and per_layer == {}
    assert math.isnan(activation_dormancy(agent.policy, None))
    assert math.isnan(activation_dormancy(agent.policy, empty))


def test_neuronal_dormancy_controller_directionality():
    """8. Higher absolute dormancy decreases strength (dorm_weight=-0.2)."""
    from adaptive_plasticity.distribution_shifts import AdaptivePlasticityController
    c = AdaptivePlasticityController(seed=42, min_severity=0.0, max_severity=1.0)
    assert c.dorm_weight == pytest.approx(-0.2)
    s_low = c.update({"repr_change": 0.5, "perf_change": 0.0,
                      "activation_dormancy": 0.0, "param_magnitudes": {}})
    s_high = c.update({"repr_change": 0.5, "perf_change": 0.0,
                       "activation_dormancy": 0.5, "param_magnitudes": {}})
    assert s_low > s_high
    assert s_low == pytest.approx(0.2)
    assert s_high == pytest.approx(0.1)


def test_neuronal_dormancy_cross_width_comparability():
    """9. Metric is defined in [0,1] regardless of hidden width."""
    probe = _synthetic_probe(8, n=256, seed=5)
    for width in (64, 256):
        agent = IQLAgent(8, 2, hidden_dim=width, num_layers=2, device="cpu")
        overall, per_layer = activation_dormancy_from_probe(agent.policy, probe)
        assert math.isfinite(overall) and 0.0 <= overall <= 1.0
        assert set(per_layer.keys()) == {0, 1}


def test_neuronal_dormancy_kind_and_schema():
    """Kind tag is neuronal-v1 and schema version is bumped past legacy."""
    assert ACTIVATION_DORMANCY_KIND == "neuronal-v1"
    assert DORM_TAU == pytest.approx(0.1)
    assert DORM_PROBE_SIZE == 256
    assert DORM_FLOOR == pytest.approx(1e-8)
    assert DIAGNOSTICS_SCHEMA_VERSION >= 2
    agent = IQLAgent(8, 2, device="cpu")
    probe = _synthetic_probe(8, n=256, seed=6)
    s = snapshot_diagnostics(agent, probe_batch=probe)
    assert s["activation_dormancy_kind"] == "neuronal-v1"
    assert s["diagnostics_schema_version"] == DIAGNOSTICS_SCHEMA_VERSION
    assert "activation_dormancy_per_layer" in s
    assert "dormancy_delta_vs_anchor" in s


def test_legacy_weight_sparsity_preserved():
    """Old weight-sparsity proxy remains available under its explicit name."""
    linear = torch.nn.Linear(4, 2)
    d = legacy_weight_sparsity(linear, eps=DORM_EPS)
    assert math.isfinite(d) and 0.0 <= d <= 1.0
    assert DORM_EPS == 1e-4