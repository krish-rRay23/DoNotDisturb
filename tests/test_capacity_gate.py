"""Unit tests for the capacity-aware intervention gate (relative-rank amendment).

Relative retention ``rho_t = sRank_t / sRank_anchor`` drives the rank
channel (anchor captured once); dormancy stays absolute. Tiny synthetic
models only; no training, no environment, no M10 imports.
"""

import math

import numpy as np
import pytest
import torch

from adaptive_plasticity.capacity_gate import (
    CAPACITY_GATE_KIND,
    CAPACITY_GATE_SCHEMA_VERSION,
    PROBE_SIZE,
    CapacityGateConfig,
    CapacityGateController,
    build_fixed_probe,
    capture_hidden_features,
    compute_capacity_metrics,
    feature_effective_rank,
    feature_numerical_rank,
    true_activation_dormancy,
)
from adaptive_plasticity.iql import IQLAgent


def _agent(obs_dim: int = 8, act_dim: int = 2, seed: int = 0) -> IQLAgent:
    torch.manual_seed(seed)
    np.random.seed(seed)
    return IQLAgent(obs_dim, act_dim, device="cpu")


def _probe(obs_dim: int = 8, n: int = 256, seed: int = 0) -> torch.Tensor:
    rng = np.random.RandomState(seed)
    return torch.as_tensor(rng.randn(n, obs_dim).astype(np.float32))


# ---------------------------------------------------------------------------
# Dormancy (unchanged)
# ---------------------------------------------------------------------------

def test_dormancy_healthy_near_zero():
    agent = _agent()
    probe = _probe()
    overall, per_layer = true_activation_dormancy(agent.policy, probe)
    assert math.isfinite(overall) and 0.0 <= overall <= 1.0
    assert set(per_layer.keys()) == {0, 1}
    assert overall <= 0.05


def test_dormancy_killed_layer_reports_one():
    agent = _agent()
    probe = _probe()
    with torch.no_grad():
        agent.policy.backbone.network[2].weight.zero_()
        agent.policy.backbone.network[2].bias.zero_()
    overall, per_layer = true_activation_dormancy(agent.policy, probe)
    assert per_layer[1] == pytest.approx(1.0)
    assert overall == pytest.approx((per_layer[0] + 1.0) / 2.0)


def test_dormancy_width_weighted_consistency():
    agent = _agent()
    probe = _probe()
    overall, per_layer = true_activation_dormancy(agent.policy, probe)
    widths = [256, 256]
    expected = sum(per_layer[i] * w for i, w in enumerate(widths)) / sum(widths)
    assert overall == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Rank metrics (unchanged computations)
# ---------------------------------------------------------------------------

def test_effective_rank_healthy_high_and_bounded():
    agent = _agent()
    probe = _probe()
    feats = capture_hidden_features(agent.policy, probe)
    assert len(feats) == 2
    er = feature_effective_rank(feats[-1])
    nr = feature_numerical_rank(feats[-1])
    assert math.isfinite(er) and 1.0 <= er <= 256.0
    assert er > 50.0
    assert math.isfinite(nr) and nr >= er * 0.5


def test_effective_rank_identity_is_full():
    x = torch.eye(32, dtype=torch.float32)
    assert feature_effective_rank(x, center=False) == pytest.approx(32.0, rel=1e-4)


def test_effective_rank_rank_one_is_one():
    x = torch.ones((64, 16), dtype=torch.float32)
    assert feature_effective_rank(x, center=False) == pytest.approx(1.0, abs=1e-6)
    assert math.isnan(feature_effective_rank(x, center=True))
    u = torch.linspace(0.0, 1.0, 64).unsqueeze(1)
    v = torch.linspace(1.0, 2.0, 16).unsqueeze(0)
    assert feature_effective_rank(u @ v, center=True) == pytest.approx(1.0, abs=1e-5)


def test_numerical_rank_collapsed_is_low():
    x = torch.ones((64, 16), dtype=torch.float32)
    assert feature_numerical_rank(x, center=False) <= 2.0
    assert math.isnan(feature_numerical_rank(x, center=True))


def test_compute_capacity_metrics_keys():
    agent = _agent()
    probe = _probe()
    m = compute_capacity_metrics(agent.policy, probe)
    assert set(m) >= {"dormancy", "dormancy_per_layer", "effective_rank",
                      "numerical_rank", "rank_layer", "hidden_widths"}
    assert m["rank_layer"] == 1
    assert m["hidden_widths"] == [256, 256]


# ---------------------------------------------------------------------------
# Relative retention + anchor-once
# ---------------------------------------------------------------------------

def test_retention_computation():
    assert CapacityGateController.retention(70.0, 100.0) == pytest.approx(0.7)
    assert CapacityGateController.retention(100.0, 100.0) == pytest.approx(1.0)
    assert CapacityGateController.retention(120.0, 100.0) == pytest.approx(1.2)
    assert math.isnan(CapacityGateController.retention(70.0, None))
    assert math.isnan(CapacityGateController.retention(math.nan, 100.0))
    assert math.isnan(CapacityGateController.retention(70.0, 0.0))
    assert math.isnan(CapacityGateController.retention(70.0, -5.0))
    assert math.isnan(CapacityGateController.retention(math.inf, 100.0))


def test_anchor_captured_once_first_wins():
    c = CapacityGateController()
    assert c.rank_anchor is None
    c.set_rank_anchor(100.0)
    assert c.rank_anchor == pytest.approx(100.0)
    c.set_rank_anchor(200.0)  # ignored without force
    assert c.rank_anchor == pytest.approx(100.0)
    c.set_rank_anchor(200.0, force=True)
    assert c.rank_anchor == pytest.approx(200.0)
    c.reset_rank_anchor()
    assert c.rank_anchor is None


def test_anchor_ignores_nonfinite():
    c = CapacityGateController()
    for bad in (math.nan, math.inf, -math.inf, 0.0, -3.0, None):
        c.set_rank_anchor(bad)
        assert c.rank_anchor is None
    c.set_rank_anchor(50.0)
    for bad in (math.nan, math.inf, None):
        c.set_rank_anchor(bad)
        assert c.rank_anchor == pytest.approx(50.0)


def test_rho_is_one_at_anchor_auto_capture():
    c = CapacityGateController()
    rec = c.update(dormancy=0.0, effective_rank=77.8)
    assert c.rank_anchor == pytest.approx(77.8)
    assert rec["rho"] == pytest.approx(1.0)
    assert rec["rank_anchor"] == pytest.approx(77.8)
    # A fresh anchor never trips the gate by itself.
    assert rec["gate_on"] is False


def test_rho_is_one_at_anchor_explicit():
    c = CapacityGateController()
    c.set_rank_anchor(40.8)
    rec = c.update(dormancy=0.0, effective_rank=40.8)
    assert rec["rho"] == pytest.approx(1.0)
    assert rec["gate_on"] is False


# ---------------------------------------------------------------------------
# Threshold crossing + hysteresis on rho
# ---------------------------------------------------------------------------

def test_gate_off_when_healthy():
    c = CapacityGateController()
    c.set_rank_anchor(130.0)
    rec = c.update(dormancy=0.0, effective_rank=130.0, numerical_rank=200.0)
    assert rec["gate_on"] is False
    assert rec["rho"] == pytest.approx(1.0)
    assert rec["intervention_strength"] == pytest.approx(0.0)


def test_gate_on_when_dormancy_high():
    c = CapacityGateController()
    c.set_rank_anchor(130.0)
    rec = c.update(dormancy=0.5, effective_rank=130.0)
    assert rec["gate_on"] is True
    assert rec["intervention_strength"] == pytest.approx(c.config.active_severity)


def test_gate_on_when_rho_low():
    c = CapacityGateController()
    c.set_rank_anchor(100.0)
    rec = c.update(dormancy=0.0, effective_rank=60.0)  # rho=0.6 <= 0.70
    assert rec["rho"] == pytest.approx(0.6)
    assert rec["gate_on"] is True


def test_gate_exact_rho_threshold_trips():
    c = CapacityGateController()
    c.set_rank_anchor(100.0)
    assert c.update(dormancy=0.0, effective_rank=70.0)["gate_on"] is True
    c2 = CapacityGateController()
    c2.set_rank_anchor(100.0)
    assert c2.update(dormancy=0.15, effective_rank=100.0)["gate_on"] is True


def test_hysteresis_rho_band():
    c = CapacityGateController()  # rho_on=0.70, rho_off=0.85
    c.set_rank_anchor(100.0)
    c.update(dormancy=0.0, effective_rank=60.0)
    assert c.gate_on is True
    # Inside the band: stays ON.
    assert c.update(dormancy=0.0, effective_rank=80.0)["gate_on"] is True
    # Recovery at/above rho_off turns OFF.
    assert c.update(dormancy=0.0, effective_rank=90.0)["gate_on"] is False


def test_hysteresis_dormancy_band():
    c = CapacityGateController()  # dorm_on=0.15, dorm_off=0.05
    c.set_rank_anchor(100.0)
    c.update(dormancy=0.5, effective_rank=100.0)
    assert c.gate_on is True
    assert c.update(dormancy=0.10, effective_rank=100.0)["gate_on"] is True
    assert c.update(dormancy=0.01, effective_rank=100.0)["gate_on"] is False


def test_hysteresis_requires_both_channels_to_recover():
    c = CapacityGateController()
    c.set_rank_anchor(100.0)
    c.update(dormancy=0.5, effective_rank=50.0)
    assert c.update(dormancy=0.0, effective_rank=50.0)["gate_on"] is True
    c2 = CapacityGateController()
    c2.set_rank_anchor(100.0)
    c2.update(dormancy=0.5, effective_rank=50.0)
    assert c2.update(dormancy=0.5, effective_rank=100.0)["gate_on"] is True


def test_gate_or_trigger_and_off_semantics():
    # OR to turn ON (either channel suffices), AND to turn OFF (both needed).
    c = CapacityGateController()
    c.set_rank_anchor(100.0)
    c.update(dormancy=0.0, effective_rank=50.0)  # rank only
    assert c.gate_on is True
    c2 = CapacityGateController()
    c2.set_rank_anchor(100.0)
    c2.update(dormancy=0.9, effective_rank=100.0)  # dormancy only
    assert c2.gate_on is True


# ---------------------------------------------------------------------------
# Cross-environment absolute-rank invariance (synthetic)
# ---------------------------------------------------------------------------

def test_cross_env_absolute_rank_invariance():
    # Hopper-like (anchor 40) and cheetah-like (anchor 140) envs: identical
    # retention must give identical decisions despite 3.5x absolute scale.
    decisions = []
    for anchor in (40.0, 140.0):
        c = CapacityGateController()
        c.set_rank_anchor(anchor)
        seq = [c.update(dormancy=0.0, effective_rank=anchor * f)["gate_on"]
               for f in (1.0, 0.6, 0.8, 0.9)]
        decisions.append(seq)
    assert decisions[0] == decisions[1] == [False, True, True, False]


def test_absolute_values_alone_do_not_decide():
    # erank=60 trips with anchor 100 (rho 0.6) but not with anchor 60 (rho 1).
    c1 = CapacityGateController()
    c1.set_rank_anchor(100.0)
    assert c1.update(dormancy=0.0, effective_rank=60.0)["gate_on"] is True
    c2 = CapacityGateController()
    c2.set_rank_anchor(60.0)
    assert c2.update(dormancy=0.0, effective_rank=60.0)["gate_on"] is False


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

def test_gate_logs_thresholds_state_strength_frequency():
    c = CapacityGateController()
    c.set_rank_anchor(100.0)
    r1 = c.update(dormancy=0.0, effective_rank=100.0)
    r2 = c.update(dormancy=0.9, effective_rank=100.0)
    for key in ("dormancy", "effective_rank", "rank_anchor", "rho",
                "dorm_on", "dorm_off", "rho_on", "rho_off",
                "gate_on", "gate_prev", "intervention_strength",
                "intervention_active", "intervention_count",
                "intervention_frequency", "timestamp"):
        assert key in r2, key
    assert r1["rho"] == pytest.approx(1.0)
    assert r1["intervention_frequency"] == pytest.approx(0.0)
    assert r2["intervention_frequency"] == pytest.approx(0.5)
    assert r2["intervention_count"] == 1
    assert r2["timestamp"] >= r1["timestamp"]
    d = c.diagnostics()
    assert d["n_evaluations"] == 2 and d["n_interventions"] == 1
    assert d["rank_anchor"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_probe_deterministic():
    obs = np.random.RandomState(0).randn(2000, 8).astype(np.float32)
    p1 = build_fixed_probe(obs, n=PROBE_SIZE, seed=123)
    p2 = build_fixed_probe(obs, n=PROBE_SIZE, seed=123)
    p3 = build_fixed_probe(obs, n=PROBE_SIZE, seed=999)
    assert p1.shape == (PROBE_SIZE, 8)
    assert torch.equal(p1, p2)
    assert not torch.equal(p1, p3)


def test_metrics_deterministic_on_fixed_probe():
    agent = _agent(seed=7)
    probe = _probe(seed=3)
    m1 = compute_capacity_metrics(agent.policy, probe)
    m2 = compute_capacity_metrics(agent.policy, probe)
    assert m1["dormancy"] == m2["dormancy"]
    assert m1["effective_rank"] == m2["effective_rank"]
    assert m1["numerical_rank"] == m2["numerical_rank"]
    assert m1["dormancy_per_layer"] == m2["dormancy_per_layer"]


def test_gate_deterministic_sequence():
    seq = [(0.0, 100.0), (0.5, 100.0), (0.10, 100.0), (0.0, 100.0), (0.0, 50.0)]
    runs = []
    for _ in range(2):
        c = CapacityGateController()
        c.set_rank_anchor(100.0)
        recs = [c.update(dormancy=d, effective_rank=r, timestamp=1000.0 + i)
                for i, (d, r) in enumerate(seq)]
        runs.append([(r["gate_on"], r["rho"]) for r in recs])
    assert runs[0] == runs[1] == [(False, 1.0), (True, 1.0), (True, 1.0),
                                  (False, 1.0), (True, 0.5)]


def test_update_from_policy_no_training_mode_leak():
    agent = _agent()
    agent.policy.train()
    probe = _probe()
    c = CapacityGateController()
    c.update_from_policy(agent.policy, probe, step=0)
    assert agent.policy.training is True
    assert c.rank_anchor is not None and math.isfinite(c.rank_anchor)


def test_capture_anchor_from_policy_is_idempotent():
    agent = _agent()
    probe = _probe()
    c = CapacityGateController()
    a1 = c.capture_anchor_from_policy(agent.policy, probe)
    a2 = c.capture_anchor_from_policy(agent.policy, probe)
    assert a1 == a2 and math.isfinite(a1)


# ---------------------------------------------------------------------------
# NaN / degenerate cases
# ---------------------------------------------------------------------------

def test_nan_rank_never_trips_and_holds_state():
    c = CapacityGateController()
    c.set_rank_anchor(100.0)
    r = c.update(dormancy=0.0, effective_rank=math.nan)
    assert r["gate_on"] is False
    assert math.isnan(r["rho"])
    assert math.isfinite(r["intervention_strength"])
    # Trip via dormancy, then NaN rank holds ON (no false recovery).
    c.update(dormancy=0.9, effective_rank=100.0)
    assert c.gate_on is True
    r = c.update(dormancy=math.nan, effective_rank=math.nan)
    assert r["gate_on"] is True
    assert math.isnan(r["rho"])


def test_nan_before_anchor_leaves_anchor_unset():
    c = CapacityGateController()
    r = c.update(dormancy=0.0, effective_rank=math.nan)
    assert c.rank_anchor is None
    assert math.isnan(r["rho"])
    assert r["gate_on"] is False
    # Later finite observation anchors normally with rho=1.0.
    r = c.update(dormancy=0.0, effective_rank=60.0)
    assert c.rank_anchor == pytest.approx(60.0)
    assert r["rho"] == pytest.approx(1.0)
    assert r["gate_on"] is False


def test_degenerate_inputs_nan_not_crash():
    assert math.isnan(feature_effective_rank(None))
    assert math.isnan(feature_numerical_rank(None))
    assert math.isnan(feature_effective_rank(torch.empty((0, 8))))
    assert math.isnan(feature_effective_rank(torch.zeros((16, 8))))
    assert math.isnan(feature_effective_rank(torch.full((16, 8), math.inf)))
    assert math.isnan(feature_numerical_rank(torch.zeros((16, 8))))
    agent = _agent()
    d, per = true_activation_dormancy(agent.policy, None)
    assert math.isnan(d) and per == {}
    d, per = true_activation_dormancy(agent.policy, torch.empty((0, 8)))
    assert math.isnan(d) and per == {}
    m = compute_capacity_metrics(agent.policy, None)
    assert math.isnan(m["dormancy"]) and math.isnan(m["effective_rank"])


def test_config_validation():
    with pytest.raises(ValueError):
        CapacityGateConfig(dorm_on=0.05, dorm_off=0.10)  # on < off
    with pytest.raises(ValueError):
        CapacityGateConfig(rho_on=0.85, rho_off=0.70)  # on > off
    with pytest.raises(ValueError):
        CapacityGateConfig(rho_on=0.0)  # non-positive
    with pytest.raises(ValueError):
        CapacityGateConfig(active_severity=1.5)


def test_schema_kind():
    assert CAPACITY_GATE_SCHEMA_VERSION == 2
    assert CAPACITY_GATE_KIND == "capacity-gate-v1"
    c = CapacityGateController()
    c.set_rank_anchor(100.0)
    rec = c.update(dormancy=0.0, effective_rank=100.0)
    assert rec["gate_kind"] == "capacity-gate-v1"
    assert rec["schema_version"] == 2
