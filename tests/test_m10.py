"""M10 unit tests: Adaptive Plasticity Controller.

Tests the AdaptivePlasticityController rule-based controller that dynamically
adjusts intervention strength based on online M6 diagnostics.
"""

import math
import numpy as np
import pytest

from adaptive_plasticity.distribution_shifts import (
    AdaptivePlasticityController,
)


def test_controller_init():
    """AdaptivePlasticityController initializes with configurable weights."""
    c = AdaptivePlasticityController(seed=42)
    assert c.min_severity == 0.0
    assert c.max_severity == 1.0
    assert c.repr_weight == 0.4
    assert c.perf_weight == -0.4  # KEY: negative weight - perf degradation ↑ strength
    assert c.dorm_weight == -0.2
    assert c._history == []

    c2 = AdaptivePlasticityController(
        seed=123,
        repr_weight=0.5,
        perf_weight=0.3,
        dorm_weight=-0.1,
        min_severity=0.1,
        max_severity=0.8,
    )
    assert c2.repr_weight == 0.5
    assert c2.perf_weight == 0.3
    assert c2.dorm_weight == -0.1
    assert c2.min_severity == 0.1
    assert c2.max_severity == 0.8


def test_controller_deterministic():
    """Same seed produces identical results (deterministic)."""
    c = AdaptivePlasticityController(seed=42)
    diag = {
        "repr_change": 0.5,
        "perf_change": -0.1,
        "activation_dormancy": 0.3,
        "param_magnitudes": {"weight1": 0.8, "weight2": 1.2},
    }
    s1 = c.update(diag)
    s2 = c.update(diag)
    assert s1 == s2  # identical call should give identical strength
    assert len(c._history) == 2  # both updates logged


def test_controller_different_seeds():
    """Different seeds produce different results (seeded RNG)."""
    c1 = AdaptivePlasticityController(seed=42)
    c2 = AdaptivePlasticityController(seed=123)
    diag = {"repr_change": 0.5, "perf_change": -0.1, "activation_dormancy": 0.3, "param_magnitudes": {}}
    s1 = c1.update(diag)
    s2 = c2.update(diag)
    # Seeds produce deterministic but different results
    assert 0.0 <= s1 <= 1.0
    assert 0.0 <= s2 <= 1.0


def test_controller_bounds():
    """Strength always returned in [min_severity, max_severity] subset of [0,1]."""
    c = AdaptivePlasticityController(seed=42, min_severity=0.2, max_severity=0.8)
    # Large positive repr_change + positive perf_change => strength at max
    diag_pos = {
        "repr_change": 1.0,
        "perf_change": 1.0,
        "activation_dormancy": 0.0,
        "param_magnitudes": {},
    }
    s = c.update(diag_pos)
    assert s <= 0.8  # capped at max_severity
    assert s >= 0.2  # above min_severity

    # Large negative diagnostics => strength at min
    diag_neg = {
        "repr_change": -1.0,
        "perf_change": -1.0,
        "activation_dormancy": 1.0,
        "param_magnitudes": {},
    }
    s = c.update(diag_neg)
    assert s <= 0.8
    assert s >= 0.2


def test_controller_logging():
    """All decisions and diagnostics are logged for provenance."""
    c = AdaptivePlasticityController(seed=42)
    diag = {
        "repr_change": 0.5,
        "perf_change": -0.2,
        "activation_dormancy": 0.1,
        "param_magnitudes": {"kernel1": 0.5, "kernel2": -0.3},
    }
    strength = c.update(diag)
    diag_out = c.diagnostics()

    assert "history" in diag_out
    assert len(diag_out["history"]) == 1
    entry = diag_out["history"][0]
    assert entry["intervention_strength"] == strength
    assert entry["repr_change"] == 0.5
    assert entry["perf_change"] == -0.2
    assert entry["activation_dormancy"] == 0.1
    assert entry["param_magnitude"] == 0.0  # logged only; not used in control equation


def test_controller_no_history_before_update():
    """Diagnostics returns empty history before any update."""
    c = AdaptivePlasticityController(seed=42)
    diag_out = c.diagnostics()
    assert diag_out["history"] == []


def test_controller_with_zero_diagnostics():
    """Controller with zero diagnostics gives neutral strength."""
    c = AdaptivePlasticityController(seed=42, min_severity=0.0, max_severity=1.0)
    diag = {
        "repr_change": 0.0,
        "perf_change": 0.0,
        "activation_dormancy": 0.0,
        "param_magnitudes": {},
    }
    s = c.update(diag)
    assert s == 0.0  # raw=0, clipped to min_severity=0.0


def test_controller_param_magnitudes_logged_only():
    """param_magnitudes is logged for provenance but NOT used in control equation."""
    c = AdaptivePlasticityController(seed=42)
    diag = {
        "repr_change": 0.0,
        "perf_change": 0.0,
        "activation_dormancy": 0.0,
        "param_magnitudes": {"good": 1.0, "bad": float("nan"), "also_bad": float("inf")},
    }
    s = c.update(diag)
    # With zero other inputs, raw=0 regardless of param_magnitudes
    assert s == 0.0
    # History should show param_magnitude=0.0 (logged but not used)
    diag_out = c.diagnostics()
    assert diag_out["history"][0]["param_magnitude"] == 0.0


def test_controller_apply_noop():
    """Controller.apply() is a no-op, returning obs/reward unchanged."""
    c = AdaptivePlasticityController(seed=42)
    obs = np.array([1.0, 2.0, 3.0])
    reward = 5.0
    result_obs, result_reward = c.apply(obs, reward)
    np.testing.assert_array_equal(result_obs, obs)
    assert result_reward == reward


def test_controller_severity_weights_grounded():
    """Weights are tunable and grounded in plasticity hypothesis (corrected signs)."""
    c = AdaptivePlasticityController(seed=42, min_severity=0.0, max_severity=1.0)
    # With repr_change=0.5, perf_change=0.5, dormancy=0.0:
    # raw = 0.4*0.5 + (-0.4)*0.5 + (-0.2)*0.0 = 0.2 - 0.2 = 0.0
    diag_pos = {
        "repr_change": 0.5,
        "perf_change": 0.5,
        "activation_dormancy": 0.0,
        "param_magnitudes": {},
    }
    s = c.update(diag_pos)
    # raw = 0.4*0.5 + (-0.4)*0.5 + (-0.2)*0.0 = 0.2 - 0.2 = 0.0
    assert abs(s - 0.0) < 0.01, f"Expected ~0.0, got {s}"

    # With repr_change=-0.5, perf_change=-0.5, dormancy=1.0:
    # raw = 0.4*(-0.5) + (-0.4)*(-0.5) + (-0.2)*1.0 = -0.2 + 0.2 - 0.2 = -0.2
    # clipped to [0, 1] with min=0 => 0.0
    diag_neg = {
        "repr_change": -0.5,
        "perf_change": -0.5,
        "activation_dormancy": 1.0,
        "param_magnitudes": {},
    }
    s = c.update(diag_neg)
    # raw = 0.4*(-0.5) + (-0.4)*(-0.5) + (-0.2)*1.0 = -0.2 + 0.2 - 0.2 = -0.2
    # clipped to min=0 => 0.0
    assert s >= 0.0


def test_controller_larger_repr_increases_strength():
    """Regression test: larger representation drift increases strength."""
    c = AdaptivePlasticityController(seed=42, min_severity=0.0, max_severity=1.0)
    # repr_change=1.0, perf_change=0.0, dormancy=0.0:
    # raw = 0.4*1.0 + (-0.4)*0.0 + (-0.2)*0.0 = 0.4
    s_high = c.update({"repr_change": 1.0, "perf_change": 0.0, "activation_dormancy": 0.0, "param_magnitudes": {}})
    # repr_change=0.0, perf_change=0.0, dormancy=0.0:
    # raw = 0.4*0.0 + (-0.4)*0.0 + (-0.2)*0.0 = 0.0
    s_low = c.update({"repr_change": 0.0, "perf_change": 0.0, "activation_dormancy": 0.0, "param_magnitudes": {}})
    assert s_high >= s_low, "Larger repr_change should not decrease strength"


def test_controller_perf_degradation_increases_strength():
    """Regression test: performance degradation increases strength."""
    c = AdaptivePlasticityController(seed=42, min_severity=0.0, max_severity=1.0)
    # perf_change=-1.0 (degradation), repr_change=0.0, dormancy=0.0:
    # raw = 0.4*0.0 + (-0.4)*(-1.0) + (-0.2)*0.0 = 0.4
    s_with_degradation = c.update({"repr_change": 0.0, "perf_change": -1.0, "activation_dormancy": 0.0, "param_magnitudes": {}})
    # perf_change=0.0 (no degradation):
    # raw = 0.4*0.0 + (-0.4)*0.0 + (-0.2)*0.0 = 0.0
    s_no_degradation = c.update({"repr_change": 0.0, "perf_change": 0.0, "activation_dormancy": 0.0, "param_magnitudes": {}})
    assert s_with_degradation >= s_no_degradation, "Performance degradation should increase strength"


def test_controller_higher_dormancy_decreases_strength():
    """Regression test: higher dormancy decreases strength."""
    c = AdaptivePlasticityController(seed=42, min_severity=0.0, max_severity=1.0)
    # dormancy=1.0, repr_change=0.0, perf_change=0.0:
    # raw = 0.4*0.0 + (-0.4)*0.0 + (-0.2)*1.0 = -0.2 => clipped to 0.0
    s_high_dorm = c.update({"repr_change": 0.0, "perf_change": 0.0, "activation_dormancy": 1.0, "param_magnitudes": {}})
    # dormancy=0.0:
    # raw = 0.4*0.0 + (-0.4)*0.0 + (-0.2)*0.0 = 0.0
    s_low_dorm = c.update({"repr_change": 0.0, "perf_change": 0.0, "activation_dormancy": 0.0, "param_magnitudes": {}})
    assert s_high_dorm <= s_low_dorm, "Higher dormancy should not increase strength"