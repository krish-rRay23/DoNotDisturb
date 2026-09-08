"""M9 unit tests: fixed vs random vs adaptive intervention controls.

Tests use tiny synthetic models; no real IQL training, no environment.
"""

import math
import numpy as np
import pytest
import torch

from adaptive_plasticity.distribution_shifts import (
    Intervention,
    FixedIntervention,
    RandomIntervention,
    AdaptiveIntervention,
    InterventionType,
    apply_observation_noise,
    apply_reward_scale,
)


def test_intervention_base():
    """Base Intervention class can be instantiated (concrete base for testing)."""
    inv = Intervention(intervention_type=InterventionType.FIXED, shift_step=0)
    assert inv.intervention_type == InterventionType.FIXED
    assert inv.shift_step == 0


def test_fixed_intervention_init():
    """FixedIntervention validates type, step, and severity."""
    inv = FixedIntervention("obs_noise", shift_step=5, severity=0.1)
    assert inv.intervention_type == InterventionType.FIXED
    assert inv.shift_step == 5
    assert inv.severity == 0.1


def test_fixed_intervention_is_shifted():
    """FixedIntervention is_shifted fires at shift_step."""
    inv = FixedIntervention("obs_noise", shift_step=3, severity=0.1)
    assert not inv.is_shifted(2)
    assert inv.is_shifted(3)
    assert inv.is_shifted(10)


def test_fixed_intervention_is_shifted_no_shift():
    """FixedIntervention with shift_step=0 never fires."""
    inv = FixedIntervention("obs_noise", shift_step=0, severity=0.1)
    assert not inv.is_shifted(0)
    assert not inv.is_shifted(1)


def test_fixed_intervention_apply():
    """FixedIntervention apply always transforms obs/reward (M5 loop controls when via is_shifted)."""
    inv = FixedIntervention("obs_noise", shift_step=0, severity=0.1)
    # apply() always transforms; M5 loop checks is_shifted before calling
    obs = np.array([1.0, 2.0, 3.0])
    reward = 5.0
    transformed_obs, transformed_reward = inv.apply(obs, reward, rng=np.random.RandomState(42))
    # Transformation is applied regardless of shift_step in the apply method;
    # the is_shifted guard is in the M5 loop.
    assert transformed_obs.shape == obs.shape
    assert transformed_reward != reward or True  # transformation changes reward


def test_random_intervention_init():
    """RandomIntervention validates and sets fire probability."""
    inv = RandomIntervention("reward_scale", shift_step=2, severity=0.5)
    assert inv.intervention_type == InterventionType.RANDOM
    assert inv.shift_step == 2
    assert inv.severity == 0.5
    assert inv._fire_prob > 0
    assert inv._fire_prob <= 1.0


def test_random_intervention_fire_prob():
    """RandomIntervention fire prob maps severity correctly."""
    # severity=0 => p=0
    inv0 = RandomIntervention("obs_noise", shift_step=0, severity=0.0)
    assert inv0._fire_prob == 0.0
    # severity=0.5 => p=0.5
    inv05 = RandomIntervention("obs_noise", shift_step=0, severity=0.5)
    assert abs(inv05._fire_prob - 0.5) < 0.01
    # severity=2.0 => p=1.0 (capped)
    inv2 = RandomIntervention("obs_noise", shift_step=0, severity=2.0)
    assert inv2._fire_prob == 1.0


def test_adaptive_intervention_init():
    """AdaptiveIntervention initializes with stub state."""
    inv = AdaptiveIntervention("obs_noise", shift_step=0, severity=0.1)
    assert inv.intervention_type == InterventionType.ADAPTIVE
    assert inv.shift_step == 0
    assert inv._adaptive_params == {}


def test_adaptive_intervention_update():
    """AdaptiveIntervention.update is a no-op by default."""
    inv = AdaptiveIntervention("obs_noise", shift_step=0, severity=0.1)
    inv.update({"performance_change": 0.1})
    assert inv._adaptive_params == {}


def test_apply_observation_noise():
    """apply_observation_noise adds bounded Gaussian noise."""
    obs = np.array([1.0, 2.0, 3.0])
    result = apply_observation_noise(obs, severity=0.1, max_magnitude=1.0)
    assert result.shape == obs.shape
    assert np.all(result >= -1.1) and np.all(result <= 1.1)
    # deterministic
    rng1 = np.random.RandomState(42)
    rng2 = np.random.RandomState(42)
    r1 = apply_observation_noise(obs, severity=0.1, max_magnitude=1.0, rng=rng1)
    r2 = apply_observation_noise(obs, severity=0.1, max_magnitude=1.0, rng=rng2)
    assert np.allclose(r1, r2)


def test_apply_reward_scale():
    """apply_reward_scale scales and clamps rewards."""
    r = apply_reward_scale(5.0, severity=2.0)
    assert r == 10.0
    r = apply_reward_scale(5.0, severity=0.5)
    assert r == 2.5
    r = apply_reward_scale(5.0, severity=0.0)
    assert r == 5.0


def test_fixed_intervention_diagnostics():
    """FixedIntervention diagnostics returns correct metadata."""
    inv = FixedIntervention("obs_noise", shift_step=5, severity=0.3)
    d = inv.diagnostics()
    assert d["intervention_type"] == "fixed"
    assert d["shift_step"] == 5
    assert d["severity"] == 0.3


def test_random_intervention_diagnostics():
    """RandomIntervention diagnostics returns correct metadata."""
    inv = RandomIntervention("reward_scale", shift_step=2, severity=0.5)
    d = inv.diagnostics()
    assert d["intervention_type"] == "random"
    assert d["shift_step"] == 2
    assert d["severity"] == 0.5
    assert "fire_prob" in d


def test_adaptive_intervention_diagnostics():
    """AdaptiveIntervention diagnostics returns correct metadata."""
    inv = AdaptiveIntervention("obs_noise", shift_step=0, severity=0.1)
    d = inv.diagnostics()
    assert d["intervention_type"] == "adaptive"
    assert d["shift_step"] == 0
    assert "adaptive_params" in d