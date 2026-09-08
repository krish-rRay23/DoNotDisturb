"""M8 unit tests: distribution shifts for offline→online RL.

Shifts are applied to the online phase only; offline D4RL data is unchanged.
"""

import math
import numpy as np
import pytest
import torch

from adaptive_plasticity.distribution_shifts import (
    DistributionShiftOrchestrator,
    apply_observation_noise,
    apply_reward_scale,
    ShiftType,
)


def test_shift_orchestrator_init():
    """Shift orchestrator validates type, step, and severity."""
    with pytest.raises(ValueError):
        DistributionShiftOrchestrator(shift_type="unknown", shift_step=0, severity=0.1)
    with pytest.raises(ValueError):
        DistributionShiftOrchestrator(shift_type="obs_noise", shift_step=-1, severity=0.1)
    with pytest.raises(ValueError):
        DistributionShiftOrchestrator(shift_type="obs_noise", shift_step=0, severity=-0.1)


def test_shift_orchestrator_is_shifted():
    """is_shifted fires at shift_step and stays on."""
    orch = DistributionShiftOrchestrator(shift_type="obs_noise", shift_step=5, severity=0.1)
    assert not orch.is_shifted(4)
    assert orch.is_shifted(5)
    assert orch.is_shifted(6)


def test_shift_orchestrator_is_shifted_no_shift():
    """is_shifted returns False when shift_step=0."""
    orch = DistributionShiftOrchestrator(shift_type="obs_noise", shift_step=0, severity=0.1)
    assert not orch.is_shifted(0)
    assert not orch.is_shifted(1)


def test_apply_observation_noise():
    """apply_observation_noise adds bounded Gaussian noise."""
    obs = np.array([1.0, 2.0, 3.0, 4.0])
    result = apply_observation_noise(obs, severity=0.1, max_magnitude=1.0)
    assert result.shape == obs.shape
    # clipped within [-1.1, 1.1] per dimension
    assert np.all(result >= -1.1) and np.all(result <= 1.1)
    # deterministic with same seed
    rng1 = np.random.RandomState(42)
    rng2 = np.random.RandomState(42)
    r1 = apply_observation_noise(obs, severity=0.1, max_magnitude=1.0, rng=rng1)
    r2 = apply_observation_noise(obs, severity=0.1, max_magnitude=1.0, rng=rng2)
    assert np.allclose(r1, r2)


def test_apply_observation_noise_no_noise():
    """zero severity gives back the original observation."""
    obs = np.array([1.0, 2.0, 3.0])
    result = apply_observation_noise(obs, severity=0.0)
    assert np.allclose(result, obs)


def test_apply_reward_scale():
    """apply_reward_scale scales and clamps rewards."""
    reward = 5.0
    # amplify
    r = apply_reward_scale(reward, severity=2.0)
    assert r == 10.0
    # attenuate
    r = apply_reward_scale(reward, severity=0.5)
    assert r == 2.5
    # no scaling
    r = apply_reward_scale(reward, severity=0.0)
    assert r == 5.0
    # clamped
    r = apply_reward_scale(-100.0, severity=3.0, min_reward=-50.0, max_reward=50.0)
    assert r == -50.0


def test_distribution_shift_orchestrator_snapshot():
    """snapshot returns the shift parameters."""
    orch = DistributionShiftOrchestrator(shift_type="reward_scale", shift_step=1000, severity=0.3)
    snap = orch.snapshot()
    assert snap["shift_type"] == "reward_scale"
    assert snap["shift_step"] == 1000
    assert snap["severity"] == 0.3