"""M7 unit tests: ReDo plasticity-preservation baseline.

Tests use tiny synthetic models; no real IQL training, no environment.
"""

import math
import numpy as np
import pytest
import torch

from adaptive_plasticity.m7 import redo_regularizer
from adaptive_plasticity.iql import IQLAgent, TanhGaussianPolicy


def test_redo_regularizer_basic():
    """redo_regularizer returns a finite scalar tensor."""
    policy = TanhGaussianPolicy(obs_dim=4, act_dim=2)
    ref_policy = TanhGaussianPolicy(obs_dim=4, act_dim=2)
    # freeze ref policy params so they don't contribute to grads
    for p in ref_policy.parameters():
        p.requires_grad = False
    reg = redo_regularizer(policy, ref_policy, obs_dim=4)
    assert isinstance(reg, torch.Tensor)
    assert reg.dim() == 0  # scalar
    assert math.isfinite(reg.item())


def test_redo_regularizer_identical_policies():
    """Identical policies should give a (near-)zero regularizer."""
    policy = TanhGaussianPolicy(obs_dim=4, act_dim=2)
    ref_policy = TanhGaussianPolicy(obs_dim=4, act_dim=2)
    # copy weights so policies are identical
    policy.load_state_dict(ref_policy.state_dict())
    for p in ref_policy.parameters():
        p.requires_grad = False
    reg = redo_regularizer(policy, ref_policy, obs_dim=4)
    assert isinstance(reg, torch.Tensor)
    assert math.isfinite(reg.item())
    assert reg.item() < 1e-4


def test_redo_logger_interval():
    """ReDoLogger maybe_log returns dict at interval steps, None otherwise."""
    from adaptive_plasticity.m7 import ReDoLogger
    logger = ReDoLogger(log_interval=5)
    agent = IQLAgent(17, 6, device="cpu")
    # step 5 should produce a dict; step 4 should return None
    d5 = logger.maybe_log(5, agent)
    d4 = logger.maybe_log(4, agent)
    assert d5 is not None
    assert d4 is None
    assert logger.step == 5


def test_redo_logger_may_log_with_term():
    """ReDoLogger maybe_log accepts and stores the redo_term value."""
    from adaptive_plasticity.m7 import ReDoLogger
    logger = ReDoLogger(log_interval=1)
    agent = IQLAgent(17, 6, device="cpu")
    d1 = logger.maybe_log(1, agent, redo_term=0.023)
    assert d1 is not None
    assert d1["redo_regularizer"] == 0.023
    assert logger.step == 1


def test_redo_logger_no_log():
    """ReDoLogger maybe_log returns None when step is not on interval."""
    from adaptive_plasticity.m7 import ReDoLogger
    logger = ReDoLogger(log_interval=10)
    agent = IQLAgent(17, 6, device="cpu")
    for s in range(1, 12):
        result = logger.maybe_log(s, agent)
        expected_step = s if s % 10 == 0 else None
        if expected_step is None:
            assert result is None, f"step {s}: expected None, got {result}"
        else:
            assert result is not None, f"step {s}: expected dict, got None"
            assert result["step"] == expected_step


def test_redo_snapshot():
    """ReDoLogger snapshot returns a dict with history."""
    from adaptive_plasticity.m7 import ReDoLogger
    logger = ReDoLogger(log_interval=1)
    agent = IQLAgent(17, 6, device="cpu")
    logger.maybe_log(1, agent, redo_term=0.1)
    logger.maybe_log(2, agent, redo_term=0.05)
    snap = logger.snapshot()
    assert "redo_regularizer_history" in snap
    assert len(snap["redo_regularizer_history"]) == 2
    assert snap["redo_regularizer_history"][0] == 0.1
    assert snap["redo_regularizer_history"][1] == 0.05