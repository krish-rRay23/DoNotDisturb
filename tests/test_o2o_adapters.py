"""External offline-to-online learners stay unavailable; the IQL adapter runs."""

from __future__ import annotations

import torch

from adaptive_plasticity.capacity_gate import CapacityGateConfig
from adaptive_plasticity.iql import IQLAgent
from adaptive_plasticity.o2o_adapters import (
    EXTERNAL_LEARNERS,
    CapacityGateSession,
    ExternalLearnerUnavailable,
    IQLLearnerAdapter,
    require_external_learner,
)


def test_external_learners_are_not_vendored():
    assert set(EXTERNAL_LEARNERS) == {"cal_ql", "pex", "ro2o"}
    for key, spec in EXTERNAL_LEARNERS.items():
        assert spec.status == "not_vendored"
        assert spec.code.startswith("https://github.com/")
        assert spec.arxiv.startswith("https://arxiv.org/")
        try:
            require_external_learner(key)
        except ExternalLearnerUnavailable as exc:
            message = str(exc)
        else:
            raise AssertionError(f"{key} should be unavailable")
        assert spec.method in message
        assert spec.code in message


def test_iql_session_evaluates_without_applying_when_the_gate_is_closed():
    torch.manual_seed(0)
    agent = IQLAgent(obs_dim=6, act_dim=2, device="cpu")
    before = agent.policy.backbone.network[0].weight.detach().clone()
    session = CapacityGateSession(
        IQLLearnerAdapter(agent),
        config=CapacityGateConfig(dorm_on=1.0, dorm_off=0.5, rho_on=0.01, rho_off=0.85, cooldown_steps=3000),
        operator="redo",
    )
    probe = torch.randn(32, 6)
    record = session.maybe_apply(probe, step=1000)
    assert record["learner"] == "iql"
    assert record["gate_operator"] == "redo"
    assert record["should_intervene"] is False
    assert record["applied"] is False
    assert torch.equal(agent.policy.backbone.network[0].weight, before)
