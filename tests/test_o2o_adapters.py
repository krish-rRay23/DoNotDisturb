"""External offline-to-online learners stay unavailable; the IQL adapter runs."""

from __future__ import annotations

from pathlib import Path

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


def _mlp(in_dim: int, hidden: int, out_dim: int) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, out_dim),
    )


class MockLearner:
    """Stand-in with the modules a wrapper must expose. Not an external codebase."""

    name = "mock"

    def __init__(self) -> None:
        torch.manual_seed(0)
        self.policy = _mlp(4, 8, 2)
        self.critic = _mlp(6, 8, 1)
        self.offline_policy = _mlp(4, 8, 2)
        self.actor_optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

    def policy_module(self) -> torch.nn.Module:
        return self.policy

    def critic_module(self) -> torch.nn.Module:
        return self.critic


def test_default_hook_directory_is_packaged_beside_the_module():
    from adaptive_plasticity import o2o_adapters

    directory = o2o_adapters.external_hook_dir()
    assert directory == Path(o2o_adapters.__file__).resolve().parent / "o2o_hooks"
    assert directory.parent.name == "adaptive_plasticity"
    assert (directory / "cal_ql.yaml").is_file()


def test_external_hooks_stay_not_yet_run_and_refuse_a_session():
    from adaptive_plasticity.o2o_adapters import load_external_hooks, open_external_session

    hooks = load_external_hooks()
    assert set(hooks) == {"cal_ql", "pex", "ro2o"}
    assert hooks["pex"].frozen_components == ("offline_policy",)
    assert "offline_policy" not in hooks["pex"].editable_components
    for key, hook in hooks.items():
        assert hook.status == "not_yet_run"
        assert hook.rho_on == 0.70
        assert hook.dorm_on == 0.15
        try:
            open_external_session(key, MockLearner())
        except ExternalLearnerUnavailable as exc:
            assert hook.learner in str(exc).lower() or EXTERNAL_LEARNERS[key].method in str(exc)
        else:
            raise AssertionError(f"{key} should not open a session")


def test_mock_learner_edits_only_the_editable_module():
    from adaptive_plasticity.o2o_adapters import CapacityGateSession

    learner = MockLearner()
    policy_before = learner.policy[0].weight.detach().clone()
    critic_before = learner.critic[0].weight.detach().clone()
    frozen_before = learner.offline_policy[0].weight.detach().clone()
    session = CapacityGateSession(
        learner,
        config=CapacityGateConfig(
            dorm_on=0.0,
            dorm_off=0.0,
            rho_on=0.01,
            rho_off=0.5,
            cooldown_steps=0,
            active_severity=0.1,
        ),
        operator="shrink_perturb",
        editable_components=("policy",),
        frozen_components=("offline_policy",),
    )
    record = session.maybe_apply(torch.randn(32, 4), step=1000)
    assert record["learner"] == "mock"
    assert record["applied"] is True
    assert not torch.equal(learner.policy[0].weight, policy_before)
    assert torch.equal(learner.critic[0].weight, critic_before)
    assert torch.equal(learner.offline_policy[0].weight, frozen_before)


def test_overlapping_frozen_and_editable_components_are_rejected():
    from adaptive_plasticity.o2o_adapters import CapacityGateSession

    try:
        CapacityGateSession(
            MockLearner(),
            editable_components=("policy",),
            frozen_components=("policy",),
        )
    except ValueError as exc:
        assert "policy" in str(exc)
    else:
        raise AssertionError("overlap should be rejected")


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
