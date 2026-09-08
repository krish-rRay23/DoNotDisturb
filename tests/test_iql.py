"""M4 unit tests: expectile loss, shapes, action bounds, seeding, config guards.

All fixtures are tiny synthetic tensors. No real datasets, no network.
"""

import numpy as np
import pytest
import torch
import yaml

from adaptive_plasticity.iql import (
    Critic,
    IQLAgent,
    TanhGaussianPolicy,
    ValueCritic,
    advantage_weights,
    expectile_loss,
    q_target_value,
    soft_update,
)
from adaptive_plasticity.reproducibility import seed_everything
from adaptive_plasticity.train_iql import build_transitions, load_training_config

OBS_DIM = 4
ACT_DIM = 2
BATCH = 16


def make_batch(count=BATCH):
    rng = np.random.default_rng(0)
    return {
        "observations": torch.as_tensor(rng.normal(size=(count, OBS_DIM)), dtype=torch.float32),
        "actions": torch.as_tensor(rng.uniform(-1, 1, size=(count, ACT_DIM)), dtype=torch.float32),
        "rewards": torch.as_tensor(rng.normal(size=(count,)), dtype=torch.float32),
        "next_observations": torch.as_tensor(rng.normal(size=(count, OBS_DIM)), dtype=torch.float32),
        "dones": torch.zeros(count, dtype=torch.float32),
    }


def test_expectile_loss_matches_hand_computation():
    diff = torch.tensor([2.0, -1.0])
    assert expectile_loss(diff, 0.7).item() == pytest.approx((0.7 * 4.0 + 0.3 * 1.0) / 2.0)
    assert expectile_loss(diff, 0.5).item() == pytest.approx(0.5 * (4.0 + 1.0) / 2.0)
    assert expectile_loss(torch.zeros(5), 0.9).item() == pytest.approx(0.0)
    with pytest.raises(ValueError):
        expectile_loss(diff, 1.5)


def test_network_output_shapes():
    seed_everything(0)
    policy = TanhGaussianPolicy(OBS_DIM, ACT_DIM, hidden_dim=32, num_layers=1)
    critic = Critic(OBS_DIM, ACT_DIM, hidden_dim=32, num_layers=1)
    value = ValueCritic(OBS_DIM, hidden_dim=32, num_layers=1)
    obs = torch.randn(BATCH, OBS_DIM)
    act = torch.rand(BATCH, ACT_DIM) * 2 - 1
    mean, std = policy(obs)
    assert mean.shape == (BATCH, ACT_DIM) and std.shape == (BATCH, ACT_DIM)
    sampled, log_prob = policy.sample(obs)
    assert sampled.shape == (BATCH, ACT_DIM) and log_prob.shape == (BATCH,)
    assert torch.isfinite(log_prob).all()
    q1, q2 = critic(obs, act)
    assert q1.shape == (BATCH,) and q2.shape == (BATCH,)
    assert value(obs).shape == (BATCH,)


def test_actions_respect_bounds():
    seed_everything(1)
    policy = TanhGaussianPolicy(OBS_DIM, ACT_DIM)
    obs = torch.randn(512, OBS_DIM)
    sampled, _ = policy.sample(obs)
    assert sampled.min().item() >= -1.0 and sampled.max().item() <= 1.0
    deterministic = policy.act(obs, deterministic=True)
    assert deterministic.min().item() >= -1.0 and deterministic.max().item() <= 1.0
    stochastic = policy.act(obs, deterministic=False)
    assert stochastic.min().item() >= -1.0 and stochastic.max().item() <= 1.0


def test_update_returns_losses_and_moves_target():
    seed_everything(2)
    agent = IQLAgent(OBS_DIM, ACT_DIM, hidden_dim=32, num_layers=1, device="cpu")
    before = [p.clone() for p in agent.target_critic.parameters()]
    losses = agent.update(make_batch())
    assert set(losses) == {"v_loss", "q_loss", "policy_loss", "advantage_mean", "weight_mean"}
    assert all(np.isfinite(v) for v in losses.values())
    after = list(agent.target_critic.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_identical_seeds_give_identical_agents_and_updates():
    seed_everything(123)
    first = IQLAgent(OBS_DIM, ACT_DIM, hidden_dim=32, num_layers=1, device="cpu")
    seed_everything(123)
    second = IQLAgent(OBS_DIM, ACT_DIM, hidden_dim=32, num_layers=1, device="cpu")
    for p1, p2 in zip(first.policy.parameters(), second.policy.parameters()):
        assert torch.equal(p1, p2)
    batch = make_batch()
    seed_everything(7)
    losses_first = first.update(batch)
    seed_everything(7)
    losses_second = second.update(batch)
    assert losses_first == pytest.approx(losses_second)


def test_updates_stay_finite_under_large_advantages():
    """Advantage weights must not overflow to inf/NaN (exp-then-clamp)."""
    seed_everything(11)
    agent = IQLAgent(OBS_DIM, ACT_DIM, hidden_dim=32, num_layers=1, device="cpu")
    batch = make_batch()
    batch["rewards"] = batch["rewards"] * 1000.0
    for _ in range(5):
        losses = agent.update(batch)
    assert all(np.isfinite(v) for v in losses.values()), losses


def test_soft_update_interpolates():
    seed_everything(3)
    agent = IQLAgent(OBS_DIM, ACT_DIM, hidden_dim=16, num_layers=1, device="cpu")
    with torch.no_grad():
        for param in agent.critic.parameters():
            param.add_(1.0)
    target_before = [p.clone() for p in agent.target_critic.parameters()]
    soft_update(agent.target_critic, agent.critic, tau=0.1)
    for before_param, after_param, source_param in zip(
        target_before, agent.target_critic.parameters(), agent.critic.parameters()
    ):
        assert torch.allclose(after_param, 0.9 * before_param + 0.1 * source_param)


def test_build_transitions_shifts_and_marks_dones():
    data = {
        "observations": np.arange(12, dtype=np.float64).reshape(6, 2),
        "actions": np.zeros((6, 1)),
        "rewards": np.ones(6),
        "terminals": np.array([0, 0, 1, 0, 0, 0], dtype=bool),
        "timeouts": np.zeros(6, dtype=bool),
    }
    transitions = build_transitions(data)
    # Plain shift everywhere except the terminal index (see leak test below).
    assert np.array_equal(transitions["next_observations"][0], data["observations"][1])
    assert np.array_equal(transitions["next_observations"][1], data["observations"][2])
    assert np.array_equal(transitions["dones"], [0, 0, 1, 0, 0, 0])
    assert transitions["observations"].dtype == np.float32


def test_build_transitions_has_no_cross_episode_leak_at_terminals():
    """Terminal transitions must not point at the next episode's start."""
    data = {
        "observations": np.arange(12, dtype=np.float64).reshape(6, 2),
        "actions": np.zeros((6, 1)),
        "rewards": np.ones(6),
        "terminals": np.array([0, 0, 1, 0, 0, 0], dtype=bool),
        "timeouts": np.array([0, 0, 0, 0, 1, 0], dtype=bool),
    }
    transitions = build_transitions(data)
    # Terminal index: successor is the observation itself (masked by done=1).
    assert np.array_equal(transitions["next_observations"][2], data["observations"][2])
    assert transitions["dones"][2] == 1.0
    # Ordinary shift elsewhere, including the documented timeout convention.
    assert np.array_equal(transitions["next_observations"][1], data["observations"][2])
    assert np.array_equal(transitions["next_observations"][4], data["observations"][5])
    assert transitions["dones"][4] == 0.0  # timeouts bootstrap per D4RL/CORL
    assert np.array_equal(transitions["next_observations"][5], data["observations"][5])


def test_q_target_terminal_vs_timeout_bootstrapping():
    """Terminals must ignore next values; non-terminals must bootstrap them."""
    rewards = torch.tensor([1.0, 2.0])
    dones = torch.tensor([1.0, 0.0])
    next_values = torch.tensor([1e6, 10.0])  # huge value exposes any terminal leak
    targets = q_target_value(rewards, dones, next_values, discount=0.99)
    assert targets[0].item() == pytest.approx(1.0)
    assert targets[1].item() == pytest.approx(2.0 + 0.99 * 10.0)


def test_advantage_weights_safe_bounded_and_monotonic():
    """Log-space clamp: identical to exp-then-clamp, never overflows."""
    advantage = torch.tensor([-1000.0, -10.0, 0.0, 1.0, 29.5, 1000.0])
    weights = advantage_weights(advantage, temperature=3.0, max_advantage=100.0)
    assert torch.isfinite(weights).all()
    assert (weights <= 100.0).all()
    assert weights[2].item() == pytest.approx(1.0)  # exp(0)
    assert weights[5].item() == pytest.approx(100.0)  # capped, not inf
    assert weights[0].item() == pytest.approx(0.0, abs=1e-20)
    assert bool((weights[1:] >= weights[:-1]).all())  # monotonic
    reference = (3.0 * advantage).exp().clamp(max=100.0)
    finite = torch.isfinite(reference)
    assert torch.allclose(weights[finite], reference[finite])
    with pytest.raises(ValueError):
        advantage_weights(advantage, temperature=3.0, max_advantage=0.0)


def test_v_update_regresses_to_target_critic_not_online_critic():
    """Perturbing the target critic must move v_loss; online-only must not."""
    batch = make_batch()
    seed_everything(21)
    base = IQLAgent(OBS_DIM, ACT_DIM, hidden_dim=32, num_layers=1, device="cpu")
    seed_everything(21)
    target_perturbed = IQLAgent(OBS_DIM, ACT_DIM, hidden_dim=32, num_layers=1, device="cpu")
    seed_everything(21)
    online_perturbed = IQLAgent(OBS_DIM, ACT_DIM, hidden_dim=32, num_layers=1, device="cpu")

    with torch.no_grad():
        for param in target_perturbed.target_critic.parameters():
            param.add_(5.0)
        for param in online_perturbed.critic.parameters():
            param.add_(5.0)

    base_losses = base.update(batch)
    target_losses = target_perturbed.update(batch)
    online_losses = online_perturbed.update(batch)

    assert target_losses["v_loss"] != pytest.approx(base_losses["v_loss"])
    assert online_losses["v_loss"] == pytest.approx(base_losses["v_loss"])
    assert online_losses["q_loss"] != pytest.approx(base_losses["q_loss"])


def test_config_guard_rejects_normalization_and_non_iql(tmp_path):
    base = {
        "seed": 0,
        "device": "cpu",
        "dataset": {"id": "halfcheetah-medium-v2"},
        "algorithm": {"name": "iql", "parameters": {}},
        "training": {"steps": 10},
        "normalization": {"observations": False, "rewards": False},
    }
    path = tmp_path / "iql.yaml"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    assert load_training_config(path)["algorithm"]["name"] == "iql"

    enabled = dict(base)
    enabled["normalization"] = {"observations": True, "rewards": False}
    path.write_text(yaml.safe_dump(enabled), encoding="utf-8")
    with pytest.raises(ValueError, match="normalization"):
        load_training_config(path)

    renamed = dict(base)
    renamed["algorithm"] = {"name": "cql", "parameters": {}}
    path.write_text(yaml.safe_dump(renamed), encoding="utf-8")
    with pytest.raises(ValueError, match="IQL"):
        load_training_config(path)
