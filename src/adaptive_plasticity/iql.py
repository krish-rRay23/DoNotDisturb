"""Standard Implicit Q-Learning (IQL) implementation (Kostrikov et al., 2021).

Contains the three function approximators (twin Q networks, V network,
tanh-squashed Gaussian policy), expectile regression for V, advantage-weighted
behavioral cloning for the policy, and soft target-Q stabilization. No
plasticity mechanism, no normalization, no research modifications.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from adaptive_plasticity.m7 import redo_regularizer

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class MLP(nn.Module):
    """Plain ReLU feed-forward network (no normalization layers)."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(previous, hidden_dim))
            layers.append(nn.ReLU())
            previous = hidden_dim
        layers.append(nn.Linear(previous, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class TanhGaussianPolicy(nn.Module):
    """Gaussian policy with tanh squashing, so actions lie in [-1, 1]."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256, num_layers: int = 2) -> None:
        super().__init__()
        self.backbone = MLP(obs_dim, hidden_dim, num_layers, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, act_dim)
        self.log_std_head = nn.Linear(hidden_dim, act_dim)
        self.act_dim = act_dim

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(observations)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std.exp()

    def sample(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample squashed actions and their corrected log-probabilities."""
        mean, std = self.forward(observations)
        normal = torch.distributions.Normal(mean, std)
        raw = normal.rsample()
        action = torch.tanh(raw)
        # Tanh change-of-variables correction (stable form).
        log_prob = normal.log_prob(raw).sum(dim=-1) - 2.0 * (np.log(2.0) - raw - nn.functional.softplus(-2.0 * raw)).sum(dim=-1)
        return action, log_prob

    def act(self, observations: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Return actions in [-1, 1]; evaluation uses the deterministic mean."""
        mean, std = self.forward(observations)
        if deterministic:
            return torch.tanh(mean)
        normal = torch.distributions.Normal(mean, std)
        return torch.tanh(normal.sample())

    def log_prob_of(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Log-probability of (already squashed) dataset actions via atanh."""
        mean, std = self.forward(observations)
        clipped = actions.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        raw = torch.atanh(clipped)
        normal = torch.distributions.Normal(mean, std)
        return normal.log_prob(raw).sum(dim=-1) - 2.0 * (np.log(2.0) - raw - nn.functional.softplus(-2.0 * raw)).sum(dim=-1)


class Critic(nn.Module):
    """Twin Q networks Q1, Q2 over concatenated (observation, action)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256, num_layers: int = 2) -> None:
        super().__init__()
        self.q1 = MLP(obs_dim + act_dim, hidden_dim, num_layers, 1)
        self.q2 = MLP(obs_dim + act_dim, hidden_dim, num_layers, 1)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat([observations, actions], dim=-1)
        return self.q1(inputs).squeeze(-1), self.q2(inputs).squeeze(-1)


class ValueCritic(nn.Module):
    """State value network V(s)."""

    def __init__(self, obs_dim: int, hidden_dim: int = 256, num_layers: int = 2) -> None:
        super().__init__()
        self.v = MLP(obs_dim, hidden_dim, num_layers, 1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.v(observations).squeeze(-1)


def expectile_loss(differences: torch.Tensor, expectile: float) -> torch.Tensor:
    """Asymmetric squared loss: |tau - 1(diff < 0)| * diff^2, averaged."""
    if not 0.0 < expectile < 1.0:
        raise ValueError(f"expectile must lie in (0, 1), got {expectile}")
    weight = torch.where(differences < 0.0, 1.0 - expectile, expectile)
    return (weight * differences.pow(2)).mean()


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """In-place Polyak averaging of target parameters toward source parameters."""
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(tau * source_param.data)


def q_target_value(
    rewards: torch.Tensor, dones: torch.Tensor, next_values: torch.Tensor, discount: float
) -> torch.Tensor:
    """One-step Q target ``r + (1 - done) * gamma * V(s')``.

    Terminals (``dones == 1``) stop bootstrapping. Timeouts bootstrap per the
    D4RL/CORL convention (see ``build_transitions``); this matches the trusted
    reference implementation exactly.
    """
    return rewards + (1.0 - dones) * discount * next_values


def advantage_weights(
    advantage: torch.Tensor, temperature: float, max_advantage: float = 100.0
) -> torch.Tensor:
    """Standard IQL weights ``min(exp(beta * A), max_advantage)``.

    Computed in log-space: ``exp(min(beta * A, log(max)))`` equals
    ``exp(beta * A).clamp(max)`` up to float rounding but can never overflow
    float32 before the cap is applied. A final clamp trims the last ulp so
    the bound holds exactly.
    """
    if max_advantage <= 0.0:
        raise ValueError(f"max_advantage must be positive, got {max_advantage}")
    capped = (temperature * advantage).clamp(max=math.log(max_advantage)).exp()
    return capped.clamp(max=max_advantage)


class IQLAgent:
    """Standard IQL agent: expectile V, MSE Q with target, AWR policy."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        *,
        hidden_dim: int = 256,
        num_layers: int = 2,
        expectile: float = 0.7,
        temperature: float = 3.0,
        discount: float = 0.99,
        target_tau: float = 0.005,
        max_advantage: float = 100.0,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.expectile = expectile
        self.temperature = temperature
        self.discount = discount
        self.target_tau = target_tau
        self.max_advantage = max_advantage

        self.policy = TanhGaussianPolicy(obs_dim, act_dim, hidden_dim, num_layers).to(self.device)
        self.critic = Critic(obs_dim, act_dim, hidden_dim, num_layers).to(self.device)
        self.target_critic = copy.deepcopy(self.critic)
        self.value = ValueCritic(obs_dim, hidden_dim, num_layers).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.policy.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=critic_lr)

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """One gradient step on V, Q, and the policy. Returns scalar losses.

        Follows the CORL/reference IQL order and semantics: the V update
        regresses toward the TARGET critic's min-Q via expectile regression;
        the Q update bootstraps V values computed before the V step; the
        policy update reuses the V-step advantage for weighting. The update is
        deterministic given parameters and batch (no sampling inside).
        """
        observations = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_observations = batch["next_observations"].to(self.device)
        dones = batch["dones"].to(self.device)

        with torch.no_grad():
            next_values = self.value(next_observations)
            target_q = torch.min(*self.target_critic(observations, actions))

        # V update: expectile regression of the target critic's min-Q.
        values = self.value(observations)
        advantage = target_q - values
        v_loss = expectile_loss(advantage, self.expectile)
        self.value_optimizer.zero_grad()
        v_loss.backward()
        self.value_optimizer.step()

        # Q update: MSE toward the one-step target (pre-update next values).
        with torch.no_grad():
            q_targets = q_target_value(rewards, dones, next_values, self.discount)
        q1, q2 = self.critic(observations, actions)
        q_loss = ((q1 - q_targets).pow(2) + (q2 - q_targets).pow(2)).mean() / 2.0
        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        soft_update(self.target_critic, self.critic, self.target_tau)

        # Policy update: advantage-weighted regression on dataset actions,
        # reusing the V-step advantage (reference semantics).
        with torch.no_grad():
            weight = advantage_weights(advantage.detach(), self.temperature, self.max_advantage)
        log_prob = self.policy.log_prob_of(observations, actions)
        policy_loss = -(weight * log_prob).mean()

        # ReDo representation-distribution regularizer (M7, optional):
        # matches current policy activations to a frozen reference distribution,
        # preventing representation drift during offline-to-online adaptation.
        # Only applied when the reference module has a 'backbone' attribute
        # (i.e., a TanhGaussianPolicy), to avoid breaking standard IQL agents
        # that use a Critic as the reference.
        redo_term = None
        if hasattr(self.target_critic, "backbone"):
            redo_term = redo_regularizer(self.policy, self.target_critic,
                                         obs_dim=getattr(self.policy, "act_dim", 4))
        if redo_term is not None:
            policy_loss = policy_loss + redo_term
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        return {
            "v_loss": float(v_loss.item()),
            "q_loss": float(q_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "advantage_mean": float(advantage.mean().item()),
            "weight_mean": float(weight.mean().item()),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "value": self.value.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.policy.load_state_dict(state["policy"])
        self.critic.load_state_dict(state["critic"])
        self.target_critic.load_state_dict(state["target_critic"])
        self.value.load_state_dict(state["value"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.value_optimizer.load_state_dict(state["value_optimizer"])
