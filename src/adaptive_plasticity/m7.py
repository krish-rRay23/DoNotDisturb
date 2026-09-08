from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


def redo_regularizer(
    policy: nn.Module,
    reference_policy: nn.Module,
    epsilon: float = 1e-2,
    obs_dim: int | None = None,
    n_obs: int = 32,
) -> torch.Tensor:
    """ReDo representation distribution regularizer.

    Matches the activation distribution of *policy* to that of
    *reference_policy* by aligning means and variances of the
    backbone features.  This is a minimal, faithful baseline
    provenance-mapped to the original ReDo idea of constraining
    representation drift during offline-to-online adaptation.

    The regularizer is an ``L2``-penalty on the difference between
    normalized activation statistics (mean and standard deviation):

        mean_pol  = mean(backbone(policy.obs_1..n))
        var_pol   = var(backbone(policy.obs_1..n))
        mean_ref  = mean(backbone(reference_policy.obs_1..n))
        var_ref   = var(backbone(reference_policy.obs_1..n))

    ``regularizer = ||mean_pol - mean_ref||^2 + ||std_pol - std_ref||^2``

    Only parameters marked ``requires_grad`` in *policy* contribute
    to the gradient.

    Args:
        policy: current (adapting) policy module.
        reference_policy: frozen policy whose activation stats are
            the target distribution.
        epsilon: small constant for numerical stability.
        obs_dim: observation dimension; if ``None``, inferred from
            the policy's ``act_dim`` attribute or defaults to ``4``.
        n_obs: number of synthetic observations to use for
            statistics estimation.  Using more than one observation
            avoids the degenerate ``n_samples=1`` case where
            variance-based normalization collapses all activations
            to zero.

    Returns:
        Scalar tensor representing the ReDo regularization loss.
    """
    if obs_dim is None:
        obs_dim = getattr(policy, "act_dim", 4)

    policy_backbone = policy.backbone
    ref_backbone = reference_policy.backbone

    # build synthetic observations: rows i.i.d. ~ Uniform(-1, 1)
    # so that the backbone features have non-degenerate statistics.
    obs = torch.rand(n_obs, obs_dim, dtype=torch.float32) * 2 - 1

    with torch.no_grad():
        ref_features = ref_backbone(obs)

    policy_features = policy_backbone(obs)

    # align feature dimensions: take the minimum shared size
    min_dim = min(policy_features.shape[-1], ref_features.shape[-1])
    policy_features = policy_features[..., :min_dim]
    ref_features = ref_features[..., :min_dim]

    # compute population mean and std across the observation dimension
    n = policy_features.shape[0]  # = n_obs

    mean_pol = policy_features.mean(dim=0)
    std_pol = policy_features.std(dim=0, unbiased=False) + epsilon

    mean_ref = ref_features.mean(dim=0)
    std_ref = ref_features.std(dim=0, unbiased=False) + epsilon

    # L2 on mean difference + L2 on std difference
    mean_diff = (mean_pol - mean_ref).pow(2).sum()
    std_diff = (std_pol - std_ref).pow(2).sum()

    return mean_diff + std_diff


class ReDoLogger:
    """Lightweight logger that tracks ReDo regularization during training.

    Stores the most recent regularization value and a short history.
    Designed to be agnostic to the training loop; the caller adds
    ``redo_term`` to the policy loss and then calls ``maybe_log``.
    """

    def __init__(self, log_interval: int = 100) -> None:
        if log_interval <= 0:
            raise ValueError("log_interval must be a positive integer")
        self.log_interval = log_interval
        self.step = 0
        self._redo_history: list[float] = []
        self._predecessor_params: dict[str, torch.Tensor] | None = None

    def maybe_log(
        self,
        step: int,
        agent: "IQLAgent",
        redo_term: float | None = None,
    ) -> dict[str, Any] | None:
        if step % self.log_interval != 0:
            return None
        self.step = step
        if redo_term is not None:
            self._redo_history.append(float(redo_term))
        # keep only the most recent value for quick access
        recent = self._redo_history[-1:] if self._redo_history else [math.nan]
        return {
            "step": step,
            "redo_regularizer": recent[0],
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "redo_regularizer_history": self._redo_history,
        }