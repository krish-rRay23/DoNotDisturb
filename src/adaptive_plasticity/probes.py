"""Probe distributions for capacity telemetry.

The published CapacityGate runs monitor a uniform sample of the online replay
buffer (``online_buffer``), falling back to a frozen offline probe until that
buffer holds a full probe. Two additional sources are provided for the
probe-distribution ablation and are not part of the reported manuscript:

* ``offline_fixed``: the same frozen offline batch on every evaluation.
* ``recent_window``: a sample from the most recent online transitions.

The rank anchor captured at the start of online adaptation stays on the
frozen offline probe. These helpers only choose the batch used for later
monitoring, so a change of source is not confounded with a moving anchor.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

PROBE_SOURCES: tuple[str, ...] = ("offline_fixed", "online_buffer", "recent_window")

__all__ = [
    "PROBE_SOURCES",
    "recent_observation_indices",
    "sample_capacity_probe",
]


def recent_observation_indices(
    count: int,
    write_idx: int,
    max_size: int,
    window: int,
) -> np.ndarray:
    """Indices of the most recent ``window`` transitions in a ring buffer.

    ``count`` is the number of valid transitions, capped at ``max_size``.
    ``write_idx`` is the next write position. When the buffer is not full,
    valid rows occupy ``[0, count)``.
    """
    if count <= 0 or window <= 0 or max_size <= 0:
        return np.empty((0,), dtype=np.int64)
    n_valid = min(int(count), int(max_size))
    window = min(int(window), n_valid)
    if n_valid < int(max_size):
        start = n_valid - window
        return np.arange(start, n_valid, dtype=np.int64)
    newest = (int(write_idx) - 1) % int(max_size)
    start = (newest - window + 1) % int(max_size)
    if start <= newest:
        return np.arange(start, newest + 1, dtype=np.int64)
    return np.concatenate(
        [
            np.arange(start, int(max_size), dtype=np.int64),
            np.arange(0, newest + 1, dtype=np.int64),
        ]
    )


def _as_probe_tensor(observations: np.ndarray, device: torch.device | str | None) -> torch.Tensor:
    tensor = torch.as_tensor(np.asarray(observations, dtype=np.float32))
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor


def sample_capacity_probe(
    *,
    source: str,
    offline_probe: torch.Tensor,
    online_observations: np.ndarray | None,
    online_count: int,
    write_idx: int,
    max_size: int,
    probe_size: int,
    recent_window: int,
    rng: np.random.RandomState,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Draw the monitoring probe for one capacity evaluation.

    ``online_buffer`` matches the published runner: one
    ``RandomState.choice(count, size=probe_size, replace=False)`` call when
    the buffer holds at least ``probe_size`` rows, otherwise the frozen
    offline probe and no RNG draw. ``offline_fixed`` never touches ``rng``.
    ``recent_window`` samples the recent slice with replacement only when
    that slice is shorter than ``probe_size``.
    """
    if source not in PROBE_SOURCES:
        raise ValueError(f"Unknown probe source {source!r}. Expected one of {PROBE_SOURCES}.")
    if not isinstance(rng, np.random.RandomState):
        raise TypeError("rng must be numpy.random.RandomState.")
    if int(probe_size) < 1:
        raise ValueError(f"probe_size must be positive, got {probe_size}.")
    if int(recent_window) < 1:
        raise ValueError(f"recent_window must be positive, got {recent_window}.")

    meta: dict[str, Any] = {
        "probe_source": source,
        "probe_size": int(probe_size),
        "recent_window": int(recent_window),
        "fallback": None,
        "n_pool": 0,
        "replace": False,
    }
    if source == "offline_fixed":
        meta["n_pool"] = int(offline_probe.shape[0]) if offline_probe.ndim >= 1 else 0
        return offline_probe, meta

    count = int(online_count)
    if online_observations is None:
        raise ValueError(f"probe source {source!r} requires online observations.")

    if source == "online_buffer":
        meta["n_pool"] = count
        if count >= int(probe_size):
            indices = rng.choice(count, size=int(probe_size), replace=False)
            return _as_probe_tensor(online_observations[indices], device), meta
        meta["fallback"] = "offline_fixed"
        return offline_probe, meta

    pool = recent_observation_indices(count, int(write_idx), int(max_size), int(recent_window))
    meta["n_pool"] = int(pool.size)
    if pool.size == 0:
        meta["fallback"] = "offline_fixed"
        return offline_probe, meta
    replace = pool.size < int(probe_size)
    meta["replace"] = bool(replace)
    chosen = rng.choice(pool, size=int(probe_size), replace=replace)
    return _as_probe_tensor(online_observations[chosen], device), meta
