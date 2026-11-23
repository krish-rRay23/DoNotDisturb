"""Probe-distribution helpers. No environment and no training."""

from __future__ import annotations

import numpy as np
import torch

from adaptive_plasticity.probes import recent_observation_indices, sample_capacity_probe


def test_online_buffer_matches_historical_choice():
    observations = np.arange(500 * 3, dtype=np.float32).reshape(500, 3)
    offline = torch.full((8, 3), -1.0)
    historical = np.random.RandomState(7)
    expected_idx = historical.choice(400, size=256, replace=False)
    rng = np.random.RandomState(7)
    probe, meta = sample_capacity_probe(
        source="online_buffer",
        offline_probe=offline,
        online_observations=observations,
        online_count=400,
        write_idx=400,
        max_size=500,
        probe_size=256,
        recent_window=1024,
        rng=rng,
        device="cpu",
    )
    np.testing.assert_array_equal(probe.numpy(), observations[expected_idx])
    assert meta["fallback"] is None
    assert meta["n_pool"] == 400


def test_online_buffer_falls_back_without_consuming_rng():
    offline = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    rng = np.random.RandomState(1)
    state = rng.get_state()
    probe, meta = sample_capacity_probe(
        source="online_buffer",
        offline_probe=offline,
        online_observations=np.zeros((10, 2), dtype=np.float32),
        online_count=10,
        write_idx=10,
        max_size=100,
        probe_size=16,
        recent_window=32,
        rng=rng,
    )
    assert probe is offline
    assert meta["fallback"] == "offline_fixed"
    assert rng.get_state()[1][:5].tolist() == state[1][:5].tolist()


def test_offline_fixed_ignores_online_buffer():
    offline = torch.ones((4, 2))
    rng = np.random.RandomState(3)
    state = rng.get_state()
    probe, meta = sample_capacity_probe(
        source="offline_fixed",
        offline_probe=offline,
        online_observations=np.zeros((100, 2), dtype=np.float32),
        online_count=100,
        write_idx=100,
        max_size=100,
        probe_size=4,
        recent_window=8,
        rng=rng,
    )
    assert probe is offline
    assert meta["probe_source"] == "offline_fixed"
    assert rng.get_state()[1][:5].tolist() == state[1][:5].tolist()


def test_recent_window_stays_inside_the_slice():
    observations = np.arange(50, dtype=np.float32).reshape(50, 1)
    rng = np.random.RandomState(0)
    probe, meta = sample_capacity_probe(
        source="recent_window",
        offline_probe=torch.zeros((4, 1)),
        online_observations=observations,
        online_count=50,
        write_idx=50,
        max_size=80,
        probe_size=4,
        recent_window=6,
        rng=rng,
        device="cpu",
    )
    values = set(probe.numpy().reshape(-1).tolist())
    assert values <= {44.0, 45.0, 46.0, 47.0, 48.0, 49.0}
    assert meta["n_pool"] == 6
    assert meta["fallback"] is None


def test_recent_indices_wrap_when_the_ring_is_full():
    # max_size 5, write_idx 2, newest is index 1. Last 3 writes are 4, 0, 1.
    indices = recent_observation_indices(count=5, write_idx=2, max_size=5, window=3)
    np.testing.assert_array_equal(indices, np.array([4, 0, 1]))


def test_recent_window_falls_back_when_the_buffer_is_empty():
    offline = torch.ones((4, 2))
    rng = np.random.RandomState(5)
    state = rng.get_state()
    probe, meta = sample_capacity_probe(
        source="recent_window",
        offline_probe=offline,
        online_observations=np.zeros((10, 2), dtype=np.float32),
        online_count=0,
        write_idx=0,
        max_size=10,
        probe_size=4,
        recent_window=8,
        rng=rng,
    )
    assert probe is offline
    assert meta["fallback"] == "offline_fixed"
    assert meta["n_pool"] == 0
    assert rng.get_state()[1][:5].tolist() == state[1][:5].tolist()


def test_recent_window_samples_with_replacement_when_the_slice_is_short():
    observations = np.arange(20, dtype=np.float32).reshape(20, 1)
    probe, meta = sample_capacity_probe(
        source="recent_window",
        offline_probe=torch.zeros((4, 1)),
        online_observations=observations,
        online_count=20,
        write_idx=20,
        max_size=20,
        probe_size=8,
        recent_window=3,
        rng=np.random.RandomState(2),
        device="cpu",
    )
    assert probe.shape == (8, 1)
    assert meta["replace"] is True
    assert meta["n_pool"] == 3
    values = set(probe.numpy().reshape(-1).tolist())
    assert values <= {17.0, 18.0, 19.0}


def test_unknown_probe_source_is_rejected():
    try:
        sample_capacity_probe(
            source="latest_only",
            offline_probe=torch.zeros((2, 1)),
            online_observations=np.zeros((4, 1), dtype=np.float32),
            online_count=4,
            write_idx=4,
            max_size=4,
            probe_size=2,
            recent_window=2,
            rng=np.random.RandomState(0),
        )
    except ValueError as exc:
        assert "latest_only" in str(exc)
    else:
        raise AssertionError("unknown probe source should raise")


def test_recent_window_is_deterministic():
    observations = np.random.RandomState(0).randn(40, 3).astype(np.float32)
    kwargs = dict(
        source="recent_window",
        offline_probe=torch.zeros((8, 3)),
        online_observations=observations,
        online_count=40,
        write_idx=40,
        max_size=100,
        probe_size=8,
        recent_window=10,
        device="cpu",
    )
    first, _ = sample_capacity_probe(rng=np.random.RandomState(11), **kwargs)
    second, _ = sample_capacity_probe(rng=np.random.RandomState(11), **kwargs)
    torch.testing.assert_close(first, second)
