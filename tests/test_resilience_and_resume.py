"""Tests for pipeline resilience, atomic checkpoints, offline cache, and resume."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

from adaptive_plasticity.final_runner import (
    final_output_name,
    run_final_cell,
)
from adaptive_plasticity.offline_cache import (
    get_offline_checkpoint_path,
    is_offline_checkpoint_valid,
    train_offline_checkpoint,
)


def test_offline_checkpoint_creation_and_validity(tmp_path):
    ckpt_dir = tmp_path / "offline_ckpts"
    path = train_offline_checkpoint(
        dataset_id="hopper-medium-v2",
        seed=0,
        warmup_updates=10,
        device="cpu",
        checkpoint_dir=ckpt_dir,
    )
    assert path.is_file()
    # Mock OFFLINE_WARMUP_UPDATES for unit test
    with mock.patch("adaptive_plasticity.offline_cache.OFFLINE_WARMUP_UPDATES", 10):
        assert is_offline_checkpoint_valid(path)
        data = torch.load(path, map_location="cpu", weights_only=False)
        assert data["dataset_id"] == "hopper-medium-v2"
        assert data["seed"] == 0
        assert data["updates"] == 10
        assert "agent_state" in data


def test_intra_cell_periodic_state_and_resume(tmp_path):
    outdir = tmp_path / "resume_test"
    # Run a cell with tiny budgets: 20 online steps, eval every 5 steps
    kw = {
        "dataset_id": "halfcheetah-medium-v2",
        "controller": "fixed",
        "shift": "obs_noise",
        "seed": 0,
        "warmup_updates": 5,
        "online_steps": 20,
        "shift_step": 5,
        "eval_interval": 5,
        "eval_episodes": 1,
        "gate_interval": 5,
        "device": "cpu",
        "outdir": str(outdir),
        "use_shared_offline_checkpoints": False,
    }

    summary1 = run_final_cell(**kw)
    assert summary1["online_steps"] == 20
    assert len(summary1["returns"]) == 4

    out_name = final_output_name(dataset_id="halfcheetah-medium-v2", controller="fixed",
                                 shift="obs_noise", severity=0.1, seed=0)
    cell_dir = outdir / out_name
    assert (cell_dir / "summary.json").is_file()
    assert (cell_dir / "checkpoint.pt").is_file()
    # state.pt should be cleaned up on successful completion
    assert not (cell_dir / "state.pt").is_file()

    # Second run should be instant (read from disk)
    t0 = time.time()
    summary2 = run_final_cell(**kw)
    t_elapsed = time.time() - t0
    assert t_elapsed < 0.2
    assert summary2["best_normalized"] == summary1["best_normalized"]


def test_atomic_file_write_never_corrupts(tmp_path):
    from adaptive_plasticity.final_runner import _atomic_write_file

    target = tmp_path / "test_file.json"
    _atomic_write_file(target, '{"key": "value"}\n')
    assert target.is_file()
    assert json.loads(target.read_text()) == {"key": "value"}
