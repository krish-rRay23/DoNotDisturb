"""Shared Offline Pre-training Cache for FINAL 135-run Study.

Pre-trains and caches the 15 base offline checkpoints (3 envs x 5 seeds,
25,000 updates each) to eliminate 3,000,000 redundant offline gradient updates
and ensure strictly paired initial model weights across online experimental arms.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from adaptive_plasticity.iql import IQLAgent
from adaptive_plasticity.m5 import load_offline_data, make_env
from adaptive_plasticity.reproducibility import (
    detect_device,
    git_commit_hash,
    seed_everything,
    utc_timestamp,
)

OFFLINE_WARMUP_UPDATES = 25_000
DEFAULT_CHECKPOINT_DIR = Path("results/offline_checkpoints")


def offline_checkpoint_name(dataset_id: str, seed: int) -> str:
    """Standard filename for shared offline base checkpoint."""
    return f"offline_{dataset_id}_seed{seed}.pt"


def get_offline_checkpoint_path(
    dataset_id: str,
    seed: int,
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
) -> Path:
    return Path(checkpoint_dir) / offline_checkpoint_name(dataset_id, seed)


def is_offline_checkpoint_valid(path: Path) -> bool:
    """Verify that an offline checkpoint file exists and is readable."""
    if not path.is_file():
        return False
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
        return isinstance(data, dict) and "agent_state" in data and data.get("updates", 0) == OFFLINE_WARMUP_UPDATES
    except Exception:
        return False


def train_offline_checkpoint(
    dataset_id: str,
    seed: int,
    warmup_updates: int = OFFLINE_WARMUP_UPDATES,
    device: str = "cuda",
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
    deterministic: bool = True,
) -> Path:
    """Pre-train one offline base checkpoint and save atomically.

    Uses GPU-resident offline data sampling for maximum throughput.
    """
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    target_path = get_offline_checkpoint_path(dataset_id, seed, ckpt_dir)

    if is_offline_checkpoint_valid(target_path):
        return target_path

    t0 = time.time()
    device = detect_device(device)
    torch_device = torch.device(device)

    seed_everything(seed, deterministic=deterministic)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)

    offline_data = load_offline_data(dataset_id)
    obs_dim = int(offline_data["observations"].shape[1])
    act_dim = int(offline_data["actions"].shape[1])
    offline_count = int(offline_data["observations"].shape[0])

    agent = IQLAgent(obs_dim, act_dim, device=device)
    _, rng = make_env(dataset_id, seed=seed)

    # Fast GPU-resident data tensors (zero PCI-e transfers during gradient loop)
    if device.startswith("cuda"):
        offline_gpu = {
            k: torch.as_tensor(v, device=torch_device)
            for k, v in offline_data.items()
        }
        for _ in range(1, int(warmup_updates) + 1):
            indices = rng.randint(0, offline_count, size=256)
            batch = {k: v[indices] for k, v in offline_gpu.items()}
            agent.update(batch)
    else:
        for _ in range(1, int(warmup_updates) + 1):
            indices = rng.randint(0, offline_count, size=256)
            batch = {k: torch.as_tensor(v[indices], device=torch_device)
                     for k, v in offline_data.items()}
            agent.update(batch)

    runtime = time.time() - t0

    payload = {
        "dataset_id": dataset_id,
        "seed": int(seed),
        "updates": int(warmup_updates),
        "agent_state": agent.state_dict(),
        "created_at": utc_timestamp(),
        "git_commit": git_commit_hash(),
        "runtime_seconds": runtime,
    }

    # Atomic write to avoid partial/corrupted files
    fd, tmp_path = tempfile.mkstemp(dir=ckpt_dir, prefix="tmp_offline_", suffix=".pt")
    os.close(fd)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, target_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    return target_path


def ensure_all_offline_checkpoints(
    envs: tuple[str, ...] = ("halfcheetah-medium-v2", "hopper-medium-v2", "walker2d-medium-v2"),
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    warmup_updates: int = OFFLINE_WARMUP_UPDATES,
    device: str = "cuda",
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
) -> list[Path]:
    """Ensure all 15 base offline checkpoints exist, training any missing ones."""
    paths: list[Path] = []
    total = len(envs) * len(seeds)
    count = 0
    print(f"=== Checking {total} shared offline base checkpoints ===", flush=True)

    for env in envs:
        for seed in seeds:
            count += 1
            target = get_offline_checkpoint_path(env, seed, checkpoint_dir)
            if is_offline_checkpoint_valid(target):
                print(f"[{count}/{total}] OK: {target.name}", flush=True)
                paths.append(target)
            else:
                print(f"[{count}/{total}] Training {target.name} ({warmup_updates} updates) ...", flush=True)
                p = train_offline_checkpoint(
                    dataset_id=env,
                    seed=seed,
                    warmup_updates=warmup_updates,
                    device=device,
                    checkpoint_dir=checkpoint_dir,
                )
                paths.append(p)

    print(f"=== All {total} offline base checkpoints ready ===\n", flush=True)
    return paths
