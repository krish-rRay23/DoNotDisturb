"""Reproducibility, runtime metadata, and run-directory utilities."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import random
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp suitable for metadata."""
    return datetime.now(UTC).isoformat()


def generate_run_id(prefix: str = "run") -> str:
    """Create a sortable, collision-resistant identifier for one run."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}-{uuid4().hex[:8]}"


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch RNGs.

    Deterministic behavior is opt-in because it can reduce performance or reject
    nondeterministic CUDA operations. It must be selected by experiment config.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic


def detect_device(requested: str = "auto") -> str:
    """Resolve a requested device without changing global torch state."""
    normalized = requested.lower()
    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if normalized not in {"cpu", "cuda"} and not normalized.startswith("cuda:"):
        raise ValueError(f"Unsupported device request: {requested}")
    if normalized.startswith("cuda:") and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested but CUDA is not available.")
    return normalized


def git_commit_hash(cwd: str | Path | None = None) -> str | None:
    """Return the current Git commit hash, or ``None`` outside a Git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, check=True,
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() or None


def package_versions(packages: tuple[str, ...] = ("numpy", "PyYAML", "torch")) -> dict[str, str | None]:
    """Return installed distribution versions for the requested packages."""
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def create_metadata(config: dict[str, Any], device: str, repository_root: str | Path | None = None) -> dict[str, Any]:
    """Capture timestamped configuration and runtime information for a run."""
    cuda_available = torch.cuda.is_available()
    cuda_name = torch.cuda.get_device_name(0) if cuda_available else None
    return {
        "created_at": utc_timestamp(),
        "run_id": generate_run_id(),
        "config": config,
        "git_commit": git_commit_hash(repository_root),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "package_versions": package_versions(),
        "device": {"selected": device, "cuda_available": cuda_available, "cuda_device_name": cuda_name, "cuda_version": torch.version.cuda},
    }


def prepare_run_directories(root: str | Path, run_id: str) -> dict[str, Path]:
    """Create and return standard directories for a single experiment run."""
    root_path = Path(root)
    directories = {name: root_path / name / run_id for name in ("logs", "checkpoints", "results", "plots")}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def write_metadata(metadata: dict[str, Any], path: str | Path) -> Path:
    """Write JSON metadata to *path* and return its normalized path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return target

