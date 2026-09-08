"""Short, non-training validation for MuJoCo environments and Minari APIs."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import minari
import mujoco
import numpy as np
import torch


ENVIRONMENT_IDS = ("HalfCheetah-v5", "Hopper-v5", "Walker2d-v5")


def runtime_report() -> dict[str, Any]:
    """Return CPU/GPU and package information without requiring CUDA."""
    cuda_available = torch.cuda.is_available()
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "gymnasium_version": importlib.metadata.version("gymnasium"),
        "mujoco_version": mujoco.__version__,
        "minari_version": importlib.metadata.version("minari"),
    }


def validate_environment(environment_id: str, seed: int = 42) -> dict[str, Any]:
    """Create, reset, step, and close one MuJoCo environment.

    Only one sampled action and one step are taken; this is not an episode or a
    training run. The same seeded reset is repeated in a fresh environment.
    """
    environment = gym.make(environment_id)
    duplicate = gym.make(environment_id)
    try:
        observation, reset_info = environment.reset(seed=seed)
        duplicate_observation, _ = duplicate.reset(seed=seed)
        action = environment.action_space.sample()
        step_result = environment.step(action)
        if len(step_result) != 5:
            raise AssertionError(f"{environment_id} returned {len(step_result)} values from step; expected 5.")
        next_observation, reward, terminated, truncated, step_info = step_result
        if observation.shape != environment.observation_space.shape:
            raise AssertionError("Reset observation shape differs from observation space.")
        if action.shape != environment.action_space.shape:
            raise AssertionError("Sampled action shape differs from action space.")
        if next_observation.shape != environment.observation_space.shape:
            raise AssertionError("Step observation shape differs from observation space.")
        if not isinstance(reward, (float, np.floating)):
            raise AssertionError("Reward is not a scalar float.")
        if not isinstance(terminated, (bool, np.bool_)) or not isinstance(truncated, (bool, np.bool_)):
            raise AssertionError("Termination flags are not boolean values.")
        if not isinstance(reset_info, dict) or not isinstance(step_info, dict):
            raise AssertionError("Reset/step info is not a dictionary.")
        deterministic = bool(np.array_equal(observation, duplicate_observation))
        if not deterministic:
            raise AssertionError("Same seed did not produce an equivalent initial observation.")
        return {
            "environment_id": environment_id,
            "observation_shape": list(observation.shape),
            "action_shape": list(action.shape),
            "step_structure": [type(value).__name__ for value in step_result],
            "deterministic_reset": deterministic,
            "status": "validated",
        }
    finally:
        environment.close()
        duplicate.close()


def minari_api_report() -> dict[str, Any]:
    """Inspect Minari's installed API and locally present datasets only.

    This deliberately does not call ``download_dataset`` or ``load_dataset``.
    Listing remote catalogues retrieves only their small metadata indexes; it
    never transfers a dataset payload. Both the ``D4RL`` and ``mujoco``
    namespaces are queried because MuJoCo locomotion datasets are hosted under
    ``mujoco/`` (e.g. ``mujoco/halfcheetah/medium-v0``), not under ``D4RL/``.
    """
    local_datasets = minari.list_local_datasets()
    public_api = sorted(
        name for name in ("list_local_datasets", "list_remote_datasets", "download_dataset", "load_dataset")
        if callable(getattr(minari, name, None))
    )
    targets = ("halfcheetah", "hopper", "walker2d")
    matching_local = [dataset_id for dataset_id in local_datasets if any(target in dataset_id.lower() for target in targets)]
    report: dict[str, Any] = {
        "minari_version": importlib.metadata.version("minari"),
        "public_api": public_api,
        "local_dataset_count": len(local_datasets),
        "matching_local_datasets": matching_local,
        "download_performed": False,
        "dataset_loading_api": "minari.load_dataset(dataset_id)",
        "dataset_download_api": "minari.download_dataset(dataset_id)",
    }
    remote_lister = getattr(minari, "list_remote_datasets", None)
    if callable(remote_lister):
        remote_datasets: dict[str, Any] = {}
        remote_errors: dict[str, str] = {}
        for prefix in ("D4RL", "mujoco"):
            try:
                listing = remote_lister(prefix=prefix)
                identifiers = listing.keys() if isinstance(listing, dict) else listing
                for dataset_id in identifiers:
                    remote_datasets[dataset_id] = prefix
            except Exception as error:  # Network/catalogue access is optional for local use.
                remote_errors[prefix] = f"{type(error).__name__}: {error}"
        if remote_datasets:
            report["remote_catalogue_accessible"] = True
            report["remote_prefixes_queried"] = ["D4RL", "mujoco"]
            report["matching_remote_datasets"] = sorted(
                dataset_id for dataset_id in remote_datasets if any(target in dataset_id.lower() for target in targets)
            )
        else:
            report["remote_catalogue_accessible"] = False
            report["remote_prefixes_queried"] = ["D4RL", "mujoco"]
            if remote_errors:
                report["remote_catalogue_error"] = "; ".join(f"{prefix}: {message}" for prefix, message in sorted(remote_errors.items()))
    return report


def main() -> None:
    """Print a JSON M2 validation report and optionally save it to disk."""
    parser = argparse.ArgumentParser(description="Validate M2 runtime dependencies without training or dataset downloads.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    arguments = parser.parse_args()
    report = {
        "runtime": runtime_report(),
        "environments": [validate_environment(environment_id, arguments.seed) for environment_id in ENVIRONMENT_IDS],
        "minari": minari_api_report(),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
