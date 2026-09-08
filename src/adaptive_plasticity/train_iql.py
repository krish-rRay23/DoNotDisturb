"""M4 reproducible IQL baseline training.

Trains standard Implicit Q-Learning on a validated M3 D4RL v2 dataset and
evaluates the policy in the live environment (evaluation episodes are
separate from the training data). Reuses M1 reproducibility/logging
infrastructure and the M3 dataset loader. No plasticity mechanism, no
normalization, no research modifications.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import yaml

from adaptive_plasticity.datasets import load_local_dataset
from adaptive_plasticity.iql import IQLAgent
from adaptive_plasticity.logging_utils import configure_logging
from adaptive_plasticity.reproducibility import (
    create_metadata,
    detect_device,
    git_commit_hash,
    package_versions,
    prepare_run_directories,
    seed_everything,
    write_metadata,
)

# D4RL reference scores for reporting normalized performance only.
# Never used to modify training data or rewards.
REFERENCE_SCORES: dict[str, dict[str, float]] = {
    "halfcheetah-medium-v2": {"random": -280.178953, "expert": 12135.0},
    "hopper-medium-v2": {"random": -20.272305, "expert": 3234.3},
    "walker2d-medium-v2": {"random": 1.629008, "expert": 4592.3},
    "halfcheetah-medium-replay-v2": {"random": -280.178953, "expert": 12135.0},
    "hopper-medium-replay-v2": {"random": -20.272305, "expert": 3234.3},
    "walker2d-medium-replay-v2": {"random": 1.629008, "expert": 4592.3},
}

ENVIRONMENT_IDS: dict[str, str] = {
    "halfcheetah-medium-v2": "HalfCheetah-v5",
    "hopper-medium-v2": "Hopper-v5",
    "walker2d-medium-v2": "Walker2d-v5",
    "halfcheetah-medium-replay-v2": "HalfCheetah-v5",
    "hopper-medium-replay-v2": "Hopper-v5",
    "walker2d-medium-replay-v2": "Walker2d-v5",
}


def load_training_config(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate an IQL training configuration."""
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Training configuration root must be a YAML mapping.")
    for key in ("seed", "device", "dataset", "algorithm", "training"):
        if key not in config:
            raise ValueError(f"Training configuration is missing required key: {key}")
    normalization = config.get("normalization", {})
    if normalization.get("observations", False) or normalization.get("rewards", False):
        raise ValueError("Observation/reward normalization is not implemented and must stay disabled.")
    if config["algorithm"].get("name") != "iql":
        raise ValueError(f"Unsupported algorithm {config['algorithm'].get('name')!r}; M4 implements standard IQL only.")
    return config


def build_transitions(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Assemble (s, a, r, s', done) arrays from raw D4RL arrays.

    D4RL transition semantics (matching ``d4rl.qlearning_dataset`` and the
    CORL reference): ``dones`` are the true ``terminals`` only. Successors
    shift by one step, with two explicit boundary rules so no cross-episode
    transition is silently fabricated:

    * terminal indices have no successor state: their ``next_observations``
      entry is the observation itself. The value is unused because
      ``dones == 1`` masks bootstrapping, so this is provably behavior-neutral
      hygiene, asserted by regression tests.
    * timeout indices bootstrap per the D4RL/CORL convention with the stored
      successor. The true post-timeout state is not recorded in D4RL files;
      this long-standing reference convention is stated here instead of being
      silent.

    No scaling or filtering is applied.
    """
    observations = np.asarray(data["observations"], dtype=np.float32)
    count = observations.shape[0]
    terminals = np.asarray(data["terminals"]).astype(bool)
    next_observations = np.empty_like(observations)
    next_observations[:-1] = observations[1:]
    next_observations[-1] = observations[-1]
    next_observations[terminals] = observations[terminals]
    return {
        "observations": observations,
        "actions": np.asarray(data["actions"], dtype=np.float32),
        "rewards": np.asarray(data["rewards"], dtype=np.float32),
        "next_observations": next_observations,
        "dones": terminals.astype(np.float32),
    }


def evaluate_policy(
    agent: IQLAgent,
    environment_id: str,
    episodes: int,
    base_seed: int,
) -> dict[str, Any]:
    """Evaluate the deterministic policy in the live environment.

    Evaluation episodes run in the simulator and are fully separate from the
    offline training data.
    """
    returns: list[float] = []
    environment = gym.make(environment_id)
    try:
        agent.policy.eval()
        with torch.no_grad():
            for episode in range(episodes):
                observation, _ = environment.reset(seed=base_seed + episode)
                done = False
                total = 0.0
                while not done:
                    obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=agent.device).unsqueeze(0)
                    action = agent.policy.act(obs_tensor, deterministic=True).squeeze(0).cpu().numpy()
                    observation, reward, terminated, truncated, _ = environment.step(action)
                    total += float(reward)
                    done = bool(terminated or truncated)
                returns.append(total)
        agent.policy.train()
    finally:
        environment.close()
    returns_array = np.asarray(returns, dtype=np.float64)
    return {
        "episodes": episodes,
        "returns": returns,
        "mean": float(returns_array.mean()),
        "std": float(returns_array.std()),
    }


def normalized_score(dataset_id: str, mean_return: float) -> float | None:
    """D4RL normalized score for reporting only (100 * (R - random) / (expert - random))."""
    reference = REFERENCE_SCORES.get(dataset_id)
    if reference is None:
        return None
    return 100.0 * (mean_return - reference["random"]) / (reference["expert"] - reference["random"])


def save_checkpoint(agent: IQLAgent, path: Path, payload: dict[str, Any]) -> Path:
    """Persist agent, optimizers, and run context to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"agent": agent.state_dict(), **payload}, path)
    return path


def train(config: dict[str, Any], output_root: str | Path, logger: Any = None) -> dict[str, Any]:
    """Run one seeded IQL training run. Returns the run summary."""
    output_root = Path(output_root)
    seed = int(config["seed"])
    dataset_id = str(config["dataset"]["id"])
    params = dict(config["algorithm"]["parameters"])
    batch_size = int(params.pop("batch_size"))
    params["num_layers"] = int(params.pop("hidden_layers"))
    total_steps = int(config["training"]["steps"])
    eval_interval = int(config["training"].get("eval_interval", 50000))
    eval_episodes = int(config["training"].get("eval_episodes", 10))
    checkpoint_interval = int(config["training"].get("checkpoint_interval", 100000))

    seed_everything(seed, bool(config.get("deterministic", False)))
    device = detect_device(str(config.get("device", "auto")))

    metadata = create_metadata(config, device, repository_root=output_root)
    run_id = metadata["run_id"]
    paths = prepare_run_directories(output_root, run_id)
    if logger is None:
        logger = configure_logging(paths["logs"] / "run.jsonl", str(config.get("logging", {}).get("level", "INFO")))
    write_metadata(metadata, paths["results"] / "metadata.json")

    data = load_local_dataset(dataset_id)
    transitions = build_transitions(data)
    agent = IQLAgent(
        transitions["observations"].shape[1], transitions["actions"].shape[1], **params, device=device
    )
    # GPU-resident tensors with index sampling via a run-seeded torch
    # generator: seeded and reproducible, avoids per-step host transfers.
    torch_data = {
        name: torch.as_tensor(array, device=agent.device) for name, array in transitions.items()
    }
    sampler = torch.Generator(device=agent.device).manual_seed(seed)
    count = transitions["observations"].shape[0]

    environment_id = ENVIRONMENT_IDS[dataset_id]
    checkpoints: list[str] = []
    last_eval: dict[str, Any] = {}
    start = time.perf_counter()
    for step in range(1, total_steps + 1):
        indices = torch.randint(count, (batch_size,), generator=sampler, device=agent.device)
        batch = {name: array[indices] for name, array in torch_data.items()}
        losses = agent.update(batch)
        if step % 1000 == 0:
            logger.info(
                f"step={step} q_loss={losses['q_loss']:.4f} v_loss={losses['v_loss']:.4f} "
                f"policy_loss={losses['policy_loss']:.4f}",
                extra={"event": "train_step", "step": step, **losses},
            )
        if step % eval_interval == 0 or step == total_steps:
            last_eval = evaluate_policy(agent, environment_id, eval_episodes, base_seed=1000 + seed)
            last_eval["normalized"] = normalized_score(dataset_id, last_eval["mean"])
            last_eval["step"] = step
            logger.info(
                f"eval step={step} return={last_eval['mean']:.1f}+/-{last_eval['std']:.1f}",
                extra={"event": "evaluation", **last_eval},
            )
        if step % checkpoint_interval == 0 or step == total_steps:
            checkpoint = save_checkpoint(
                agent,
                paths["checkpoints"] / f"checkpoint_step_{step}.pt",
                {"step": step, "config": config, "run_id": run_id, "dataset_id": dataset_id, "seed": seed},
            )
            checkpoints.append(str(checkpoint))
    runtime_seconds = time.perf_counter() - start

    summary = {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "environment_id": environment_id,
        "seed": seed,
        "device": device,
        "training_steps": total_steps,
        "batch_size": batch_size,
        "batch_sampler": "torch.Generator manual_seed=run seed, randint with replacement",
        "runtime_seconds": runtime_seconds,
        "steps_per_second": total_steps / runtime_seconds,
        "final_eval": last_eval,
        "checkpoints": checkpoints,
        "git_commit": git_commit_hash(output_root),
        "package_versions": package_versions(("numpy", "PyYAML", "torch", "gymnasium", "h5py")),
    }
    summary_path = paths["results"] / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    logger.info(f"Training complete: {total_steps} steps in {runtime_seconds:.1f}s.", extra={"event": "train_complete"})
    return summary


def main() -> None:
    """Parse arguments, resolve overrides, and run one IQL training run."""
    parser = argparse.ArgumentParser(description="Train a reproducible standard IQL baseline.")
    parser.add_argument("--config", type=Path, default=Path("configs/iql.yaml"))
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("."))
    arguments = parser.parse_args()

    config = load_training_config(arguments.config)
    if arguments.dataset is not None:
        config["dataset"]["id"] = arguments.dataset
    if arguments.seed is not None:
        config["seed"] = arguments.seed
    if arguments.steps is not None:
        config["training"]["steps"] = arguments.steps
    if arguments.device is not None:
        config["device"] = arguments.device
    if config["dataset"]["id"] not in ENVIRONMENT_IDS:
        raise KeyError(f"Unknown M4 dataset {config['dataset']['id']!r}. Known: {sorted(ENVIRONMENT_IDS)}")

    summary = train(config, arguments.output_root)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    print(f"Run: {summary['run_id']} | final return: {summary['final_eval']['mean']:.1f} "
          f"(normalized: {summary['final_eval']['normalized']:.1f})")


if __name__ == "__main__":
    main()
