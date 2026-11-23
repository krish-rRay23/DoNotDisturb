"""One-cell confirm-gated smoke on a synthetic fixture.

This is the smallest confirm-gated training path. It steps a real Hopper
environment a few times on CPU and trains on arrays built in memory. Those
arrays are not a D4RL dataset. The summary is labeled SYNTHETIC FIXTURE and
is not a manuscript result.

``CONFIRM-FIXTURE-SMOKE`` is not ``CONFIRM-FINAL-150``, ``CONFIRM-PILOT-9``,
or any other matrix token. A dry run writes the plan and does not train.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

from adaptive_plasticity.final_runner import run_final_cell
from adaptive_plasticity.m5 import make_env
from adaptive_plasticity.train_iql import build_transitions

FIXTURE_CONFIRM_TOKEN = "CONFIRM-FIXTURE-SMOKE"
FIXTURE_DATASET_ID = "fixture-hopper"
FIXTURE_ENV_ID = "hopper-medium-v2"
FIXTURE_GYM_NAME = "Hopper-v5"
FIXTURE_ONLINE_STEPS = 4
FIXTURE_WARMUP_UPDATES = 2
FIXTURE_ROWS = 64
FIXTURE_PROBE_SIZE = 32
FIXTURE_BANNER = "SYNTHETIC FIXTURE. These numbers are not experimental results."
OUTPUT_NAME = "FIXTURE-SMOKE-capacity_gate-none-seed0"


def synthetic_transitions(
    obs_dim: int,
    act_dim: int,
    *,
    n: int = FIXTURE_ROWS,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """In-memory transitions. Not a downloaded dataset and not a result."""
    rng = np.random.RandomState(seed)
    observations = rng.randn(n, obs_dim).astype(np.float32)
    actions = rng.uniform(-1.0, 1.0, size=(n, act_dim)).astype(np.float32)
    rewards = rng.randn(n).astype(np.float32)
    terminals = np.zeros(n, dtype=bool)
    terminals[-1] = True
    return build_transitions(
        {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "terminals": terminals,
        }
    )


def fixture_plan() -> dict:
    return {
        "confirm_token": FIXTURE_CONFIRM_TOKEN,
        "dataset_id": FIXTURE_DATASET_ID,
        "data_source": "synthetic_fixture",
        "environment_name": FIXTURE_GYM_NAME,
        "controller": "capacity_gate",
        "shift": "none",
        "seed": 0,
        "device": "cpu",
        "online_steps": FIXTURE_ONLINE_STEPS,
        "warmup_updates": FIXTURE_WARMUP_UPDATES,
        "offline_rows": FIXTURE_ROWS,
        "fixture": True,
        "banner": FIXTURE_BANNER,
        "executed": False,
    }


def execute_fixture_smoke(outdir: Path) -> dict:
    """Run the one cell. Caller has already checked the confirm token."""
    probe_env, _ = make_env(FIXTURE_ENV_ID, seed=0)
    try:
        obs_dim = int(probe_env.observation_space.shape[0])
        act_dim = int(probe_env.action_space.shape[0])
    finally:
        probe_env.close()
    offline = synthetic_transitions(obs_dim, act_dim)

    def _env(dataset_id: str, *, seed: int, idx: int = 0):
        if dataset_id != FIXTURE_DATASET_ID:
            raise RuntimeError(
                f"Fixture smoke refuses to open {dataset_id!r}."
            )
        # Two-step cap so the final eval cannot roll out a full episode.
        env = gym.make(FIXTURE_GYM_NAME, max_episode_steps=2)
        env_seed = seed + idx
        env.reset(seed=env_seed)
        rng = np.random.RandomState(env_seed)
        return env, rng

    outdir.mkdir(parents=True, exist_ok=True)
    summary = run_final_cell(
        dataset_id=FIXTURE_DATASET_ID,
        controller="capacity_gate",
        shift="none",
        seed=0,
        online_steps=FIXTURE_ONLINE_STEPS,
        warmup_updates=FIXTURE_WARMUP_UPDATES,
        shift_step=FIXTURE_ONLINE_STEPS,
        update_freq=1,
        gate_interval=2,
        online_ratio=0.0,
        eval_interval=FIXTURE_ONLINE_STEPS + 1,
        eval_episodes=1,
        device="cpu",
        outdir=outdir,
        deterministic=True,
        use_shared_offline_checkpoints=False,
        output_name=OUTPUT_NAME,
        probe_source="online_buffer",
        recent_window=FIXTURE_PROBE_SIZE,
        probe_size=FIXTURE_PROBE_SIZE,
        gate_operator="shrink_perturb",
        fixture=True,
        offline_data=offline,
        env_factory=_env,
    )
    summary["data_source"] = "synthetic_fixture"
    summary["offline_rows"] = FIXTURE_ROWS
    summary["environment_name"] = FIXTURE_GYM_NAME
    summary["banner"] = FIXTURE_BANNER
    summary_path = outdir / OUTPUT_NAME / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="One-cell CPU smoke on a synthetic fixture. Not a result."
    )
    parser.add_argument("--outdir", type=str, default="results/fixture_smoke")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--confirm-matrix",
        type=str,
        default=None,
        help=f"Must equal {FIXTURE_CONFIRM_TOKEN} to train the one cell.",
    )
    args = parser.parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plan_path = outdir / "FIXTURE_plan.json"
    plan_path.write_text(
        json.dumps(fixture_plan(), indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"FIXTURE plan: 1 cell, {FIXTURE_ONLINE_STEPS} online steps, "
        f"device cpu, data_source synthetic_fixture. Not a result.",
        flush=True,
    )
    print(f"Wrote {plan_path}")
    if not args.execute:
        print("Dry run only. Training was not launched.")
        return
    if args.confirm_matrix != FIXTURE_CONFIRM_TOKEN:
        print(
            f"ERROR: --execute requires --confirm-matrix {FIXTURE_CONFIRM_TOKEN} "
            f"(received {args.confirm_matrix!r}). Refusing.",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(2)
    print(f"CONFIRMED: launching the fixture smoke ({FIXTURE_CONFIRM_TOKEN}).", flush=True)
    summary = execute_fixture_smoke(outdir)
    print(
        f"Fixture smoke finished: update_steps={summary['update_steps']} "
        f"device={summary['environment']['device']} fixture={summary['fixture']}. "
        "Not a result.",
        flush=True,
    )


if __name__ == "__main__":
    main()
