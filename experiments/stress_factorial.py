"""Factorial harness for the 24-run non-stationary stress study + 9-run stable validation.

Matrices:
1. STABLE VALIDATION (9 cells):
   - Env: hopper-medium-v2
   - Shifts: none, obs_noise, reward_scale
   - Controllers: none, fixed, capacity_gate
   - Seed: 0
   - Purpose: Validates that the final active online-buffer probe produces 0 false positives
     under stable/mild conditions, directly bridging the 135-run study.

2. STRESS STUDY (24 cells):
   - Envs: halfcheetah-medium-v2, walker2d-medium-v2
   - Shift: actuator_cripple (step 5,000)
   - Controllers: none, fixed, redo, capacity_gate
   - Seeds: 0, 1, 2
   - Purpose: Proves selective plasticity recovery under true dynamics collapse.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adaptive_plasticity.final_runner import (
    REGIME_SEVERITY,
    _REF_RETURNS,
    final_protocol,
    run_final_cell,
)

STRESS_PROTOCOL_VERSION = "stress-v1"
CONFIRM_STRESS_TOKEN = "CONFIRM-STRESS-24"
CONFIRM_STABLE_TOKEN = "CONFIRM-STABLE-9"

STRESS_ENVS = ("halfcheetah-medium-v2", "walker2d-medium-v2")
STRESS_CONTROLLERS = ("none", "fixed", "redo", "capacity_gate")
STRESS_SEEDS = (0, 1, 2)

STABLE_ENVS = ("hopper-medium-v2",)
STABLE_SHIFTS = ("none", "obs_noise", "reward_scale")
STABLE_CONTROLLERS = ("none", "fixed", "capacity_gate")
STABLE_SEEDS = (0,)


def stress_output_name(
    *,
    dataset_id: str,
    controller: str,
    shift: str,
    severity: float,
    seed: int,
) -> str:
    return f"STRESS-{dataset_id}-{controller}-{shift}-sev{severity:g}-seed{seed}"


def stable_output_name(
    *,
    dataset_id: str,
    controller: str,
    shift: str,
    severity: float,
    seed: int,
) -> str:
    return f"STABLE_VAL-{dataset_id}-{controller}-{shift}-sev{severity:g}-seed{seed}"


def build_stress_plan(*, device: str = "cpu") -> list[dict[str, Any]]:
    plan = []
    for env in STRESS_ENVS:
        for ctrl in STRESS_CONTROLLERS:
            for seed in STRESS_SEEDS:
                plan.append({
                    "dataset_id": env,
                    "controller": ctrl,
                    "shift": "actuator_cripple",
                    "severity": 1.0,
                    "shift_step": 5_000,
                    "online_steps": 25_000,
                    "seed": seed,
                    "device": device,
                    "output_name": stress_output_name(
                        dataset_id=env, controller=ctrl, shift="actuator_cripple",
                        severity=1.0, seed=seed,
                    ),
                })
    return plan


def build_stable_validation_plan(*, device: str = "cpu") -> list[dict[str, Any]]:
    plan = []
    for env in STABLE_ENVS:
        for shift in STABLE_SHIFTS:
            for ctrl in STABLE_CONTROLLERS:
                for seed in STABLE_SEEDS:
                    severity = REGIME_SEVERITY[shift]
                    shift_step = 0 if shift == "none" else 5_000
                    plan.append({
                        "dataset_id": env,
                        "controller": ctrl,
                        "shift": shift,
                        "severity": severity,
                        "shift_step": shift_step,
                        "online_steps": 25_000,
                        "seed": seed,
                        "device": device,
                        "output_name": stable_output_name(
                            dataset_id=env, controller=ctrl, shift=shift,
                            severity=severity, seed=seed,
                        ),
                    })
    return plan


def _execute_worker_task(kwargs: dict[str, Any]) -> dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    return run_final_cell(**kwargs)


def execute_plan(
    plan: list[dict[str, Any]],
    outdir: Path,
    manifest_prefix: str = "STRESS",
    workers: int = 6,
) -> list[dict[str, Any]]:
    manifest_path = outdir / f"{manifest_prefix}_manifest.jsonl"
    rliable_csv = outdir / f"{manifest_prefix}_rliable_scores.csv"

    completed = []
    pending = []
    for cell in plan:
        cell_out = outdir / cell["output_name"]
        summary_file = cell_out / "summary.json"
        ckpt_file = cell_out / "checkpoint.pt"
        if summary_file.is_file() and ckpt_file.is_file():
            try:
                data = json.loads(summary_file.read_text(encoding="utf-8"))
                completed.append(data)
                continue
            except Exception:
                pass
        kw = dict(cell)
        kw["outdir"] = str(outdir)
        pending.append(kw)

    print(f"[{manifest_prefix}] Total: {len(plan)} | Cached: {len(completed)} | To execute: {len(pending)}")

    if pending:
        workers = max(1, min(int(workers), len(pending)))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_execute_worker_task, kw): kw["output_name"] for kw in pending}
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    res = fut.result()
                    completed.append(res)
                    print(f"  [DONE] {name} (runtime: {res.get('runtime_seconds', 0):.1f}s, "
                          f"norm: {res.get('normalized', [-1])[-1]:.2f})", flush=True)
                except Exception as e:
                    print(f"  [FAIL] {name}: {e}", file=sys.stderr, flush=True)
                    raise

    # Write manifest
    with open(manifest_path, "w", encoding="utf-8") as f:
        for item in completed:
            f.write(json.dumps(item) + "\n")
    print(f"Wrote manifest to {manifest_path}")

    # Write rliable csv
    with open(rliable_csv, "w", encoding="utf-8") as f:
        f.write("environment,algorithm,seed,score\n")
        for item in completed:
            env = item.get("environment", item.get("dataset_id"))
            ctrl = item.get("controller")
            seed = item.get("seed")
            score = item.get("normalized", [0.0])[-1]
            f.write(f"{env},{ctrl},{seed},{score}\n")
    print(f"Wrote rliable scores to {rliable_csv}")

    return completed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Stress & Stable Validation Factorial Harness")
    parser.add_argument("--execute", action="store_true", help="Launch execution")
    parser.add_argument("--confirm-matrix", type=str, default="", help="Confirmation token")
    parser.add_argument("--stable-validate", action="store_true", help="Run 9-cell stable validation")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--outdir", type=str, default="results/stress_study")

    args = parser.parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.stable_validate:
        plan = build_stable_validation_plan(device=args.device)
        print(f"=== STABLE VALIDATION PLAN (9 cells) ===")
        for p in plan:
            print(f"  {p['output_name']}")
        if args.execute:
            if args.confirm_matrix != CONFIRM_STABLE_TOKEN:
                print(f"ERROR: --stable-validate requires --confirm-matrix {CONFIRM_STABLE_TOKEN}", file=sys.stderr)
                sys.exit(2)
            execute_plan(plan, outdir, manifest_prefix="STABLE_VAL", workers=args.workers)
        return

    # Default: Stress study
    plan = build_stress_plan(device=args.device)
    print(f"=== 24-CELL STRESS STUDY PLAN ===")
    for p in plan:
        print(f"  {p['output_name']}")
    if args.execute:
        if args.confirm_matrix != CONFIRM_STRESS_TOKEN:
            print(f"ERROR: --execute requires --confirm-matrix {CONFIRM_STRESS_TOKEN}", file=sys.stderr)
            sys.exit(2)
        execute_plan(plan, outdir, manifest_prefix="STRESS", workers=args.workers)
    else:
        print("Dry run only: execution not launched.")


if __name__ == "__main__":
    main()
