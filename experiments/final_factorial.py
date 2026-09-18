"""Final factorial study harness: FINAL confirmatory protocol (approved).

Primary matrix: 3 envs x 3 regimes x 3 arms (none/fixed/capacity_gate)
  x 5 seeds = 135 cells.
M10 reference plan: halfcheetah-medium-v2 only x 3 regimes x m10ref
  x 5 seeds = 15 cells (separate plan, never part of the primary matrix).
Total: 150 runs.
Protocol: 25,000 offline updates, 25,000 online steps, shift_step 5,000,
  online_ratio 0.5, update_freq 1, eval every 2,500 steps x 10 episodes.
Regime severities: none=0.0, obs_noise sigma=0.1, reward_scale scale=0.5.

Modes:
  --dry-run   (default) enumerate + validate both plans, write
              results/FINAL_plan.json (135 primary) and
              results/FINAL_reference_plan.json (15 reference) with
              complete protocol metadata. No training, no env interaction.
  --smoke     non-training gate-metric check (synthetic probes, cost timing).
  --smoke-run miniature loop verification (tiny budgets, real env + data):
              asserts online update counts, shift timing, and determinism.
   --execute   run the full 150-cell study ONLY with --confirm-matrix
               CONFIRM-FINAL-150. Anything else refuses. DO NOT LAUNCH
               except when authorized.

Execution (no science changes — mechanics only):
  --workers N parallel cell workers sharing this process (default 9, as
              used in the M10 study). Each cell still writes its own unique
              output directory. Completed cells (summary.json + checkpoint.pt
              present and readable) are skipped/resumed; any cell failure
              stops further scheduling and aborts after in-flight cells
              finish. Progress prints as [i/N] launch/finish/skip lines.

Pre-pilot mode (NOT the final study):
  --pilot     small 9-cell verification matrix on hopper-medium-v2 only
              (1 env x 3 regimes x 3 arms x 1 seed), reduced budgets
              (5k offline / 5k online), PILOT- output prefix, PILOT_plan.json
              artifacts, and its own CONFIRM-PILOT-9 token. Reuses the frozen
              final runner read-only with overridden budgets; frozen
              CapacityGate logic, thresholds, intervention mechanics, M10,
              and final defaults are untouched. --pilot --execute runs ONLY
              with --confirm-matrix CONFIRM-PILOT-9 (the FINAL token is
              rejected in pilot mode and vice versa).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adaptive_plasticity.final_runner import (
    CONFIRM_TOKEN,
    FINAL_CONTROLLERS,
    FINAL_ENVS,
    FINAL_SEEDS,
    FINAL_SHIFTS,
    PRIMARY_CONTROLLERS,
    REFERENCE_CONTROLLERS,
    REFERENCE_ENVS,
    REGIME_SEVERITY,
    final_output_name,
    final_protocol,
    run_final_cell,
)

#: Default arms for the primary confirmatory matrix (m10ref excluded;
#: it runs only via the separate reference plan).
DEFAULT_CONTROLLERS = ("none", "fixed", "capacity_gate")

#: Default parallel cell workers (same count used in the M10 study).
DEFAULT_WORKERS: int = 9

# ---------------------------------------------------------------------------
# PRE-PILOT mode (9 runs). Reduced budgets ONLY; frozen runner/gate reused
# read-only. Nothing here changes final-study defaults or thresholds.
# ---------------------------------------------------------------------------

#: Distinct protocol label so pilot artifacts can never be mistaken for FINAL.
PILOT_PROTOCOL_VERSION: str = "pilot-v1"

#: Pilot matrix: 1 env x 3 regimes x 3 arms x 1 seed = 9 cells.
PILOT_ENVS: tuple[str, ...] = ("hopper-medium-v2",)
PILOT_SHIFTS: tuple[str, ...] = ("none", "obs_noise", "reward_scale")
PILOT_CONTROLLERS: tuple[str, ...] = ("none", "fixed", "capacity_gate")
PILOT_SEEDS: tuple[int, ...] = (0,)

#: Confirmation token for pilot execution (FINAL token is rejected in pilot).
PILOT_CONFIRM_TOKEN: str = "CONFIRM-PILOT-9"

#: Reduced pilot budgets (execution kwargs only; final defaults untouched).
PILOT_OFFLINE_WARMUP_UPDATES: int = 5_000
PILOT_ONLINE_STEPS: int = 5_000
PILOT_SHIFT_STEP: int = 1_000
PILOT_ONLINE_RATIO: float = 0.5
PILOT_UPDATE_FREQ: int = 1
PILOT_EVAL_INTERVAL: int = 1_000
PILOT_EVAL_EPISODES: int = 5
PILOT_GATE_INTERVAL: int = 1_000
PILOT_BATCH_SIZE: int = 256
PILOT_PROBE_SIZE: int = 256


def pilot_protocol() -> dict:
    """Pilot protocol dict: frozen final thresholds/severities, reduced budgets."""
    frozen = final_protocol()
    return {
        "protocol_version": PILOT_PROTOCOL_VERSION,
        "based_on": frozen["protocol_version"],
        "offline_warmup_updates": PILOT_OFFLINE_WARMUP_UPDATES,
        "online_steps": PILOT_ONLINE_STEPS,
        "shift_step": PILOT_SHIFT_STEP,
        "regime_severity": dict(REGIME_SEVERITY),
        "online_ratio": PILOT_ONLINE_RATIO,
        "update_freq": PILOT_UPDATE_FREQ,
        "batch_size": PILOT_BATCH_SIZE,
        "eval_interval": PILOT_EVAL_INTERVAL,
        "eval_episodes": PILOT_EVAL_EPISODES,
        "gate_interval": PILOT_GATE_INTERVAL,
        "probe_size": PILOT_PROBE_SIZE,
        "noise_rng_offset": frozen["noise_rng_offset"],
        "deterministic": True,
        # Reuse the frozen gate thresholds verbatim (no retuning).
        "gate_thresholds": dict(frozen["gate_thresholds"]),
    }


def pilot_output_name(
    *,
    dataset_id: str,
    controller: str,
    shift: str,
    severity: float,
    seed: int,
) -> str:
    """Unified pilot identity: ``PILOT-<env>-<ctrl>-<shift>-sev<g>-seed<N>``."""
    return "PILOT-%s-%s-%s-sev%g-seed%d" % (dataset_id, controller, shift, float(severity), int(seed))


def build_pilot_plan(*, device: str = "cuda") -> list[dict]:
    """Enumerate the 9-cell pre-pilot matrix (fixed spec, ignores CLI filters)."""
    protocol = pilot_protocol()
    plan: list[dict] = []
    for env in PILOT_ENVS:
        for shift in PILOT_SHIFTS:
            for ctrl in PILOT_CONTROLLERS:
                for seed in PILOT_SEEDS:
                    severity = REGIME_SEVERITY[shift]
                    shift_step = 0 if shift == "none" else PILOT_SHIFT_STEP
                    plan.append({
                        "dataset_id": env,
                        "controller": ctrl,
                        "shift": shift,
                        "shift_step": shift_step,
                        "severity": severity,
                        "seed": int(seed),
                        "device": device,
                        # Reduced-budget overrides forwarded to run_final_cell.
                        "warmup_updates": PILOT_OFFLINE_WARMUP_UPDATES,
                        "online_steps": PILOT_ONLINE_STEPS,
                        "update_freq": PILOT_UPDATE_FREQ,
                        "gate_interval": PILOT_GATE_INTERVAL,
                        "online_ratio": PILOT_ONLINE_RATIO,
                        "eval_interval": PILOT_EVAL_INTERVAL,
                        "eval_episodes": PILOT_EVAL_EPISODES,
                        "output_name": pilot_output_name(
                            dataset_id=env, controller=ctrl, shift=shift,
                            severity=severity, seed=seed),
                        "protocol": dict(protocol, shift_step=shift_step,
                                          severity=severity),
                    })
    names = [c["output_name"] for c in plan]
    assert len(names) == len(set(names)), "pilot output names must be unique"
    assert len(plan) == 9, f"pilot plan must be 9 cells, got {len(plan)}"
    return plan


def build_plan(
    *,
    seeds=FINAL_SEEDS,
    controllers=DEFAULT_CONTROLLERS,
    envs=FINAL_ENVS,
    shifts=FINAL_SHIFTS,
    device: str = "cuda",
) -> list[dict]:
    """Enumerate a factorial plan; every cell carries the full protocol.

    Defaults enumerate the 135-cell primary matrix (3 envs x 3 regimes x
    3 arms x 5 seeds). Pass explicit ``controllers``/``envs`` for subsets
    or the M10-reference plan (see ``build_reference_plan``).
    """
    ctrls = list(controllers)
    for c in ctrls:
        if c not in FINAL_CONTROLLERS:
            raise ValueError(f"Unknown controller {c!r}")
    for e in envs:
        if e not in FINAL_ENVS:
            raise ValueError(f"Unknown env {e!r}")
    for s in shifts:
        if s not in FINAL_SHIFTS:
            raise ValueError(f"Unknown shift {s!r}")
    protocol = final_protocol()
    plan: list[dict] = []
    for env in envs:
        for shift in shifts:
            for ctrl in ctrls:
                for seed in seeds:
                    severity = REGIME_SEVERITY[shift]
                    shift_step = 0 if shift == "none" else protocol["shift_step"]
                    plan.append({
                        "dataset_id": env,
                        "controller": ctrl,
                        "shift": shift,
                        "shift_step": shift_step,
                        "severity": severity,
                        "seed": int(seed),
                        "device": device,
                        "output_name": final_output_name(
                            dataset_id=env, controller=ctrl, shift=shift,
                            severity=severity, seed=seed),
                        "protocol": dict(protocol, shift_step=shift_step,
                                         severity=severity),
                    })
    names = [c["output_name"] for c in plan]
    assert len(names) == len(set(names)), "output names must be unique"
    return plan


def build_primary_plan(*, device: str = "cuda") -> list[dict]:
    """Approved 135-cell primary matrix (m10ref excluded by design)."""
    plan = build_plan(seeds=FINAL_SEEDS, controllers=PRIMARY_CONTROLLERS,
                      envs=FINAL_ENVS, shifts=FINAL_SHIFTS, device=device)
    assert len(plan) == 135, f"primary plan must be 135 cells, got {len(plan)}"
    assert "m10ref" not in {c["controller"] for c in plan}
    return plan


def build_reference_plan(*, device: str = "cuda") -> list[dict]:
    """Approved 15-cell M10-reference plan (halfcheetah only, m10ref only)."""
    plan = build_plan(seeds=FINAL_SEEDS, controllers=REFERENCE_CONTROLLERS,
                      envs=REFERENCE_ENVS, shifts=FINAL_SHIFTS, device=device)
    assert len(plan) == 15, f"reference plan must be 15 cells, got {len(plan)}"
    return plan


def run_smoke(outdir: Path) -> dict:
    """Non-training smoke: synthetic probe + gate eval + cost timing."""
    import numpy as np
    import torch
    from adaptive_plasticity.capacity_gate import (
        CapacityGateController, compute_capacity_metrics,
    )
    from adaptive_plasticity.iql import IQLAgent

    results = {"cells": [], "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for env in FINAL_ENVS:
        torch.manual_seed(0)
        agent = IQLAgent(17, 6, device="cpu")
        probe = torch.as_tensor(
            np.random.RandomState(0).randn(256, 17).astype(np.float32))
        t0 = time.perf_counter()
        metrics = compute_capacity_metrics(agent.policy, probe)
        cost = time.perf_counter() - t0
        gate = CapacityGateController()
        rec = gate.update(dormancy=metrics["dormancy"],
                          effective_rank=metrics["effective_rank"])
        results["cells"].append({
            "dataset_id": env, "controller": "capacity_gate",
            "dormancy": metrics["dormancy"],
            "effective_rank": metrics["effective_rank"],
            "numerical_rank": metrics["numerical_rank"],
            "gate_on": rec["gate_on"],
            "eval_cost_seconds": cost,
        })
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "FINAL_smoke.json").write_text(
        json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")
    return results


def run_smoke_run(outdir: Path) -> dict:
    """Miniature loop verification with tiny budgets (real env + data).

    Verifies: online updates execute every step, shift activates exactly at
    shift_step, observation noise is deterministic across repeats, and arm
    trigger logic differs as intended. Uses halfcheetah + obs_noise only.
    """
    tiny = {"online_steps": 30, "warmup_updates": 5, "shift_step": 10,
            "eval_interval": 15, "eval_episodes": 1, "gate_interval": 5,
            "device": "cpu", "outdir": outdir / "smoke_run"}
    checks: dict = {"cells": []}

    def run_cell(**kw):
        return run_final_cell(dataset_id="halfcheetah-medium-v2", shift="obs_noise",
                              seed=0, **{**tiny, **kw})

    s_fixed = run_cell(controller="fixed")
    s_gate = run_cell(controller="capacity_gate")
    s_none = run_cell(controller="none")
    # Determinism: identical repeat must match exactly (modulo timestamps).
    s_repeat = run_final_cell(dataset_id="halfcheetah-medium-v2", shift="obs_noise",
                              seed=0, controller="fixed", **tiny)
    for s in (s_fixed, s_gate, s_none):
        assert s["update_steps"] == tiny["online_steps"], (
            f"update_steps={s['update_steps']} != online_steps")
        assert s["online_steps"] == tiny["online_steps"]
        assert len(s["returns"]) == 2, f"expected 2 evals, got {len(s['returns'])}"
    # Shift timing: fixed arm applies exactly steps [10..29].
    assert s_fixed["applied_intervention_steps"] == tiny["online_steps"] - tiny["shift_step"]
    # none arm never applies.
    assert s_none["applied_intervention_steps"] == 0
    # Determinism across repeats (returns, normalized, applied counts, updates).
    for key in ("returns", "normalized", "update_steps", "applied_intervention_steps"):
        assert s_fixed[key] == s_repeat[key], f"nondeterministic {key}"
    assert (outdir / "smoke_run" / s_fixed["output_name"] / "checkpoint.pt").exists()
    assert (outdir / "smoke_run" / s_gate["output_name"] / f"gate_{s_gate['output_name']}.jsonl").exists()
    gate_lines = (outdir / "smoke_run" / s_gate["output_name"]
                  / f"gate_{s_gate['output_name']}.jsonl").read_text().splitlines()
    assert len(gate_lines) == tiny["online_steps"] // tiny["gate_interval"]
    checks["cells"] = [
        {"controller": s["controller"], "update_steps": s["update_steps"],
         "applied": s["applied_intervention_steps"], "evals": len(s["returns"]),
         "gate_log_lines": len(gate_lines) if s["controller"] == "capacity_gate" else 0}
        for s in (s_fixed, s_gate, s_none)]
    checks["determinism_repeat_match"] = True
    checks["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (outdir / "FINAL_smoke_run.json").write_text(
        json.dumps(checks, indent=2, default=str) + "\n", encoding="utf-8")
    return checks


def write_manifest(summaries: list[dict], outdir: Path, *, prefix: str = "FINAL") -> tuple[Path, Path]:
    """Machine-readable manifest (JSONL) + rliable-ready CSV of scores."""
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = outdir / f"{prefix}_manifest.jsonl"
    with open(manifest, "w", encoding="utf-8") as f:
        for s in summaries:
            f.write(json.dumps({
                "env": s["dataset_id"], "controller": s["controller"],
                "shift": s["shift"], "seed": s["seed"],
                "score": s.get("best_normalized"),
                "output_name": s.get("output_name"),
            }, default=str) + "\n")
    csv_path = outdir / f"{prefix}_rliable_scores.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["env", "controller", "shift", "seed", "score"])
        w.writeheader()
        for s in summaries:
            w.writerow({"env": s["dataset_id"], "controller": s["controller"],
                        "shift": s["shift"], "seed": s["seed"],
                        "score": s.get("best_normalized")})
    return manifest, csv_path


def cell_completed(outdir: str | Path, cell: dict) -> dict | None:
    """Return the saved summary when a cell already finished, else None.

    A cell counts as completed only when BOTH ``summary.json`` (readable)
    and ``checkpoint.pt`` exist in its unique output directory. Anything
    else (missing or unreadable files) means the cell must (re-)run.
    Read-only: never creates, modifies, or launches anything.
    """
    cell_dir = Path(outdir) / cell["output_name"]
    summary_path = cell_dir / "summary.json"
    ckpt_path = cell_dir / "checkpoint.pt"
    if not (summary_path.is_file() and ckpt_path.is_file()):
        return None
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _cell_runner_kwargs(cell: dict, outdir: str | Path) -> dict:
    """Build the ``run_final_cell`` kwargs for one planned cell."""
    kwargs: dict = {
        "dataset_id": cell["dataset_id"], "controller": cell["controller"],
        "shift": cell["shift"], "seed": cell["seed"],
        "severity": cell["severity"], "shift_step": cell["shift_step"],
        "device": cell.get("device", "cuda"),
        "outdir": outdir,
    }
    for key in ("warmup_updates", "online_steps", "update_freq",
                "gate_interval", "online_ratio", "eval_interval",
                "eval_episodes", "output_name"):
        if key in cell:
            kwargs[key] = cell[key]
    return kwargs


def execute_plan(plan, outdir, *, runner=None, manifest_prefix: str = "FINAL",
                 workers: int = 1):
    """Invoke the final runner for every planned cell.

    Called ONLY after the mandatory confirmation guard in ``main`` has
    accepted the exact confirmation token. ``runner`` defaults to the real
    final runner (resolved at call time) and is injectable so tests can
    prove the execution path is entered without training. Cells may carry
    optional pilot budget overrides (warmup_updates, online_steps,
    update_freq, gate_interval, online_ratio, eval_interval,
    eval_episodes); final cells omit them and fall back to the frozen
    runner defaults.

    Execution mechanics (no science impact):

    * ``workers`` caps how many cells run concurrently (ThreadPoolExecutor,
      shared process). ``workers=1`` is strictly sequential.
    * Completed cells (see ``cell_completed``) are skipped and their saved
      summaries reused, so re-invocation resumes the matrix.
    * The first cell failure stops further scheduling: cells not yet
      started never run; in-flight cells finish; then the original
      exception is re-raised and no manifest is written.
    * Returned summaries (and manifest rows) always follow plan order.
    * Progress prints as ``[i/N]`` launch/finish/skip/FAIL lines, where N
      is the plan size (150 for the full study).
    """
    if runner is None:
        runner = run_final_cell
    workers = int(workers)
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers!r}")
    outdir = Path(outdir)
    total = len(plan)
    summaries: list = [None] * total
    pending: list[tuple[int, dict]] = []
    for idx, cell in enumerate(plan):
        saved = cell_completed(outdir, cell)
        if saved is not None:
            summaries[idx] = saved
            print(f"[{idx + 1}/{total}] skipped (completed) "
                  f"{cell['output_name']}", flush=True)
        else:
            pending.append((idx, cell))
    if not pending:
        write_manifest(summaries, outdir, prefix=manifest_prefix)
        return summaries

    if workers == 1:
        for idx, cell in pending:
            print(f"[{idx + 1}/{total}] launching {cell['output_name']} ...",
                  flush=True)
            try:
                summaries[idx] = runner(**_cell_runner_kwargs(cell, outdir))
            except Exception as exc:
                print(f"[{idx + 1}/{total}] FAILED {cell['output_name']}: "
                      f"{exc!r}", file=sys.stderr, flush=True)
                raise
            print(f"[{idx + 1}/{total}] finished {cell['output_name']}",
                  flush=True)
        write_manifest(summaries, outdir, prefix=manifest_prefix)
        return summaries

    is_mock = (
        "mock" in type(runner).__module__
        or "unittest.mock" in str(type(runner))
        or getattr(runner, "_mock_name", None) is not None
    )
    pool_cls = ThreadPoolExecutor if is_mock else ProcessPoolExecutor
    first_failure: BaseException | None = None
    with pool_cls(max_workers=workers) as pool:
        future_to_pos: dict = {}
        queue = iter(pending)

        def _submit(idx: int, cell: dict) -> None:
            print(f"[{idx + 1}/{total}] launching {cell['output_name']} ...",
                  flush=True)
            fut = pool.submit(runner, **_cell_runner_kwargs(cell, outdir))
            future_to_pos[fut] = (idx, cell)

        for _ in range(min(workers, len(pending))):
            idx, cell = next(queue)
            _submit(idx, cell)
        while future_to_pos and first_failure is None:
            done, _ = wait(future_to_pos, return_when=FIRST_COMPLETED)
            for fut in done:
                idx, cell = future_to_pos.pop(fut)
                try:
                    summaries[idx] = fut.result()
                except Exception as exc:  # noqa: BLE001 — must record, then stop
                    first_failure = exc
                    print(f"[{idx + 1}/{total}] FAILED "
                          f"{cell['output_name']}: {exc!r}",
                          file=sys.stderr, flush=True)
                else:
                    print(f"[{idx + 1}/{total}] finished "
                          f"{cell['output_name']}", flush=True)
            if first_failure is None:
                for _ in range(len(done)):
                    try:
                        idx, cell = next(queue)
                    except StopIteration:
                        break
                    _submit(idx, cell)
        if first_failure is not None:
            for fut in future_to_pos:
                fut.cancel()
    if first_failure is not None:
        raise first_failure
    write_manifest(summaries, outdir, prefix=manifest_prefix)
    return summaries


def print_pilot_verification_hint(summaries: list[dict]) -> None:
    """Print the intended pilot checks (no assertions; for human review)."""
    by_shift: dict[str, dict[str, int]] = {}
    for s in summaries:
        by_shift.setdefault(s["shift"], {})[s["controller"]] = int(
            s.get("applied_intervention_steps", -1))
    print("Pilot verification hints (expect none==0, fixed==4 interventions, "
          "gate in [0, fixed]):")
    for shift in PILOT_SHIFTS:
        row = by_shift.get(shift, {})
        print(f"  shift={shift}: " + ", ".join(
            f"{c} applied={row.get(c, '?')}" for c in PILOT_CONTROLLERS))
    print("Also verify at runtime: gate starts OFF on healthy adaptation, "
          "gate_*.jsonl + checkpoint.pt + summary.json exist per PILOT- dir, "
          "and 5 evals per run.")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Final factorial study harness.")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--controllers", type=str, default="none,fixed,capacity_gate",
                        help="Primary-matrix arms filter (default: the approved "
                             "none,fixed,capacity_gate set; m10ref runs only via "
                             "the separate reference plan).")
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--pilot", action="store_true",
                        help="Pre-pilot verification mode: fixed 9-cell matrix "
                             "(hopper-medium-v2 x 3 regimes x 3 arms x seed 0) "
                             "with reduced budgets. Writes PILOT_* artifacts "
                             "only; never touches FINAL defaults.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Execution device for every cell (default cuda; "
                             "fails loudly if unavailable).")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help="Parallel cell workers sharing this process "
                             f"(default {DEFAULT_WORKERS}, as used in the M10 "
                             "study). 1 = strictly sequential. Completed cells "
                             "are skipped; any failure stops scheduling.")
    parser.add_argument("--confirm-matrix", type=str, default=None,
                        help=f"Must equal {CONFIRM_TOKEN} to launch the matrix "
                             f"(or {PILOT_CONFIRM_TOKEN} with --pilot).")
    args = parser.parse_args(argv)

    if int(args.workers) < 1:
        print(f"ERROR: --workers must be >= 1 (received {args.workers!r}). "
              f"Refusing.", file=sys.stderr, flush=True)
        sys.exit(2)

    outdir = Path(args.outdir)

    # ----- PRE-PILOT branch (fully separate from the FINAL path) -----
    if args.pilot:
        if args.smoke or args.smoke_run:
            print("ERROR: --pilot cannot be combined with --smoke/--smoke-run "
                  "(those are FINAL-only checks). Refusing.",
                  file=sys.stderr, flush=True)
            sys.exit(2)
        plan = build_pilot_plan(device=args.device)
        print(f"PILOT plan: {len(plan)} cells "
              f"({len(PILOT_ENVS)} envs x {len(PILOT_SHIFTS)} shifts x "
              f"{len(PILOT_CONTROLLERS)} ctrls x {len(PILOT_SEEDS)} seeds) "
              f"[PRE-PILOT, not FINAL]")
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "PILOT_plan.json").write_text(
            json.dumps({"cells": plan, "n_cells": len(plan),
                        "protocol": pilot_protocol()}, indent=2) + "\n",
            encoding="utf-8")
        print(f"Wrote {outdir / 'PILOT_plan.json'}")
        if args.execute:
            if args.confirm_matrix != PILOT_CONFIRM_TOKEN:
                print(f"ERROR: --pilot --execute requires --confirm-matrix "
                      f"{PILOT_CONFIRM_TOKEN} (received {args.confirm_matrix!r}). "
                      f"Refusing.", file=sys.stderr, flush=True)
                sys.exit(2)
            print(f"CONFIRMED: launching PILOT execution of {len(plan)} cells.",
                  flush=True)
            summaries = execute_plan(plan, outdir, manifest_prefix="PILOT",
                                         workers=args.workers)
            print_pilot_verification_hint(summaries)
            print(f"Executed {len(summaries)} PILOT cells.", flush=True)
        else:
            print("PILOT dry run only: training matrix NOT launched (as required).")
        return

    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip() != "")
    controllers = tuple(c.strip() for c in args.controllers.split(",") if c.strip())
    for c in controllers:
        if c not in PRIMARY_CONTROLLERS:
            print(f"ERROR: --controllers accepts only primary arms "
                  f"{list(PRIMARY_CONTROLLERS)} (received {c!r}); m10ref runs "
                  f"only via the separate reference plan. Refusing.",
                  file=sys.stderr, flush=True)
            sys.exit(2)

    plan = build_plan(seeds=seeds, controllers=controllers, device=args.device)
    reference_plan = build_reference_plan(device=args.device)
    print(f"Primary plan: {len(plan)} cells "
          f"({len(FINAL_ENVS)} envs x {len(FINAL_SHIFTS)} shifts x "
          f"{len(controllers)} ctrls x {len(seeds)} seeds)")
    print(f"Reference plan: {len(reference_plan)} cells "
          f"({len(REFERENCE_ENVS)} envs x {len(FINAL_SHIFTS)} shifts x "
          f"{len(REFERENCE_CONTROLLERS)} ctrls x {len(FINAL_SEEDS)} seeds)")
    print(f"Total: {len(plan) + len(reference_plan)} runs")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "FINAL_plan.json").write_text(
        json.dumps({"cells": plan, "n_cells": len(plan),
                    "protocol": final_protocol()}, indent=2) + "\n",
        encoding="utf-8")
    print(f"Wrote {outdir / 'FINAL_plan.json'}")
    (outdir / "FINAL_reference_plan.json").write_text(
        json.dumps({"cells": reference_plan,
                    "n_cells": len(reference_plan),
                    "protocol": final_protocol()}, indent=2) + "\n",
        encoding="utf-8")
    print(f"Wrote {outdir / 'FINAL_reference_plan.json'}")

    if args.smoke:
        smoke = run_smoke(outdir)
        for cell in smoke["cells"]:
            print(f"  {cell['dataset_id']}: dorm={cell['dormancy']:.4f} "
                  f"erank={cell['effective_rank']:.1f} gate={cell['gate_on']} "
                  f"cost={cell['eval_cost_seconds']*1000:.1f}ms")

    if args.smoke_run:
        checks = run_smoke_run(outdir)
        for cell in checks["cells"]:
            print(f"  {cell['controller']}: updates={cell['update_steps']} "
                  f"applied={cell['applied']} evals={cell['evals']} "
                  f"gate_lines={cell['gate_log_lines']}")
        print(f"  determinism repeat match: {checks['determinism_repeat_match']}")

    if args.execute:
        # Mandatory confirmation guard: exact token match required. The plan
        # above is ALWAYS written first, so any refusal looks like "stops
        # after plan generation" -- the ERROR goes to stderr and echoes the
        # received value so a near-miss token is diagnosable.
        if args.confirm_matrix != CONFIRM_TOKEN:
            print(f"ERROR: --execute requires --confirm-matrix {CONFIRM_TOKEN} "
                  f"(received {args.confirm_matrix!r}). Refusing.",
                  file=sys.stderr, flush=True)
            sys.exit(2)
        print(f"CONFIRMED: launching execution of {len(plan)} primary + "
              f"{len(reference_plan)} reference cells.", flush=True)
        summaries = execute_plan(plan, outdir, manifest_prefix="FINAL",
                                     workers=args.workers)
        ref_summaries = execute_plan(reference_plan, outdir,
                                     manifest_prefix="FINAL_reference",
                                     workers=args.workers)
        print(f"Executed {len(summaries)} primary + {len(ref_summaries)} "
              f"reference cells.", flush=True)
    else:
        print("Dry run only: training matrix NOT launched (as required).")


if __name__ == "__main__":
    main()
