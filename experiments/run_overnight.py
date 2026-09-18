"""Master Overnight Execution Orchestrator for FINAL 135-run Study.

Single idempotent command to run each night:
  .venv\Scripts\python experiments\run_overnight.py

Features:
  * Automatically pre-trains and caches the 15 shared offline base checkpoints.
  * Runs the 135-cell primary matrix with 2-worker concurrency.
  * Atomically saves checkpoints and skips completed runs.
  * Graceful Windows signal handling (SIGINT, SIGTERM, SIGBREAK).
  * Automatically triggers final analysis and plot generation once all 135 runs finish.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import final_factorial as ff
from adaptive_plasticity.final_runner import (
    CONFIRM_TOKEN,
    FINAL_ENVS,
    FINAL_SEEDS,
    FINAL_SHIFTS,
    PRIMARY_CONTROLLERS,
    final_output_name,
    run_final_cell,
)
from adaptive_plasticity.offline_cache import (
    DEFAULT_CHECKPOINT_DIR,
    ensure_all_offline_checkpoints,
)

STATUS_FILE = Path("results/FINAL_status.json")


def update_status_file(
    outdir: Path,
    total_cells: int,
    completed_cells: int,
    in_flight_cells: list[str],
    status: str,
) -> None:
    """Atomically record live status for multi-session tracking."""
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "total_cells": total_cells,
        "completed_cells": completed_cells,
        "remaining_cells": total_cells - completed_cells,
        "progress_percent": round((completed_cells / max(1, total_cells)) * 100.0, 1),
        "in_flight": list(in_flight_cells),
    }
    tmp = outdir / "tmp_status.json"
    target = outdir / "FINAL_status.json"
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, target)


def execute_matrix_resilient(
    plan: list[dict],
    outdir: Path,
    workers: int = 2,
    manifest_prefix: str = "FINAL",
) -> tuple[list[dict], int]:
    """Execute the matrix with atomic progress, graceful shutdown, and skip/resume."""
    outdir.mkdir(parents=True, exist_ok=True)
    total = len(plan)
    summaries: list[dict | None] = [None] * total
    pending: list[tuple[int, dict]] = []

    # Check already completed cells
    for idx, cell in enumerate(plan):
        saved = ff.cell_completed(outdir, cell)
        if saved is not None:
            summaries[idx] = saved
        else:
            pending.append((idx, cell))

    already_done = total - len(pending)
    print(f"\n=======================================================", flush=True)
    print(f" MATRIX STATUS: {already_done}/{total} cells completed ({len(pending)} pending)", flush=True)
    print(f"=======================================================\n", flush=True)

    if not pending:
        valid_summaries = [s for s in summaries if s is not None]
        ff.write_manifest(valid_summaries, outdir, prefix=manifest_prefix)
        update_status_file(outdir, total, total, [], "ALL_COMPLETED")
        return valid_summaries, 0

    stop_requested = False

    def _signal_handler(signum, frame):
        nonlocal stop_requested
        print("\n[SHUTDOWN] Signal received. Finishing in-flight cells and exiting cleanly...", flush=True)
        stop_requested = True

    # Register Windows / POSIX termination signals
    try:
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, _signal_handler)
    except Exception:
        pass

    completed_in_session = 0
    in_flight_names: list[str] = []
    first_failure: BaseException | None = None

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="overnight-cell") as pool:
        future_to_pos: dict = {}
        queue = iter(pending)

        def _submit(idx: int, cell: dict) -> None:
            print(f"[{already_done + completed_in_session + len(future_to_pos) + 1}/{total}] "
                  f"launching {cell['output_name']} ...", flush=True)
            fut = pool.submit(run_final_cell, **ff._cell_runner_kwargs(cell, outdir))
            future_to_pos[fut] = (idx, cell)
            in_flight_names.append(cell["output_name"])
            update_status_file(outdir, total, already_done + completed_in_session, in_flight_names, "RUNNING")

        # Initial queue fill
        for _ in range(min(workers, len(pending))):
            if stop_requested:
                break
            idx, cell = next(queue)
            _submit(idx, cell)

        # Worker loop
        while future_to_pos and first_failure is None:
            done, _ = wait(future_to_pos, return_when=FIRST_COMPLETED)
            for fut in done:
                idx, cell = future_to_pos.pop(fut)
                if cell["output_name"] in in_flight_names:
                    in_flight_names.remove(cell["output_name"])
                try:
                    summary = fut.result()
                    summaries[idx] = summary
                    completed_in_session += 1
                    curr_completed = already_done + completed_in_session
                    print(f"[{curr_completed}/{total}] FINISHED {cell['output_name']} "
                          f"({summary.get('runtime_seconds', 0):.1f}s)", flush=True)
                    # Update manifest after every single cell
                    valid = [s for s in summaries if s is not None]
                    ff.write_manifest(valid, outdir, prefix=manifest_prefix)
                    update_status_file(outdir, total, curr_completed, in_flight_names,
                                       "STOPPING" if stop_requested else "RUNNING")
                except Exception as exc:
                    first_failure = exc
                    print(f"[ERROR] Cell failed: {cell['output_name']}: {exc!r}", file=sys.stderr, flush=True)

            if first_failure is None and not stop_requested:
                for _ in range(len(done)):
                    try:
                        idx, cell = next(queue)
                    except StopIteration:
                        break
                    _submit(idx, cell)

        if stop_requested:
            print(f"\n[SESSION COMPLETE] Cleanly stopped. Completed {completed_in_session} cells this session.", flush=True)

    if first_failure is not None:
        raise first_failure

    valid_summaries = [s for s in summaries if s is not None]
    ff.write_manifest(valid_summaries, outdir, prefix=manifest_prefix)
    final_status = "ALL_COMPLETED" if len(valid_summaries) == total else "PARTIAL_SESSION_ENDED"
    update_status_file(outdir, total, len(valid_summaries), [], final_status)
    return valid_summaries, completed_in_session


def run_auto_analysis(outdir: Path) -> None:
    """Run post-run statistical analysis once all 135 runs finish."""
    print("\n=======================================================", flush=True)
    print(" ALL 135 RUNS COMPLETED! RUNNING AUTOMATED FINAL ANALYSIS", flush=True)
    print("=======================================================\n", flush=True)
    try:
        import final_analysis
        audit = final_analysis.audit_plan(outdir)
        print("Completeness audit:", f"{audit['n_primary_complete']}/{audit['n_primary']} primary cells complete")
        if audit.get("audit_status") == "complete" or audit.get("primary_complete"):
            final_analysis.main(["--results-dir", str(outdir)])
            print("\n[SUCCESS] Final analysis generated all figures, tables, and IQM statistics in", outdir)
        else:
            print("[NOTICE] Audit indicated missing runs for full analysis.")
    except Exception as exc:
        print(f"[WARNING] Analysis execution encountered: {exc!r}", file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Master Overnight Runner for FINAL Study.")
    parser.add_argument("--workers", type=int, default=2, help="Concurrency worker count (default 2).")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device (default cuda).")
    parser.add_argument("--outdir", type=str, default="results", help="Results output directory.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=======================================================")
    print(" ADAPTIVE PLASTICITY CONTROL: OVERNIGHT EXECUTION HARNESS")
    print(f" Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f" Device: {args.device} | Workers: {args.workers} | Output: {outdir}")
    print("=======================================================\n")

    # Step 1: Pre-flight & Ensure 15 Shared Offline Base Checkpoints
    ensure_all_offline_checkpoints(
        envs=FINAL_ENVS,
        seeds=FINAL_SEEDS,
        warmup_updates=25_000,
        device=args.device,
        checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
    )

    # Step 2: Build Primary 135-Run Plan
    plan = ff.build_primary_plan(device=args.device)
    plan_file = outdir / "FINAL_plan.json"
    if not plan_file.exists():
        plan_doc = {"protocol": ff.final_protocol(), "cells": plan}
        plan_file.write_text(json.dumps(plan_doc, indent=2, default=str) + "\n", encoding="utf-8")

    # Step 3: Resilient Matrix Execution
    summaries, completed_now = execute_matrix_resilient(
        plan=plan,
        outdir=outdir,
        workers=args.workers,
        manifest_prefix="FINAL",
    )

    # Step 4: If all 135 completed, automatically trigger analysis
    if len(summaries) == len(plan):
        run_auto_analysis(outdir)
    else:
        print(f"\n>>> Progress saved: {len(summaries)}/{len(plan)} cells complete. "
              f"Run this command again in the next session to resume.", flush=True)


if __name__ == "__main__":
    main()
