"""Launch CapacityGate ablations.

Default mode prints the plan and does not train. Every planned cell is
``not_yet_run`` until this script actually finishes it. Pilot budgets are a
launch check of a few thousand environment steps. They are not manuscript
results.

Examples
--------
python experiments/run_ablations.py
python experiments/run_ablations.py --smoke
python experiments/run_ablations.py --pilot --execute --confirm-matrix CONFIRM-ABLATION-PILOT --device cpu
python experiments/run_ablations.py --probe-pilot --execute --confirm-matrix CONFIRM-ABLATION-PILOT --device cpu
python experiments/run_ablations.py --execute --confirm-matrix CONFIRM-ABLATION-FULL --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adaptive_plasticity.ablations import (
    FULL_CONFIRM_TOKEN,
    PILOT_CONFIRM_TOKEN,
    ablation_output_name,
    apply_pilot_budget,
    load_all_studies,
    select_cells,
    select_probe_pilot,
)


def _smoke(cells: list[dict]) -> None:
    import numpy as np
    import torch

    from adaptive_plasticity.capacity_gate import CapacityGateConfig
    from adaptive_plasticity.probes import sample_capacity_probe

    for cell in cells:
        CapacityGateConfig(
            dorm_on=float(cell["dorm_on"]),
            dorm_off=float(cell["dorm_off"]),
            rho_on=float(cell["rho_on"]),
            rho_off=float(cell["rho_off"]),
            active_severity=float(cell["active_severity"]),
            cooldown_steps=int(cell["cooldown_steps"]),
        )
    observations = np.arange(300 * 4, dtype=np.float32).reshape(300, 4)
    offline = torch.zeros((16, 4), dtype=torch.float32)
    rng = np.random.RandomState(0)
    for source, window in (
        ("offline_fixed", 1024),
        ("online_buffer", 1024),
        ("recent_window", 32),
    ):
        probe, meta = sample_capacity_probe(
            source=source,
            offline_probe=offline,
            online_observations=observations,
            online_count=300,
            write_idx=300,
            max_size=10_000,
            probe_size=16,
            recent_window=window,
            rng=rng,
            device="cpu",
        )
        if probe.shape[0] != 16 or meta["probe_source"] != source:
            raise RuntimeError(f"Smoke probe failed for {source}: {meta}")
    print(f"Smoke ok: {len(cells)} planned cells validated; three probe sources sampled.")


def _execute(cells: list[dict], *, outdir: Path, device: str, pilot: bool) -> None:
    from adaptive_plasticity.capacity_gate import CapacityGateConfig
    from adaptive_plasticity.final_runner import run_final_cell

    outdir.mkdir(parents=True, exist_ok=True)
    for index, cell in enumerate(cells, start=1):
        name = ablation_output_name(cell, pilot=pilot)
        print(f"[{index}/{len(cells)}] {name}", flush=True)
        gate_config = CapacityGateConfig(
            dorm_on=float(cell["dorm_on"]),
            dorm_off=float(cell["dorm_off"]),
            rho_on=float(cell["rho_on"]),
            rho_off=float(cell["rho_off"]),
            active_severity=float(cell["active_severity"]),
            cooldown_steps=int(cell["cooldown_steps"]),
        )
        summary = run_final_cell(
            dataset_id=cell["dataset_id"],
            controller=cell["controller"],
            shift=cell["shift"],
            seed=int(cell["seed"]),
            severity=float(cell["severity"]),
            shift_step=int(cell["shift_step"]),
            online_steps=int(cell["online_steps"]),
            warmup_updates=int(cell["warmup_updates"]),
            update_freq=1,
            gate_interval=int(cell["gate_interval"]),
            online_ratio=float(cell["online_ratio"]),
            eval_interval=int(cell["eval_interval"]),
            eval_episodes=int(cell["eval_episodes"]),
            device=device,
            outdir=outdir,
            gate_config=gate_config,
            deterministic=True,
            use_shared_offline_checkpoints=not pilot,
            output_name=name,
            probe_source=cell["probe_source"],
            recent_window=int(cell["recent_window"]),
            probe_size=int(cell["probe_size"]),
            gate_operator=cell["gate_operator"],
        )
        print(
            f"  finished {name}: interventions={summary.get('applied_intervention_steps')} "
            f"probe_source={summary.get('probe_source')} status=pilot_finished"
            if pilot
            else f"  finished {name}: interventions={summary.get('applied_intervention_steps')}",
            flush=True,
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="CapacityGate ablation plan and launcher")
    parser.add_argument("--study", default=None, help="One study name. Default: all studies.")
    parser.add_argument("--pilot", action="store_true", help="Restrict to the short mixed pilot cells.")
    parser.add_argument(
        "--probe-pilot",
        action="store_true",
        help="Four hopper seed-0 cells covering every probe source, at the pilot budget.",
    )
    parser.add_argument("--execute", action="store_true", help="Train the selected cells.")
    parser.add_argument("--confirm-matrix", default="", help="Required confirmation token for --execute.")
    parser.add_argument("--smoke", action="store_true", help="Validate configs and sample synthetic probes.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--write-plan", default=None, help="Optional path for the JSON plan.")
    args = parser.parse_args(argv)

    if args.pilot and args.probe_pilot:
        print("ERROR: pass only one of --pilot and --probe-pilot.", file=sys.stderr)
        sys.exit(2)
    studies = load_all_studies(REPO_ROOT)
    short_run = bool(args.pilot or args.probe_pilot)
    if args.probe_pilot:
        cells = select_probe_pilot(studies)
    else:
        cells = select_cells(studies, pilot=args.pilot, study=args.study)
        if args.pilot:
            cells = [apply_pilot_budget(cell) for cell in cells]
    label = "Probe pilot" if args.probe_pilot else "Pilot" if args.pilot else "Full"
    print(
        f"{label} ablation plan: {len(cells)} cells. "
        "Status of every planned cell before execution: not_yet_run."
    )
    for cell in cells:
        print(
            f"  {ablation_output_name(cell, pilot=short_run)} "
            f"probe={cell['probe_source']} operator={cell['gate_operator']} "
            f"rho_on={cell['rho_on']} dorm_on={cell['dorm_on']} "
            f"cooldown={cell['cooldown_steps']} steps={cell['online_steps']}"
        )
    if args.write_plan:
        destination = Path(args.write_plan)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": "not_yet_run" if not args.execute else "execution_requested",
            "pilot": bool(short_run),
            "n_cells": len(cells),
            "cells": cells,
        }
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote plan to {destination}")

    if args.smoke:
        _smoke(cells)

    if not args.execute:
        print("Dry run only. Pass --execute and the matching --confirm-matrix token to train.")
        return

    expected = PILOT_CONFIRM_TOKEN if short_run else FULL_CONFIRM_TOKEN
    if args.confirm_matrix != expected:
        print(f"ERROR: --execute requires --confirm-matrix {expected}", file=sys.stderr)
        sys.exit(2)
    outdir = Path(args.outdir or ("results/ablation_pilot" if short_run else "results/ablations"))
    print(f"Executing into {outdir} on device {args.device}.")
    print("These runs are not manuscript results.")
    _execute(cells, outdir=outdir, device=args.device, pilot=short_run)


if __name__ == "__main__":
    main()
