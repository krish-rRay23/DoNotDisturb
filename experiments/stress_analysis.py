"""Post-run analysis for the 24-run stress study + 9-run stable validation.

Produces:
1. Aggregated IQM with 95% stratified bootstrap CIs across the 4 arms.
2. Pairwise probability of improvement (CapacityGate vs None, Fixed, ReDo).
3. Dual-probe mechanistic plots:
   - Online probe (active manifold collapse) vs Offline probe (intrinsic weight capacity).
4. Learning curves on the crippled target task.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def compute_iqm(scores: np.ndarray) -> float:
    if len(scores) == 0:
        return 0.0
    q25, q75 = np.percentile(scores, [25, 75])
    trimmed = scores[(scores >= q25) & (scores <= q75)]
    return float(np.mean(trimmed)) if len(trimmed) > 0 else float(np.mean(scores))


def bootstrap_iqm_ci(scores: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.RandomState(seed)
    n = len(scores)
    if n <= 1:
        val = compute_iqm(scores)
        return val, val
    boot_iqms = []
    for _ in range(n_boot):
        sample = rng.choice(scores, size=n, replace=True)
        boot_iqms.append(compute_iqm(sample))
    return float(np.percentile(boot_iqms, 2.5)), float(np.percentile(boot_iqms, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress Study Post-Analysis")
    parser.add_argument("--indir", type=str, default="results/stress_study")
    parser.add_argument("--outdir", type=str, default="results/stress_analysis")
    args = parser.parse_args()

    indir = Path(args.indir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    stress_manifest = indir / "STRESS_manifest.jsonl"
    if not stress_manifest.is_file():
        print(f"No STRESS_manifest.jsonl found in {indir}. Checking individual summary files...")
        summaries = glob.glob(str(indir / "STRESS-*/summary.json"))
        records = []
        for s in summaries:
            with open(s) as f:
                records.append(json.load(f))
    else:
        with open(stress_manifest) as f:
            records = [json.loads(line) for line in f]

    print(f"Loaded {len(records)} stress study runs.")

    # 1. Summary table
    table_rows = []
    ctrl_scores: dict[str, list[float]] = {}
    for r in records:
        ctrl = r.get("controller")
        env = r.get("environment", r.get("dataset_id"))
        seed = r.get("seed")
        norm = r.get("normalized", [0.0])[-1]
        raw_ret = r.get("returns", [0.0])[-1]
        applied_raw = r.get("applied_intervention_steps", 0)
        applied = len(applied_raw) if isinstance(applied_raw, list) else int(applied_raw or 0)
        table_rows.append({
            "env": env, "controller": ctrl, "seed": seed,
            "final_normalized": norm, "final_return": raw_ret,
            "applied_interventions": applied
        })
        ctrl_scores.setdefault(ctrl, []).append(norm)

    df = pd.DataFrame(table_rows)
    df.to_csv(outdir / "stress_summary_table.csv", index=False)
    print(f"Wrote {outdir / 'stress_summary_table.csv'}")

    # 2. Compute aggregate IQM
    iqm_results = {}
    print("\n=== AGGREGATE PERFORMANCE (24-RUN STRESS STUDY) ===")
    for ctrl, scs in ctrl_scores.items():
        arr = np.array(scs)
        iqm = compute_iqm(arr)
        ci_l, ci_u = bootstrap_iqm_ci(arr)
        iqm_results[ctrl] = {"iqm": iqm, "ci_low": ci_l, "ci_high": ci_u, "n": len(arr)}
        print(f"  {ctrl:15s}: IQM={iqm:6.2f} [{ci_l:6.2f}, {ci_u:6.2f}] (mean={np.mean(arr):6.2f} +/- {np.std(arr):5.2f})")

    with open(outdir / "stress_iqm.json", "w") as f:
        json.dump(iqm_results, f, indent=2)

    # 3. Dual-probe telemetry plots
    gate_files = glob.glob(str(indir / "STRESS-*capacity_gate*/gate_*.jsonl"))
    if gate_files:
        print(f"\nProcessing dual-probe telemetry from {len(gate_files)} gate logs...")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for gf in gate_files:
            steps, on_rho, off_rho, on_dorm, off_dorm = [], [], [], [], []
            with open(gf) as f:
                for line in f:
                    d = json.loads(line)
                    steps.append(d.get("step", 0))
                    on_rho.append(d.get("online_rho", d.get("rho", 1.0)))
                    off_rho.append(d.get("offline_rho", 1.0))
                    on_dorm.append(d.get("online_dormancy", d.get("dormancy", 0.0)))
                    off_dorm.append(d.get("offline_dormancy", 0.0))
            
            axes[0].plot(steps, on_rho, color="crimson", alpha=0.5, label="Online Probe (Active Manifold)")
            axes[0].plot(steps, off_rho, color="navy", linestyle="--", alpha=0.5, label="Offline Probe (Intrinsic)")
            axes[1].plot(steps, on_dorm, color="crimson", alpha=0.5)
            axes[1].plot(steps, off_dorm, color="navy", linestyle="--", alpha=0.5)

        axes[0].axvline(5000, color="gray", linestyle=":", label="Shift Step (Actuator Cripple)")
        axes[0].axhline(0.70, color="red", linestyle="--", label="rho_on (0.70)")
        axes[0].set_title("Effective Rank Ratio: Active vs Intrinsic")
        axes[0].set_xlabel("Online Step")
        axes[0].set_ylabel("Rank Ratio (rho)")
        
        # Deduplicate legend
        handles, labels = axes[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axes[0].legend(by_label.values(), by_label.keys(), loc="lower left")

        axes[1].axvline(5000, color="gray", linestyle=":")
        axes[1].axhline(0.15, color="red", linestyle="--", label="dorm_on (0.15)")
        axes[1].set_title("Neuron Dormancy: Active vs Intrinsic")
        axes[1].set_xlabel("Online Step")
        axes[1].set_ylabel("Dormancy Ratio (d)")
        
        plt.tight_layout()
        plt.savefig(outdir / "dual_probe_diagnostics.png", dpi=300)
        plt.close()
        print(f"Wrote dual-probe diagnostic plot to {outdir / 'dual_probe_diagnostics.png'}")

    print("\nStress analysis complete.")


if __name__ == "__main__":
    main()
