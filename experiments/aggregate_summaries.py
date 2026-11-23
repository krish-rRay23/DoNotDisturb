"""Write a per-cell table and IQM intervals from summary.json files.

The output is computed from those files. It is not a manuscript table.
Missing cells are left out rather than filled.

Example
-------
python experiments/aggregate_summaries.py --indir results/ablation_pilot --outdir results/aggregate_check --replicates 200
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adaptive_plasticity.summary_stats import iqm_by_controller, load_summaries, per_cell_rows


def _filter_seeds(rows: list[dict], seeds: tuple[int, ...] | None) -> list[dict]:
    if seeds is None:
        return rows
    allowed = set(seeds)
    return [row for row in rows if row.get("seed") in allowed]


def write_report(
    indir: str | Path,
    outdir: str | Path,
    *,
    seeds: tuple[int, ...] | None = None,
    replicates: int = 2000,
    seed: int = 0,
) -> dict:
    rows = _filter_seeds(per_cell_rows(load_summaries(indir)), seeds)
    by_controller = iqm_by_controller(rows, replicates=replicates, seed=seed)
    destination = Path(outdir)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        "banner": "Computed from the supplied summary.json files. Not a manuscript result.",
        "indir": str(indir),
        "seeds": None if seeds is None else list(seeds),
        "n_rows": len(rows),
        "n_scored_rows": sum(row["score"] is not None for row in rows),
        "per_cell": rows,
        "iqm_by_controller": by_controller,
    }
    (destination / "aggregate.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (destination / "per_cell.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset_id",
                "controller",
                "shift",
                "seed",
                "score",
                "applied_intervention_steps",
                "probe_source",
                "gate_operator",
                "duplicate_identity",
                "manuscript_status",
                "summary_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Aggregate summary.json files. Does not invent scores.")
    parser.add_argument("--indir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seeds", default="", help="Optional comma-separated seed filter.")
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    seeds = tuple(int(part) for part in args.seeds.split(",") if part.strip()) or None
    payload = write_report(
        args.indir,
        args.outdir,
        seeds=seeds,
        replicates=args.replicates,
        seed=args.seed,
    )
    print(payload["banner"])
    print(f"Rows: {payload['n_rows']}. Scored rows: {payload['n_scored_rows']}.")
    if not payload["iqm_by_controller"]:
        print("No numeric scores. No IQM was computed.")
        return
    for controller, stats in payload["iqm_by_controller"].items():
        print(
            f"{controller}: IQM {stats['iqm']:.4f} "
            f"[{stats['ci_low']:.4f}, {stats['ci_high']:.4f}] "
            f"n={stats['n_scores']} (computed from these logs only)"
        )


if __name__ == "__main__":
    main()
