"""Report which stress-test cells are in the manuscript and which are missing.

This script does not recompute the paper table and does not invent scores.
Local ``summary.json`` files are optional.

Examples
--------
python experiments/analyze_stress_coverage.py
python experiments/analyze_stress_coverage.py --indir results/stress_study --seeds 0,1,2,3,4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adaptive_plasticity.stress_reports import coverage_report, load_summaries, render_coverage_report


def _parse_seeds(text: str) -> tuple[int, ...]:
    seeds = tuple(int(part) for part in text.split(",") if part.strip())
    if not seeds:
        raise SystemExit("Provide at least one seed, for example 0,1,2,3,4.")
    return seeds


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Stress-test coverage against the manuscript table")
    parser.add_argument("--indir", default="results/stress_study", help="Directory of run summaries, if any.")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    args = parser.parse_args(argv)
    seeds = _parse_seeds(args.seeds)
    summaries = load_summaries(REPO_ROOT / args.indir)
    report = coverage_report(summaries, seeds=seeds)
    sys.stdout.write(render_coverage_report(report))
    missing = [row for row in report["rows"] if row["status"] == "not_yet_run"]
    if missing:
        print("Launch the unpublished seed extension (not run by this script) with:")
        print(
            "  python experiments/stress_factorial.py --seeds 3,4 --execute "
            "--confirm-matrix CONFIRM-STRESS-EXT --device cpu "
            "--outdir results/stress_study_extension"
        )
        print(
            "CapacityGate seeds 1 and 2 are in the 24-cell harness but the manuscript "
            "reports seed 0 only. Do not treat a future aggregate over those seeds as "
            "the published table."
        )


if __name__ == "__main__":
    main()
