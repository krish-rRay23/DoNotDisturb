"""Write LaTeX tables from summary.json files already on disk.

The numbers are computed from those files. They are not manuscript tables.
An empty directory writes the banner and no numeric rows.

Example
-------
python experiments/render_latex_tables.py --indir results/ablation_pilot --outdir results/latex_tables
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adaptive_plasticity.latex_tables import write_tables


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render LaTeX tables from supplied summary.json files.")
    parser.add_argument("--indir", type=Path, required=True, help="Directory to search for summary.json files.")
    parser.add_argument("--outdir", type=Path, required=True, help="Directory for the .tex files and values.json.")
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="Mark the output as a synthetic fixture. Also set when a summary has fixture=true.",
    )
    parser.add_argument("--replicates", type=int, default=200, help="Stratified bootstrap replicates for the IQM row.")
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap seed. Not a training seed.")
    args = parser.parse_args(argv)
    rendered = write_tables(
        args.indir,
        args.outdir,
        fixture=args.fixture,
        replicates=args.replicates,
        seed=args.seed,
    )
    print(rendered["values"]["banner"])
    print(f"rows={rendered['values']['n_rows']} fixture={str(rendered['values']['fixture']).lower()}")


if __name__ == "__main__":
    main()
