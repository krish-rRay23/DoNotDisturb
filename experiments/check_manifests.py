"""Read-only sha256 check for every registered dataset.

Does not download and does not rewrite a manifest. A missing raw file is
reported as absent and does not fail the default check. A present file
whose digest differs, or whose manifest or sha256 is missing, exits 1.
Only a file that is actually absent is exempt. ``--require-present`` also
exits 2 when any registered file is absent. The default exit code stays 0
when files are simply not checked out, so ``make check`` can run on a
fresh clone.

Example
-------
python experiments/check_manifests.py
python experiments/check_manifests.py --require-present
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adaptive_plasticity.datasets import check_manifest_sha256, registry


def dataset_state(report: dict) -> str:
    """Classify one hash report.

    ``absent`` is only a raw file that is missing or empty. A file that is
    on disk without a manifest sha256 is ``unverified``, not absent.
    """
    if not report.get("raw_present"):
        return "absent"
    if report.get("match") is True:
        return "match"
    if report.get("match") is False:
        return "mismatch"
    return "unverified"


def summarize_hashes(root: str | Path = ".", *, require_present: bool = False) -> dict:
    """Compare each registered file with its manifest. Never writes."""
    reports = [check_manifest_sha256(dataset_id, root) for dataset_id in registry()]
    states = {report["canonical_id"]: dataset_state(report) for report in reports}
    mismatches = [dataset_id for dataset_id, state in states.items() if state == "mismatch"]
    unverified = [dataset_id for dataset_id, state in states.items() if state == "unverified"]
    absent = [dataset_id for dataset_id, state in states.items() if state == "absent"]
    status = 0
    if mismatches or unverified:
        status = 1
    elif require_present and absent:
        status = 2
    return {
        "banner": "Read-only manifest hash check. No download and no rewrite.",
        "reports": reports,
        "states": states,
        "mismatches": mismatches,
        "unverified": unverified,
        "absent": absent,
        "status": status,
        "require_present": require_present,
    }


def format_report(summary: dict) -> str:
    """One line per dataset, then the banner."""
    lines = [summary["banner"]]
    states = summary.get("states")
    for report in summary["reports"]:
        state = states[report["canonical_id"]] if states else dataset_state(report)
        lines.append(f"{report['canonical_id']}: {state}")
    if summary["mismatches"] or summary["unverified"]:
        lines.append("A present file does not match a manifest sha256, or has no sha256 to compare.")
    elif summary["absent"] and summary["require_present"]:
        lines.append("A registered raw file is absent.")
    elif summary["absent"]:
        lines.append("Absent files were not treated as failures. Pass --require-present to require them.")
    else:
        lines.append("Every registered raw file matches its manifest sha256.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Check registered dataset sha256 values without writing.")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="Project root containing data/.")
    parser.add_argument(
        "--require-present",
        action="store_true",
        help="Exit 2 when any registered raw file is absent.",
    )
    args = parser.parse_args(argv)
    summary = summarize_hashes(args.root, require_present=args.require_present)
    print(format_report(summary), end="")
    if summary["status"]:
        raise SystemExit(summary["status"])


if __name__ == "__main__":
    main()
