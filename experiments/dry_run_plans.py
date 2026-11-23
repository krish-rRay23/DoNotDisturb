"""Dry-run every published plan launcher and report which raw datasets exist.

This does not train and does not download. Raw HDF5 files stay gitignored.
Acquire a missing file with:

    python -m adaptive_plasticity.datasets --acquire <id>

Acquisition rewrites ``data/manifests/<id>.json`` even when the bytes were
already present. If the sha256 is unchanged, restore the tracked manifest
with ``git checkout -- data/manifests/<id>.json`` so a download does not
become a commit.
"""

from __future__ import annotations

import argparse
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from adaptive_plasticity.datasets import missing_dataset_ids, registry

PRIMARY_DATASETS: tuple[str, ...] = (
    "halfcheetah-medium-v2",
    "hopper-medium-v2",
    "walker2d-medium-v2",
)


def optional_registry_datasets() -> tuple[str, ...]:
    """Registry ids the published runners do not load.

    These are the medium-replay locomotion files. ``antmaze-umaze-v2`` is
    not in the registry, so it cannot appear here and cannot be acquired.
    """
    return tuple(dataset_id for dataset_id in registry() if dataset_id not in PRIMARY_DATASETS)


def missing_primary_datasets(root: str | Path = ".") -> list[str]:
    """Return primary dataset ids whose raw file is absent or empty."""
    return missing_dataset_ids(PRIMARY_DATASETS, root)


def _capture(fn, argv: list[str]) -> str:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        fn(argv)
    return buffer.getvalue()


def launch_dry_runs() -> dict[str, str]:
    """Call each launcher without ``--execute``. Returns captured stdout."""
    import analyze_stress_coverage
    import final_factorial
    import run_ablations
    import stress_factorial

    runs = {
        "primary": _capture(final_factorial.main, ["--device", "cpu"]),
        "stress": _capture(stress_factorial.main, ["--device", "cpu", "--seeds", "0,1,2"]),
        "stress_extension": _capture(
            stress_factorial.main, ["--device", "cpu", "--seeds", "3,4", "--outdir", "results/stress_study_extension"]
        ),
        "stable": _capture(stress_factorial.main, ["--stable-validate", "--device", "cpu"]),
        "ablations": _capture(run_ablations.main, []),
        "ablation_smoke": _capture(run_ablations.main, ["--smoke"]),
        "stress_coverage": _capture(analyze_stress_coverage.main, ["--seeds", "0,1,2,3,4"]),
    }
    expected_fragments = {
        "primary": "Dry run only: training matrix NOT launched",
        "stress": "Dry run only: execution not launched",
        "stress_extension": "STRESS SEED EXTENSION (not yet run)",
        "stable": "STABLE VALIDATION PLAN (9 cells)",
        "ablations": "Dry run only",
        "ablation_smoke": "Smoke ok:",
        "stress_coverage": "not_yet_run",
    }
    for name, fragment in expected_fragments.items():
        if fragment not in runs[name]:
            raise RuntimeError(f"{name} dry-run did not contain {fragment!r}.\n{runs[name][-500:]}")
    return runs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Dry-run plan launchers and check raw datasets")
    parser.add_argument(
        "--plans-only",
        action="store_true",
        help="Skip the dataset-file check. Use this when the HDF5 files are not on disk.",
    )
    parser.add_argument(
        "--require-registry",
        action="store_true",
        help="Exit 2 if any registered raw file is missing, including medium-replay.",
    )
    args = parser.parse_args(argv)
    launch_dry_runs()
    print("Dry-run launchers exited without starting training.")
    if args.plans_only:
        print("Dataset check skipped (--plans-only).")
        return
    missing = missing_primary_datasets(REPO_ROOT)
    optional = optional_registry_datasets()
    optional_missing = missing_dataset_ids(optional, REPO_ROOT)
    for dataset_id in PRIMARY_DATASETS:
        state = "missing" if dataset_id in missing else "present"
        print(f"dataset {dataset_id}: {state}")
    for dataset_id in optional:
        state = "missing" if dataset_id in optional_missing else "present"
        print(f"dataset {dataset_id}: {state} (not required by the primary plan)")
    print("antmaze-umaze-v2 is not in the dataset registry and is not acquired.")
    if missing or (args.require_registry and optional_missing):
        needed = list(missing)
        if args.require_registry:
            needed.extend(dataset_id for dataset_id in optional_missing if dataset_id not in needed)
        joined = " ".join(needed)
        print(
            "Missing raw files. Acquire each one, then restore an unchanged manifest:\n"
            f"  python -m adaptive_plasticity.datasets --acquire {joined}\n"
            "  git checkout -- data/manifests/<id>.json",
            file=sys.stderr,
        )
        sys.exit(2)
    if optional_missing:
        print(
            "Optional medium-replay files are absent. The published runners do not load them. "
            "No training was started."
        )
        return
    print("Registered raw datasets are present and gitignored. No training was started.")


if __name__ == "__main__":
    main()
