"""Plan launchers dry-run without training or downloading."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import dry_run_plans


def test_primary_dataset_ids_are_the_three_locomotion_suites():
    assert dry_run_plans.PRIMARY_DATASETS == (
        "halfcheetah-medium-v2",
        "hopper-medium-v2",
        "walker2d-medium-v2",
    )


def test_optional_registry_ids_are_the_medium_replay_files():
    assert dry_run_plans.optional_registry_datasets() == (
        "halfcheetah-medium-replay-v2",
        "hopper-medium-replay-v2",
        "walker2d-medium-replay-v2",
    )
    assert set(dry_run_plans.PRIMARY_DATASETS).isdisjoint(dry_run_plans.optional_registry_datasets())


def test_launchers_dry_run_without_training():
    captured = dry_run_plans.launch_dry_runs()
    assert "Primary plan: 135 cells" in captured["primary"]
    assert "Reference plan: 15 cells" in captured["primary"]
    assert "24-CELL STRESS STUDY PLAN" in captured["stress"]
    assert "STRESS SEED EXTENSION (not yet run)" in captured["stress_extension"]
    assert "125" in captured["ablations"]
    assert "Smoke ok:" in captured["ablation_smoke"]
