"""Published dry-run matrices stay 150 confirmatory runs and 24 stress runs."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import final_factorial
import stress_factorial


def test_final_dry_run_plan_is_135_primary_plus_15_reference():
    primary = final_factorial.build_primary_plan(device="cpu")
    reference = final_factorial.build_reference_plan(device="cpu")
    assert len(primary) == 135
    assert len(reference) == 15
    assert len(primary) + len(reference) == 150
    assert final_factorial.CONFIRM_TOKEN == "CONFIRM-FINAL-150"


def test_stress_dry_run_plan_is_24_cells_and_an_extension_is_not():
    plan = stress_factorial.build_stress_plan()
    assert len(plan) == 24
    assert len(plan) == (
        len(stress_factorial.STRESS_ENVS)
        * len(stress_factorial.STRESS_CONTROLLERS)
        * len(stress_factorial.STRESS_SEEDS)
    )
    identities = {(cell["dataset_id"], cell["controller"], cell["seed"]) for cell in plan}
    assert len(identities) == 24
    assert {cell["dataset_id"] for cell in plan} == set(stress_factorial.STRESS_ENVS)
    assert {cell["controller"] for cell in plan} == {"none", "fixed", "redo", "capacity_gate"}
    assert {cell["seed"] for cell in plan} == {0, 1, 2}
    assert {cell["shift"] for cell in plan} == {"actuator_cripple"}
    assert {cell["severity"] for cell in plan} == {1.0}
    extension = stress_factorial.build_stress_plan(seeds=(3, 4))
    assert len(extension) == 16
    assert len(extension) != 24
    assert stress_factorial.CONFIRM_STRESS_TOKEN == "CONFIRM-STRESS-24"
    assert stress_factorial.CONFIRM_STRESS_EXT_TOKEN == "CONFIRM-STRESS-EXT"
