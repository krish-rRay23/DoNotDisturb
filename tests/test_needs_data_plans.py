"""Needs-data items have a not-yet-run plan and a line in the run guide."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import pytest

from adaptive_plasticity.ablations import load_all_studies
from adaptive_plasticity.final_runner import PRIMARY_CONTROLLERS
from adaptive_plasticity.pending_plans import (
    PendingPlanRefused,
    expand_pending_plan,
    load_pending_plans,
    refuse_pending_plan,
)
from adaptive_plasticity.placeholders import load_placeholders

import dry_run_plans
import stress_factorial


FORBIDDEN = ("0.9915", "1.6501", "40.37")


def test_every_needs_data_phrase_has_one_plan_and_is_listed():
    review = (REPO_ROOT / "docs" / "reviewer_risk_review.md").read_text(encoding="utf-8")
    needs = review.split("## needs-data", 1)[1]
    guide = (REPO_ROOT / "docs" / "run_guide.md").read_text(encoding="utf-8")
    plans = load_pending_plans(REPO_ROOT)
    assert len(plans) == 10
    for name, document in plans.items():
        phrase = document["reviewer_phrase"]
        assert phrase in needs
        assert f"configs/pending/{name}.yaml" in guide
        assert document["status"] == "not_yet_run"
        assert document["executable"] is False
    for token in FORBIDDEN:
        assert token not in guide
        for document in plans.values():
            assert token not in document["why_not_run"]


def test_new_grids_are_outside_the_published_plans():
    plans = load_pending_plans(REPO_ROOT)
    extra_seeds = expand_pending_plan(plans["more_confirmatory_seeds"])
    assert len(extra_seeds) == 3 * 3 * 3 * 3
    assert {row["seed"] for row in extra_seeds} == {5, 6, 7}
    hopper = expand_pending_plan(plans["hopper_actuator_stress"])
    assert len(hopper) == 4 * 3
    assert {row["dataset_id"] for row in hopper} == {"hopper-medium-v2"}
    assert "hopper-medium-v2" not in stress_factorial.STRESS_ENVS
    resets = expand_pending_plan(plans["primary_reset_arms"])
    assert len(resets) == 3 * 3 * 2 * 5
    assert {row["controller"] for row in resets} == {"redo", "redo_periodic"}
    assert set(PRIMARY_CONTROLLERS).isdisjoint({row["controller"] for row in resets})
    for name in (
        "multiseed_capacitygate_stress",
        "unapplied_revision_figures",
        "broader_domains",
        "external_learners",
        "gate_threshold_sweep",
        "per_environment_signed_rank",
        "primary_gate_firing",
    ):
        assert expand_pending_plan(plans[name]) == []
    assert len(load_all_studies(REPO_ROOT)) == 125
    placeholders = load_placeholders(REPO_ROOT)
    for name in (
        "halfcheetah_medium_replay",
        "hopper_medium_replay",
        "walker2d_medium_replay",
        "halfcheetah_expert",
        "hopper_expert",
        "walker2d_expert",
        "manipulation",
        "vision",
        "antmaze_umaze",
    ):
        assert placeholders[name]["executable"] is False


def test_pending_plans_refuse_to_launch_and_the_dry_run_prints_them():
    with pytest.raises(PendingPlanRefused, match="not_yet_run"):
        refuse_pending_plan("hopper_actuator_stress", REPO_ROOT)
    text = dry_run_plans.describe_pending_plans()
    assert "pending hopper_actuator_stress: kind=grid cells=12" in text
    assert "pending unapplied_revision_figures: kind=blocked cells=0" in text
    assert "Pending plans are not_yet_run. No training was started." in text
