"""Stress coverage reports manuscript cells and leaves extensions not-yet-run."""

from __future__ import annotations

import json
from pathlib import Path

from adaptive_plasticity.stress_reports import (
    MANUSCRIPT_STRESS,
    coverage_report,
    extension_cells,
    load_summaries,
    render_coverage_report,
)


def test_manuscript_table_reports_capacity_gate_seed_zero_only():
    assert set(MANUSCRIPT_STRESS[("walker2d-medium-v2", "capacity_gate")]) == {0}
    assert MANUSCRIPT_STRESS[("walker2d-medium-v2", "capacity_gate")][0] == 7.23
    assert MANUSCRIPT_STRESS[("halfcheetah-medium-v2", "fixed")][0] == 15.29
    cells = extension_cells((0, 1, 2, 3, 4))
    unpublished = [cell for cell in cells if not cell["in_manuscript"]]
    assert unpublished
    assert {cell["status"] for cell in unpublished} == {"not_yet_run"}
    # 2 envs x 4 controllers x 5 seeds, minus the two CapacityGate seeds the
    # manuscript omits inside {0,1,2} and the two CapacityGate seeds 3 and 4
    # on each env: reported cells are 2*3*3 + 2*1 = 20.
    assert sum(cell["in_manuscript"] for cell in cells) == 20


def test_missing_directory_does_not_invent_scores(tmp_path: Path):
    report = coverage_report(load_summaries(tmp_path / "absent"), seeds=(0, 1))
    text = render_coverage_report(report)
    assert "not_yet_run" in text
    assert report["n_local_summaries"] == 0
    assert "8.58" in text


def test_local_summary_is_not_rewritten_as_manuscript(tmp_path: Path):
    run = tmp_path / "STRESS-halfcheetah-medium-v2-capacity_gate-actuator_cripple-sev1-seed1"
    run.mkdir()
    (run / "summary.json").write_text(
        json.dumps(
            {
                "dataset_id": "halfcheetah-medium-v2",
                "controller": "capacity_gate",
                "shift": "actuator_cripple",
                "seed": 1,
                "normalized": [1.0, 2.5],
            }
        ),
        encoding="utf-8",
    )
    report = coverage_report(load_summaries(tmp_path), seeds=(0, 1))
    row = next(
        item
        for item in report["rows"]
        if item["dataset_id"] == "halfcheetah-medium-v2"
        and item["controller"] == "capacity_gate"
        and item["seed"] == 1
    )
    assert row["local_score"] == 2.5
    assert row["manuscript_score"] is None
    assert row["status"] == "local_summary_present"
    assert row["in_manuscript"] is False


def test_non_stress_summary_does_not_match_the_actuator_table():
    ordinary = {
        "dataset_id": "halfcheetah-medium-v2",
        "controller": "capacity_gate",
        "shift": "none",
        "seed": 0,
        "normalized": [99.0],
    }
    stress = {
        "dataset_id": "halfcheetah-medium-v2",
        "controller": "capacity_gate",
        "shift": "actuator_cripple",
        "seed": 0,
        "normalized": [8.58],
    }
    # Later non-stress file must not overwrite the actuator summary.
    report = coverage_report([stress, ordinary], seeds=(0,))
    row = next(
        item
        for item in report["rows"]
        if item["dataset_id"] == "halfcheetah-medium-v2"
        and item["controller"] == "capacity_gate"
        and item["seed"] == 0
    )
    assert row["local_score"] == 8.58
    assert row["matches_manuscript_rounding"] is True

    only_ordinary = coverage_report([ordinary], seeds=(0,))
    ordinary_row = next(
        item
        for item in only_ordinary["rows"]
        if item["dataset_id"] == "halfcheetah-medium-v2"
        and item["controller"] == "capacity_gate"
        and item["seed"] == 0
    )
    assert ordinary_row["local_score"] is None
    assert ordinary_row["present"] is False
