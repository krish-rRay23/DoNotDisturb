"""Ablation plans stay labeled not-yet-run and match the published defaults."""

from __future__ import annotations

from adaptive_plasticity.ablations import (
    FULL_CONFIRM_TOKEN,
    PILOT_BUDGET,
    PILOT_CONFIRM_TOKEN,
    apply_pilot_budget,
    load_all_studies,
    select_cells,
)


def test_plan_is_not_yet_run_and_has_unique_cells():
    cells = load_all_studies()
    assert len(cells) == 125
    assert {cell["status"] for cell in cells} == {"not_yet_run"}
    names = {
        f"{cell['study']}-{cell['cell_id']}-{cell['dataset_id']}-seed{cell['seed']}"
        for cell in cells
    }
    assert len(names) == 125


def test_published_reference_cell_matches_manuscript_defaults():
    cells = load_all_studies()
    reference = [
        cell
        for cell in cells
        if cell["study"] == "probe_distributions"
        and cell["cell_id"] == "online_buffer"
        and cell["dataset_id"] == "hopper-medium-v2"
        and cell["seed"] == 0
    ]
    assert len(reference) == 1
    cell = reference[0]
    assert cell["probe_source"] == "online_buffer"
    assert cell["gate_operator"] == "shrink_perturb"
    assert cell["rho_on"] == 0.70
    assert cell["rho_off"] == 0.85
    assert cell["dorm_on"] == 0.15
    assert cell["dorm_off"] == 0.08
    assert cell["cooldown_steps"] == 3000
    assert cell["probe_size"] == 256
    assert cell["online_steps"] == 25000


def test_probe_study_covers_three_sources_on_three_environments():
    cells = [cell for cell in load_all_studies() if cell["study"] == "probe_distributions"]
    sources = {cell["probe_source"] for cell in cells}
    datasets = {cell["dataset_id"] for cell in cells}
    assert sources == {"offline_fixed", "online_buffer", "recent_window"}
    assert datasets == {
        "halfcheetah-medium-v2",
        "hopper-medium-v2",
        "walker2d-medium-v2",
    }
    windows = {cell["recent_window"] for cell in cells if cell["probe_source"] == "recent_window"}
    assert windows == {256, 1024}


def test_pilot_is_short_and_covers_each_new_path():
    pilot = [apply_pilot_budget(cell) for cell in select_cells(load_all_studies(), pilot=True)]
    assert len(pilot) == 7
    assert {cell["online_steps"] for cell in pilot} == {PILOT_BUDGET["online_steps"]}
    assert PILOT_BUDGET["online_steps"] <= 2000
    assert {cell["dataset_id"] for cell in pilot} == {"hopper-medium-v2"}
    ids = {(cell["study"], cell["cell_id"]) for cell in pilot}
    assert ("probe_distributions", "offline_fixed") in ids
    assert ("probe_distributions", "recent_window_1024") in ids
    assert ("probe_distributions", "recent_window_256") in ids
    assert ("gate_thresholds", "rho_on_0.95") in ids
    assert ("gate_thresholds", "dorm_on_0.05") in ids
    assert ("cooldown", "cooldown_0") in ids
    assert ("gate_operator", "redo") in ids
    assert PILOT_CONFIRM_TOKEN != FULL_CONFIRM_TOKEN
