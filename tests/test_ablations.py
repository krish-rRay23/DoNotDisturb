"""Ablation plans stay labeled not-yet-run and match the published defaults."""

from __future__ import annotations

from adaptive_plasticity.ablations import (
    FULL_CONFIRM_TOKEN,
    PILOT_BUDGET,
    PILOT_CONFIRM_TOKEN,
    apply_pilot_budget,
    load_all_studies,
    select_cells,
    select_probe_pilot,
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


PUBLISHED_FIXED = {
    "dataset_id": "hopper-medium-v2",
    "controller": "capacity_gate",
    "shift": "none",
    "severity": 0.0,
    "shift_step": 5000,
    "online_steps": 25000,
    "warmup_updates": 25000,
    "online_ratio": 0.5,
    "gate_interval": 1000,
    "eval_interval": 2500,
    "eval_episodes": 10,
    "probe_source": "online_buffer",
    "probe_size": 256,
    "recent_window": 1024,
    "gate_operator": "shrink_perturb",
    "rho_on": 0.70,
    "rho_off": 0.85,
    "dorm_on": 0.15,
    "dorm_off": 0.08,
    "cooldown_steps": 3000,
    "active_severity": 0.1,
    "status": "not_yet_run",
}


def _assert_published_except(cell: dict, varied: set[str]) -> None:
    for key, expected in PUBLISHED_FIXED.items():
        if key in varied:
            continue
        assert cell[key] == expected, f"{cell['study']}:{cell['cell_id']} seed{cell['seed']} {key}"


def test_threshold_cooldown_and_operator_grids_stay_one_factor():
    cells = load_all_studies()
    by_study: dict[str, list[dict]] = {}
    for cell in cells:
        by_study.setdefault(cell["study"], []).append(cell)
    assert {study: len(rows) for study, rows in by_study.items()} == {
        "probe_distributions": 60,
        "gate_thresholds": 35,
        "cooldown": 20,
        "gate_operator": 10,
    }

    thresholds = [cell for cell in by_study["gate_thresholds"] if cell["seed"] == 0]
    assert {cell["dataset_id"] for cell in by_study["gate_thresholds"]} == {"hopper-medium-v2"}
    by_id = {cell["cell_id"]: cell for cell in thresholds}
    assert set(by_id) == {
        "published_thresholds",
        "rho_on_0.50",
        "rho_on_0.85",
        "rho_on_0.95",
        "dorm_on_0.05",
        "dorm_on_0.08",
        "dorm_on_0.25",
    }
    assert (by_id["published_thresholds"]["rho_on"], by_id["published_thresholds"]["rho_off"]) == (0.70, 0.85)
    assert (by_id["rho_on_0.50"]["rho_on"], by_id["rho_on_0.50"]["rho_off"]) == (0.50, 0.85)
    assert (by_id["rho_on_0.85"]["rho_on"], by_id["rho_on_0.85"]["rho_off"]) == (0.85, 0.90)
    assert (by_id["rho_on_0.95"]["rho_on"], by_id["rho_on_0.95"]["rho_off"]) == (0.95, 0.97)
    assert (by_id["dorm_on_0.05"]["dorm_on"], by_id["dorm_on_0.05"]["dorm_off"]) == (0.05, 0.05)
    assert (by_id["dorm_on_0.08"]["dorm_on"], by_id["dorm_on_0.08"]["dorm_off"]) == (0.08, 0.05)
    assert (by_id["dorm_on_0.25"]["dorm_on"], by_id["dorm_on_0.25"]["dorm_off"]) == (0.25, 0.08)
    for cell in by_study["gate_thresholds"]:
        _assert_published_except(cell, {"rho_on", "rho_off", "dorm_on", "dorm_off"})
        assert 0.0 <= cell["dorm_off"] <= cell["dorm_on"] <= 1.0
        assert 0.0 < cell["rho_on"] <= cell["rho_off"]

    cooldowns = [cell for cell in by_study["cooldown"] if cell["seed"] == 0]
    assert {cell["cooldown_steps"] for cell in cooldowns} == {0, 1000, 3000, 10000}
    for cell in by_study["cooldown"]:
        _assert_published_except(cell, {"cooldown_steps"})

    operators = [cell for cell in by_study["gate_operator"] if cell["seed"] == 0]
    assert {cell["cell_id"]: cell["gate_operator"] for cell in operators} == {
        "shrink_perturb": "shrink_perturb",
        "redo": "redo",
    }
    for cell in by_study["gate_operator"]:
        _assert_published_except(cell, {"gate_operator"})


def test_probe_pilot_covers_every_source_at_the_short_budget():
    cells = select_probe_pilot(load_all_studies())
    assert len(cells) == 4
    assert {cell["probe_source"] for cell in cells} == {
        "offline_fixed",
        "online_buffer",
        "recent_window",
    }
    assert {cell["cell_id"] for cell in cells} == {
        "offline_fixed",
        "online_buffer",
        "recent_window_1024",
        "recent_window_256",
    }
    assert {cell["dataset_id"] for cell in cells} == {"hopper-medium-v2"}
    assert {cell["seed"] for cell in cells} == {0}
    assert {cell["online_steps"] for cell in cells} == {PILOT_BUDGET["online_steps"]}
    assert {cell["status"] for cell in cells} == {"not_yet_run"}
