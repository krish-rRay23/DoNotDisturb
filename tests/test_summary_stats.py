"""IQM and per-cell tables on synthetic summaries. No manuscript numbers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from adaptive_plasticity.summary_stats import (
    iqm,
    iqm_by_controller,
    iqm_with_ci,
    load_summaries,
    paired_comparison,
    per_cell_rows,
)
from aggregate_summaries import write_report
from final_analysis import iqm as final_iqm


def test_iqm_matches_the_final_analysis_definition():
    scores = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    assert iqm(scores) == 4.5
    assert iqm(scores) == final_iqm(scores)
    assert np.isnan(iqm([]))


def test_bootstrap_interval_is_reproducible_and_does_not_invent_empty_data():
    groups = [np.array([1.0, 2.0, 3.0, 4.0]), np.array([10.0, 11.0, 12.0, 13.0])]
    first = iqm_with_ci(groups, replicates=50, seed=1)
    second = iqm_with_ci(groups, replicates=50, seed=1)
    assert first == second
    assert first["n_scores"] == 8
    assert first["source"] == "computed_from_supplied_summaries"
    empty = iqm_with_ci([], replicates=10, seed=0)
    assert np.isnan(empty["iqm"])
    assert empty["n_scores"] == 0


def test_per_cell_table_keeps_duplicates_and_missing_scores(tmp_path: Path):
    run_a = tmp_path / "a"
    run_b = tmp_path / "b"
    run_a.mkdir()
    run_b.mkdir()
    (run_a / "summary.json").write_text(
        json.dumps(
            {
                "dataset_id": "hopper-medium-v2",
                "controller": "capacity_gate",
                "shift": "none",
                "seed": 0,
                "normalized": [1.0, 3.0],
                "applied_intervention_steps": 0,
                "probe_source": "online_buffer",
            }
        ),
        encoding="utf-8",
    )
    (run_b / "summary.json").write_text(
        json.dumps(
            {
                "dataset_id": "hopper-medium-v2",
                "controller": "capacity_gate",
                "shift": "none",
                "seed": 0,
                "normalized": [],
            }
        ),
        encoding="utf-8",
    )
    rows = per_cell_rows(load_summaries(tmp_path))
    assert len(rows) == 2
    assert rows[0]["score"] == 3.0
    assert rows[0]["duplicate_identity"] is False
    assert rows[1]["score"] is None
    assert rows[1]["duplicate_identity"] is True
    assert {row["manuscript_status"] for row in rows} == {"not_a_manuscript_number"}
    report = iqm_by_controller(rows, replicates=20, seed=0)
    assert report["capacity_gate"]["n_scores"] == 1
    payload = write_report(tmp_path, tmp_path / "out", replicates=20, seed=0)
    assert payload["n_rows"] == 2
    assert payload["n_scored_rows"] == 1
    assert "Not a manuscript result" in payload["banner"]
    assert (tmp_path / "out" / "per_cell.csv").is_file()


def test_seed_filter_drops_other_seeds(tmp_path: Path):
    for seed, score in ((0, 1.0), (3, 9.0)):
        run = tmp_path / f"s{seed}"
        run.mkdir()
        (run / "summary.json").write_text(
            json.dumps(
                {
                    "dataset_id": "hopper-medium-v2",
                    "controller": "none",
                    "shift": "actuator_cripple",
                    "seed": seed,
                    "normalized": [score],
                }
            ),
            encoding="utf-8",
        )
    payload = write_report(tmp_path, tmp_path / "out", seeds=(3,), replicates=10, seed=0)
    assert payload["n_rows"] == 1
    assert payload["per_cell"][0]["seed"] == 3
    assert payload["per_cell"][0]["score"] == 9.0


def test_paired_comparison_uses_only_complete_unique_cells():
    rows = [
        {"dataset_id": "hopper-medium-v2", "shift": "none", "seed": 0, "controller": "none", "score": 4.0},
        {"dataset_id": "hopper-medium-v2", "shift": "none", "seed": 1, "controller": "none", "score": 6.0},
        {"dataset_id": "hopper-medium-v2", "shift": "none", "seed": 2, "controller": "none", "score": 8.0},
        {"dataset_id": "hopper-medium-v2", "shift": "none", "seed": 0, "controller": "fixed", "score": 1.0},
        {"dataset_id": "hopper-medium-v2", "shift": "none", "seed": 1, "controller": "fixed", "score": 2.0},
        {"dataset_id": "hopper-medium-v2", "shift": "none", "seed": 1, "controller": "fixed", "score": 9.0},
        {"dataset_id": "hopper-medium-v2", "shift": "obs_noise", "seed": 0, "controller": "fixed", "score": 3.0},
    ]
    report = paired_comparison(rows, "none", "fixed")
    assert report["n_pairs"] == 1
    assert report["pairs"][0]["difference"] == 3.0
    assert report["mean_difference"] == 3.0
    assert report["n_left_greater"] == 1
    assert report["manuscript_status"] == "not_a_manuscript_number"
    reasons = {item["reason"] for item in report["unpaired"]}
    assert reasons == {"duplicate_scored_rows", "missing_partner"}
    assert report["wilcoxon_statistic"] == 0.0
    empty = paired_comparison([], "none", "fixed")
    assert empty["n_pairs"] == 0
    assert empty["wilcoxon_statistic"] is None
    assert empty["mean_difference"] is None
    zeros = paired_comparison(
        [
            {"dataset_id": "a", "shift": "none", "seed": 0, "controller": "none", "score": 1.0},
            {"dataset_id": "a", "shift": "none", "seed": 0, "controller": "fixed", "score": 1.0},
        ],
        "none",
        "fixed",
    )
    assert zeros["n_ties"] == 1
    assert zeros["wilcoxon_statistic"] is None
    assert "zero" in zeros["wilcoxon_note"]


def test_environment_alias_does_not_collapse_distinct_datasets():
    rows = []
    for environment, left_score, right_score in (
        ("halfcheetah-medium-v2", 4.0, 1.0),
        ("walker2d-medium-v2", 8.0, 2.0),
    ):
        rows.append(
            {
                "environment": environment,
                "controller": "none",
                "protocol": {"shift": "actuator_cripple"},
                "seed": 0,
                "score": left_score,
            }
        )
        rows.append(
            {
                "environment": environment,
                "controller": "fixed",
                "protocol": {"shift": "actuator_cripple"},
                "seed": 0,
                "score": right_score,
            }
        )
    rows.append({"controller": "none", "seed": 0, "score": 5.0})
    rows.append({"controller": "fixed", "seed": 0, "score": 6.0})
    report = paired_comparison(rows, "none", "fixed")
    assert report["n_pairs"] == 2
    assert {pair["dataset_id"] for pair in report["pairs"]} == {
        "halfcheetah-medium-v2",
        "walker2d-medium-v2",
    }
    assert {pair["difference"] for pair in report["pairs"]} == {3.0, 6.0}
    incomplete = [item for item in report["unpaired"] if item["reason"] == "incomplete_identity"]
    assert len(incomplete) == 2
    table = per_cell_rows(
        [
            {"environment": "hopper-medium-v2", "controller": "none", "shift": "none", "seed": 0, "score": 1.0},
            {"controller": "none", "seed": 1, "score": 2.0},
            {"controller": "none", "seed": 1, "normalized": [3.0]},
        ]
    )
    assert table[0]["dataset_id"] == "hopper-medium-v2"
    assert table[0]["duplicate_identity"] is False
    assert table[1]["dataset_id"] is None
    assert table[2]["dataset_id"] is None
    assert table[1]["duplicate_identity"] is False
    assert table[2]["duplicate_identity"] is False
