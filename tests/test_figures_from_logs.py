"""Score figures are means of supplied logs, or they are not written."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from adaptive_plasticity.figures_from_logs import scores_by_controller, write_score_figure
import reproduce_all_tables_and_figures as repro


def test_missing_scores_do_not_write_a_figure(tmp_path: Path):
    report = write_score_figure([{"controller": "none"}], tmp_path)
    assert report["ok"] is False
    assert report["wrote_figure"] is False
    assert list(tmp_path.iterdir()) == []


def test_bar_means_come_from_the_supplied_scores(tmp_path: Path):
    summaries = [
        {"controller": "none", "normalized": [1.0, 3.0]},
        {"controller": "none", "score": 5.0},
        {"controller": "fixed", "normalized": [10.0]},
        {"controller": None, "score": 99.0},
    ]
    assert scores_by_controller(summaries) == {"none": [3.0, 5.0], "fixed": [10.0]}
    m5_summary = {"controller_type": "adaptive", "normalized": [2.0, 4.0]}
    assert scores_by_controller([m5_summary]) == {"adaptive": [4.0]}
    both = {"controller": "none", "controller_type": "adaptive", "score": 1.0}
    assert scores_by_controller([both]) == {"none": [1.0]}
    report = write_score_figure(summaries, tmp_path)
    assert report["wrote_figure"] is True
    payload = json.loads((tmp_path / "plotted.json").read_text(encoding="utf-8"))
    assert payload["banner"].startswith("Plotted from the supplied summaries")
    assert payload["controllers"]["none"] == {"n": 2, "mean": 4.0}
    assert payload["controllers"]["fixed"] == {"n": 1, "mean": 10.0}
    assert (tmp_path / "scores_from_logs.png").is_file()
    assert "38.78" not in json.dumps(payload)


def test_empty_iqm_is_not_a_zero_and_stress_gate_is_not_filled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    assert math.isnan(repro.compute_iqm([]))
    monkeypatch.setattr(repro, "STRESS_STUDY_DIR", tmp_path)
    table = repro.reproduce_table_2()
    assert set(table["CapacityGate"]) == {"logs absent"}
    exported = table.to_csv(index=False)
    assert "8.58" not in exported
    assert "7.23" not in exported
