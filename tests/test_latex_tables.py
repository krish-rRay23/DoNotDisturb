"""LaTeX tables are rendered from synthetic fixtures, not from the manuscript."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from adaptive_plasticity.latex_tables import build_tables, render_mean_table, render_seed_table, write_tables
import render_latex_tables

MANUSCRIPT_TOKENS = ("38.78", "12.30", "8.58", "7.23", "40.37", "0.9915", "1.6501")


def _summary(controller, scores, *, seed=0, shift="none", dataset="fixture-env", fixture=True, controller_type=None):
    document = {
        "fixture": fixture,
        "dataset_id": dataset,
        "shift": shift,
        "seed": seed,
        "normalized": list(scores),
    }
    if controller is not None:
        document["controller"] = controller
    if controller_type is not None:
        document["controller_type"] = controller_type
    return document


def test_renderer_source_does_not_embed_manuscript_numbers():
    source = (REPO_ROOT / "src" / "adaptive_plasticity" / "latex_tables.py").read_text(encoding="utf-8")
    for token in MANUSCRIPT_TOKENS:
        assert token not in source


def test_empty_input_renders_no_numeric_table():
    text = render_mean_table([])
    assert "No scored rows" in text
    assert r"\begin{tabular}" not in text
    for token in MANUSCRIPT_TOKENS:
        assert token not in text


def test_synthetic_fixture_mean_and_sample_std_are_marked():
    summaries = [
        _summary("none", [1.0], seed=0),
        _summary("none", [3.0], seed=1),
    ]
    rendered = build_tables(summaries, replicates=20, seed=0)
    text = rendered["mean_table"]
    assert "SYNTHETIC FIXTURE" in text
    assert "Not a manuscript table" in text
    assert r"2.00 \pm 1.41" in text
    assert r"obs\_noise" not in text
    cell = rendered["values"]["cells"][0]
    assert cell["mean"] == pytest.approx(2.0)
    assert cell["std"] == pytest.approx(math.sqrt(2.0))
    assert rendered["values"]["fixture"] is True
    assert rendered["values"]["iqm_by_controller"]["none"]["iqm"] == pytest.approx(2.0)
    for token in MANUSCRIPT_TOKENS:
        assert token not in text


def test_single_score_omits_a_standard_deviation():
    text = render_mean_table([_summary("fixed", [4.0])], fixture=True)
    assert r"4.00$ (n=1)" in text
    assert r"\pm" not in text.split("(n=1)")[0].split("fixed")[-1]


def test_missing_seed_stays_blank_and_underscores_are_escaped():
    summaries = [
        _summary("none", [1.0], seed=0, shift="obs_noise"),
        _summary("fixed", [2.0], seed=1, shift="obs_noise"),
    ]
    text = render_seed_table(summaries, fixture=True)
    assert r"obs\_noise" in text
    assert "SYNTHETIC FIXTURE" in text
    fixed_line = next(line for line in text.splitlines() if "fixed" in line and "fixture-env" in line)
    assert "---" in fixed_line
    assert r"$2.00$" in fixed_line


def test_duplicate_identity_is_not_averaged_twice():
    summaries = [
        _summary("none", [1.0], seed=0, fixture=False),
        _summary("none", [9.0], seed=0, fixture=False),
    ]
    rendered = build_tables(summaries, fixture=False)
    assert rendered["values"]["n_rows"] == 1
    assert rendered["values"]["cells"][0]["mean"] == pytest.approx(1.0)
    assert "SYNTHETIC FIXTURE" not in rendered["mean_table"]


def test_m5_controller_type_is_grouped_when_controller_is_absent():
    summary = _summary(None, [2.0, 4.0], controller_type="adaptive")
    rendered = build_tables([summary], fixture=True)
    assert rendered["values"]["cells"][0]["controller"] == "adaptive"
    assert rendered["values"]["cells"][0]["mean"] == pytest.approx(4.0)


def test_controller_field_wins_over_controller_type():
    summary = _summary("none", [2.0], controller_type="adaptive")
    rendered = build_tables([summary], fixture=True)
    assert rendered["values"]["cells"][0]["controller"] == "none"


def test_writer_round_trip_uses_only_the_supplied_directory(tmp_path):
    source = tmp_path / "logs"
    source.mkdir()
    (source / "summary.json").write_text(
        json.dumps(_summary("none", [1.0, 3.0], fixture=True)),
        encoding="utf-8",
    )
    outdir = tmp_path / "tables"
    write_tables(source, outdir, replicates=10)
    stored = json.loads((outdir / "values.json").read_text(encoding="utf-8"))
    assert stored["fixture"] is True
    assert stored["source"] == "supplied_summaries"
    assert "SYNTHETIC FIXTURE" in (outdir / "cell_means.tex").read_text(encoding="utf-8")
    assert "Not a manuscript table" in (outdir / "seed_scores.tex").read_text(encoding="utf-8")
    render_latex_tables.main(["--indir", str(tmp_path / "missing"), "--outdir", str(tmp_path / "empty"), "--fixture"])
    empty = (tmp_path / "empty" / "cell_means.tex").read_text(encoding="utf-8")
    assert "No scored rows" in empty
    for token in MANUSCRIPT_TOKENS:
        assert token not in empty
