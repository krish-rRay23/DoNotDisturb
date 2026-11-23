"""The GPU run guide names the four confirm tokens and does not invent a score."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adaptive_plasticity.ablations import load_all_studies, select_cells

FORBIDDEN = ("0.9915", "1.6501", "40.37", "38.78", "12.30")


def test_run_guide_names_the_four_confirm_gated_runs_and_the_regenerators():
    text = (REPO_ROOT / "docs" / "run_guide.md").read_text(encoding="utf-8")
    for token in (
        "CONFIRM-FINAL-150",
        "CONFIRM-STRESS-24",
        "CONFIRM-STABLE-9",
        "CONFIRM-ABLATION-FULL",
    ):
        assert token in text
    assert "--device cuda" in text
    assert "export MUJOCO_GL=glfw" in text
    assert "python experiments/final_analysis.py --results-dir results/final_study" in text
    assert "python experiments/reproduce_all_tables_and_figures.py" in text
    assert "python experiments/render_latex_tables.py --indir results/final_study" in text
    assert "python experiments/analyze_stress_coverage.py --indir results/stress_study --seeds 0,1,2" in text
    assert "not_the_manuscript_table" in text
    assert "stress_iqm.json" in text
    assert "not_yet_run" in text
    assert "STABLE_VAL" in text
    assert "--outdir results/stable_validation" in text
    assert "results/stress_study" in text
    assert "refuses when the 135-cell primary matrix is incomplete" in text
    assert "refuses while the 150-cell study is incomplete" not in text
    assert "does not stamp PARTIAL-DATA" in text
    full = select_cells(load_all_studies(REPO_ROOT), pilot=False)
    assert f"The full ablation plan is {len(full)} cells" in text
    for token in FORBIDDEN:
        assert token not in text
    readme = (REPO_ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    assert "run_guide.md" in readme
