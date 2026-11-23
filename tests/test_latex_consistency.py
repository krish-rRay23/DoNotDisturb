"""The manuscript's references resolve, and symbols are named before they are used."""

from __future__ import annotations

from pathlib import Path

from adaptive_plasticity.latex_consistency import check_manuscript, check_repository

REPO_ROOT = Path(__file__).resolve().parents[1]
TEX = REPO_ROOT / "research_paper" / "iclr2027" / "iclr2027_conference.tex"


def test_manuscript_refs_labels_and_figures_resolve():
    problems = check_repository(REPO_ROOT)
    assert problems == []


def test_a_missing_label_and_a_label_before_its_caption_fail():
    tex = r"""
\begin{table}
\label{tab:early}
\caption{Too late.}
\end{table}
Table~\ref{tab:missing} and Table~\ref{tab:early}.
"""
    problems = check_manuscript(tex)
    assert any("tab:missing" in item for item in problems)
    assert any("tab:early" in item and "before the caption" in item for item in problems)


def test_notation_is_named_at_first_use_and_no_number_was_added():
    tex = TEX.read_text(encoding="utf-8")
    assert "$K$" not in tex
    assert "H_l" not in tex
    assert "Let $n_l$ be the number of units" in tex
    assert r"with discount $\gamma$" in tex
    dormancy = tex.split("Activation Dormancy", 1)[1].split(r"\subsection", 1)[0]
    assert r"\tau_{\mathrm{redo}} = 0.10" in dormancy
    for label in (
        "fig:iqm_intro",
        "fig:prob_intro",
        "fig:telemetry_trajectories_app",
        "fig:clean_vs_shifted_app",
    ):
        assert r"\subref{" + label + "}" in tex
    abstract = tex.split(r"\begin{abstract}", 1)[1].split(r"\end{abstract}", 1)[0]
    assert abstract.find(r"($\rho$)") < abstract.find(r"\rho(t)")
    assert abstract.find("($d$)") < abstract.find(r"d \le 0.085")
