"""Manuscript wording stays on the transcribed numbers and does not invent citations."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEX = REPO_ROOT / "research_paper" / "iclr2027" / "iclr2027_conference.tex"
BIB = REPO_ROOT / "research_paper" / "iclr2027" / "iclr2027_conference.bib"

UNVERIFIED = ("0.9915", "1.6501", "40.37")
INVENTED_KEYS = ("kingma2015adam", "he2015delving", "polyak1964", "vitter1985", "brockman2016gym")


def test_limitations_keep_the_published_dormancy_band_and_name_the_stress_gap():
    tex = TEX.read_text(encoding="utf-8")
    assert r"\section{Limitations}" in tex
    assert r"$7\text{--}12\%$" in tex
    assert "seed 0 only" in tex
    assert "not an arm of the 135-cell primary matrix" in tex
    assert r"\textsc{CapacityGate}" in tex
    for token in UNVERIFIED:
        assert token not in tex


def test_redo_is_named_as_an_approximate_low_disruption_reset():
    tex = TEX.read_text(encoding="utf-8")
    assert "approximate, low-disruption reset" in tex
    assert "Neuron Recycling" not in tex
    assert r"\textsc{CapacityGate} is displayed" in tex
    assert r"mean $\pm$ standard deviation across 3 paired seeds" in tex
    assert "dormant neuron recycling (ReDo)" not in tex
    assert r"$\rho(t) > 0.70, d(t) < 0.15$" in tex


def test_missing_citations_are_todos_and_not_new_bibliography_entries():
    tex = TEX.read_text(encoding="utf-8")
    bib = BIB.read_text(encoding="utf-8")
    for needle in (
        "TODO: cite Adam",
        "TODO: cite Kaiming initialization",
        "TODO: cite Polyak averaging",
        "TODO: cite reservoir sampling",
        "TODO: cite Gym",
    ):
        assert needle in tex
    for key in INVENTED_KEYS:
        assert key not in bib
        assert key not in tex
    assert "duplicates agarwal2021deep" in bib
