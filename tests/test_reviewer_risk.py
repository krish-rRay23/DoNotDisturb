"""Scope sentences match the reviewer note. No new score is introduced."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEX = REPO_ROOT / "research_paper" / "iclr2027" / "iclr2027_conference.tex"
NOTE = REPO_ROOT / "docs" / "reviewer_risk_review.md"

UNVERIFIED = ("0.9915", "1.6501", "40.37")


def test_every_risk_item_is_tagged_and_the_scope_sentences_are_in_the_manuscript():
    note = NOTE.read_text(encoding="utf-8")
    tags = [line.split("tag:", 1)[1].strip() for line in note.splitlines() if line.startswith("- tag:")]
    assert tags
    assert set(tags) <= {"addressable-now", "needs-data"}
    assert tags.count("addressable-now") >= 8
    assert tags.count("needs-data") >= 8
    tex = TEX.read_text(encoding="utf-8")
    for phrase in (
        "The reported IQM and the signed-rank test use the 135 primary factorial runs.",
        "Hopper is not in that stress protocol.",
        "fails for unconditional periodic Shrink-and-Perturb",
        r"P(\text{None} > \text{Fixed}) = 0.933",
        "Seed-Level Primary Benchmark Data",
        "does not treat them as independent replicates of one task.",
        "are not in that aggregate.",
        "Resets are not an arm of the 135-cell matrix.",
        "does not show that every plasticity operator has that effect.",
        r"\textbf{IQL only.}",
    ):
        assert phrase in tex
    for phrase in (
        "Exhaustive Seed-Level",
        "fatal for O2O transfer",
        "fails fundamentally in",
        "across four regimes",
    ):
        assert phrase not in tex
    for token in UNVERIFIED:
        assert token not in tex
        assert token not in note or "not manuscript numbers" in note
