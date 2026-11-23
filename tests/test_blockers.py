"""The blockers page names what is still external and what would unblock it."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_blockers_page_names_each_external_blocker_and_its_unblock():
    text = (REPO_ROOT / "docs" / "blockers.md").read_text(encoding="utf-8")
    index = (REPO_ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    assert "blockers.md" in index
    for heading in (
        "## Revised manuscript numbers",
        "## GPU runs",
        "## Cal-QL, PEX, and RO2O",
        "## AntMaze adapter",
    ):
        assert heading in text
        body = text.split(heading, 1)[1].split("\n## ", 1)[0]
        assert "What unblocks it" in body
    assert "docs/pending_revision_numbers.md" in text
    assert "0.9915" in text and "1.6501" in text and "40.37" in text
    assert "do not copy them into the paper" in text
    for token in (
        "CONFIRM-FINAL-150",
        "CONFIRM-STRESS-24",
        "CONFIRM-STABLE-9",
        "CONFIRM-ABLATION-FULL",
    ):
        assert token in text
    assert "results/stable_validation" in text
    assert "docs/run_guide.md" in text
    assert "open_external_session" in text
    assert "ExternalLearnerUnavailable" in text
    assert "antmaze-umaze-v2" in text
    assert "not in the dataset registry" in text
    assert "refuse_placeholder" in text
    # The page may name the pending figures. It must not present a new score.
    assert "38.78" not in text
    assert "12.30" not in text
