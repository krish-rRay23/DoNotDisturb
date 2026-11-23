"""The manuscript job uses a pinned TeX image and cannot apt-install TeX Live."""

from __future__ import annotations

from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "tests.yml"


def test_latex_job_uses_a_pinned_image_and_a_short_timeout():
    text = WORKFLOW.read_text(encoding="utf-8")
    latex = text.split("\n  latex:", 1)[1]
    assert "image: ghcr.io/xu-cheng/texlive-small:20261001" in latex
    assert "timeout-minutes: 15" in latex
    assert "timeout 4m tlmgr install algorithms eso-pic multirow" in latex
    assert "apt-get" not in latex
    assert "texlive-latex-extra" not in latex
    assert "cancel-in-progress: true" in text
