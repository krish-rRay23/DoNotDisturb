"""The manuscript job uses a pinned TeX image and cannot apt-install TeX Live."""

from __future__ import annotations

from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "tests.yml"


def test_latex_job_uses_a_pinned_image_and_a_short_timeout():
    text = WORKFLOW.read_text(encoding="utf-8")
    latex = text.split("\n  latex:", 1)[1]
    assert "image: ghcr.io/xu-cheng/texlive-small:20261001" in latex
    assert "timeout-minutes: 15" in latex
    assert "tlmgr option repository https://texlive.info/tlnet-archive/2026/10/01/tlnet" in latex
    assert "timeout 4m tlmgr install algorithms eso-pic multirow" in latex
    assert "apt-get" not in latex
    assert "texlive-latex-extra" not in latex
    assert "cancel-in-progress: true" in text
    assert "uses: actions/upload-artifact@v4" in latex
    assert "name: manuscript-pdf" in latex
    assert "path: research_paper/iclr2027/iclr2027_conference.pdf" in latex
    assert "if-no-files-found: error" in latex
