"""Public pages must repeat the manuscript tokens and omit unverified ones."""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_plasticity.claim_check import (
    ClaimMismatch,
    assert_public_claims,
    check_claims,
    check_surface_sync,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_repository_public_surfaces_match_the_transcribed_table():
    report = assert_public_claims(REPO_ROOT)
    assert report["ok"] is True
    assert report["source"] == "transcribed_tokens_not_recomputed"
    assert report["missing"] == []
    assert report["forbidden"] == []


def test_a_dropped_token_and_a_forbidden_revision_figure_both_fail(tmp_path: Path):
    claims = {
        "surfaces": {"readme": "README.md"},
        "require": [{"token": "38.78", "surfaces": ["readme"]}],
        "forbid": [{"token": "40.37", "surfaces": ["readme"], "reason": "not in the manuscript"}],
    }
    readme = tmp_path / "README.md"
    readme.write_text("IQM 38.78 only.\n", encoding="utf-8")
    assert check_claims(tmp_path, claims)["ok"] is True
    readme.write_text("IQM 12.30 and a ReDo IQM of 40.37.\n", encoding="utf-8")
    report = check_claims(tmp_path, claims)
    assert report["ok"] is False
    assert report["missing"] == [{"token": "38.78", "surface": "readme"}]
    assert report["forbidden"][0]["token"] == "40.37"
    with pytest.raises(ClaimMismatch):
        assert_public_claims(tmp_path, claims)


def _surfaces() -> tuple[str, str, str]:
    tex = (REPO_ROOT / "research_paper/iclr2027/iclr2027_conference.tex").read_text(encoding="utf-8")
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    preview = (REPO_ROOT / "research_paper/iclr2027/preview.html").read_text(encoding="utf-8")
    return tex, readme, preview


def test_readme_and_preview_keep_the_manuscript_decimals():
    tex, readme, preview = _surfaces()
    report = check_surface_sync(tex=tex, readme=readme, preview=preview)
    assert report["ok"] is True
    assert report["missing_abstract_decimals"] == []
    assert report["decimals_not_in_manuscript"] == []
    assert report["source"] == "transcribed_tokens_not_recomputed"


def test_an_edited_abstract_decimal_must_reach_readme_and_preview():
    tex, readme, preview = _surfaces()
    abstract, rest = tex.split(r"\end{abstract}", 1)
    edited = abstract.replace("38.78", "38.79", 1) + r"\end{abstract}" + rest
    report = check_surface_sync(tex=edited, readme=readme, preview=preview)
    assert report["ok"] is False
    assert {"token": "38.79", "surface": "readme"} in report["missing_abstract_decimals"]
    assert {"token": "38.79", "surface": "preview"} in report["missing_abstract_decimals"]


def test_a_readme_result_decimal_that_is_not_in_the_manuscript_fails():
    tex, readme, preview = _surfaces()
    edited = readme.replace("## Result", "## Result\n\nA fabricated IQM of 99.99.\n", 1)
    report = check_surface_sync(tex=tex, readme=edited, preview=preview)
    assert report["ok"] is False
    assert {"token": "99.99", "surface": "readme"} in report["decimals_not_in_manuscript"]
