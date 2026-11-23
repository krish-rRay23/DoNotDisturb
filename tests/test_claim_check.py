"""Public pages must repeat the manuscript tokens and omit unverified ones."""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_plasticity.claim_check import ClaimMismatch, assert_public_claims, check_claims

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
