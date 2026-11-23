"""A new decimal in the manuscript fails until the claim map names it."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adaptive_plasticity.claim_checklist import CLAIMS
from adaptive_plasticity.claim_coverage import (
    check_numeric_claims,
    check_repository,
    extract_numeric_tokens,
)

FORBIDDEN = ("0.9915", "1.6501", "40.37")


def test_current_manuscript_decimals_are_mapped():
    report = check_repository(REPO_ROOT)
    assert report["missing_from_map"] == []
    assert report["missing_from_tex"] == []
    assert report["unknown_claim_ids"] == []
    assert report["ok"] is True
    text = (REPO_ROOT / "docs" / "claim_numeric_tokens.yaml").read_text(encoding="utf-8")
    for token in FORBIDDEN:
        assert token not in text


def test_a_new_decimal_without_a_claim_entry_fails():
    tex = (REPO_ROOT / "research_paper/iclr2027/iclr2027_conference.tex").read_text(encoding="utf-8")
    token_map = yaml.safe_load((REPO_ROOT / "docs" / "claim_numeric_tokens.yaml").read_text(encoding="utf-8"))
    report = check_numeric_claims(tex + "\n$99.99$\n", token_map)
    assert report["ok"] is False
    assert report["missing_from_map"] == ["99.99"]


def test_a_dimension_is_not_a_numeric_claim():
    tokens = extract_numeric_tokens(r"\begin{subfigure}[b]{0.48\textwidth} $38.78$ \end{subfigure}")
    assert tokens == {"38.78"}


def test_a_stale_map_token_and_an_unknown_claim_fail():
    tex = "$1.25$"
    stale = check_numeric_claims(tex, {"primary_cells": ["1.25", "9.50"]})
    assert stale["ok"] is False
    assert stale["missing_from_tex"] == ["9.50"]
    unknown = check_numeric_claims(tex, {"not_a_claim": ["1.25"]})
    assert unknown["unknown_claim_ids"] == ["not_a_claim"]
    assert unknown["ok"] is False


def test_claim_ids_in_the_map_exist():
    document = yaml.safe_load((REPO_ROOT / "docs" / "claim_numeric_tokens.yaml").read_text(encoding="utf-8"))
    known = {claim["claim_id"] for claim in CLAIMS}
    assert set(document) <= known
    assert "protocol_settings" in known
