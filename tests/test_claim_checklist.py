"""The claim checklist reports absent manuscript logs as unverified and does not read them."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adaptive_plasticity.claim_checklist import CLAIMS, assess_claims, render_markdown

FORBIDDEN = ("0.9915", "1.6501", "40.37")
MANUSCRIPT_TOKENS = ("38.78", "12.30", "[37.43, 40.03]", "[9.36, 15.10]", "26.03", "[0.99, 1.03]", "0.085")


def test_absent_manuscript_logs_are_unverified(tmp_path):
    rows = {row["claim_id"]: row for row in assess_claims(tmp_path)}
    for claim in CLAIMS:
        if claim["role"] == "manuscript":
            assert rows[claim["claim_id"]]["status"] == "unverified"
            assert rows[claim["claim_id"]]["logs_present"] is False
    assert rows["short_pilots"]["status"] == "not_a_manuscript_result"
    assert rows["older_final_summary"]["status"] == "not_the_manuscript_table"
    assert rows["pooled_stress_iqm"]["status"] == "not_the_manuscript_table"


def test_a_present_log_is_not_read_and_is_not_called_verified(tmp_path):
    target = tmp_path / "results" / "final_study" / "FINAL_analysis_summary.json"
    target.parent.mkdir(parents=True)
    target.write_text('{"planted": "do-not-copy-this-score"}', encoding="utf-8")
    rows = {row["claim_id"]: row for row in assess_claims(tmp_path)}
    assert rows["aggregate_iqm"]["status"] == "present_not_recomputed"
    assert rows["paired_comparison"]["status"] == "present_not_recomputed"
    assert rows["stress_table"]["status"] == "unverified"
    text = render_markdown(tmp_path)
    assert "do-not-copy-this-score" not in text
    assert "present_not_recomputed" in text
    assert "verified result" not in text


def test_committed_checklist_matches_this_checkout_and_keeps_tokens():
    text = render_markdown(REPO_ROOT)
    committed = (REPO_ROOT / "docs" / "claim_log_checklist.md").read_text(encoding="utf-8")
    assert committed == text
    for token in MANUSCRIPT_TOKENS:
        assert token in committed
    assert "`unverified`" in committed
    for token in FORBIDDEN:
        assert token not in committed
    source = (REPO_ROOT / "src" / "adaptive_plasticity" / "claim_checklist.py").read_text(encoding="utf-8")
    for token in FORBIDDEN:
        assert token not in source
