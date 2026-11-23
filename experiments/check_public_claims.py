"""Fail if README, the manuscript, or the HTML preview drift apart.

Tokens come from docs/manuscript_claims.yaml. They are not recomputed.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adaptive_plasticity.claim_check import check_claims, check_surface_sync


def main(argv: list[str] | None = None) -> None:
    del argv
    report = check_claims(REPO_ROOT)
    sync = check_surface_sync(
        tex=(REPO_ROOT / "research_paper/iclr2027/iclr2027_conference.tex").read_text(encoding="utf-8"),
        readme=(REPO_ROOT / "README.md").read_text(encoding="utf-8"),
        preview=(REPO_ROOT / "research_paper/iclr2027/preview.html").read_text(encoding="utf-8"),
    )
    if report["ok"] and sync["ok"]:
        print("Public claim tokens match the transcribed manuscript table.")
        print("README and preview decimals match the manuscript.")
        print("No result was recomputed.")
        return
    for item in report["missing"]:
        print(f"missing {item['token']!r} on {item['surface']}", file=sys.stderr)
    for item in report["forbidden"]:
        print(
            f"forbidden {item['token']!r} on {item['surface']}: {item['reason']}",
            file=sys.stderr,
        )
    for item in sync["missing_abstract_decimals"]:
        print(
            f"abstract decimal {item['token']!r} is missing from {item['surface']}",
            file=sys.stderr,
        )
    for item in sync["decimals_not_in_manuscript"]:
        print(
            f"{item['surface']} decimal {item['token']!r} is not in the manuscript",
            file=sys.stderr,
        )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
