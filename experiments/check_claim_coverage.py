"""Fail if the manuscript prints a decimal the claim map does not name.

The map is docs/claim_numeric_tokens.yaml. Tokens are transcribed. No score
is recomputed and no log is opened.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adaptive_plasticity.claim_coverage import check_repository


def main(argv: list[str] | None = None) -> None:
    del argv
    report = check_repository(REPO_ROOT)
    if report["ok"]:
        print("Manuscript decimals are listed in the claim map.")
        print("No result was recomputed.")
        return
    for claim_id in report["unknown_claim_ids"]:
        print(f"unknown claim id {claim_id!r}", file=sys.stderr)
    for token in report["missing_from_map"]:
        print(
            f"numeric claim {token!r} is in the manuscript and not in the claim map",
            file=sys.stderr,
        )
    for token in report["missing_from_tex"]:
        print(
            f"claim-map token {token!r} is not printed in the manuscript",
            file=sys.stderr,
        )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
