"""Fail if README, the manuscript, or the HTML preview drift apart.

Tokens come from docs/manuscript_claims.yaml. They are not recomputed.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adaptive_plasticity.claim_check import check_claims


def main(argv: list[str] | None = None) -> None:
    del argv
    report = check_claims(REPO_ROOT)
    if report["ok"]:
        print("Public claim tokens match the transcribed manuscript table.")
        print("No result was recomputed.")
        return
    for item in report["missing"]:
        print(f"missing {item['token']!r} on {item['surface']}", file=sys.stderr)
    for item in report["forbidden"]:
        print(
            f"forbidden {item['token']!r} on {item['surface']}: {item['reason']}",
            file=sys.stderr,
        )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
