"""Write docs/claim_log_checklist.md from the working tree.

The writer does not read log contents. Run it from the repository root:

    python experiments/write_claim_checklist.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adaptive_plasticity.claim_checklist import render_markdown

DESTINATION = REPO_ROOT / "docs" / "claim_log_checklist.md"


def main() -> None:
    text = render_markdown(REPO_ROOT)
    DESTINATION.write_text(text, encoding="utf-8")
    print(f"Wrote {DESTINATION}")
    print("No log was opened and no score was recomputed.")


if __name__ == "__main__":
    main()
