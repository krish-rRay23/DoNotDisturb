"""Check that public pages repeat the manuscript tokens, and no others we forbade.

The token list is transcribed. This module does not recompute an IQM, a
p-value, or a cell mean. A missing token is a consistency failure. It is
not filled in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

__all__ = ["load_claims", "check_claims", "ClaimMismatch"]

DEFAULT_CLAIMS = Path(__file__).resolve().parents[2] / "docs" / "manuscript_claims.yaml"


class ClaimMismatch(RuntimeError):
    """A public surface is missing a manuscript token or contains a forbidden one."""


def load_claims(path: str | Path | None = None) -> dict[str, Any]:
    """Read the transcribed token table. It does not recompute a result."""
    claims_path = Path(path) if path is not None else DEFAULT_CLAIMS
    document = yaml.safe_load(claims_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Claim table {claims_path} must be a mapping.")
    for key in ("surfaces", "require", "forbid"):
        if key not in document:
            raise ValueError(f"Claim table {claims_path} is missing {key!r}.")
    return document


def check_claims(root: str | Path, claims: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return missing and forbidden hits. ``ok`` is true only when both are empty."""
    base = Path(root)
    table = claims if claims is not None else load_claims(base / "docs" / "manuscript_claims.yaml")
    texts: dict[str, str] = {}
    for name, relative in table["surfaces"].items():
        path = base / relative
        if not path.is_file():
            texts[name] = ""
        else:
            texts[name] = path.read_text(encoding="utf-8")
    missing: list[dict[str, str]] = []
    forbidden: list[dict[str, str]] = []
    for item in table["require"]:
        token = str(item["token"])
        for surface in item["surfaces"]:
            if token not in texts.get(surface, ""):
                missing.append({"token": token, "surface": surface})
    for item in table["forbid"]:
        token = str(item["token"])
        for surface in item["surfaces"]:
            if token in texts.get(surface, ""):
                forbidden.append(
                    {
                        "token": token,
                        "surface": surface,
                        "reason": str(item.get("reason", "")),
                    }
                )
    return {
        "ok": not missing and not forbidden,
        "missing": missing,
        "forbidden": forbidden,
        "source": "transcribed_tokens_not_recomputed",
    }


def assert_public_claims(root: str | Path, claims: dict[str, Any] | None = None) -> dict[str, Any]:
    """Raise ClaimMismatch when a required token is missing or a forbidden token is present."""
    report = check_claims(root, claims)
    if report["ok"]:
        return report
    parts = []
    if report["missing"]:
        parts.append(f"missing {report['missing']}")
    if report["forbidden"]:
        parts.append(f"forbidden {report['forbidden']}")
    raise ClaimMismatch("; ".join(parts))
