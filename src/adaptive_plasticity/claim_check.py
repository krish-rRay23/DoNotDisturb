"""Check that public pages repeat the manuscript tokens, and no others we forbade.

The token list is transcribed. This module does not recompute an IQM, a
p-value, or a cell mean. A missing token is a consistency failure. It is
not filled in.
"""

from __future__ import annotations

import re
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

# A sentence period may follow a decimal. A further digit may not.
_DECIMAL = re.compile(r"(?<![\w.])([-+]?\d+\.\d+)(?!\.?\d)")
_STYLE = re.compile(r"<style\b.*?</style>", re.I | re.S)
_SECTION_NUMBER = re.compile(r"Section\s+\d+\.\d+")

__all__ = [
    "load_claims",
    "check_claims",
    "check_surface_sync",
    "ClaimMismatch",
]

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


def _decimals(text: str) -> list[str]:
    return _DECIMAL.findall(text)


def _decimal_values(text: str) -> set[Decimal]:
    return {Decimal(token) for token in _decimals(text)}


def manuscript_abstract(tex: str) -> str:
    return tex.split(r"\begin{abstract}", 1)[1].split(r"\end{abstract}", 1)[0]


def readme_result_section(readme: str) -> str:
    return readme.split("## Result", 1)[1].split("## What is not a result yet", 1)[0]


def preview_prose(preview: str) -> str:
    """Drop the stylesheet and section numbers such as ``Section 5.4``."""
    return _SECTION_NUMBER.sub("Section", _STYLE.sub("", preview))


def check_surface_sync(*, tex: str, readme: str, preview: str) -> dict[str, Any]:
    """The README and the HTML preview must keep the manuscript's decimals.

    Abstract decimals are copied as written, and each one must be its own
    decimal on the public page. ``1.4`` does not match ``1.44``. A result
    decimal on either public page must be the same value as some decimal
    in the manuscript, so ``0.5`` still matches ``0.50``. This does not
    recompute a score.
    """
    abstract = manuscript_abstract(tex)
    prose = preview_prose(preview)
    missing: list[dict[str, str]] = []
    surface_tokens = {
        "readme": set(_decimals(readme)),
        "preview": set(_decimals(prose)),
    }
    for token in dict.fromkeys(_decimals(abstract)):
        for surface, tokens in surface_tokens.items():
            if token not in tokens:
                missing.append({"token": token, "surface": surface})
    manuscript_values = _decimal_values(tex)
    extra: list[dict[str, str]] = []
    for surface, text in (
        ("readme", readme_result_section(readme)),
        ("preview", prose),
    ):
        for token in dict.fromkeys(_decimals(text)):
            if Decimal(token) not in manuscript_values:
                extra.append({"token": token, "surface": surface})
    return {
        "ok": not missing and not extra,
        "missing_abstract_decimals": missing,
        "decimals_not_in_manuscript": extra,
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
