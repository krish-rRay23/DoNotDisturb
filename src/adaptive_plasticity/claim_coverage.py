"""Fail when the manuscript prints a decimal that the claim map does not name.

Tokens are transcribed strings. This module does not recompute a score and
does not open a run log. A TeX dimension such as ``0.48\\textwidth`` is not
a numeric claim.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from adaptive_plasticity.claim_checklist import CLAIMS

DEFAULT_TEX = Path("research_paper/iclr2027/iclr2027_conference.tex")
DEFAULT_MAP = Path("docs/claim_numeric_tokens.yaml")

_DECIMAL = re.compile(r"(?<![\w.])([-+]?\d+\.\d+)(?![\w.])")
_POWER = re.compile(r"10\^\{(-?\d+)\}")
_DIMENSION = re.compile(
    r"\s*(?:\\(?:textwidth|linewidth|columnwidth|hsize)|(?:in|em|ex|cm|mm|pt|pc)\b)"
)


def strip_tex_comments(tex: str) -> str:
    """Drop unescaped ``%`` comments. ``\\%`` stays."""
    kept: list[str] = []
    for line in tex.splitlines():
        out: list[str] = []
        index = 0
        while index < len(line):
            if line[index] == "%" and (index == 0 or line[index - 1] != "\\"):
                break
            out.append(line[index])
            index += 1
        kept.append("".join(out))
    return "\n".join(kept)


def extract_numeric_tokens(tex: str) -> set[str]:
    """Signed decimals and ``10^{n}`` powers, excluding TeX dimensions."""
    body = strip_tex_comments(tex)
    tokens: set[str] = set()
    for match in _DECIMAL.finditer(body):
        if _DIMENSION.match(body[match.end() : match.end() + 24]):
            continue
        tokens.add(match.group(1).lstrip("+"))
    for match in _POWER.finditer(body):
        tokens.add(f"10^{{{match.group(1)}}}")
    return tokens


def load_token_map(path: str | Path) -> dict[str, list[str]]:
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Token map {path} must be a mapping of claim id to tokens.")
    parsed: dict[str, list[str]] = {}
    for claim_id, tokens in document.items():
        if not isinstance(tokens, list) or not all(isinstance(item, str) for item in tokens):
            raise ValueError(f"Claim {claim_id} must list strings.")
        parsed[str(claim_id)] = [item.strip() for item in tokens]
    return parsed


def check_numeric_claims(
    tex: str,
    token_map: dict[str, list[str]],
    claim_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Compare printed tokens with the claim map.

    ``missing_from_map`` is a new printed number with no claim entry.
    ``missing_from_tex`` is a map entry that the manuscript no longer prints.
    ``unknown_claim_ids`` are map keys that are not claims.
    """
    known = claim_ids if claim_ids is not None else {claim["claim_id"] for claim in CLAIMS}
    printed = extract_numeric_tokens(tex)
    mapped: set[str] = set()
    unknown = sorted(claim_id for claim_id in token_map if claim_id not in known)
    for tokens in token_map.values():
        mapped.update(tokens)
    return {
        "ok": not unknown and printed == mapped,
        "missing_from_map": sorted(printed - mapped),
        "missing_from_tex": sorted(mapped - printed),
        "unknown_claim_ids": unknown,
        "source": "transcribed_tokens_not_recomputed",
    }


def check_repository(root: str | Path = ".") -> dict[str, Any]:
    base = Path(root)
    tex = (base / DEFAULT_TEX).read_text(encoding="utf-8")
    token_map = load_token_map(base / DEFAULT_MAP)
    return check_numeric_claims(tex, token_map)
