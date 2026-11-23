"""Structural checks for the manuscript. No scores are read or written.

The checker looks for references that do not resolve, labels that are
defined twice, figures that are not on disk, and bibliography keys that
are cited but missing. It does not compile TeX and it does not change a
number.
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = ["check_manuscript", "log_problems"]

_COMMENT = re.compile(r"(?<!\\)%.*")
_BEGIN = re.compile(r"\\begin\{([^}]+)\}")
_END = re.compile(r"\\end\{([^}]+)\}")
_LABEL = re.compile(r"\\label\{([^}]+)\}")
_REF = re.compile(r"\\(?:auto|eq|page|[Cc]|sub)?ref\{([^}]+)\}")
_CITE = re.compile(r"\\cite[a-zA-Z*]*\{([^}]+)\}")
_BIBKEY = re.compile(r"@\w+\s*\{([^,\s]+)\s*,")
_GRAPHIC = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
_FLOAT = re.compile(
    r"\\begin\{(figure\*?|table\*?|subfigure|algorithm)\}(.*?)\\end\{\1\}",
    re.S,
)
_CAPTION = re.compile(r"\\caption(\[[^\]]*\])?\{")


def _strip_comments(tex: str) -> str:
    return "\n".join(_COMMENT.sub("", line) for line in tex.splitlines())


def check_manuscript(
    tex: str,
    *,
    bib: str | None = None,
    figure_root: Path | None = None,
) -> list[str]:
    """Return human-readable problems. An empty list means the structure holds."""
    source = _strip_comments(tex)
    problems: list[str] = []

    stack: list[tuple[str, int]] = []
    for line_number, line in enumerate(source.splitlines(), 1):
        for match in _BEGIN.finditer(line):
            stack.append((match.group(1), line_number))
        for match in _END.finditer(line):
            name = match.group(1)
            if not stack or stack[-1][0] != name:
                problems.append(
                    f"\\end{{{name}}} at line {line_number} does not close the open environment."
                )
            else:
                stack.pop()
    for name, line_number in stack:
        problems.append(f"\\begin{{{name}}} at line {line_number} is never closed.")

    labels = _LABEL.findall(source)
    seen: set[str] = set()
    for label in labels:
        if label in seen:
            problems.append(f"Label {label!r} is defined more than once.")
        seen.add(label)
    for ref in sorted(set(_REF.findall(source))):
        if ref not in seen:
            problems.append(f"Reference {ref!r} has no matching label.")

    for match in _FLOAT.finditer(source):
        body = match.group(2)
        captions = [item.start() for item in _CAPTION.finditer(body)]
        for label_match in _LABEL.finditer(body):
            if captions and not any(pos < label_match.start() for pos in captions):
                problems.append(
                    f"Label {label_match.group(1)!r} comes before the caption, "
                    "so the reference would not resolve to that float."
                )

    if bib is not None:
        keys = set(_BIBKEY.findall(bib))
        cited: set[str] = set()
        for group in _CITE.findall(source):
            cited.update(part.strip() for part in group.split(",") if part.strip())
        for key in sorted(cited - keys):
            problems.append(f"Citation {key!r} is not in the bibliography.")

    if figure_root is not None:
        for relative in _GRAPHIC.findall(source):
            if not (figure_root / relative).is_file():
                problems.append(f"Figure file {relative!r} is not on disk.")
    return problems


def log_problems(log: str) -> list[str]:
    """Problems a TeX log still reports after the structural check."""
    problems: list[str] = []
    for line in log.splitlines():
        lowered = line.lower()
        if line.startswith("!") or "emergency stop" in lowered or "fatal error" in lowered:
            problems.append(line.strip())
        elif "undefined" in lowered and ("reference" in lowered or "citation" in lowered):
            problems.append(line.strip())
    return problems


def check_repository(root: str | Path) -> list[str]:
    base = Path(root)
    tex_path = base / "research_paper" / "iclr2027" / "iclr2027_conference.tex"
    bib_path = base / "research_paper" / "iclr2027" / "iclr2027_conference.bib"
    return check_manuscript(
        tex_path.read_text(encoding="utf-8"),
        bib=bib_path.read_text(encoding="utf-8"),
        figure_root=tex_path.parent,
    )
