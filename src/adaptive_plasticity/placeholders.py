"""Domains that are not part of the reported benchmark.

These records exist so a later run has a named config. They are not loaded
by the 125-cell ablation plan, and nothing here trains or scores them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "load_placeholders",
    "refuse_placeholder",
    "PlaceholderNotRunnable",
]


class PlaceholderNotRunnable(RuntimeError):
    """Raised when a caller tries to launch a domain that has not been wired."""


def placeholder_dir(root: str | Path | None = None) -> Path:
    """Directory of the not-yet-run placeholder configs."""
    base = Path(__file__).resolve().parents[2] if root is None else Path(root)
    return base / "configs" / "placeholders"


def load_placeholders(root: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Load placeholder configs. Every file must stay not-yet-run and not executable."""
    loaded: dict[str, dict[str, Any]] = {}
    for path in sorted(placeholder_dir(root).glob("*.yaml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError(f"Placeholder {path} must be a mapping.")
        for key in ("name", "status", "executable", "dataset_id", "why_not_run"):
            if key not in document:
                raise ValueError(f"Placeholder {path} is missing {key!r}.")
        if document["status"] != "not_yet_run":
            raise ValueError(f"Placeholder {document['name']!r} must stay not_yet_run.")
        if document["executable"] is not False:
            raise ValueError(f"Placeholder {document['name']!r} must set executable: false.")
        loaded[str(document["name"])] = document
    if not loaded:
        raise FileNotFoundError(f"No placeholder configs in {placeholder_dir(root)}.")
    return loaded


def refuse_placeholder(name: str, root: str | Path | None = None) -> None:
    """Stop a launch. The message repeats why the domain is not runnable."""
    document = load_placeholders(root)[name]
    raise PlaceholderNotRunnable(
        f"{name} is not_yet_run and executable is false. {document['why_not_run']}"
    )
