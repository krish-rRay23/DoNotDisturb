"""Not-yet-run plans for the needs-data items in the reviewer note.

Loading and expanding these files does not train, download, or score.
Every plan stays ``not_yet_run`` and ``executable: false``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "PendingPlanRefused",
    "expand_pending_plan",
    "load_pending_plans",
    "pending_plan_dir",
    "refuse_pending_plan",
]


class PendingPlanRefused(RuntimeError):
    """Raised when a caller tries to launch a plan that has not been run."""


def pending_plan_dir(root: str | Path | None = None) -> Path:
    """Directory of the needs-data plan files."""
    base = Path(__file__).resolve().parents[2] if root is None else Path(root)
    return base / "configs" / "pending"


def load_pending_plans(root: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Load every pending plan. Each one must stay not-yet-run and not executable."""
    loaded: dict[str, dict[str, Any]] = {}
    for path in sorted(pending_plan_dir(root).glob("*.yaml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError(f"Pending plan {path} must be a mapping.")
        for key in ("name", "status", "executable", "kind", "reviewer_phrase", "why_not_run"):
            if key not in document:
                raise ValueError(f"Pending plan {path} is missing {key!r}.")
        if document["status"] != "not_yet_run":
            raise ValueError(f"Pending plan {document['name']!r} must stay not_yet_run.")
        if document["executable"] is not False:
            raise ValueError(f"Pending plan {document['name']!r} must set executable: false.")
        if document["kind"] not in ("grid", "analysis", "existing", "blocked"):
            raise ValueError(f"Pending plan {document['name']!r} has unknown kind {document['kind']!r}.")
        loaded[str(document["name"])] = document
    if not loaded:
        raise FileNotFoundError(f"No pending plans in {pending_plan_dir(root)}.")
    return loaded


def expand_pending_plan(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a grid plan into cells. Other kinds have no cells and no scores."""
    if document["kind"] != "grid":
        return []
    grid = document.get("grid")
    if not isinstance(grid, dict):
        raise ValueError(f"Pending plan {document['name']!r} needs a grid.")
    datasets = [str(item) for item in grid["datasets"]]
    shifts = [str(item) for item in grid["shifts"]]
    controllers = [str(item) for item in grid["controllers"]]
    seeds = [int(item) for item in grid["seeds"]]
    if not datasets or not shifts or not controllers or not seeds:
        raise ValueError(f"Pending plan {document['name']!r} has an empty grid.")
    rows: list[dict[str, Any]] = []
    for dataset_id in datasets:
        for shift in shifts:
            for controller in controllers:
                for seed in seeds:
                    rows.append(
                        {
                            "plan": document["name"],
                            "dataset_id": dataset_id,
                            "shift": shift,
                            "controller": controller,
                            "seed": seed,
                            "status": "not_yet_run",
                        }
                    )
    return rows


def refuse_pending_plan(name: str, root: str | Path | None = None) -> None:
    """Stop a launch. The message repeats why the plan is not runnable."""
    document = load_pending_plans(root)[name]
    raise PendingPlanRefused(
        f"{name} is not_yet_run and executable is false. {document['why_not_run']}"
    )
