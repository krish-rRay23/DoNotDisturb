"""Plan builders for CapacityGate ablations that have not been run.

The YAML files under ``configs/ablations/`` are the source of the grids.
Expanding a study enumerates cells and labels every cell ``not_yet_run``.
This module does not train, score, or invent outcomes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

STUDY_FILES: tuple[str, ...] = (
    "probe_distributions.yaml",
    "gate_thresholds.yaml",
    "cooldown.yaml",
    "gate_operator.yaml",
)

PILOT_CONFIRM_TOKEN = "CONFIRM-ABLATION-PILOT"
FULL_CONFIRM_TOKEN = "CONFIRM-ABLATION-FULL"

#: Budgets for a launch check. These are not manuscript runs.
PILOT_BUDGET: dict[str, int] = {
    "online_steps": 2_000,
    "warmup_updates": 50,
    "eval_interval": 2_000,
    "eval_episodes": 1,
    "gate_interval": 500,
}

__all__ = [
    "STUDY_FILES",
    "PILOT_CONFIRM_TOKEN",
    "FULL_CONFIRM_TOKEN",
    "PILOT_BUDGET",
    "ablation_config_dir",
    "load_study",
    "expand_study",
    "load_all_studies",
    "ablation_output_name",
    "select_cells",
    "apply_pilot_budget",
]


def ablation_config_dir(root: str | Path | None = None) -> Path:
    base = Path(__file__).resolve().parents[2] if root is None else Path(root)
    return base / "configs" / "ablations"


def load_study(path: str | Path) -> dict[str, Any]:
    study_path = Path(path)
    with study_path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"Ablation study {study_path} must be a mapping.")
    for key in ("name", "description", "status", "defaults", "cells"):
        if key not in document:
            raise ValueError(f"Ablation study {study_path} is missing {key!r}.")
    if document["status"] != "not_yet_run":
        raise ValueError(
            f"Ablation study {document['name']!r} must stay status 'not_yet_run' "
            "until a completed run is analyzed. Do not mark planned cells as finished."
        )
    if not isinstance(document["cells"], list) or not document["cells"]:
        raise ValueError(f"Ablation study {study_path} has no cells.")
    return document


def _validate_thresholds(row: dict[str, Any], where: str) -> None:
    dorm_off = float(row["dorm_off"])
    dorm_on = float(row["dorm_on"])
    rho_on = float(row["rho_on"])
    rho_off = float(row["rho_off"])
    if not (0.0 <= dorm_off <= dorm_on <= 1.0):
        raise ValueError(f"{where}: require 0 <= dorm_off <= dorm_on <= 1, got {dorm_off}, {dorm_on}.")
    if not (0.0 < rho_on <= rho_off):
        raise ValueError(f"{where}: require 0 < rho_on <= rho_off, got {rho_on}, {rho_off}.")
    if int(row["cooldown_steps"]) < 0:
        raise ValueError(f"{where}: cooldown_steps must be non-negative.")
    if row["probe_source"] not in ("offline_fixed", "online_buffer", "recent_window"):
        raise ValueError(f"{where}: unknown probe_source {row['probe_source']!r}.")
    if row["gate_operator"] not in ("shrink_perturb", "redo"):
        raise ValueError(f"{where}: unknown gate_operator {row['gate_operator']!r}.")


def expand_study(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand defaults × cells × datasets × seeds. Every row is not-yet-run."""
    defaults = dict(document["defaults"])
    seeds = [int(seed) for seed in defaults.pop("seeds")]
    datasets = list(defaults.pop("datasets"))
    if not seeds or not datasets:
        raise ValueError(f"Study {document['name']!r} needs at least one seed and dataset.")
    rows: list[dict[str, Any]] = []
    for spec in document["cells"]:
        if "id" not in spec:
            raise ValueError(f"Study {document['name']!r} has a cell without an id.")
        cell_datasets = list(spec.get("datasets", datasets))
        pilot_datasets = set(spec.get("pilot_datasets", cell_datasets))
        for dataset_id in cell_datasets:
            for seed in seeds:
                row = dict(defaults)
                for key, value in spec.items():
                    if key in {"datasets", "pilot_datasets"}:
                        continue
                    row[key] = value
                row.update(
                    {
                        "study": document["name"],
                        "cell_id": spec["id"],
                        "dataset_id": dataset_id,
                        "seed": int(seed),
                        "status": "not_yet_run",
                        "include_in_pilot": bool(spec.get("include_in_pilot", False))
                        and dataset_id in pilot_datasets
                        and int(seed) == seeds[0],
                    }
                )
                _validate_thresholds(row, f"{document['name']}:{spec['id']}:{dataset_id}:seed{seed}")
                rows.append(row)
    return rows


def load_all_studies(root: str | Path | None = None) -> list[dict[str, Any]]:
    config_dir = ablation_config_dir(root)
    rows: list[dict[str, Any]] = []
    for name in STUDY_FILES:
        path = config_dir / name
        rows.extend(expand_study(load_study(path)))
    identities = [ablation_output_name(row, pilot=False) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Ablation output names are not unique.")
    return rows


def ablation_output_name(cell: dict[str, Any], *, pilot: bool) -> str:
    prefix = "PILOT-ABL" if pilot else "ABL"
    return (
        f"{prefix}-{cell['study']}-{cell['cell_id']}-"
        f"{cell['dataset_id']}-{cell['controller']}-seed{int(cell['seed'])}"
    )


def select_cells(cells: list[dict[str, Any]], *, pilot: bool, study: str | None = None) -> list[dict[str, Any]]:
    selected = cells
    if study is not None:
        selected = [cell for cell in selected if cell["study"] == study]
        if not selected:
            known = sorted({cell["study"] for cell in cells})
            raise ValueError(f"Unknown ablation study {study!r}. Known: {known}.")
    if pilot:
        selected = [cell for cell in selected if cell["include_in_pilot"]]
    return selected


def apply_pilot_budget(cell: dict[str, Any]) -> dict[str, Any]:
    updated = dict(cell)
    updated.update(PILOT_BUDGET)
    updated["status"] = "not_yet_run"
    updated["pilot"] = True
    return updated
