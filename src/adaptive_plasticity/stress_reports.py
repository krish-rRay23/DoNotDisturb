"""Coverage report for the actuator-crippling stress study.

Numbers in ``MANUSCRIPT_STRESS`` are the rounded values printed in
``research_paper/iclr2027/iclr2027_conference.tex`` (Table ``tab:stress_results``).
They are not recomputed here. Local summaries are reported separately and are
never written back as manuscript results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: Rounded manuscript table. CapacityGate is reported for seed 0 only.
MANUSCRIPT_STRESS: dict[tuple[str, str], dict[int, float]] = {
    ("halfcheetah-medium-v2", "none"): {0: 7.99, 1: 5.49, 2: 5.21},
    ("halfcheetah-medium-v2", "fixed"): {0: 15.29, 1: 12.32, 2: 18.18},
    ("halfcheetah-medium-v2", "redo"): {0: 11.40, 1: 12.25, 2: 17.29},
    ("halfcheetah-medium-v2", "capacity_gate"): {0: 8.58},
    ("walker2d-medium-v2", "none"): {0: 7.23, 1: 10.93, 2: 6.48},
    ("walker2d-medium-v2", "fixed"): {0: -0.71, 1: -0.15, 2: -0.73},
    ("walker2d-medium-v2", "redo"): {0: 8.45, 1: 9.34, 2: 6.29},
    ("walker2d-medium-v2", "capacity_gate"): {0: 7.23},
}

MANUSCRIPT_STRESS_SUMMARY: dict[tuple[str, str], str] = {
    ("halfcheetah-medium-v2", "none"): "6.23 ± 1.53",
    ("halfcheetah-medium-v2", "fixed"): "15.26 ± 2.93",
    ("halfcheetah-medium-v2", "redo"): "13.64 ± 3.18",
    ("halfcheetah-medium-v2", "capacity_gate"): "8.58 (seed 0 only)",
    ("walker2d-medium-v2", "none"): "8.21 ± 2.38",
    ("walker2d-medium-v2", "fixed"): "-0.53 ± 0.33",
    ("walker2d-medium-v2", "redo"): "8.03 ± 1.57",
    ("walker2d-medium-v2", "capacity_gate"): "7.23 (seed 0 only; abstains)",
}

STRESS_ENVS: tuple[str, ...] = ("halfcheetah-medium-v2", "walker2d-medium-v2")
STRESS_CONTROLLERS: tuple[str, ...] = ("none", "fixed", "redo", "capacity_gate")
PUBLISHED_SEEDS: tuple[int, ...] = (0, 1, 2)

__all__ = [
    "MANUSCRIPT_STRESS",
    "MANUSCRIPT_STRESS_SUMMARY",
    "STRESS_ENVS",
    "STRESS_CONTROLLERS",
    "PUBLISHED_SEEDS",
    "extension_cells",
    "load_summaries",
    "coverage_report",
    "render_coverage_report",
]


def extension_cells(seeds: tuple[int, ...] = (0, 1, 2, 3, 4)) -> list[dict[str, Any]]:
    """List stress cells for ``seeds`` and mark which ones the manuscript reports."""
    cells: list[dict[str, Any]] = []
    for dataset_id in STRESS_ENVS:
        for controller in STRESS_CONTROLLERS:
            reported = MANUSCRIPT_STRESS[(dataset_id, controller)]
            for seed in seeds:
                in_manuscript = int(seed) in reported
                cells.append(
                    {
                        "dataset_id": dataset_id,
                        "controller": controller,
                        "seed": int(seed),
                        "shift": "actuator_cripple",
                        "in_manuscript": in_manuscript,
                        "manuscript_score": reported.get(int(seed)),
                        "status": "reported" if in_manuscript else "not_yet_run",
                    }
                )
    return cells


def _summary_shift(summary: dict[str, Any]) -> str | None:
    """Shift recorded on a run summary. Protocol is only a fallback."""
    shift = summary.get("shift")
    if shift is None and isinstance(summary.get("protocol"), dict):
        shift = summary["protocol"].get("shift")
    if shift is None:
        return None
    return str(shift)


def _final_normalized(summary: dict[str, Any]) -> float | None:
    scores = summary.get("normalized")
    if isinstance(scores, list) and scores:
        try:
            return float(scores[-1])
        except (TypeError, ValueError):
            return None
    return None


def load_summaries(indir: str | Path) -> list[dict[str, Any]]:
    """Read ``summary.json`` files under a results directory. Missing dir → []."""
    root = Path(indir)
    if not root.is_dir():
        return []
    summaries: list[dict[str, Any]] = []
    for path in sorted(root.glob("**/summary.json")):
        try:
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return summaries


def coverage_report(
    summaries: list[dict[str, Any]],
    *,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    match_atol: float = 0.02,
) -> dict[str, Any]:
    """Compare local summaries to the manuscript table without rewriting either.

    A local score 'matches' a manuscript entry only when both the cell identity
    and the rounded value agree within ``match_atol``. Mismatches are listed.
    Cells the manuscript does not report stay ``not_yet_run`` even if a local
    file is absent.
    """
    by_key: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for summary in summaries:
        dataset_id = summary.get("dataset_id") or summary.get("environment")
        controller = summary.get("controller")
        seed = summary.get("seed")
        shift = _summary_shift(summary)
        if dataset_id is None or controller is None or seed is None or shift is None:
            continue
        # The manuscript table is the actuator-crippling study. A same-seed
        # confirmatory run must not be scored against that table.
        if shift != "actuator_cripple":
            continue
        by_key[(str(dataset_id), str(controller), shift, int(seed))] = summary

    rows: list[dict[str, Any]] = []
    n_present = 0
    n_match = 0
    n_mismatch = 0
    for cell in extension_cells(seeds):
        key = (cell["dataset_id"], cell["controller"], cell["shift"], cell["seed"])
        summary = by_key.get(key)
        local_score = _final_normalized(summary) if summary is not None else None
        present = local_score is not None
        manuscript_score = cell["manuscript_score"]
        match: bool | None
        if manuscript_score is None or local_score is None:
            match = None
        else:
            match = abs(float(local_score) - float(manuscript_score)) <= match_atol
        if present:
            n_present += 1
        if match is True:
            n_match += 1
        elif match is False:
            n_mismatch += 1
        status = cell["status"]
        if not cell["in_manuscript"]:
            status = "local_summary_present" if present else "not_yet_run"
        elif present and match is False:
            status = "local_summary_disagrees_with_manuscript"
        elif present and match is True:
            status = "local_summary_matches_manuscript_rounding"
        rows.append(
            {
                **cell,
                "local_score": local_score,
                "present": present,
                "matches_manuscript_rounding": match,
                "status": status,
            }
        )
    return {
        "source": "research_paper/iclr2027/iclr2027_conference.tex Table tab:stress_results",
        "n_cells": len(rows),
        "n_local_summaries": n_present,
        "n_rounding_matches": n_match,
        "n_rounding_mismatches": n_mismatch,
        "rows": rows,
    }


def render_coverage_report(report: dict[str, Any]) -> str:
    """Text report of which stress cells have local summaries. It does not fill a missing cell."""
    lines = [
        "Stress-test coverage (manuscript numbers are copied from the tex table; local files are separate).",
        f"Source: {report['source']}",
        (
            f"Cells in this plan: {report['n_cells']}. "
            f"Local summaries: {report['n_local_summaries']}. "
            f"Rounding matches: {report['n_rounding_matches']}. "
            f"Rounding mismatches: {report['n_rounding_mismatches']}."
        ),
        "",
        "environment | controller | seed | manuscript | local | status",
    ]
    for row in report["rows"]:
        manuscript = "—" if row["manuscript_score"] is None else f"{row['manuscript_score']:.2f}"
        local = "—" if row["local_score"] is None else f"{row['local_score']:.4f}"
        lines.append(
            f"{row['dataset_id']} | {row['controller']} | {row['seed']} | "
            f"{manuscript} | {local} | {row['status']}"
        )
    lines.append("")
    lines.append("Manuscript mean ± std (not recomputed):")
    for key, text in MANUSCRIPT_STRESS_SUMMARY.items():
        lines.append(f"  {key[0]} {key[1]}: {text}")
    lines.append("")
    lines.append(
        "Seeds the manuscript does not report are not-yet-run. "
        "Do not paste local pilot scores into the paper."
    )
    return "\n".join(lines) + "\n"
