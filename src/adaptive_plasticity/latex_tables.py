"""Render LaTeX tables from supplied run summaries.

Every number is computed from those summaries. A missing score is omitted,
and a duplicate identity is not averaged in twice. The text says it is not
a manuscript table. Pass ``fixture=True``, or set ``fixture`` on a summary,
and the text is also marked as a synthetic fixture.

The sample standard deviation uses ``ddof=1`` and is left out when a cell
has one score. The IQM block calls ``iqm_by_controller`` on the same rows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from adaptive_plasticity.summary_stats import iqm_by_controller, load_summaries, per_cell_rows

COMPUTED_BANNER = "% Computed from supplied summaries. Not a manuscript table."
FIXTURE_BANNER = "% SYNTHETIC FIXTURE. These numbers are not experimental results."
EMPTY_BANNER = "% No scored rows. No table was rendered."


def normalize_summaries(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy records so M5 field names are visible to the summary aggregator.

    Factorial summaries use ``controller`` and ``shift``. M5 summaries use
    ``controller_type`` and ``shift_type``. A field that is already set is
    kept. A missing or empty ``shift_type`` is the M5 no-shift case, labeled
    ``-noshift``, which is the token ``m5.py`` already uses for that case.
    """
    normalized: list[dict[str, Any]] = []
    for summary in summaries:
        copied: dict[str, Any] | None = None

        def ensure() -> dict[str, Any]:
            nonlocal copied
            if copied is None:
                copied = dict(summary)
            return copied

        if summary.get("controller") is None and summary.get("controller_type") is not None:
            ensure()["controller"] = summary["controller_type"]
        if summary.get("shift") is None and "shift_type" in summary:
            shift_type = summary.get("shift_type")
            ensure()["shift"] = "-noshift" if shift_type in (None, "") else shift_type
        normalized.append(copied if copied is not None else summary)
    return normalized


def _is_fixture(summaries: list[dict[str, Any]], fixture: bool | None) -> bool:
    marked = any(bool(summary.get("fixture")) for summary in summaries)
    if fixture is None:
        return marked
    return bool(fixture) or marked


def usable_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Scored, complete, non-duplicate rows. Incomplete rows stay out of the table."""
    rows: list[dict[str, Any]] = []
    for row in per_cell_rows(normalize_summaries(summaries)):
        score = row.get("score")
        if score is None or row.get("duplicate_identity"):
            continue
        if None in (row.get("dataset_id"), row.get("controller"), row.get("shift"), row.get("seed")):
            continue
        value = float(score)
        if not np.isfinite(value):
            continue
        kept = dict(row)
        kept["score"] = value
        rows.append(kept)
    return rows


def _seed_key(value: Any) -> tuple:
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


def group_cells(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One cell per dataset, shift, and controller, with scores kept by seed."""
    grouped: dict[tuple[Any, Any, Any], list[float]] = {}
    by_seed: dict[tuple[Any, Any, Any], dict[Any, float]] = {}
    for row in rows:
        key = (row["dataset_id"], row["shift"], row["controller"])
        grouped.setdefault(key, []).append(float(row["score"]))
        by_seed.setdefault(key, {})[row["seed"]] = float(row["score"])
    cells: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda item: (str(item[0]), str(item[1]), str(item[2]))):
        values = np.asarray(grouped[key], dtype=float)
        cells.append(
            {
                "dataset_id": key[0],
                "shift": key[1],
                "controller": key[2],
                "n": int(values.size),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if values.size >= 2 else None,
                "median": float(np.median(values)),
                "min": float(values.min()),
                "max": float(values.max()),
                "by_seed": {str(seed): score for seed, score in sorted(by_seed[key].items(), key=lambda item: _seed_key(item[0]))},
            }
        )
    return cells


def _escape(value: Any) -> str:
    text = str(value)
    text = text.replace("\\", r"\textbackslash{}")
    for char, replacement in (("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_")):
        text = text.replace(char, replacement)
    return text


def _fmt(value: float) -> str:
    return f"{float(value):.2f}"


def _mean_tex(cell: dict[str, Any]) -> str:
    if cell["std"] is None:
        return rf"${_fmt(cell['mean'])}$ (n=1)"
    return rf"${_fmt(cell['mean'])} \pm {_fmt(cell['std'])}$"


def _preamble(is_fixture: bool, empty: bool) -> list[str]:
    lines = [COMPUTED_BANNER]
    if is_fixture:
        lines.append(FIXTURE_BANNER)
    if empty:
        lines.append(EMPTY_BANNER)
    return lines


def render_mean_table(
    summaries: list[dict[str, Any]],
    *,
    fixture: bool | None = None,
    replicates: int = 200,
    seed: int = 0,
) -> str:
    """Mean and sample-std table, plus an IQM row per controller."""
    rows = usable_rows(summaries)
    is_fixture = _is_fixture(summaries, fixture)
    cells = group_cells(rows)
    lines = _preamble(is_fixture, not cells)
    if not cells:
        return "\n".join(lines) + "\n"
    lines.extend(
        [
            r"\begin{tabular}{llll}",
            r"\toprule",
            r"Dataset & Shift & Controller & Mean $\pm$ sample std \\",
            r"\midrule",
        ]
    )
    for cell in cells:
        lines.append(
            f"{_escape(cell['dataset_id'])} & {_escape(cell['shift'])} & {_escape(cell['controller'])} & {_mean_tex(cell)} \\\\"
        )
    lines.append(r"\midrule")
    for controller, stats in iqm_by_controller(rows, replicates=replicates, seed=seed).items():
        lines.append(
            "IQM & supplied & "
            f"{_escape(controller)} & ${_fmt(stats['iqm'])}$ "
            f"[{_fmt(stats['ci_low'])}, {_fmt(stats['ci_high'])}] \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            "% Sample standard deviation uses ddof=1 and is omitted when n=1.",
            "% The IQM row is computed from these supplied rows only.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_seed_table(summaries: list[dict[str, Any]], *, fixture: bool | None = None) -> str:
    """One column per seed that appears in the supplied rows. Missing seeds stay blank."""
    rows = usable_rows(summaries)
    is_fixture = _is_fixture(summaries, fixture)
    cells = group_cells(rows)
    lines = _preamble(is_fixture, not cells)
    if not cells:
        return "\n".join(lines) + "\n"
    seed_labels: list[str] = []
    for cell in cells:
        for label in cell["by_seed"]:
            if label not in seed_labels:
                seed_labels.append(label)
    seed_labels = sorted(seed_labels, key=_seed_key)
    header = " & ".join(["Dataset", "Shift", "Controller", *[f"seed {_escape(label)}" for label in seed_labels]])
    lines.extend(
        [
            r"\begin{tabular}{" + ("l" * (3 + len(seed_labels))) + "}",
            r"\toprule",
            header + r" \\",
            r"\midrule",
        ]
    )
    for cell in cells:
        values = [f"${_fmt(cell['by_seed'][label])}$" if label in cell["by_seed"] else "---" for label in seed_labels]
        lines.append(
            " & ".join(
                [
                    _escape(cell["dataset_id"]),
                    _escape(cell["shift"]),
                    _escape(cell["controller"]),
                    *values,
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            "% A blank seed is missing from the supplied summaries. It was not filled.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_tables(
    summaries: list[dict[str, Any]],
    *,
    fixture: bool | None = None,
    replicates: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """Return the TeX strings and the JSON-ready values those strings print."""
    rows = usable_rows(summaries)
    values = {
        "banner": "Computed from supplied summaries. Not a manuscript table.",
        "fixture": _is_fixture(summaries, fixture),
        "n_rows": len(rows),
        "cells": group_cells(rows),
        "iqm_by_controller": iqm_by_controller(rows, replicates=replicates, seed=seed),
        "source": "supplied_summaries",
    }
    return {
        "mean_table": render_mean_table(summaries, fixture=fixture, replicates=replicates, seed=seed),
        "seed_table": render_seed_table(summaries, fixture=fixture),
        "values": values,
    }


def write_tables(
    indir: str | Path,
    outdir: str | Path,
    *,
    fixture: bool = False,
    replicates: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """Write ``cell_means.tex``, ``seed_scores.tex``, and ``values.json``."""
    rendered = build_tables(load_summaries(indir), fixture=fixture, replicates=replicates, seed=seed)
    destination = Path(outdir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "cell_means.tex").write_text(rendered["mean_table"], encoding="utf-8")
    (destination / "seed_scores.tex").write_text(rendered["seed_table"], encoding="utf-8")
    (destination / "values.json").write_text(
        json.dumps(rendered["values"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return rendered
