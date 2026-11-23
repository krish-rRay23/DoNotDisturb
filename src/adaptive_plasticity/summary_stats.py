"""Aggregate scores from run summaries.

Every number this module returns is computed from the summaries it is given.
It does not read the manuscript and it does not fill in a missing cell.
The interquartile mean and the stratified bootstrap match
``experiments/final_analysis.py`` so a synthetic check can compare them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "iqm",
    "stratified_bootstrap_ci",
    "iqm_with_ci",
    "load_summaries",
    "per_cell_rows",
    "iqm_by_controller",
]


def iqm(scores) -> float:
    """Interquartile mean: mean of the middle 50% of sorted scores."""
    values = np.sort(np.asarray(scores, dtype=float))
    count = values.size
    if count == 0:
        return float("nan")
    low = int(np.floor(0.25 * count))
    high = int(np.ceil(0.75 * count))
    middle = values[low:high]
    if middle.size == 0:
        return float(values[count // 2])
    return float(middle.mean())


def stratified_bootstrap_ci(
    groups: list[np.ndarray],
    stat_fn,
    *,
    replicates: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """95% percentile interval, resampling scores inside each group."""
    rng = np.random.RandomState(int(seed))
    prepared = [np.asarray(group, dtype=float) for group in groups]
    if not prepared or any(group.size == 0 for group in prepared):
        return float("nan"), float("nan")
    estimates = np.empty(int(replicates), dtype=float)
    for index in range(int(replicates)):
        pooled = np.concatenate(
            [group[rng.randint(0, group.size, size=group.size)] for group in prepared]
        )
        estimates[index] = stat_fn(pooled)
    if not np.all(np.isfinite(estimates)):
        return float("nan"), float("nan")
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def iqm_with_ci(
    group_scores: list[np.ndarray],
    *,
    replicates: int = 2000,
    seed: int = 0,
) -> dict[str, float | int]:
    """IQM of the pooled scores plus a stratified bootstrap interval."""
    pooled = np.concatenate([np.asarray(group, dtype=float) for group in group_scores]) if group_scores else np.empty(0)
    point = iqm(pooled)
    low, high = stratified_bootstrap_ci(group_scores, iqm, replicates=replicates, seed=seed)
    return {
        "iqm": point,
        "ci_low": low,
        "ci_high": high,
        "n_scores": int(pooled.size),
        "n_groups": len(group_scores),
        "replicates": int(replicates),
        "seed": int(seed),
        "source": "computed_from_supplied_summaries",
    }


def load_summaries(indir: str | Path) -> list[dict[str, Any]]:
    """Read ``summary.json`` files. A missing directory yields an empty list."""
    root = Path(indir)
    if not root.is_dir():
        return []
    loaded: list[dict[str, Any]] = []
    for path in sorted(root.glob("**/summary.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(document, dict):
            document["_summary_path"] = str(path)
            loaded.append(document)
    return loaded


def _final_score(summary: dict[str, Any]) -> float | None:
    scores = summary.get("normalized")
    if isinstance(scores, list) and scores:
        try:
            return float(scores[-1])
        except (TypeError, ValueError):
            return None
    if "score" in summary:
        try:
            return float(summary["score"])
        except (TypeError, ValueError):
            return None
    return None


def per_cell_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per summary. Duplicate identities are kept and flagged.

    Rows without a numeric score are kept with ``score`` null so a missing
    result stays visible.
    """
    seen: dict[tuple[Any, ...], int] = {}
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        identity = (
            summary.get("dataset_id"),
            summary.get("controller"),
            summary.get("shift"),
            summary.get("seed"),
        )
        seen[identity] = seen.get(identity, 0) + 1
        rows.append(
            {
                "dataset_id": summary.get("dataset_id"),
                "controller": summary.get("controller"),
                "shift": summary.get("shift"),
                "seed": summary.get("seed"),
                "score": _final_score(summary),
                "applied_intervention_steps": summary.get("applied_intervention_steps"),
                "probe_source": summary.get("probe_source"),
                "gate_operator": summary.get("gate_operator"),
                "summary_path": summary.get("_summary_path"),
                "duplicate_identity": seen[identity] > 1,
                "manuscript_status": "not_a_manuscript_number",
            }
        )
    return rows


def iqm_by_controller(
    rows: list[dict[str, Any]],
    *,
    replicates: int = 2000,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """IQM per controller. Strata are ``(dataset_id, shift)`` groups.

    A controller with no numeric scores is omitted. The result is labeled
    as computed from the supplied rows.
    """
    by_controller: dict[str, dict[tuple[Any, Any], list[float]]] = {}
    for row in rows:
        if row.get("score") is None or row.get("controller") is None:
            continue
        strata = by_controller.setdefault(str(row["controller"]), {})
        key = (row.get("dataset_id"), row.get("shift"))
        strata.setdefault(key, []).append(float(row["score"]))
    report: dict[str, dict[str, Any]] = {}
    for controller, strata in sorted(by_controller.items()):
        groups = [np.asarray(values, dtype=float) for values in strata.values()]
        stats = iqm_with_ci(groups, replicates=replicates, seed=seed)
        stats["controller"] = controller
        stats["strata"] = [
            {"dataset_id": key[0], "shift": key[1], "n": len(values)}
            for key, values in sorted(strata.items(), key=lambda item: (str(item[0][0]), str(item[0][1])))
        ]
        report[controller] = stats
    return report
