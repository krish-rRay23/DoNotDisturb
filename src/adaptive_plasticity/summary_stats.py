"""Aggregate scores from run summaries.

Every number this module returns is computed from the summaries it is given.
It does not read the manuscript and it does not fill in a missing cell.
The interquartile mean and the stratified bootstrap match
``experiments/final_analysis.py`` so a synthetic check can compare them.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
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
    "paired_comparison",
    "paired_effect_size",
]


def iqm(scores: Sequence[float] | np.ndarray) -> float:
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
    stat_fn: Callable[[np.ndarray], float],
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


def _cell_fields(record: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    """Dataset, controller, shift, and seed, with the legacy aliases.

    Stress summaries may store the dataset under ``environment`` and the
    shift under ``protocol``. Those aliases are the same cell, not a new one.
    """
    dataset_id = record.get("dataset_id")
    if dataset_id is None:
        dataset_id = record.get("environment")
    shift = record.get("shift")
    if shift is None and isinstance(record.get("protocol"), dict):
        shift = record["protocol"].get("shift")
    return dataset_id, record.get("controller"), shift, record.get("seed")


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
        dataset_id, controller, shift, seed = _cell_fields(summary)
        complete = None not in (dataset_id, controller, shift, seed)
        duplicate = False
        if complete:
            identity = (dataset_id, controller, shift, seed)
            seen[identity] = seen.get(identity, 0) + 1
            duplicate = seen[identity] > 1
        rows.append(
            {
                "dataset_id": dataset_id,
                "controller": controller,
                "shift": shift,
                "seed": seed,
                "score": _final_score(summary),
                "applied_intervention_steps": summary.get("applied_intervention_steps"),
                "probe_source": summary.get("probe_source"),
                "gate_operator": summary.get("gate_operator"),
                "summary_path": summary.get("_summary_path"),
                "duplicate_identity": duplicate,
                "manuscript_status": "not_a_manuscript_number",
            }
        )
    return rows


def paired_effect_size(differences: Sequence[float] | np.ndarray) -> dict[str, Any]:
    """Cohen's dz for paired differences: mean divided by the sample standard deviation.

    Fewer than two differences, or a zero sample standard deviation, leaves
    the effect size empty. This is not a manuscript effect size.
    """
    values = np.asarray(differences, dtype=float)
    count = int(values.size)
    if count < 2:
        return {
            "n": count,
            "cohens_dz": None,
            "note": "need at least two paired differences for a sample standard deviation",
        }
    std = float(values.std(ddof=1))
    if std == 0.0 or not np.isfinite(std):
        return {
            "n": count,
            "cohens_dz": None,
            "note": "sample standard deviation is zero",
        }
    return {
        "n": count,
        "cohens_dz": float(values.mean() / std),
        "note": "mean paired difference divided by its sample standard deviation; not a manuscript effect size",
    }


def paired_comparison(
    rows: list[dict[str, Any]],
    left_controller: str,
    right_controller: str,
) -> dict[str, Any]:
    """Paired left-minus-right scores on ``(dataset_id, shift, seed)``.

    ``environment`` is accepted as ``dataset_id``, and a shift stored only
    on ``protocol`` is accepted as ``shift``. A cell is paired only when
    both controllers have exactly one numeric score and every identity
    field is present. Incomplete rows are listed one by one so two
    different environments cannot collapse into one pair. Duplicates and
    missing partners stay listed and are not filled in. The Wilcoxon
    statistic is computed from these rows only. It is not a manuscript
    result.
    """
    from scipy.stats import wilcoxon

    def _index(controller: str) -> tuple[dict[tuple[Any, ...], float], list[dict[str, Any]]]:
        grouped: dict[tuple[Any, ...], list[float]] = {}
        skipped: list[dict[str, Any]] = []
        for row in rows:
            dataset_id, row_controller, shift, seed = _cell_fields(row)
            if row_controller != controller:
                continue
            if row.get("score") is None:
                continue
            if None in (dataset_id, shift, seed):
                skipped.append(
                    {
                        "dataset_id": dataset_id,
                        "shift": shift,
                        "seed": seed,
                        "controller": controller,
                        "reason": "incomplete_identity",
                    }
                )
                continue
            identity = (dataset_id, shift, seed)
            grouped.setdefault(identity, []).append(float(row["score"]))
        unique: dict[tuple[Any, ...], float] = {}
        for identity, scores in grouped.items():
            if len(scores) == 1:
                unique[identity] = scores[0]
            else:
                skipped.append(
                    {
                        "dataset_id": identity[0],
                        "shift": identity[1],
                        "seed": identity[2],
                        "controller": controller,
                        "reason": "duplicate_scored_rows",
                    }
                )
        return unique, skipped

    left, left_skipped = _index(left_controller)
    right, right_skipped = _index(right_controller)
    pairs: list[dict[str, Any]] = []
    unpaired: list[dict[str, Any]] = left_skipped + right_skipped
    for identity in sorted(set(left) | set(right), key=lambda item: (str(item[0]), str(item[1]), str(item[2]))):
        if identity not in left or identity not in right:
            present = left_controller if identity in left else right_controller
            unpaired.append(
                {
                    "dataset_id": identity[0],
                    "shift": identity[1],
                    "seed": identity[2],
                    "controller": present,
                    "reason": "missing_partner",
                }
            )
            continue
        difference = left[identity] - right[identity]
        pairs.append(
            {
                "dataset_id": identity[0],
                "shift": identity[1],
                "seed": identity[2],
                "left_score": left[identity],
                "right_score": right[identity],
                "difference": difference,
            }
        )
    differences = np.asarray([pair["difference"] for pair in pairs], dtype=float)
    wilcoxon_statistic = None
    wilcoxon_pvalue = None
    wilcoxon_note = "not_computed"
    if differences.size and np.any(differences != 0):
        try:
            result = wilcoxon(differences, zero_method="wilcox", alternative="two-sided")
            wilcoxon_statistic = float(result.statistic)
            wilcoxon_pvalue = float(result.pvalue)
            wilcoxon_note = "two-sided signed-rank on left minus right; zero differences dropped"
        except ValueError as exc:
            wilcoxon_note = f"not_computed: {exc}"
    elif differences.size:
        wilcoxon_note = "not_computed: all paired differences are zero"
    return {
        "banner": "Computed from the supplied summary.json files. Not a manuscript result.",
        "left_controller": left_controller,
        "right_controller": right_controller,
        "difference": "left_minus_right",
        "n_pairs": int(differences.size),
        "mean_difference": float(differences.mean()) if differences.size else None,
        "sample_std_difference": float(differences.std(ddof=1)) if differences.size > 1 else None,
        "median_difference": float(np.median(differences)) if differences.size else None,
        "n_left_greater": int(np.count_nonzero(differences > 0)) if differences.size else 0,
        "n_right_greater": int(np.count_nonzero(differences < 0)) if differences.size else 0,
        "n_ties": int(np.count_nonzero(differences == 0)) if differences.size else 0,
        "wilcoxon_statistic": wilcoxon_statistic,
        "wilcoxon_pvalue": wilcoxon_pvalue,
        "wilcoxon_note": wilcoxon_note,
        "effect_size": paired_effect_size(differences),
        "multiple_comparisons": {
            "adjustment": "none",
            "n_tests_in_this_report": 1 if differences.size else 0,
            "note": (
                "The p-value in this report is unadjusted. It is not a manuscript "
                "test, and it is not corrected for any other comparison."
            ),
        },
        "pairs": pairs,
        "unpaired": unpaired,
        "manuscript_status": "not_a_manuscript_number",
    }


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
        dataset_id, controller, shift, _seed = _cell_fields(row)
        if row.get("score") is None or controller is None or dataset_id is None or shift is None:
            continue
        strata = by_controller.setdefault(str(controller), {})
        key = (dataset_id, shift)
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
