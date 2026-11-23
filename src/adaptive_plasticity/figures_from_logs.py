"""Draw score bars from supplied run summaries.

Every plotted number is the mean of scores in those summaries. The module
does not look up a manuscript table and does not fill a missing controller.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

__all__ = ["scores_by_controller", "write_score_figure"]


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


def _controller_name(summary: dict[str, Any]) -> Any:
    """Factorial summaries use ``controller``. M5 summaries use ``controller_type``."""
    controller = summary.get("controller")
    if controller is None:
        controller = summary.get("controller_type")
    return controller


def scores_by_controller(summaries: list[dict[str, Any]]) -> dict[str, list[float]]:
    """Final numeric scores grouped by controller. Empty groups are omitted."""
    grouped: dict[str, list[float]] = {}
    for summary in summaries:
        controller = _controller_name(summary)
        score = _final_score(summary)
        if controller is None or score is None:
            continue
        grouped.setdefault(str(controller), []).append(score)
    return grouped


def write_score_figure(summaries: list[dict[str, Any]], outdir: str | Path) -> dict[str, Any]:
    """Write a bar chart and ``plotted.json``, or write nothing if there are no scores."""
    grouped = scores_by_controller(summaries)
    destination = Path(outdir)
    if not grouped:
        return {
            "ok": False,
            "reason": "no numeric scores",
            "wrote_figure": False,
            "banner": "No figure was written. Missing scores were not filled from the manuscript.",
        }
    destination.mkdir(parents=True, exist_ok=True)
    controllers = sorted(grouped)
    means = [float(np.mean(grouped[name])) for name in controllers]
    payload = {
        "banner": "Plotted from the supplied summaries. Not a manuscript figure.",
        "controllers": {
            name: {"n": len(grouped[name]), "mean": mean}
            for name, mean in zip(controllers, means)
        },
    }
    (destination / "plotted.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    figure, axis = plt.subplots(figsize=(6, 3.5))
    axis.bar(controllers, means, color="#4c78a8", edgecolor="black")
    axis.set_ylabel("Mean of supplied final scores")
    axis.set_title("Not a manuscript figure")
    figure.tight_layout()
    figure_path = destination / "scores_from_logs.png"
    figure.savefig(figure_path)
    plt.close(figure)
    return {
        "ok": True,
        "reason": "plotted supplied means",
        "wrote_figure": True,
        "figure": str(figure_path),
        "plotted": payload,
    }
