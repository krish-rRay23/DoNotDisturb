"""Post-run analysis for the frozen FINAL-v1 study (read-only pipeline).

Consumes the completed run directories of the 135-cell primary matrix
(3 envs x 3 regimes x none/fixed/capacity_gate x 5 seeds) plus the 15-cell
M10-reference plan (halfcheetah-medium-v2 x 3 regimes x m10ref x 5 seeds)
and produces per-run scores, learning curves, env x regime x method
aggregation, IQM with 95% stratified bootstrap CIs, pairwise probability of
improvement, CapacityGate activation/capacity diagnostics, clean-vs-shifted
comparisons, paper-ready CSV/JSON tables, and publication-quality plots.

Read-only by construction: this module never imports training code paths,
never modifies run directories, plans, thresholds, or protocol values, and
never invents results. Statistics follow Agarwal et al. (2021): IQM with
stratified bootstrap percentile CIs (resample seeds within each
env x regime group), hand-rolled with numpy (no rliable dependency).

Completeness gating: the default mode REFUSES to analyze incomplete data
(exit 2 with a reason-coded report). ``--audit-only`` writes just the
completeness audit. ``--allow-partial`` analyzes the complete subset with
an explicit PARTIAL-DATA banner stamped into every artifact (dev use only).

Score definition: the per-run final normalized score is the LAST element
of the run's ``normalized`` evaluation trajectory (``normalized[-1]``).
Note this differs from ``FINAL_manifest.jsonl``, whose ``score`` field is
``best_normalized``; the choice is recorded in every output artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adaptive_plasticity.final_runner import (  # read-only constants only
    FINAL_ENVS,
    FINAL_SEEDS,
    FINAL_SHIFTS,
    PRIMARY_CONTROLLERS,
    PROTOCOL_VERSION,
    REFERENCE_CONTROLLERS,
    REFERENCE_ENVS,
)

#: Expected study shape (validated against the plan files, never assumed).
EXPECTED_PRIMARY_CELLS: int = 135
EXPECTED_REFERENCE_CELLS: int = 15

#: Score definition recorded in every artifact.
SCORE_DEFINITION: str = "last-normalized-eval"

#: Summary keys required for a run to count as complete.
REQUIRED_SUMMARY_KEYS: tuple[str, ...] = (
    "output_name", "dataset_id", "controller", "shift", "seed",
    "normalized", "returns", "applied_intervention_steps",
    "online_steps", "protocol",
)

#: Protocol budget fields that must match the plan (catches stale runs).
PROTOCOL_BUDGET_KEYS: tuple[str, ...] = (
    "protocol_version", "offline_warmup_updates", "online_steps",
    "shift_step", "eval_interval", "eval_episodes", "online_ratio",
    "update_freq",
)

#: Audit status codes.
STATUS_COMPLETE = "complete"
STATUS_MISSING = "missing"

FIGURE_DPI: int = 150

_METHOD_ORDER = ("none", "fixed", "capacity_gate")
_REF_ORDER = ("m10ref",)


# ---------------------------------------------------------------------------
# Plans + audit
# ---------------------------------------------------------------------------

def load_plan(path: str | Path) -> tuple[list[dict], dict]:
    """Load a plan file; return (cells in file order, protocol)."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(doc["cells"]), dict(doc["protocol"])


def _protocol_mismatch(summary_proto: dict, plan_proto: dict) -> str | None:
    """Return a mismatch description, or None when budgets agree."""
    for key in PROTOCOL_BUDGET_KEYS:
        if summary_proto.get(key) != plan_proto.get(key):
            return f"{key}: run={summary_proto.get(key)!r} plan={plan_proto.get(key)!r}"
    return None


def audit_cell(results_dir: str | Path, cell: dict, plan_protocol: dict) -> dict:
    """Classify one planned cell. Read-only; never creates or launches."""
    out = {"output_name": cell["output_name"], "status": STATUS_MISSING,
           "reason": "no-summary", "detail": None}
    summary_path = Path(results_dir) / cell["output_name"] / "summary.json"
    if not summary_path.is_file():
        return out
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        out["reason"] = "unparseable-summary"
        out["detail"] = repr(exc)
        return out
    if not isinstance(summary, dict):
        out["reason"] = "unparseable-summary"
        return out
    for key in REQUIRED_SUMMARY_KEYS:
        if key not in summary:
            out["reason"] = "missing-key"
            out["detail"] = key
            return out
    for key in ("dataset_id", "controller", "shift", "seed"):
        if summary.get(key) != cell.get(key):
            out["reason"] = "identity-mismatch"
            out["detail"] = f"{key}: summary={summary.get(key)!r} plan={cell.get(key)!r}"
            return out
    cell_proto = cell.get("protocol") or plan_protocol
    mismatch = _protocol_mismatch(summary.get("protocol", {}), cell_proto)
    if mismatch is not None:
        out["reason"] = "protocol-mismatch"
        out["detail"] = mismatch
        return out
    if not isinstance(summary["normalized"], list) or not summary["normalized"]:
        out["reason"] = "empty-trajectory"
        return out
    cell_dir = Path(results_dir) / cell["output_name"]
    if summary.get("controller") == "capacity_gate":
        gate_log = cell_dir / f"gate_{cell['output_name']}.jsonl"
        if not gate_log.is_file() or gate_log.stat().st_size == 0:
            out["reason"] = "missing-gate-log"
            return out
    if summary.get("controller") == "m10ref":
        ref_log = cell_dir / f"m10ref_{cell['output_name']}.jsonl"
        if not ref_log.is_file() or ref_log.stat().st_size == 0:
            out["reason"] = "missing-m10ref-log"
            return out
    if not (cell_dir / "checkpoint.pt").is_file():
        out["reason"] = "missing-checkpoint"
        return out
    out["status"] = STATUS_COMPLETE
    out["reason"] = None
    return out


def audit_study(results_dir: str | Path, primary_cells: list[dict],
                reference_cells: list[dict], plan_protocol: dict) -> dict:
    """Audit planned cells in deterministic plan order."""
    primary = [audit_cell(results_dir, c, c.get("protocol", plan_protocol)) for c in primary_cells]
    reference = [audit_cell(results_dir, c, c.get("protocol", plan_protocol)) for c in reference_cells]
    n_pri_complete = sum(1 for a in primary if a["status"] == STATUS_COMPLETE)
    n_ref_complete = sum(1 for a in reference if a["status"] == STATUS_COMPLETE)
    n_pri_total = len(primary_cells)
    n_ref_total = len(reference_cells)

    pri_ok = (n_pri_total > 0) and (n_pri_complete == n_pri_total)
    ref_ok = (n_ref_total == 0) or (n_ref_complete == n_ref_total)
    overall_complete = pri_ok and ref_ok

    return {
        "n_primary": n_pri_total,
        "n_reference": n_ref_total,
        "n_complete": n_pri_complete + n_ref_complete,
        "n_primary_complete": n_pri_complete,
        "n_reference_complete": n_ref_complete,
        "primary_complete": pri_ok,
        "audit_status": "complete" if pri_ok else "incomplete",
        "complete": overall_complete,
        "cells": {"primary": primary, "reference": reference},
    }


def audit_plan(results_dir: str | Path) -> dict:
    """Helper for external callers to audit study completeness."""
    results_dir = Path(results_dir)
    primary_cells, plan_protocol = load_plan(results_dir / "FINAL_plan.json")
    if (results_dir / "FINAL_reference_plan.json").is_file():
        try:
            reference_cells, _ = load_plan(results_dir / "FINAL_reference_plan.json")
        except Exception:
            reference_cells = []
    else:
        reference_cells = []
    return audit_study(results_dir, primary_cells, reference_cells, plan_protocol)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def load_gate_log(results_dir: str | Path, output_name: str) -> list[dict]:
    """Load gate telemetry records sorted by step (empty list if absent)."""
    path = Path(results_dir) / output_name / f"gate_{output_name}.jsonl"
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    records.sort(key=lambda r: r.get("step", 0))
    return records


def extract_run(results_dir: str | Path, cell: dict) -> dict:
    """Per-run record: LAST-normalized score plus trajectories and metadata.

    Caller must only pass cells the audit classified complete.
    """
    summary = json.loads(
        (Path(results_dir) / cell["output_name"] / "summary.json").read_text(
            encoding="utf-8"))
    normalized = [float(v) for v in summary["normalized"]]
    returns = [float(v) for v in summary["returns"]]
    eval_interval = int(summary["protocol"]["eval_interval"])
    eval_steps = [(i + 1) * eval_interval for i in range(len(normalized))]
    gate = summary.get("gate", {})
    return {
        "output_name": cell["output_name"],
        "env": cell["dataset_id"],
        "regime": cell["shift"],
        "method": cell["controller"],
        "seed": int(cell["seed"]),
        "severity": float(cell["severity"]),
        "score": normalized[-1],
        "final_return": returns[-1],
        "eval_steps": eval_steps,
        "normalized": normalized,
        "returns": returns,
        "applied_intervention_steps": int(summary["applied_intervention_steps"]),
        "gate_applicable": bool(gate.get("applicable", False)),
        "gate_intervention_frequency": float(gate.get("intervention_frequency", 0.0))
        if gate.get("applicable") else 0.0,
        "gate_n_evaluations": int(gate.get("n_evaluations", 0))
        if gate.get("applicable") else 0,
        "rank_anchor": gate.get("rank_anchor"),
    }


# ---------------------------------------------------------------------------
# Aggregation + robust statistics (Agarwal et al. 2021, hand-rolled)
# ---------------------------------------------------------------------------

def aggregate(records: list[dict]) -> list[dict]:
    """Group records by (env, regime, method); seeds sorted deterministically."""
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for rec in records:
        groups.setdefault((rec["env"], rec["regime"], rec["method"]), []).append(rec)
    rows = []
    for key in sorted(groups):
        recs = sorted(groups[key], key=lambda r: r["seed"])
        scores = np.array([r["score"] for r in recs], dtype=float)
        rows.append({
            "env": key[0], "regime": key[1], "method": key[2],
            "n": len(recs), "seeds": [r["seed"] for r in recs],
            "scores": [float(v) for v in scores],
            "mean": float(scores.mean()),
            "std": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
            "median": float(np.median(scores)),
            "min": float(scores.min()), "max": float(scores.max()),
        })
    return rows


def iqm(scores) -> float:
    """Interquartile mean: mean of the middle 50% of sorted scores."""
    x = np.sort(np.asarray(scores, dtype=float))
    n = x.size
    if n == 0:
        return float("nan")
    lo = int(np.floor(0.25 * n))
    hi = int(np.ceil(0.75 * n))
    mid = x[lo:hi]
    if mid.size == 0:
        return float(x[n // 2])
    return float(mid.mean())


def stratified_bootstrap_ci(groups: list[np.ndarray], stat_fn,
                            *, replicates: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile CI: resample seeds with replacement within each group."""
    rng = np.random.RandomState(int(seed))
    groups = [np.asarray(g, dtype=float) for g in groups]
    estimates = np.empty(replicates, dtype=float)
    for b in range(replicates):
        pooled = np.concatenate([
            g[rng.randint(0, g.size, size=g.size)] for g in groups])
        estimates[b] = stat_fn(pooled)
    if not np.all(np.isfinite(estimates)):
        return float("nan"), float("nan")
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def iqm_with_ci(group_scores: list[np.ndarray], *, replicates: int = 2000,
                seed: int = 0) -> dict:
    """IQM over pooled group scores plus 95% stratified bootstrap CI."""
    pooled = np.concatenate([np.asarray(g, dtype=float) for g in group_scores])
    point = iqm(pooled)
    lo, hi = stratified_bootstrap_ci(group_scores, iqm,
                                     replicates=replicates, seed=seed)
    return {"iqm": point, "ci_low": lo, "ci_high": hi, "n_scores": int(pooled.size)}


def _pairwise_win_rate(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    wins = (a[:, None] > b[None, :]).mean()
    ties = (a[:, None] == b[None, :]).mean()
    return float(wins + 0.5 * ties)


def prob_improvement(a_groups: list[np.ndarray], b_groups: list[np.ndarray], *,
                     replicates: int = 2000, seed: int = 0) -> dict:
    """P(score_A > score_B): seed-pair win rate per group, averaged over groups.

    95% CI via stratified bootstrap over seeds within each group.
    """
    a_groups = [np.asarray(g, dtype=float) for g in a_groups]
    b_groups = [np.asarray(g, dtype=float) for g in b_groups]
    point = float(np.mean([_pairwise_win_rate(a, b)
                           for a, b in zip(a_groups, b_groups)]))
    rng = np.random.RandomState(int(seed))
    estimates = np.empty(replicates, dtype=float)
    for i in range(replicates):
        estimates[i] = np.mean([
            _pairwise_win_rate(
                a[rng.randint(0, a.size, size=a.size)],
                b[rng.randint(0, b.size, size=b.size)])
            for a, b in zip(a_groups, b_groups)])
    return {"prob": point, "ci_low": float(np.percentile(estimates, 2.5)),
            "ci_high": float(np.percentile(estimates, 97.5))}


# ---------------------------------------------------------------------------
# Gate activation + capacity trajectories + clean-vs-shifted
# ---------------------------------------------------------------------------

def gate_activation(records: list[dict], gate_interval: int) -> dict:
    """Activation frequency, ON-run durations, and onset from gate records."""
    flags = [bool(r.get("gate_on", False)) for r in records]
    n = len(flags)
    runs: list[int] = []
    current = 0
    onset = None
    for i, on in enumerate(flags):
        if on:
            if onset is None:
                onset = records[i].get("step")
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return {
        "n_records": n,
        "fraction_on": float(sum(flags) / n) if n else 0.0,
        "final_frequency": float(records[-1].get("intervention_frequency", 0.0)) if n else 0.0,
        "max_on_records": int(max(runs)) if runs else 0,
        "mean_on_records": float(np.mean(runs)) if runs else 0.0,
        "max_on_steps": int(max(runs)) * int(gate_interval) if runs else 0,
        "onset_step": onset,
        "n_activations": len(runs),
    }


def clean_vs_shifted(group_rows: list[dict]) -> list[dict]:
    """Per env x method: shifted-vs-clean absolute and relative change."""
    by_key = {(r["env"], r["regime"], r["method"]): r for r in group_rows}
    out = []
    for env in sorted({r["env"] for r in group_rows}):
        for method in sorted({r["method"] for r in group_rows}):
            clean = by_key.get((env, "none", method))
            if clean is None:
                continue
            for regime in sorted({r["regime"] for r in group_rows} - {"none"}):
                shifted = by_key.get((env, regime, method))
                if shifted is None:
                    continue
                drop = shifted["mean"] - clean["mean"]
                out.append({
                    "env": env, "method": method, "regime": regime,
                    "clean_mean": clean["mean"], "shifted_mean": shifted["mean"],
                    "abs_change": drop,
                    "rel_change": (drop / clean["mean"]
                                   if clean["mean"] != 0 else float("nan")),
                })
    return out


# ---------------------------------------------------------------------------
# Tables + plots
# ---------------------------------------------------------------------------

def _apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": FIGURE_DPI,
        "savefig.dpi": FIGURE_DPI,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def write_tables(group_rows: list[dict], path: str | Path) -> Path:
    """Paper-ready CSV: one row per env x regime x method."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "env": r["env"], "regime": r["regime"], "method": r["method"],
        "n": r["n"], "mean": r["mean"], "std": r["std"], "median": r["median"],
        "min": r["min"], "max": r["max"],
        "scores": ";".join(f"{v:.6f}" for v in r["scores"]),
    } for r in group_rows]).to_csv(path, index=False)
    return path


def _curve_axes(records: list[dict], envs: list[str], regimes: list[str],
                methods: list[str], title: str) -> plt.Figure:
    _apply_style()
    fig, axes = plt.subplots(len(envs), len(regimes),
                             figsize=(4 * len(regimes), 3 * len(envs)),
                             sharex=True, sharey=False, squeeze=False)
    by_key: dict[tuple[str, str, str], list[dict]] = {}
    for rec in records:
        by_key.setdefault((rec["env"], rec["regime"], rec["method"]), []).append(rec)
    for i, env in enumerate(envs):
        for j, regime in enumerate(regimes):
            ax = axes[i][j]
            for method in methods:
                recs = sorted(by_key.get((env, regime, method), []),
                              key=lambda r: r["seed"])
                if not recs:
                    continue
                steps = recs[0]["eval_steps"]
                stack = np.array([r["normalized"] for r in recs], dtype=float)
                mean, std = stack.mean(axis=0), stack.std(axis=0)
                ax.plot(steps, mean, label=method)
                ax.fill_between(steps, mean - std, mean + std, alpha=0.2)
            ax.set_title(f"{env} / {regime}")
            ax.set_xlabel("online steps")
            ax.set_ylabel("normalized score")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_learning_curves(records: list[dict], plots_dir: Path) -> Path:
    """Mean +/- std normalized-score curves per env x regime (primary)."""
    envs = sorted({r["env"] for r in records})
    regimes = sorted({r["regime"] for r in records})
    fig = _curve_axes(records, envs, regimes, list(_METHOD_ORDER),
                      "FINAL-v1 primary learning curves (mean +/- std over seeds)")
    path = plots_dir / "learning_curves.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_reference_curves(records: list[dict], plots_dir: Path) -> Path:
    """M10-reference learning curves (kept separate from primary)."""
    if not records:
        _apply_style()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title("M10-reference learning curves (none executed)")
        path = plots_dir / "reference_curves.png"
        fig.savefig(path)
        plt.close(fig)
        return path
    envs = sorted({r["env"] for r in records})
    regimes = sorted({r["regime"] for r in records})
    fig = _curve_axes(records, envs, regimes, list(_REF_ORDER),
                      "FINAL-v1 M10-reference learning curves (mean +/- std over seeds)")
    path = plots_dir / "reference_curves.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_iqm(iqm_by_method: dict, plots_dir: Path) -> Path:
    """IQM per primary method with 95% stratified bootstrap CIs."""
    _apply_style()
    methods = [m for m in _METHOD_ORDER if m in iqm_by_method]
    vals = [iqm_by_method[m]["iqm"] for m in methods]
    lo = [v - iqm_by_method[m]["ci_low"] for v, m in zip(vals, methods)]
    hi = [iqm_by_method[m]["ci_high"] - v for v, m in zip(vals, methods)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(methods, vals, yerr=[lo, hi], capsize=5)
    ax.set_title("FINAL-v1 primary IQM with 95% stratified bootstrap CIs")
    ax.set_ylabel("IQM normalized score (last eval)")
    fig.tight_layout()
    path = plots_dir / "iqm.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_prob_improvement(prob_matrix: dict, plots_dir: Path) -> Path:
    """Heatmap of P(row method > column method)."""
    _apply_style()
    methods = [m for m in _METHOD_ORDER if m in prob_matrix]
    mat = np.array([[prob_matrix[a][b]["prob"] for b in methods]
                    for a in methods])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mat, vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(methods)), methods, rotation=20)
    ax.set_yticks(range(len(methods)), methods)
    for i in range(len(methods)):
        for j in range(len(methods)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center")
    ax.set_title("P(improvement): P(row > column)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    path = plots_dir / "prob_improvement.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_gate_activation(gate_by_run: dict, plots_dir: Path) -> Path:
    """Raster of gate ON/OFF per capacity_gate run (rows sorted by group)."""
    _apply_style()
    names = sorted(gate_by_run)
    if not names:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.set_title("Gate activation: no capacity_gate runs")
        path = plots_dir / "gate_activation.png"
        fig.savefig(path)
        plt.close(fig)
        return path
    width = max(len(v) for v in gate_by_run.values())
    mat = np.zeros((len(names), width))
    for i, name in enumerate(names):
        flags = gate_by_run[name]
        mat[i, :len(flags)] = np.asarray(flags, dtype=float)
    fig, ax = plt.subplots(figsize=(max(6, width * 0.5), max(3, len(names) * 0.25)))
    ax.imshow(mat, aspect="auto", interpolation="nearest")
    ax.set_title("CapacityGate ON (yellow) / OFF (dark) per run x gate eval")
    ax.set_xlabel("gate evaluation index")
    ax.set_ylabel("run (sorted)")
    fig.tight_layout()
    path = plots_dir / "gate_activation.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_capacity_trajectories(records: list[dict], plots_dir: Path) -> Path:
    """rho and dormancy vs online steps for capacity_gate runs (mean +/- std)."""
    _apply_style()
    envs = sorted({r["env"] for r in records})
    regimes = sorted({r["regime"] for r in records})
    fig, axes = plt.subplots(2, max(1, len(envs)),
                             figsize=(4 * max(1, len(envs)), 6),
                             sharex=True, squeeze=False)
    for j, env in enumerate(envs):
        for regime in regimes:
            recs = sorted(
                [r for r in records if r["env"] == env and r["regime"] == regime
                 and r["method"] == "capacity_gate"],
                key=lambda r: r["seed"])
            trajs = [t for t in (rec.get("capacity_traj") for rec in recs)
                     if t]
            if not trajs:
                continue
            steps = trajs[0]["steps"]
            for i, metric in enumerate(("rho", "dormancy")):
                stack = np.array([t[metric] for t in trajs], dtype=float)
                mean = np.nanmean(stack, axis=0)
                std = np.nanstd(stack, axis=0)
                axes[i][j].plot(steps, mean, label=regime)
                axes[i][j].fill_between(steps, mean - std, mean + std, alpha=0.2)
                axes[i][j].set_title(f"{env} / {metric}")
                axes[i][j].set_xlabel("online steps")
        axes[0][j].legend()
    fig.suptitle("Capacity trajectories (mean +/- std over seeds)")
    fig.tight_layout()
    path = plots_dir / "capacity_trajectories.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_clean_shifted(comparisons: list[dict], plots_dir: Path) -> Path:
    """Absolute clean-vs-shifted change per env x method x regime."""
    _apply_style()
    envs = sorted({c["env"] for c in comparisons})
    fig, axes = plt.subplots(1, max(1, len(envs)),
                             figsize=(4 * max(1, len(envs)), 4), squeeze=False)
    for j, env in enumerate(envs):
        ax = axes[0][j]
        rows = [c for c in comparisons if c["env"] == env]
        labels = [f"{c['method']}/{c['regime']}" for c in rows]
        vals = [c["abs_change"] for c in rows]
        ax.bar(range(len(rows)), vals)
        ax.set_xticks(range(len(rows)), labels, rotation=45, ha="right")
        ax.set_title(f"{env}: shifted - clean")
        ax.set_ylabel("abs change in mean normalized score")
        ax.axhline(0.0, color="black", linewidth=0.8)
    fig.tight_layout()
    path = plots_dir / "clean_vs_shifted.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_analysis(*, results_dir: Path, plots_dir: Path,
                 replicates: int = 2000, bootstrap_seed: int = 0,
                 partial: bool = False, primary_cells: list[dict] | None = None,
                 reference_cells: list[dict] | None = None,
                 plan_protocol: dict | None = None) -> dict:
    """Full pipeline on audited-complete data; returns the summary payload.

    Plans load from the results dir (with 135/15 shape validation) unless
    injected directly (used by tests with synthetic fixtures).
    """
    if primary_cells is None or reference_cells is None or plan_protocol is None:
        primary_cells, plan_protocol = load_plan(results_dir / "FINAL_plan.json")
        reference_cells, _ = load_plan(results_dir / "FINAL_reference_plan.json")
        if len(primary_cells) != EXPECTED_PRIMARY_CELLS:
            raise ValueError(
                f"primary plan has {len(primary_cells)} cells, expected "
                f"{EXPECTED_PRIMARY_CELLS}")
        if len(reference_cells) != EXPECTED_REFERENCE_CELLS:
            raise ValueError(
                f"reference plan has {len(reference_cells)} cells, expected "
                f"{EXPECTED_REFERENCE_CELLS}")

    audit = audit_study(results_dir, primary_cells, reference_cells, plan_protocol)
    if not audit.get("primary_complete") and not audit["complete"] and not partial:
        raise RuntimeError(
            f"incomplete study: {audit['n_primary_complete']}/{len(primary_cells)} primary complete; "
            "refusing to analyze (see audit).")

    def _complete(cells, section):
        status = {a["output_name"]: a for a in audit["cells"][section]}
        return [c for c in cells if status[c["output_name"]]["status"] == STATUS_COMPLETE]

    primary_records = [extract_run(results_dir, c)
                       for c in _complete(primary_cells, "primary")]
    reference_records = [extract_run(results_dir, c)
                         for c in _complete(reference_cells, "reference")]

    # Capacity trajectories for gate runs (rho/dormancy vs steps).
    gate_interval = int(plan_protocol.get("gate_interval", 1000))
    gate_by_run: dict[str, list[bool]] = {}
    for rec in primary_records:
        if rec["method"] != "capacity_gate":
            continue
        logs = load_gate_log(results_dir, rec["output_name"])
        gate_by_run[rec["output_name"]] = [bool(r.get("gate_on", False)) for r in logs]
        rec["capacity_traj"] = {
            "steps": [r.get("step", 0) for r in logs],
            "rho": [r.get("rho", float("nan")) for r in logs],
            "dormancy": [r.get("dormancy", float("nan")) for r in logs],
        }
        rec["gate_activation"] = gate_activation(logs, gate_interval)

    group_rows = aggregate(primary_records)
    ref_rows = aggregate(reference_records)

    by_method: dict[str, list[np.ndarray]] = {}
    for row in group_rows:
        by_method.setdefault(row["method"], []).append(
            np.array(row["scores"], dtype=float))
    iqm_by_method = {m: iqm_with_ci(gs, replicates=replicates,
                                    seed=bootstrap_seed)
                     for m, gs in sorted(by_method.items())}

    pairs = [("capacity_gate", "none"), ("capacity_gate", "fixed"), ("fixed", "none")]
    prob_matrix: dict[str, dict[str, dict]] = {m: {} for m in by_method}
    for a, b in pairs:
        if a in by_method and b in by_method:
            prob_matrix[a][b] = prob_improvement(
                by_method[a], by_method[b],
                replicates=replicates, seed=bootstrap_seed)
            rev = prob_improvement(by_method[b], by_method[a],
                                   replicates=replicates,
                                   seed=bootstrap_seed + 1)
            prob_matrix[b][a] = rev
    for m in by_method:
        prob_matrix[m][m] = {"prob": 0.5, "ci_low": 0.5, "ci_high": 0.5}

    comparisons = clean_vs_shifted(group_rows)

    plots_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = [
        str(plot_learning_curves(primary_records, plots_dir)),
        str(plot_reference_curves(reference_records, plots_dir)),
        str(plot_iqm(iqm_by_method, plots_dir)),
        str(plot_prob_improvement(prob_matrix, plots_dir)),
        str(plot_gate_activation(gate_by_run, plots_dir)),
        str(plot_capacity_trajectories(primary_records, plots_dir)),
        str(plot_clean_shifted(comparisons, plots_dir)),
    ]

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "score_definition": SCORE_DEFINITION,
        "partial_data": bool(partial),
        "bootstrap": {"replicates": replicates, "seed": bootstrap_seed},
        "audit": {
            "n_primary": audit["n_primary"],
            "n_reference": audit["n_reference"],
            "n_complete": audit["n_complete"],
            "cells": audit["cells"],
        },
        "runs": primary_records,
        "reference_runs": reference_records,
        "groups": [{k: v for k, v in r.items()} for r in group_rows],
        "reference_groups": [{k: v for k, v in r.items()} for r in ref_rows],
        "iqm_by_method": iqm_by_method,
        "prob_improvement": prob_matrix,
        "clean_vs_shifted": comparisons,
        "figures": figure_paths,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    # Capacity trajectories live in figures; drop bulky per-step arrays from
    # the JSON runs table but keep scalar gate activation stats.
    for rec in payload["runs"]:
        rec.pop("capacity_traj", None)
        rec.pop("normalized", None)
        rec.pop("returns", None)
        rec.pop("eval_steps", None)
    return payload


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Read-only post-run analysis for the frozen FINAL-v1 study.")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--plots-dir", type=str, default="plots/final")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--audit-only", action="store_true",
                        help="Write only the completeness audit, then exit.")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Analyze the complete subset (dev only); every "
                             "artifact is stamped PARTIAL-DATA.")
    args = parser.parse_args(argv)
    results_dir = Path(args.results_dir)
    plots_dir = Path(args.plots_dir)

    primary_cells, plan_protocol = load_plan(results_dir / "FINAL_plan.json")
    reference_cells, _ = load_plan(results_dir / "FINAL_reference_plan.json")
    audit = audit_study(results_dir, primary_cells, reference_cells, plan_protocol)
    audit_path = results_dir / "FINAL_analysis_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, default=str) + "\n",
                          encoding="utf-8")
    print(f"Audit: {audit['n_complete']}/150 complete "
          f"({audit['n_primary_complete']}/135 primary, "
          f"{audit['n_reference_complete']}/15 reference)")
    if args.audit_only:
        print(f"Wrote {audit_path}")
        return
    if not audit.get("primary_complete") and not audit["complete"] and not args.allow_partial:
        print("ERROR: study incomplete — refusing to analyze. "
              "Re-run with --audit-only for the reason-coded report.",
              file=sys.stderr, flush=True)
        for section in ("primary", "reference"):
            for a in audit["cells"][section]:
                if a["status"] != STATUS_COMPLETE:
                    print(f"  {section}/{a['output_name']}: "
                          f"{a['status']}:{a['reason']}",
                          file=sys.stderr, flush=True)
        sys.exit(2)
    if args.allow_partial and not audit["complete"]:
        print("WARNING: PARTIAL-DATA mode — results cover only the "
              f"{audit['n_complete']}/150 complete runs.", flush=True)

    payload = run_analysis(results_dir=results_dir, plots_dir=plots_dir,
                           replicates=int(args.bootstrap_replicates),
                           bootstrap_seed=int(args.bootstrap_seed),
                           partial=bool(args.allow_partial))
    summary_path = results_dir / "FINAL_analysis_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, default=str) + "\n",
                            encoding="utf-8")
    primary_groups = [r for r in payload["groups"]]
    ref_groups = [r for r in payload["reference_groups"]]
    write_tables(primary_groups, results_dir / "FINAL_paper_tables.csv")
    write_tables(ref_groups, results_dir / "FINAL_reference_tables.csv")
    print(f"Wrote {summary_path}")
    print("Wrote FINAL_paper_tables.csv + FINAL_reference_tables.csv")
    for method in sorted(payload["iqm_by_method"]):
        s = payload["iqm_by_method"][method]
        print(f"  IQM {method}: {s['iqm']:.2f} "
              f"[{s['ci_low']:.2f}, {s['ci_high']:.2f}]")
    for fig in payload["figures"]:
        print(f"Wrote {fig}")


if __name__ == "__main__":
    main()
