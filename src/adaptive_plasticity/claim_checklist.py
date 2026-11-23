"""Map manuscript claims to the logs that would support them.

This module does not open those logs and does not recompute a number.
A missing working-tree path is ``unverified``. A path that exists is
``present_not_recomputed``: the file was not read. Short pilots and the
known non-manuscript summaries keep their own labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Each log path is relative to the repository root. The supplementary zip
# is not unpacked by this check, so a path that exists only inside that zip
# is still unverified here.
CLAIMS: tuple[dict[str, Any], ...] = (
    {
        "claim_id": "aggregate_iqm",
        "manuscript": (
            "Table tab:main_results and the abstract. "
            "Transcribed tokens: 38.78, 12.30, [37.43, 40.03], [9.36, 15.10]."
        ),
        "script": "experiments/final_analysis.py",
        "logs": ("results/final_study/FINAL_analysis_summary.json",),
        "role": "manuscript",
        "note": (
            "REPRODUCIBILITY.md names this path inside the supplementary zip "
            "as the summary that rounds to the manuscript. This checklist does not open it."
        ),
    },
    {
        "claim_id": "paired_comparison",
        "manuscript": (
            "The aggregate statistical paragraph. Transcribed tokens: "
            "26.03, 19.09, 18.0, 1.44, 9.15, 9.69, 20.72, 31.59, 22.27, 0.933."
        ),
        "script": "experiments/final_analysis.py",
        "logs": ("results/final_study/FINAL_analysis_summary.json",),
        "role": "manuscript",
        "note": "Same archived summary as the IQM row. The statistic is not recomputed here.",
    },
    {
        "claim_id": "primary_cells",
        "manuscript": "Tables tab:main_results and tab:per_seed_full. 135 primary cells.",
        "script": "experiments/final_factorial.py",
        "logs": ("results/final_study",),
        "role": "manuscript",
        "note": "A present directory is not a checked score. summary.json files are not read.",
    },
    {
        "claim_id": "stress_table",
        "manuscript": (
            "Table tab:stress_results and Figure fig:stress_operator_asymmetry. "
            "CapacityGate in that table is seed 0 only."
        ),
        "script": "experiments/stress_factorial.py",
        "logs": ("results/stress_study/STRESS-*/summary.json",),
        "role": "manuscript",
        "note": (
            "Only STRESS- run summaries count. A STABLE_VAL summary in the same directory "
            "is the nine-cell validation and is not this table. Summaries are not read."
        ),
    },
    {
        "claim_id": "stress_telemetry",
        "manuscript": "Section sec:telemetry and Table tab:telemetry_audit. HalfCheetah stress, seed 0.",
        "script": "experiments/stress_factorial.py",
        "logs": (
            "results/stress_study/STRESS-halfcheetah-medium-v2-capacity_gate-actuator_cripple-sev1-seed0/summary.json",
        ),
        "role": "manuscript",
        "note": "The gate log next to that summary is the telemetry trace. It is not read here.",
    },
    {
        "claim_id": "stable_rank",
        "manuscript": "Section sec:abstention. Transcribed tokens: [0.99, 1.03], 0.085.",
        "script": "experiments/final_factorial.py",
        "logs": ("results/final_study",),
        "role": "manuscript",
        "note": "The interval is the manuscript sentence. Gate jsonl files are not parsed.",
    },
    {
        "claim_id": "learning_curves",
        "manuscript": "Figure fig:learning_curves.",
        "script": "experiments/reproduce_all_tables_and_figures.py",
        "logs": (
            "results/reproduced_figures/figure2_online_adaptation_learning_curves.png",
            "plots/final/learning_curves.png",
        ),
        "role": "manuscript",
        "note": (
            "reproduce_all_tables_and_figures.py writes the first path. "
            "final_analysis.py writes the second path under its default --plots-dir. "
            "Neither writer uses results/final_analysis/learning_curves.png. The files are not opened."
        ),
    },
    {
        "claim_id": "older_final_summary",
        "manuscript": "Not a manuscript table. Do not use this file for Table tab:main_results.",
        "script": "none",
        "logs": ("results/FINAL_analysis_summary.json",),
        "role": "not_the_table",
        "note": (
            "REPRODUCIBILITY.md says the copy at results/FINAL_analysis_summary.json "
            "inside the supplementary zip does not match the paper."
        ),
    },
    {
        "claim_id": "pooled_stress_iqm",
        "manuscript": "Not Table tab:stress_results. Do not use this file for the stress means.",
        "script": "none",
        "logs": ("results/stress_analysis/stress_iqm.json",),
        "role": "not_the_table",
        "note": (
            "REPRODUCIBILITY.md says this archive pools CapacityGate seeds the paper does not report."
        ),
    },
    {
        "claim_id": "ablation_grid",
        "manuscript": "Not in the manuscript tables. The 125-cell grid is not_yet_run.",
        "script": "experiments/run_ablations.py",
        "logs": ("results/ablations",),
        "role": "manuscript",
        "note": "No full ablation log is a result in the paper.",
    },
    {
        "claim_id": "short_pilots",
        "manuscript": "Not a manuscript claim. The 2,000-step hopper pilots are launch checks.",
        "script": "experiments/run_ablations.py",
        "logs": ("results/ablation_pilot", "results/probe_pilot"),
        "role": "not_manuscript",
        "note": "Pilot scores are not comparable to the 25,000-step tables and are not read here.",
    },
)


def _log_present(root: Path, relative: str) -> bool:
    if any(char in relative for char in "*?[]"):
        return any(root.glob(relative))
    path = root / relative
    if path.is_file():
        return True
    if path.is_dir():
        return any(path.glob("**/summary.json"))
    return False


def assess_claims(root: str | Path = ".") -> list[dict[str, Any]]:
    """Return one row per claim. Never reads a log's contents."""
    base = Path(root)
    rows: list[dict[str, Any]] = []
    for claim in CLAIMS:
        present = any(_log_present(base, relative) for relative in claim["logs"])
        if claim["role"] == "not_manuscript":
            status = "not_a_manuscript_result"
        elif claim["role"] == "not_the_table":
            status = "not_the_manuscript_table"
        elif present:
            status = "present_not_recomputed"
        else:
            status = "unverified"
        rows.append(
            {
                "claim_id": claim["claim_id"],
                "manuscript": claim["manuscript"],
                "script": claim["script"],
                "logs": list(claim["logs"]),
                "status": status,
                "logs_present": present,
                "note": claim["note"],
            }
        )
    return rows


def render_markdown(root: str | Path = ".") -> str:
    """Reviewer checklist. Numbers in the text are transcribed tokens, not recomputed scores."""
    rows = assess_claims(root)
    lines = [
        "# Claim to log checklist",
        "",
        "This page maps each manuscript claim to the script and the working-tree log that would carry it.",
        "It does not recompute a score and it does not unpack `DoNotDisturb_ICLR2027_Supplementary.zip`.",
        "A log that exists only inside that zip is `unverified` until it is on disk, and a file that is on disk is still not read.",
        "",
        "`unverified` means the working-tree log is absent.",
        "`present_not_recomputed` means the path exists and its contents were not opened.",
        "`not_the_manuscript_table` means the path is the wrong archive even if it is later unpacked.",
        "`not_a_manuscript_result` means the run is a launch check, not a paper table.",
        "",
        "Figures named in `docs/pending_revision_numbers.md` are not rows here and are not applied.",
        "`experiments/check_public_claims.py` only checks that transcribed tokens are still on the public pages.",
        "",
        "| Claim | Manuscript location | Script | Working-tree log | Status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        logs = "<br>".join(f"`{path}`" for path in row["logs"])
        lines.append(
            "| `{claim_id}` | {manuscript} | `{script}` | {logs} | `{status}` |".format(
                claim_id=row["claim_id"],
                manuscript=row["manuscript"].replace("|", "/"),
                script=row["script"],
                logs=logs,
                status=row["status"],
            )
        )
    lines.extend(["", "## Notes", ""])
    for row in rows:
        lines.append(f"- `{row['claim_id']}` ({row['status']}): {row['note']}")
    lines.append("")
    return "\n".join(lines)
