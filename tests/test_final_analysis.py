"""Unit tests for the FINAL-v1 post-run analysis pipeline.

Synthetic fixtures only (tmp_path): no training, no real results, no GPU.
Covers audit classification, strict incomplete-data refusal, last-score
extraction, aggregation, IQM/bootstrap/P(improvement) math, gate activation
stats, primary/reference separation, and plot/table artifact creation.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import final_analysis as fa


PLAN_PROTOCOL = {
    "protocol_version": "final-v1",
    "offline_warmup_updates": 25_000,
    "online_steps": 25_000,
    "shift_step": 5_000,
    "eval_interval": 2_500,
    "eval_episodes": 10,
    "online_ratio": 0.5,
    "update_freq": 1,
    "gate_interval": 1_000,
}

GATE_THRESH = {"dorm_on": 0.15, "dorm_off": 0.05, "rho_on": 0.70, "rho_off": 0.85}


def _cell(env, ctrl, shift, seed, severity=0.1, shift_step=5_000):
    name = f"FINAL-{env}-{ctrl}-{shift}-sev{severity:g}-seed{seed}"
    return {"dataset_id": env, "controller": ctrl, "shift": shift,
            "shift_step": 0 if shift == "none" else shift_step,
            "severity": severity, "seed": seed, "device": "cpu",
            "output_name": name, "protocol": dict(PLAN_PROTOCOL)}


def _write_run(results_dir, cell, normalized=(10.0, 20.0, 30.0),
               gate_on_pattern=(), protocol_override=None):
    d = Path(results_dir) / cell["output_name"]
    d.mkdir(parents=True, exist_ok=True)
    proto = dict(PLAN_PROTOCOL)
    proto.update(protocol_override or {})
    summary = {
        "output_name": cell["output_name"], "dataset_id": cell["dataset_id"],
        "controller": cell["controller"], "shift": cell["shift"],
        "seed": cell["seed"], "severity": cell["severity"],
        "normalized": list(normalized),
        "returns": [v * 10.0 for v in normalized],
        "best_normalized": max(normalized) if len(normalized) else 0.0,
        "applied_intervention_steps": 3,
        "online_steps": 25_000, "update_steps": 25_000,
        "protocol": proto,
        "gate": {"applicable": cell["controller"] == "capacity_gate",
                 "intervention_frequency": 0.2,
                 "n_evaluations": len(gate_on_pattern),
                 "rank_anchor": 80.0,
                 "config": dict(GATE_THRESH, active_severity=0.1),
                 "kind": "capacity-gate-v1", "schema_version": 2},
        "m10ref": {"applicable": cell["controller"] == "m10ref"},
    }
    (d / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (d / "checkpoint.pt").write_bytes(b"fake")
    if cell["controller"] == "capacity_gate":
        lines = []
        for i, on in enumerate(gate_on_pattern or [False] * 3):
            lines.append({"step": (i + 1) * 1000, "gate_on": bool(on),
                          "gate_prev": False, "rho": 0.95, "dormancy": 0.02,
                          "effective_rank": 76.0, "numerical_rank": 200.0,
                          "rank_anchor": 80.0,
                          "intervention_strength": 0.1 if on else 0.0,
                          "intervention_count": i if on else 0,
                          "intervention_frequency": 0.2,
                          "shift_active": True})
        (d / f"gate_{cell['output_name']}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in lines) + "\n", encoding="utf-8")
    if cell["controller"] == "m10ref":
        (d / f"m10ref_{cell['output_name']}.jsonl").write_text(
            json.dumps({"step": 1000, "intervention_strength": 0.3}) + "\n",
            encoding="utf-8")
    return cell


def _write_plans(results_dir, primary, reference):
    (Path(results_dir) / "FINAL_plan.json").write_text(json.dumps(
        {"cells": primary, "n_cells": len(primary),
         "protocol": PLAN_PROTOCOL}) + "\n", encoding="utf-8")
    (Path(results_dir) / "FINAL_reference_plan.json").write_text(json.dumps(
        {"cells": reference, "n_cells": len(reference),
         "protocol": PLAN_PROTOCOL}) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def test_audit_complete_run(tmp_path):
    cell = _cell("hopper-medium-v2", "none", "none", 0, 0.0)
    _write_run(tmp_path, cell)
    out = fa.audit_cell(tmp_path, cell, PLAN_PROTOCOL)
    assert out["status"] == "complete"
    assert out["reason"] is None


def test_audit_missing_run(tmp_path):
    cell = _cell("hopper-medium-v2", "none", "none", 0, 0.0)
    out = fa.audit_cell(tmp_path, cell, PLAN_PROTOCOL)
    assert (out["status"], out["reason"]) == ("missing", "no-summary")


def test_audit_unparseable_summary(tmp_path):
    cell = _cell("hopper-medium-v2", "none", "none", 0, 0.0)
    d = tmp_path / cell["output_name"]
    d.mkdir(parents=True)
    (d / "summary.json").write_text("{broken", encoding="utf-8")
    assert fa.audit_cell(tmp_path, cell, PLAN_PROTOCOL)["reason"] == "unparseable-summary"


def test_audit_protocol_mismatch(tmp_path):
    cell = _cell("hopper-medium-v2", "none", "none", 0, 0.0)
    _write_run(tmp_path, cell, protocol_override={"online_steps": 50_000})
    out = fa.audit_cell(tmp_path, cell, PLAN_PROTOCOL)
    assert out["reason"] == "protocol-mismatch"
    assert "online_steps" in out["detail"]


def test_audit_empty_trajectory(tmp_path):
    cell = _cell("hopper-medium-v2", "none", "none", 0, 0.0)
    _write_run(tmp_path, cell, normalized=())
    assert fa.audit_cell(tmp_path, cell, PLAN_PROTOCOL)["reason"] == "empty-trajectory"


def test_audit_missing_gate_log(tmp_path):
    cell = _cell("hopper-medium-v2", "capacity_gate", "obs_noise", 0)
    _write_run(tmp_path, cell)
    (tmp_path / cell["output_name"] / f"gate_{cell['output_name']}.jsonl").unlink()
    assert fa.audit_cell(tmp_path, cell, PLAN_PROTOCOL)["reason"] == "missing-gate-log"


def test_audit_missing_checkpoint(tmp_path):
    cell = _cell("hopper-medium-v2", "none", "none", 0, 0.0)
    _write_run(tmp_path, cell)
    (tmp_path / cell["output_name"] / "checkpoint.pt").unlink()
    assert fa.audit_cell(tmp_path, cell, PLAN_PROTOCOL)["reason"] == "missing-checkpoint"


def test_audit_identity_mismatch(tmp_path):
    cell = _cell("hopper-medium-v2", "none", "none", 0, 0.0)
    _write_run(tmp_path, cell)
    # Same output_name on disk (seed 0 in summary) but plan cell claims seed 99.
    other = dict(cell, seed=99)
    out = fa.audit_cell(tmp_path, other, PLAN_PROTOCOL)
    assert out["reason"] == "identity-mismatch"


def test_audit_study_counts_and_order(tmp_path):
    primary = [_cell("hopper-medium-v2", "none", "none", 0, 0.0),
               _cell("hopper-medium-v2", "fixed", "obs_noise", 1)]
    reference = [_cell("halfcheetah-medium-v2", "m10ref", "none", 0, 0.0)]
    _write_run(tmp_path, primary[0])
    audit = fa.audit_study(tmp_path, primary, reference, PLAN_PROTOCOL)
    assert (audit["n_primary"], audit["n_reference"]) == (2, 1)
    assert audit["n_complete"] == 1
    assert audit["complete"] is False
    assert [a["output_name"] for a in audit["cells"]["primary"]] == [
        c["output_name"] for c in primary]


# ---------------------------------------------------------------------------
# Strict refusal + audit-only
# ---------------------------------------------------------------------------

def test_main_refuses_incomplete_study(tmp_path):
    primary = [_cell("hopper-medium-v2", "none", "none", 0, 0.0)]
    _write_plans(tmp_path, primary, [])
    with pytest.raises(SystemExit) as exc:
        fa.main(["--results-dir", str(tmp_path)])
    assert exc.value.code == 2
    assert not (tmp_path / "FINAL_analysis_summary.json").exists()
    assert (tmp_path / "FINAL_analysis_audit.json").exists()


def test_main_audit_only_writes_audit(tmp_path):
    primary = [_cell("hopper-medium-v2", "none", "none", 0, 0.0)]
    _write_plans(tmp_path, primary, [])
    fa.main(["--results-dir", str(tmp_path), "--audit-only"])
    audit = json.loads((tmp_path / "FINAL_analysis_audit.json").read_text())
    assert audit["n_complete"] == 0
    assert not (tmp_path / "FINAL_analysis_summary.json").exists()


# ---------------------------------------------------------------------------
# Extraction / aggregation / stats math
# ---------------------------------------------------------------------------

def test_extract_uses_last_normalized_score(tmp_path):
    cell = _cell("hopper-medium-v2", "fixed", "obs_noise", 2)
    _write_run(tmp_path, cell, normalized=(5.0, 50.0, 15.0))
    rec = fa.extract_run(tmp_path, cell)
    assert rec["score"] == pytest.approx(15.0)  # last, not best (50.0)
    assert rec["eval_steps"] == [2500, 5000, 7500]
    assert (rec["env"], rec["regime"], rec["method"], rec["seed"]) == \
        ("hopper-medium-v2", "obs_noise", "fixed", 2)


def test_aggregate_group_means(tmp_path):
    cells = [_cell("hopper-medium-v2", "none", "none", s, 0.0) for s in (0, 1)]
    for cell, score in zip(cells, (10.0, 20.0)):
        _write_run(tmp_path, cell, normalized=(score,))
    recs = [fa.extract_run(tmp_path, c) for c in cells]
    rows = fa.aggregate(recs)
    assert len(rows) == 1
    assert rows[0]["mean"] == pytest.approx(15.0)
    assert rows[0]["median"] == pytest.approx(15.0)
    assert rows[0]["seeds"] == [0, 1]


def test_iqm_known_value():
    assert fa.iqm([1, 2, 3, 4, 5, 6, 7, 8]) == pytest.approx(4.5)
    assert fa.iqm([7.0]) == pytest.approx(7.0)
    assert np.isnan(fa.iqm([]))


def test_bootstrap_ci_deterministic_and_sane():
    groups = [np.array([10.0, 12.0, 11.0, 13.0, 12.0]),
              np.array([20.0, 22.0, 21.0, 23.0, 22.0])]
    r1 = fa.iqm_with_ci(groups, replicates=100, seed=7)
    r2 = fa.iqm_with_ci(groups, replicates=100, seed=7)
    assert r1 == r2
    assert r1["ci_low"] <= r1["iqm"] <= r1["ci_high"]
    assert r1["n_scores"] == 10


def test_prob_improvement_self_and_dominance():
    groups = [np.array([1.0, 2.0, 3.0, 4.0])]
    same = fa.prob_improvement(groups, groups, replicates=50, seed=0)
    assert same["prob"] == pytest.approx(0.5)
    better = [np.array([10.0, 11.0, 12.0])]
    worse = [np.array([1.0, 2.0, 3.0])]
    dom = fa.prob_improvement(better, worse, replicates=50, seed=0)
    assert dom["prob"] == pytest.approx(1.0)
    assert dom["ci_low"] == pytest.approx(1.0)


def test_gate_activation_duration_and_onset():
    recs = [{"step": 1000, "gate_on": False, "intervention_frequency": 0.0},
            {"step": 2000, "gate_on": True, "intervention_frequency": 0.25},
            {"step": 3000, "gate_on": True, "intervention_frequency": 0.5},
            {"step": 4000, "gate_on": False, "intervention_frequency": 0.4},
            {"step": 5000, "gate_on": True, "intervention_frequency": 0.5}]
    stats = fa.gate_activation(recs, gate_interval=1000)
    assert stats["fraction_on"] == pytest.approx(0.6)
    assert stats["max_on_records"] == 2
    assert stats["max_on_steps"] == 2000
    assert stats["onset_step"] == 2000
    assert stats["n_activations"] == 2


def test_gate_activation_never_on():
    recs = [{"step": 1000, "gate_on": False, "intervention_frequency": 0.0}]
    stats = fa.gate_activation(recs, gate_interval=1000)
    assert stats["max_on_steps"] == 0
    assert stats["onset_step"] is None


def test_clean_vs_shifted_math():
    rows = [
        {"env": "hopper-medium-v2", "regime": "none", "method": "fixed",
         "mean": 100.0},
        {"env": "hopper-medium-v2", "regime": "obs_noise", "method": "fixed",
         "mean": 80.0},
    ]
    out = fa.clean_vs_shifted(rows)
    assert len(out) == 1
    assert out[0]["abs_change"] == pytest.approx(-20.0)
    assert out[0]["rel_change"] == pytest.approx(-0.2)


# ---------------------------------------------------------------------------
# End-to-end on a miniature synthetic study
# ---------------------------------------------------------------------------

def _mini_study(tmp_path):
    envs = ["hopper-medium-v2", "walker2d-medium-v2"]
    primary, reference = [], []
    for env in envs:
        for shift, sev in (("none", 0.0), ("obs_noise", 0.1)):
            for ctrl in ("none", "fixed", "capacity_gate"):
                for seed in (0, 1, 2):
                    c = _cell(env, ctrl, shift, seed, sev)
                    primary.append(c)
                    base = 100.0 - (10.0 if shift != "none" else 0.0)
                    _write_run(tmp_path, c,
                               normalized=(base - 5, base, base + 5),
                               gate_on_pattern=(False, True, False))
    for shift, sev in (("none", 0.0), ("obs_noise", 0.1)):
        for seed in (0, 1, 2):
            c = _cell("halfcheetah-medium-v2", "m10ref", shift, seed, sev)
            reference.append(c)
            _write_run(tmp_path, c, normalized=(40.0, 42.0, 44.0))
    return primary, reference


def test_end_to_end_mini_study(tmp_path):
    primary, reference = _mini_study(tmp_path)
    plots = tmp_path / "plots"
    payload = fa.run_analysis(
        results_dir=tmp_path, plots_dir=plots, replicates=50,
        bootstrap_seed=0, primary_cells=primary, reference_cells=reference,
        plan_protocol=dict(PLAN_PROTOCOL))
    assert payload["protocol_version"] == "final-v1"
    assert payload["score_definition"] == "last-normalized-eval"
    assert payload["partial_data"] is False
    assert len(payload["runs"]) == len(primary)
    assert len(payload["reference_runs"]) == len(reference)
    # Primary/reference separation: no m10ref in groups, no gate arms in ref.
    assert {r["method"] for r in payload["groups"]} == {
        "none", "fixed", "capacity_gate"}
    assert {r["method"] for r in payload["reference_groups"]} == {"m10ref"}
    assert set(payload["iqm_by_method"]) == {"none", "fixed", "capacity_gate"}
    assert payload["prob_improvement"]["capacity_gate"]["none"]["prob"] == \
        pytest.approx(0.5)  # identical synthetic scores across methods
    gate_runs = [r for r in payload["runs"] if r["method"] == "capacity_gate"]
    assert all("gate_activation" in r for r in gate_runs)
    assert len(payload["clean_vs_shifted"]) == 2 * 3  # envs x methods
    # Tables + figures written.
    fa.write_tables(payload["groups"], tmp_path / "FINAL_paper_tables.csv")
    fa.write_tables(payload["reference_groups"],
                    tmp_path / "FINAL_reference_tables.csv")
    import pandas as pd
    assert len(pd.read_csv(tmp_path / "FINAL_paper_tables.csv")) == len(primary) // 3
    for fig in payload["figures"]:
        assert Path(fig).is_file(), fig
    assert len(payload["figures"]) == 7


def test_run_analysis_refuses_partial_without_flag(tmp_path):
    primary, reference = _mini_study(tmp_path)
    (tmp_path / primary[0]["output_name"] / "summary.json").unlink()
    with pytest.raises(RuntimeError, match="incomplete"):
        fa.run_analysis(results_dir=tmp_path, plots_dir=tmp_path / "p",
                        replicates=10, primary_cells=primary,
                        reference_cells=reference,
                        plan_protocol=dict(PLAN_PROTOCOL))


def test_run_analysis_partial_flag_proceeds(tmp_path):
    primary, reference = _mini_study(tmp_path)
    (tmp_path / primary[0]["output_name"] / "summary.json").unlink()
    payload = fa.run_analysis(
        results_dir=tmp_path, plots_dir=tmp_path / "p", replicates=10,
        primary_cells=primary, reference_cells=reference,
        plan_protocol=dict(PLAN_PROTOCOL), partial=True)
    assert payload["partial_data"] is True
    assert len(payload["runs"]) == len(primary) - 1
