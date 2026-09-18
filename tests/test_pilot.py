"""Pre-pilot mode tests: 9-cell spec, isolation from the frozen FINAL study.

No training, no environment. The pilot reuses the frozen runner read-only
with reduced budgets; these tests prove the FINAL defaults are untouched.
"""

import json
import sys
from collections import Counter
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import final_factorial as ff
from adaptive_plasticity.final_runner import (
    CONFIRM_TOKEN,
    FINAL_CONTROLLERS,
    FINAL_ENVS,
    FINAL_SEEDS,
    FINAL_SHIFTS,
    REGIME_SEVERITY,
    final_protocol,
)
from final_factorial import (
    build_plan,
    build_pilot_plan,
    pilot_output_name,
    pilot_protocol,
)


def test_pilot_plan_has_9_cells():
    plan = build_pilot_plan()
    assert len(plan) == 9  # 1 env x 3 regimes x 3 arms x 1 seed


def test_pilot_dimensions():
    plan = build_pilot_plan()
    assert Counter(c["dataset_id"] for c in plan) == {"hopper-medium-v2": 9}
    assert Counter(c["shift"] for c in plan) == {
        "none": 3, "obs_noise": 3, "reward_scale": 3}
    assert Counter(c["controller"] for c in plan) == {
        "none": 3, "fixed": 3, "capacity_gate": 3}
    assert Counter(c["seed"] for c in plan) == {0: 9}
    assert "m10ref" not in {c["controller"] for c in plan}
    names = [c["output_name"] for c in plan]
    assert len(set(names)) == 9
    assert all(n.startswith("PILOT-") for n in names)


def test_pilot_severity_and_shift_step():
    for cell in build_pilot_plan():
        assert cell["severity"] == REGIME_SEVERITY[cell["shift"]]
        assert cell["severity"] in (0.0, 0.1, 0.5)
        assert cell["shift_step"] == (0 if cell["shift"] == "none" else 1_000)
        assert f"sev{cell['severity']:g}" in cell["output_name"]


def test_pilot_budgets():
    for cell in build_pilot_plan():
        assert cell["warmup_updates"] == 5_000
        assert cell["online_steps"] == 5_000
        assert cell["update_freq"] == 1
        assert cell["online_ratio"] == 0.5
        assert cell["eval_interval"] == 1_000
        assert cell["eval_episodes"] == 5
        assert cell["gate_interval"] == 1_000
        p = cell["protocol"]
        assert p["protocol_version"] == "pilot-v1"
        assert p["offline_warmup_updates"] == 5_000
        assert p["online_steps"] == 5_000
        assert p["eval_interval"] == 1_000
        assert p["eval_episodes"] == 5


def test_pilot_reuses_frozen_thresholds_and_severities():
    pp = pilot_protocol()
    fp = final_protocol()
    assert pp["gate_thresholds"] == fp["gate_thresholds"]
    assert pp["regime_severity"] == fp["regime_severity"]
    assert pp["based_on"] == fp["protocol_version"]
    assert pp["protocol_version"] != fp["protocol_version"]


def test_pilot_output_prefix_distinct_from_final():
    from adaptive_plasticity.final_runner import final_output_name
    pilot = pilot_output_name(
        dataset_id="hopper-medium-v2", controller="fixed",
        shift="obs_noise", severity=0.1, seed=0)
    final = final_output_name(
        dataset_id="hopper-medium-v2", controller="fixed",
        shift="obs_noise", severity=0.1, seed=0)
    assert pilot.startswith("PILOT-")
    assert final.startswith("FINAL-")
    assert pilot != final


def test_final_defaults_match_approved_protocol():
    assert final_protocol()["offline_warmup_updates"] == 25_000
    assert final_protocol()["online_steps"] == 25_000
    assert final_protocol()["shift_step"] == 5_000
    assert final_protocol()["eval_interval"] == 2_500
    assert final_protocol()["eval_episodes"] == 10
    assert final_protocol()["protocol_version"] == "final-v1"
    assert len(build_plan()) == 135  # primary matrix (m10ref excluded)
    assert len(ff.build_primary_plan()) == 135
    assert len(ff.build_reference_plan()) == 15
    assert CONFIRM_TOKEN == "CONFIRM-FINAL-150"
    assert ff.PILOT_CONFIRM_TOKEN == "CONFIRM-PILOT-9"
    assert ff.PILOT_CONFIRM_TOKEN != CONFIRM_TOKEN


def _fake_summary(**kw):
    return {"dataset_id": kw["dataset_id"], "controller": kw["controller"],
            "shift": kw["shift"], "seed": kw["seed"],
            "best_normalized": 50.0, "output_name": "X",
            "applied_intervention_steps": 0}


def test_pilot_dry_run_writes_pilot_plan_only(tmp_path):
    with mock.patch.object(ff, "run_final_cell") as runner:
        ff.main(["--pilot", "--outdir", str(tmp_path)])
    runner.assert_not_called()
    assert (tmp_path / "PILOT_plan.json").exists()
    assert not (tmp_path / "FINAL_plan.json").exists()
    plan_doc = json.loads((tmp_path / "PILOT_plan.json").read_text())
    assert plan_doc["n_cells"] == 9
    assert plan_doc["protocol"]["protocol_version"] == "pilot-v1"


def test_pilot_execute_without_confirmation_runs_nothing(tmp_path):
    with mock.patch.object(ff, "run_final_cell") as runner:
        import pytest
        with pytest.raises(SystemExit) as exc:
            ff.main(["--pilot", "--execute", "--outdir", str(tmp_path)])
    assert exc.value.code == 2
    runner.assert_not_called()


def test_pilot_execute_rejects_final_token(tmp_path):
    import pytest
    with mock.patch.object(ff, "run_final_cell") as runner:
        with pytest.raises(SystemExit) as exc:
            ff.main(["--pilot", "--execute", "--confirm-matrix", CONFIRM_TOKEN,
                      "--outdir", str(tmp_path)])
    assert exc.value.code == 2
    runner.assert_not_called()


def test_pilot_execute_with_pilot_token_runs_9(tmp_path):
    with mock.patch.object(ff, "run_final_cell", side_effect=_fake_summary) as runner:
        ff.main(["--pilot", "--execute", "--confirm-matrix", ff.PILOT_CONFIRM_TOKEN,
                 "--device", "cpu", "--outdir", str(tmp_path)])
    assert runner.call_count == 9
    assert (tmp_path / "PILOT_manifest.jsonl").exists()
    assert (tmp_path / "PILOT_rliable_scores.csv").exists()
    assert not (tmp_path / "FINAL_manifest.jsonl").exists()
    for call in runner.call_args_list:
        assert call.kwargs["device"] == "cpu"
        assert call.kwargs["online_steps"] == 5_000
        assert call.kwargs["warmup_updates"] == 5_000
        assert call.kwargs["eval_interval"] == 1_000
        assert call.kwargs["eval_episodes"] == 5
        assert call.kwargs["update_freq"] == 1
        assert call.kwargs["online_ratio"] == 0.5


def test_final_execute_rejects_pilot_token(tmp_path):
    import pytest
    with mock.patch.object(ff, "run_final_cell") as runner:
        with pytest.raises(SystemExit) as exc:
            ff.main(["--execute", "--confirm-matrix", ff.PILOT_CONFIRM_TOKEN,
                      "--outdir", str(tmp_path)])
    assert exc.value.code == 2
    runner.assert_not_called()
