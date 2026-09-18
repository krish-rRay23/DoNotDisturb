"""Mocked execution tests for controlled parallel harness (--workers).

No training, no environment, no GPU. The runner is always a lightweight
mock; the 150-cell plans are pure enumeration (no env interaction).
Proves: worker cap, exact-once scheduling of all 150 cells, stop-on-failure,
and completed-cell skip/resume.
"""

import json
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import final_factorial as ff
from adaptive_plasticity.final_runner import CONFIRM_TOKEN


def _full_study_plan():
    return (ff.build_primary_plan(device="cpu")
            + ff.build_reference_plan(device="cpu"))


def _key(cell):
    return (cell["dataset_id"], cell["controller"], cell["shift"], cell["seed"])


def test_default_workers_is_nine():
    assert ff.DEFAULT_WORKERS == 9


def test_main_forwards_default_workers_nine(tmp_path):
    with mock.patch.object(ff, "execute_plan",
                           return_value=[]) as exec_mock:
        ff.main(["--execute", "--confirm-matrix", CONFIRM_TOKEN,
                 "--seeds", "0", "--controllers", "none",
                 "--outdir", str(tmp_path)])
    assert exec_mock.call_count == 2  # primary + reference
    for call in exec_mock.call_args_list:
        assert call.kwargs["workers"] == 9


def test_main_rejects_workers_zero(tmp_path):
    with mock.patch.object(ff, "run_final_cell") as runner:
        with pytest.raises(SystemExit) as exc:
            ff.main(["--execute", "--confirm-matrix", CONFIRM_TOKEN,
                     "--workers", "0", "--outdir", str(tmp_path)])
    assert exc.value.code == 2
    runner.assert_not_called()


def test_execute_plan_rejects_workers_zero(tmp_path):
    with pytest.raises(ValueError):
        ff.execute_plan([], tmp_path, workers=0)


def test_workers_cap_concurrency_and_schedule_all_150_once(tmp_path):
    plan = _full_study_plan()
    assert len(plan) == 150
    lock = threading.Lock()
    state = {"active": 0, "max_active": 0, "calls": Counter()}

    def runner(**kw):
        key = (kw["dataset_id"], kw["controller"], kw["shift"], kw["seed"])
        with lock:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
            state["calls"][key] += 1
        try:
            time.sleep(0.003)  # force overlap between workers
        finally:
            with lock:
                state["active"] -= 1
        return {"dataset_id": kw["dataset_id"], "controller": kw["controller"],
                "shift": kw["shift"], "seed": kw["seed"],
                "best_normalized": 1.0,
                "output_name": ff.final_output_name(
                    dataset_id=kw["dataset_id"], controller=kw["controller"],
                    shift=kw["shift"], severity=kw["severity"], seed=kw["seed"])}

    summaries = ff.execute_plan(plan, tmp_path, runner=runner,
                                manifest_prefix="FINAL", workers=9)
    # Cap respected, parallelism actually happened.
    assert state["max_active"] <= 9
    assert state["max_active"] >= 2
    # Every one of the 150 cells ran exactly once ...
    assert sum(state["calls"].values()) == 150
    assert all(v == 1 for v in state["calls"].values())
    assert {_key(c) for c in plan} == set(state["calls"])
    # ... and summaries/manifest follow plan order.
    assert [s["output_name"] for s in summaries] == [
        c["output_name"] for c in plan]
    manifest_lines = (tmp_path / "FINAL_manifest.jsonl").read_text().splitlines()
    assert len(manifest_lines) == 150
    assert json.loads(manifest_lines[0])["output_name"] == plan[0]["output_name"]


def test_failure_stops_further_scheduling_sequential(tmp_path):
    plan = ff.build_plan(seeds=(0, 1), controllers=("none", "fixed"),
                         device="cpu")  # 3 envs x 3 shifts x 2 ctrls x 2 seeds
    fail_name = plan[2]["output_name"]
    seen = []

    def runner(**kw):
        name = ff.final_output_name(
            dataset_id=kw["dataset_id"], controller=kw["controller"],
            shift=kw["shift"], severity=kw["severity"], seed=kw["seed"])
        seen.append(name)
        if name == fail_name:
            raise RuntimeError("boom")
        return {"dataset_id": kw["dataset_id"], "controller": kw["controller"],
                "shift": kw["shift"], "seed": kw["seed"],
                "best_normalized": 1.0, "output_name": name}

    with pytest.raises(RuntimeError, match="boom"):
        ff.execute_plan(plan, tmp_path, runner=runner, workers=1)
    # Cells 0..2 ran; nothing after the failure was ever scheduled.
    assert seen == [c["output_name"] for c in plan[:3]]
    assert not (tmp_path / "FINAL_manifest.jsonl").exists()


def test_failure_stops_further_scheduling_parallel(tmp_path):
    plan = ff.build_plan(seeds=(0,), controllers=("none", "fixed"),
                         device="cpu")  # 18 cells
    fail_name = plan[0]["output_name"]
    seen = []
    lock = threading.Lock()

    def runner(**kw):
        name = ff.final_output_name(
            dataset_id=kw["dataset_id"], controller=kw["controller"],
            shift=kw["shift"], severity=kw["severity"], seed=kw["seed"])
        with lock:
            seen.append(name)
        if name == fail_name:
            raise RuntimeError("fast-boom")
        time.sleep(0.2)  # in-flight cells finish; nothing new is scheduled
        return {"dataset_id": kw["dataset_id"], "controller": kw["controller"],
                "shift": kw["shift"], "seed": kw["seed"],
                "best_normalized": 1.0, "output_name": name}

    with pytest.raises(RuntimeError, match="fast-boom"):
        ff.execute_plan(plan, tmp_path, runner=runner, workers=3)
    # Only the initial in-flight window (<= 3 cells) ever started.
    assert len(seen) <= 3
    assert fail_name in seen
    for cell in plan[3:]:
        assert cell["output_name"] not in seen
    assert not (tmp_path / "FINAL_manifest.jsonl").exists()


def test_completed_cells_are_skipped_and_resumed(tmp_path):
    plan = ff.build_plan(seeds=(0,), controllers=("none",), device="cpu")
    assert len(plan) == 9
    skipped_idx = {1, 4, 7}

    def saved_summary(cell):
        return {"dataset_id": cell["dataset_id"],
                "controller": cell["controller"], "shift": cell["shift"],
                "seed": cell["seed"], "best_normalized": 99.0,
                "output_name": cell["output_name"]}

    for i, cell in enumerate(plan):
        if i in skipped_idx:
            d = tmp_path / cell["output_name"]
            d.mkdir(parents=True, exist_ok=True)
            (d / "summary.json").write_text(
                json.dumps(saved_summary(cell)), encoding="utf-8")
            (d / "checkpoint.pt").write_bytes(b"fake-checkpoint")
    # Corrupt summary + present checkpoint => treated as incomplete (re-run).
    corrupt = plan[2]
    d = tmp_path / corrupt["output_name"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text("{not valid json", encoding="utf-8")
    (d / "checkpoint.pt").write_bytes(b"fake-checkpoint")

    ran = []

    def runner(**kw):
        name = ff.final_output_name(
            dataset_id=kw["dataset_id"], controller=kw["controller"],
            shift=kw["shift"], severity=kw["severity"], seed=kw["seed"])
        ran.append(name)
        return {"dataset_id": kw["dataset_id"], "controller": kw["controller"],
                "shift": kw["shift"], "seed": kw["seed"],
                "best_normalized": 1.0, "output_name": name}

    summaries = ff.execute_plan(plan, tmp_path, runner=runner, workers=3)
    # Skipped cells never invoked the runner; everything else ran once.
    for i in skipped_idx:
        assert plan[i]["output_name"] not in ran
    assert len(ran) == len(plan) - len(skipped_idx)
    assert corrupt["output_name"] in ran
    # Saved summaries reused in place (plan order preserved).
    for i in skipped_idx:
        assert summaries[i] == saved_summary(plan[i])
    assert [s["output_name"] for s in summaries] == [
        c["output_name"] for c in plan]
    assert len((tmp_path / "FINAL_manifest.jsonl").read_text().splitlines()) == 9


def test_sequential_workers_one_matches_workers_n_order(tmp_path):
    plan = ff.build_plan(seeds=(0,), controllers=("fixed",), device="cpu")

    def runner(**kw):
        return {"dataset_id": kw["dataset_id"], "controller": kw["controller"],
                "shift": kw["shift"], "seed": kw["seed"],
                "best_normalized": 2.0,
                "output_name": ff.final_output_name(
                    dataset_id=kw["dataset_id"], controller=kw["controller"],
                    shift=kw["shift"], severity=kw["severity"], seed=kw["seed"])}

    s_seq = ff.execute_plan(plan, tmp_path / "a", runner=runner, workers=1)
    s_par = ff.execute_plan(plan, tmp_path / "b", runner=runner, workers=4)
    assert [s["output_name"] for s in s_seq] == [
        s["output_name"] for s in s_par] == [c["output_name"] for c in plan]
