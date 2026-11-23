"""The smallest confirm-gated path runs a few CPU steps on a synthetic fixture."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import fixture_smoke  # noqa: E402

FIXTURE_BANNER = fixture_smoke.FIXTURE_BANNER
FIXTURE_CONFIRM_TOKEN = fixture_smoke.FIXTURE_CONFIRM_TOKEN
FIXTURE_ONLINE_STEPS = fixture_smoke.FIXTURE_ONLINE_STEPS
OUTPUT_NAME = fixture_smoke.OUTPUT_NAME
main = fixture_smoke.main


def test_fixture_flag_refuses_to_read_a_dataset_file(tmp_path):
    from adaptive_plasticity.final_runner import run_final_cell

    with pytest.raises(ValueError, match="does not read a D4RL file"):
        run_final_cell(
            dataset_id="fixture-hopper",
            controller="none",
            shift="none",
            seed=0,
            online_steps=1,
            warmup_updates=0,
            device="cpu",
            outdir=tmp_path,
            use_shared_offline_checkpoints=False,
            fixture=True,
        )


def test_dry_run_writes_the_plan_and_does_not_train(tmp_path):
    main(["--outdir", str(tmp_path)])
    plan = json.loads((tmp_path / "FIXTURE_plan.json").read_text(encoding="utf-8"))
    assert plan["confirm_token"] == FIXTURE_CONFIRM_TOKEN
    assert plan["data_source"] == "synthetic_fixture"
    assert plan["device"] == "cpu"
    assert plan["online_steps"] == FIXTURE_ONLINE_STEPS
    assert plan["executed"] is False
    assert plan["banner"] == FIXTURE_BANNER
    assert not (tmp_path / OUTPUT_NAME).exists()


def test_wrong_confirm_token_refuses_before_training(tmp_path):
    with pytest.raises(SystemExit) as raised:
        main(["--outdir", str(tmp_path), "--execute", "--confirm-matrix", "CONFIRM-FINAL-150"])
    assert raised.value.code == 2
    assert not (tmp_path / OUTPUT_NAME).exists()
    plan = json.loads((tmp_path / "FIXTURE_plan.json").read_text(encoding="utf-8"))
    assert plan["executed"] is False


def test_confirm_token_runs_a_few_cpu_steps_on_the_fixture(tmp_path):
    main(
        [
            "--outdir",
            str(tmp_path),
            "--execute",
            "--confirm-matrix",
            FIXTURE_CONFIRM_TOKEN,
        ]
    )
    cell = tmp_path / OUTPUT_NAME
    summary = json.loads((cell / "summary.json").read_text(encoding="utf-8"))
    assert summary["fixture"] is True
    assert summary["banner"] == FIXTURE_BANNER
    assert summary["data_source"] == "synthetic_fixture"
    assert summary["dataset_id"] == "fixture-hopper"
    assert summary["controller"] == "capacity_gate"
    assert summary["environment"]["device"] == "cpu"
    assert summary["online_steps"] == FIXTURE_ONLINE_STEPS
    assert summary["update_steps"] == FIXTURE_ONLINE_STEPS
    assert summary["best_normalized"] is None
    assert summary["offline_rows"] == 64
    assert (cell / "checkpoint.pt").is_file()
    gate_log = cell / f"gate_{OUTPUT_NAME}.jsonl"
    records = [json.loads(line) for line in gate_log.read_text(encoding="utf-8").splitlines() if line]
    assert records
    text = (cell / "summary.json").read_text(encoding="utf-8")
    for forbidden in ("38.78", "40.37", "0.9915", "1.6501"):
        assert forbidden not in text
