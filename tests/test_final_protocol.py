"""Protocol tests: FINAL 150-run study (135 primary + 15 M10-reference).

No training, no environment. Verifies the plan shapes, per-cell protocol
metadata, arm trigger isolation, and isolated seeded observation noise.
"""

import sys
from collections import Counter
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import final_factorial as ff
from adaptive_plasticity.final_runner import (
    CONFIRM_TOKEN,
    REGIME_SEVERITY,
    apply_regime_transform,
    final_protocol,
    sample_mixed_batch_fast,
    should_apply_intervention,
)
from adaptive_plasticity.m5 import OnlineBuffer, sample_mixed_batch
from final_factorial import build_plan, build_primary_plan, build_reference_plan


def test_primary_plan_has_135_cells():
    plan = build_primary_plan()
    assert len(plan) == 135  # 3 envs x 3 regimes x 3 arms x 5 seeds
    assert len(build_plan()) == 135  # default plan IS the primary matrix


def test_reference_plan_has_15_cells():
    plan = build_reference_plan()
    assert len(plan) == 15  # halfcheetah x 3 regimes x m10ref x 5 seeds


def test_total_study_is_150_runs():
    assert len(build_primary_plan()) + len(build_reference_plan()) == 150


def test_primary_plan_dimensions():
    plan = build_primary_plan()
    assert Counter(c["dataset_id"] for c in plan) == {
        "halfcheetah-medium-v2": 45, "hopper-medium-v2": 45, "walker2d-medium-v2": 45}
    assert Counter(c["shift"] for c in plan) == {
        "none": 45, "obs_noise": 45, "reward_scale": 45}
    assert Counter(c["controller"] for c in plan) == {
        "none": 45, "fixed": 45, "capacity_gate": 45}
    assert Counter(c["seed"] for c in plan) == {0: 27, 1: 27, 2: 27, 3: 27, 4: 27}
    assert "m10ref" not in {c["controller"] for c in plan}
    names = [c["output_name"] for c in plan]
    assert len(set(names)) == 135


def test_reference_plan_dimensions():
    plan = build_reference_plan()
    assert Counter(c["dataset_id"] for c in plan) == {"halfcheetah-medium-v2": 15}
    assert Counter(c["shift"] for c in plan) == {
        "none": 5, "obs_noise": 5, "reward_scale": 5}
    assert Counter(c["controller"] for c in plan) == {"m10ref": 15}
    assert Counter(c["seed"] for c in plan) == {0: 3, 1: 3, 2: 3, 3: 3, 4: 3}
    names = [c["output_name"] for c in plan]
    assert len(set(names)) == 15


def test_primary_and_reference_names_disjoint():
    primary = {c["output_name"] for c in build_primary_plan()}
    reference = {c["output_name"] for c in build_reference_plan()}
    assert not (primary & reference)


def test_plan_severity_and_shift_step():
    for cell in build_primary_plan() + build_reference_plan():
        assert cell["severity"] == REGIME_SEVERITY[cell["shift"]]
        assert cell["shift_step"] == (0 if cell["shift"] == "none" else 5_000)
        assert f"sev{cell['severity']:g}" in cell["output_name"]


def test_plan_cells_carry_complete_protocol():
    required = {"offline_warmup_updates", "online_steps", "shift_step",
                "online_ratio", "update_freq", "eval_interval", "eval_episodes",
                "gate_interval", "deterministic", "gate_thresholds",
                "protocol_version", "regime_severity"}
    for cell in build_primary_plan() + build_reference_plan():
        assert required <= set(cell["protocol"]), cell["output_name"]
        p = cell["protocol"]
        assert p["online_steps"] == 25_000 and p["update_freq"] == 1
        assert p["eval_interval"] == 2_500 and p["eval_episodes"] == 10
        assert p["offline_warmup_updates"] == 25_000


def test_regime_severity_map():
    assert REGIME_SEVERITY == {"none": 0.0, "obs_noise": 0.1, "reward_scale": 0.5}


def test_trigger_truth_table():
    # none arm never applies
    assert should_apply_intervention(controller="none", step=10_000, shift_step=5_000, gate_on=True) is False
    assert should_apply_intervention(controller="none", step=1_000, shift_step=5_000, gate_on=False) is False

    # fixed arm applies unconditionally during intervention phase (step >= shift_step)
    assert should_apply_intervention(controller="fixed", step=1_000, shift_step=5_000) is False
    assert should_apply_intervention(controller="fixed", step=5_000, shift_step=5_000) is True
    assert should_apply_intervention(controller="fixed", step=10_000, shift_step=5_000) is True

    # capacity_gate triggers autonomously on gate_on without needing an oracle shift detector
    assert should_apply_intervention(controller="capacity_gate", gate_on=False) is False
    assert should_apply_intervention(controller="capacity_gate", gate_on=True) is True

    # m10ref triggers based on M10 heuristic strength
    assert should_apply_intervention(controller="m10ref", m10_strength=0.0) is False
    assert should_apply_intervention(controller="m10ref", m10_strength=0.3) is True


def test_obs_noise_deterministic_and_isolated():
    obs = np.arange(8, dtype=np.float32)
    r1 = np.random.RandomState(0 + 7919)
    r2 = np.random.RandomState(0 + 7919)
    o1, _ = apply_regime_transform(obs, 1.0, shift="obs_noise", severity=0.1, noise_rng=r1)
    o2, _ = apply_regime_transform(obs, 1.0, shift="obs_noise", severity=0.1, noise_rng=r2)
    np.testing.assert_array_equal(o1, o2)
    assert not np.array_equal(o1, obs)  # noise actually modifies obs
    r3 = np.random.RandomState(1 + 7919)
    o3, _ = apply_regime_transform(obs, 1.0, shift="obs_noise", severity=0.1, noise_rng=r3)
    assert not np.array_equal(o1, o3)
    # reward_scale path is deterministic and preserves obs exactly.
    o4, w4 = apply_regime_transform(obs, 2.0, shift="reward_scale", severity=0.5, noise_rng=None)
    np.testing.assert_array_equal(o4, obs)
    assert w4 == pytest.approx(1.0)
    # none regime is a passthrough.
    o5, w5 = apply_regime_transform(obs, 2.0, shift="none", severity=0.0, noise_rng=None)
    np.testing.assert_array_equal(o5, obs)
    assert w5 == pytest.approx(2.0)


def test_confirm_token_set():
    assert CONFIRM_TOKEN == "CONFIRM-FINAL-150"


def test_final_protocol_values():
    p = final_protocol()
    assert p["offline_warmup_updates"] == 25_000
    assert p["online_steps"] == 25_000
    assert p["shift_step"] == 5_000
    assert p["online_ratio"] == 0.5
    assert p["update_freq"] == 1
    assert p["eval_interval"] == 2_500
    assert p["eval_episodes"] == 10


def test_final_protocol_relative_rank_thresholds():
    p = final_protocol()
    assert p["gate_thresholds"] == {
        "dorm_on": 0.15, "dorm_off": 0.05, "rho_on": 0.70, "rho_off": 0.85}


def _fake_summary(**kw):
    return {"dataset_id": kw["dataset_id"], "controller": kw["controller"],
            "shift": kw["shift"], "seed": kw["seed"],
            "best_normalized": 50.0, "output_name": "X"}


def test_execute_without_confirmation_runs_nothing(tmp_path):
    with mock.patch.object(ff, "run_final_cell") as runner:
        with pytest.raises(SystemExit) as exc:
            ff.main(["--execute", "--outdir", str(tmp_path)])
    assert exc.value.code == 2
    runner.assert_not_called()


def test_execute_wrong_token_runs_nothing(tmp_path):
    with mock.patch.object(ff, "run_final_cell") as runner:
        with pytest.raises(SystemExit) as exc:
            ff.main(["--execute", "--confirm-matrix", "WRONG",
                     "--outdir", str(tmp_path)])
    assert exc.value.code == 2
    runner.assert_not_called()


def test_execute_with_confirmation_invokes_runner(tmp_path):
    # Primary subset (3 envs x 3 shifts x 1 ctrl x 1 seed = 9) PLUS the full
    # 15-cell reference plan; runner mocked (no training).
    with mock.patch.object(ff, "run_final_cell", side_effect=_fake_summary) as runner:
        ff.main(["--execute", "--confirm-matrix", CONFIRM_TOKEN,
                 "--seeds", "0", "--controllers", "none",
                 "--outdir", str(tmp_path)])
    assert runner.call_count == 9 + 15
    assert (tmp_path / "FINAL_manifest.jsonl").exists()
    assert (tmp_path / "FINAL_rliable_scores.csv").exists()
    assert (tmp_path / "FINAL_reference_manifest.jsonl").exists()
    assert (tmp_path / "FINAL_reference_rliable_scores.csv").exists()
    assert (tmp_path / "FINAL_plan.json").exists()
    assert (tmp_path / "FINAL_reference_plan.json").exists()


def test_dry_run_writes_both_plans_and_invokes_nothing(tmp_path):
    import json
    with mock.patch.object(ff, "run_final_cell") as runner:
        ff.main(["--outdir", str(tmp_path)])
    runner.assert_not_called()
    primary = json.loads((tmp_path / "FINAL_plan.json").read_text())
    reference = json.loads((tmp_path / "FINAL_reference_plan.json").read_text())
    assert primary["n_cells"] == 135
    assert reference["n_cells"] == 15


def test_dry_run_invokes_nothing(tmp_path):
    with mock.patch.object(ff, "run_final_cell") as runner:
        ff.main(["--outdir", str(tmp_path)])
    runner.assert_not_called()


def test_controllers_rejects_m10ref(tmp_path):
    with mock.patch.object(ff, "run_final_cell") as runner:
        with pytest.raises(SystemExit) as exc:
            ff.main(["--controllers", "m10ref", "--outdir", str(tmp_path)])
    assert exc.value.code == 2
    runner.assert_not_called()


def test_plan_cells_contain_device_cuda():
    for cell in build_plan():
        assert cell["device"] == "cuda"


def test_execute_forwards_device_cuda(tmp_path):
    with mock.patch.object(ff, "run_final_cell", side_effect=_fake_summary) as runner:
        ff.main(["--execute", "--confirm-matrix", CONFIRM_TOKEN,
                 "--seeds", "0", "--controllers", "none",
                 "--outdir", str(tmp_path)])
    assert runner.call_count == 9 + 15
    for call in runner.call_args_list:
        assert call.kwargs["device"] == "cuda"


def test_execute_device_override_forwarded(tmp_path):
    with mock.patch.object(ff, "run_final_cell", side_effect=_fake_summary) as runner:
        ff.main(["--execute", "--confirm-matrix", CONFIRM_TOKEN,
                 "--seeds", "0", "--controllers", "none",
                 "--device", "cpu", "--outdir", str(tmp_path)])
    assert runner.call_count == 9 + 15
    for call in runner.call_args_list:
        assert call.kwargs["device"] == "cpu"


def _small_replay(obs_dim=4, act_dim=2, n_off=1000, n_on=500, seed=0):
    rng = np.random.RandomState(seed)
    offline = {
        "observations": rng.randn(n_off, obs_dim).astype(np.float32),
        "actions": rng.randn(n_off, act_dim).astype(np.float32),
        "rewards": rng.randn(n_off).astype(np.float32),
        "next_observations": rng.randn(n_off, obs_dim).astype(np.float32),
        "dones": (rng.rand(n_off) > 0.9).astype(np.float32),
    }
    buf = OnlineBuffer(obs_dim=obs_dim, act_dim=act_dim, max_size=10_000)
    brng = np.random.RandomState(seed + 1)
    for i in range(n_on):
        buf.add(offline["observations"][i % n_off], offline["actions"][i % n_off],
                float(i), offline["next_observations"][i % n_off], 0.0)
    assert len(buf) == n_on
    return offline, buf


def _call_fast(offline, buf, ratio, seed, batch_size=256):
    return sample_mixed_batch_fast(
        online_buf=buf, offline_data=offline, offline_count=len(offline["rewards"]),
        batch_size=batch_size, online_ratio=ratio,
        rng=np.random.RandomState(seed), device="cpu")


def test_fast_sampler_deterministic():
    offline, buf = _small_replay()
    for ratio in (0.0, 0.5, 1.0):
        m1, n1o, n1f = _call_fast(offline, buf, ratio, seed=7)
        m2, n2o, n2f = _call_fast(offline, buf, ratio, seed=7)
        assert (n1o, n1f) == (n2o, n2f)
        for k in m1:
            assert torch.equal(m1[k], m2[k])
    # Different seed draws a different batch.
    m3, _, _ = _call_fast(offline, buf, 0.5, seed=8)
    m1, _, _ = _call_fast(offline, buf, 0.5, seed=7)
    assert not torch.equal(m1["observations"], m3["observations"])


def test_fast_sampler_exact_batch_sizes():
    import torch
    offline, buf = _small_replay()
    for ratio, exp_on in ((0.0, 0), (0.5, 128), (1.0, 256)):
        mixed, n_on, n_off = _call_fast(offline, buf, ratio, seed=3)
        assert (n_on, n_off) == (exp_on, 256 - exp_on)
        assert set(mixed) == set(offline)
        for k, v in mixed.items():
            assert isinstance(v, torch.Tensor)
            assert v.shape[0] == 256, (ratio, k, v.shape)
            assert v.dtype == torch.float32


def test_fast_sampler_offline_fallback_when_buffer_short():
    offline, buf = _small_replay(n_on=10)
    mixed, n_on, n_off = _call_fast(offline, buf, 0.5, seed=3)
    assert (n_on, n_off) == (0, 256)
    assert mixed["observations"].shape[0] == 256


def test_fast_sampler_matches_frozen_contract():
    import torch
    from adaptive_plasticity.m5 import sample_mixed_batch as frozen
    offline, buf = _small_replay()
    for ratio in (0.0, 0.5, 1.0):
        fast, fo, ff_ = _call_fast(offline, buf, ratio, seed=11)
        ref, ro, rf_ = frozen(
            online_buf=buf, offline_data=offline,
            offline_count=len(offline["rewards"]), batch_size=256,
            online_ratio=ratio, rng=np.random.RandomState(11), device="cpu")
        assert (fo, ff_) == (ro, rf_)
        assert set(fast) == set(ref)
        for k in ref:
            assert fast[k].shape == ref[k].shape
            assert fast[k].dtype == ref[k].dtype
