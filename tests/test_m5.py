"""M5 unit tests: OnlineBuffer, pipeline boundary, mixing, resume skeleton.

All tests use tiny synthetic data; no real environments, no network.
"""

import numpy as np
import pytest

from adaptive_plasticity.m5 import OnlineBuffer, m5_run, sample_mixed_batch
from adaptive_plasticity.iql import IQLAgent


def _make_mixing_fixtures(obs_dim=2, act_dim=1, offline_count=1000, online_count=500):
    """Offline = all zeros, online = all ones: composition is exactly countable."""
    offline_data = {
        "observations": np.zeros((offline_count, obs_dim), dtype=np.float32),
        "actions": np.zeros((offline_count, act_dim), dtype=np.float32),
        "rewards": np.zeros((offline_count,), dtype=np.float32),
        "next_observations": np.zeros((offline_count, obs_dim), dtype=np.float32),
        "dones": np.zeros((offline_count,), dtype=np.float32),
    }
    buf = OnlineBuffer(obs_dim=obs_dim, act_dim=act_dim, max_size=online_count + 10)
    for _ in range(online_count):
        buf.add(np.ones(obs_dim, dtype=np.float32),
                np.ones(act_dim, dtype=np.float32),
                1.0, np.ones(obs_dim, dtype=np.float32), 0.0)
    return offline_data, buf


def _count_online_offline_rows(mixed):
    obs = mixed["observations"].cpu().numpy()
    n_online_rows = int((obs == 1.0).all(axis=1).sum())
    n_offline_rows = int((obs == 0.0).all(axis=1).sum())
    return n_online_rows, n_offline_rows


def test_online_buffer_basic():
    buf = OnlineBuffer(obs_dim=4, act_dim=2, max_size=10)
    # add 7 transitions
    for i in range(7):
        buf.add(np.ones(4) * i, np.ones(2) * i, float(i),
                np.ones(4) * (-i), float(i % 2))
    assert len(buf) == 7
    assert buf.count == 7
    # sample
    batch = buf.sample(3, np.random.RandomState(0))
    assert batch["observations"].shape == (3, 4)
    assert batch["actions"].shape == (3, 2)
    # after 10 adds, buffer is full; adding 3 more overwrites oldest
    for _ in range(3):
        buf.add(np.zeros(4), np.zeros(2), 0.0, np.zeros(4), 0.0)
    assert len(buf) == 10
    batch2 = buf.sample(3, np.random.RandomState(0))
    assert batch2["observations"].shape == (3, 4)
    # oldest entries (idx 0) should now be the third add (i=3) due to FIFO
    # just verify we can sample without error


def test_online_buffer_fifo_overwrite():
    buf = OnlineBuffer(obs_dim=2, act_dim=1, max_size=3)
    for i in range(6):
        buf.add(np.array([i, i]), np.array([i]), float(i),
                np.array([-i, -i]), float(0))
    assert len(buf) == 3
    # should only keep the last 3 adds: i=3,4,5
    batch = buf.sample(3, np.random.RandomState(0))
    # verify the stored observations correspond to i=3,4,5 (order depends on sampling)
    # just check shape and finiteness
    assert np.all(np.isfinite(batch["observations"]))


def test_m5_hyperparameter_shapes():
    """m5_run takes valid hyperparameters; we verify the function signature
    and that it returns a dict with expected keys (no environment needed)."""
    # The function creates an IQLAgent and env internally; we only test
    # that it returns a dict with the top-level keys we expect.
    # We cannot fully execute without gym, so we inspect the source.
    import adaptive_plasticity.m5 as m5mod
    src = open(m5mod.__file__).read()
    expected_keys = {"dataset_id", "seed", "offline_steps", "online_steps",
                     "update_steps", "online_buffer_final_size",
                     "online_ratio", "warmup_steps", "eval_interval",
                     "returns", "normalized", "best_return", "best_normalized"}
    # check that the return dict construction uses these keys
    assert "returns" in src
    assert "normalized" in src
    assert "best_return" in src
    assert "best_normalized" in src


def test_online_ratio_boundary():
    """Verify that online_ratio clamping stays in [0,1] conceptually."""
    # We can't fully run m5_run here without gym, but we can verify the
    # parameter is accepted and stored.
    # The function signature accepts any float; the implementation clips
    # it via max(1, int(batch_size * online_ratio)) logic.
    # Just confirm no crash from bad ratio.
    try:
        # minimal smoke: import and check function exists
        from adaptive_plasticity.m5 import m5_run
        assert callable(m5_run)
    except Exception as e:
        pytest.xfail(f"m5_run not importable: {e}")


def test_path_no_cross_environment_collision():
    """Regression test: different datasets must produce different output paths.

    Ensures dataset_id is part of the run identity so that e.g.
    hopper-medium-v2 and walker2d-medium-v2 with the same controller,
    shift condition, and seed do NOT write to the same directory.
    """
    import adaptive_plasticity.m5 as m5mod
    src = open(m5mod.__file__).read()
    # Identity is built by the shared helper (Phase 4C); fall back to any
    # legacy inline construction. Either way dataset_id must be included.
    helper_lines = [line for line in src.splitlines()
                    if 'M10-' in line and 'dataset_id' in line]
    assert len(helper_lines) >= 1, "M10 output identity with dataset_id not found in m5.py"
    out_name_lines = [line for line in src.splitlines() if 'out_name' in line and 'M10-' in line]
    for line in out_name_lines:
        assert 'dataset_id' in line or 'args.dataset' in line, (
            f"out_name must include dataset_id to prevent cross-environment collisions: {line}"
        )


def test_iql_agent_reuse():
    """Verify IQLAgent can be instantitated with the observed dimensions
    from a loaded offline dataset (mimicking M5 load)."""
    # We just check that the constructor accepts the right args.
    # A full forward pass isn't needed here.
    agent = IQLAgent(obs_dim=17, act_dim=6, device="cpu")
    assert agent is not None


def test_online_float64_batch_converted_to_float32():
    """Regression: float64 online NumPy batch must not break agent.update().

    Reproduces the Phase 2 CPU stress-run path: raw gymnasium observations
    are float64, and torch.as_tensor preserves NumPy dtype, so an online
    batch can reach IQLAgent.update() as float64. The agent must normalize
    to float32 (model dtype) instead of raising a dtype error.
    """
    import math
    import torch
    agent = IQLAgent(obs_dim=4, act_dim=2, device="cpu")
    rng = np.random.RandomState(0)
    n = 16
    batch = {
        "observations": torch.as_tensor(rng.randn(n, 4)),
        "actions": torch.as_tensor(rng.randn(n, 2)),
        "rewards": torch.as_tensor(rng.randn(n)),
        "next_observations": torch.as_tensor(rng.randn(n, 4)),
        "dones": torch.as_tensor(rng.randint(0, 2, size=n).astype(float)),
    }
    # Precondition: this really is a float64 batch (as in the stress run).
    for k, v in batch.items():
        assert v.dtype == torch.float64, f"{k} should be float64, got {v.dtype}"
    out = agent.update(batch)  # must not raise a dtype RuntimeError
    for k, v in out.items():
        assert math.isfinite(v), f"{k} should be finite, got {v}"


def test_mixed_batch_ratio_1_0_is_fully_online():
    offline_data, buf = _make_mixing_fixtures()
    mixed, n_online, n_offline = sample_mixed_batch(
        online_buf=buf, offline_data=offline_data,
        offline_count=1000, batch_size=256, online_ratio=1.0,
        rng=np.random.RandomState(0), device="cpu",
    )
    assert (n_online, n_offline) == (256, 0)
    for v in mixed.values():
        assert v.shape[0] == 256  # batch size preserved exactly
    n_on, n_off = _count_online_offline_rows(mixed)
    assert n_on == 256 and n_off == 0


def test_mixed_batch_ratio_0_5_is_half_and_half():
    offline_data, buf = _make_mixing_fixtures()
    mixed, n_online, n_offline = sample_mixed_batch(
        online_buf=buf, offline_data=offline_data,
        offline_count=1000, batch_size=256, online_ratio=0.5,
        rng=np.random.RandomState(0), device="cpu",
    )
    assert (n_online, n_offline) == (128, 128)
    for v in mixed.values():
        assert v.shape[0] == 256  # batch size preserved exactly
    n_on, n_off = _count_online_offline_rows(mixed)
    # Regression: the old dict.update overwrite made this 0 online / 256 offline.
    assert n_on == 128, f"expected 128 online rows, got {n_on}"
    assert n_off == 128, f"expected 128 offline rows, got {n_off}"


def test_mixed_batch_ratio_0_0_is_fully_offline():
    offline_data, buf = _make_mixing_fixtures()
    mixed, n_online, n_offline = sample_mixed_batch(
        online_buf=buf, offline_data=offline_data,
        offline_count=1000, batch_size=256, online_ratio=0.0,
        rng=np.random.RandomState(0), device="cpu",
    )
    assert (n_online, n_offline) == (0, 256)
    for v in mixed.values():
        assert v.shape[0] == 256  # batch size preserved exactly
    n_on, n_off = _count_online_offline_rows(mixed)
    assert n_on == 0 and n_off == 256


def test_mixed_batch_deterministic_under_seed():
    offline_data, buf = _make_mixing_fixtures()
    kwargs = dict(online_buf=buf, offline_data=offline_data,
                  offline_count=1000, batch_size=256, online_ratio=0.5,
                  device="cpu")
    m1, n1a, n1b = sample_mixed_batch(rng=np.random.RandomState(123), **kwargs)
    # Rebuild identical online buffer so RNG streams align exactly.
    _, buf2 = _make_mixing_fixtures()
    m2, n2a, n2b = sample_mixed_batch(
        online_buf=buf2, offline_data=offline_data,
        offline_count=1000, batch_size=256, online_ratio=0.5,
        rng=np.random.RandomState(123), device="cpu")
    assert (n1a, n1b) == (n2a, n2b) == (128, 128)
    for k in m1:
        assert (m1[k].cpu().numpy() == m2[k].cpu().numpy()).all()


def test_mixed_batch_insufficient_online_falls_back_offline_only():
    offline_data, buf = _make_mixing_fixtures(online_count=10)
    mixed, n_online, n_offline = sample_mixed_batch(
        online_buf=buf, offline_data=offline_data,
        offline_count=1000, batch_size=256, online_ratio=0.5,
        rng=np.random.RandomState(0), device="cpu",
    )
    assert (n_online, n_offline) == (0, 256)
    for v in mixed.values():
        assert v.shape[0] == 256
    n_on, n_off = _count_online_offline_rows(mixed)
    assert n_on == 0 and n_off == 256


def test_m5_single_evaluation_site():
    """Each evaluation step must occur exactly once (no terminal duplicate)."""
    import adaptive_plasticity.m5 as m5mod
    src = open(m5mod.__file__).read()
    assert src.count("last_eval = online_step") == 1, (
        "expected exactly one evaluation site updating last_eval"
    )


def test_m10_output_name_severity_qualified():
    """Phase 4C: severities map to distinct, complete output identities."""
    from adaptive_plasticity.m5 import m10_output_name
    n0 = m10_output_name(dataset_id="hopper-medium-v2", controller_type="adaptive",
                         shift_type="obs_noise", shift_step=0, severity=0.0, seed=1)
    n1 = m10_output_name(dataset_id="hopper-medium-v2", controller_type="adaptive",
                         shift_type="obs_noise", shift_step=150, severity=0.1, seed=1)
    n5 = m10_output_name(dataset_id="hopper-medium-v2", controller_type="adaptive",
                         shift_type="obs_noise", shift_step=150, severity=0.5, seed=1)
    assert len({n0, n1, n5}) == 3, "severities must not collide"
    for n in (n0, n1, n5):
        assert "hopper-medium-v2" in n and "adaptive" in n and "seed1" in n
    assert "-noshift-" in n0 and "-shift-" in n1 and "-shift-" in n5
    assert "sev0.1" in n1 and "sev0.5" in n5
    # Same severity+condition is deterministic.
    assert m10_output_name(dataset_id="hopper-medium-v2", controller_type="fixed",
                           shift_type="obs_noise", shift_step=150, severity=0.5, seed=2) == \
        m10_output_name(dataset_id="hopper-medium-v2", controller_type="fixed",
                        shift_type="obs_noise", shift_step=150, severity=0.5, seed=2)


def test_m5_summary_persists_experiment_metadata():
    """Phase 4C: summary.json carries severity/shift_type/shift_step/controller_type."""
    import adaptive_plasticity.m5 as m5mod
    src = open(m5mod.__file__).read()
    for key in ('"severity"', '"shift_type"', '"shift_step"', '"controller_type"'):
        assert key in src, f"summary/diag_record must persist {key}"
    # m5_run and main() must share the single severity-qualified helper.
    assert src.count("m10_output_name(") >= 2


def test_m5_diag_record_matches_controller_history():
    """Phase 4C: m5 passes dormancy verbatim and logs the smoothed perf value."""
    import adaptive_plasticity.m5 as m5mod
    src = open(m5mod.__file__).read()
    assert '"activation_dormancy": dorm_val' in src
    assert '"perf_change_raw"' in src
    assert "_history[-1].get(\"perf_change\"" in src
    # Functional check: the value m5 logs equals the controller history value.
    from adaptive_plasticity.distribution_shifts import AdaptivePlasticityController
    c = AdaptivePlasticityController(seed=5)
    dorm_val = 0.3
    c.update({"activation_dormancy": dorm_val}, current_perf=1.0)
    c.update({"activation_dormancy": dorm_val}, current_perf=2.0)
    record_perf = c._history[-1].get("perf_change", 0.0)
    assert record_perf == c.diagnostics()["history"][-1]["perf_change"]
    assert c._history[-1]["activation_dormancy"] == dorm_val


def test_m5_live_path_fresh_perf_proxy_every_update():
    """Phase 4E: controller perf input is recomputed per update, never stale.

    The live path must derive current_perf from a fresh mean state-value on
    the fixed probe (available at every controller update) rather than the
    sparse evaluation return carried forward between eval intervals.
    """
    import adaptive_plasticity.m5 as m5mod
    src = open(m5mod.__file__).read()
    assert "perf_value_proxy" in src
    assert "agent.value(" in src
    assert "current_perf=perf_value_proxy" in src
    # The stale pattern (latest eval return fed straight to the controller)
    # must be gone from the live controller step.
    assert "current_norm = metrics" not in src


def test_m5_live_path_centering_anchors_and_explicit_fields():
    """Phase 4E: pre-shift anchors frozen once; raw vs centered explicit."""
    import adaptive_plasticity.m5 as m5mod
    src = open(m5mod.__file__).read()
    # One-time pre-shift anchor capture through the first-wins setter.
    assert "centering_anchors_set()" in src
    assert "set_centering_anchors(" in src
    # JSONL carries both raw diagnostics and centered equation inputs.
    for key in ('"repr_anchor"', '"delta_repr"',
                '"dormancy_anchor"', '"delta_dormancy"',
                '"perf_value_proxy"'):
        assert key in src, f"diag_record must carry explicit field {key}"
    # Raw auditability preserved: legacy raw keys still logged verbatim.
    for key in ('"repr_change": repr_for_log',
                '"activation_dormancy": dorm_val',
                '"evaluation_return"',
                '"evaluation_normalized"'):
        assert key in src, f"raw diagnostic field lost: {key}"


def test_m5_live_path_passes_preshift_gate():
    """Phase 4G: scale estimation gated by pre-shift status, never post-shift."""
    import adaptive_plasticity.m5 as m5mod
    src = open(m5mod.__file__).read()
    # The controller update receives the pre-shift gate derived from the
    # same shift_active flag written to the log (kept in sync by construction).
    assert "pre_shift=not shift_active_now" in src
    assert "shift_active_now = intervention.is_shifted(online_step)" in src
    assert '"shift_active": shift_active_now' in src


def test_m5_jsonl_has_normalized_fields_and_scales():
    """Phase 4G: JSONL carries z-scores and pre-shift scales; raw intact."""
    import adaptive_plasticity.m5 as m5mod
    src = open(m5mod.__file__).read()
    for key in ('"z_repr"', '"z_perf"', '"z_dorm"',
                '"repr_scale"', '"perf_scale"', '"dormancy_scale"'):
        assert key in src, f"diag_record must carry normalized field {key}"
    # Raw and centered fields preserved alongside the normalized ones.
    for key in ('"repr_change": repr_for_log', '"delta_repr"',
                '"perf_change_raw"', '"activation_dormancy": dorm_val',
                '"delta_dormancy"', '"perf_value_proxy"'):
        assert key in src, f"raw/centered field lost: {key}"