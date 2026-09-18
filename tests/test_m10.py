"""M10 unit tests: Adaptive Plasticity Controller.

Tests the AdaptivePlasticityController rule-based controller that dynamically
adjusts intervention strength based on online M6 diagnostics.
"""

import math
import numpy as np
import pytest

from adaptive_plasticity.distribution_shifts import (
    AdaptivePlasticityController,
)


def test_controller_init():
    """AdaptivePlasticityController initializes with configurable weights."""
    c = AdaptivePlasticityController(seed=42)
    assert c.min_severity == 0.0
    assert c.max_severity == 1.0
    assert c.repr_weight == 0.4
    assert c.perf_weight == -0.4  # KEY: negative weight - perf degradation ↑ strength
    assert c.dorm_weight == -0.2
    assert c._history == []

    c2 = AdaptivePlasticityController(
        seed=123,
        repr_weight=0.5,
        perf_weight=0.3,
        dorm_weight=-0.1,
        min_severity=0.1,
        max_severity=0.8,
    )
    assert c2.repr_weight == 0.5
    assert c2.perf_weight == 0.3
    assert c2.dorm_weight == -0.1
    assert c2.min_severity == 0.1
    assert c2.max_severity == 0.8


def test_controller_deterministic():
    """Same seed produces identical results (deterministic)."""
    c = AdaptivePlasticityController(seed=42)
    diag = {
        "repr_change": 0.5,
        "perf_change": -0.1,
        "activation_dormancy": 0.3,
        "param_magnitudes": {"weight1": 0.8, "weight2": 1.2},
    }
    s1 = c.update(diag)
    s2 = c.update(diag)
    assert s1 == s2  # identical call should give identical strength
    assert len(c._history) == 2  # both updates logged


def test_controller_different_seeds():
    """Different seeds produce different results (seeded RNG)."""
    c1 = AdaptivePlasticityController(seed=42)
    c2 = AdaptivePlasticityController(seed=123)
    diag = {"repr_change": 0.5, "perf_change": -0.1, "activation_dormancy": 0.3, "param_magnitudes": {}}
    s1 = c1.update(diag)
    s2 = c2.update(diag)
    # Seeds produce deterministic but different results
    assert 0.0 <= s1 <= 1.0
    assert 0.0 <= s2 <= 1.0


def test_controller_bounds():
    """Strength always returned in [min_severity, max_severity] subset of [0,1]."""
    c = AdaptivePlasticityController(seed=42, min_severity=0.2, max_severity=0.8)
    # Large positive repr_change + positive perf_change => strength at max
    diag_pos = {
        "repr_change": 1.0,
        "perf_change": 1.0,
        "activation_dormancy": 0.0,
        "param_magnitudes": {},
    }
    s = c.update(diag_pos)
    assert s <= 0.8  # capped at max_severity
    assert s >= 0.2  # above min_severity

    # Large negative diagnostics => strength at min
    diag_neg = {
        "repr_change": -1.0,
        "perf_change": -1.0,
        "activation_dormancy": 1.0,
        "param_magnitudes": {},
    }
    s = c.update(diag_neg)
    assert s <= 0.8
    assert s >= 0.2


def test_controller_logging():
    """All decisions and diagnostics are logged for provenance."""
    c = AdaptivePlasticityController(seed=42)
    diag = {
        "repr_change": 0.5,
        "perf_change": -0.2,
        "activation_dormancy": 0.1,
        "param_magnitudes": {"kernel1": 0.5, "kernel2": -0.3},
    }
    strength = c.update(diag)
    diag_out = c.diagnostics()

    assert "history" in diag_out
    assert len(diag_out["history"]) == 1
    entry = diag_out["history"][0]
    assert entry["intervention_strength"] == strength
    assert entry["repr_change"] == 0.5
    assert entry["perf_change"] == -0.2
    assert entry["activation_dormancy"] == 0.1
    assert entry["param_magnitude"] == 0.0  # logged only; not used in control equation


def test_controller_no_history_before_update():
    """Diagnostics returns empty history before any update."""
    c = AdaptivePlasticityController(seed=42)
    diag_out = c.diagnostics()
    assert diag_out["history"] == []


def test_controller_with_zero_diagnostics():
    """Controller with zero diagnostics gives neutral strength."""
    c = AdaptivePlasticityController(seed=42, min_severity=0.0, max_severity=1.0)
    diag = {
        "repr_change": 0.0,
        "perf_change": 0.0,
        "activation_dormancy": 0.0,
        "param_magnitudes": {},
    }
    s = c.update(diag)
    assert s == 0.0  # raw=0, clipped to min_severity=0.0


def test_controller_param_magnitudes_logged_only():
    """param_magnitudes is logged for provenance but NOT used in control equation."""
    c = AdaptivePlasticityController(seed=42)
    diag = {
        "repr_change": 0.0,
        "perf_change": 0.0,
        "activation_dormancy": 0.0,
        "param_magnitudes": {"good": 1.0, "bad": float("nan"), "also_bad": float("inf")},
    }
    s = c.update(diag)
    # With zero other inputs, raw=0 regardless of param_magnitudes
    assert s == 0.0
    # History should show param_magnitude=0.0 (logged but not used)
    diag_out = c.diagnostics()
    assert diag_out["history"][0]["param_magnitude"] == 0.0


def test_controller_apply_noop():
    """Controller.apply() is a no-op, returning obs/reward unchanged."""
    c = AdaptivePlasticityController(seed=42)
    obs = np.array([1.0, 2.0, 3.0])
    reward = 5.0
    result_obs, result_reward = c.apply(obs, reward)
    np.testing.assert_array_equal(result_obs, obs)
    assert result_reward == reward


def test_controller_severity_weights_grounded():
    """Weights are tunable and grounded in plasticity hypothesis (corrected signs)."""
    c = AdaptivePlasticityController(seed=42, min_severity=0.0, max_severity=1.0)
    # With repr_change=0.5, perf_change=0.5, dormancy=0.0:
    # raw = 0.4*0.5 + (-0.4)*0.5 + (-0.2)*0.0 = 0.2 - 0.2 = 0.0
    diag_pos = {
        "repr_change": 0.5,
        "perf_change": 0.5,
        "activation_dormancy": 0.0,
        "param_magnitudes": {},
    }
    s = c.update(diag_pos)
    # raw = 0.4*0.5 + (-0.4)*0.5 + (-0.2)*0.0 = 0.2 - 0.2 = 0.0
    assert abs(s - 0.0) < 0.01, f"Expected ~0.0, got {s}"

    # With repr_change=-0.5, perf_change=-0.5, dormancy=1.0:
    # raw = 0.4*(-0.5) + (-0.4)*(-0.5) + (-0.2)*1.0 = -0.2 + 0.2 - 0.2 = -0.2
    # clipped to [0, 1] with min=0 => 0.0
    diag_neg = {
        "repr_change": -0.5,
        "perf_change": -0.5,
        "activation_dormancy": 1.0,
        "param_magnitudes": {},
    }
    s = c.update(diag_neg)
    # raw = 0.4*(-0.5) + (-0.4)*(-0.5) + (-0.2)*1.0 = -0.2 + 0.2 - 0.2 = -0.2
    # clipped to min=0 => 0.0
    assert s >= 0.0


def test_controller_larger_repr_increases_strength():
    """Regression test: larger representation drift increases strength."""
    c = AdaptivePlasticityController(seed=42, min_severity=0.0, max_severity=1.0)
    # repr_change=1.0, perf_change=0.0, dormancy=0.0:
    # raw = 0.4*1.0 + (-0.4)*0.0 + (-0.2)*0.0 = 0.4
    s_high = c.update({"repr_change": 1.0, "perf_change": 0.0, "activation_dormancy": 0.0, "param_magnitudes": {}})
    # repr_change=0.0, perf_change=0.0, dormancy=0.0:
    # raw = 0.4*0.0 + (-0.4)*0.0 + (-0.2)*0.0 = 0.0
    s_low = c.update({"repr_change": 0.0, "perf_change": 0.0, "activation_dormancy": 0.0, "param_magnitudes": {}})
    assert s_high >= s_low, "Larger repr_change should not decrease strength"


def test_controller_perf_degradation_increases_strength():
    """Regression test: performance degradation increases strength."""
    c = AdaptivePlasticityController(seed=42, min_severity=0.0, max_severity=1.0)
    # perf_change=-1.0 (degradation), repr_change=0.0, dormancy=0.0:
    # raw = 0.4*0.0 + (-0.4)*(-1.0) + (-0.2)*0.0 = 0.4
    s_with_degradation = c.update({"repr_change": 0.0, "perf_change": -1.0, "activation_dormancy": 0.0, "param_magnitudes": {}})
    # perf_change=0.0 (no degradation):
    # raw = 0.4*0.0 + (-0.4)*0.0 + (-0.2)*0.0 = 0.0
    s_no_degradation = c.update({"repr_change": 0.0, "perf_change": 0.0, "activation_dormancy": 0.0, "param_magnitudes": {}})
    assert s_with_degradation >= s_no_degradation, "Performance degradation should increase strength"


def test_controller_higher_dormancy_decreases_strength():
    """Regression test: higher dormancy decreases strength."""
    c = AdaptivePlasticityController(seed=42, min_severity=0.0, max_severity=1.0)
    # dormancy=1.0, repr_change=0.0, perf_change=0.0:
    # raw = 0.4*0.0 + (-0.4)*0.0 + (-0.2)*1.0 = -0.2 => clipped to 0.0
    s_high_dorm = c.update({"repr_change": 0.0, "perf_change": 0.0, "activation_dormancy": 1.0, "param_magnitudes": {}})
    # dormancy=0.0:
    # raw = 0.4*0.0 + (-0.4)*0.0 + (-0.2)*0.0 = 0.0
    s_low_dorm = c.update({"repr_change": 0.0, "perf_change": 0.0, "activation_dormancy": 0.0, "param_magnitudes": {}})
    assert s_high_dorm <= s_low_dorm, "Higher dormancy should not increase strength"


def test_controller_dormancy_changes_strength():
    """Regression test: nonzero activation_dormancy changes controller strength.

    With identical repr_change and perf_change, increasing dormancy
    must decrease strength (dorm_weight=-0.2 < 0).
    Proves the controller receives and uses the real dormancy value
    from m5.py's diagnostic log, not a hardcoded 0.0.
    """
    c = AdaptivePlasticityController(seed=42, min_severity=0.0, max_severity=1.0)
    repr_ch = 0.5
    perf_ch = 0.0

    s_zero = c.update({
        "repr_change": repr_ch,
        "perf_change": perf_ch,
        "activation_dormancy": 0.0,
        "param_magnitudes": {},
    })
    s_nonzero = c.update({
        "repr_change": repr_ch,
        "perf_change": perf_ch,
        "activation_dormancy": 0.5,
        "param_magnitudes": {},
    })

    # raw = 0.4*repr_ch + (-0.4)*perf_ch + (-0.2)*dormancy
    # s_zero:  raw = 0.4*0.5 + 0 + 0 = 0.2
    # s_nonzero: raw = 0.4*0.5 + 0 + (-0.2)*0.5 = 0.2 - 0.1 = 0.1
    assert s_zero > s_nonzero, (
        f"Nonzero dormancy should decrease strength: "
        f"s_zero={s_zero:.4f} <= s_nonzero={s_nonzero:.4f}"
    )
    assert math.isclose(s_zero, 0.2, rel_tol=1e-6)
    assert math.isclose(s_nonzero, 0.1, rel_tol=1e-6)


def test_controller_missing_anchor_is_nan_not_zero():
    """Missing anchor yields NaN (distinguishable from genuine 0.0)."""
    import torch
    from adaptive_plasticity.iql import IQLAgent
    c = AdaptivePlasticityController(seed=42)
    agent = IQLAgent(17, 6, device="cpu")
    assert math.isnan(c._compute_temporal_repr_change(agent))
    # update() with agent but no anchor: history keeps NaN, strength uses 0.0
    s = c.update({"activation_dormancy": 0.0}, agent=agent, current_perf=1.0)
    assert math.isnan(c._history[-1]["repr_change"])
    assert s == 0.0


def test_controller_anchor_not_refreshed_after_update():
    """Phase 1 regression: the anchor is FIXED, not replaced per update.

    set_reference_agent() after the first capture is a no-op (unless
    force/reset_anchor is used), and update() itself never mutates the
    anchor. Drift therefore accumulates vs the run-start anchor instead of
    collapsing to a per-step delta.
    """
    import torch
    from adaptive_plasticity.iql import IQLAgent
    c = AdaptivePlasticityController(seed=42)
    agent = IQLAgent(17, 6, device="cpu")
    c.set_reference_agent(agent)
    anchor_before = {k: v.clone() for k, v in c._prev_agent_params.items()}

    # Perturb the live agent: distance vs anchor must be positive.
    with torch.no_grad():
        for p in agent.policy.parameters():
            p.add_(0.5)
    d1 = c._compute_temporal_repr_change(agent)
    assert math.isfinite(d1) and d1 > 0.0

    # update() must not mutate the anchor.
    c.update({"activation_dormancy": 0.0}, agent=agent, current_perf=1.0)
    for k in anchor_before:
        assert torch.equal(c._prev_agent_params[k], anchor_before[k])

    # A second set_reference_agent() with drifted params must NOT replace it.
    with torch.no_grad():
        for p in agent.policy.parameters():
            p.add_(0.5)
    c.set_reference_agent(agent)
    for k in anchor_before:
        assert torch.equal(c._prev_agent_params[k], anchor_before[k])

    # Distance kept growing vs the ORIGINAL anchor (not reset to ~0).
    d2 = c._compute_temporal_repr_change(agent)
    assert math.isfinite(d2) and d2 >= d1 and d2 > 0.0

    # Explicit reset is the only way to move the anchor.
    c.reset_anchor(agent)
    d3 = c._compute_temporal_repr_change(agent)
    assert math.isfinite(d3) and abs(d3) < 1e-6


def test_controller_perf_ema_smooths_single_eval_spike():
    """Phase 4C: a single noisy evaluation cannot dominate controller strength.

    EMA(alpha=0.5): control input == alpha * raw single-step delta, so a
    one-off spike moves strength half as far as the old raw delta would.
    """
    from adaptive_plasticity.distribution_shifts import PERF_EMA_ALPHA
    assert PERF_EMA_ALPHA == pytest.approx(0.5)
    c = AdaptivePlasticityController(seed=7)
    assert c.perf_ema_alpha == pytest.approx(0.5)
    base = {"repr_change": 0.0, "activation_dormancy": 0.0, "param_magnitudes": {}}
    c.update(dict(base), current_perf=5.0)
    c.update(dict(base), current_perf=5.0)
    c.update(dict(base), current_perf=5.0)
    s = c.update(dict(base), current_perf=9.0)
    last = c._history[-1]
    assert last["perf_change_raw"] == pytest.approx(4.0)
    assert last["perf_change"] == pytest.approx(0.5 * 4.0)
    assert abs(last["perf_change"]) < abs(last["perf_change_raw"])
    # Strength uses the smoothed value: raw_total = -0.4*2.0 = -0.8 -> clip 0.
    assert s == pytest.approx(0.0)


def test_controller_perf_ema_deterministic():
    """Phase 4C: identical evaluation sequences give identical strengths/history."""
    seq = [5.0, 5.0, 6.5, 6.0, 2.0, 4.0]
    run = []
    for _ in range(2):
        c = AdaptivePlasticityController(seed=11)
        strengths = [
            c.update({"repr_change": 0.01, "activation_dormancy": 0.1,
                      "param_magnitudes": {}}, current_perf=p)
            for p in seq
        ]
        run.append((strengths, [dict(h) for h in c._history]))
    assert run[0][0] == run[1][0]
    assert run[0][1] == run[1][1]
    # Smoothed deltas are bounded by raw deltas elementwise after warmup.
    for h in run[0][1][1:]:
        assert abs(h["perf_change"]) <= abs(h["perf_change_raw"]) + 1e-12


def test_controller_input_log_consistency():
    """Phase 4C: logged history matches the inputs that entered the equation."""
    c = AdaptivePlasticityController(seed=21)
    dorm_in = 0.25
    s = c.update({"repr_change": 0.1, "activation_dormancy": dorm_in,
                  "param_magnitudes": {}}, current_perf=3.0)
    s2 = c.update({"repr_change": 0.1, "activation_dormancy": dorm_in,
                   "param_magnitudes": {}}, current_perf=4.0)
    # Dormancy input is logged verbatim.
    assert c._history[0]["activation_dormancy"] == dorm_in
    assert c._history[1]["activation_dormancy"] == dorm_in
    # Raw delta preserved; smoothed input drives strength deterministically.
    assert c._history[1]["perf_change_raw"] == pytest.approx(1.0)
    assert c._history[1]["perf_change"] == pytest.approx(0.5)
    expected_raw = 0.4 * 0.1 + (-0.4) * 0.5 + (-0.2) * dorm_in
    assert s2 == pytest.approx(max(0.0, min(1.0, expected_raw)))
    # diagnostics() provenance mirrors _history exactly.
    logged = c.diagnostics()["history"]
    assert logged[-1]["perf_change"] == c._history[-1]["perf_change"]
    assert logged[-1]["perf_change_raw"] == c._history[-1]["perf_change_raw"]
    assert logged[-1]["intervention_strength"] == s2
    assert s == pytest.approx(0.0)  # first call: perf 0, raw_total<0 -> clip


def test_controller_legacy_direct_perf_path_unchanged():
    """Phase 4C: diagnostics-input path (no current_perf) keeps direct semantics."""
    c = AdaptivePlasticityController(seed=33)
    s = c.update({"repr_change": 0.0, "perf_change": -1.0,
                  "activation_dormancy": 0.0, "param_magnitudes": {}})
    assert c._history[-1]["perf_change"] == pytest.approx(-1.0)
    assert c._history[-1]["perf_change_raw"] == pytest.approx(-1.0)
    # raw_total = -0.4 * -1.0 = 0.4
    assert s == pytest.approx(0.4)


def test_controller_centering_anchors_stable():
    """Phase 4E: pre-shift anchors are first-wins; explicit reset only."""
    c = AdaptivePlasticityController(seed=51)
    assert not c.centering_anchors_set()
    c.set_centering_anchors(repr_anchor=0.004, dormancy_anchor=0.10)
    assert c.centering_anchors_set()
    assert c._repr_anchor == pytest.approx(0.004)
    assert c._dormancy_anchor == pytest.approx(0.10)
    # Second capture without force is ignored (stability).
    c.set_centering_anchors(repr_anchor=0.9, dormancy_anchor=0.9)
    assert c._repr_anchor == pytest.approx(0.004)
    assert c._dormancy_anchor == pytest.approx(0.10)
    # Non-finite candidates never create or overwrite anchors.
    c2 = AdaptivePlasticityController(seed=52)
    c2.set_centering_anchors(repr_anchor=float("nan"), dormancy_anchor=float("inf"))
    assert not c2.centering_anchors_set()
    # Explicit reset is the only way to move them.
    c.reset_centering_anchors(repr_anchor=0.02, dormancy_anchor=0.05)
    assert c._repr_anchor == pytest.approx(0.02)
    assert c._dormancy_anchor == pytest.approx(0.05)


def test_controller_centered_signals_feed_equation():
    """Phase 4E: with anchors set, deltas (not raws) drive strength."""
    c = AdaptivePlasticityController(seed=53)
    c.set_centering_anchors(repr_anchor=0.004, dormancy_anchor=0.10)
    s = c.update({"repr_change": 0.014, "activation_dormancy": 0.12,
                  "param_magnitudes": {}}, current_perf=5.0)
    last = c._history[-1]
    # Raw preserved verbatim; centered deltas explicit.
    assert last["repr_change"] == pytest.approx(0.014)
    assert last["activation_dormancy"] == pytest.approx(0.12)
    assert last["repr_anchor"] == pytest.approx(0.004)
    assert last["dormancy_anchor"] == pytest.approx(0.10)
    assert last["delta_repr"] == pytest.approx(0.010)
    assert last["delta_dormancy"] == pytest.approx(0.02)
    # First perf call -> smoothed 0; equation: 0.4*0.01 - 0.2*0.02 = 0.0.
    assert last["perf_change"] == pytest.approx(0.0)
    assert s == pytest.approx(0.0)
    # Second update: perf EMA still 0 on flat signal; drifted repr moves it.
    s2 = c.update({"repr_change": 0.104, "activation_dormancy": 0.12,
                   "param_magnitudes": {}}, current_perf=5.0)
    last2 = c._history[-1]
    assert last2["delta_repr"] == pytest.approx(0.100)
    assert last2["delta_dormancy"] == pytest.approx(0.02)
    # raw_total = 0.4*0.1 - 0.4*0 - 0.2*0.02 = 0.036
    assert s2 == pytest.approx(0.036)
    # provenance mirrors history exactly.
    logged = c.diagnostics()["history"][-1]
    assert logged["delta_repr"] == last2["delta_repr"]
    assert logged["delta_dormancy"] == last2["delta_dormancy"]


def test_controller_unanchored_passthrough_matches_legacy():
    """Phase 4E: without anchors, deltas equal raws (legacy bit-identical)."""
    c = AdaptivePlasticityController(seed=54)
    s = c.update({"repr_change": 0.05, "perf_change": -0.5,
                  "activation_dormancy": 0.2, "param_magnitudes": {}})
    last = c._history[-1]
    assert last["delta_repr"] == pytest.approx(0.05)
    assert last["delta_dormancy"] == pytest.approx(0.2)
    assert last["repr_anchor"] is None and last["dormancy_anchor"] is None
    # raw_total = 0.4*0.05 + (-0.4)*(-0.5) + (-0.2)*0.2 = 0.02+0.2-0.04 = 0.18
    assert s == pytest.approx(0.18)


def test_controller_fresh_perf_signal_every_update():
    """Phase 4E: a fresh current_perf each call yields a live perf delta.

    Guards the stale-between-evals regression: consecutive updates with a
    moving signal must each record a nonzero raw delta (not a repeated 0.0
    from a carried-forward evaluation).
    """
    c = AdaptivePlasticityController(seed=55)
    base = {"repr_change": 0.0, "activation_dormancy": 0.0, "param_magnitudes": {}}
    seq = [5.0, 5.5, 6.0, 5.75, 6.25]
    for p in seq:
        c.update(dict(base), current_perf=p)
    raws = [h["perf_change_raw"] for h in c._history]
    assert raws[0] == pytest.approx(0.0)  # first call anchors EMA
    assert all(r != pytest.approx(0.0) for r in raws[1:])
    assert raws[1:] == pytest.approx([0.5, 0.5, -0.25, 0.5])
    # Deterministic: same seed + same fresh sequence -> identical outputs.
    c2 = AdaptivePlasticityController(seed=55)
    s1 = [c2.update(dict(base), current_perf=p) for p in seq]
    c3 = AdaptivePlasticityController(seed=55)
    s2 = [c3.update(dict(base), current_perf=p) for p in seq]
    assert s1 == s2
    assert [dict(h) for h in c2._history] == [dict(h) for h in c3._history]


def test_controller_centered_no_nan_inf_strength():
    """Phase 4E: NaN/Inf diagnostics stay auditable but never poison strength."""
    c = AdaptivePlasticityController(seed=56)
    c.set_centering_anchors(repr_anchor=0.01, dormancy_anchor=0.10)
    for diag in ({"repr_change": float("nan"), "activation_dormancy": 0.1, "param_magnitudes": {}},
                 {"repr_change": 0.02, "activation_dormancy": float("inf"), "param_magnitudes": {}},
                 {"repr_change": float("nan"), "activation_dormancy": float("nan"), "param_magnitudes": {}}):
        s = c.update(dict(diag), current_perf=float("nan"))
        assert math.isfinite(s) and 0.0 <= s <= 1.0
    # Raw NaNs preserved in history (distinguishable from genuine 0.0).
    assert math.isnan(c._history[0]["repr_change"])
    assert math.isnan(c._history[0]["delta_repr"])
    assert math.isinf(c._history[1]["activation_dormancy"])
    assert math.isnan(c._history[2]["delta_dormancy"])
    assert all(math.isfinite(h["intervention_strength"]) for h in c._history)


def test_controller_scale_estimated_from_preshift_only():
    """Phase 4G: scales are the population std of finite pre-shift inputs."""
    from adaptive_plasticity.distribution_shifts import DIAG_SCALE_FLOOR
    assert DIAG_SCALE_FLOOR == pytest.approx(1e-6)
    c = AdaptivePlasticityController(seed=61)
    c.set_centering_anchors(repr_anchor=0.0, dormancy_anchor=0.0)
    seq = [(0.02, 0.10, 0.30), (0.04, -0.10, 0.34), (0.06, 0.20, 0.32)]
    for r, p, d in seq:
        c.update({"repr_change": r, "perf_change": p,
                  "activation_dormancy": d, "param_magnitudes": {}},
                 pre_shift=True)
    # Hand-computed population stds over the first two updates' deltas
    # (use-then-update: third update normalizes against updates 1-2).
    last = c._history[-1]
    assert last["repr_scale"] == pytest.approx(0.01)
    assert last["perf_scale"] == pytest.approx(0.10)
    assert last["dormancy_scale"] == pytest.approx(0.02)
    # Final scales additionally fold in the third update: cross-check the
    # Welford estimator against an independent population-std computation.
    import statistics
    assert c.channel_scale("repr") == pytest.approx(statistics.pstdev([0.02, 0.04, 0.06]))
    assert c.channel_scale("perf") == pytest.approx(statistics.pstdev([0.10, -0.10, 0.20]))
    assert c.channel_scale("dorm") == pytest.approx(statistics.pstdev([0.30, 0.34, 0.32]))
    assert last["z_repr"] == pytest.approx(6.0)
    assert last["z_perf"] == pytest.approx(2.0)
    assert last["z_dorm"] == pytest.approx(16.0)
    # raw_total = 0.4*6 - 0.4*2 - 0.2*16 = -1.6 -> clip 0.
    assert last["intervention_strength"] == pytest.approx(0.0)
    assert last["pre_shift"] is True


def test_controller_scale_floor():
    """Phase 4G: zero pre-shift variation normalizes against the floor."""
    from adaptive_plasticity.distribution_shifts import DIAG_SCALE_FLOOR
    c = AdaptivePlasticityController(seed=62)
    c.set_centering_anchors(repr_anchor=0.0, dormancy_anchor=0.0)
    for _ in range(3):
        c.update({"repr_change": 0.05, "perf_change": 0.10,
                  "activation_dormancy": 0.30, "param_magnitudes": {}},
                 pre_shift=True)
    assert c.channel_scale("repr") == pytest.approx(DIAG_SCALE_FLOOR)
    assert c.channel_scale("perf") == pytest.approx(DIAG_SCALE_FLOOR)
    assert c.channel_scale("dorm") == pytest.approx(DIAG_SCALE_FLOOR)
    last = c._history[-1]
    assert last["z_repr"] == pytest.approx(0.05 / DIAG_SCALE_FLOOR)
    assert last["z_perf"] == pytest.approx(0.10 / DIAG_SCALE_FLOOR)
    assert last["z_dorm"] == pytest.approx(0.30 / DIAG_SCALE_FLOOR)
    assert math.isfinite(last["intervention_strength"])
    assert 0.0 <= last["intervention_strength"] <= 1.0


def test_controller_no_postshift_leakage():
    """Phase 4G: post-shift samples never enter the scale estimates."""
    c = AdaptivePlasticityController(seed=63)
    c.set_centering_anchors(repr_anchor=0.0, dormancy_anchor=0.0)
    for r, p, d in [(0.02, 0.10, 0.30), (0.04, -0.10, 0.34)]:
        c.update({"repr_change": r, "perf_change": p,
                  "activation_dormancy": d, "param_magnitudes": {}},
                 pre_shift=True)
    frozen = (c.channel_scale("repr"), c.channel_scale("perf"),
              c.channel_scale("dorm"))
    assert frozen[0] == pytest.approx(0.01)
    # Extreme post-shift observations must not move the scales ...
    s = c.update({"repr_change": 1000.0, "perf_change": -1000.0,
                  "activation_dormancy": 500.0, "param_magnitudes": {}},
                 pre_shift=False)
    assert (c.channel_scale("repr"), c.channel_scale("perf"),
            c.channel_scale("dorm")) == pytest.approx(frozen)
    # ... yet the equation still uses the frozen scales on fresh deltas.
    last = c._history[-1]
    assert last["z_repr"] == pytest.approx(1000.0 / frozen[0])
    assert last["z_perf"] == pytest.approx(-1000.0 / frozen[1])
    assert last["z_dorm"] == pytest.approx(500.0 / frozen[2])
    assert last["pre_shift"] is False
    assert math.isfinite(s) and 0.0 <= s <= 1.0
    # Legacy omission (pre_shift=None) also never accumulates.
    c2 = AdaptivePlasticityController(seed=64)
    c2.update({"repr_change": 3.0, "perf_change": 3.0,
               "activation_dormancy": 3.0, "param_magnitudes": {}})
    assert (c2.channel_scale("repr"), c2.channel_scale("perf"),
            c2.channel_scale("dorm")) == (1.0, 1.0, 1.0)


def test_controller_normalization_deterministic():
    """Phase 4G: identical gated sequences give identical outputs."""
    seq = [(0.02, 0.10, 0.30, True), (0.04, -0.10, 0.34, True),
           (0.06, 0.20, 0.32, True), (0.50, -0.40, 0.60, False)]
    runs = []
    for _ in range(2):
        c = AdaptivePlasticityController(seed=65)
        c.set_centering_anchors(repr_anchor=0.0, dormancy_anchor=0.0)
        strengths = [c.update({"repr_change": r, "perf_change": p,
                               "activation_dormancy": d, "param_magnitudes": {}},
                              pre_shift=f) for r, p, d, f in seq]
        runs.append((strengths, [dict(h) for h in c._history]))
    assert runs[0][0] == runs[1][0]
    assert runs[0][1] == runs[1][1]


def test_controller_hand_computed_normalized_output():
    """Phase 4G: end-to-end hand computation through scales into strength."""
    c = AdaptivePlasticityController(seed=66)
    c.set_centering_anchors(repr_anchor=0.0, dormancy_anchor=0.0)
    for r, p, d in [(0.01, 0.02, 0.05), (0.03, 0.06, 0.15)]:
        c.update({"repr_change": r, "perf_change": p,
                  "activation_dormancy": d, "param_magnitudes": {}},
                 pre_shift=True)
    # Population stds: repr 0.01, perf 0.02, dorm 0.05 (hand-verified).
    assert c.channel_scale("repr") == pytest.approx(0.01)
    assert c.channel_scale("perf") == pytest.approx(0.02)
    assert c.channel_scale("dorm") == pytest.approx(0.05)
    s = c.update({"repr_change": 0.02, "perf_change": 0.01,
                  "activation_dormancy": 0.05, "param_magnitudes": {}},
                 pre_shift=False)
    last = c._history[-1]
    assert last["z_repr"] == pytest.approx(2.0)
    assert last["z_perf"] == pytest.approx(0.5)
    assert last["z_dorm"] == pytest.approx(1.0)
    # raw = 0.4*2.0 - 0.4*0.5 - 0.2*1.0 = 0.4 exactly (interior, unclipped).
    assert s == pytest.approx(0.4)
    # Frozen by the post-shift call above.
    assert c.channel_scale("repr") == pytest.approx(0.01)


def test_controller_normalized_no_nan_inf_strength():
    """Phase 4G: non-finite inputs skip scale estimation, never poison output."""
    c = AdaptivePlasticityController(seed=67)
    c.set_centering_anchors(repr_anchor=0.01, dormancy_anchor=0.10)
    c.update({"repr_change": 0.02, "perf_change": 0.05,
              "activation_dormancy": 0.12, "param_magnitudes": {}},
             current_perf=1.0, pre_shift=True)
    counts_before = [list(st) for st in c._scale_state.values()]
    for diag, perf in [({"repr_change": float("nan"), "activation_dormancy": 0.1,
                         "param_magnitudes": {}}, 2.0),
                       ({"repr_change": 0.02, "activation_dormancy": float("inf"),
                         "param_magnitudes": {}}, 3.0),
                       ({"repr_change": 0.02, "activation_dormancy": 0.12,
                         "param_magnitudes": {}}, float("nan"))]:
        s = c.update(dict(diag), current_perf=perf, pre_shift=True)
        assert math.isfinite(s) and 0.0 <= s <= 1.0
    # NaN/Inf channels skipped; finite channels still accumulated.
    # (setup + updates 2-3 finite for repr; setup + updates 1,3 for dorm.)
    assert c._scale_state["repr"][0] == pytest.approx(counts_before[0][0] + 2.0)
    assert c._scale_state["dorm"][0] == pytest.approx(counts_before[2][0] + 2.0)
    assert math.isnan(c._history[1]["delta_repr"])
    assert math.isnan(c._history[1]["z_repr"])
    assert math.isinf(c._history[2]["activation_dormancy"])
    assert all(math.isfinite(h["intervention_strength"]) for h in c._history)