"""The hyperparameter table and the compute note come from the repo, not a new run."""

from __future__ import annotations

from pathlib import Path

import yaml

from adaptive_plasticity.capacity_gate import CapacityGateConfig
from adaptive_plasticity.final_runner import (
    ACTUATOR_CRIPPLE_SEVERITY,
    PUBLISHED_GATE,
    REGIME_SEVERITY,
    final_protocol,
)
from adaptive_plasticity.iql import IQLAgent

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "research_paper" / "iclr2027" / "iclr2027_conference.tex"


def _section(tex: str, start: str, end: str) -> str:
    return tex.split(start, 1)[1].split(end, 1)[0]


def _comma(value: int) -> str:
    return f"{value:,}"


def test_hyperparameter_rows_match_the_runner_and_the_iql_config():
    tex = TEX.read_text(encoding="utf-8")
    table = _section(tex, "\\label{app:hyperparams}", "\\section{Compute}")
    protocol = final_protocol()
    iql = yaml.safe_load((ROOT / "configs" / "iql.yaml").read_text(encoding="utf-8"))
    parameters = iql["algorithm"]["parameters"]
    agent = IQLAgent(1, 1)
    assert parameters["hidden_dim"] == agent.policy.backbone.network[0].out_features
    assert parameters["hidden_layers"] == 2
    assert parameters["max_advantage"] == agent.max_advantage
    assert PUBLISHED_GATE["dorm_off"] != CapacityGateConfig().dorm_off
    assert PUBLISHED_GATE["cooldown_steps"] != CapacityGateConfig().cooldown_steps
    assert f"${parameters['hidden_layers']}$ layers, width ${parameters['hidden_dim']}$" in table
    assert f"${protocol['eval_episodes']}$ episodes every ${_comma(protocol['eval_interval'])}$" in table
    assert f"Advantage cap & ${int(parameters['max_advantage'])}$" in table
    assert f"Observation-noise severity & ${REGIME_SEVERITY['obs_noise']:.2f}$" in table
    assert f"Reward-scale severity & ${REGIME_SEVERITY['reward_scale']:.2f}$" in table
    assert f"Actuator-cripple severity & ${ACTUATOR_CRIPPLE_SEVERITY:.1f}$" in table
    assert f"${PUBLISHED_GATE['dorm_off']:.2f}$" in table
    assert f"${_comma(int(PUBLISHED_GATE['cooldown_steps']))}$" in table
    assert "class defaults are not the values passed into the run" in table
    assert "No new score is reported." in table


def test_compute_is_unknown_where_the_repository_does_not_record_it():
    tex = TEX.read_text(encoding="utf-8")
    compute = _section(tex, "\\section{Compute}", "\\section{Diagnostic Telemetry")
    assert "They are unknown." in compute
    assert "No compute figure is reported." in compute
    for token in ("0.9915", "1.6501", "40.37", "RTX", "3050"):
        assert token not in compute
