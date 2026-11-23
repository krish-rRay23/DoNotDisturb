"""Interfaces for running CapacityGate beside another offline-to-online learner.

``IQLLearnerAdapter`` wraps the IQL agent in this repository. Cal-QL, PEX,
and RO2O are not vendored. Their specs record the upstream paper and
repository and the methods a wrapper must implement. Calling
``require_external_learner`` raises until that wrapper exists. No scores from
those codebases are reported here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from adaptive_plasticity.capacity_gate import CapacityGateConfig, CapacityGateController
from adaptive_plasticity.interventions import apply_plasticity_intervention

EXTERNAL_STATUS = "not_vendored"

__all__ = [
    "EXTERNAL_STATUS",
    "ExternalLearnerSpec",
    "ExternalLearnerUnavailable",
    "EXTERNAL_LEARNERS",
    "require_external_learner",
    "IQLLearnerAdapter",
    "CapacityGateSession",
    "GateHookConfig",
    "capacity_config_from_hook",
    "load_gate_hook",
    "load_external_hooks",
    "open_external_session",
]


@dataclass(frozen=True)
class ExternalLearnerSpec:
    """What is required before CapacityGate can sit on an external learner."""

    key: str
    method: str
    paper: str
    arxiv: str
    code: str
    framework: str
    status: str
    wrapper_requirements: tuple[str, ...]
    notes: str


class ExternalLearnerUnavailable(RuntimeError):
    """Raised when an external offline-to-online codebase is not wrapped."""


EXTERNAL_LEARNERS: dict[str, ExternalLearnerSpec] = {
    "cal_ql": ExternalLearnerSpec(
        key="cal_ql",
        method="Cal-QL",
        paper=(
            "Nakamoto, Zhai, Singh, Sobol Mark, Ma, Finn, Kumar, and Levine. "
            "Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning. "
            "NeurIPS 2023."
        ),
        arxiv="https://arxiv.org/abs/2303.05479",
        code="https://github.com/nakamotoo/Cal-QL",
        framework="jax",
        status=EXTERNAL_STATUS,
        wrapper_requirements=(
            "Load an offline Cal-QL checkpoint and expose the online fine-tuning step.",
            "Provide observation batches that CapacityGate can run through a policy and critic.",
            "Apply shrink-and-perturb or ReDo to the parameters that the online optimizer steps, and clear the corresponding optimizer state.",
            "Keep the Cal-QL calibration term unchanged; the gate only decides whether an intervention runs.",
        ),
        notes=(
            "The upstream repository is JAX/Flax and its scripts cover AntMaze and Adroit. "
            "A locomotion D4RL fine-tuning entry point is not in that repository, so a "
            "faithful wrapper needs a locomotion port before any comparison is meaningful. "
            "This package does not vendor or reimplement Cal-QL."
        ),
    ),
    "pex": ExternalLearnerSpec(
        key="pex",
        method="PEX",
        paper=(
            "Zhang, Xu, and Yu. Policy Expansion for Bridging Offline-to-Online "
            "Reinforcement Learning. ICLR 2023."
        ),
        arxiv="https://arxiv.org/abs/2302.00935",
        code="https://github.com/Haichao-Zhang/PEX",
        framework="pytorch",
        status=EXTERNAL_STATUS,
        wrapper_requirements=(
            "Train or load the offline policy with main_offline.py and expand it in main_online.py.",
            "Expose the frozen offline policy and the learnable online policy as modules whose hidden ReLU activations can be probed.",
            "Intervene only on the learnable online policy and its critic. Leave the frozen offline policy untouched.",
            "Pass the same probe batch the gate used into ReDo when the operator is redo.",
        ),
        notes=(
            "Upstream PEX is PyTorch and includes a locomotion example "
            "(halfcheetah-random-v2) as well as AntMaze. Clone that repository beside "
            "this one and implement PEXLearnerAdapter with the methods on IQLLearnerAdapter. "
            "This package does not vendor PEX."
        ),
    ),
    "ro2o": ExternalLearnerSpec(
        key="ro2o",
        method="RO2O",
        paper=(
            "Wen, Yu, Yang, Chen, Bai, and Wang. Towards Robust Offline-to-Online "
            "Reinforcement Learning via Uncertainty and Smoothness. JAIR 81, 2024."
        ),
        arxiv="https://arxiv.org/abs/2309.16973",
        code="https://github.com/BattleWen/RO2O",
        framework="pytorch",
        status=EXTERNAL_STATUS,
        wrapper_requirements=(
            "Load a checkpoint produced by algorithms/offline/ro2o_mujoco.py.",
            "Hook the online update inside algorithms/finetune/ro2o_ft_mujoco.py.",
            "Expose the ensemble critic and the actor, including which parameters the smoothness and uncertainty penalties update.",
            "Apply a gated intervention to those parameters without removing the RO2O penalties.",
        ),
        notes=(
            "Upstream RO2O is PyTorch and already has MuJoCo offline and fine-tuning "
            "scripts. This package does not vendor RO2O or report RO2O scores."
        ),
    ),
}


@dataclass(frozen=True)
class GateHookConfig:
    """Thresholds and which modules a wrapper may edit.

    External learners stay ``not_yet_run`` until their code is wrapped.
    Loading a hook does not import that code and does not report a score.
    """

    learner: str
    operator: str = "shrink_perturb"
    dorm_on: float = 0.15
    dorm_off: float = 0.08
    rho_on: float = 0.70
    rho_off: float = 0.85
    cooldown_steps: int = 3000
    active_severity: float = 0.1
    editable_components: tuple[str, ...] = ("critic", "policy")
    frozen_components: tuple[str, ...] = ()
    status: str = "not_yet_run"


def capacity_config_from_hook(hook: GateHookConfig) -> CapacityGateConfig:
    """Build the in-repo gate config from a hook. Does not construct a learner."""
    if hook.operator not in ("shrink_perturb", "redo"):
        raise ValueError(f"Unknown gate operator {hook.operator!r}.")
    overlap = set(hook.editable_components) & set(hook.frozen_components)
    if overlap:
        raise ValueError(f"Components cannot be both editable and frozen: {sorted(overlap)}.")
    return CapacityGateConfig(
        dorm_on=float(hook.dorm_on),
        dorm_off=float(hook.dorm_off),
        rho_on=float(hook.rho_on),
        rho_off=float(hook.rho_off),
        active_severity=float(hook.active_severity),
        cooldown_steps=int(hook.cooldown_steps),
    )


def _as_tuple(value: Any, field: str) -> tuple[str, ...]:
    if isinstance(value, str) or value is None:
        raise ValueError(f"{field} must be a list of component names.")
    return tuple(str(item) for item in value)


def load_gate_hook(path: str | Path) -> GateHookConfig:
    """Read one hook file. External hooks must stay ``not_yet_run``."""
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Gate hook {path} must be a mapping.")
    learner = str(document["learner"])
    status = str(document.get("status", "not_yet_run"))
    if learner in EXTERNAL_LEARNERS and status != "not_yet_run":
        raise ValueError(
            f"{learner} is not wrapped. Its hook must stay status 'not_yet_run', got {status!r}."
        )
    if learner not in EXTERNAL_LEARNERS and learner != "iql":
        raise ValueError(f"Unknown learner {learner!r}.")
    hook = GateHookConfig(
        learner=learner,
        operator=str(document.get("operator", "shrink_perturb")),
        dorm_on=float(document.get("dorm_on", 0.15)),
        dorm_off=float(document.get("dorm_off", 0.08)),
        rho_on=float(document.get("rho_on", 0.70)),
        rho_off=float(document.get("rho_off", 0.85)),
        cooldown_steps=int(document.get("cooldown_steps", 3000)),
        active_severity=float(document.get("active_severity", 0.1)),
        editable_components=_as_tuple(
            document.get("editable_components", ["critic", "policy"]),
            "editable_components",
        ),
        frozen_components=_as_tuple(document.get("frozen_components", []), "frozen_components"),
        status=status,
    )
    capacity_config_from_hook(hook)
    return hook


def external_hook_dir(root: str | Path | None = None) -> Path:
    base = Path(__file__).resolve().parents[2] if root is None else Path(root)
    return base / "configs" / "o2o"


def load_external_hooks(root: str | Path | None = None) -> dict[str, GateHookConfig]:
    """Load the Cal-QL, PEX, and RO2O hook files. None of them import upstream code."""
    hooks: dict[str, GateHookConfig] = {}
    for path in sorted(external_hook_dir(root).glob("*.yaml")):
        hook = load_gate_hook(path)
        hooks[hook.learner] = hook
    missing = set(EXTERNAL_LEARNERS) - set(hooks)
    if missing:
        raise FileNotFoundError(f"Missing external hook files for {sorted(missing)}.")
    return hooks


def open_external_session(key: str, adapter: Any) -> None:
    """Refuse to build a session for an unwrapped external learner.

    The adapter argument is accepted so a future wrapper can use the same
    call. Today every external key raises ``ExternalLearnerUnavailable``.
    """
    del adapter
    require_external_learner(key)


def require_external_learner(key: str) -> ExternalLearnerSpec:
    """Refuse to pretend an external learner is available.

    Returns nothing on success. Cal-QL, PEX, and RO2O always raise until a
    wrapper is added in this repository.
    """
    if key not in EXTERNAL_LEARNERS:
        known = ", ".join(sorted(EXTERNAL_LEARNERS))
        raise KeyError(f"Unknown external learner {key!r}. Known: {known}.")
    spec = EXTERNAL_LEARNERS[key]
    requirements = "\n".join(f"  - {item}" for item in spec.wrapper_requirements)
    raise ExternalLearnerUnavailable(
        f"{spec.method} is {spec.status} and is not importable from this repository.\n"
        f"Paper: {spec.paper}\n"
        f"arXiv: {spec.arxiv}\n"
        f"Code: {spec.code}\n"
        f"Framework: {spec.framework}\n"
        f"Wrapper requirements:\n{requirements}\n"
        f"{spec.notes}"
    )


class IQLLearnerAdapter:
    """View of the in-repo IQL agent used by CapacityGateSession."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent

    @property
    def name(self) -> str:
        return "iql"

    def policy_module(self) -> torch.nn.Module:
        return self.agent.policy

    def critic_module(self) -> torch.nn.Module:
        return self.agent.critic

    def actor_optimizer(self) -> torch.optim.Optimizer | None:
        return getattr(self.agent, "actor_optimizer", None)

    def critic_optimizer(self) -> torch.optim.Optimizer | None:
        return getattr(self.agent, "critic_optimizer", None)


class CapacityGateSession:
    """Run CapacityGate diagnostics, and optionally one operator, on an adapter.

    ``evaluate`` never mutates parameters. ``maybe_apply`` calls the existing
    intervention helper only when the gate's ``should_intervene`` flag is set.
    """

    def __init__(
        self,
        adapter: Any,
        config: CapacityGateConfig | None = None,
        operator: str = "shrink_perturb",
        *,
        editable_components: tuple[str, ...] = ("critic", "policy"),
        frozen_components: tuple[str, ...] = (),
    ) -> None:
        if operator not in ("shrink_perturb", "redo"):
            raise ValueError(f"Unknown gate operator {operator!r}.")
        overlap = set(editable_components) & set(frozen_components)
        if overlap:
            raise ValueError(f"Components cannot be both editable and frozen: {sorted(overlap)}.")
        self.adapter = adapter
        self.operator = operator
        self.editable_components = tuple(editable_components)
        self.frozen_components = tuple(frozen_components)
        self.gate = CapacityGateController(config=config or CapacityGateConfig())

    @classmethod
    def from_hook(cls, adapter: Any, hook: GateHookConfig) -> "CapacityGateSession":
        """Build a session from a hook. External learners still have to be wrapped."""
        if hook.learner in EXTERNAL_LEARNERS:
            open_external_session(hook.learner, adapter)
        return cls(
            adapter,
            config=capacity_config_from_hook(hook),
            operator=hook.operator,
            editable_components=hook.editable_components,
            frozen_components=hook.frozen_components,
        )

    def _subject(self) -> Any:
        """Object whose ``policy`` and ``critic`` modules the gate can read.

        The in-repo IQL adapter exposes ``.agent``. A mock or a future wrapper
        can be that object itself.
        """
        return self.adapter.agent if hasattr(self.adapter, "agent") else self.adapter

    def evaluate(
        self,
        probe_batch: torch.Tensor,
        step: int,
    ) -> dict[str, Any]:
        record = self.gate.update_from_agent(self._subject(), probe_batch, step=step)
        record["learner"] = self.adapter.name
        record["gate_operator"] = self.operator
        return record

    def maybe_apply(
        self,
        probe_batch: torch.Tensor,
        step: int,
        *,
        rng: Any = None,
    ) -> dict[str, Any]:
        record = self.evaluate(probe_batch, step)
        record["applied"] = False
        if record.get("should_intervene", False):
            apply_plasticity_intervention(
                self._subject(),
                intervention_type=self.operator,
                severity=self.gate.config.active_severity,
                shrink=0.1,
                probe_batch=probe_batch,
                rng=rng,
                target_components=self.editable_components,
            )
            record["applied"] = True
        return record
