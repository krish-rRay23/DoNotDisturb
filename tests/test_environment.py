import numpy as np
import pytest

from adaptive_plasticity.environment_validation import ENVIRONMENT_IDS, runtime_report, validate_environment


@pytest.mark.parametrize("environment_id", ENVIRONMENT_IDS)
def test_mujoco_environment_reset_step_and_determinism(environment_id: str) -> None:
    report = validate_environment(environment_id, seed=314)
    assert report["status"] == "validated"
    assert report["deterministic_reset"] is True
    assert report["observation_shape"]
    assert report["action_shape"]
    assert len(report["step_structure"]) == 5


def test_environment_observations_are_equivalent_for_same_seed() -> None:
    first = validate_environment("HalfCheetah-v5", seed=99)
    second = validate_environment("HalfCheetah-v5", seed=99)
    assert first["observation_shape"] == second["observation_shape"]


def test_runtime_report_is_cpu_safe() -> None:
    report = runtime_report()
    assert report["python_version"]
    assert isinstance(report["cuda_available"], bool)
    if not report["cuda_available"]:
        assert report["gpu_name"] is None

