import random

import numpy as np
import torch

from adaptive_plasticity.reproducibility import create_metadata, detect_device, prepare_run_directories, seed_everything


def test_seed_reproducibility() -> None:
    seed_everything(123)
    first = (random.random(), np.random.rand(), torch.rand(1).item())
    seed_everything(123)
    second = (random.random(), np.random.rand(), torch.rand(1).item())
    assert first == second


def test_metadata_contains_runtime_fields(tmp_path) -> None:
    metadata = create_metadata({"seed": 1}, "cpu", tmp_path)
    assert metadata["hostname"]
    assert metadata["created_at"]
    assert "torch" in metadata["package_versions"]
    assert metadata["device"]["selected"] == "cpu"


def test_device_detection() -> None:
    assert detect_device("auto") in {"cpu", "cuda"}
    assert detect_device("cpu") == "cpu"


def test_prepare_checkpoint_directory(tmp_path) -> None:
    paths = prepare_run_directories(tmp_path, "test-run")
    assert paths["checkpoints"] == tmp_path / "checkpoints" / "test-run"
    assert paths["checkpoints"].is_dir()

