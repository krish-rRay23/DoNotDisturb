from pathlib import Path

from adaptive_plasticity.config import load_config


def test_load_base_config() -> None:
    config = load_config(Path("configs/base.yaml"))
    assert config["seed"] == 42
    assert config["training"]["steps"] is None

