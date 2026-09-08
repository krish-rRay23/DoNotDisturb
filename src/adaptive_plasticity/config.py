"""YAML configuration loading and minimal validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping from *path*.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the file is not a YAML mapping or lacks required M1 keys.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a YAML mapping.")

    required_keys = {"seed", "device", "deterministic", "environment", "dataset", "algorithm", "training", "logging"}
    missing = required_keys.difference(config)
    if missing:
        raise ValueError(f"Configuration is missing required keys: {sorted(missing)}")
    return config

