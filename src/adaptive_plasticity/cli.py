"""Configuration-only smoke-test entry point for M1."""

from __future__ import annotations

import argparse
from pathlib import Path

from adaptive_plasticity.config import load_config
from adaptive_plasticity.logging_utils import configure_logging
from adaptive_plasticity.reproducibility import create_metadata, detect_device, prepare_run_directories, seed_everything, write_metadata


def main() -> None:
    """Initialize a reproducible run; no algorithm or training is performed."""
    parser = argparse.ArgumentParser(description="Initialize an Adaptive Plasticity RL experiment run.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML experiment configuration.")
    parser.add_argument("--output-root", type=Path, default=Path("."), help="Root for generated run artifacts.")
    arguments = parser.parse_args()

    config = load_config(arguments.config)
    device = detect_device(str(config["device"]))
    seed_everything(int(config["seed"]), bool(config["deterministic"]))
    metadata = create_metadata(config, device, repository_root=arguments.output_root)
    paths = prepare_run_directories(arguments.output_root, metadata["run_id"])
    logger = configure_logging(paths["logs"] / "run.jsonl", str(config["logging"]["level"]))
    metadata_path = write_metadata(metadata, paths["results"] / "metadata.json")
    logger.info("M1 smoke test initialized; no training was run.", extra={"event": "smoke_test_initialized"})
    print(f"Run initialized: {metadata['run_id']}")
    print(f"Device: {device}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()

