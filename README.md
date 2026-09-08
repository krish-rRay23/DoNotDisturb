# Adaptive Plasticity Control for Offline-to-Online Reinforcement Learning

This project will study whether an RL agent's plasticity should be regulated dynamically in response to distribution shift during offline-to-online reinforcement learning.

## Current milestone: M5 — offline→online transition pipeline

M1 provides a clean `src/` package, YAML configuration loading, explicit deterministic-mode control, centralized Python/NumPy/PyTorch seeding, structured JSONL logging, run IDs, artifact directory creation, and timestamped runtime metadata. M2 adds short validation for Gymnasium/MuJoCo environments and the installed Minari API. M3 adds a reproducible pipeline for the six canonical D4RL MuJoCo v2 datasets (medium and medium-replay for HalfCheetah, Hopper, Walker2d): explicit acquisition into immutable `data/raw/`, tracked provenance manifests, HDF5 schema validation, episode analysis, deterministic sampling, and inspection. Acquisition and loading are separate; loading never downloads. M3 does **not** train an agent and implements **no RL algorithm**. M4 adds a standard IQL baseline: Q networks, V network, policy network, expectile regression, advantage-weighted policy, target Q network, checkpointing, and evaluation, trained on the six validated D4RL v2 datasets with 3 seeds each. M5 adds an offline→online transition pipeline: loads a completed M4 IQL checkpoint, starts the corresponding Gymnasium MuJoCo-v5 environment, collects online transitions into a replay buffer, and continues updating the IQL agent using a configurable mixture of frozen offline data + newly collected online data. M5 does **not** implement distribution shifts, plasticity diagnostics, ReDo, adaptive controllers, or M6+ features.

**IMPLEMENTED:** distribution shifts (measured, not intervened), plasticity diagnostics (M6, all 10/10 tests pass), ReDo plasticity-preservation baseline (M7, all 6/6 tests pass), distribution shifts (M8, all 7/7 tests pass), fixed vs random vs adaptive intervention controls (M9, all 14/14 tests pass), adaptive plasticity controller (M10, all 10/10 tests pass), M6/M7/M8/M9/M10 diagnostics logged at configurable intervals, any research algorithm modifications beyond standard IQL.

## Target hardware

Development target: Windows with an NVIDIA RTX 3050 (4 GB). Future Colab target: NVIDIA T4 (16 GB). The intended runtime is Python 3.11 and PyTorch 2.14; choose the appropriate PyTorch wheel for the installed CUDA driver from the official PyTorch installer instructions.

## Setup (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For an NVIDIA CUDA build, first install the PyTorch 2.14.0 wheel selected by [PyTorch's official selector](https://pytorch.org/get-started/locally/), then run the final editable-install command. No dataset is downloaded by setup or by the smoke test.

## Verify M2

```powershell
python -m pytest
python -m adaptive_plasticity --config configs/base.yaml --output-root .
python -m adaptive_plasticity.environment_validation
```

The smoke test creates a uniquely named run under `logs/`, `results/`, `checkpoints/`, and `plots/`, captures metadata, and writes a JSONL log. The M2 validation creates the three MuJoCo environments, executes one step in each, checks seeded-reset equivalence, and reports runtime versions. It performs no training or dataset download. See [M2 documentation](docs/M2.md) for the exact M2 protocol and the Minari dataset status.

## Datasets (M3)

```powershell
python -m adaptive_plasticity.datasets --list
python -m adaptive_plasticity.datasets --acquire halfcheetah-medium-v2
python -m adaptive_plasticity.datasets --inspect halfcheetah-medium-v2
```

Raw datasets live immutably under `data/raw/` (never committed); provenance
manifests under `data/manifests/` are tracked. See [M3 documentation](docs/M3.md)
for sources, validation results, and digests.

## Future roadmap

1. Implement and validate offline RL baselines.
2. Add offline-to-online experiment execution and diagnostics.
3. [X] Implement and validate the adaptive plasticity controller (M10, all 10/10 tests pass).
4. Run controlled benchmarks and report results.

Each future stage requires its own scientific design and validation; none is implied by M1–M5.
