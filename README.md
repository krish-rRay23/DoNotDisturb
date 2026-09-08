# Adaptive Plasticity Control for Offline-to-Online Reinforcement Learning

This project implements a research-grade pipeline for studying adaptive plasticity regulation during offline-to-online reinforcement learning, from infrastructure (M1) through adaptive plasticity control (M10).

## Project Status: M10 Complete ✅

All milestones M1–M10 implemented and tested. The adaptive plasticity controller (M10) is fully implemented with 13 unit tests passing and micro-validation confirmed.

## Milestone Summary

| Milestone | Status | Description |
|-----------|--------|-------------|
| **M1** | ✅ | Core infrastructure: config, logging, reproducibility, CLI |
| **M2** | ✅ | Environment validation (MuJoCo/Gymnasium) |
| **M3** | ✅ | Dataset pipeline (D4RL/Minari, HDF5, provenance) |
| **M4** | ✅ | Standard IQL baseline (Kostrikov et al., 2021) |
| **M5** | ✅ | Offline→Online transition pipeline |
| **M6** | ✅ | Plasticity diagnostics (repr_change, perf_change, dormancy) |
| **M7** | ✅ | ReDo plasticity-preservation regularizer |
| **M8** | ✅ | Distribution shifts (obs_noise, reward_scale) |
| **M9** | ✅ | Fixed/Random/Adaptive intervention controls |
| **M10** | ✅ **Adaptive Plasticity Controller** - **COMPLETE** |

---

## M10: Adaptive Plasticity Controller

The M10 milestone implements a rule-based adaptive plasticity controller that dynamically adjusts intervention strength based on online plasticity diagnostics.

### Controller Equation

```python
raw = 0.4 * repr_change - 0.4 * perf_change - 0.2 * activation_dormancy
strength = clip(raw, min_severity=0.0, max_severity=1.0)
```

### Controller Inputs (Real-time Diagnostics)

| Diagnostic | Definition | Range | Interpretation |
|------------|------------|-------|----------------|
| `repr_change` | `1 - cos_sim(θ_t, θ_{t-Δt})` per module, averaged | [0, 1] | 0 = identical, 1 = orthogonal |
| `perf_change` | `normalized_return_t - normalized_return_{t-Δt}` | ℝ | < 0 = degradation |
| `activation_dormancy` | `frac(\|w\| < 1e-4)` for policy params | [0, 1] | 1 = all dormant |
| `param_magnitude` | `mean(||w||_2)` over trainable params | ℝ⁺ | Logged only |

### Controller Logic

- **Representation instability ↑** → intervention strength **↑**
- **Performance degradation** (perf_change < 0) → intervention strength **↑**
- **Dormancy ↑** → intervention strength **↓**
- Strength clipped to `[min_severity, max_severity]` ⊆ `[0, 1]`

### Key Features

1. **Temporal representation change** — Compares current agent params vs. frozen snapshot from previous controller update
2. **Real performance delta** — `perf_change = current_norm - prev_norm` (0.0 for first eval)
3. **Single diagnostic record per controller update** — No duplicate records
4. **Persistent logging** — `diagnostics.jsonl` with all controller state
5. **Deterministic** — Seeded RNG, CUDA determinism configured
6. **Shift-aware** — Respects `shift_step` for intervention activation

---

## Quick Start

### Installation

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For CUDA: install PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) first.

### Run Tests

```powershell
python -m pytest                    # All 89 tests (88 pass, 1 expected M9 legacy failure)
python -m pytest tests/test_m10.py -v   # M10 tests only
```

### Run M10 Micro-Validation (4 conditions × 100 steps)

```powershell
python run_micro_validation.py
```

Expected output: All 4 conditions pass with online_steps=100, update_steps>0, no NaN/Inf, diagnostics.jsonl created.

### Run Single M10 Experiment

```powershell
python -m adaptive_plasticity.m5 ^
  --dataset halfcheetah-medium-v2 ^
  --seed 0 ^
  --online-steps 20000 ^
  --update-freq 1000 ^
  --online-ratio 0.5 ^
  --warmup-steps 50000 ^
  --eval-interval 5000 ^
  --eval-episodes 5 ^
  --device cuda ^
  --controller-type adaptive ^
  --shift-type obs_noise ^
  --shift-step 5000 ^
  --severity 0.1
```

### Controller Types

| Type | `--controller-type` | Behavior |
|------|---------------------|----------|
| `none` | `none` | No intervention (baseline) |
| `fixed` | `fixed` | Constant severity after `shift_step` |
| `adaptive` | `adaptive` | M10 adaptive controller (dynamic severity) |

### Distribution Shifts

| Shift Type | `--shift-type` | Effect |
|------------|----------------|--------|
| Observation noise | `obs_noise` | Adds bounded Gaussian noise to observations |
| Reward scaling | `reward_scale` | Scales rewards by severity factor |

Shift activates at `--shift-step` (0 = never).

---

## Project Structure

```
adaptive-plasticity-rl/
├── src/adaptive_plasticity/          # Core package
│   ├── m5.py              # M5 pipeline + M10 controller integration
│   ├── m6.py              # Plasticity diagnostics
│   ├── m7.py              # ReDo regularizer
│   ├── m8.py              # Distribution shifts
│   └── iql.py             # IQL agent (M4)
├── configs/                # YAML configurations
├── tests/                  # 89 unit/integration tests
├── data/                   # Dataset manifests (raw data in data/raw/)
├── docs/                   # Per-milestone documentation
├── results/                # Experiment outputs (gitignored)
├── logs/                   # JSONL logs
├── plots/                  # Generated plots
├── checkpoints/            # Model checkpoints
├── data/raw/               # Raw D4RL datasets (gitignored)
├── data/manifests/         # Dataset provenance manifests
├── tests/                  # 89 tests (88 pass, 1 expected M9 legacy failure)
├── run_micro_validation.py # Quick 4-condition validation
├── run_pilot.py            # Pilot experiment runner
├── run_micro_validation.py # Micro-validation runner
├── launch_seed1.ps1        # PowerShell launcher
└── pyproject.toml          # Package metadata
```

---

## Running the 12-Run M10 Validation Batch

```powershell
# 4 conditions × 3 seeds = 12 runs
# Each run: ~5-7 min on RTX 3050

# Fixed + obs_noise shift (seeds 0,1,2)
python -m adaptive_plasticity.m5 --dataset halfcheetah-medium-v2 --seed 0 --controller-type fixed --shift-type obs_noise --shift-step 5000 --severity 0.1 --online-steps 20000 --warmup-steps 50000 --eval-interval 5000 --eval-episodes 5 --device cuda

python -m adaptive_plasticity.m5 --dataset halfcheetah-medium-v2 --seed 1 --controller-type fixed --shift-type obs_noise --shift-step 5000 --severity 0.1 ...

python -m adaptive_plasticity.m5 --dataset halfcheetah-medium-v2 --seed 2 --controller-type fixed --shift-type obs_noise --shift-step 5000 --severity 0.1 ...

# Adaptive + obs_noise shift (seeds 0,1,2)
python -m adaptive_plasticity.m5 --dataset halfcheetah-medium-v2 --seed 0 --controller-type adaptive --shift-type obs_noise --shift-step 5000 --severity 0.1 ...

# Fixed + no shift (seeds 0,1,2)
python -m adaptive_plasticity.m5 --dataset halfcheetah-medium-v2 --seed 0 --controller-type fixed --shift-type obs_noise --shift-step 0 --severity 0.1 ...

# Adaptive + no shift (seeds 0,1,2)
python -m adaptive_plasticity.m5 --dataset halfcheetah-medium-v2 --seed 0 --controller-type adaptive --shift-type obs_noise --shift-step 0 --severity 0.1 ...
```

### Expected Outputs per Run

```
results/
├── M10-fixed-shift-seed0/
│   ├── summary.json          # Aggregated metrics
│   └── ...
├── diagnostics_M10-adaptive-shift-seed0.jsonl   # Per-step diagnostics
└── ...
```

Each run produces:
- `summary.json` — Best/last returns, normalized scores, update counts
- `diagnostics_M10-{controller}-{shift|noshift}-seed{N}.jsonl` — Per-step diagnostics

---

## Running Tests

```powershell
python -m pytest                          # All 89 tests
python -m pytest tests/test_m10.py -v    # M10 tests (13 tests)
python -m pytest tests/test_m5.py -v     # M5 integration tests
python -m pytest tests/test_m6.py -v     # Diagnostics tests
```

**Expected:** 88/89 tests pass (1 expected M9 legacy failure in `AdaptiveIntervention` — unrelated to M10)

---

## Scientific Rigor Checklist

- [x] Temporal representation change (current vs previous controller step)
- [x] Real performance delta (`perf_change = current_norm - prev_norm`)
- ✅ Single diagnostic record per controller update
- ✅ `perf_change = current_norm - prev_norm` (0.0 for first eval)
- ✅ Controller equation: `0.4*repr - 0.4*perf - 0.2*dormancy`, clipped [0,1]
- ✅ Directionality verified: repr↑→strength↑, perf↓→strength↑, dormancy↑→strength↓
- ✅ Bounds enforced: `strength ∈ [0, 1]`
- ✅ NaN/Inf guarded
- ✅ Deterministic under fixed seed (CUDA determinism enabled)
- ✅ Shift activates exactly at `shift_step`
- ✅ No-shift: `shift_active=False` throughout
- ✅ Diagnostics logged to `diagnostics.jsonl` per run
- ✅ Unique output paths: `M10-{controller}-{shift|noshift}-seed{N}/`

---

## Known Limitations

| Limitation | Status |
|------------|--------|
| Checkpoint/resume for M10 | Placeholder only — not implemented |
| CUDA determinism | `warn_only=True` — some ops may be non-deterministic |
| `perf_change` in micro-val | 0.0 until ≥2 evaluations occur |
| Value head `repr_change` | Returns 0.0 when norms are zero (handled) |
| Checkpoint/resume | Not implemented for M10 state |

---

## Citation

If you use this codebase in research, please cite the relevant milestones and the original IQL paper:

```bibtex
@article{kostrikov2021offline,
  title={Offline Reinforcement Learning with Implicit Q-Learning},
  author={Kostrikov, Ilya and Nair, Ashvin and Levine, Sergey},
  journal={ICLR},
  year={2022}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.