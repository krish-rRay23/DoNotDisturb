# Adaptive Plasticity Control for Offline-to-Online Reinforcement Learning

<p align="center">
  <strong>Can an RL agent learn when to preserve plasticity and when to restore it?</strong>
</p>

<p align="center">
  A research-oriented study of adaptive plasticity control under distribution shift.
</p>

---

## Overview

Offline-to-online reinforcement learning combines **offline pretraining** with continued interaction in a changing environment.

A central challenge is **plasticity**: the ability of a neural network to continue learning from new data without destroying useful representations acquired previously.

Most plasticity-preserving approaches apply a **fixed intervention** throughout training.

This project investigates a different question:

> **Should the amount of plasticity intervention remain fixed when the learning regime itself is changing?**

We study an **Adaptive Plasticity Controller** that adjusts intervention strength using signals from the agent's learning dynamics.

The goal is to determine whether **adaptive control of plasticity** can provide a better trade-off between:

- retaining useful prior representations,
- adapting to new distributions,
- maintaining performance under distribution shift.

---

## Research Question

### Main question

> **Can an RL agent dynamically regulate its plasticity according to the degree of distribution shift, instead of relying on a fixed plasticity-preserving intervention?**

### Hypothesis

A fixed intervention may be unnecessarily restrictive during periods where rapid adaptation is required and insufficient when prior representations become unstable.

We therefore hypothesize:

> **An adaptive controller can preserve prior knowledge when stability matters and restore plasticity when adaptation is required.**

This project treats plasticity intervention as a **meta-control problem over learning dynamics**.

---

## Method

```text
Offline Dataset
      │
      ▼
   IQL Agent
      │
      ▼
Offline → Online Transition
      │
      ▼
   Online RL
      │
      ├───────────────┐
      │               │
      ▼               ▼
   Fixed          Adaptive
Intervention      Controller
      │               │
      └───────┬───────┘
              ▼
       Distribution Shift
              │
              ▼
        RL Performance
```

### Adaptive Controller

The controller uses three diagnostics:

- **Representation change**
- **Performance change**
- **Activation dormancy**

The current rule-based controller computes:

```text
raw =
    0.4 × representation_change
  − 0.4 × performance_change
  − 0.2 × activation_dormancy

strength = clip(raw, min_severity, max_severity)
```

The resulting intervention strength is applied dynamically during online learning.

---

## Experimental Setup

### Environments

- HalfCheetah-v5
- Hopper-v5
- Walker2d-v5

### Offline data

D4RL-style datasets:

- `medium`
- `medium-replay`

for the three environments.

### Offline algorithm

**Implicit Q-Learning (IQL)** is used for offline pretraining.

### Baselines

The M10 study compares:

```text
Fixed Intervention
        vs
Adaptive Intervention
```

under:

```text
No Distribution Shift
        vs
Distribution Shift
```

with three random seeds per condition.

---

## Distribution Shifts

Controlled shifts can be introduced during the online phase.

### Observation Noise

```text
obs' = obs + N(0, severity²)
```

### Reward Scaling

Rewards can also be modified through controlled scaling.

The shift mechanism is deterministic with respect to the experiment seed and activates only after the configured shift step.

---

## Experimental Progress

The project is organized into twelve research milestones.

| Milestone | Description | Status |
|---|---|---|
| **M1** | Reproducible research repository | ✅ |
| **M2** | Environment and dataset validation | ✅ |
| **M3** | Dataset pipeline | ✅ |
| **M4** | IQL offline baseline | ✅ |
| **M5** | Offline → online RL pipeline | ✅ |
| **M6** | Plasticity diagnostics | ✅ |
| **M7** | ReDo-inspired baseline | ✅ |
| **M8** | Controlled distribution shifts | ✅ |
| **M9** | Fixed/random intervention controls | ✅ |
| **M10** | Adaptive Plasticity Controller | ✅ |
| **M11** | Learned controller | Conditional |
| **M12** | Final study, analysis and paper artifacts | Pending |

**M11 is intentionally conditional.** A learned controller will only be implemented if the M10 evidence provides a strong enough scientific basis.

---

## Current M10 Experiment

The completed M10 study consists of:

```text
3 environments
× 2 conditions
× 2 controller types
× 3 seeds
= 36 runs
```

Controller types:

```text
Adaptive
Fixed
```

Conditions:

```text
Shift
No-shift
```

Each run uses the same training budget and evaluation protocol so that the comparison isolates the effect of the intervention strategy.

---

## Results

The current M10 study is analyzed across:

### Primary metric

**Final normalized return**

### Secondary metrics

- Best normalized return
- Learning-curve AUC
- Shift-induced performance degradation
- Controller intervention strength
- Representation dynamics
- Performance change
- Activation dormancy

Results are reported across individual seeds rather than relying only on aggregate averages.

> **Important:** The project does not assume that Adaptive will outperform Fixed in every environment. The scientific objective is to determine where adaptive control helps, where it fails, and why.

Detailed experiment outputs live under:

```text
results/
```

Final analysis is stored under:

```text
results/M10_analysis/
```

---

## Repository Structure

```text
adaptive-plasticity-control/
│
├── configs/
│   └── Experiment configuration files
│
├── src/
│   └── adaptive_plasticity/
│       ├── algorithms/
│       ├── datasets/
│       ├── environments/
│       ├── diagnostics/
│       ├── interventions/
│       └── experiments/
│
├── tests/
│   └── Unit and integration tests
│
├── scripts/
│   └── Training, evaluation and analysis utilities
│
├── experiments/
│   └── Experiment definitions and launch configurations
│
├── results/
│   ├── M10_analysis/
│   └── experiment outputs
│
├── checkpoints/
│   └── Model checkpoints
│
├── plots/
│   └── Generated visualizations
│
├── logs/
│   └── Training and diagnostic logs
│
├── docs/
│   └── Research notes and documentation
│
├── data/
│   ├── raw/
│   └── processed/
│
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## Reproducibility

The project is designed around deterministic and auditable experiments.

Each experiment records:

```text
environment
dataset
seed
algorithm configuration
shift configuration
controller configuration
training budget
evaluation configuration
software/runtime metadata
```

Raw datasets are kept separate from processed experiment artifacts.

Experiment outputs are stored independently for each seed and condition to avoid accidental aggregation or overwriting.

---

## Installation

Python **3.11+** is recommended.

```bash
git clone https://github.com/<username>/adaptive-plasticity-control.git
cd adaptive-plasticity-control

python -m venv .venv
```

### Windows

```powershell
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -e .
```

---

## Quick Start

Run the experiment launcher:

```bash
python -m adaptive_plasticity
```

Specific experiments are configured through:

```text
configs/
```

Training outputs, checkpoints, logs and evaluation results are written to the configured experiment directories.

---

## Testing

Run:

```bash
pytest
```

The test suite covers core components including:

- dataset loading
- deterministic sampling
- IQL updates
- offline-to-online transitions
- intervention behavior
- distribution shifts
- plasticity diagnostics
- adaptive controller logic
- experiment reproducibility

---

## Research Philosophy

This repository is intended as a **research artifact**, not simply an RL implementation.

### 1. Controlled comparisons

Adaptive and fixed interventions are evaluated under the same environments, seeds, training budgets and shift configurations.

### 2. Mechanistic analysis

Performance alone is not sufficient.

The project also examines **why** the controller behaves differently by tracking representation dynamics, performance changes and neuron activity.

### 3. Negative results matter

A controller is not considered successful merely because one benchmark improves.

The final conclusion is based on:

```text
performance
+
consistency across seeds
+
robustness under shift
+
controller behavior
+
ablation evidence
```

---

## Research Roadmap

```text
M1  ── Reproducibility
M2  ── Environment / Dataset Validation
M3  ── Dataset Pipeline
M4  ── IQL Baseline
M5  ── Offline → Online RL
M6  ── Plasticity Diagnostics
M7  ── ReDo Baseline
M8  ── Distribution Shifts
M9  ── Fixed / Random Controls
M10 ── Adaptive Plasticity Controller
       │
       ├──► Analyze evidence
       │
       └──► M11 Learned Controller
                │
                ▼
              M12
       Final Study + Paper
```

---

## Why This Project?

The underlying idea is simple:

> **Learning dynamics are not stationary, so plasticity control should not necessarily be stationary either.**

Instead of asking only:

> "How do we preserve plasticity?"

this project asks:

> **"Can an agent decide how much plasticity it needs right now?"**

That shifts plasticity preservation from a static regularization problem toward a **dynamic control problem over the learning process itself**.

---

## Citation

A formal citation will be added with the final research report.

```bibtex
@software{ray_adaptive_plasticity_control,
  author  = {Krish Ray},
  title   = {Adaptive Plasticity Control for Offline-to-Online Reinforcement Learning},
  year    = {2026},
  url     = {https://github.com/<username>/adaptive-plasticity-control}
}
```

---

## Status

**Research prototype — M10 experimental study complete.**

The next stage is rigorous analysis of the 36-run study, followed by a decision on whether a learned controller is scientifically justified.

```text
Build → Measure → Analyze → Validate → Publish
```
