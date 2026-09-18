# Do Not Disturb: When Plasticity Interventions Degrade Offline-to-Online Reinforcement Learning

[![Paper](https://img.shields.io/badge/Paper-ICLR%202027%20Submission-blue.svg)](research_paper/iclr2027/iclr2027_conference.tex)
[![Preview](https://img.shields.io/badge/HTML-Interactive%20Preview-brightgreen.svg)](research_paper/iclr2027/preview.html)
[![Tests](https://img.shields.io/badge/Tests-88%2B%20Passing-success.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official research repository and codebase for the paper:
**"Do Not Disturb: When Plasticity Interventions Degrade Offline-to-Online Reinforcement Learning"** (Target: ICLR 2027).

---

## Executive Summary

Maintaining neural network plasticity is widely regarded as an indispensable prerequisite for continual reinforcement learning (RL). A vibrant literature advocates periodic parameter interventions—such as network resets, Shrink-and-Perturb, and dormant neuron recycling—to counteract capacity loss, dead units, and feature rank collapse. While effective in continual supervised learning and online RL from scratch, these methods rest on an implicit assumption: **that plasticity restoration is universally benign and can be scheduled unconditionally.**

In this work, we demonstrate that this assumption fails fundamentally in **offline-to-online (O2O) continuous-control RL**. When transitioning from static pretraining to online interaction, agents inherit structured, high-performing policy and value manifolds. Across standard D4RL continuous-control locomotion benchmarks, we show that unconditional periodic Shrink-and-Perturb induces severe **intervention vulnerability**, destabilizing converged locomotion policies and causing a catastrophic **68.3% collapse in aggregate Interquartile Mean (IQM) normalized return** ($38.78 \to 12.30$, paired Wilcoxon signed-rank $W = 18.0$, $p = 1.44 \times 10^{-11}$).

To address this vulnerability, we formulate the **"Do Not Disturb" principle**:
> *An agent should only intervene to restore representation capacity when diagnostic telemetry confirms that capacity has collapsed; otherwise, converged policy manifolds should be left undisturbed.*

We introduce **`CapacityGate`**, a closed-loop diagnostic framework that monitors feature effective rank ($\rho$) and neuron dormancy ($d$) using uniform replay reservoir sampling and refractory cooldown control. Furthermore, under non-stationary physical stress (actuator crippling), we uncover a profound **operator stability asymmetry**: zero-functional-shift neuron recycling (ReDo) preserves balance manifolds where unconstrained weight perturbation causes total dynamical collapse.

---

## Core Research Questions & Findings

### RQ1: Intervention Vulnerability
*Does unconditional periodic parameter intervention restore or degrade policy performance during offline-to-online transfer in continuous locomotion?*
- **Finding**: Unconditional periodic intervention is overwhelmingly detrimental in stable transfer. Across 45 paired evaluations ($N=150$ total matched runs), periodic Shrink-and-Perturb degrades performance in **93.3% of comparisons**, dropping aggregate IQM from $38.78$ to $12.30$. In balance-critical bipedal locomotion (Walker2d), it induces immediate and permanent falling ($31.45 \to 2.14$).

### RQ2: Diagnostic Feasibility & Abstention
*Can online representation health metrics reliably detect intact capacity and govern intervention decisions through closed-loop abstention?*
- **Finding**: In stable and moderately shifted O2O transfer, representations do not spontaneously collapse: relative effective rank remains near baseline ($\rho \in [0.99, 1.03]$) and dormant neuron fractions remain minimal ($d \le 0.085$). `CapacityGate` detects that capacity remains intact and safely **abstains from intervention**, preserving 100% of baseline performance ($38.78$ IQM) and completely preventing perturbation-induced collapse.

### RQ3: Operator Sensitivity Under Non-Stationarity
*Under genuine physical distribution shifts (actuator crippling), how do functional-shift versus zero-functional-shift intervention operators differ in preserving continuous control equilibria?*
- **Finding**: Intervention operators exhibit a stark stability asymmetry dictated by their functional shift:
  - On **Walker2d** (bipedal balance), non-zero functional shift operators ($\Delta f_\theta(x) \neq 0$) cause total dynamic collapse ($-0.53 \pm 0.33$). In contrast, zero-functional-shift dormant neuron recycling ($W_{\text{out}} = 0$, ReDo) preserves viable walking gait dynamics ($8.03 \pm 1.57$, matching the unperturbed baseline).
  - On **HalfCheetah** (unconstrained balance), breaking the offline prior via weight perturbation promotes alternative galloping gaits, boosting return ($15.26$ vs. $6.23$ baseline).

---

## Key Empirical Results

### Primary Confirmatory Benchmark (135 Runs, 5 Paired Seeds)

| Environment | Shift Regime | Baseline (None) | Fixed (Shrink-Perturb) | CapacityGate |
| :--- | :--- | :---: | :---: | :---: |
| **HalfCheetah-v2** | None | $35.32 \pm 2.29$ | $10.71 \pm 5.02$ | $\mathbf{35.32 \pm 2.29}$ |
| | Obs Noise ($\sigma=0.1$) | $34.81 \pm 4.14$ | $17.89 \pm 9.92$ | $\mathbf{34.81 \pm 4.14}$ |
| | Reward Scale ($\alpha=0.5$) | $37.85 \pm 2.83$ | $9.49 \pm 7.39$ | $\mathbf{37.85 \pm 2.83}$ |
| **Hopper-v2** | None | $40.98 \pm 2.46$ | $14.87 \pm 8.45$ | $\mathbf{40.98 \pm 2.46}$ |
| | Obs Noise ($\sigma=0.1$) | $41.32 \pm 4.70$ | $36.85 \pm 18.05$ | $\mathbf{41.32 \pm 4.70}$ |
| | Reward Scale ($\alpha=0.5$) | $36.93 \pm 4.20$ | $24.56 \pm 7.38$ | $\mathbf{36.93 \pm 4.20}$ |
| **Walker2d-v2** | None | $31.45 \pm 12.65$ | $2.14 \pm 5.79$ | $\mathbf{31.45 \pm 12.65}$ |
| | Obs Noise ($\sigma=0.1$) | $36.16 \pm 19.97$ | $6.45 \pm 9.05$ | $\mathbf{36.16 \pm 19.97}$ |
| | Reward Scale ($\alpha=0.5$) | $64.33 \pm 11.22$ | $1.94 \pm 5.64$ | $\mathbf{64.33 \pm 11.22}$ |
| **Aggregate IQM** | **Stratified Bootstrap [95% CI]** | **38.78** [37.43, 40.03] | **12.30** [9.36, 15.10] | **38.78** [37.43, 40.03] |

### Non-Stationary Stress Study (Actuator Crippling at Step 5,000)

| Environment | Baseline (None) | Fixed (Shrink-Perturb) | ReDo (Neuron Recycling) | CapacityGate |
| :--- | :---: | :---: | :---: | :---: |
| **HalfCheetah-v2** | $6.23 \pm 1.53$ | $\mathbf{15.26 \pm 2.93}$ | $13.64 \pm 3.18$ | $8.58$ (Single-Cell Validated) |
| **Walker2d-v2** | $8.21 \pm 2.38$ | $-0.53 \pm 0.33$ | $\mathbf{8.03 \pm 1.57}$ | $7.23$ (Baseline Parity) |

---

## Architecture Overview

```
adaptive-plasticity-rl/
├── configs/                            # Experiment and diagnostic configuration files
├── data/manifests/                     # D4RL dataset SHA256 provenance manifests
├── experiments/                        # Reproducible experiment runners
│   ├── final_factorial.py              # Primary 135-run confirmatory factorial suite
│   ├── stress_factorial.py             # 24-run non-stationary stress study suite
│   ├── final_analysis.py               # Statistical evaluation & rliable integration
│   └── stress_analysis.py              # Stress test diagnostics & plotting
├── research_paper/                     # Complete ICLR 2027 conference submission
│   ├── iclr2027/
│   │   ├── iclr2027_conference.tex     # Full LaTeX manuscript (all sections & proofs)
│   │   ├── iclr2027_conference.bib     # Curated 2016-2026 bibliography
│   │   ├── iclr2027_conference.sty     # Official conference style file
│   │   ├── preview.html                # Interactive HTML paper preview
│   │   └── figures/                    # Publication-grade vector and PNG figures
│   └── iclr2027_paper_submission.zip   # Overleaf-ready submission archive
├── src/adaptive_plasticity/            # Core Python package
│   ├── capacity_gate.py                # Closed-loop diagnostic CapacityGate controller
│   ├── interventions.py                # Shrink-and-Perturb and ReDo operators
│   ├── final_runner.py                 # Resilient online adaptation runner
│   ├── offline_cache.py                # Decoupled offline pretraining cache
│   ├── distribution_shifts.py          # Observation noise, reward scaling & actuator crippling
│   ├── iql.py                          # Implicit Q-Learning baseline implementation
│   └── m6.py                           # Effective rank (sRank) & dormancy diagnostics
└── tests/                              # Pytest test suite (>88 unit and integration tests)
```

---

## Quick Start

### 1. Installation

```bash
# Create and activate Python 3.11 virtual environment
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install package in editable mode
pip install --upgrade pip
pip install -e ".[dev]"
```

### 2. Run Test Suite

```bash
# Run all unit and integration tests
pytest tests/ -q

# Run specific CapacityGate and intervention tests
pytest tests/test_capacity_gate.py -v
pytest tests/test_interventions.py -v
pytest tests/test_final_protocol.py -v
```

### 3. Run Experiments

```bash
# Run primary 135-run factorial experiment suite across 3 seeds/environments
python experiments/final_factorial.py --workers 4

# Run non-stationary physical stress study (actuator crippling)
python experiments/stress_factorial.py --workers 2

# Compute rigorous statistical analysis (IQM, Wilcoxon signed-rank, CIs)
python scratch/full_scientific_analysis.py
```

### 4. Paper Artifacts & Verification

- **Full LaTeX Manuscript**: Located in [`research_paper/iclr2027/iclr2027_conference.tex`](research_paper/iclr2027/iclr2027_conference.tex).
- **Interactive Preview**: Open [`research_paper/iclr2027/preview.html`](research_paper/iclr2027/preview.html) in any modern web browser.
- **Overleaf Submission ZIP**: [`research_paper/iclr2027_paper_submission.zip`](research_paper/iclr2027_paper_submission.zip).
- **LaTeX Syntax Audit**: Run `python scratch/validate_latex.py` to verify zero unescaped characters, matching environments, and cross-reference validity.

---

## Citation

If you find this work or codebase useful in your research, please cite:

```bibtex
@article{anukulana2027donotdisturb,
  title={Do Not Disturb: When Plasticity Interventions Degrade Offline-to-Online Reinforcement Learning},
  author={Anonymous Authors},
  journal={International Conference on Learning Representations (ICLR)},
  year={2027}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.