# Do Not Disturb: When Plasticity Interventions Degrade Offline-to-Online Reinforcement Learning

[![Paper](https://img.shields.io/badge/Paper-ICLR%202027%20Submission-blue.svg)](research_paper/iclr2027/iclr2027_conference.tex)
[![Preview](https://img.shields.io/badge/HTML-Interactive%20Preview-brightgreen.svg)](research_paper/iclr2027/preview.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official code for **"Do Not Disturb: When Plasticity Interventions Degrade Offline-to-Online Reinforcement Learning"** (ICLR 2027 submission).

The numbers below are copied from [`research_paper/iclr2027/iclr2027_conference.tex`](research_paper/iclr2027/iclr2027_conference.tex). There is no separate revised manuscript, no `README_UPDATED.md`, and no response-to-reviewers file in this repository. Nothing in the new ablation plans has been run, and those plans do not change the tables.

ReDo is an **approximate, low-disruption** reset. Units below a relative activation threshold have their incoming weights reinitialized and their outgoing weights zeroed. The forward map is unchanged only for units that were already silent. It is not an identity operator.

---

## Result

Unconditional plasticity restoration is not benign for an offline-to-online locomotion agent. The agent already has a structured policy and value function. Periodic Shrink-and-Perturb during online fine-tuning collapses aggregate interquartile-mean (IQM) normalized return by 68.3%, from **38.78** to **12.30** (paired Wilcoxon signed-rank \(W = 18.0\), \(p = 1.44 \times 10^{-11}\), mean paired difference \(+26.03 \pm 19.09\), \(N = 45\) seed-matched pairs).

**CapacityGate** monitors relative effective rank \(\rho\) and activation dormancy \(d\) on a uniform sample of the online replay buffer, with hysteresis and a refractory cooldown. In the tested stable and moderately shifted transfers, capacity stays intact (\(\rho \in [0.99, 1.03]\), \(d \le 0.085\)) and the gate records **zero interventions** across the 45 primary cells (0 of 1,125 evaluations). Return stays at the baseline IQM of **38.78**.

Under sudden actuator crippling, the operator matters. Shrink-and-Perturb drops Walker2d to **\(-0.53 \pm 0.33\)**. The approximate, low-disruption ReDo reset stays near the unperturbed baseline at **\(8.03 \pm 1.57\)** (baseline \(8.21 \pm 2.38\)). On HalfCheetah, where falling is impossible, both operators exceed the crippled baseline. CapacityGate in that stress table is a **seed-0** cell, not a three-seed mean.

The benchmark is 150 runs: 135 primary factorial runs (3 environments × 3 regimes × 3 arms × 5 seeds) and 15 frozen reference runs. The stress harness is a separate 24-cell plan; the manuscript reports CapacityGate there for seed 0 only.

---

## Questions and findings

**RQ1. Does unconditional periodic intervention help or hurt offline-to-online locomotion?**
Periodic Shrink-and-Perturb is worse in 42 of 45 matched pairs (\(P(\text{none} > \text{fixed}) = 0.933\)). On Walker2d with no extra shift the cell mean falls from \(31.45 \pm 12.65\) to \(2.14 \pm 5.79\).

**RQ2. Can rank and dormancy tell the agent to abstain?**
In the stable and moderate-shift matrix they can. CapacityGate matches the baseline cell by cell because it does not fire. That is measured abstention, not a claim that the same thresholds will abstain under every shift.

**RQ3. Under actuator crippling, how does a large functional change compare with an approximate, low-disruption reset?**
On Walker2d, Shrink-and-Perturb destroys balance and ReDo does not. On HalfCheetah, Shrink-and-Perturb reaches \(15.26 \pm 2.93\) and ReDo \(13.64 \pm 3.18\), against a baseline of \(6.23 \pm 1.53\).

---

## Primary benchmark

Final normalized return after 25,000 online steps. Mean ± standard deviation over 5 paired seeds. CapacityGate matches the baseline in every cell below because it did not intervene.

| Environment | Regime | Baseline | Shrink-and-Perturb | CapacityGate |
| --- | --- | ---: | ---: | ---: |
| halfcheetah-medium-v2 | none | \(35.32 \pm 2.29\) | \(10.71 \pm 5.02\) | \(35.32 \pm 2.29\) |
| | obs. noise \(\sigma=0.1\) | \(34.81 \pm 4.14\) | \(17.89 \pm 9.92\) | \(34.81 \pm 4.14\) |
| | reward scale \(0.5\) | \(37.85 \pm 2.83\) | \(9.49 \pm 7.39\) | \(37.85 \pm 2.83\) |
| hopper-medium-v2 | none | \(40.98 \pm 2.46\) | \(14.87 \pm 8.45\) | \(40.98 \pm 2.46\) |
| | obs. noise \(\sigma=0.1\) | \(41.32 \pm 4.70\) | \(36.85 \pm 18.05\) | \(41.32 \pm 4.70\) |
| | reward scale \(0.5\) | \(36.93 \pm 4.20\) | \(24.56 \pm 7.38\) | \(36.93 \pm 4.20\) |
| walker2d-medium-v2 | none | \(31.45 \pm 12.65\) | \(2.14 \pm 5.79\) | \(31.45 \pm 12.65\) |
| | obs. noise \(\sigma=0.1\) | \(36.16 \pm 19.97\) | \(6.45 \pm 9.05\) | \(36.16 \pm 19.97\) |
| | reward scale \(0.5\) | \(64.33 \pm 11.22\) | \(1.94 \pm 5.64\) | \(64.33 \pm 11.22\) |
| **Aggregate IQM** | 95% stratified bootstrap CI | **38.78** [37.43, 40.03] | **12.30** [9.36, 15.10] | **38.78** [37.43, 40.03] |

The same aggregate comparison has a paired \(t\)-statistic \(t = 9.15\) (\(p = 9.69 \times 10^{-12}\)) and a bootstrap 95% interval on the mean paired difference of \([20.72, 31.59]\) (median paired difference \(+22.27\)).

## Actuator-crippling stress table

Joint 0 is zeroed on HalfCheetah and joint 2 on Walker2d, starting at step 5,000. Means for `none`, `fixed`, and `redo` use seeds 0, 1, and 2. CapacityGate is seed 0 only.

| Environment | Baseline | Shrink-and-Perturb | ReDo | CapacityGate (seed 0) |
| --- | ---: | ---: | ---: | ---: |
| halfcheetah-medium-v2 | \(6.23 \pm 1.53\) | \(15.26 \pm 2.93\) | \(13.64 \pm 3.18\) | 8.58 |
| walker2d-medium-v2 | \(8.21 \pm 2.38\) | \(-0.53 \pm 0.33\) | \(8.03 \pm 1.57\) | 7.23 (matches that seed's baseline) |

On HalfCheetah seed 0 the unperturbed baseline is 7.99 and CapacityGate, which does intervene after the rank trigger, finishes at 8.58. The three-seed baseline mean is 6.23. Those are different summaries of different seeds.

---

## What is not a result yet

These paths exist so they can be launched with one command. They have **not been run**. Do not cite them as findings.

- **Probe distribution.** Published monitoring uses a uniform draw from the online buffer, with the frozen offline batch as the rank anchor and as the fallback before the buffer can fill a probe. The ablation adds the frozen offline batch on every step and recent windows of 256 and 1,024 transitions, on all three locomotion datasets. Config: `configs/ablations/probe_distributions.yaml`.
- **Gate thresholds, cooldown, and operator.** One-factor sweeps around the published settings (\(\rho_{\mathrm{on}}=0.70\), \(\rho_{\mathrm{off}}=0.85\), \(d_{\mathrm{on}}=0.15\), \(d_{\mathrm{off}}=0.08\), cooldown 3,000, Shrink-and-Perturb). The alternate operator under the gate is ReDo. Configs: `configs/ablations/`.
- **More stress seeds.** The manuscript does not report CapacityGate seeds 1–2 or any seed above 2. `experiments/analyze_stress_coverage.py` prints that coverage. It does not fill in the missing cells.
- **Cal-QL, PEX, and RO2O.** Not vendored. `src/adaptive_plasticity/o2o_adapters.py` records the papers, the upstream repositories, and the methods a wrapper must implement. Calling one raises `ExternalLearnerUnavailable`. The in-repo learner is IQL.

An older note in `docs/final_study_v2_scientific_report.md` describes a different protocol and different cell means. It is not the manuscript table.

The supplementary zip's README quotes IQM intervals \([37.43, 40.06]\) and \([9.31, 15.23]\). The manuscript, and `results/final_study/FINAL_analysis_summary.json` inside that zip, use **[37.43, 40.03]** and **[9.36, 15.10]** after rounding. Use the manuscript intervals.

---

## Layout

```
configs/ablations/                 # probe, threshold, cooldown, operator grids
experiments/final_factorial.py     # 135 + 15 confirmatory plan
experiments/stress_factorial.py    # 24-cell stress harness and seed extension
experiments/run_ablations.py       # ablation dry-run, smoke, and launch
experiments/analyze_stress_coverage.py
src/adaptive_plasticity/           # CapacityGate, ReDo, IQL, probe sources, adapters
research_paper/iclr2027/           # manuscript
tests/                             # unit tests, including the new plans
```

Published CapacityGate defaults, also the reference cells in the ablation configs: probe source `online_buffer`, probe size 256, dormancy on/off 0.15/0.08, relative rank on/off 0.70/0.85, cooldown 3,000 steps, operator `shrink_perturb`, intervention severity 0.1.

---

## Install and check

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
# Headless MuJoCo here needs glfw. MUJOCO_GL=egl failed to import.
export MUJOCO_GL=glfw
make check
```

`make check` runs the unit tests (skipping cells that load gitignored datasets), the plan dry-run, a read-only manifest hash check, and the public-claim token check. It does not train and it does not download. The Makefile sets `MUJOCO_GL=glfw` when the variable is unset. An absent raw file does not fail the hash check; a present file with a different sha256 does.

`requirements.txt` pins every direct dependency with `==` (Python 3.12). `torch==2.14.0` installs a CUDA build; this environment has no GPU, and the runners accept `--device cpu`.

Full MuJoCo training is intentionally not started by the commands above. Exact launch commands, dataset acquisition, and the confirm tokens are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

Raw D4RL files are not in git. Acquire them before any training command:

```bash
python -m adaptive_plasticity.datasets --acquire hopper-medium-v2
```

Repeat for `halfcheetah-medium-v2` and `walker2d-medium-v2`.

---

## Citation

The manuscript is an anonymous conference submission. The bibliography entry in the paper stays anonymous.

```bibtex
@inproceedings{donotdisturb2027,
  title={Do Not Disturb: When Plasticity Interventions Degrade Offline-to-Online Reinforcement Learning},
  author={Anonymous Authors},
  booktitle={International Conference on Learning Representations},
  year={2027}
}
```

## License

MIT. See [LICENSE](LICENSE).
