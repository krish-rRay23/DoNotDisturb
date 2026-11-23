# Reproducibility

Reported numbers come from `research_paper/iclr2027/iclr2027_conference.tex`. This file does not recompute them. A revised manuscript and a response to reviewers are **not in the repository**. `README_UPDATED.md` is not in the repository.

Pinned packages for the checks run with this revision are in `requirements.txt` (CPython 3.12, Linux x86_64). `torch==2.14.0` is the cu130 wheel. `torch.cuda.is_available()` was false here, so every command below uses CPU unless you pass `--device cuda` on a machine with a GPU.

Headless MuJoCo: `MUJOCO_GL=egl` failed to import here (`eglQueryString` missing). `MUJOCO_GL=glfw` imported and stepped `Hopper-v5`. Export it before training if EGL is unavailable:

```bash
export MUJOCO_GL=glfw
```

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

## Manuscript numbers (do not replace these from a pilot)

Primary aggregate IQM, stratified bootstrap 95% CI, from Table 1 of the manuscript:

| Arm | IQM | 95% CI |
| --- | ---: | --- |
| none | 38.78 | [37.43, 40.03] |
| fixed (Shrink-and-Perturb) | 12.30 | [9.36, 15.10] |
| CapacityGate | 38.78 | [37.43, 40.03] |

Paired Wilcoxon on the 45 baseline-minus-fixed pairs: \(W = 18.0\), \(p = 1.44 \times 10^{-11}\), mean paired difference \(+26.03 \pm 19.09\).

The copy of `FINAL_analysis_summary.json` under `results/final_study/` inside `DoNotDisturb_ICLR2027_Supplementary.zip` rounds to those intervals (IQM 38.7848 → 38.78, CI high 40.025 → 40.03, fixed CI [9.362, 15.100] → [9.36, 15.10]). An older summary at `results/FINAL_analysis_summary.json` in the same zip does **not** match the paper (fixed IQM about 40.6). Do not use it. The README inside that zip quotes [37.43, 40.06] and [9.31, 15.23]; those are not the manuscript intervals.

Stress means are the manuscript's Table `tab:stress_results`. CapacityGate is seed 0 only (HalfCheetah 8.58, Walker2d 7.23). `results/stress_analysis/stress_iqm.json` in the zip pools six CapacityGate scores, including seeds the paper does not report, and is not that table.

`docs/final_study_v2_scientific_report.md` is an earlier protocol with different cell means. It is not a source for the paper.

## Checks that do not train

```bash
pytest tests/ -q
# CI uses the same suite without the cells that load gitignored D4RL files:
MUJOCO_GL=glfw pytest tests/ -q -m "not needs_offline_data"

python experiments/final_factorial.py --help
python experiments/stress_factorial.py
python experiments/run_ablations.py
python experiments/run_ablations.py --smoke
python experiments/analyze_stress_coverage.py
python experiments/check_public_claims.py
python experiments/aggregate_summaries.py --indir results/ablation_pilot --outdir results/aggregate_check --replicates 200
python -c "from adaptive_plasticity.o2o_adapters import require_external_learner; require_external_learner('cal_ql')"
```

The last command exits with `ExternalLearnerUnavailable`. That is the expected result: Cal-QL, PEX, and RO2O are not installed here. The same call with `pex` or `ro2o` raises as well. Hook files under `src/adaptive_plasticity/o2o_hooks/` record thresholds and which modules a future wrapper may edit. They are package data, so they load from an installed copy. They do not import those repositories.

`python experiments/check_public_claims.py` exits 0 when the README, the manuscript, and the HTML preview still contain the transcribed tokens. It does not recompute them. `python experiments/stress_factorial.py` prints the 24-cell plan and does not train. `python experiments/run_ablations.py` prints 125 not-yet-run cells. `--smoke` checks that every cell builds a `CapacityGateConfig` and that the three probe sources can be sampled on synthetic data.

`--pair none,fixed` on `aggregate_summaries.py` adds a left-minus-right comparison for cells that have one numeric score on each controller. It also reports Cohen's dz for those pairs and states that the p-value is unadjusted. It does not fill missing cells, and the statistic is not a manuscript result:

```bash
python experiments/aggregate_summaries.py --indir results/ablation_pilot --outdir results/aggregate_check --pair none,fixed --replicates 200
```

Rebuilding the paper figures from archived logs, when the supplementary results tree is unpacked:

```bash
python experiments/reproduce_all_tables_and_figures.py
```

That script reads saved logs. It does not rerun MuJoCo. It does not draw the stress bars from the numbers printed in the paper. A missing CapacityGate stress log stays `logs absent`. To plot means from whatever `summary.json` files you have, without treating the picture as a manuscript figure:

```bash
python -c "import json,sys; from adaptive_plasticity.figures_from_logs import write_score_figure; print(write_score_figure([json.load(open(p)) for p in sys.argv[1:]], 'results/figures_from_logs'))" results/ablation_pilot/*/summary.json
```

## Datasets

Training needs the raw D4RL files. They are gitignored.

```bash
python -m adaptive_plasticity.datasets --acquire hopper-medium-v2
python -m adaptive_plasticity.datasets --acquire halfcheetah-medium-v2
python -m adaptive_plasticity.datasets --acquire walker2d-medium-v2
python -m adaptive_plasticity.datasets --acquire halfcheetah-medium-replay-v2
python -m adaptive_plasticity.datasets --acquire hopper-medium-replay-v2
python -m adaptive_plasticity.datasets --acquire walker2d-medium-replay-v2
```

The three `medium-v2` files are what the published runners load. The three `medium-replay-v2` files are in the registry and are not part of the 135-cell plan. `antmaze-umaze-v2` is not in the registry, so `--acquire` cannot download it.

Acquisition downloads a Wayback snapshot of the official D4RL file and rewrites `data/manifests/<id>.json` even when the file was already on disk. Raw HDF5 files are gitignored. If the sha256 is unchanged, restore the tracked manifest before committing. To compare a local file with the committed sha256 without writing the manifest:

```bash
python -m adaptive_plasticity.datasets --check-hash hopper-medium-v2
```

Exit 0 means the bytes match. Exit 1 means they differ. Exit 2 means the raw file or the manifest sha256 is absent. The command does not download.

Restore an unchanged manifest after `--acquire` with:

```bash
git checkout -- data/manifests/hopper-medium-v2.json
git checkout -- data/manifests/halfcheetah-medium-v2.json
git checkout -- data/manifests/walker2d-medium-v2.json
```

If that host is unreachable, training cannot start. MuJoCo environments are `HalfCheetah-v5`, `Hopper-v5`, and `Walker2d-v5` via Gymnasium.

Confirm the launchers and the raw files without training:

```bash
python experiments/dry_run_plans.py
```

A missing primary file exits 2. A missing medium-replay file is printed and does not fail that command, because the published runners do not load it. `--require-registry` also exits 2 when a medium-replay file is absent. `--plans-only` skips the file check, which is what the unit test uses when the HDF5 files are not checked out.

The three medium-replay files were acquired with the commands above. Each download's byte count and sha256 matched the committed manifest (`local_bytes` and `sha256`). The acquire step rewrote those manifests, and they were restored with `git checkout` so the validated fields stayed in git. The HDF5 files stayed gitignored.

`configs/placeholders/halfcheetah_medium_replay.yaml` names an extra locomotion file that is in the registry and not in the primary plan. `configs/placeholders/antmaze_umaze.yaml` names AntMaze, which is not registered and has no environment adapter. Neither is runnable:

```bash
python -c "from adaptive_plasticity.placeholders import refuse_placeholder; refuse_placeholder('antmaze_umaze')"
```

## Manuscript training commands (long; not launched here)

These are the published protocols. They are full MuJoCo runs (25,000 offline updates and 25,000 online steps per cell, except where the harness says otherwise). Do not start them unless you intend to wait for the full matrix.

Primary 135-cell matrix plus 15 reference cells:

```bash
python experiments/final_factorial.py --execute --confirm-matrix CONFIRM-FINAL-150 --device cpu --workers 1
```

24-cell stress harness (seeds 0, 1, 2 for none, fixed, redo, and capacity_gate). The paper still reports CapacityGate for seed 0 only, even though this plan includes seeds 1 and 2:

```bash
python experiments/stress_factorial.py --execute --confirm-matrix CONFIRM-STRESS-24 --device cpu --workers 1 --outdir results/stress_study
```

Stable 9-cell validation bundled with the same harness:

```bash
python experiments/stress_factorial.py --stable-validate --execute --confirm-matrix CONFIRM-STABLE-9 --device cpu --workers 1
```

## New ablations (not yet run)

Full grids, 125 cells, manuscript budgets, hopper for the one-factor sweeps and all three environments for the probe-source sweep:

```bash
python experiments/run_ablations.py --execute --confirm-matrix CONFIRM-ABLATION-FULL --device cpu --outdir results/ablations
```

Short pilot of the new paths only (7 cells, hopper-medium-v2, seed 0, 2,000 online steps, 50 offline warmup updates, 1 evaluation episode). This is a launch check. It is not a result. A 2,000-step run with 50 warmup updates is not the manuscript protocol, so intervention counts from it are not evidence about the 25,000-step gate.

```bash
export MUJOCO_GL=glfw
python experiments/run_ablations.py --pilot --execute --confirm-matrix CONFIRM-ABLATION-PILOT --device cpu --outdir results/ablation_pilot
```

That command was executed once in this environment after `hopper-medium-v2` was acquired. All 7 cells exited 0. Outputs stayed in gitignored `results/ablation_pilot/`. Do not copy those normalized returns into the paper.

The four-cell probe pilot (offline fixed, online buffer, recent window 1024, recent window 256; hopper-medium-v2, seed 0, same short budget) is separate from the seven-cell mixed pilot:

```bash
python experiments/run_ablations.py --probe-pilot --execute --confirm-matrix CONFIRM-ABLATION-PILOT --device cpu --outdir results/probe_pilot
```

The seven pilot cells are:

| Study | Cell | What changes |
| --- | --- | --- |
| probe_distributions | offline_fixed | monitor the frozen offline batch |
| probe_distributions | recent_window_1024 | last 1,024 online transitions |
| probe_distributions | recent_window_256 | last 256 online transitions |
| gate_thresholds | rho_on_0.95 | rank trigger 0.95, off-threshold 0.97 |
| gate_thresholds | dorm_on_0.05 | dormancy trigger 0.05 |
| cooldown | cooldown_0 | no refractory period |
| gate_operator | redo | approximate, low-disruption ReDo under the gate |

One study at a time:

```bash
python experiments/run_ablations.py --study probe_distributions
python experiments/run_ablations.py --study gate_thresholds
python experiments/run_ablations.py --study cooldown
python experiments/run_ablations.py --study gate_operator
```

Published reference settings inside those configs: `probe_source=online_buffer`, `probe_size=256`, `rho_on=0.70`, `rho_off=0.85`, `dorm_on=0.15`, `dorm_off=0.08`, `cooldown_steps=3000`, `gate_operator=shrink_perturb`. The rank anchor is always the frozen offline probe. Only the monitoring batch, the thresholds, the cooldown, or the operator change.

## Stress seeds the paper does not report

Print coverage, including seeds 3 and 4 and CapacityGate seeds 1 and 2:

```bash
python experiments/analyze_stress_coverage.py --seeds 0,1,2,3,4 --indir results/stress_study
```

Launch only the extra seeds (not run here). Output names stay distinct and the manifest prefix is `STRESS_EXT`:

```bash
python experiments/stress_factorial.py --seeds 3,4 --execute --confirm-matrix CONFIRM-STRESS-EXT --device cpu --workers 1 --outdir results/stress_study_extension
```

If local summaries ever disagree with the rounded manuscript values, the coverage script says so. It does not edit the paper.

## External offline-to-online methods

| Key | Method | Code | What is missing |
| --- | --- | --- | --- |
| `cal_ql` | Cal-QL (Nakamoto et al., NeurIPS 2023) | https://github.com/nakamotoo/Cal-QL | JAX/Flax. Upstream scripts are AntMaze and Adroit, not this locomotion suite. A wrapper must expose policy and critic probes and apply the gate without dropping the calibration term. |
| `pex` | PEX (Zhang, Xu, and Yu, ICLR 2023) | https://github.com/Haichao-Zhang/PEX | PyTorch. Intervene on the learnable online policy only; leave the frozen offline policy alone. |
| `ro2o` | RO2O (Wen et al., JAIR 2024) | https://github.com/BattleWen/RO2O | PyTorch, with MuJoCo fine-tuning scripts. Hook `ro2o_ft_mujoco.py` without removing the uncertainty and smoothness penalties. |

No scores from these methods are in the manuscript or in this repository.

## Paper source

```bash
# syntax-level check used during submission preparation
python scratch/validate_latex.py
```

Do not run `scratch/generate_final_manuscript.py` or the `scratch/apply_*.py` editors. They are disabled because they would overwrite the manuscript or reintroduce the old claim that ReDo is an exact zero-functional-shift identity.
