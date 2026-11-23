# Running the four confirm-gated matrices

This page is for a machine with a GPU and the raw D4RL files. It does not start those runs. CI does not start them. Printed scores from a finished run are not manuscript numbers until the log named in `claim_log_checklist.md` is the file that produced them, and this page does not copy a score into the paper.

Acquire the six registry datasets and check their sha256 values with the commands in `REPRODUCIBILITY.md` before training. A missing primary file stops the final matrix.

Headless MuJoCo in the environment that prepared this repository needed GLFW. Export it before any command that imports MuJoCo when EGL does not import:

```bash
export MUJOCO_GL=glfw
```

`--device cuda` is the GPU setting. It is separate from `MUJOCO_GL`. `experiments/final_factorial.py` defaults to `cuda` and refuses to start when that device is unavailable. The stress and ablation launchers default to `cpu`, so the commands below pass `--device cuda`.

## Dry run

These commands print the plans and do not train.

```bash
python experiments/final_factorial.py --device cuda --outdir results/final_study
python experiments/stress_factorial.py --device cuda --outdir results/stress_study
python experiments/stress_factorial.py --stable-validate --device cuda --outdir results/stable_validation
python experiments/run_ablations.py --device cuda
```

The final plan is 135 primary cells plus 15 reference cells (150). The stress plan is 24 cells. The stable plan is 9 cells. The full ablation plan is 125 cells, and every planned cell is `not_yet_run`. The 125-cell grid is not a manuscript table.

## Needs-data plans

`docs/reviewer_risk_review.md` lists items that need a run or a source file. Each one has a file under `configs/pending/`. Every file is `not_yet_run` and `executable: false`. `python experiments/dry_run_plans.py --plans-only` prints them and does not train. Nothing in this section is a result.

| Reviewer item | Plan | What it is |
| --- | --- | --- |
| More confirmatory seeds | `configs/pending/more_confirmatory_seeds.yaml` | Seeds 5–7 on the three medium-v2 tasks, three confirmatory shifts, and three primary controllers. Not `CONFIRM-FINAL-150`. |
| Multi-seed CapacityGate stress | `configs/pending/multiseed_capacitygate_stress.yaml` | Points at `CONFIRM-STRESS-24`, which already includes CapacityGate seeds 0, 1, and 2. The manuscript reports seed 0 only. Seeds 1 and 2 are not manuscript numbers. |
| Unapplied revision figures | `configs/pending/unapplied_revision_figures.yaml` | No cells and no command. The figures stay in `docs/pending_revision_numbers.md`. |
| Hopper actuator crippling | `configs/pending/hopper_actuator_stress.yaml` | `hopper-medium-v2`, `actuator_cripple`, four controllers, seeds 0–2. Hopper is not in `CONFIRM-STRESS-24`. |
| ReDo and periodic resets on the confirmatory grid | `configs/pending/primary_reset_arms.yaml` | Controllers `redo` and `redo_periodic` on the confirmatory environments, shifts, and seeds 0–4. Not the 135-cell plan. |
| Medium-replay, expert, manipulation, vision, AntMaze | `configs/pending/broader_domains.yaml` | Placeholder files in `configs/placeholders/`. Medium-replay ids are in the dataset registry. Expert, manipulation, vision, and AntMaze are not. |
| Cal-QL, PEX, and RO2O | `configs/pending/external_learners.yaml` | Hook files `src/adaptive_plasticity/o2o_hooks/cal_ql.yaml`, `pex.yaml`, and `ro2o.yaml`. `open_external_session` still raises. |
| Gate-threshold sweep | `configs/pending/gate_threshold_sweep.yaml` | `configs/ablations/gate_thresholds.yaml`, already inside `CONFIRM-ABLATION-FULL`. Every cell is `not_yet_run`. |
| Per-environment signed-rank test | `configs/pending/per_environment_signed_rank.yaml` | Analysis only. It does not emit a p-value. |
| Primary-cell gate firing | `configs/pending/primary_gate_firing.yaml` | Analysis only. It does not read a log and does not add an intervention count. |

The medium-replay placeholders are `configs/placeholders/halfcheetah_medium_replay.yaml`, `hopper_medium_replay.yaml`, and `walker2d_medium_replay.yaml`. The expert placeholders are `halfcheetah_expert.yaml`, `hopper_expert.yaml`, and `walker2d_expert.yaml`. Manipulation and vision are `manipulation.yaml` and `vision.yaml`, with no dataset id. AntMaze stays `antmaze_umaze.yaml`.

## Execute

Each command trains only when `--confirm-matrix` matches that plan. Any other token refuses.

```bash
python experiments/final_factorial.py --execute --confirm-matrix CONFIRM-FINAL-150 --device cuda --outdir results/final_study
python experiments/stress_factorial.py --execute --confirm-matrix CONFIRM-STRESS-24 --device cuda --workers 6 --outdir results/stress_study
python experiments/stress_factorial.py --stable-validate --execute --confirm-matrix CONFIRM-STABLE-9 --device cuda --workers 6 --outdir results/stable_validation
python experiments/run_ablations.py --execute --confirm-matrix CONFIRM-ABLATION-FULL --device cuda --outdir results/ablations
```

The final launcher defaults to 9 workers. The stress launcher defaults to 6. `--workers 1` runs one cell at a time. A cell that already has `summary.json` and `checkpoint.pt` is skipped.

`CONFIRM-STRESS-24` is seeds 0, 1, and 2. The manuscript stress table reports CapacityGate for seed 0 only. Seeds outside `0,1,2` need `CONFIRM-STRESS-EXT` and are not this plan. `STABLE_VAL-*` summaries go to `results/stable_validation` and are not the stress table. `CONFIRM-ABLATION-PILOT` and `CONFIRM-PILOT-9` are short launch checks, not these four runs.

## Regenerate tables and figures from the outputs

Run these after the cells finish. They read logs. They do not train, and they do not edit the manuscript.

```bash
python experiments/final_analysis.py --results-dir results/final_study
python experiments/reproduce_all_tables_and_figures.py
python experiments/render_latex_tables.py --indir results/final_study --outdir results/latex_tables
python experiments/render_latex_tables.py --indir results/stress_study --outdir results/latex_tables_stress
python experiments/analyze_stress_coverage.py --indir results/stress_study --seeds 0,1,2
python experiments/write_claim_checklist.py
```

`final_analysis.py` writes `results/final_study/FINAL_analysis_summary.json`, `FINAL_paper_tables.csv`, and plots under `plots/final/`, including `learning_curves.png`. It refuses when the 135-cell primary matrix is incomplete. A missing reference cell does not trip that refusal and does not stamp PARTIAL-DATA; the primary outputs are still written. `--audit-only` lists which of the 150 cells are missing. `--allow-partial` is the switch that stamps PARTIAL-DATA.

`reproduce_all_tables_and_figures.py` reads `results/final_study/FINAL_paper_tables.csv` and `results/stress_study/STRESS-*/summary.json`. It writes `results/reproduced_figures/`, including `figure2_online_adaptation_learning_curves.png`. It does not draw the stress bars from manuscript constants. Those bars come from `adaptive_plasticity.figures_from_logs.write_score_figure` on supplied summaries.

`render_latex_tables.py` reads every `summary.json` under `--indir` and prints a banner that the table was computed from supplied summaries and is not a manuscript table. The stress command above stays on `results/stress_study` because that directory has the `STRESS-*` runs. `STABLE_VAL-*` stays in `results/stable_validation`, so those nine cells are not folded into that table.

`write_claim_checklist.py` rewrites `docs/claim_log_checklist.md` from which of those paths exist. A file on disk is `present_not_recomputed`. The writer does not open the file.

`experiments/stress_analysis.py` writes `results/stress_analysis/stress_iqm.json`. That file pools CapacityGate seeds and is `not_the_manuscript_table` in the checklist. Do not use it for the stress means.
