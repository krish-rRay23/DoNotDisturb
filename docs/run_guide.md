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
python experiments/stress_factorial.py --stable-validate --device cuda --outdir results/stress_study
python experiments/run_ablations.py --device cuda
```

The final plan is 135 primary cells plus 15 reference cells (150). The stress plan is 24 cells. The stable plan is 9 cells. The full ablation plan is 125 cells, and every planned cell is `not_yet_run`. The 125-cell grid is not a manuscript table.

## Execute

Each command trains only when `--confirm-matrix` matches that plan. Any other token refuses.

```bash
python experiments/final_factorial.py --execute --confirm-matrix CONFIRM-FINAL-150 --device cuda --outdir results/final_study
python experiments/stress_factorial.py --execute --confirm-matrix CONFIRM-STRESS-24 --device cuda --workers 6 --outdir results/stress_study
python experiments/stress_factorial.py --stable-validate --execute --confirm-matrix CONFIRM-STABLE-9 --device cuda --workers 6 --outdir results/stress_study
python experiments/run_ablations.py --execute --confirm-matrix CONFIRM-ABLATION-FULL --device cuda --outdir results/ablations
```

The final launcher defaults to 9 workers. The stress launcher defaults to 6. `--workers 1` runs one cell at a time. A cell that already has `summary.json` and `checkpoint.pt` is skipped.

`CONFIRM-STRESS-24` is seeds 0, 1, and 2. The manuscript stress table reports CapacityGate for seed 0 only. Seeds outside `0,1,2` need `CONFIRM-STRESS-EXT` and are not this plan. `STABLE_VAL-*` summaries share `results/stress_study` and are not the stress table. `CONFIRM-ABLATION-PILOT` and `CONFIRM-PILOT-9` are short launch checks, not these four runs.

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

`final_analysis.py` writes `results/final_study/FINAL_analysis_summary.json`, `FINAL_paper_tables.csv`, and plots under `plots/final/`, including `learning_curves.png`. It refuses while the 150-cell study is incomplete.

`reproduce_all_tables_and_figures.py` reads `results/final_study/FINAL_paper_tables.csv` and `results/stress_study/STRESS-*/summary.json`. It writes `results/reproduced_figures/`, including `figure2_online_adaptation_learning_curves.png`. It does not draw the stress bars from manuscript constants. Those bars come from `adaptive_plasticity.figures_from_logs.write_score_figure` on supplied summaries.

`render_latex_tables.py` prints a banner that the table was computed from supplied summaries and is not a manuscript table.

`write_claim_checklist.py` rewrites `docs/claim_log_checklist.md` from which of those paths exist. A file on disk is `present_not_recomputed`. The writer does not open the file.

`experiments/stress_analysis.py` writes `results/stress_analysis/stress_iqm.json`. That file pools CapacityGate seeds and is `not_the_manuscript_table` in the checklist. Do not use it for the stress means.
