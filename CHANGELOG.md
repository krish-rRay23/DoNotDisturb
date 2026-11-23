# Changelog

## Unreleased

Aligned the public text with the manuscript in `research_paper/iclr2027/iclr2027_conference.tex` and added launch paths for experiments that have not been run.

### Merged pull requests #14–#19

`requirements.txt` already pinned every direct dependency with `==` before these pull requests. `REPRODUCIBILITY.md` already tells a headless machine to export `MUJOCO_GL=glfw` (`egl` did not import here). Neither item needed a new pin or a new environment note.

- #14. Public claim check. The README, the manuscript, and `preview.html` are checked against transcribed tokens. No result is recomputed.
- #15. Figures from logs. Score bars are drawn only from supplied summaries. A missing score writes no figure. M5 summaries are grouped by `controller_type`.
- #16. Paired effect size. Cohen's dz is the mean paired difference divided by its sample standard deviation, and the p-value is labeled unadjusted. It is not a manuscript test.
- #17. `make check`. One entry point runs the unit tests, the plan dry-run, a read-only manifest hash check, and the public-claim check. A present file with no sha256 fails. An absent file does not.
- #18. LaTeX tables from supplied summaries. Means, sample standard deviations, and IQMs come from those files. M5 `shift_type`, including the no-shift case, is kept. A fixture is labeled as a fixture.
- #19. Limitations and wording. The limitations section states scope, and missing citations are marked `TODO: cite`. No manuscript number was added or replaced.

### Source of numbers

- No revised manuscript, response to reviewers, or `README_UPDATED.md` is in the repository. Reported figures stay the manuscript figures. They were not recomputed and no new scores were invented.
- ReDo is described as an approximate, low-disruption reset in the README, the manuscript (abstract, research questions, contributions, method, stress discussion, conclusion), and `preview.html`. The implementation is unchanged: a relative activation threshold, incoming weights reinitialized, outgoing weights zeroed. That map is an identity only for units that were already silent.
- The supplementary zip README's IQM intervals ([37.43, 40.06] and [9.31, 15.23]) are not the manuscript intervals ([37.43, 40.03] and [9.36, 15.10]). `REPRODUCIBILITY.md` records which archived JSON matches the paper.
- `docs/final_study_v2_scientific_report.md` and `results/stress_analysis/stress_iqm.json` inside the supplementary zip are not the manuscript tables.

### Experiments

- Probe sources `offline_fixed`, `online_buffer`, and `recent_window` in `src/adaptive_plasticity/probes.py`. The published runner still uses `online_buffer` and still anchors rank on the frozen offline probe.
- Ablation configs and `experiments/run_ablations.py`. 125 full cells, all `not_yet_run`. A 7-cell pilot uses 2,000 online steps.
- `CapacityGate` accepts `gate_operator` `shrink_perturb` (published) or `redo`.
- Stress harness accepts `--seeds`. Anything other than `0,1,2` requires `CONFIRM-STRESS-EXT` and is not a manuscript result. `experiments/analyze_stress_coverage.py` lists missing cells.
- `src/adaptive_plasticity/o2o_adapters.py` documents Cal-QL, PEX, and RO2O and refuses to import them. The IQL adapter is local. Those codebases are not vendored.

### Reproducibility

- Added pinned `requirements.txt` and `REPRODUCIBILITY.md`.
- Disabled `scratch/generate_final_manuscript.py` and the `scratch/apply_*.py` editors so they cannot overwrite the manuscript with the old ReDo wording.

### Launch check, not a result

A 7-cell hopper pilot (2,000 online steps, 50 warmup updates, seed 0, CPU, `MUJOCO_GL=glfw`) finished. Logs are under `results/ablation_pilot/`, which is gitignored. Intervention counts from that short, barely trained run are launch telemetry only. They are not manuscript findings and are not comparable to the 25,000-step tables.

### Review fixes

- `build_plan(..., shifts=("actuator_cripple",))` uses `default_severity`, so the stress shift keeps severity 1.0. `REGIME_SEVERITY` stays the three confirmatory regimes.
- Stress coverage matches summaries on environment, controller, seed, and `shift == actuator_cripple`. A same-seed confirmatory run is not compared with the actuator table.

### Paper wording

- The abstract and contribution (2) now use the rank interval already stated in the Diagnostic Abstention section, \(\rho(t) \in [0.99, 1.03]\), and the abstract's Walker2d stress sentence uses the mean \(\pm\) standard deviation already in Table `tab:stress_results` (\(8.03 \pm 1.57\), \(-0.53 \pm 0.33\)). No new number was introduced. The unconfirmed interval [0.9915, 1.6501] and a ReDo IQM of 40.37 were not applied. The HTML preview matches those two sentences. `d < 0.08` in the preview abstract was the tex's \(d \le 0.085\).
- The limitations section now states the locomotion scope, that \textsc{CapacityGate} does not explore, that hysteresis is relative to the existing \(7\text{--}12\%\) active-network band, and that \textsc{CapacityGate} stress is seed 0 only. ReDo is not described as an arm of the 135-cell matrix. No number was added or replaced.
- ReDo is named as an approximate, low-disruption reset in the operator subsection, the stress table, the stress caption, and the literature matrix. Bare "CapacityGate" in that caption is \textsc{CapacityGate}.
- Adam, Kaiming initialization, Polyak averaging, reservoir sampling, and Gym are marked `TODO: cite`. No bibliography entry was added for them. `agarwal2021rliable` is commented as a duplicate of the cited `agarwal2021deep` key.
- Figure captions use $\rho(t)$ and $d(t)$ for the telemetry trajectories, and the stress-figure caption says "standard deviation" and names ReDo as an approximate, low-disruption reset. The inequalities $0.70$ and $0.15$ and the printed returns are unchanged.

### Reproducibility entry point

- `make check` runs the unit tests, the plan dry-run, a read-only manifest hash check, and the public-claim token check. It does not train or download. `MUJOCO_GL` defaults to `glfw` because `egl` did not import in this environment. An absent raw file does not fail the hash check. A present file fails it when the sha256 differs or when the manifest has no sha256 to compare. Every direct dependency in `requirements.txt` was already pinned with `==`; a unit test rejects an unpinned line. The unverified revision figures were not applied.

### Continuous integration

- `.github/workflows/tests.yml` runs `make check` on pull requests and on `main`. `MUJOCO_GL=glfw` under Xvfb. Tests marked `needs_offline_data` are skipped so CI does not download datasets or run the short training cells that need them.

### Public claim check

- `experiments/check_public_claims.py` checks that the README, the manuscript, and the HTML preview still contain the tokens transcribed in `docs/manuscript_claims.yaml`. It does not recompute them. The unverified revision figures `0.9915`, `1.6501`, and `40.37` are forbidden on those three surfaces. The HTML preview now also states the paired-\(t\) statistic and the bootstrap interval on the mean paired difference, both already in the manuscript.

### LaTeX tables from logs

- `experiments/render_latex_tables.py` writes a mean table and a seed table from supplied `summary.json` files. The numbers are the mean and the sample standard deviation of those scores, plus an IQM from `iqm_by_controller` on the same rows. No scored row means no numeric table. A one-score cell does not print a standard deviation. Duplicate identities are not averaged twice. M5 summaries are grouped by `controller_type` when `controller` is absent, and by `shift_type` when `shift` is absent. A missing `shift_type` is the M5 no-shift case and is labeled `-noshift`, the token already used in `m5.py`. It is not dropped and it is not filled from the manuscript. Output from a fixture is labeled `SYNTHETIC FIXTURE` and is not a manuscript table.

### Figures from logs

- `src/adaptive_plasticity/figures_from_logs.py` writes a bar chart of means from supplied summaries and a `plotted.json` that lists those means. No scores means no file. The output is labeled as not a manuscript figure. Factorial summaries are grouped by `controller`. M5 summaries, which store `controller_type` instead, are grouped by that field.
- `experiments/reproduce_all_tables_and_figures.py` no longer creates an output directory on import, no longer paints the stress bars from manuscript constants, and no longer fills a missing CapacityGate stress cell with 8.58 or 7.23. An empty IQM is NaN rather than 0. Learning-curve and stress figures are skipped when their logs are absent.

### Dataset hash check

- `python -m adaptive_plasticity.datasets --check-hash <id>` compares a local raw file with the sha256 in `data/manifests/<id>.json`. It does not download and it does not rewrite the manifest. Exit 1 is a mismatch. Exit 2 means the file or the recorded sha256 is absent.

### Dataset dry-run

- `experiments/dry_run_plans.py` launches the primary, stress, stress-extension, stable, and ablation planners without `--execute`, and reports whether the three primary HDF5 files are present. It does not download or commit them.
- `REPRODUCIBILITY.md` notes that `--acquire` rewrites the tracked manifest and should be restored when the sha256 is unchanged.
- The same dry-run also reports the three registered medium-replay files. They are not required by the published runners, so a missing replay file does not fail the command. `--require-registry` does. A missing-file message prints one `--acquire` command per id, because the dataset CLI accepts a single id. `antmaze-umaze-v2` is not in the registry. The replay files were acquired; their sha256 values matched the committed manifests, the manifests were restored, and the HDF5 files were not committed.

### Probe pilot

- `--probe-pilot` selects four hopper seed-0 cells, one per probe-distribution row, at the 2,000-step budget. The mixed seven-cell pilot is unchanged. Both stay `not_yet_run` until executed. Unit tests cover an empty recent window, replacement on a short window, and an unknown source.
- The threshold, cooldown, and operator grids stay one-factor: 35 threshold cells, 20 cooldown cells, and 10 operator cells, all `not_yet_run`. A unit test locks the varied field and every other published setting, including the probe.

### Summary aggregation

- `experiments/aggregate_summaries.py` writes a per-cell table and a per-controller IQM with a stratified bootstrap interval from whatever `summary.json` files it is given. Empty inputs produce no IQM. The output is labeled as not a manuscript result. The IQM definition matches `experiments/final_analysis.py`.
- `--pair left,right` adds a left-minus-right comparison on cells that have exactly one numeric score for each controller. `environment` is read as `dataset_id`, and a shift stored only on `protocol` is read as `shift`. Rows that are still missing an identity field are listed separately and are not paired with each other. Duplicate rows and missing partners are listed and are not filled in. The Wilcoxon statistic is computed from those supplied pairs only. Cohen's dz is the mean paired difference divided by its sample standard deviation, and it is omitted when that deviation is undefined. The p-value is labeled unadjusted and is not a manuscript test.

### External learner hooks

- `src/adaptive_plasticity/o2o_hooks/` records gate thresholds and editable versus frozen modules for Cal-QL, PEX, and RO2O. The files ship with the package, so `load_external_hooks()` does not look two directories above the installed module. Each file stays `not_yet_run`. `open_external_session` still raises. A mock learner test checks that an intervention edits only the modules named as editable.

### Placeholders

- `configs/placeholders/` names `halfcheetah-medium-replay-v2` (in the dataset registry, not in the 135-cell plan) and `antmaze-umaze-v2` (not registered and not wired). Both are `not_yet_run`. `refuse_placeholder` raises. They are not part of the ablation plan.

### Plan sizes

- Unit tests lock the dry-run matrices at 150 confirmatory runs (135 primary plus 15 reference) and 24 stress cells (2 environments, 4 controllers, seeds 0–2). A seed extension is a different length and is not that 24-cell plan. No training was launched.

### Claim checklist

- `docs/claim_log_checklist.md` maps each manuscript claim to the script and the working-tree log that would carry it. A missing log is `unverified`. A file that is on disk is `present_not_recomputed` and is not opened. The supplementary zip is not unpacked. Short pilots stay `not_a_manuscript_result`. The older final summary and the pooled stress IQM stay `not_the_manuscript_table`. The learning-curve row points at `results/reproduced_figures/figure2_online_adaptation_learning_curves.png` and `plots/final/learning_curves.png`, the two files those scripts write. The stress-table row counts only `STRESS-*/summary.json`, so a `STABLE_VAL` summary in the same directory does not mark that table present.

### Claim coverage

- `make check` runs `experiments/check_claim_coverage.py`. Every decimal and every `10^{n}` power printed in the manuscript must appear under a claim id in `docs/claim_numeric_tokens.yaml`. A TeX dimension is not a claim. The check does not recompute a score. `protocol_settings` holds controller and optimizer settings, not scores.

### Quickstart

- The README quickstart runs `make check` (tests, plan dry-run, hash check, public-claim check, and claim-coverage) and points the four confirm-gated GPU runs at `docs/run_guide.md`. No manuscript number was added.

### Run guide

- `docs/run_guide.md` lists the four confirm-gated full runs (`CONFIRM-FINAL-150`, `CONFIRM-STRESS-24`, `CONFIRM-STABLE-9`, `CONFIRM-ABLATION-FULL`) and the commands that rebuild tables and figures from their outputs. No run was launched. No manuscript number was added.

### Fixture smoke

- `experiments/fixture_smoke.py` is the smallest confirm-gated path: one capacity-gate cell, 4 online steps, CPU, Hopper-v5. The offline batch is built in memory and is not a D4RL file. `--execute` requires `CONFIRM-FIXTURE-SMOKE`. The summary is labeled `SYNTHETIC FIXTURE` and is not a result. No manuscript number was added.

### Dead launchers

- Removed `experiments/run_overnight.py`, `experiments/benchmark_concurrency.py`, and `experiments/mechanistic_stress_test.py`. Nothing in the package or the tests imported them, and none of them required a confirm token. The documented launchers are unchanged. `activation_dormancy` no longer accepts the ignored `eps` argument; weight sparsity stays on `legacy_weight_sparsity`.

### Not run

- Full MuJoCo matrices (135+15 confirmatory runs and the 24-cell stress study).
- Cal-QL, PEX, and RO2O.
- Stress seeds the manuscript does not report.

### Revision numbers still pending

`docs/pending_revision_numbers.md` lists every sentence that would change if a revision reports rank in [0.9915, 1.6501], multi-seed CapacityGate stress, or a ReDo IQM of 40.37. Those figures are not in `main` or in `cursor/revision-consistency-5776`. They are not copied into the paper or the README.
