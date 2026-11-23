# Changelog

## Unreleased

Aligned the public text with the manuscript in `research_paper/iclr2027/iclr2027_conference.tex` and added launch paths for experiments that have not been run.

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

### Continuous integration

- `.github/workflows/tests.yml` runs pytest on pull requests and on `main`. `MUJOCO_GL=glfw` under Xvfb. Tests marked `needs_offline_data` are skipped so CI does not download datasets or run the short training cells that need them.

### Public claim check

- `experiments/check_public_claims.py` checks that the README, the manuscript, and the HTML preview still contain the tokens transcribed in `docs/manuscript_claims.yaml`. It does not recompute them. The unverified revision figures `0.9915`, `1.6501`, and `40.37` are forbidden on those three surfaces. The HTML preview now also states the paired-\(t\) statistic and the bootstrap interval on the mean paired difference, both already in the manuscript.

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

### Not run

- Full MuJoCo matrices (135+15 confirmatory runs and the 24-cell stress study).
- Cal-QL, PEX, and RO2O.
- Stress seeds the manuscript does not report.

### Revision numbers still pending

`docs/pending_revision_numbers.md` lists every sentence that would change if a revision reports rank in [0.9915, 1.6501], multi-seed CapacityGate stress, or a ReDo IQM of 40.37. Those figures are not in `main` or in `cursor/revision-consistency-5776`. They are not copied into the paper or the README.
