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

### Not run

- Full MuJoCo matrices (135+15 confirmatory runs and the 24-cell stress study).
- Cal-QL, PEX, and RO2O.
- Stress seeds the manuscript does not report.
