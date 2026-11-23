# Claim to log checklist

This page maps each manuscript claim to the script and the working-tree log that would carry it.
It does not recompute a score and it does not unpack `DoNotDisturb_ICLR2027_Supplementary.zip`.
A log that exists only inside that zip is `unverified` until it is on disk, and a file that is on disk is still not read.

`unverified` means the working-tree log is absent.
`present_not_recomputed` means the path exists and its contents were not opened.
`not_the_manuscript_table` means the path is the wrong archive even if it is later unpacked.
`not_a_manuscript_result` means the run is a launch check, not a paper table.

Figures named in `docs/pending_revision_numbers.md` are not rows here and are not applied.
`experiments/check_public_claims.py` only checks that transcribed tokens are still on the public pages.

| Claim | Manuscript location | Script | Working-tree log | Status |
| --- | --- | --- | --- | --- |
| `aggregate_iqm` | Table tab:main_results and the abstract. Transcribed tokens: 38.78, 12.30, [37.43, 40.03], [9.36, 15.10]. | `experiments/final_analysis.py` | `results/final_study/FINAL_analysis_summary.json` | `unverified` |
| `paired_comparison` | The aggregate statistical paragraph. Transcribed tokens: 26.03, 19.09, 18.0, 1.44, 9.15, 9.69, 20.72, 31.59, 22.27, 0.933. | `experiments/final_analysis.py` | `results/final_study/FINAL_analysis_summary.json` | `unverified` |
| `primary_cells` | Tables tab:main_results and tab:per_seed_full. 135 primary cells. | `experiments/final_factorial.py` | `results/final_study` | `unverified` |
| `stress_table` | Table tab:stress_results and Figure fig:stress_operator_asymmetry. CapacityGate in that table is seed 0 only. | `experiments/stress_factorial.py` | `results/stress_study/STRESS-*/summary.json` | `unverified` |
| `stress_telemetry` | Section sec:telemetry and Table tab:telemetry_audit. HalfCheetah stress, seed 0. | `experiments/stress_factorial.py` | `results/stress_study/STRESS-halfcheetah-medium-v2-capacity_gate-actuator_cripple-sev1-seed0/summary.json` | `unverified` |
| `stable_rank` | Section sec:abstention. Transcribed tokens: [0.99, 1.03], 0.085. | `experiments/final_factorial.py` | `results/final_study` | `unverified` |
| `learning_curves` | Figure fig:learning_curves. | `experiments/reproduce_all_tables_and_figures.py` | `results/reproduced_figures/figure2_online_adaptation_learning_curves.png`<br>`plots/final/learning_curves.png` | `unverified` |
| `older_final_summary` | Not a manuscript table. Do not use this file for Table tab:main_results. | `none` | `results/FINAL_analysis_summary.json` | `not_the_manuscript_table` |
| `pooled_stress_iqm` | Not Table tab:stress_results. Do not use this file for the stress means. | `none` | `results/stress_analysis/stress_iqm.json` | `not_the_manuscript_table` |
| `ablation_grid` | Not in the manuscript tables. The 125-cell grid is not_yet_run. | `experiments/run_ablations.py` | `results/ablations` | `unverified` |
| `short_pilots` | Not a manuscript claim. The 2,000-step hopper pilots are launch checks. | `experiments/run_ablations.py` | `results/ablation_pilot`<br>`results/probe_pilot` | `not_a_manuscript_result` |
| `protocol_settings` | Method sections and Table tab:hyperparams. Thresholds, probe settings, and optimizer settings, not a reported score. | `src/adaptive_plasticity/final_runner.py` | `src/adaptive_plasticity/capacity_gate.py` | `not_a_manuscript_result` |

## Notes

- `aggregate_iqm` (unverified): REPRODUCIBILITY.md names this path inside the supplementary zip as the summary that rounds to the manuscript. This checklist does not open it.
- `paired_comparison` (unverified): Same archived summary as the IQM row. The statistic is not recomputed here.
- `primary_cells` (unverified): A present directory is not a checked score. summary.json files are not read.
- `stress_table` (unverified): Only STRESS- run summaries count. A STABLE_VAL summary in the same directory is the nine-cell validation and is not this table. Summaries are not read.
- `stress_telemetry` (unverified): The gate log next to that summary is the telemetry trace. It is not read here.
- `stable_rank` (unverified): The interval is the manuscript sentence. Gate jsonl files are not parsed.
- `learning_curves` (unverified): reproduce_all_tables_and_figures.py writes the first path. final_analysis.py writes the second path under its default --plots-dir. Neither writer uses results/final_analysis/learning_curves.png. The files are not opened.
- `older_final_summary` (not_the_manuscript_table): REPRODUCIBILITY.md says the copy at results/FINAL_analysis_summary.json inside the supplementary zip does not match the paper.
- `pooled_stress_iqm` (not_the_manuscript_table): REPRODUCIBILITY.md says this archive pools CapacityGate seeds the paper does not report.
- `ablation_grid` (unverified): No full ablation log is a result in the paper.
- `short_pilots` (not_a_manuscript_result): Pilot scores are not comparable to the 25,000-step tables and are not read here.
- `protocol_settings` (not_a_manuscript_result): These decimals are the published controller and optimizer settings. They are not scores and are not read from a run log.
