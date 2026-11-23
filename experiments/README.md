# Experiments

Runners for the manuscript protocol and for ablations that have not been run.
Commands and the source of every reported number are in `REPRODUCIBILITY.md`.

| Script | What it does | Trains by default? |
| --- | --- | --- |
| `final_factorial.py` | 135-cell confirmatory plan plus the 15-cell reference plan | No. `--execute` needs `CONFIRM-FINAL-150`. |
| `stress_factorial.py` | 24-cell actuator-crippling harness (seeds 0–2). `--seeds` outside `0,1,2` is an extension | No. Published token `CONFIRM-STRESS-24`; extension token `CONFIRM-STRESS-EXT`. |
| `run_ablations.py` | Probe source, thresholds, cooldown, and gate operator | No. Pilot token `CONFIRM-ABLATION-PILOT`; full token `CONFIRM-ABLATION-FULL`. |
| `analyze_stress_coverage.py` | Compares local summaries to the manuscript stress table | No. |
| `reproduce_all_tables_and_figures.py` | Rebuilds tables and figures from saved logs | No training. Needs the archived run logs. |
| `fixture_smoke.py` | One capacity-gate cell, 4 CPU steps, synthetic offline batch | No. `--execute` needs `CONFIRM-FIXTURE-SMOKE`. The summary is a `SYNTHETIC FIXTURE`, not a result. |

`configs/ablations/` holds the grids. Every planned cell is `not_yet_run` until a launcher finishes it. Pilot budgets are a few thousand environment steps and are not manuscript results.

Cal-QL, PEX, and RO2O are not wrapped. See `src/adaptive_plasticity/o2o_adapters.py`.
