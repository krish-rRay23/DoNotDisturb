# Remaining blockers

These items are not done. Nothing on this page is a result. No training run was launched to write it.

## Revised manuscript numbers

What is blocked. `docs/pending_revision_numbers.md` lists three figures that are still not applied: relative rank in [0.9915, 1.6501], multi-seed CapacityGate stress, and a ReDo IQM of 40.37. They are not in the manuscript, the README, or `preview.html`.

What unblocks it. A source file in this repository that states those figures and which sentence each one replaces. Until that file is present, do not copy them into the paper, the README, or `preview.html`. The public claim check rejects those three tokens on those surfaces.

## GPU runs

What is blocked. The four confirm-gated matrices in `docs/run_guide.md` have not been launched from this environment. There is no GPU here, and CI does not start them.

| Token | Output directory | Plan |
| --- | --- | --- |
| `CONFIRM-FINAL-150` | `results/final_study` | 135 primary cells plus 15 reference cells |
| `CONFIRM-STRESS-24` | `results/stress_study` | 24 cells, seeds 0–2 |
| `CONFIRM-STABLE-9` | `results/stable_validation` | 9 cells. Not the stress table |
| `CONFIRM-ABLATION-FULL` | `results/ablations` | 125 cells, `not_yet_run`, not a manuscript table |

What unblocks it.

- A machine where `--device cuda` is available. `experiments/final_factorial.py` refuses to start when CUDA was requested and is missing. The stress and ablation launchers default to CPU, so the guide passes `--device cuda` explicitly.
- The six registry datasets, acquired and sha256-matched with the commands in `REPRODUCIBILITY.md`. A missing primary file stops the final matrix. `export MUJOCO_GL=glfw` when EGL does not import. `MUJOCO_GL` is separate from `--device`.
- The execute commands in `docs/run_guide.md`. Any other confirm token refuses.
- After the cells finish, the regenerate commands in that guide. `final_analysis.py` refuses while the 135-cell primary matrix is incomplete. A missing reference cell does not trip that refusal and does not stamp `PARTIAL-DATA`. `STABLE_VAL-*` stays in `results/stable_validation`, because `render_latex_tables.py` reads every `summary.json` under the directory it is given.
- A checklist row stays `unverified` until the log named in `docs/claim_log_checklist.md` is on disk. A new score is not a manuscript number until the claim map lists it. This page does not add one.

Seeds outside `0,1,2` need `CONFIRM-STRESS-EXT` and are not the 24-cell plan. The manuscript stress table reports CapacityGate for seed 0 only.

## Cal-QL, PEX, and RO2O

What is blocked. Those codebases are not vendored. `open_external_session` in `src/adaptive_plasticity/o2o_adapters.py` raises `ExternalLearnerUnavailable`. The hook files under `src/adaptive_plasticity/o2o_hooks/` stay `not_yet_run`. No score from those learners exists here.

What unblocks it. A wrapper in this repository that implements the methods on `IQLLearnerAdapter`, using the upstream code below. Cloning the repository is not enough: the session still raises until that wrapper exists.

- Cal-QL. [nakamotoo/Cal-QL](https://github.com/nakamotoo/Cal-QL) (JAX/Flax). Load an offline checkpoint, expose observation batches the gate can probe, apply shrink-and-perturb or ReDo to the parameters the online optimizer steps, and leave the calibration term unchanged. That repository's scripts cover AntMaze and Adroit. It does not ship a locomotion D4RL fine-tuning entry point, so a locomotion comparison also needs that port.
- PEX. [Haichao-Zhang/PEX](https://github.com/Haichao-Zhang/PEX). Train or load the offline policy and expand it online. Probe the learnable online policy. Intervene only on that policy and its critic. Leave the frozen offline policy untouched.
- RO2O. [BattleWen/RO2O](https://github.com/BattleWen/RO2O). Load a checkpoint from `algorithms/offline/ro2o_mujoco.py` and hook the online update in `algorithms/finetune/ro2o_ft_mujoco.py`. Apply a gated intervention without removing the uncertainty and smoothness penalties.

## AntMaze adapter

What is blocked. `configs/placeholders/antmaze_umaze.yaml` names `antmaze-umaze-v2`. That id is not in the dataset registry, `environment` is null, `executable` is false, and `refuse_placeholder` raises. The IQL agent in this repository is the continuous-control locomotion agent. Nothing has been run.

What unblocks it. Register the dataset, add a goal-reaching environment adapter, and decide whether the same gate applies to that action space. Until those exist the placeholder stays `not_yet_run` and must not be cited.

`halfcheetah-medium-replay-v2` is a different placeholder. It is in the dataset registry and uses HalfCheetah-v5, but it is not in the 135-cell plan and it has no score.
