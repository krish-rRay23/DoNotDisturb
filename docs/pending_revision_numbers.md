# Pending revision numbers

The three claims below are still not applied. Reproducibility checks must not copy them into the paper, the README, or `REPRODUCIBILITY.md`.

Searched on 2026-11-23, after pull request #1 merged into `main` (`17ecdc8`). The public surfaces at `main` `33bd694` still do not contain these figures.

## What was searched

- `main` at `17ecdc855b4345e0abbb4d9294a642a2f30fed31`
- `cursor/revision-consistency-5776` at `d232225f464251d319926c76f5ff0df90b6f362d` (the only other branch)
- Filenames and text for a revised manuscript, a response to reviewers, `README_UPDATED.md`, `0.9915`, `1.6501`, and a ReDo IQM of `40.37`

None of those files or figures are in the repository. GitHub code search on `krish-rRay23/DoNotDisturb` also returned no matches. The only manuscript source is `research_paper/iclr2027/iclr2027_conference.tex`. The HTML preview repeats that manuscript. `scratch/generate_final_manuscript.py` and `scratch/apply_*.py` exit immediately and are not a revision.

Krish described a revision that reports, among other things, relative rank in `[0.9915, 1.6501]`, multi-seed stress results, and a ReDo IQM of `40.37`. Those values are **not written into the paper, the README, or the reproducibility notes** until a source file is in the repo. This document lists the sentences that would have to change, and the value each one has now.

## Claim: relative rank in [0.9915, 1.6501]

The manuscript does not contain `0.9915` or `1.6501`. It contains two different rank intervals, plus a one-point trigger. A source file has to say which sentence the new interval replaces. Do not replace the stress telemetry interval with the stable-transfer interval, or the reverse, without that source.

| Location | What it says now |
| --- | --- |
| `research_paper/iclr2027/iclr2027_conference.tex` abstract | \(\rho \approx 1.0\), \(d \le 0.085\) |
| same file, contribution (2) | \(\rho \approx 1.0\), \(d \le 0.085\) |
| same file, Diagnostic Abstention (`sec:abstention`) | \(\rho(t) \in [0.99, 1.03]\) against \(\rho_{\mathrm{on}} = 0.70\); \(d(t) \in [4.3\%, 8.5\%]\) against \(d_{\mathrm{on}} = 0.15\) |
| same file, HalfCheetah stress telemetry (Seed 0, pre-shift steps 1,000–4,000) | \(d = 0.043\), \(\rho \in [0.85, 0.98]\); later \(\rho = 0.6948\) at step 7,000 |
| same file, method | trigger thresholds \(\rho_{\mathrm{on}} = 0.70\), \(\rho_{\mathrm{off}} = 0.85\). These are gates, not observed ranges. |
| `research_paper/iclr2027/preview.html` abstract | same \(\rho \approx 1.0\), \(d \le 0.085\) wording as the tex abstract |
| `README.md`, Result | \(\rho \in [0.99, 1.03]\), \(d \le 0.085\) |
| `REPRODUCIBILITY.md` | does not restate the rank interval; it points at the tex |
| `configs/ablations/*.yaml` | `rho_on: 0.70`, `rho_off: 0.85` as the published gate, not an observed range |

## Claim: multi-seed stress results

`none`, `fixed`, and `redo` already have seeds 0, 1, and 2 in Table `tab:stress_results`. CapacityGate does not. The table prints `---` for CapacityGate seeds 1 and 2. Extending CapacityGate to more seeds would replace every "seed 0 only" sentence below. The per-seed numbers that are already printed should stay until a source file replaces them.

| Location | What it says now |
| --- | --- |
| `iclr2027_conference.tex` Table `tab:stress_results` | HalfCheetah CapacityGate: seed 0 \(8.58\), seeds 1–2 blank, summary "8.58 (Single-Cell Validated)". Walker2d CapacityGate: seed 0 \(7.23\), seeds 1–2 blank, summary "7.23 (Baseline Parity)". |
| same file, Figure `fig:stress_operator_asymmetry` caption | CapacityGate is "single-cell deterministic validation (Seed 0)", Walker2d \(7.23\), HalfCheetah \(8.58\) vs that seed's baseline \(7.99\). |
| same file, stress prose and appendix `tab:telemetry_audit` | HalfCheetah actuator telemetry is Seed 0 only. |
| `preview.html` stress table | CapacityGate column is the seed-0 summary (HalfCheetah 8.58, Walker2d 7.23). |
| `README.md`, Actuator-crippling stress table | "CapacityGate (seed 0)": HalfCheetah 8.58, Walker2d 7.23. |
| `REPRODUCIBILITY.md` | "CapacityGate is seed 0 only (HalfCheetah 8.58, Walker2d 7.23)." |
| `src/adaptive_plasticity/stress_reports.py` `MANUSCRIPT_STRESS` | `capacity_gate` maps are `{0: 8.58}` and `{0: 7.23}` only. |

Seeds 3 and 4 are not in the manuscript for any stress arm. `experiments/stress_factorial.py --seeds 3,4` is an extension and requires `CONFIRM-STRESS-EXT`. It is not a result.

## Claim: ReDo IQM of 40.37

The manuscript has no ReDo aggregate IQM. ReDo is not an arm of the 135-cell primary matrix. Table 1's aggregate rows are:

| Arm | IQM | 95% CI |
| --- | ---: | --- |
| `none` | 38.78 | [37.43, 40.03] |
| `fixed` | 12.30 | [9.36, 15.10] |
| CapacityGate | 38.78 | [37.43, 40.03] |

A ReDo IQM of 40.37 would be a new row. It is not in the tex, the preview, `README.md`, or `REPRODUCIBILITY.md`. The ReDo numbers that do exist are stress-table means, not an IQM:

| Location | Current ReDo figure |
| --- | --- |
| Table `tab:stress_results`, HalfCheetah | seeds 11.40, 12.25, 17.29; mean \(13.64 \pm 3.18\) |
| Table `tab:stress_results`, Walker2d | seeds 8.45, 9.34, 6.29; mean \(8.03 \pm 1.57\) |
| abstract | "\(8.03\) normalized return on Walker2d" and "\(-0.53\)" for weight perturbation. The \(\pm\) bands are in the table and in contribution (3), not in this abstract sentence. |
| contribution (3) | Walker2d ReDo \(8.03 \pm 1.57\), Fixed \(-0.53 \pm 0.33\) |
| `README.md` stress table | HalfCheetah ReDo \(13.64 \pm 3.18\); Walker2d ReDo \(8.03 \pm 1.57\) |

Do not add 40.37 beside 38.78 or 12.30. If a later source defines a ReDo IQM, it also has to say the protocol: which environments, regimes, seeds, and whether the statistic is the same stratified bootstrap IQM as Table 1.

## What to push

Put the revised PDF or `.tex`, and the response to reviewers, in the repo (or point this file at their paths). After that, the substitutions are mechanical: each row above is replaced only by the value in that source, and this file can be deleted or marked applied.

## Tables rendered from logs

`experiments/render_latex_tables.py` computes cells from supplied `summary.json` files. It does not read this document and it does not read the manuscript. Do not paste `0.9915`, `1.6501`, or a ReDo IQM of `40.37` into a rendered table. A synthetic fixture used by the unit test is labeled `SYNTHETIC FIXTURE` and is not a result.
