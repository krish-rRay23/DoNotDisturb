# Reviewer risk review

This is a read of `research_paper/iclr2027/iclr2027_conference.tex` as a skeptical ICLR reviewer. It does not add a score, recompute a test, or apply the unconfirmed figures in `docs/pending_revision_numbers.md`. Those figures stay blocked in `docs/blockers.md`.

Each item is tagged `addressable-now` or `needs-data`. The manuscript edits that accompany this note are only the `addressable-now` items. They change scope sentences. They do not change a printed return.

## addressable-now

- tag: addressable-now
  **The setup paragraph used five seeds for every comparison, then mentioned the 24 stress runs in the same sentence.** The confirmatory matrix is seeds 0–4. The stress protocol is seeds 0–2, and CapacityGate in that table is seed 0 only. The setup now says so, in the same place as the limitations section already did.

- tag: addressable-now
  **“Three tasks across four regimes” reads as Hopper actuator crippling.** Hopper is in the confirmatory matrix. It is not in the stress table. The benchmarks paragraph now says actuator crippling is a separate protocol and that Hopper is not in it.

- tag: addressable-now
  **The opening claim was broader than the operator that was measured.** The first paragraph named resets, Shrink-and-Perturb, and dormant-neuron recycling, then said the unconditional-plasticity assumption fails in offline-to-online control. The measured collapse is periodic Shrink-and-Perturb on the 135-run matrix. The abstract, the HTML preview, and the introduction now say that. Resets and recycling are not claimed as arms of that matrix.

- tag: addressable-now
  **Figure 1 attributed \(P = 0.933\) to Baseline and CapacityGate together.** The reported probability is \(P(\text{None} > \text{Fixed}) = 0.933\). CapacityGate matches Baseline on the primary cells, so that figure is not a separate comparison. The caption and the preview say so. The number is unchanged.

- tag: addressable-now
  **The appendix was titled “Exhaustive.”** The table is the 135 primary cells, seeds 0–4. It is not the stress table and it is not the 15 reference runs. The section is now “Seed-Level Primary Benchmark Data.”

- tag: addressable-now
  **The signed-rank test was easy to read as 45 independent replicates of one task.** \(N = 45\) pools environment-regime pairs. The setup now says the test pools those pairs and does not treat them as independent replicates of one task. No new p-value was computed.

- tag: addressable-now
  **The 15 frozen reference runs were counted inside the 150 and left next to the IQM.** The reported IQM and the signed-rank test use the 135 primary factorial runs. The reference plan in `src/adaptive_plasticity/final_runner.py` is HalfCheetah only and is excluded from that primary matrix. The setup now says the reference runs are HalfCheetah only and are not in that aggregate. No reference-run score was added.

- tag: addressable-now
  **The literature matrix called network resetting “fatal for O2O transfer.”** Resets are not an arm of the 135-cell matrix. The cell now says that, instead of stating a result the matrix does not contain.

- tag: addressable-now
  **The discussion stated the hazard as a fact about interventions in general.** It now scopes that sentence to the periodic Shrink-and-Perturb comparison, and it says the manuscript does not show the same effect for every plasticity operator.

- tag: addressable-now
  **The learner and the confirmatory arms were easy to miss.** Limitations now state that online fine-tuning is Implicit Q-Learning, and that the confirmatory arms are `none`, periodic Shrink-and-Perturb, and CapacityGate. Other offline-to-online algorithms are not compared.

## needs-data

These items stay open. Fixing them needs a run or a source file that is not in the manuscript. This note does not fill them.

- tag: needs-data
  **Five confirmatory seeds is a thin sample for an aggregate IQM.** Adding seeds would be a new run. The paper now states the count it has. It does not call that count large.

- tag: needs-data
  **CapacityGate under actuator crippling is seed 0 only.** Seeds 1 and 2 are blank in Table `tab:stress_results`. Multi-seed CapacityGate stress is one of the three unapplied figures in `docs/pending_revision_numbers.md`. Do not copy it in until a source file in this repository states it.

- tag: needs-data
  **The rank interval [0.9915, 1.6501] and a ReDo IQM of 40.37 are not manuscript numbers.** They are named in `docs/pending_revision_numbers.md` and rejected by the public claim check. They are not applied here.

- tag: needs-data
  **Hopper has no actuator-cripple result.** The stress table is HalfCheetah and Walker2d. A Hopper stress cell would be a new run.

- tag: needs-data
  **ReDo and periodic resets are outside the 135-cell matrix.** ReDo in the stress table is three seeds on two environments. Putting either operator into the confirmatory matrix would be a new run.

- tag: needs-data
  **The datasets are the three medium-v2 locomotion tasks.** Medium-replay, expert, manipulation, vision, and AntMaze are not in the manuscript. AntMaze is still the placeholder in `docs/blockers.md`.

- tag: needs-data
  **Cal-QL, PEX, and RO2O are not compared.** The code is not vendored. `docs/blockers.md` says what a wrapper would have to do. No score from those learners exists here.

- tag: needs-data
  **The gate thresholds are one published pair.** Limitations already say the runs are not a threshold sweep. A sweep would be new data.

- tag: needs-data
  **There is no per-environment signed-rank test in the manuscript.** The pooled test is now labeled as pooled. Computing a new p-value would be a new result. This note does not compute one.

- tag: needs-data
  **On the 45 primary cells the gate does not fire, so those cells do not measure a benefit from intervening.** The firing trace that is already printed is the HalfCheetah seed-0 stress telemetry. A further comparison would be a new result.
