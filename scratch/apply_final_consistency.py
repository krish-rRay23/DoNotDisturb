import re

with open('research_paper/iclr2027/iclr2027_conference.tex', encoding='utf-8') as f:
    tex = f.read()

# 1. Experimental setup: Mann-Whitney -> paired Wilcoxon signed-rank
old_setup_stat = r"Statistical aggregation follows the rigorous protocols recommended by \citet{agarwal2021rliable}, computing Interquartile Mean (IQM) scores with 2,000 stratified bootstrap replicates and non-parametric Mann-Whitney $U$ significance tests."
new_setup_stat = r"Statistical aggregation follows the rigorous protocols recommended by \citet{agarwal2021rliable}, computing Interquartile Mean (IQM) scores with 2,000 stratified bootstrap replicates and non-parametric paired Wilcoxon signed-rank significance tests on matched environment/shift/seed pairs."
assert old_setup_stat in tex, "Setup stat text not found"
tex = tex.replace(old_setup_stat, new_setup_stat)

# 2. Figure 1 caption: "Evaluating 150 independent online adaptation runs" -> "Evaluating 150 total runs with matched environment/shift/seed comparisons"
old_fig1_cap = r"Evaluating 150 independent online adaptation runs across D4RL locomotion tasks"
new_fig1_cap = r"Evaluating 150 total runs with matched environment/shift/seed comparisons across D4RL locomotion tasks"
assert old_fig1_cap in tex, "Fig 1 caption text not found"
tex = tex.replace(old_fig1_cap, new_fig1_cap)

# Also in Introduction summary and Conclusion:
tex = tex.replace(r"across 150 evaluated runs (Figure~\ref{fig:aggregate_benchmark})",
                  r"across 150 total runs with matched environment/shift/seed comparisons (Figure~\ref{fig:aggregate_benchmark})")
tex = tex.replace(r"across 150 benchmark runs.",
                  r"across 150 total runs with matched environment/shift/seed comparisons.")

# 3. Figure 4 caption: clearly distinguish CapacityGate as single-cell validation
old_fig4_cap = r"""\caption{\textbf{Operator Stability Asymmetry Under Severe Physical Non-Stationarity.} Evaluation normalized returns under sudden joint actuator crippling at step 5,000. In balance-critical Walker2d, weight-perturbation operators (Fixed) inject non-zero functional shifts that destroy the bipedal equilibrium manifold ($-0.53 \pm 0.33$). In sharp contrast, dormant neuron recycling (ReDo) preserves balance identically ($8.03 \pm 1.57$, matching Baseline $8.21 \pm 2.38$) due to its zero-functional-shift formulation ($W_{\mathrm{out}} = 0$). In unconstrained HalfCheetah, disrupting the offline prior aids in discovering an alternate front-joint galloping gait, enabling Fixed ($15.26 \pm 2.93$) and ReDo ($13.64 \pm 3.18$) to outperform the passive baseline ($6.23 \pm 1.53$).}"""

new_fig4_cap = r"""\caption{\textbf{Operator Stability Asymmetry Under Severe Physical Non-Stationarity.} Evaluation normalized returns under sudden joint actuator crippling at step 5,000. Bars for Baseline (None), Fixed Periodic (Shrink-and-Perturb), and ReDo (Neuron Recycling) depict mean $\pm$ std across 3 paired random seeds. CapacityGate is displayed with hatched bars to clearly denote single-cell deterministic validation (Seed 0) rather than a 3-seed mean: on Walker2d Seed 0, CapacityGate safely abstains from intervention, preserving Baseline parity ($7.23$); on HalfCheetah Seed 0, it executes spaced pulses with refractory cooldown, achieving $8.58$ vs. $7.99$ unperturbed. In balance-critical Walker2d, weight-perturbation operators (Fixed) inject non-zero functional shifts that destroy the bipedal equilibrium manifold ($-0.53 \pm 0.33$). In sharp contrast, dormant neuron recycling (ReDo) preserves balance ($8.03 \pm 1.57$, matching Baseline $8.21 \pm 2.38$) due to its zero-functional-shift formulation ($W_{\mathrm{out}} = 0$). In unconstrained HalfCheetah, disrupting the offline prior aids in discovering an alternate front-joint galloping gait, enabling Fixed ($15.26 \pm 2.93$) and ReDo ($13.64 \pm 3.18$) to outperform the passive baseline ($6.23 \pm 1.53$).}"""

assert old_fig4_cap in tex, "Fig 4 caption text not found"
tex = tex.replace(old_fig4_cap, new_fig4_cap)

# 4 & 5. Remove remaining overclaims: "must", "necessary", "essential"
tex = tex.replace(r"periodic, scheduled intervention is necessary to maintain learning agility.",
                  r"periodic, scheduled intervention is widely assumed to maintain learning agility.")

tex = tex.replace(r"converged policy manifolds must be left undisturbed.",
                  r"converged policy manifolds should be left undisturbed.")

tex = tex.replace(r"diagnostic probes must sample uniformly across the replay buffer",
                  r"diagnostic probes should sample uniformly across the replay buffer")

tex = tex.replace(r"probe transitions must be sampled \emph{uniformly at random without replacement}",
                  r"probe transitions are sampled \emph{uniformly at random without replacement}")

tex = tex.replace(r"To guarantee exact experimental reproducibility",
                  r"To preserve exact experimental reproducibility")

tex = tex.replace(r"An intelligent RL agent must follow the ``Do Not Disturb'' principle, leaving healthy policy manifolds intact.",
                  r"This reinforces the ``Do Not Disturb'' principle, leaving healthy policy manifolds intact.")

tex = tex.replace(r"representation capacity is a necessary, but not sufficient, condition for policy adaptation",
                  r"representation capacity is a prerequisite, but not a guarantee, for policy adaptation")

tex = tex.replace(r"Deactivation thresholds must be calibrated to the baseline dormancy",
                  r"Deactivation thresholds should be calibrated to the baseline dormancy")

tex = tex.replace(r"future work must focus on conservative, diagnostic-gated intervention policies",
                  r"future work can build upon conservative, diagnostic-gated intervention policies")

with open('research_paper/iclr2027/iclr2027_conference.tex', 'w', encoding='utf-8') as f:
    f.write(tex)

print("Successfully applied consistency updates to iclr2027_conference.tex!")
