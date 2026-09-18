import re

with open('research_paper/iclr2027/iclr2027_conference.tex', encoding='utf-8') as f:
    tex = f.read()

# 1. Update Abstract statistics & overclaims
old_abs_target = r"causing a catastrophic 68.3\% collapse in aggregate Interquartile Mean (IQM) normalized return ($38.78 \to 12.30$, $p = 2.11 \times 10^{-19}$). To resolve this dilemma, we articulate the \textbf{``Do Not Disturb''} principle: an agent should intervene if and only if representation capacity is demonstrably impaired. We introduce \textsc{CapacityGate}, a closed-loop diagnostic framework that monitors feature effective rank and neuron dormancy using uniform replay reservoir sampling and refractory cooldown control. In stable and moderately shifted transfer, \textsc{CapacityGate} detects that representation capacity remains intact ($\rho \approx 1.0, d < 0.08$) and safely abstains from intervention, preserving baseline performance ($38.78$ IQM) and acting as an essential safeguard against perturbation-induced collapse."

new_abs_target = r"causing a catastrophic 68.3\% collapse in aggregate Interquartile Mean (IQM) normalized return ($38.78 \to 12.30$, paired Wilcoxon signed-rank $W = 18.0$, $p = 1.44 \times 10^{-11}$, mean paired difference $+26.03 \pm 19.09$). To address this vulnerability, we articulate the \textbf{``Do Not Disturb''} principle: an agent should intervene only when representation capacity is demonstrably impaired. We introduce \textsc{CapacityGate}, a closed-loop diagnostic framework that monitors feature effective rank and neuron dormancy using uniform replay reservoir sampling and refractory cooldown control. In stable and moderately shifted transfer, \textsc{CapacityGate} detects that representation capacity remains intact ($\rho \approx 1.0, d < 0.08$) and safely abstains from intervention, preserving baseline performance ($38.78$ IQM) and protecting converged policies against perturbation-induced collapse."

assert old_abs_target in tex, "Abstract target text not found!"
tex = tex.replace(old_abs_target, new_abs_target)

# 2. Update Contribution 1 & 2 in Introduction
old_intro_contrib = r"""\begin{enumerate}
    \item \textbf{Intervention Vulnerability in Stable O2O Transfer}: Unconditional periodic Shrink-and-Perturb \citep{ash2020warmstarting} collapses aggregate IQM normalized return by 68.3\% ($38.78 \to 12.30$, Mann-Whitney $U = 114.0$, $p = 2.11 \times 10^{-19}$) across 150 evaluated runs (Figure~\ref{fig:aggregate_benchmark}). In balance-critical tasks such as Hopper and Walker2d, periodic perturbation induces immediate and permanent gait failure (dropping normalized return on Walker2d to $2.14 \pm 5.79$ vs. $31.45 \pm 12.65$ baseline).
    \item \textbf{Diagnostic Abstention as Active Protection}: In stable and moderately shifted O2O transfer, representations do not spontaneously collapse: relative effective rank remains near baseline ($\rho \in [0.99, 1.03]$) and dormant neuron fractions remain minimal ($d \le 0.085$). By detecting that capacity remains intact, \textsc{CapacityGate} achieves \emph{zero observed interventions across tested stable-transfer conditions}, preserving baseline performance identically ($38.78$ IQM) and acting as a necessary circuit breaker against perturbation-induced collapse."""

new_intro_contrib = r"""\begin{enumerate}
    \item \textbf{Intervention Vulnerability in Stable O2O Transfer}: Unconditional periodic Shrink-and-Perturb \citep{ash2020warmstarting} collapses aggregate IQM normalized return by 68.3\% ($38.78 \to 12.30$, paired Wilcoxon signed-rank $W = 18.0$, $p = 1.44 \times 10^{-11}$) across 150 evaluated runs (Figure~\ref{fig:aggregate_benchmark}). In balance-critical tasks such as Hopper and Walker2d, periodic perturbation induces immediate and permanent gait failure (dropping normalized return on Walker2d to $2.14 \pm 5.79$ vs. $31.45 \pm 12.65$ baseline).
    \item \textbf{Diagnostic Abstention as Active Protection}: In stable and moderately shifted O2O transfer, representations do not spontaneously collapse: relative effective rank remains near baseline ($\rho \in [0.99, 1.03]$) and dormant neuron fractions remain minimal ($d \le 0.085$). By detecting that capacity remains intact, \textsc{CapacityGate} achieves \emph{zero observed interventions across tested stable-transfer conditions}, preserving baseline performance identically ($38.78$ IQM) and preventing perturbation-induced collapse."""

assert old_intro_contrib in tex, "Intro contrib target not found!"
tex = tex.replace(old_intro_contrib, new_intro_contrib)

# 3. Update Table 1 wording (remove "Proves")
tex = tex.replace(r"Proves open-loop parameter destruction is catastrophic for converged balance manifolds.",
                  r"Shows open-loop parameter destruction is catastrophic for converged balance manifolds.")
tex = tex.replace(r"Proves diagnostic abstention protects converged policies; identifies operator asymmetry.",
                  r"Demonstrates that diagnostic abstention protects converged policies; reveals operator asymmetry.")
tex = tex.replace(r"the first systematic investigation", r"a systematic investigation")

# 4. Update Section 3 cooldown sentence
tex = tex.replace(r"parameter intervention is actuated if and only if at least $\Delta_{\mathrm{cool}} = 3,000$",
                  r"parameter intervention is actuated only when at least $\Delta_{\mathrm{cool}} = 3,000$")

# 5. Update Results section statistical paragraph
old_stat_p = r"""A non-parametric Mann-Whitney $U$ test confirms that this collapse is statistically decisive ($U = 114.0$, $p = 2.11 \times 10^{-19}$). As depicted in Figure~\ref{fig:aggregate_benchmark}b, the empirical probability of improvement of Fixed over Baseline is merely $P(\text{Fixed} > \text{None}) = 0.067$ ($95\%$ CI $[0.013, 0.124]$), while Baseline outperforms Fixed in 93.3\% of comparisons ($P(\text{None} > \text{Fixed}) = 0.933$)."""

new_stat_p = r"""Because our experimental evaluation pairs seeds identically across intervention arms, we conduct a paired non-parametric Wilcoxon signed-rank test on the paired differences $(\text{Baseline} - \text{Fixed})$ across all $N=45$ matched evaluations. The analysis confirms a decisive performance penalty for unconditional intervention (Wilcoxon signed-rank $W = 18.0$, $p = 1.44 \times 10^{-11}$; paired $t$-test $t = 9.15, p = 9.69 \times 10^{-12}$). Baseline outperforms Fixed in 42 out of 45 paired evaluations (93.3\% positive pairs), with a mean paired difference of $+26.03 \pm 19.09$ normalized return (95\% bootstrap CI $[20.72, 31.59]$) and a median paired difference of $+22.27$ (IQR: $[17.38, 33.87]$). As depicted in Figure~\ref{fig:aggregate_benchmark}b, the empirical probability of improvement of Fixed over Baseline is merely $P(\text{Fixed} > \text{None}) = 0.067$ ($95\%$ CI $[0.013, 0.124]$), while Baseline outperforms Fixed in 93.3\% of comparisons ($P(\text{None} > \text{Fixed}) = 0.933$)."""

assert old_stat_p in tex, "Stat paragraph target not found!"
tex = tex.replace(old_stat_p, new_stat_p)

# 6. Update Section 5.3 to introduce Figure 4 (stress_operator_asymmetry.png) and strengthen narrative
old_stress_sec = r"""\subsection{Operator Stability Asymmetry Under Non-Stationary Stress}
\label{sec:stress}

What occurs when an agent experiences a genuine, non-stationary physical disruption? We evaluate this question in our 24-run stress benchmark under sudden actuator crippling ($a_{\mathrm{crippled}} = 0$ at step 5,000). Table~\ref{tab:stress_results} reports the normalized returns across 3 paired seeds for Baseline (\texttt{none}), Fixed Periodic (\texttt{fixed}), Dormant Neuron Recycling (\texttt{redo}), and \textsc{CapacityGate}.

The empirical data reveals a sharp \textbf{operator stability asymmetry}:
\begin{itemize}
    \item \textbf{Walker2d Balance Collapse}: Walker2d requires continuous bipedal coordination to avoid falling. Unconditional Shrink-and-Perturb (\texttt{fixed}) injects non-zero functional shifts ($\Delta f_\theta(x) \neq 0$), destroying the agent's balance manifold and collapsing return to $-0.53 \pm 0.33$. In stark contrast, ReDo recycles dormant neurons while setting outgoing weights to zero ($W_{\mathrm{out}} = 0$), preserving input-output mappings identically ($f_{\theta'}(x) \equiv f_\theta(x)$). As a result, ReDo preserves locomotion stability, achieving $8.03 \pm 1.57$ (matching the unperturbed baseline of $8.21 \pm 2.38$).
    \item \textbf{HalfCheetah Gait Re-Exploration}: Unlike Walker2d, HalfCheetah cannot fall over. Disabling the rear actuator severely impedes forward velocity ($6.23 \pm 1.53$ baseline). In this unconstrained setting, breaking the offline prior via weight perturbation or neuron recycling aids in discovering a front-joint galloping gait, enabling Fixed ($15.26 \pm 2.93$) and ReDo ($13.64 \pm 3.18$) to outperform the passive baseline.
\end{itemize}

This comparison establishes that \emph{the choice of intervention operator is as decisive as the decision to intervene}. Operators that induce arbitrary functional shifts are catastrophic for delicate continuous control tasks, whereas zero-functional-shift operators can refresh latent units without destabilizing physical equilibrium.

\begin{table}[t]"""

new_stress_sec = r"""\subsection{Operator Stability Asymmetry Under Non-Stationary Stress}
\label{sec:stress}

What occurs when an agent experiences a genuine, non-stationary physical disruption? We evaluate this question in our 24-run stress benchmark under sudden actuator crippling ($a_{\mathrm{crippled}} = 0$ at step 5,000). As illustrated in Figure~\ref{fig:stress_operator_asymmetry} and detailed in Table~\ref{tab:stress_results}, the empirical data demonstrates that intervention efficacy is \textbf{both shift-regime- and operator-dependent}:

\begin{figure}[t]
\centering
\includegraphics[width=0.88\textwidth]{figures/stress_operator_asymmetry.png}
\caption{\textbf{Operator Stability Asymmetry Under Severe Physical Non-Stationarity.} Evaluation normalized returns under sudden joint actuator crippling at step 5,000. In balance-critical Walker2d, weight-perturbation operators (Fixed) inject non-zero functional shifts that destroy the bipedal equilibrium manifold ($-0.53 \pm 0.33$). In sharp contrast, dormant neuron recycling (ReDo) preserves balance identically ($8.03 \pm 1.57$, matching Baseline $8.21 \pm 2.38$) due to its zero-functional-shift formulation ($W_{\mathrm{out}} = 0$). In unconstrained HalfCheetah, disrupting the offline prior aids in discovering an alternate front-joint galloping gait, enabling Fixed ($15.26 \pm 2.93$) and ReDo ($13.64 \pm 3.18$) to outperform the passive baseline ($6.23 \pm 1.53$).}
\label{fig:stress_operator_asymmetry}
\end{figure}

\begin{itemize}
    \item \textbf{Walker2d Balance Collapse (Fixed $-0.53$ vs. ReDo $8.03$ vs. Baseline $8.21$)}: Walker2d requires delicate, continuous bipedal coordination to avoid falling. Unconditional Shrink-and-Perturb (\texttt{fixed}) injects non-zero functional shifts ($\Delta f_\theta(x) \neq 0$), destabilizing the agent's balance manifold and collapsing return to $-0.53 \pm 0.33$. In stark contrast, ReDo recycles dormant neurons while setting outgoing weights to zero ($W_{\mathrm{out}} = 0$), preserving input-output mappings identically ($f_{\theta'}(x) \equiv f_\theta(x)$). As a result, ReDo maintains viable walking gait dynamics, achieving $8.03 \pm 1.57$ (matching the unperturbed baseline of $8.21 \pm 2.38$).
    \item \textbf{HalfCheetah Gait Re-Exploration (Fixed $15.26$ vs. ReDo $13.64$ vs. Baseline $6.23$)}: Unlike Walker2d, HalfCheetah cannot fall over. Disabling the rear actuator severely impedes forward velocity under the offline prior ($6.23 \pm 1.53$ baseline). In this unconstrained forward propulsion task, disrupting the offline prior via weight perturbation or neuron recycling promotes discovering an alternate front-joint galloping gait, enabling Fixed ($15.26 \pm 2.93$) and ReDo ($13.64 \pm 3.18$) to substantially outperform the passive baseline.
\end{itemize}

This comparison establishes that \emph{the choice of intervention operator is as decisive as the decision to intervene}. Operators that induce arbitrary functional shifts are catastrophic for balance-critical continuous control tasks, whereas zero-functional-shift operators can refresh latent units without destabilizing physical equilibrium.

\begin{table}[t]"""

assert old_stress_sec in tex, "Stress section target not found!"
tex = tex.replace(old_stress_sec, new_stress_sec)

# 7. Update Discussion circuit breaker wording
tex = tex.replace(r"Gating acts as an essential circuit breaker", r"Gating provides an effective circuit-breaking mechanism")

with open('research_paper/iclr2027/iclr2027_conference.tex', 'w', encoding='utf-8') as f:
    f.write(tex)

print("Updated iclr2027_conference.tex with audited paired stats and strengthened stress narrative!")
