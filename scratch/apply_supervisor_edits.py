import re

with open('research_paper/iclr2027/iclr2027_conference.tex', encoding='utf-8') as f:
    tex = f.read()

# 1. Add Research Questions before Summary of Key Findings & Contributions
rq_block = r"""\subsection{Research Questions}
To systematically investigate the efficacy and risks of plasticity interventions in offline-to-online continuous control, we formulate three core research questions:
\begin{itemize}
    \item \textbf{RQ1 (Intervention Vulnerability)}: \emph{Does unconditional periodic parameter intervention restore or degrade policy performance during offline-to-online transfer in continuous locomotion?}
    \item \textbf{RQ2 (Diagnostic Feasibility \& Abstention)}: \emph{Can online representation health metrics (feature effective rank and activation dormancy) reliably detect intact capacity and govern intervention decisions through closed-loop abstention?}
    \item \textbf{RQ3 (Operator Sensitivity Under Non-Stationarity)}: \emph{Under genuine physical distribution shifts (actuator crippling), how do functional-shift versus zero-functional-shift intervention operators differ in preserving continuous control equilibria?}
\end{itemize}

\subsection{Summary of Key Findings \& Contributions}"""

tex = tex.replace(r"\subsection{Summary of Key Findings \& Contributions}", rq_block)

# 2. Update Conclusion to Conclusion and Future Work with concrete future directions
old_conclusion = r"""\section{Conclusion}
\label{sec:conclusion}

In this paper, we conducted an empirical investigation into the behavior of plasticity interventions in offline-to-online continuous-control reinforcement learning. We showed that unconditional periodic Shrink-and-Perturb induces severe intervention vulnerability, causing a catastrophic 68.3\% collapse in aggregate IQM return across 150 total runs with matched environment/shift/seed comparisons. We formulated the ``Do Not Disturb'' principle and introduced \textsc{CapacityGate}, demonstrating that closed-loop diagnostic abstention effectively shields converged policies from unnecessary perturbations. Furthermore, we demonstrated that under physical non-stationarity, intervention operators exhibit a stark stability asymmetry: zero-functional-shift neuron recycling (ReDo) maintains viable continuous balance manifolds, whereas weight perturbation causes total dynamical collapse. 

Our findings indicate that plasticity interventions in offline-to-online RL should not be treated as indiscriminate background routines. Instead, future work can build upon conservative, diagnostic-gated intervention policies that respect the stability of pre-trained representations."""

new_conclusion = r"""\section{Conclusion and Future Work}
\label{sec:conclusion}

In this paper, we conducted an empirical investigation into the behavior of plasticity interventions in offline-to-online continuous-control reinforcement learning. Across 150 total runs with matched environment/shift/seed comparisons, we showed that unconditional periodic Shrink-and-Perturb induces severe intervention vulnerability, causing a catastrophic 68.3\% collapse in aggregate IQM return. We formulated the ``Do Not Disturb'' principle and introduced \textsc{CapacityGate}, demonstrating that closed-loop diagnostic abstention effectively shields converged policies from unnecessary perturbations. Furthermore, under physical non-stationarity, intervention operators exhibit a stark stability asymmetry: zero-functional-shift neuron recycling (ReDo) maintains viable continuous balance manifolds, whereas weight perturbation causes total dynamical collapse. 

Our findings indicate that plasticity interventions in offline-to-online RL should not be treated as indiscriminate background routines, but rather as regime- and operator-dependent operations governed by capacity telemetry.

\paragraph{Future Work}
Several promising directions emerge from this study:
\begin{enumerate}
    \item \textbf{Exploration-Oriented Zero-Shift Operators}: While ReDo demonstrates that zero-functional-shift operators ($W_{\mathrm{out}} = 0$) preserve physical balance, discovering novel gaits under altered kinematics requires directed exploration. Developing hybrid operators that combine the structural safety of zero-functional-shift recycling with targeted exploratory policy perturbations represents an important frontier for continuous control.
    \item \textbf{Adaptive and Self-Calibrating Thresholds}: In this work, diagnostic thresholds for rank ratio ($\rho_{\mathrm{on}}, \rho_{\mathrm{off}}$) and activation dormancy ($d_{\mathrm{on}}, d_{\mathrm{off}}$) were calibrated empirically. Extending \textsc{CapacityGate} to adaptively tune thresholds via online running statistics, Bayesian optimization, or meta-learning could enhance cross-domain generalizability without manual calibration.
    \item \textbf{High-Dimensional Visual Control and Robotics}: Investigating whether the ``Do Not Disturb'' principle generalizes to pixel-based continuous control (e.g., DeepMind Control Suite from pixels) and physical robotic hardware transitions will clarify the interplay between visual representation drift and control manifold preservation.
    \item \textbf{Multi-Task and Lifelong Transfer}: Scaling closed-loop diagnostic gating to sequences of heterogeneous tasks will determine how capacity telemetry scales across non-stationary continual learning lifetimes.
\end{enumerate}"""

assert old_conclusion in tex, "Old conclusion block not found"
tex = tex.replace(old_conclusion, new_conclusion)

# 3. Fix Appendix Table Numbering so tables are Table A.1 and Table A.2 instead of Table 5 and Table 6
old_appendix_header = r"""\newpage
\appendix
\section{Exhaustive Seed-Level Benchmark Data}
\label{app:per_seed}

Table~\ref{tab:per_seed_full} provides the complete, unabridged breakdown of final evaluation normalized returns across all 135 primary factorial runs in the confirmatory study, reporting individual performance for each of the 5 paired random seeds."""

new_appendix_header = r"""\newpage
\appendix
\renewcommand{\thetable}{A.\arabic{table}}
\setcounter{table}{0}
\renewcommand{\thefigure}{A.\arabic{figure}}
\setcounter{figure}{0}

\section{Exhaustive Seed-Level Benchmark Data}
\label{app:per_seed}

Table~\ref{tab:per_seed_full} (Appendix Table A.1) provides the complete, unabridged breakdown of final evaluation normalized returns across all 135 primary factorial runs in the confirmatory study, reporting individual performance for each of the 5 paired random seeds."""

assert old_appendix_header in tex, "Old appendix header not found"
tex = tex.replace(old_appendix_header, new_appendix_header)

tex = tex.replace(r"Table~\ref{tab:hyperparams} specifies the primary hyperparameters",
                  r"Table~\ref{tab:hyperparams} (Appendix Table A.2) specifies the primary hyperparameters")

with open('research_paper/iclr2027/iclr2027_conference.tex', 'w', encoding='utf-8') as f:
    f.write(tex)

print("Successfully updated paper with Supervisor suggestions (Research Questions, Conclusion & Future Work, Appendix Table numbering)!")
