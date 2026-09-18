import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import shutil

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10.5,
    'ytick.labelsize': 10.5,
    'legend.fontsize': 9.5,
    'figure.titlesize': 14,
    'lines.linewidth': 1.8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

fig_dir = Path("research_paper/iclr2027/figures")
fig_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# 1. IQM Bar Chart with exact stratified bootstrap 95% CIs
# -------------------------------------------------------------
methods = ['Baseline (None)', 'Fixed Periodic', 'CapacityGate']
iqm_vals = [38.78, 12.30, 38.78]
ci_lows = [37.43, 9.36, 37.43]
ci_highs = [40.03, 15.10, 40.03]

yerr = [
    [val - low for val, low in zip(iqm_vals, ci_lows)],
    [high - val for val, high in zip(iqm_vals, ci_highs)]
]

colors = ['#2563eb', '#ea580c', '#16a34a']

fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=300)
bars = ax.bar(methods, iqm_vals, yerr=yerr, capsize=6, color=colors, alpha=0.88, edgecolor='black', linewidth=1.1, width=0.55)

for bar, val in zip(bars, iqm_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 2.5, f'{val:.2f}', ha='center', va='bottom', fontsize=10.5, fontweight='bold')

ax.set_ylabel('Normalized Return (IQM)', fontweight='bold')
ax.set_title('Aggregate Online Performance (150 Paired Runs)\nwith 95% Stratified Bootstrap CIs', pad=12, fontweight='bold')
ax.set_ylim(0, 48)
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(['Baseline\n(None)', 'Fixed Periodic\n(Shrink-Perturb)', 'CapacityGate\n(Ours)'], fontweight='bold')

fig.tight_layout()
fig.savefig(fig_dir / "iqm.png")
plt.close(fig)
print("Saved iqm.png")

# -------------------------------------------------------------
# 2. Probability of Improvement Heatmap
# -------------------------------------------------------------
matrix = np.array([
    [0.50, 0.933, 0.50],
    [0.067, 0.50, 0.067],
    [0.50, 0.933, 0.50]
])

labels = ['Baseline', 'Fixed', 'CapacityGate']

fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=300)
cax = ax.imshow(matrix, cmap='Blues', vmin=0.0, vmax=1.0)

ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(labels, fontweight='bold', fontsize=10.5)
ax.set_yticklabels(labels, fontweight='bold', fontsize=10.5)

ax.set_xlabel('Column Method', fontweight='bold', labelpad=8)
ax.set_ylabel('Row Method', fontweight='bold', labelpad=8)
ax.set_title('Empirical Probability of Improvement\nP(Row > Column)', pad=12, fontweight='bold')

for i in range(3):
    for j in range(3):
        val = matrix[i, j]
        color = 'white' if val > 0.65 else 'black'
        text = f"{val:.3f}" if val not in [0.5, 0.0] else f"{val:.2f}"
        if i == 0 and j == 1: text = "0.933"
        if i == 1 and j in [0, 2]: text = "0.067"
        if i == 2 and j == 1: text = "0.933"
        ax.text(j, i, text, ha="center", va="center", color=color, fontweight='bold', fontsize=11)

cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
cbar.set_label('Probability of Superiority', rotation=270, labelpad=15, fontweight='bold')

fig.tight_layout()
fig.savefig(fig_dir / "prob_improvement.png")
plt.close(fig)
print("Saved prob_improvement.png")

# -------------------------------------------------------------
# 3. Stress-Study Operator Stability Asymmetry Figure
# -------------------------------------------------------------
envs = ['Walker2d\n(Balance-Critical)', 'HalfCheetah\n(Unconstrained Forward)']
x = np.arange(len(envs))
width = 0.19

# 3-Seed Means and Stds for None, Fixed, ReDo
none_means = [8.21, 6.23]
none_stds = [2.38, 1.53]

fixed_means = [-0.53, 15.26]
fixed_stds = [0.33, 2.93]

redo_means = [8.03, 13.64]
redo_stds = [1.57, 3.18]

# Single-Cell Deterministic Validation (Seed 0) for CapacityGate
gate_single_cell = [7.23, 8.58]

fig, ax = plt.subplots(figsize=(8.4, 4.6), dpi=300)

rects1 = ax.bar(x - 1.5*width, none_means, width, yerr=none_stds, capsize=4.5,
                label='Baseline (None, 3-Seed Mean)', color='#2563eb', alpha=0.9, edgecolor='black', linewidth=1)
rects2 = ax.bar(x - 0.5*width, fixed_means, width, yerr=fixed_stds, capsize=4.5,
                label='Fixed (Shrink-Perturb, 3-Seed Mean)', color='#dc2626', alpha=0.9, edgecolor='black', linewidth=1)
rects3 = ax.bar(x + 0.5*width, redo_means, width, yerr=redo_stds, capsize=4.5,
                label='ReDo (Neuron Recycling, 3-Seed Mean)', color='#16a34a', alpha=0.9, edgecolor='black', linewidth=1)
rects4 = ax.bar(x + 1.5*width, gate_single_cell, width,
                label='CapacityGate (Single-Cell Validated, Seed 0)',
                color='#f3e8ff', edgecolor='#9333ea', hatch='///', linewidth=1.4)

ax.set_ylabel('Evaluation Normalized Return', fontweight='bold')
ax.set_title('Operator Stability Asymmetry Under Severe Physical Non-Stationarity', pad=14, fontweight='bold', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(envs, fontweight='bold', fontsize=11)
ax.axhline(0, color='gray', linestyle='-', linewidth=0.8)
ax.set_ylim(-3.8, 21.5)
ax.legend(loc='upper left', framealpha=0.95, edgecolor='#cbd5e1', ncol=1)

# Annotations emphasizing single-cell vs 3-seed
ax.annotate('Single-Cell Validated\n(Seed 0: 7.23 Parity)',
            xy=(0 + 1.5*width, 7.23), xytext=(0.42, 4.5),
            arrowprops=dict(facecolor='#9333ea', shrink=0.08, width=1.2, headwidth=5),
            fontsize=8.5, fontweight='bold', color='#7e22ce', ha='center',
            bbox=dict(boxstyle="round,pad=0.25", fc="#faf5ff", ec="#d8b4fe", lw=1))

ax.annotate('Single-Cell Validated\n(Seed 0: 8.58 vs 7.99)',
            xy=(1 + 1.5*width, 8.58), xytext=(1.40, 5.5),
            arrowprops=dict(facecolor='#9333ea', shrink=0.08, width=1.2, headwidth=5),
            fontsize=8.5, fontweight='bold', color='#7e22ce', ha='center',
            bbox=dict(boxstyle="round,pad=0.25", fc="#faf5ff", ec="#d8b4fe", lw=1))

ax.annotate('Total Balance Collapse\n(Weights Destabilized: -0.53)',
            xy=(0 - 0.5*width, -0.6), xytext=(-0.35, -2.8),
            arrowprops=dict(facecolor='#dc2626', shrink=0.08, width=1.5, headwidth=6),
            fontsize=8.5, fontweight='bold', color='#b91c1c', ha='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="#fee2e2", ec="#fca5a5", lw=1))

ax.annotate('Preserves Balance Gait\n(Zero Functional Shift: 8.03)',
            xy=(0 + 0.5*width, 8.1), xytext=(0.02, 13.5),
            arrowprops=dict(facecolor='#16a34a', shrink=0.08, width=1.5, headwidth=6),
            fontsize=8.5, fontweight='bold', color='#15803d', ha='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="#dcfce7", ec="#86efac", lw=1))

ax.annotate('Disrupting Prior Aids\nGait Discovery (15.26)',
            xy=(1 - 0.5*width, 15.3), xytext=(0.88, 18.8),
            arrowprops=dict(facecolor='#dc2626', shrink=0.08, width=1.5, headwidth=6),
            fontsize=8.5, fontweight='bold', color='#b91c1c', ha='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="#fee2e2", ec="#fca5a5", lw=1))

fig.tight_layout()
fig.savefig(fig_dir / "stress_operator_asymmetry.png")
plt.close(fig)
print("Saved updated stress_operator_asymmetry.png with hatched single-cell CapacityGate bars!")
