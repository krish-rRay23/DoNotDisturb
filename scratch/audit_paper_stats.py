import pandas as pd
from scipy import stats
import numpy as np

df = pd.read_csv('results/final_study/FINAL_rliable_scores.csv')
print("Columns:", df.columns.tolist())
piv = df.pivot(index=['env', 'shift', 'seed'], columns='controller', values='score')
print("Pivoted shape:", piv.shape)

fixed = piv['fixed']
none = piv['none']
cap = piv['capacity_gate']

# Paired Wilcoxon
wilc = stats.wilcoxon(fixed, none)
# Mann-Whitney U
u_stat, u_pval = stats.mannwhitneyu(fixed, none)

print(f"\n=== OVERALL (N=45 paired cells) ===")
print(f"Fixed: mean={fixed.mean():.2f}, median={fixed.median():.2f}")
print(f"None:  mean={none.mean():.2f}, median={none.median():.2f}")
print(f"Cap:   mean={cap.mean():.2f}, median={cap.median():.2f}")
print(f"Mann-Whitney U: stat={u_stat}, p-value={u_pval:.4e}")
print(f"Wilcoxon signed-rank: stat={wilc.statistic}, p-value={wilc.pvalue:.4e}")
print(f"None - Cap difference: {(none - cap).abs().max():.6f}")

# Cell by cell breakdown
print("\n=== CELL-BY-CELL BREAKDOWN ===")
summary = df.groupby(['env', 'shift', 'controller'])['score'].agg(['mean', 'std', 'median']).reset_index()
print(summary.to_string())

# Stress study breakdown
df_stress = pd.read_csv('results/stress_study/STRESS_rliable_scores.csv')
print("\n=== STRESS STUDY BREAKDOWN ===")
summary_stress = df_stress.groupby(['env', 'controller'])['score'].agg(['mean', 'std', 'median']).reset_index()
print(summary_stress.to_string())
