import os
import json
import glob
import numpy as np
from scipy import stats

results_dir = "results"
envs = ["halfcheetah-medium-v2", "hopper-medium-v2", "walker2d-medium-v2"]
regimes = [
    ("none", "0"),
    ("obs_noise", "0.1"),
    ("reward_scale", "0.5")
]
arms = ["none", "fixed", "capacity_gate"]
seeds = [0, 1, 2, 3, 4]

data = {}
for env in envs:
    data[env] = {}
    for reg, sev in regimes:
        data[env][reg] = {}
        for arm in arms:
            data[env][reg][arm] = {}
            for seed in seeds:
                cell_name = f"FINAL-{env}-{arm}-{reg}-{f'sev{sev}'}-seed{seed}"
                cell_path = os.path.join(results_dir, cell_name, "summary.json")
                if not os.path.exists(cell_path):
                    raise FileNotFoundError(f"Missing {cell_path}")
                with open(cell_path, "r") as f:
                    s = json.load(f)
                norm_evals = s.get("normalized", [])
                last_score = float(norm_evals[-1])
                last3_score = float(np.mean(norm_evals[-3:]))
                data[env][reg][arm][seed] = {
                    "last": last_score,
                    "last3": last3_score,
                    "norm_evals": norm_evals,
                    "applied_intervention_steps": s.get("applied_intervention_steps", 0)
                }

def bootstrap_ci(diffs, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    diffs = np.array(diffs)
    n = len(diffs)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means[b] = np.mean(sample)
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))

def exact_sign_test(diffs):
    pos = sum(d > 0 for d in diffs)
    neg = sum(d < 0 for d in diffs)
    n_non_zero = pos + neg
    if n_non_zero == 0:
        return 1.0, pos, neg
    k = min(pos, neg)
    res = stats.binomtest(k, n_non_zero, 0.5, alternative='two-sided')
    return float(res.pvalue), pos, neg

def paired_wilcoxon(diffs):
    if all(d == 0 for d in diffs):
        return 1.0
    try:
        res = stats.wilcoxon(diffs, zero_method='pratt')
        return float(res.pvalue)
    except Exception as e:
        return 1.0

# 1. Check identity of 'none' regime across arms
print("=== PRIORITY 1: 'none' REGIME IDENTITY CHECK ===")
max_diff = 0.0
for env in envs:
    print(f"\n--- {env} ---")
    for seed in seeds:
        s_none = data[env]["none"]["none"][seed]["last"]
        s_fixed = data[env]["none"]["fixed"][seed]["last"]
        s_gate = data[env]["none"]["capacity_gate"][seed]["last"]
        d1 = abs(s_none - s_fixed)
        d2 = abs(s_none - s_gate)
        max_diff = max(max_diff, d1, d2)
        print(f"Seed {seed} | None: {s_none:10.6f} | Fixed: {s_fixed:10.6f} | Gate: {s_gate:10.6f} | |Fixed-None|={d1:.2e} |Gate-None|={d2:.2e}")
print(f"\nMaximum absolute difference across all seeds/arms in 'none' regime: {max_diff:.2e}")

# 2. Priority 3 & 4: Compute paired statistics
print("\n=== PRIORITY 3 & 4: PAIRED STATS (Fixed vs None) ===")

results_table = []
p_values_last = []
p_values_last3 = []

for env in envs:
    for reg, sev in regimes:
        fixed_last = [data[env][reg]["fixed"][s]["last"] for s in seeds]
        none_last = [data[env][reg]["none"][s]["last"] for s in seeds]
        diffs_last = [f - n for f, n in zip(fixed_last, none_last)]
        
        fixed_last3 = [data[env][reg]["fixed"][s]["last3"] for s in seeds]
        none_last3 = [data[env][reg]["none"][s]["last3"] for s in seeds]
        diffs_last3 = [f - n for f, n in zip(fixed_last3, none_last3)]
        
        mean_diff_last = float(np.mean(diffs_last))
        ci_last = bootstrap_ci(diffs_last)
        pval_sign_last, pos_l, neg_l = exact_sign_test(diffs_last)
        pval_wilc_last = paired_wilcoxon(diffs_last)
        
        mean_diff_last3 = float(np.mean(diffs_last3))
        ci_last3 = bootstrap_ci(diffs_last3)
        pval_sign_last3, pos_l3, neg_l3 = exact_sign_test(diffs_last3)
        pval_wilc_last3 = paired_wilcoxon(diffs_last3)
        
        results_table.append({
            "env": env,
            "regime": reg,
            "mean_none_last": float(np.mean(none_last)),
            "mean_fixed_last": float(np.mean(fixed_last)),
            "mean_diff_last": mean_diff_last,
            "diffs_last": [float(x) for x in diffs_last],
            "ci_last": ci_last,
            "pval_sign_last": pval_sign_last,
            "pval_wilc_last": pval_wilc_last,
            "pos_neg_last": [pos_l, neg_l],
            "mean_none_last3": float(np.mean(none_last3)),
            "mean_fixed_last3": float(np.mean(fixed_last3)),
            "mean_diff_last3": mean_diff_last3,
            "diffs_last3": [float(x) for x in diffs_last3],
            "ci_last3": ci_last3,
            "pval_sign_last3": pval_sign_last3,
            "pval_wilc_last3": pval_wilc_last3,
            "pos_neg_last3": [pos_l3, neg_l3],
        })
        p_values_last.append(pval_wilc_last)
        p_values_last3.append(pval_wilc_last3)

# Holm-Bonferroni correction
def holm_bonferroni(pvals):
    n = len(pvals)
    indexed_p = sorted(enumerate(pvals), key=lambda x: x[1])
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed_p):
        multiplier = n - rank
        p_adj = min(1.0, p * multiplier)
        running_max = max(running_max, p_adj)
        adjusted[orig_idx] = running_max
    return adjusted

holm_last = holm_bonferroni(p_values_last)
holm_last3 = holm_bonferroni(p_values_last3)

for idx, r in enumerate(results_table):
    r["holm_wilc_last"] = float(holm_last[idx])
    r["holm_wilc_last3"] = float(holm_last3[idx])

print("\n--- PRIMARY ENDPOINT: LAST EVALUATION SCORE ---")
print(f"{'Env':<22} | {'Regime':<12} | {'Mean None':<9} | {'Mean Fixed':<10} | {'Mean Diff':<10} | {'95% CI':<22} | {'(+/-)':<5} | {'Sign p':<8} | {'Wilc p':<8} | {'Holm p':<8}")
for r in results_table:
    ci_str = f"[{r['ci_last'][0]:+.2f}, {r['ci_last'][1]:+.2f}]"
    sn_str = f"{r['pos_neg_last'][0]}/{r['pos_neg_last'][1]}"
    print(f"{r['env']:<22} | {r['regime']:<12} | {r['mean_none_last']:<9.2f} | {r['mean_fixed_last']:<10.2f} | {r['mean_diff_last']:<+10.2f} | {ci_str:<22} | {sn_str:<5} | {r['pval_sign_last']:<8.4f} | {r['pval_wilc_last']:<8.4f} | {r['holm_wilc_last']:<8.4f}")

print("\n--- SENSITIVITY ENDPOINT: MEAN OF LAST 3 EVALUATIONS ---")
print(f"{'Env':<22} | {'Regime':<12} | {'Mean None':<9} | {'Mean Fixed':<10} | {'Mean Diff':<10} | {'95% CI':<22} | {'(+/-)':<5} | {'Sign p':<8} | {'Wilc p':<8} | {'Holm p':<8}")
for r in results_table:
    ci_str = f"[{r['ci_last3'][0]:+.2f}, {r['ci_last3'][1]:+.2f}]"
    sn_str = f"{r['pos_neg_last3'][0]}/{r['pos_neg_last3'][1]}"
    print(f"{r['env']:<22} | {r['regime']:<12} | {r['mean_none_last3']:<9.2f} | {r['mean_fixed_last3']:<10.2f} | {r['mean_diff_last3']:<+10.2f} | {ci_str:<22} | {sn_str:<5} | {r['pval_sign_last3']:<8.4f} | {r['pval_wilc_last3']:<8.4f} | {r['holm_wilc_last3']:<8.4f}")

with open("scratch/stats_output.json", "w") as f:
    json.dump(results_table, f, indent=2)
print("\nSaved stats to scratch/stats_output.json")
