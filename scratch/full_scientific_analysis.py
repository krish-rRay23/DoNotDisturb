import os
import json
import glob
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = "results/final_study_v2"
OUTPUT_DIR = "results/final_study_v2_analysis"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

ENVS = ["halfcheetah-medium-v2", "hopper-medium-v2", "walker2d-medium-v2"]
SHIFTS = ["none", "obs_noise", "reward_scale"]
PRIMARY_CONTROLLERS = ["none", "fixed", "capacity_gate"]
SEEDS = [0, 1, 2, 3, 4]

# Step 1: Load and Audit Dataset
expected_primary_runs = []
for env in ENVS:
    for shift in SHIFTS:
        sev = 0.0 if shift == "none" else (0.1 if shift == "obs_noise" else 0.5)
        for ctrl in PRIMARY_CONTROLLERS:
            for seed in SEEDS:
                folder_name = f"FINAL-{env}-{ctrl}-{shift}-sev{sev}-seed{seed}"
                expected_primary_runs.append((env, ctrl, shift, sev, seed, folder_name, False))

expected_ref_runs = []
for shift in SHIFTS:
    sev = 0.0 if shift == "none" else (0.1 if shift == "obs_noise" else 0.5)
    for seed in SEEDS:
        folder_name = f"FINAL-halfcheetah-medium-v2-m10ref-{shift}-sev{sev}-seed{seed}"
        expected_ref_runs.append(("halfcheetah-medium-v2", "m10ref", shift, sev, seed, folder_name, True))

all_expected = expected_primary_runs + expected_ref_runs

rows = []
complete_count = 0
corrupt_count = 0
missing_count = 0

for env, ctrl, shift, sev, seed, folder_name, is_ref in all_expected:
    dir_path = os.path.join(RESULTS_DIR, folder_name)
    if not os.path.exists(dir_path):
        missing_count += 1
        continue
        
    summary_path = os.path.join(dir_path, "summary.json")
    meta_path = os.path.join(dir_path, "metadata.json")
    ckpt_path = os.path.join(dir_path, "checkpoint.pt")
    gate_files = glob.glob(os.path.join(dir_path, "gate_*.jsonl"))
    
    if not (os.path.exists(summary_path) and os.path.exists(meta_path) and os.path.exists(ckpt_path) and gate_files):
        corrupt_count += 1
        continue
        
    with open(summary_path, "r") as f:
        summary = json.load(f)
        
    gate_records = []
    with open(gate_files[0], "r") as f:
        for line in f:
            if line.strip():
                gate_records.append(json.loads(line))
                
    norm_scores = summary.get("normalized", [])
    returns = summary.get("returns", [])
    
    if len(norm_scores) != 10 or len(gate_records) != 25:
        corrupt_count += 1
        continue
        
    complete_count += 1
    
    pre_shift_score = norm_scores[1]  # step 5000
    post_shift_score = norm_scores[2] # step 7500
    final_score = norm_scores[-1]     # step 25000
    last3_score = float(np.mean(norm_scores[-3:]))
    best_score = summary.get("best_normalized", np.nan)
    
    degradation = pre_shift_score - post_shift_score
    recovery = final_score - post_shift_score
    net_change = final_score - pre_shift_score
    
    applied_interventions = summary.get("applied_intervention_steps", 0)
    gate_info = summary.get("gate", {})
    gate_frequency = gate_info.get("intervention_frequency", 0.0)
    gate_n_interventions = gate_info.get("n_interventions", 0)
    rank_anchor = gate_info.get("rank_anchor", np.nan)
    
    dorm_seq = [r.get("dormancy", np.nan) for r in gate_records]
    erank_seq = [r.get("effective_rank", np.nan) for r in gate_records]
    rho_seq = [r.get("rho", np.nan) for r in gate_records]
    gate_on_seq = [r.get("gate_on", False) for r in gate_records]
    interv_active_seq = [r.get("intervention_active", False) for r in gate_records]
    
    pre_shift_dorm = float(np.nanmean([r.get("dormancy", np.nan) for r in gate_records if r.get("step", 0) <= 5000]))
    post_shift_dorm = float(np.nanmean([r.get("dormancy", np.nan) for r in gate_records if r.get("step", 0) > 5000]))
    
    pre_shift_erank = float(np.nanmean([r.get("effective_rank", np.nan) for r in gate_records if r.get("step", 0) <= 5000]))
    post_shift_erank = float(np.nanmean([r.get("effective_rank", np.nan) for r in gate_records if r.get("step", 0) > 5000]))

    pre_shift_rho = float(np.nanmean([r.get("rho", np.nan) for r in gate_records if r.get("step", 0) <= 5000]))
    post_shift_rho = float(np.nanmean([r.get("rho", np.nan) for r in gate_records if r.get("step", 0) > 5000]))
    
    gate_triggers_count = sum(1 for r in gate_records if r.get("gate_on", False))
    gate_triggers_post_shift = sum(1 for r in gate_records if r.get("gate_on", False) and r.get("step", 0) > 5000)
    
    row = {
        "env": env,
        "controller": ctrl,
        "shift": shift,
        "severity": sev,
        "seed": seed,
        "is_ref": is_ref,
        "folder_name": folder_name,
        "pre_shift_score": pre_shift_score,
        "post_shift_score": post_shift_score,
        "final_score": final_score,
        "last3_score": last3_score,
        "best_score": best_score,
        "degradation": degradation,
        "recovery": recovery,
        "net_change": net_change,
        "applied_interventions": applied_interventions,
        "gate_frequency": gate_frequency,
        "gate_n_interventions": gate_n_interventions,
        "gate_triggers_count": gate_triggers_count,
        "gate_triggers_post_shift": gate_triggers_post_shift,
        "rank_anchor": rank_anchor,
        "pre_shift_dorm": pre_shift_dorm,
        "post_shift_dorm": post_shift_dorm,
        "pre_shift_erank": pre_shift_erank,
        "post_shift_erank": post_shift_erank,
        "pre_shift_rho": pre_shift_rho,
        "post_shift_rho": post_shift_rho,
        "norm_scores_seq": norm_scores,
        "returns_seq": returns,
        "dorm_seq": dorm_seq,
        "erank_seq": erank_seq,
        "rho_seq": rho_seq,
        "gate_on_seq": gate_on_seq,
        "interv_active_seq": interv_active_seq,
        "runtime_seconds": summary.get("runtime_seconds", np.nan)
    }
    rows.append(row)

df = pd.DataFrame(rows)

print(f"=== AUDIT SUMMARY ===")
print(f"Total Runs Expected: {len(all_expected)}")
print(f"Complete & Valid Runs: {complete_count}")
print(f"Missing Runs: {missing_count}")
print(f"Corrupt Runs: {corrupt_count}")

# Save CSV without sequence lists
df_flat = df.drop(columns=["norm_scores_seq", "returns_seq", "dorm_seq", "erank_seq", "rho_seq", "gate_on_seq", "interv_active_seq"])
df_flat.to_csv(os.path.join(OUTPUT_DIR, "flat_results_150_runs.csv"), index=False)

# Helper statistical functions
def bootstrap_ci(arr, n_boot=10000, seed=42):
    arr = np.array(arr)
    if len(arr) == 0 or np.all(np.isnan(arr)):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    n = len(arr)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        boot_means[b] = np.mean(rng.choice(arr, size=n, replace=True))
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))

def exact_sign_test(diffs):
    diffs = np.array(diffs)
    pos = int(np.sum(diffs > 0))
    neg = int(np.sum(diffs < 0))
    n = pos + neg
    if n == 0:
        return 1.0, pos, neg
    k = min(pos, neg)
    res = stats.binomtest(k, n, 0.5, alternative='two-sided')
    return float(res.pvalue), pos, neg

def paired_wilcoxon(diffs):
    diffs = np.array(diffs)
    if np.all(diffs == 0) or np.all(diffs == diffs[0]):
        return 1.0
    try:
        res = stats.wilcoxon(diffs, zero_method='pratt')
        return float(res.pvalue)
    except Exception:
        return 1.0

def holm_bonferroni(pvals):
    pvals = np.array(pvals)
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

def rliable_iqm(scores, n_boot=10000, seed=42):
    scores = np.array(scores)
    iqm_val = stats.trim_mean(scores, proportiontocut=0.25)
    rng = np.random.default_rng(seed)
    n = len(scores)
    boot_iqms = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.choice(scores, size=n, replace=True)
        boot_iqms[b] = stats.trim_mean(sample, proportiontocut=0.25)
    ci_low = float(np.percentile(boot_iqms, 2.5))
    ci_high = float(np.percentile(boot_iqms, 97.5))
    return float(iqm_val), ci_low, ci_high

# -------------------------------------------------------------
# STEP 2 & 3: Results Summary and Statistical Comparisons
# -------------------------------------------------------------
summary_cells = []
paired_comparisons = []

# Group by env, shift, controller
for env in ENVS:
    for shift in SHIFTS:
        sub_all = df[(df["env"] == env) & (df["shift"] == shift)]
        
        # Get baseline 'none' controller scores by seed
        none_sub = sub_all[sub_all["controller"] == "none"].sort_values("seed")
        none_final = none_sub["final_score"].values
        none_last3 = none_sub["last3_score"].values
        
        ctrls_in_cell = sub_all["controller"].unique()
        for ctrl in ctrls_in_cell:
            ctrl_sub = sub_all[sub_all["controller"] == ctrl].sort_values("seed")
            final_vals = ctrl_sub["final_score"].values
            last3_vals = ctrl_sub["last3_score"].values
            pre_vals = ctrl_sub["pre_shift_score"].values
            post_vals = ctrl_sub["post_shift_score"].values
            deg_vals = ctrl_sub["degradation"].values
            rec_vals = ctrl_sub["recovery"].values
            
            applied_steps = ctrl_sub["applied_interventions"].values
            gate_trig_counts = ctrl_sub["gate_triggers_count"].values
            post_shift_dorms = ctrl_sub["post_shift_dorm"].values
            post_shift_eranks = ctrl_sub["post_shift_erank"].values
            post_shift_rhos = ctrl_sub["post_shift_rho"].values
            
            final_mean = float(np.mean(final_vals))
            final_std = float(np.std(final_vals, ddof=1)) if len(final_vals) > 1 else 0.0
            final_median = float(np.median(final_vals))
            final_iqm, iqm_low, iqm_high = rliable_iqm(final_vals)
            
            last3_mean = float(np.mean(last3_vals))
            last3_std = float(np.std(last3_vals, ddof=1)) if len(last3_vals) > 1 else 0.0
            
            deg_mean = float(np.mean(deg_vals))
            rec_mean = float(np.mean(rec_vals))
            
            summary_cells.append({
                "env": env,
                "shift": shift,
                "controller": ctrl,
                "n_seeds": len(final_vals),
                "final_mean": final_mean,
                "final_std": final_std,
                "final_median": final_median,
                "final_iqm": final_iqm,
                "final_iqm_ci": [iqm_low, iqm_high],
                "last3_mean": last3_mean,
                "last3_std": last3_std,
                "pre_shift_mean": float(np.mean(pre_vals)),
                "post_shift_mean": float(np.mean(post_vals)),
                "degradation_mean": deg_mean,
                "recovery_mean": rec_mean,
                "mean_applied_interventions": float(np.mean(applied_steps)),
                "mean_gate_triggers": float(np.mean(gate_trig_counts)),
                "mean_post_shift_dorm": float(np.nanmean(post_shift_dorms)),
                "mean_post_shift_erank": float(np.nanmean(post_shift_eranks)),
                "mean_post_shift_rho": float(np.nanmean(post_shift_rhos))
            })
            
            # Paired comparison vs 'none'
            if ctrl != "none" and len(final_vals) == len(none_final):
                diffs_final = final_vals - none_final
                diffs_last3 = last3_vals - none_last3
                
                mean_diff_final = float(np.mean(diffs_final))
                ci_final = bootstrap_ci(diffs_final)
                p_sign_final, pos_f, neg_f = exact_sign_test(diffs_final)
                p_wilc_final = paired_wilcoxon(diffs_final)
                
                mean_diff_last3 = float(np.mean(diffs_last3))
                ci_last3 = bootstrap_ci(diffs_last3)
                p_sign_last3, pos_l3, neg_l3 = exact_sign_test(diffs_last3)
                p_wilc_last3 = paired_wilcoxon(diffs_last3)
                
                paired_comparisons.append({
                    "env": env,
                    "shift": shift,
                    "comparison": f"{ctrl} vs none",
                    "target_ctrl": ctrl,
                    "ref_ctrl": "none",
                    "mean_diff_final": mean_diff_final,
                    "ci_final": ci_final,
                    "p_sign_final": p_sign_final,
                    "p_wilc_final": p_wilc_final,
                    "pos_neg_final": [pos_f, neg_f],
                    "mean_diff_last3": mean_diff_last3,
                    "ci_last3": ci_last3,
                    "p_sign_last3": p_sign_last3,
                    "p_wilc_last3": p_wilc_last3,
                    "pos_neg_last3": [pos_l3, neg_l3],
                    "diffs_final": [float(x) for x in diffs_final]
                })

        # Paired comparison: CapacityGate vs Fixed
        fixed_sub = sub_all[sub_all["controller"] == "fixed"].sort_values("seed")
        gate_sub = sub_all[sub_all["controller"] == "capacity_gate"].sort_values("seed")
        if len(fixed_sub) == 5 and len(gate_sub) == 5:
            diffs_gate_fixed = gate_sub["final_score"].values - fixed_sub["final_score"].values
            diffs_l3_gate_fixed = gate_sub["last3_score"].values - fixed_sub["last3_score"].values
            
            mean_diff = float(np.mean(diffs_gate_fixed))
            ci = bootstrap_ci(diffs_gate_fixed)
            p_sign, pos_g, neg_g = exact_sign_test(diffs_gate_fixed)
            p_wilc = paired_wilcoxon(diffs_gate_fixed)
            
            paired_comparisons.append({
                "env": env,
                "shift": shift,
                "comparison": "capacity_gate vs fixed",
                "target_ctrl": "capacity_gate",
                "ref_ctrl": "fixed",
                "mean_diff_final": mean_diff,
                "ci_final": ci,
                "p_sign_final": p_sign,
                "p_wilc_final": p_wilc,
                "pos_neg_final": [pos_g, neg_g],
                "mean_diff_last3": float(np.mean(diffs_l3_gate_fixed)),
                "ci_last3": bootstrap_ci(diffs_l3_gate_fixed),
                "p_sign_last3": exact_sign_test(diffs_l3_gate_fixed)[0],
                "p_wilc_last3": paired_wilcoxon(diffs_l3_gate_fixed),
                "pos_neg_last3": [exact_sign_test(diffs_l3_gate_fixed)[1], exact_sign_test(diffs_l3_gate_fixed)[2]],
                "diffs_final": [float(x) for x in diffs_gate_fixed]
            })

# Apply Holm-Bonferroni correction across all paired comparisons
p_wilc_final_list = [c["p_wilc_final"] for c in paired_comparisons]
p_wilc_last3_list = [c["p_wilc_last3"] for c in paired_comparisons]

holm_final = holm_bonferroni(p_wilc_final_list)
holm_last3 = holm_bonferroni(p_wilc_last3_list)

for idx, c in enumerate(paired_comparisons):
    c["holm_wilc_final"] = float(holm_final[idx])
    c["holm_wilc_last3"] = float(holm_last3[idx])

df_summary = pd.DataFrame(summary_cells)
df_summary.to_csv(os.path.join(OUTPUT_DIR, "summary_by_cell.csv"), index=False)

df_paired = pd.DataFrame(paired_comparisons)
df_paired.to_csv(os.path.join(OUTPUT_DIR, "paired_comparisons.csv"), index=False)

print("\nSummary and Paired Comparison tables created successfully.")

# -------------------------------------------------------------
# STEP 4 & 5: Diagnostic Gating Analysis
# -------------------------------------------------------------
gate_analysis = []
for row in rows:
    gate_records = row["gate_records"]
    if not gate_records:
        continue
    
    # Detailed gate inspection
    dorm_vals = [r.get("dormancy", np.nan) for r in gate_records]
    erank_vals = [r.get("effective_rank", np.nan) for r in gate_records]
    rho_vals = [r.get("rho", np.nan) for r in gate_records]
    steps = [r.get("step", 0) for r in gate_records]
    gate_ons = [r.get("gate_on", False) for r in gate_records]
    
    # Check trigger conditions: dorm >= dorm_on (0.15) OR rho <= rho_on (0.7)
    dorm_ons = [r.get("dorm_on", 0.15) for r in gate_records]
    rho_ons = [r.get("rho_on", 0.7) for r in gate_records]
    
    dorm_trig_count = sum(1 for d, d_on in zip(dorm_vals, dorm_ons) if d >= d_on)
    rho_trig_count = sum(1 for r, r_on in zip(rho_vals, rho_ons) if r <= r_on)
    
    max_dorm = float(np.nanmax(dorm_vals))
    min_dorm = float(np.nanmin(dorm_vals))
    max_erank = float(np.nanmax(erank_vals))
    min_erank = float(np.nanmin(erank_vals))
    min_rho = float(np.nanmin(rho_vals))
    max_rho = float(np.nanmax(rho_vals))
    
    gate_analysis.append({
        "env": row["env"],
        "shift": row["shift"],
        "controller": row["controller"],
        "seed": row["seed"],
        "applied_interventions": row["applied_interventions"],
        "gate_triggers_count": row["gate_triggers_count"],
        "gate_triggers_post_shift": row["gate_triggers_post_shift"],
        "dorm_trig_count": dorm_trig_count,
        "rho_trig_count": rho_trig_count,
        "max_dormancy": max_dorm,
        "min_dormancy": min_dorm,
        "min_effective_rank": min_erank,
        "max_effective_rank": max_erank,
        "min_rho": min_rho,
        "max_rho": max_rho,
        "rank_anchor": row["rank_anchor"]
    })

df_gate = pd.DataFrame(gate_analysis)
df_gate.to_csv(os.path.join(OUTPUT_DIR, "gate_diagnostics_detail.csv"), index=False)

print("Gate diagnostics detailed table created.")

# Print key findings from gate analysis
print("\n=== GATE ACTIVATION SUMMARY ===")
gate_summary_by_ctrl_shift = df_gate.groupby(["env", "shift", "controller"])[["applied_interventions", "gate_triggers_count", "max_dormancy", "min_rho"]].mean()
print(gate_summary_by_ctrl_shift.to_string())

# -------------------------------------------------------------
# STEP 6: Visualizations (Plots)
# -------------------------------------------------------------

# Plot 1: Learning Curves by Env and Shift
eval_steps = [2500 * (i + 1) for i in range(10)]
colors = {"none": "#333333", "fixed": "#d95f02", "capacity_gate": "#7570b3", "m10ref": "#e7298a"}
styles = {"none": "-", "fixed": "--", "capacity_gate": "-.", "m10ref": ":"}

fig, axes = plt.subplots(len(ENVS), len(SHIFTS), figsize=(15, 12), sharex=True, sharey='row')
for i, env in enumerate(ENVS):
    for j, shift in enumerate(SHIFTS):
        ax = axes[i, j]
        sub = df[(df["env"] == env) & (df["shift"] == shift)]
        for ctrl in sub["controller"].unique():
            ctrl_sub = sub[sub["controller"] == ctrl]
            curves = np.array(ctrl_sub["norm_scores_seq"].tolist())
            mean_curve = np.mean(curves, axis=0)
            std_curve = np.std(curves, axis=0) / np.sqrt(len(curves))
            
            ax.plot(eval_steps, mean_curve, label=ctrl, color=colors.get(ctrl, "gray"), linestyle=styles.get(ctrl, "-"), linewidth=2)
            ax.fill_between(eval_steps, mean_curve - std_curve, mean_curve + std_curve, color=colors.get(ctrl, "gray"), alpha=0.15)
            
        ax.axvline(x=5000, color='red', linestyle=':', alpha=0.7, label='Shift Step' if i==0 and j==0 else "")
        if i == 0:
            ax.set_title(f"Shift: {shift.upper()}", fontsize=12, fontweight='bold')
        if j == 0:
            ax.set_ylabel(f"{env}\nNormalized Score", fontsize=11, fontweight='bold')
        if i == len(ENVS) - 1:
            ax.set_xlabel("Online Steps", fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.5)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=5, fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(os.path.join(PLOTS_DIR, "learning_curves_all.png"), dpi=300)
plt.close()

# Plot 2: Dormancy & Effective Rank Trajectories over Steps
fig, axes = plt.subplots(len(ENVS), 2, figsize=(14, 10), sharex=True)
gate_steps = [1000 * i for i in range(1, 26)]

for i, env in enumerate(ENVS):
    # Left: Dormancy
    ax_d = axes[i, 0]
    sub_d = df[df["env"] == env]
    for shift in SHIFTS:
        shift_sub = sub_d[sub_d["shift"] == shift]
        dorm_curves = np.array(shift_sub["dorm_seq"].tolist())
        mean_dorm = np.nanmean(dorm_curves, axis=0)
        ax_d.plot(gate_steps, mean_dorm, label=f"{shift}", linewidth=2)
    ax_d.axhline(y=0.15, color='r', linestyle='--', label='dorm_on (0.15)')
    ax_d.axvline(x=5000, color='gray', linestyle=':', alpha=0.7)
    ax_d.set_ylabel(f"{env}\nDormancy", fontsize=11, fontweight='bold')
    ax_d.grid(True, linestyle='--', alpha=0.5)
    if i == 0: ax_d.set_title("Dormancy Trajectory", fontsize=12, fontweight='bold')
    if i == len(ENVS) - 1: ax_d.set_xlabel("Online Steps", fontsize=11)

    # Right: Effective Rank
    ax_r = axes[i, 1]
    for shift in SHIFTS:
        shift_sub = sub_d[sub_d["shift"] == shift]
        erank_curves = np.array(shift_sub["erank_seq"].tolist())
        mean_erank = np.nanmean(erank_curves, axis=0)
        ax_r.plot(gate_steps, mean_erank, label=f"{shift}", linewidth=2)
    ax_r.axvline(x=5000, color='gray', linestyle=':', alpha=0.7)
    ax_r.set_ylabel(f"{env}\nEffective Rank", fontsize=11, fontweight='bold')
    ax_r.grid(True, linestyle='--', alpha=0.5)
    if i == 0: ax_r.set_title("Effective Rank Trajectory", fontsize=12, fontweight='bold')
    if i == len(ENVS) - 1: ax_r.set_xlabel("Online Steps", fontsize=11)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=4, fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(PLOTS_DIR, "diagnostic_trajectories.png"), dpi=300)
plt.close()

# Plot 3: Paired Differences (Fixed - None, CapacityGate - None)
fig, ax = plt.subplots(figsize=(12, 6))
paired_primary = [p for p in paired_comparisons if p["comparison"] in ["fixed vs none", "capacity_gate vs none"]]

y_labels = []
mean_diffs = []
ci_lows = []
ci_highs = []
colors_list = []

for p in paired_primary:
    lbl = f"{p['env'].split('-')[0]} | {p['shift']} | {p['comparison']}"
    y_labels.append(lbl)
    mean_diffs.append(p["mean_diff_final"])
    ci_lows.append(p["ci_final"][0])
    ci_highs.append(p["ci_final"][1])
    colors_list.append("#d95f02" if "fixed" in p["comparison"] else "#7570b3")

y_pos = np.arange(len(y_labels))
err_low = np.array(mean_diffs) - np.array(ci_lows)
err_high = np.array(ci_highs) - np.array(mean_diffs)

ax.errorbar(mean_diffs, y_pos, xerr=[err_low, err_high], fmt='o', color='black', ecolor=colors_list, elinewidth=2, capsize=4, markersize=7)
ax.axvline(x=0.0, color='red', linestyle='--', alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(y_labels, fontsize=10)
ax.set_xlabel("Mean Score Difference vs None (95% Bootstrap CI)", fontsize=11, fontweight='bold')
ax.set_title("Paired Difference in Final Score (Step 25,000)", fontsize=12, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "paired_differences_ci.png"), dpi=300)
plt.close()

print("\nSaved all plots to results/final_study_v2_analysis/plots/")
