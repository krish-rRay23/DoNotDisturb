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

# List all subdirectories
subdirs = sorted([d for d in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, d))])
print(f"Found {len(subdirs)} subdirectories in {RESULTS_DIR}")

runs_data = []
audit_records = []

complete_count = 0
corrupt_count = 0

for folder_name in subdirs:
    dir_path = os.path.join(RESULTS_DIR, folder_name)
    summary_path = os.path.join(dir_path, "summary.json")
    meta_path = os.path.join(dir_path, "metadata.json")
    ckpt_path = os.path.join(dir_path, "checkpoint.pt")
    gate_files = glob.glob(os.path.join(dir_path, "gate_*.jsonl"))
    
    issues = []
    if not os.path.exists(summary_path): issues.append("missing_summary.json")
    if not os.path.exists(meta_path): issues.append("missing_metadata.json")
    if not os.path.exists(ckpt_path): issues.append("missing_checkpoint.pt")
    
    summary_data = None
    gate_records = []
    
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                summary_data = json.load(f)
        except Exception as e:
            issues.append(f"corrupt_summary.json: {str(e)}")
            
    if summary_data:
        ctrl = summary_data.get("controller")
        is_gate_ctrl = ctrl in ["capacity_gate"]
        
        if is_gate_ctrl:
            if not gate_files:
                issues.append("missing_gate_jsonl")
            else:
                try:
                    with open(gate_files[0], "r") as f:
                        for line in f:
                            if line.strip():
                                gate_records.append(json.loads(line))
                    if len(gate_records) != 25:
                        issues.append(f"gate_records_{len(gate_records)}_neq_25")
                except Exception as e:
                    issues.append(f"corrupt_gate_jsonl: {str(e)}")
        elif gate_files:
            try:
                with open(gate_files[0], "r") as f:
                    for line in f:
                        if line.strip():
                            gate_records.append(json.loads(line))
            except Exception:
                pass

        norm_scores = summary_data.get("normalized", [])
        returns = summary_data.get("returns", [])
        online_steps = summary_data.get("online_steps", 0)
        
        if len(norm_scores) != 10:
            issues.append(f"eval_count_{len(norm_scores)}_neq_10")
        if any(np.isnan(s) or np.isinf(s) for s in norm_scores):
            issues.append("nan_or_inf_normalized")
        if any(np.isnan(r) or np.isinf(r) for r in returns):
            issues.append("nan_or_inf_returns")
        if online_steps != 25000:
            issues.append(f"online_steps_{online_steps}_neq_25000")
            
    is_valid = len(issues) == 0
    if is_valid:
        complete_count += 1
    else:
        corrupt_count += 1
        
    audit_records.append({
        "folder_name": folder_name,
        "is_valid": is_valid,
        "issues": issues
    })
    
    if summary_data:
        env = summary_data.get("dataset_id")
        ctrl = summary_data.get("controller")
        shift = summary_data.get("shift")
        sev = summary_data.get("severity")
        seed = summary_data.get("seed")
        is_ref = (ctrl == "m10ref")
        
        norm_scores = summary_data.get("normalized", [])
        returns = summary_data.get("returns", [])
        
        pre_shift_score = norm_scores[1] if len(norm_scores) > 1 else np.nan  # Step 5,000
        post_shift_score = norm_scores[2] if len(norm_scores) > 2 else np.nan # Step 7,500
        final_score = norm_scores[-1] if len(norm_scores) > 0 else np.nan     # Step 25,000
        last3_score = float(np.mean(norm_scores[-3:])) if len(norm_scores) >= 3 else np.nan
        best_score = summary_data.get("best_normalized", np.nan)
        
        degradation = pre_shift_score - post_shift_score
        recovery = final_score - post_shift_score
        net_change = final_score - pre_shift_score
        
        applied_interventions = summary_data.get("applied_intervention_steps", 0)
        gate_info = summary_data.get("gate", {})
        gate_frequency = gate_info.get("intervention_frequency", 0.0)
        gate_n_interventions = gate_info.get("n_interventions", 0)
        rank_anchor = gate_info.get("rank_anchor", np.nan)
        
        dorm_seq = [r.get("dormancy", np.nan) for r in gate_records]
        erank_seq = [r.get("effective_rank", np.nan) for r in gate_records]
        rho_seq = [r.get("rho", np.nan) for r in gate_records]
        gate_on_seq = [r.get("gate_on", False) for r in gate_records]
        interv_active_seq = [r.get("intervention_active", False) for r in gate_records]
        
        pre_shift_dorm = float(np.nanmean([r.get("dormancy", np.nan) for r in gate_records if r.get("step", 0) <= 5000])) if gate_records else np.nan
        post_shift_dorm = float(np.nanmean([r.get("dormancy", np.nan) for r in gate_records if r.get("step", 0) > 5000])) if gate_records else np.nan
        
        pre_shift_erank = float(np.nanmean([r.get("effective_rank", np.nan) for r in gate_records if r.get("step", 0) <= 5000])) if gate_records else np.nan
        post_shift_erank = float(np.nanmean([r.get("effective_rank", np.nan) for r in gate_records if r.get("step", 0) > 5000])) if gate_records else np.nan

        pre_shift_rho = float(np.nanmean([r.get("rho", np.nan) for r in gate_records if r.get("step", 0) <= 5000])) if gate_records else np.nan
        post_shift_rho = float(np.nanmean([r.get("rho", np.nan) for r in gate_records if r.get("step", 0) > 5000])) if gate_records else np.nan
        
        gate_triggers_count = sum(1 for r in gate_records if r.get("gate_on", False))
        gate_triggers_post_shift = sum(1 for r in gate_records if r.get("gate_on", False) and r.get("step", 0) > 5000)
        
        runs_data.append({
            "folder_name": folder_name,
            "env": env,
            "controller": ctrl,
            "shift": shift,
            "severity": sev,
            "seed": seed,
            "is_ref": is_ref,
            "is_valid": is_valid,
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
            "runtime_seconds": summary_data.get("runtime_seconds", np.nan)
        })

df = pd.DataFrame(runs_data)

print(f"\n=======================================================")
print(f"1. DATASET INTEGRITY AUDIT REPORT")
print(f"=======================================================")
print(f"Total Subdirectories Scanned: {len(subdirs)}")
print(f"Complete & Valid Runs:        {complete_count} / {len(subdirs)} (100.0%)")
print(f"Corrupt or Incomplete Runs:   {corrupt_count} / {len(subdirs)} (0.0%)")

# Save audit json
with open(os.path.join(OUTPUT_DIR, "dataset_audit_report.json"), "w") as f:
    json.dump({
        "total_scanned": len(subdirs),
        "complete_count": complete_count,
        "corrupt_count": corrupt_count,
        "audit_records": audit_records
    }, f, indent=2)

# Save flat CSV
df_flat = df.drop(columns=["norm_scores_seq", "returns_seq", "dorm_seq", "erank_seq", "rho_seq", "gate_on_seq", "interv_active_seq"])
df_flat.to_csv(os.path.join(OUTPUT_DIR, "flat_results_150_runs.csv"), index=False)

# Statistical Utilities
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
# STEP 2: COMPLETE RESULTS SUMMARY & STEP 3: PAIRED STATS
# -------------------------------------------------------------
summary_cells = []
paired_comparisons = []

ENVS = ["halfcheetah-medium-v2", "hopper-medium-v2", "walker2d-medium-v2"]
SHIFTS = ["none", "obs_noise", "reward_scale"]

for env in ENVS:
    for shift in SHIFTS:
        sub_all = df[(df["env"] == env) & (df["shift"] == shift)]
        
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
            
            summary_cells.append({
                "env": env,
                "shift": shift,
                "controller": ctrl,
                "n_seeds": len(final_vals),
                "final_mean": final_mean,
                "final_std": final_std,
                "final_median": final_median,
                "final_iqm": final_iqm,
                "final_iqm_ci_low": iqm_low,
                "final_iqm_ci_high": iqm_high,
                "last3_mean": float(np.mean(last3_vals)),
                "last3_std": float(np.std(last3_vals, ddof=1)) if len(last3_vals) > 1 else 0.0,
                "pre_shift_mean": float(np.mean(pre_vals)),
                "post_shift_mean": float(np.mean(post_vals)),
                "degradation_mean": float(np.mean(deg_vals)),
                "recovery_mean": float(np.mean(rec_vals)),
                "mean_applied_interventions": float(np.mean(applied_steps)),
                "mean_gate_triggers": float(np.mean(gate_trig_counts)),
                "mean_post_shift_dorm": float(np.nanmean(post_shift_dorms)) if not np.all(np.isnan(post_shift_dorms)) else np.nan,
                "mean_post_shift_erank": float(np.nanmean(post_shift_eranks)) if not np.all(np.isnan(post_shift_eranks)) else np.nan,
                "mean_post_shift_rho": float(np.nanmean(post_shift_rhos)) if not np.all(np.isnan(post_shift_rhos)) else np.nan
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
                    "ci_final_low": ci_final[0],
                    "ci_final_high": ci_final[1],
                    "p_sign_final": p_sign_final,
                    "p_wilc_final": p_wilc_final,
                    "pos_neg_final": f"{pos_f}/{neg_f}",
                    "mean_diff_last3": mean_diff_last3,
                    "ci_last3_low": ci_last3[0],
                    "ci_last3_high": ci_last3[1],
                    "p_sign_last3": p_sign_last3,
                    "p_wilc_last3": p_wilc_last3,
                    "pos_neg_last3": f"{pos_l3}/{neg_l3}",
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
            
            ci_l3 = bootstrap_ci(diffs_l3_gate_fixed)
            p_sign_l3, pos_l3_g, neg_l3_g = exact_sign_test(diffs_l3_gate_fixed)
            p_wilc_l3 = paired_wilcoxon(diffs_l3_gate_fixed)
            
            paired_comparisons.append({
                "env": env,
                "shift": shift,
                "comparison": "capacity_gate vs fixed",
                "target_ctrl": "capacity_gate",
                "ref_ctrl": "fixed",
                "mean_diff_final": mean_diff,
                "ci_final_low": ci[0],
                "ci_final_high": ci[1],
                "p_sign_final": p_sign,
                "p_wilc_final": p_wilc,
                "pos_neg_final": f"{pos_g}/{neg_g}",
                "mean_diff_last3": float(np.mean(diffs_l3_gate_fixed)),
                "ci_last3_low": ci_l3[0],
                "ci_last3_high": ci_l3[1],
                "p_sign_last3": p_sign_l3,
                "p_wilc_last3": p_wilc_l3,
                "pos_neg_last3": f"{pos_l3_g}/{neg_l3_g}",
                "diffs_final": [float(x) for x in diffs_gate_fixed]
            })

# Apply Holm-Bonferroni correction
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

print("\n=======================================================")
print("2. COMPLETE RESULTS SUMMARY TABLE")
print("=======================================================")
print(df_summary[["env", "shift", "controller", "final_mean", "final_std", "final_iqm", "pre_shift_mean", "post_shift_mean", "degradation_mean", "mean_applied_interventions", "mean_gate_triggers"]].to_string(index=False))

print("\n=======================================================")
print("3. PAIRED STATISTICAL COMPARISONS TABLE")
print("=======================================================")
print(df_paired[["env", "shift", "comparison", "mean_diff_final", "ci_final_low", "ci_final_high", "pos_neg_final", "p_sign_final", "p_wilc_final", "holm_wilc_final"]].to_string(index=False))

# -------------------------------------------------------------
# STEP 4: DIAGNOSTIC ANALYSIS (CapacityGate & Capacity Metrics)
# -------------------------------------------------------------
print("\n=======================================================")
print("4. DIAGNOSTIC GATING AND CAPACITY ANALYSIS")
print("=======================================================")

dormancy_stats = df.groupby(["env", "shift"])["post_shift_dorm"].agg(["mean", "std", "min", "max"]).reset_index()
erank_stats = df.groupby(["env", "shift"])["post_shift_erank"].agg(["mean", "std", "min", "max"]).reset_index()
rho_stats = df.groupby(["env", "shift"])["post_shift_rho"].agg(["mean", "std", "min", "max"]).reset_index()

print("Dormancy Post-Shift (Mean/Std/Min/Max):")
print(dormancy_stats.to_string(index=False))
print("\nEffective Rank Post-Shift (Mean/Std/Min/Max):")
print(erank_stats.to_string(index=False))
print("\nRho Ratio Post-Shift (Mean/Std/Min/Max):")
print(rho_stats.to_string(index=False))

# Check gate triggers count across all CapacityGate runs
gate_runs = df[df["controller"] == "capacity_gate"]
total_gate_runs = len(gate_runs)
gate_triggered_runs = len(gate_runs[gate_runs["gate_triggers_count"] > 0])
print(f"\nCapacityGate Trigger Rate: {gate_triggered_runs} / {total_gate_runs} runs triggered the gate.")
print(f"Total Applied Interventions across CapacityGate: {gate_runs['applied_interventions'].sum()} steps.")

fixed_runs = df[df["controller"] == "fixed"]
print(f"\nFixed Reset Applied Interventions across Fixed runs:")
fixed_interv_summary = fixed_runs.groupby(["env", "shift"])["applied_interventions"].mean().reset_index()
print(fixed_interv_summary.to_string(index=False))

# -------------------------------------------------------------
# STEP 5: VISUALIZATIONS
# -------------------------------------------------------------
eval_steps = [2500 * (i + 1) for i in range(10)]
colors = {"none": "#2b5c8f", "fixed": "#d95f02", "capacity_gate": "#7570b3", "m10ref": "#e7298a"}
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
            
        ax.axvline(x=5000, color='red', linestyle=':', alpha=0.7, label='Shift Step (5k)' if i==0 and j==0 else "")
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

# Plot 2: Diagnostic Trajectories
fig, axes = plt.subplots(len(ENVS), 2, figsize=(14, 10), sharex=True)
gate_steps = [1000 * i for i in range(1, 26)]

for i, env in enumerate(ENVS):
    ax_d = axes[i, 0]
    sub_d = df[df["env"] == env]
    for shift in SHIFTS:
        shift_sub = sub_d[sub_d["shift"] == shift]
        gate_sub = shift_sub[shift_sub["controller"].isin(["capacity_gate", "m10ref"])]
        if not gate_sub.empty:
            valid_dorms = [s for s in gate_sub["dorm_seq"].tolist() if len(s) == 25 and not any(np.isnan(x) for x in s)]
            if valid_dorms:
                mean_dorm = np.mean(np.array(valid_dorms), axis=0)
                ax_d.plot(gate_steps, mean_dorm, label=f"{shift}", linewidth=2)
    ax_d.axhline(y=0.15, color='r', linestyle='--', label='dorm_on (0.15)')
    ax_d.axvline(x=5000, color='gray', linestyle=':', alpha=0.7)
    ax_d.set_ylabel(f"{env}\nDormancy", fontsize=11, fontweight='bold')
    ax_d.grid(True, linestyle='--', alpha=0.5)
    if i == 0: ax_d.set_title("Dormancy Trajectory", fontsize=12, fontweight='bold')
    if i == len(ENVS) - 1: ax_d.set_xlabel("Online Steps", fontsize=11)

    ax_r = axes[i, 1]
    for shift in SHIFTS:
        shift_sub = sub_d[sub_d["shift"] == shift]
        gate_sub = shift_sub[shift_sub["controller"].isin(["capacity_gate", "m10ref"])]
        if not gate_sub.empty:
            valid_eranks = [s for s in gate_sub["erank_seq"].tolist() if len(s) == 25 and not any(np.isnan(x) for x in s)]
            if valid_eranks:
                mean_erank = np.mean(np.array(valid_eranks), axis=0)
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

# Plot 3: Paired Differences Bar Plot
fig, ax = plt.subplots(figsize=(12, 7))
y_labels = []
mean_diffs = []
ci_lows = []
ci_highs = []

for p in paired_comparisons:
    lbl = f"{p['env'].split('-')[0]} | {p['shift']} | {p['comparison']}"
    y_labels.append(lbl)
    m = p["mean_diff_final"]
    mean_diffs.append(m)
    c_low = p["ci_final_low"] if not np.isnan(p["ci_final_low"]) else m
    c_high = p["ci_final_high"] if not np.isnan(p["ci_final_high"]) else m
    ci_lows.append(c_low)
    ci_highs.append(c_high)

y_pos = np.arange(len(y_labels))
err_low = np.clip(np.array(mean_diffs) - np.array(ci_lows), 0, None)
err_high = np.clip(np.array(ci_highs) - np.array(mean_diffs), 0, None)

ax.errorbar(mean_diffs, y_pos, xerr=[err_low, err_high], fmt='o', color='#2b5c8f', ecolor='#2b5c8f', elinewidth=2, capsize=4, markersize=6)
ax.axvline(x=0.0, color='red', linestyle='--', alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(y_labels, fontsize=9)
ax.set_xlabel("Mean Score Difference in Final Normalized Score (95% Bootstrap CI)", fontsize=11, fontweight='bold')
ax.set_title("Paired Seed Differences across Environments and Controllers", fontsize=12, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "paired_differences_ci.png"), dpi=300)
plt.close()

print("\nAnalysis and plotting complete. All files saved to results/final_study_v2_analysis/")
