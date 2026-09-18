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
os.makedirs(OUTPUT_DIR, exist_ok=True)

ENVS = ["halfcheetah-medium-v2", "hopper-medium-v2", "walker2d-medium-v2"]
SHIFTS = ["none", "obs_noise", "reward_scale"]
PRIMARY_CONTROLLERS = ["none", "fixed", "capacity_gate"]
SEEDS = [0, 1, 2, 3, 4]

# Step 1: Audit Dataset Integrity
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

print(f"Total expected run directories: {len(all_expected)} (135 primary + 15 reference)")

missing_dirs = []
corrupt_runs = []
complete_runs = []
runs_data = []

for env, ctrl, shift, sev, seed, folder_name, is_ref in all_expected:
    dir_path = os.path.join(RESULTS_DIR, folder_name)
    if not os.path.exists(dir_path):
        missing_dirs.append(folder_name)
        continue
    
    summary_path = os.path.join(dir_path, "summary.json")
    meta_path = os.path.join(dir_path, "metadata.json")
    ckpt_path = os.path.join(dir_path, "checkpoint.pt")
    
    gate_pattern = os.path.join(dir_path, "gate_*.jsonl")
    gate_files = glob.glob(gate_pattern)
    
    has_summary = os.path.exists(summary_path)
    has_meta = os.path.exists(meta_path)
    has_ckpt = os.path.exists(ckpt_path)
    has_gate = len(gate_files) > 0
    
    issues = []
    if not has_summary: issues.append("missing_summary.json")
    if not has_meta: issues.append("missing_metadata.json")
    if not has_ckpt: issues.append("missing_checkpoint.pt")
    if not has_gate: issues.append("missing_gate_jsonl")
    
    summary_data = None
    gate_records = []
    
    if has_summary:
        try:
            with open(summary_path, "r") as f:
                summary_data = json.load(f)
        except Exception as e:
            issues.append(f"corrupt_summary.json: {str(e)}")
            
    if has_gate:
        try:
            with open(gate_files[0], "r") as f:
                for line in f:
                    if line.strip():
                        gate_records.append(json.loads(line))
        except Exception as e:
            issues.append(f"corrupt_gate_jsonl: {str(e)}")
            
    if summary_data:
        norm_scores = summary_data.get("normalized", [])
        returns = summary_data.get("returns", [])
        online_steps = summary_data.get("online_steps", 0)
        
        if len(norm_scores) != 10:
            issues.append(f"incomplete_evals_count: {len(norm_scores)}/10")
        if any(np.isnan(s) or np.isinf(s) for s in norm_scores):
            issues.append("nan_or_inf_in_normalized_scores")
        if any(np.isnan(r) or np.isinf(r) for r in returns):
            issues.append("nan_or_inf_in_returns")
        if online_steps != 25000:
            issues.append(f"incomplete_online_steps: {online_steps}/25000")
            
    if gate_records:
        if len(gate_records) != 25:
            issues.append(f"incomplete_gate_records: {len(gate_records)}/25")
            
    if issues:
        corrupt_runs.append((folder_name, issues))
    else:
        complete_runs.append(folder_name)
        
    runs_data.append({
        "env": env,
        "controller": ctrl,
        "shift": shift,
        "severity": sev,
        "seed": seed,
        "folder_name": folder_name,
        "is_ref": is_ref,
        "summary": summary_data,
        "gate_records": gate_records,
        "issues": issues
    })

print(f"Audit Summary:")
print(f"  Complete Runs: {len(complete_runs)} / {len(all_expected)}")
print(f"  Missing Directories: {len(missing_dirs)}")
print(f"  Corrupt/Incomplete Runs: {len(corrupt_runs)}")
if corrupt_runs:
    for c, iss in corrupt_runs:
        print(f"    - {c}: {iss}")

with open(os.path.join(OUTPUT_DIR, "dataset_audit_report.json"), "w") as f:
    json.dump({
        "total_expected": len(all_expected),
        "complete_count": len(complete_runs),
        "missing_dirs": missing_dirs,
        "corrupt_runs": corrupt_runs
    }, f, indent=2)

print("\nSaved audit report to dataset_audit_report.json")
