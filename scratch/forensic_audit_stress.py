import json
import glob
import os
import numpy as np
import pandas as pd

print("=== 1. AUDIT OF CAPACITY GATE TELEMETRY IN STRESS STUDY ===")
gate_paths = sorted(glob.glob("results/stress_study/STRESS-*capacity_gate*/gate_*.jsonl"))
print(f"Found {len(gate_paths)} gate record files.")

for p in gate_paths:
    cell_name = os.path.basename(os.path.dirname(p))
    print(f"\n=======================================================")
    print(f"--- {cell_name} ---")
    with open(p, "r") as f:
        records = [json.loads(line) for line in f]
    
    first_trigger = None
    for r in records:
        step = r.get("step")
        gate_on = r.get("gate_on")
        on_dorm = r.get("online_dormancy", r.get("dormancy"))
        on_rho = r.get("online_rho", r.get("rho"))
        off_dorm = r.get("offline_dormancy", 0.0)
        off_rho = r.get("offline_rho", 0.0)
        shift_act = r.get("shift_active")
        n_int = r.get("n_interventions")
        
        if gate_on and first_trigger is None:
            first_trigger = (step, on_dorm, on_rho, shift_act)
            print(f"--> FIRST TRIGGER at step {step}: shift_active={shift_act}, on_dorm={on_dorm:.4f}, on_rho={on_rho:.4f}, off_dorm={off_dorm:.4f}, off_rho={off_rho:.4f}")
        
        print(f"step={step:5d} | shift={str(shift_act):5s} | on_dorm={on_dorm:.4f} | on_rho={on_rho:.4f} | off_dorm={off_dorm:.4f} | off_rho={off_rho:.4f} | gate_on={str(gate_on):5s} | n_int={n_int}")

print("\n=== 2. SEED-LEVEL PERFORMANCE TABLE (ALL 24 RUNS) ===")
manifest_path = "results/stress_study/STRESS_manifest.jsonl"
rows = []
with open(manifest_path, "r") as f:
    for line in f:
        d = json.loads(line)
        env = d["dataset_id"]
        ctrl = d["controller"]
        seed = d["seed"]
        score = d["normalized"][-1]  # final evaluation score
        int_count = d.get("applied_intervention_steps", 0)
        rows.append({"env": env, "controller": ctrl, "seed": seed, "final_score": score, "interventions": int_count})

df = pd.DataFrame(rows)
piv_score = df.pivot(index=["env", "seed"], columns="controller", values="final_score")
piv_int = df.pivot(index=["env", "seed"], columns="controller", values="interventions")

print("\n--- FINAL EVALUATION SCORE BY SEED ---")
print(piv_score.to_string())

print("\n--- APPLIED INTERVENTIONS BY SEED ---")
print(piv_int.to_string())

print("\n=== 3. SUMMARY BY ENV AND CONTROLLER ===")
summary = df.groupby(["env", "controller"]).agg(
    mean_score=("final_score", "mean"),
    std_score=("final_score", "std"),
    median_score=("final_score", "median"),
    mean_int=("interventions", "mean")
)
print(summary.to_string())
