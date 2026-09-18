import json, glob, os
import pandas as pd
import numpy as np

print("=== COUNTERFACTUAL ANALYSIS OF GATE TURN-OFF AND INTERVENTION LOGIC ===")

# Let's inspect the exact values in HalfCheetah and Walker2d runs post-trigger
gate_paths = sorted(glob.glob("results/stress_study/STRESS-*capacity_gate*/gate_*.jsonl"))

for p in gate_paths:
    cell_name = os.path.basename(os.path.dirname(p))
    with open(p, "r") as f:
        records = [json.loads(line) for line in f]
    
    print(f"\n=======================================================")
    print(f"--- {cell_name} ---")
    
    # Analyze why the gate stayed on
    # Default condition to turn off: dorm <= 0.05 AND rho >= 0.85
    # Let's test alternative turn-off rules:
    # Rule A (Baseline): dorm <= 0.05 and rho >= 0.85
    # Rule B: dorm <= 0.12 and rho >= 0.85
    # Rule C: dorm <= 0.15 and rho >= 0.80
    # Rule D: rank-only recovery: rho >= 0.85 (regardless of dorm)
    # Rule E: Single-shot / Refractory cooldown (e.g. at most 1 intervention per 5000 steps)
    
    records_df = pd.DataFrame(records)
    print("Post-trigger summary statistics (from trigger step onward):")
    trig_steps = [r["step"] for r in records if r.get("gate_on")]
    if not trig_steps:
        print("  Never triggered!")
        continue
    first_trig = min(trig_steps)
    post = records_df[records_df["step"] >= first_trig]
    
    print(f"  First trigger step: {first_trig}")
    print(f"  Online dormancy range: min={post['online_dormancy'].min():.4f}, mean={post['online_dormancy'].mean():.4f}, median={post['online_dormancy'].median():.4f}, max={post['online_dormancy'].max():.4f}")
    print(f"  Online rho range:      min={post['online_rho'].min():.4f}, mean={post['online_rho'].mean():.4f}, median={post['online_rho'].median():.4f}, max={post['online_rho'].max():.4f}")
    print(f"  Offline dormancy range: min={post['offline_dormancy'].min():.4f}, mean={post['offline_dormancy'].mean():.4f}, median={post['offline_dormancy'].median():.4f}, max={post['offline_dormancy'].max():.4f}")
    print(f"  Fraction of post-trigger steps where dorm <= 0.05: {(post['online_dormancy'] <= 0.05).mean():.2%}")
    print(f"  Fraction of post-trigger steps where dorm <= 0.10: {(post['online_dormancy'] <= 0.10).mean():.2%}")
    print(f"  Fraction of post-trigger steps where dorm <= 0.15: {(post['online_dormancy'] <= 0.15).mean():.2%}")
    print(f"  Fraction of post-trigger steps where rho >= 0.85:   {(post['online_rho'] >= 0.85).mean():.2%}")

