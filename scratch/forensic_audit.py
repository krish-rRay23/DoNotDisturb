import glob
import json
import numpy as np

# 1. Inspect gate logs around step 3000-6000 across all capacity_gate runs
cg_files = glob.glob('results/stress_study/STRESS-*capacity_gate*/gate_*.jsonl')
print(f"Found {len(cg_files)} capacity_gate logs.")

for gf in sorted(cg_files):
    name = gf.split('STRESS-')[1].split('/')[0].split('\\')[0]
    print(f"\n--- {name} ---")
    with open(gf) as f:
        for line in f:
            d = json.loads(line)
            step = d['step']
            if step in (3000, 4000, 5000, 6000):
                print(f"  Step {step:5d} | shift_active={d.get('shift_active')} | gate_on={d.get('gate_on')} | on_dorm={d.get('online_dormancy', 0):.3f} | off_dorm={d.get('offline_dormancy', 0):.3f} | on_rho={d.get('online_rho', 0):.3f} | int_count={d.get('intervention_count')}")

