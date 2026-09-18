import json, glob, os
import pandas as pd

print("=== SIMULATION OF ALTERNATIVE GATE LOGIC ON EXISTING TELEMETRY ===")

gate_paths = sorted(glob.glob("results/stress_study/STRESS-*capacity_gate*/gate_*.jsonl"))

def simulate_gate(records, dorm_on=0.15, dorm_off=0.05, rho_on=0.70, rho_off=0.85, cooldown=0, rank_only_off=False):
    gate_on = False
    interventions = 0
    last_int_step = -99999
    
    events = []
    
    for r in records:
        step = r["step"]
        dorm = r.get("online_dormancy", r.get("dormancy"))
        rho = r.get("online_rho", r.get("rho"))
        
        should_turn_on = (dorm >= dorm_on) or (rho <= rho_on)
        if rank_only_off:
            should_turn_off = (rho >= rho_off)
        else:
            should_turn_off = (dorm <= dorm_off) and (rho >= rho_off)
            
        if not gate_on:
            if should_turn_on and (step - last_int_step >= cooldown):
                gate_on = True
                interventions += 1
                last_int_step = step
                events.append((step, "ON", dorm, rho))
        else:
            if should_turn_off:
                gate_on = False
                events.append((step, "OFF", dorm, rho))
            else:
                # Still on
                if step - last_int_step >= cooldown:
                    interventions += 1
                    last_int_step = step
                    
    return interventions, events

print("\nEvaluating different controller settings across the 6 runs:")
configs = [
    ("Default (Frozen)", {"dorm_on": 0.15, "dorm_off": 0.05, "rho_on": 0.70, "rho_off": 0.85, "cooldown": 0, "rank_only_off": False}),
    ("Rank-Only Turn-Off", {"dorm_on": 0.15, "dorm_off": 0.05, "rho_on": 0.70, "rho_off": 0.85, "cooldown": 0, "rank_only_off": True}),
    ("Higher Dorm-Off (0.12)", {"dorm_on": 0.15, "dorm_off": 0.12, "rho_on": 0.70, "rho_off": 0.85, "cooldown": 0, "rank_only_off": False}),
    ("Higher Dorm-Off (0.15)", {"dorm_on": 0.15, "dorm_off": 0.15, "rho_on": 0.70, "rho_off": 0.85, "cooldown": 0, "rank_only_off": False}),
    ("Cooldown 5k Steps", {"dorm_on": 0.15, "dorm_off": 0.05, "rho_on": 0.70, "rho_off": 0.85, "cooldown": 5000, "rank_only_off": False}),
    ("Single-Shot / Pulse", {"dorm_on": 0.15, "dorm_off": 0.05, "rho_on": 0.70, "rho_off": 0.85, "cooldown": 99999, "rank_only_off": False}),
]

results = []
for p in gate_paths:
    cell_name = os.path.basename(os.path.dirname(p))
    with open(p, "r") as f:
        records = [json.loads(line) for line in f]
    
    row = {"cell": cell_name}
    for name, cfg in configs:
        n_ints, evts = simulate_gate(records, **cfg)
        row[name] = n_ints
    results.append(row)

res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))
