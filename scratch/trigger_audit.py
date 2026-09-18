import json, glob, os

print("=== TRIGGER AUDIT ACROSS ALL 6 CAPACITY_GATE RUNS ===")
gate_paths = sorted(glob.glob("results/stress_study/STRESS-*capacity_gate*/gate_*.jsonl"))

for p in gate_paths:
    cell_name = os.path.basename(os.path.dirname(p))
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
        
        if gate_on and first_trigger is None:
            first_trigger = (step, on_dorm, on_rho, off_dorm, off_rho, shift_act)
            
    print(f"\n{cell_name}:")
    if first_trigger:
        s, od, orh, ofd, ofr, sh = first_trigger
        print(f"  First trigger at step {s}: shift_active={sh}, on_dorm={od:.4f}, on_rho={orh:.4f} (off_dorm={ofd:.4f}, off_rho={ofr:.4f})")
    else:
        print("  NEVER TRIGGERED (gate remained OFF throughout 25k steps)")
    
    # print steps 1000 to 7000
    for r in records:
        s = r.get("step")
        if s <= 7000:
            print(f"    step {s:5d}: shift={str(r.get('shift_active')):5s} | on_dorm={r.get('online_dormancy'):.4f} | on_rho={r.get('online_rho'):.4f} | off_dorm={r.get('offline_dormancy'):.4f} | off_rho={r.get('offline_rho'):.4f} | gate_on={str(r.get('gate_on')):5s}")
