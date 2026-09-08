import json
import os

print('=== Extended M10 Validation ===')

# Check the extended run
diag_path = 'results/diagnostics_M10-adaptive-shift-seed0.jsonl'
if os.path.exists(diag_path):
    with open(diag_path) as f:
        lines = f.readlines()
    print('Total records:', len(lines))
    
    # Check perf_change values
    perf_changes = [json.loads(l)['perf_change'] for l in lines]
    print('perf_change values:', [round(v, 6) for v in perf_changes])
    
    # Check repr_change
    repr_changes = [json.loads(l)['repr_change'] for l in lines]
    print('repr_change range:', min(repr_changes), 'to', max(repr_changes))
    
    # Check controller_strength
    strengths = [json.loads(l)['controller_strength'] for l in lines]
    print('strength range:', min(strengths), 'to', max(strengths))
    
    # Check perf_change non-zero count
    non_zero = sum(1 for v in [json.loads(l)['perf_change'] for l in lines] if abs(v) > 1e-10)
    print('Non-zero perf_change count:', non_zero, '/', len(lines))
    
    # Check controller_strength uniqueness
    strengths = [json.loads(l)['controller_strength'] for l in lines]
    unique_strengths = len(set([round(s, 10) for s in strengths]))
    print('Unique strength values:', unique_strengths)