import json
import os

print('=== FINAL VERIFICATION ===')
print()

conditions = [
    ('fixed_obs_noise_s0', 'fixed', 'obs_noise', 50, 0),
    ('adaptive_obs_noise_s0', 'adaptive', 'obs_noise', 50, 0),
    ('fixed_no_shift_s0', 'fixed', 'obs_noise', 0, 0),
    ('adaptive_no_shift_s0', 'adaptive', 'obs_noise', 0, 0),
]

for name, ctrl_type, shift_type, shift_step, seed in conditions:
    print(name)
    
    # Check summary.json
    cstr = {'fixed': 'fixed', 'adaptive': 'adaptive'}[ctrl_type]
    sstr = '-shift' if shift_step > 0 else '-noshift'
    oname = 'M10-' + cstr + sstr + '-seed' + str(seed)
    
    found = False
    for f in os.listdir('results'):
        if f.startswith('M10-') and f.endswith('seed' + str(seed)):
            if ('shift' in f and 'noshift' not in f) == (shift_step > 0):
                if ('adaptive' in f and 'adaptive' in ctrl_type) or ('fixed' in f and 'fixed' in ctrl_type):
                    with open('results/' + f + '/summary.json') as f:
                        s = json.load(f)
                    print('  summary.json: best_normalized=' + str(s.get('best_normalized')))
                    found = True
                    break
    if not found:
        print('  summary.json: NOT FOUND')
    
    # Check diagnostics
    cstr = {'fixed': 'fixed', 'adaptive': 'adaptive'}[ctrl_type]
    sstr = '-shift' if shift_step > 0 else '-noshift'
    oname = 'M10-' + cstr + sstr + '-seed' + str(seed)
    diag_path = 'results/diagnostics_' + oname + '.jsonl'
    if os.path.exists(diag_path):
        with open(diag_path) as f:
            lines = f.readlines()
        print('  diagnostics.jsonl: ' + str(len(lines)) + ' records')
        if lines:
            first = json.loads(lines[0])
            last = json.loads(lines[-1])
            print('  shift_active: ' + str(first.get('shift_active')) + ' -> ' + str(last.get('shift_active')))
            print('  repr_change: ' + str(first.get('repr_change')) + ' -> ' + str(last.get('repr_change')))
            print('  controller_strength: ' + str(first.get('controller_strength')) + ' -> ' + str(last.get('controller_strength')))
            
            repr_values = [json.loads(l).get('repr_change', 0) for l in lines]
            if any(r != 0 for r in repr_values):
                print('  [PASS] repr_change is non-zero (temporal)')
            else:
                print('  [WARN] repr_change is always zero')
            
            strengths = [json.loads(l).get('controller_strength', 0) for l in lines]
            if len(set(strengths)) > 1:
                print('  [PASS] controller_strength changes over time')
            else:
                print('  [WARN] controller_strength is constant')
        else:
            print('  diagnostics.jsonl: NOT FOUND')
        print()

print('=== ALL CHECKS COMPLETE ===')