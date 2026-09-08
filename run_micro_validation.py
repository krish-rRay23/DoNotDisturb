from adaptive_plasticity.m5 import m5_run
import json
import os
import math

print('=== M10 MICRO-VALIDATION ===')
print()

conditions = [
    ('fixed_obs_noise_s0', 'fixed', 'obs_noise', 50, 0),
    ('adaptive_obs_noise_s0', 'adaptive', 'obs_noise', 50, 0),
    ('fixed_no_shift_s0', 'fixed', 'obs_noise', 0, 0),
    ('adaptive_no_shift_s0', 'adaptive', 'obs_noise', 0, 0),
]

all_passed = True

for name, ctrl_type, shift_type, shift_step, seed in conditions:
    print(name)
    try:
        result = m5_run(
            dataset_id='halfcheetah-medium-v2',
            seed=seed,
            online_steps=100,
            update_freq=10,
            online_ratio=0.5,
            warmup_steps=6,
            eval_interval=50,
            eval_episodes=1,
            device='cuda',
            controller_type=ctrl_type,
            shift_step=shift_step,
            severity=0.1,
            shift_type=shift_type,
        )
        
        # Check 1: online_steps
        assert result['online_steps'] == 100
        print('  online_steps: PASS')
        
        # Check 2: update_steps > 0
        assert result['update_steps'] > 0
        print('  update_steps: ' + str(result['update_steps']) + ' > 0: PASS')
        
        # Check 3: no NaN/Inf
        br = result.get('best_return', float('nan'))
        bn = result.get('best_normalized', float('nan'))
        assert math.isfinite(br) and math.isfinite(bn)
        print('  no NaN/Inf: PASS')
        
        # Check output path
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
            all_passed = False
        
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
                
                # Check repr_change is not always zero
                repr_values = [json.loads(l).get('repr_change', 0) for l in lines]
                if any(r != 0 for r in repr_values):
                    print('  [PASS] repr_change is non-zero (temporal)')
                else:
                    print('  [WARN] repr_change is always zero')
                
                # Check controller_strength changes
                strengths = [json.loads(l).get('controller_strength', 0) for l in lines]
                if len(set(strengths)) > 1:
                    print('  [PASS] controller_strength changes over time')
                else:
                    print('  [WARN] controller_strength is constant')
        else:
            print('  [FAIL] diagnostics.jsonl NOT FOUND')
            all_passed = False
        
        print()
        
    except Exception as e:
        print('  [FAIL] ERROR: ' + str(e))
        import traceback
        traceback.print_exc()
        all_passed = False
        print()

print('=== ALL MICRO-VALIDATION TESTS PASSED ===' if all_passed else '=== SOME TESTS FAILED ===')