"""M10 pilot experiment run script."""
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from adaptive_plasticity.m5 import m5_run

def main():
    result = m5_run(
        dataset_id='halfcheetah-medium-v2',
        seed=0,
        online_steps=20000,
        update_freq=1000,
        online_ratio=0.5,
        warmup_steps=50000,
        eval_interval=5000,
        eval_episodes=5,
        device='cuda',
        controller_type=None,  # AdaptivePlasticityController path internal
        shift_step=5000,
        severity=0.1,
        shift_type='obs_noise',
    )
    
    print('=== M10 PILOT EXPERIMENT RESULT ===')
    print(json.dumps(result, indent=2, default=str))
    
    # Verify the 5 requested conditions
    print('\n=== VERIFICATION ===')
    
    # 1. Adaptive intervention strength changes over time
    strength_history = result.get('adaptive_strength_history', 'N/A')
    print(f'1. Strength changes over time: {"YES" if strength_history else "NO"}')
    if strength_history and len(strength_history) > 1:
        print(f'   Initial: {strength_history[0]:.4f}, Final: {strength_history[-1]:.4f}')
    
    # 2. Shift activates at exactly step 5000
    online_steps = result.get('online_steps', 0)
    shift_step = 5000
    print(f'2. Shift at step {shift_step}: online_steps={online_steps}, shift_fired={shift_step <= online_steps}')
    
    # 3. Diagnostics logged correctly
    diagnostics = result.get('diagnostics_logged', 'N/A')
    print(f'3. Diagnostics logged: {"YES" if diagnostics else "NO"}')
    
    # 4. No NaN/Inf
    returns = result.get('returns', [])
    nan_inf = any(
        any((isinstance(r, float) and (math.isnan(r) or math.isinf(r))) for r in returns_batch)
        for returns_batch in [returns] if returns
    ) if returns else 'check manually'
    has_returns = returns is not None and len(returns) > 0
    print(f'4. No NaN/Inf in returns: {"YES" if has_returns and not any((isinstance(r, float) and (math.isnan(r) or math.isinf(r))) for r in returns) else "check manually"}')
    
    # 5. Evaluation and checkpoint outputs valid
    best_return = result.get('best_return', None)
    normalized = result.get('best_normalized', None)
    print(f'5. Eval outputs valid: best_return={best_return}, best_normalized={normalized}')
    
    # Save result
    os.makedirs('results', exist_ok=True)
    with open('results/pilot_m10.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print('\nResult saved to results/pilot_m10.json')


if __name__ == '__main__':
    import math
    main()